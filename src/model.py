import os
import warnings
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms

from unet import UNet
from utils import backwarp_2d, device, reverse_events, to_voxel_grid

warnings.filterwarnings("ignore", category=UserWarning)  # no more nasty torch warnings


class Warp(nn.Module):
    def __init__(self):
        super().__init__()
        self.flow_network = UNet(5, 2)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        flow: torch.Tensor = self.flow_network(
            torch.cat([x["before.voxel_grid_rev"], x["after.voxel_grid"],])
        )
        warped, _ = backwarp_2d(
            src=torch.cat([x["before.rgb_image"], x["after.rgb_image"],]),
            y_disp=flow[:, 0, ...],
            x_disp=flow[:, 1, ...],
        )
        (before_flow, after_flow) = torch.chunk(flow, chunks=2)
        (before_warped, after_warped) = torch.chunk(warped, chunks=2)
        return {
            "middle.before_warped": before_warped,
            "middle.after_warped": after_warped,
            "before.flow": before_flow,
            "after.flow": after_flow,
        }


class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion_network: UNet = UNet(16, 3)

    def forward(self, x: Dict[str, Any]) -> torch.Tensor:
        return self.fusion_network(
            torch.cat(
                [
                    x["before.voxel_grid"],
                    x["before.rgb_image"],
                    x["after.voxel_grid"],
                    x["after.rgb_image"],
                ],
                dim=1,
            )
        )


class RefineWarp:
    def __init__(self):
        self.warp: Warp = Warp()
        self.fusion: Fusion = Fusion()
        self.flow_ref: UNet = UNet(9, 4)

    def to_device(self, device: torch.device) -> None:
        self.device = device
        self.warp = self.warp.to(self.device)
        self.fusion = self.fusion.to(self.device)
        self.flow_ref = self.flow_ref.to(self.device)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        x.update(self.warp.forward(x))
        x["middle.fusion"] = self.fusion.forward(x)
        residual = self.flow_ref(
            torch.cat(
                [x["middle.{}_warped".format(packet)] for packet in ["after", "before"]]
                + [x["middle.fusion"]],
                dim=1,
            )
        )
        residual = torch.cat(torch.chunk(residual, 2, dim=1), dim=0)
        refined, _ = backwarp_2d(
            src=torch.cat([x["middle.after_warped"], x["middle.before_warped"]]),
            y_disp=residual[:, 0, ...],
            x_disp=residual[:, 1, ...],
        )
        return torch.chunk(refined, 2)


class AttentionAverage:
    def __init__(self):
        self.refine_warp: RefineWarp = RefineWarp()
        self.attention_network: UNet = UNet(14, 3)
        self.device: torch.device = torch.device("cpu")  # default, not recommended
        self.to_device(device)

    def to_device(self, device: torch.device) -> None:
        self.device = device
        self.attention_network = self.attention_network.to(self.device)
        self.refine_warp.to_device(self.device)

    def forward(self, x: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        before_refined, after_refined = self.refine_warp.forward(x)
        num, _, h, w = x["middle.fusion"].shape
        attention_scores: torch.Tensor = self.attention_network(
            torch.cat(
                [
                    x["after.flow"],
                    after_refined,
                    x["before.flow"],
                    before_refined,
                    x["middle.fusion"],
                    x["middle.weight"] * torch.ones((num, 1, h, w), device=device),
                ],
                dim=1,
            )
        )
        attention: torch.Tensor = F.softmax(attention_scores, dim=1)
        average: torch.Tensor = (
            attention[:, 0, ...].unsqueeze(1) * before_refined
            + attention[:, 1, ...].unsqueeze(1) * after_refined
            + attention[:, 2, ...].unsqueeze(1) * x["middle.fusion"]
        )
        return average, attention

    def run(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        left_events: np.ndarray,
        right_events: np.ndarray,
        t_diff: float,
    ) -> torch.Tensor:
        # convert images to tensors
        to_tensor: Callable[
            [np.ndarray], torch.Tensor
        ] = lambda x: transforms.ToTensor()(x).to(self.device)

        v: int = 5  # voxel bin size
        s: Tuple[int, int, int] = left_frame.shape
        x: Dict[str, Any] = {  # data structure to be passed around
            "before.rgb_image": to_tensor(left_frame).unsqueeze(0),
            "before.voxel_grid": to_voxel_grid(left_events, s, v),
            "before.voxel_grid_rev": to_voxel_grid(reverse_events(left_events), s, v),
            "middle.weight": t_diff,
            "after.rgb_image": to_tensor(right_frame).unsqueeze(0),
            "after.voxel_grid": to_voxel_grid(right_events, s, v),
        }

        with torch.no_grad():
            frame, _ = self.forward(x)
        return torch.clamp(frame.squeeze().cpu().detach(), 0, 1,)

    def load_legacy_ckpt(self, ckpt_file: str) -> None:
        assert os.path.exists(ckpt_file)
        params: Dict[str, Any] = torch.load(ckpt_file, map_location=device)["networks"]
        # from original timelens authors checkpoint.bin
        def assign_weight(conv: nn.Module, weights: np.ndarray) -> nn.Module:
            conv.weight = nn.Parameter(weights)
            return conv

        def assign_bias(conv: nn.Module, biases: np.ndarray) -> nn.Module:
            conv.bias = nn.Parameter(biases)
            return conv

        def update_conv(module: nn.Module, conv_name: str) -> None:
            old = getattr(module, conv_name)
            if module_names[-1] == "weight":
                setattr(module, conv_name, assign_weight(old, val))
            elif module_names[-1] == "bias":
                setattr(module, conv_name, assign_bias(old, val))
            else:
                raise NotImplementedError

        for key, val in params.items():
            module_names = key.split(".")
            module = self
            if module_names[0] == "attention_network":
                module = self.attention_network
            elif module_names[0] == "flow_network":
                module = self.refine_warp.warp.flow_network
            elif module_names[0] == "flow_refinement_network":
                module = self.refine_warp.flow_ref
            elif module_names[0] == "fusion_network":
                module = self.refine_warp.fusion.fusion_network
            else:
                raise NotImplementedError
            # for now only supported 2 layers deep since thats all the timelens authors use
            if "conv" in module_names[1]:
                update_conv(module, module_names[1])
            else:
                update_conv(getattr(module, module_names[1]), module_names[2])

        print("Done assigning from legacy ckpt file")

    def save_ckpt(self, ckpt_dir: str) -> None:
        print(f"Saving own ckpt data to {ckpt_dir}")

        def save_state_dict(module: str) -> None:
            print(f"Saving {module} ckpt")
            modules: List[str] = module.split(".")
            base = self if len(modules) == 1 else self.refine_warp
            torch.save(
                {"state_dict": getattr(base, modules[-1]).state_dict()},
                os.path.join(ckpt_dir, f"{module}.bin"),
            )

        save_state_dict("attention_network")
        save_state_dict("refine_warp.warp")
        save_state_dict("refine_warp.fusion")
        save_state_dict("refine_warp.flow_ref")

    def load_ckpt(self, ckpt_dir: str) -> None:
        print(f"Loading from ckpt data: {ckpt_dir}")

        def load_state_dict(module: str) -> None:
            print(f"Loading {module} ckpt")
            modules: List[str] = module.split(".")
            params = torch.load(os.path.join(ckpt_dir, f"{module}.bin"))
            modules: List[str] = module.split(".")
            base = self if len(modules) == 1 else self.refine_warp
            getattr(base, modules[-1]).load_state_dict(params["state_dict"])

        load_state_dict("attention_network")
        load_state_dict("refine_warp.warp")
        load_state_dict("refine_warp.fusion")
        load_state_dict("refine_warp.flow_ref")

