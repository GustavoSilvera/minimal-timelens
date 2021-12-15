from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Up(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv1: nn.Module = nn.Conv2d(in_dim, out_dim, 3, stride=1, padding=1)
        self.conv2: nn.Module = nn.Conv2d(2 * out_dim, out_dim, 3, stride=1, padding=1)
        self.ReLU: nn.Module = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor, skpCn: torch.Tensor) -> torch.Tensor:
        x = self.ReLU(self.conv1(F.interpolate(x, scale_factor=2, mode="bilinear")))
        x = self.ReLU(self.conv2(torch.cat((x, skpCn), 1)))
        return x


class Down(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, filter_dim: int):
        super().__init__()
        kwargs: Dict[str, Any] = {"stride": 1, "padding": (filter_dim - 1) // 2}
        self.conv1: nn.Module = nn.Conv2d(in_dim, out_dim, filter_dim, **kwargs)
        self.conv2: nn.Module = nn.Conv2d(out_dim, out_dim, filter_dim, **kwargs)
        self.ReLU: nn.Module = nn.LeakyReLU(0.1)
        layers: List[nn.Module] = [self.conv1, self.ReLU, self.conv2, self.ReLU]
        self.network: nn.Linear = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network.forward(F.avg_pool2d(x, 2))


class SizeAdapter:
    def __init__(self, min_size: Optional[int] = 64):
        self._m: int = min_size
        self.pad_w: int = None
        self.pad_h: int = None

    def closest_to_min(self, x: int, m: Optional[int] = None) -> int:
        m = self._m if m is None else m
        return int(np.ceil(x / m) * m)

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        self.pad_h = self.closest_to_min(h) - h
        self.pad_w = self.closest_to_min(w) - w
        return nn.ZeroPad2d((self.pad_w, 0, self.pad_h, 0)).forward(x)

    def unpad(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.pad_h :, self.pad_w :]


class UNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(UNet, self).__init__()
        self._size_adapter = SizeAdapter(min_size=32)
        self.conv1: nn.Module = nn.Conv2d(in_dim, 32, 7, stride=1, padding=3)
        self.conv2: nn.Module = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.conv3 = nn.Conv2d(32, out_dim, 3, stride=1, padding=1)
        self.down1: Down = Down(32, 64, 5)
        self.down2: Down = Down(64, 128, 3)
        self.down3: Down = Down(128, 256, 3)
        self.down4: Down = Down(256, 512, 3)
        self.down5: Down = Down(512, 512, 3)
        self.up1: Up = Up(512, 512)
        self.up2: Up = Up(512, 256)
        self.up3: Up = Up(256, 128)
        self.up4: Up = Up(128, 64)
        self.up5: Up = Up(64, 32)
        self.ReLU: nn.Module = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ReLU(self.conv1(self._size_adapter.pad(x)))
        s1 = self.ReLU(self.conv2(x))
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        x = self.down5(s5)
        x = self.up5(self.up4(self.up3(self.up2(self.up1(x, s5), s4), s3), s2), s1)
        return self._size_adapter.unpad(self.conv3(x))
