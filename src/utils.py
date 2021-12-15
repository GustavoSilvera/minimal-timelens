import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import io

device_str: str = "cuda" if torch.cuda.is_available() else "cpu"
device: torch.device = torch.device(device_str)
print(f"Found torch device: {device}")


def map_to_pm1(data: np.ndarray) -> np.ndarray:
    # convert polarity from 0, 1 -> -1, 1
    assert np.min(data) == 0 and np.max(data) == 1
    return 2 * data - 1


def load_im(im_name: str) -> np.ndarray:
    assert os.path.exists(im_name)
    return io.imread(im_name)


def end_th(num: int) -> str:
    """english is weird"""
    if num == 1:
        return "st"
    if num == 2:
        return "nd"
    if num == 3:
        return "rd"
    return "th"


def load_data(data_dir) -> Tuple[List[str], np.ndarray, np.ndarray]:
    image_dir: str = os.path.join(data_dir, "images")
    # lazily loading images
    image_files: Tuple[str] = tuple(os.walk(image_dir))[0]
    frame_paths: List[str] = [
        os.path.join(image_dir, f)
        for f in sorted(image_files[2])
        if ".png" in f or ".jpg" in f  # supported image formats
    ]

    event_dir = os.path.join(data_dir, "events")
    event_files = [f for f in sorted(tuple(os.walk(event_dir))[0][2]) if ".npz" in f]
    events: List[np.ndarray] = []
    for event_file in event_files:
        tmp = np.load(os.path.join(event_dir, event_file), allow_pickle=True)
        events.append(
            np.stack(
                (
                    tmp["x"].astype(np.float64).reshape((-1,)),  # x
                    tmp["y"].astype(np.float64).reshape((-1,)),  # y
                    tmp["t"].astype(np.float64).reshape((-1,)),  # timestamp
                    map_to_pm1(
                        tmp["p"].astype(np.float32).reshape((-1,))
                    ),  # polarity (-1 or 1)
                ),
                axis=-1,
            )
        )
    events: np.ndarray = np.concatenate(events)
    timestamps_files: List[str] = [f for f in image_files[2] if ".txt" in f]
    assert len(timestamps_files) == 1
    timestamps: np.ndarray = np.loadtxt(os.path.join(image_dir, timestamps_files[0]))
    return frame_paths, timestamps, events


def to_voxel_grid(
    event_sequence: np.ndarray,
    im_size: Tuple[int, int, int],
    num_bins: Optional[int] = 5,
) -> torch.Tensor:
    h, w, _ = im_size
    voxel_grid: torch.Tensor = torch.zeros(
        num_bins, h, w, dtype=torch.float32, device=device
    )
    voxel_grid_flat = voxel_grid.flatten()

    # Convert timestamps to [0, num_bins] range.
    start_t: float = np.min(event_sequence[:, 2])
    end_t: float = np.max(event_sequence[:, 2])
    duration: float = end_t - start_t
    features: torch.Tensor = torch.from_numpy(event_sequence).to(device)
    x: torch.Tensor = features[:, 0]
    y: torch.Tensor = features[:, 1]
    pol: torch.Tensor = features[:, 3].float()
    t: torch.Tensor = ((features[:, 2] - start_t) * (num_bins - 1) / duration).float()

    left_t: torch.Tensor = t.floor().long()
    left_x: torch.Tensor = x.floor().long()
    left_y: torch.Tensor = y.floor().long()

    right_t: torch.Tensor = (left_t + 1).long()
    right_x: torch.Tensor = (left_x + 1).long()
    right_y: torch.Tensor = (left_y + 1).long()

    for xb in [left_x, right_x]:
        for yb in [left_y, right_y]:
            for tb in [left_t, right_t]:
                mask: torch.Tensor = (
                    (0 <= xb)
                    & (0 <= yb)
                    & (0 <= tb)
                    & (xb <= w - 1)
                    & (yb <= h - 1)
                    & (tb <= num_bins - 1)
                )
                lin_idx: torch.Tensor = xb + yb * w + tb * w * h
                weight: torch.Tensor = (
                    pol
                    * (1 - torch.abs(xb - x))
                    * (1 - torch.abs(yb - y))
                    * (1 - torch.abs(tb - t))
                )
                voxel_grid_flat.index_add_(
                    dim=0, index=lin_idx[mask], source=weight[mask].float()
                )

    return voxel_grid.unsqueeze(0)


def backwarp_2d(
    src: torch.Tensor, y_disp: torch.Tensor, x_disp: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    w: int = src.shape[-1]
    h: int = src.shape[-2]

    def compute_src_coordinates(
        y_disp: torch.Tensor, x_disp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w: int = y_disp.shape[-1]
        h: int = y_disp.shape[-2]

        x, y = torch.meshgrid([torch.arange(w), torch.arange(h)])
        x: torch.Tensor = x.transpose(0, 1).float().to(device) + x_disp.squeeze(1)
        y: torch.Tensor = y.transpose(0, 1).float().to(device) + y_disp.squeeze(1)
        oob_mask: torch.Tensor = ((x < 0) | (x >= w) | (y < 0) | (y >= h))
        return y, x, oob_mask

    y_src, x_src, mask = compute_src_coordinates(y_disp, x_disp)
    x_src: torch.Tensor = ((2.0 / float(w - 1)) * x_src - 1).masked_fill(mask, 0)
    y_src: torch.Tensor = ((2.0 / float(h - 1)) * y_src - 1).masked_fill(mask, 0)
    mask: torch.Tensor = mask.unsqueeze(1)
    grid_src = torch.stack([x_src, y_src], -1)
    target = torch.nn.functional.grid_sample(src, grid_src, align_corners=True)
    target.masked_fill_(mask.expand_as(target), 0)
    return target, mask


def reverse_events(events: np.ndarray) -> np.ndarray:
    if len(events) == 0:
        return events
    reversed_events: np.ndarray = events.copy()
    # start_t: float = np.min(events[:, 2])
    end_t: float = np.max(reversed_events[:, 2])
    reversed_events[:, 2] = end_t - reversed_events[:, 2]  # reverse flow of time
    reversed_events[:, 3] = -1 * reversed_events[:, 3]  # negate the polarity
    reversed_events = np.copy(np.flipud(reversed_events))  # flip rows (copy +strides)
    return reversed_events


def split_events_by_t(
    events: np.ndarray, t: float, t_min: float, t_max: float
) -> Tuple[np.ndarray, np.ndarray]:
    assert t_min < t < t_max
    start_t: float = np.min(events[:, 2])
    end_t: float = np.max(events[:, 2])
    if not (start_t <= t <= end_t):
        raise ValueError(
            '"timestamps" should be between start and end of the sequence.'
        )
    after_t_min: np.ndarray = events[:, 2] > t_min
    after_t: np.ndarray = events[:, 2] > t
    after_t_max: np.ndarray = events[:, 2] > t_max
    first_sequence: np.ndarray = events[(after_t_min * (~after_t)), :]
    second_sequence: np.ndarray = events[(after_t * (~after_t_max)), :]
    return first_sequence, second_sequence


def save_im(out_dir: str, im: np.ndarray, frame_num: int) -> None:
    im_np: np.ndarray = im
    if not isinstance(im, np.ndarray):
        im_np = np.moveaxis(im.numpy(), 0, 2)
    plt.imsave(os.path.join(out_dir, f"{frame_num:04d}.jpg"), im_np)
