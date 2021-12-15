import os
import time
from typing import List

import numpy as np

from model import AttentionAverage
from utils import end_th, load_data, load_im, save_im, split_events_by_t

out_dir: str = "out"
os.makedirs(out_dir, exist_ok=True)
data_dir: str = "data"
assert os.path.exists(data_dir)
image_set: str = "example"
assert os.path.exists(os.path.join(data_dir, image_set))
legacy_checkpoint_file: str = os.path.join(data_dir, "checkpoint.bin")
ckpt_dir: str = os.path.join(data_dir, "ckpt")
os.makedirs(ckpt_dir, exist_ok=True)
num_inserts: int = 7


def run(
    model: AttentionAverage,
    frames: List[str],
    timestamps: np.ndarray,
    events: np.ndarray,
    num_inserts: int,
) -> None:
    wall_t = time.time()
    im_idx: int = 0
    total: int = (len(frames) - 1) * (num_inserts + 1)
    for i in range(len(frames) - 1):  # between any two ground-truth images
        left_frame = load_im(frames[i])  # if i == 0 else right_frame
        right_frame = load_im(frames[i + 1])  # always load, unseen frame
        save_im(out_dir, left_frame, im_idx)
        start_t, end_t = timestamps[i], timestamps[i + 1]
        for _insert in range(num_inserts):  # new images
            insert = _insert + 1
            t_diff: float = insert / (num_inserts + 1)
            t: float = start_t + t_diff * (end_t - start_t)
            Le, Re = split_events_by_t(events, t, start_t, end_t)
            dt: float = time.time() - wall_t
            completion: float = 100 * im_idx / total
            print(
                f"inserting {insert}{end_th(insert)} frame {completion:.3f}% @ {dt:.3f}s"
            )
            new_frame: np.ndarray = model.run(left_frame, right_frame, Le, Re, t_diff)
            im_idx += 1
            save_im(out_dir, new_frame, im_idx)
        im_idx += 1
    save_im(out_dir, right_frame, im_idx)
    print(f"Done interpolation!")


if __name__ == "__main__":
    print("Starting timelens video + event interpolation")
    model: AttentionAverage = AttentionAverage()
    # model.load_legacy_ckpt(legacy_checkpoint_file)  # optional after first run
    # model.save_ckpt(ckpt_dir)  # save ckpt to smaller individual files
    model.load_ckpt(ckpt_dir)  # load network ckpt from smaller individual files
    frames, timestamps, events = load_data(os.path.join(data_dir, image_set))
    run(model, frames, timestamps, events, num_inserts)
