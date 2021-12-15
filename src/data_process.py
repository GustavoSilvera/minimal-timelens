import numpy as np
import dv  # for our inivation camera
from typing import Dict, List, Optional, Tuple
from utils import load_data, load_im
import os
import matplotlib.pyplot as plt
import shutil

data_dir: str = "data"
images_set: str = "hand"
aedat4_file: str = "events.aedat4"
np_events_file: str = "events.npy"
dvXres: Tuple[int, int] = (320, 240)
rgbres: Tuple[int, int] = (1920, 1080)
fps: float = 59.94
used_beam_splitter: bool = True
temporal_res: float = 1 / 1e6  # our camera has microsecond-level granularity
# idxs
T_IDX: int = 0
X_IDX: int = 1
Y_IDX: int = 2
P_IDX: int = 3


def event_to_np(event: np.array) -> np.ndarray:
    # need to convert from custom types
    ret = np.zeros((len(event), 4), dtype=np.int64)
    for i, data in enumerate(event):
        # need to explicitly copy (not datatypes)
        ret[i, :] = np.array([data[0], data[1], data[2], data[3]], dtype=np.int64)
    return ret


def read_data(aedat_file: str, try_load_np: Optional[bool] = True):
    events: List[np.ndarray] = []
    dv_file = os.path.join(data_dir, images_set, aedat_file)
    if try_load_np:
        np_file = os.path.join(data_dir, images_set, np_events_file)
        if os.path.exists(np_file):
            print(f"Loading events from existing file: {np_file}")
            return np.load(np_file)
    print("Performing first time Aedat->np data conversion... this might take a while")
    assert os.path.exists(dv_file)
    with dv.AedatFile(dv_file) as f:  # for first time AedatFiles
        for event in f["events"].numpy():
            events.append(event_to_np(event))
            i: int = len(events)
            if i % 20 == 0:
                print(f"Accumulated {i} events", end="\r", flush=True)
    num_events: int = len(events)
    print(f"Done! Accumulated {num_events} events")
    events: np.ndarray = np.concatenate(events)
    # save the file to a npy so we don't need to run this again
    np.save(os.path.join(data_dir, images_set, np_events_file), events)
    return events


def get_timestamps_from_events(events: np.ndarray) -> np.ndarray:
    return (events[:, T_IDX] - np.min(events[:, T_IDX])) * temporal_res


def get_buckets(
    timestamps: np.ndarray, pol: np.ndarray, bins: int
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    dt: float = np.max(timestamps) / bins
    ts = np.array([dt * i for i in range(bins)])
    p_buckets: List[Tuple[int, int]] = []
    ub: float = 0
    for i in range(bins - 1):
        t0: float = dt * (i + 0)  # lower bound
        t1: float = dt * (i + 1)  # upper bound
        lb: int = np.searchsorted(timestamps, t0) if ub is None else ub
        ub: int = np.searchsorted(timestamps, t1)
        pol_data: np.ndarray = pol[lb:ub]
        total: int = ub - lb
        p_buckets.append((np.sum(pol_data), total))

    return ts, p_buckets


def visualize_into_buckets(events: np.ndarray, num_buckets: int) -> None:
    # plot information about data
    timestamps: np.ndarray = get_timestamps_from_events(events)
    assert np.all(np.diff(timestamps) >= 0)  # sorted in increasing order
    pol: np.ndarray = events[:, P_IDX]  # 0 or 1
    fig, axs = plt.subplots(1, 1, tight_layout=True)
    fig.suptitle(f"visualizing events into {num_buckets} bins")
    plt.xlabel("Time (s)")
    plt.ylabel("# Events (+ and -)")
    bins = num_buckets  # for visualizing the data
    ts, p_buckets = get_buckets(timestamps, pol, bins)
    pos_p_bucket_val = np.array([pos for (pos, total) in p_buckets])  # only +1
    neg_p_bucket_val = np.array([total - pos for (pos, total) in p_buckets])  # only 0
    assert ts.shape == (bins,)
    assert pos_p_bucket_val.shape == neg_p_bucket_val.shape == (bins - 1,)
    axs.plot(ts[:-1], pos_p_bucket_val, c="green", linewidth=None)
    axs.plot(ts[:-1], neg_p_bucket_val, c="red", linewidth=None)
    plt.show()
    plt.cla()
    plt.clf()


def get_t_start(
    events: np.ndarray, granularity: int = 100, known_prior: Optional[float] = None
) -> float:
    """using the largest peak of no. of negative polarities to determine 
       synchrony time by considering when lights went out"""
    timestamps: np.ndarray = get_timestamps_from_events(events)
    pol: np.ndarray = events[:, P_IDX]
    if known_prior is not None:  # if we know the lights went out in this % of recording
        assert 0 < known_prior <= 1
        # we can clip the rest since we know ish where it lies
        timestamps = timestamps[: int(len(timestamps) * known_prior)]
        pol = pol[: int(len(pol) * known_prior)]
    ts, p_buckets = get_buckets(timestamps, pol, bins=granularity)
    neg_p_bucket_val = np.array([total - pos for (pos, total) in p_buckets])  # only 0
    # since we care about when the lights go 'out', find the largest diff
    bucket_idx_ord: np.ndarray = np.argsort(np.diff(neg_p_bucket_val))
    peak: int = bucket_idx_ord[-1]
    return ts[peak]


def get_ranged_events(
    all_events: np.ndarray, t_sync: float, t0: float, t1: float
) -> np.ndarray:
    timestamps: np.ndarray = get_timestamps_from_events(all_events)
    lb: np.ndarray = np.searchsorted(timestamps, t_sync + t0)
    ub: np.ndarray = np.searchsorted(timestamps, t_sync + t1)
    return events[lb:ub, :]  # everything from lower -> upper bound


def spatial_align(
    xs: np.ndarray, ys: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    print("Spatially aligning xs and ys")
    if used_beam_splitter:
        # notice that we need to flip the xs if we use a beam splitter
        xs = np.max(xs) - xs
    # first compute respective aspect ratios
    ar_dvX = dvXres[0] / dvXres[1]
    ar_rgb = rgbres[0] / rgbres[1]
    assert ar_dvX < ar_rgb  # for now, if not then align the y's
    # assuming dvX has a taller aspect ratio than rgb (by assert)
    # now we just need to align the x's together and crop off the top/bottom
    scale: float = rgbres[0] / dvXres[0]
    xs = xs * scale - 40
    ys = ys * scale * 0.9 - 130
    inner = (ys >= 0) * (ys < rgbres[1]) * (xs > 0) * (xs < rgbres[0])
    ys = ys[inner]  # crop out top/bottom/left/right that goes outside image
    xs = xs[inner]
    return xs, ys, inner


def save_events(events: np.ndarray, to_dir: str) -> None:
    os.makedirs(to_dir, exist_ok=True)
    tmp: Dict[str, np.ndarray] = {}
    x, y, inner = spatial_align(events[:, X_IDX], events[:, Y_IDX])
    _events = events[inner, :]
    tmp["t"] = get_timestamps_from_events(_events)
    tmp["x"] = x
    tmp["y"] = y
    tmp["p"] = _events[:, P_IDX]
    filepath = os.path.join(to_dir, "events.npz")
    np.savez(filepath, **tmp)
    print(f"successfully saved ranged events to {filepath}")


def extract_vid_data(
    critical_ts: Dict[str, Tuple[int, int]], events: np.ndarray, t_sync: float
) -> None:
    assert os.path.exists(data_dir)
    frame_sync = critical_ts["lights_out"][0]  # start of when lights go out

    for crit, data in critical_ts.items():
        print(f"{crit} took {(data[1] - data[0]) / fps :.3f}s")
    images_dir = os.path.join(data_dir, images_set, "images")
    # prefer jpg but also supports png
    ext: str = "jpg" if os.path.exists(
        os.path.join(images_dir, f"{1:05d}.jpg")
    ) else ".png"
    for crit_name in critical_ts.keys():
        new_dir: str = os.path.join(data_dir, crit_name)
        os.makedirs(new_dir, exist_ok=True)
        new_im_dir: str = os.path.join(new_dir, "images")
        os.makedirs(new_im_dir, exist_ok=True)
        new_im_dir_abs: str = os.path.abspath(new_im_dir)
        assert os.path.exists(new_im_dir_abs)
        frame = critical_ts[crit_name]
        # copy_files: str = f"cp {{{frame[0]:05d}..{frame[1]:05d}}}.{ext} {new_dir_abs}"
        for i in range(frame[0], frame[1] + 1):
            src: str = os.path.join(images_dir, f"{i:05d}.{ext}")
            shutil.copy(src, new_im_dir_abs)
        print(f"successfully copied files to {new_dir}")
        # also send over respective events:
        t0: float = (frame[0] - frame_sync) / fps
        t1: float = (frame[1] - frame_sync) / fps
        ranged_events = get_ranged_events(events, t_sync, t0, t1)
        new_events_dir: str = os.path.join(new_dir, "events")
        save_events(ranged_events, new_events_dir)
        # and finally, send in the image timestamps
        end_t = t1 - t0 + 1  # (include the +1 to buffer a tiny bit)
        im_timestamps = np.arange(start=0, stop=end_t, step=1 / fps)
        im_timestamps_path: str = os.path.join(new_im_dir, "timestamps.txt")
        np.savetxt(im_timestamps_path, im_timestamps)
        print(f"successfully created timestamps.txt for {crit_name}")
    return


def visualize_frames_side_by_side(image_set: str) -> None:
    custom_dir = os.path.join(data_dir, image_set)
    frames, timestamps, all_events = load_data(data_dir=custom_dir)
    side_by_side_dir = os.path.join(custom_dir, "side_by_side")
    os.makedirs(side_by_side_dir, exist_ok=True)
    for i in range(len(frames) - 1):
        rgb_im = load_im(frames[i])
        t0 = timestamps[i]
        t_idx0 = np.searchsorted(all_events[:, 2], t0)
        t1 = timestamps[i + 1]
        t_idx1 = np.searchsorted(all_events[:, 2], t1)
        events = all_events[t_idx0:t_idx1, :]
        fig, axs = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
        axs[0].imshow(rgb_im)
        xs = events[:, 0]
        ys = events[:, 1]
        pol = events[:, 3]
        colours = np.zeros((len(pol), 3), dtype=np.uint8)
        colours[pol == 1] = np.array([1, 0, 0])  # positive is red
        colours[pol == -1] = np.array([0, 0, 1])  # negative is blue
        axs[1].set_aspect("equal")
        axs[1].set_xlim([0, rgb_im.shape[1]])
        axs[1].set_ylim([0, rgb_im.shape[0]])
        axs[1].scatter(
            xs, rgb_im.shape[0] - ys, c=colours, s=1
        )  # flip according to origin
        plt.savefig(os.path.join(side_by_side_dir, f"{i:05d}.jpg"))
        plt.close()
        print(f"Rendered image: {100*i/(len(frames)-1):.3f}%", end="\r", flush=True)


def visualize_frames_overlaid(image_set: str) -> None:
    custom_dir = os.path.join(data_dir, image_set)
    frames, timestamps, all_events = load_data(data_dir=custom_dir)
    overlay_dir = os.path.join(custom_dir, "overlay")
    os.makedirs(overlay_dir, exist_ok=True)
    for i in range(len(frames) - 1):
        rgb_im = load_im(frames[i])
        t0 = timestamps[i]
        t_idx0 = np.searchsorted(all_events[:, 2], t0)
        t1 = timestamps[i + 1]
        t_idx1 = np.searchsorted(all_events[:, 2], t1)
        events = all_events[t_idx0:t_idx1, :]
        fig, axs = plt.subplots(1, 1, figsize=(5, 5), tight_layout=True)
        axs.imshow(rgb_im[::-1, :, :])
        xs = events[:, 0]
        ys = events[:, 1]
        pol = events[:, 3]
        colours = np.zeros((len(pol), 3), dtype=np.uint8)
        colours[pol == 1] = np.array([1, 0, 0])  # positive is red
        colours[pol == -1] = np.array([0, 0, 1])  # negative is blue
        axs.set_aspect("equal")
        axs.set_xlim([0, rgb_im.shape[1]])
        axs.set_ylim([0, rgb_im.shape[0]])
        axs.scatter(
            xs, rgb_im.shape[0] - ys, c=colours, s=1
        )  # flip according to origin
        plt.savefig(os.path.join(overlay_dir, f"{i:05d}.jpg"))
        plt.close()
        print(f"Rendered image: {100*i/(len(frames)-1):.3f}%", end="\r", flush=True)


if __name__ == "__main__":
    events = read_data(aedat4_file, try_load_np=True)
    granularity: int = 10000  # (num bins) bigger is better but more expensive
    visualize_into_buckets(events, granularity)
    # find where the lights turned off and trim the rest (before and after?)
    """first-time setup"""
    t_lights_off = get_t_start(events, granularity, known_prior=0.5)
    print(f"Synchrony start t={t_lights_off:.3f}s")
    critical_events: Dict[str, Tuple[int, int]] = {
        "lights_out": (215, 310),
        "hand_shake": (642, 984),
    }
    extract_vid_data(critical_events, events, t_lights_off)
    """once extract vid data has successfully ran"""
    for events in critical_events.keys():
        if events == "lights_out":
            continue
        visualize_frames_overlaid(events)
        visualize_frames_side_by_side(events)

