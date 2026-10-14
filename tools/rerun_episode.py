"""Show one Bridge episode in rerun: every downloaded camera, gripper, xyz and phases over time.

python3 tools/rerun_episode.py --data data/raw/bridge --episode 32
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import rerun as rr

from explore_episode import (CAMS, camera_present, gripper_scale, load_frames_episode,
                             load_meta_episodes, read_frames)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "submission" / "src"))
from aijgen import phases as ph


def video_file(root: Path, ep, cam: str) -> Path:
    """mp4 with this episode's frames from camera cam."""
    key = f"videos/observation.images.{cam}"
    c, f = int(ep[f"{key}/chunk_index"]), int(ep[f"{key}/file_index"])
    return root / f"videos/observation.images.{cam}/chunk-{c:03d}/file-{f:03d}.mp4"


def set_frame(t: int) -> None:
    """Put the following logs at frame t of the timeline (API differs between rerun versions)."""
    if hasattr(rr, "set_time"):
        rr.set_time("frame", sequence=t)
    else:
        rr.set_time_sequence("frame", t)


def scalar(x: float):
    return rr.Scalars(x) if hasattr(rr, "Scalars") else rr.Scalar(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="Bridge dataset root")
    ap.add_argument("--episode", type=int, required=True)
    args = ap.parse_args()

    eps = load_meta_episodes(args.data)
    ep = eps[eps["episode_index"] == args.episode].iloc[0]
    rows = load_frames_episode(args.data, ep)
    state = np.stack(rows["observation.state"].values)
    action = np.stack(rows["action"].values)
    n = len(rows)
    task = list(ep["tasks"])[0]

    c = ph.closedness(state[:, 6], closed_high=False, scale=gripper_scale(args.data))
    labels = ph.label(c) if c is not None else None

    # cameras that have real frames for this episode and whose video is downloaded
    cams = [cam for cam in CAMS if camera_present(ep, cam) and video_file(args.data, ep, cam).exists()]
    frames = {cam: read_frames(args.data, ep, cam, n) for cam in cams}
    print(f"episode {args.episode}: '{task}', {n} frames, cameras {cams}")

    rr.init(f"bridge_ep{args.episode}", spawn=True)
    rr.log("task", rr.TextDocument(task), static=True)
    for t in range(n):
        set_frame(t)
        for cam in cams:
            if frames[cam][t] is not None:
                rr.log(f"camera/{cam}", rr.Image(frames[cam][t]))
        rr.log("gripper/state", scalar(state[t, 6]))
        rr.log("gripper/action", scalar(action[t, 6]))
        if c is not None:
            rr.log("gripper/closedness", scalar(c[t]))
        for i, name in enumerate("xyz"):
            rr.log(f"position/{name}", scalar(state[t, i] - state[0, i]))
        if labels is not None and (t == 0 or labels[t] != labels[t - 1]):
            rr.log("phase", rr.TextLog(ph.PHASES[labels[t]]))


if __name__ == "__main__":
    main()