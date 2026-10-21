"""Look at one Bridge episode: gripper/xyz plot, phases, keyframes.

python3 tools/explore_episode.py --data data/raw/bridge --out data/explore --episode 0
"""
import argparse
import glob
import shutil
import sys
from pathlib import Path
import json

import av
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image



REPO = Path(__file__).resolve().parents[1]  
sys.path.insert(0, str(REPO / "submission" / "src"))

from aijgen import phases as ph

FPS = 5
CAMS = ["image_0", "image_1", "image_2", "image_3"]
PHASE_COLORS = {"pre_grasp": "#9ecae1", "immobilization": "#fdae6b", "contact": "#e6550d",
                "detach": "#a1d99b", "post_grasp": "#d9d9d9"}


def load_meta_episodes(root: Path) -> pd.DataFrame:
    '''
    return: table of all /meta
    '''
    files = sorted(glob.glob(str(root / "meta/episodes/**/*.parquet"), recursive=True))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_frames_episode(root: Path, episode: pd.Series) -> pd.DataFrame:
    c, f = int(episode["data/chunk_index"]), int(episode["data/file_index"])
    df = pd.read_parquet(root / f"data/chunk-{c:03d}/file-{f:03d}.parquet")
    df = df[df["episode_index"] == episode["episode_index"]].sort_values("frame_index")
    return df.reset_index(drop=True)


def gripper_scale(root: Path) -> float:
    """Dataset-wide gripper range (p1..p99) from meta/stats.json."""
    st = json.load(open(root / "meta" / "stats.json"))["observation.state"]
    return st["q99"][6] - st["q01"][6]


def _flatten(x) -> list[float]:
    """Recursively flatten nested lists / object arrays into a flat list of floats."""
    if isinstance(x, (list, tuple, np.ndarray)):
        out = []
        for v in x:
            out.extend(_flatten(v))
        return out
    return [float(x)]


def camera_present(ep: pd.Series, cam: str) -> bool:
    mean = _flatten(ep[f"stats/observation.images.{cam}/mean"])
    return float(np.mean(mean)) > 0.02


def read_frames(root: Path, ep: pd.Series, cam: str, n: int) -> list:
    """Return all n frames of the episode from the concatenated mp4."""
    key = f"videos/observation.images.{cam}"
    c, f = int(ep[f"{key}/chunk_index"]), int(ep[f"{key}/file_index"])
    t0 = float(ep[f"{key}/from_timestamp"])
    path = root / f"videos/observation.images.{cam}/chunk-{c:03d}/file-{f:03d}.mp4"
    frames = [None] * n
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(max(t0 - 1.0, 0) / stream.time_base), stream=stream, backward=True)
        for fr in container.decode(stream):
            t = float(fr.pts * stream.time_base)
            i = int(round((t - t0) * FPS))
            if i < 0:
                continue
            if i >= n:
                break
            if frames[i] is None:
                frames[i] = fr.to_ndarray(format="rgb24")
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="dataset root")
    ap.add_argument("--out", type=Path, required=True, help="folder for pictures")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--cam", default="image_0")
    args = ap.parse_args()

    eps = load_meta_episodes(args.data)
    ep = eps[eps["episode_index"] == args.episode].iloc[0]
    rows = load_frames_episode(args.data, ep)
    state = np.stack(rows["observation.state"].values)
    action = np.stack(rows["action"].values)
    n = len(rows)

    task = list(ep["tasks"])[0]
    print(f"episode {args.episode}: '{task}', {n} frames")
    print("cameras present:", {c: camera_present(ep, c) for c in CAMS})

    g = state[:, 6]
    c = ph.closedness(g, closed_high=False, scale=gripper_scale(args.data))   # 0 = open .. 1 = closed
    labels = ph.label(c) if c is not None else np.zeros(n, dtype=int)  # no grasp -> all pre_grasp
    phases = [ph.PHASES[i] for i in labels]
    print("gripper:", np.round(g, 2).tolist())
    print("action[6]:", np.round(action[:, 6], 2).tolist())
    print("phases:", phases)

    out = args.out / f"ep{args.episode:05d}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # plot
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for t, p in enumerate(phases):
        for a in ax:
            a.axvspan(t - 0.5, t + 0.5, color=PHASE_COLORS[p], alpha=0.35, lw=0)
    ax[0].plot(g, "k.-", label="gripper state[6]")
    ax[0].plot(action[:, 6], "r--", alpha=0.6, label="gripper action[6]")
    ax[0].legend(); ax[0].set_title(task)
    for i, name in enumerate("xyz"):
        ax[1].plot(state[:, i] - state[0, i], ".-", label=f"{name} - {name}0")
    ax[1].legend(); ax[1].set_xlabel("frame")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.5) for c in PHASE_COLORS.values()]
    fig.legend(handles, PHASE_COLORS.keys(), loc="upper right", ncol=5)
    fig.tight_layout(); fig.savefig(out / "signals.png", dpi=110); plt.close(fig)

    # frames
    frames = read_frames(args.data, ep, args.cam, n)
    missing = sum(f is None for f in frames)
    if missing:
        print(f"WARNING: {missing} frames not decoded")

    keyframes = [0, n - 1] + [t for t in range(1, n) if phases[t] != phases[t - 1]]
    for t in sorted(set(keyframes)):
        if frames[t] is not None:
            Image.fromarray(frames[t]).save(out / f"key_f{t:03d}_{phases[t]}.jpg", quality=90)

    # contact sheet of all frames
    cols = 6
    h, w = 120, 160
    rows_n = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows_n * h), "white")
    for t, fr in enumerate(frames):
        if fr is not None:
            sheet.paste(Image.fromarray(fr).resize((w, h)), ((t % cols) * w, (t // cols) * h))
    sheet.save(out / "all_frames.jpg", quality=85)
    print("saved to", out)


if __name__ == "__main__":
    main()