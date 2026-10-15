"""Arm and gripper speed around given frames: does a blurred gripper come from fast motion?

python3 tools/arm_speed.py --data data/raw/bridge --frames 39:13 47:19 13860:32
python3 tools/arm_speed.py --data data/raw/bridge --frames 39:13 --window 5
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from explore_episode import gripper_scale, load_frames_episode, load_meta_episodes

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "submission" / "src"))
from aijgen import phases as ph


def speeds(x: np.ndarray, fps: float) -> np.ndarray:
    """Per-frame speed of a signal (rows = frames), from the previous and the next frame, units per second."""
    n = len(x)
    out = np.zeros(n)
    for t in range(n):
        a, b = max(t - 1, 0), min(t + 1, n - 1) #предыдущий и следующий кадры
        if b > a:
            out[t] = np.linalg.norm(np.atleast_1d(x[b] - x[a])) / ((b - a) / fps)
    return out


def parse_pair(s: str) -> tuple[int, int]:
    """'39:13' -> (39, 13): episode and frame."""
    e, f = s.split(":")
    return int(e), int(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="Bridge dataset root")
    ap.add_argument("--frames", type=parse_pair, nargs="+", required=True,
                    help="episode:frame pairs with a blurred gripper, e.g. 39:13 47:19")
    ap.add_argument("--range", type=int, default=3, help="frames to show before and after")
    args = ap.parse_args()

    fps = float(json.load(open(args.data / "meta" / "info.json"))["fps"])
    scale = gripper_scale(args.data)
    eps = load_meta_episodes(args.data)

    summary = []
    for e, f in args.frames:
        found = eps[eps["episode_index"] == e]
        if found.empty:
            print(f"episode {e}: not in meta")
            continue
        ep = found.iloc[0]
        rows = load_frames_episode(args.data, ep)
        state = np.stack(rows["observation.state"].values)
        n = len(state)
        if not 0 <= f < n:
            print(f"episode {e}: frame {f} out of range 0..{n - 1}")
            continue

        arm = speeds(state[:, 0:3], fps)                       # m/s
        c = ph.closedness(state[:, 6], closed_high=False, scale=scale)
        grip = speeds(c, fps) if c is not None else np.zeros(n)  # closedness units per second
        labels = ph.label(c) if c is not None else None
        median = float(np.median(arm))

        print(f"\nepisode {e}, frame {f}/{n}: {list(ep['tasks'])[0]!r}")
        print(f"  arm speed over the episode: median {median:.3f} m/s, max {arm.max():.3f} m/s")
        print(f"  {'frame':>5}  {'speed m/s':>8}  {'x median':>8}  {'grip /s':>8}  {'closed':>6}  phase")
        for t in range(max(f - args.range, 0), min(f + args.range, n - 1) + 1):
            phase = ph.PHASES[labels[t]] if labels is not None else "-"
            closed = f"{c[t]:.2f}" if c is not None else "-"
            mark = "  <-- blurred" if t == f else ""
            print(f"  {t:>5}  {arm[t]:>8.3f}  {arm[t] / max(median, 1e-9):>8.1f}  {grip[t]:>8.2f}  "
                  f"{closed:>6}  {phase}{mark}")
        summary.append((e, f, arm[f], arm[f] / max(median, 1e-9), grip[f]))

    if summary:
        print("\nblurred frames:")
        print(f"  {'episode:frame':>14}  {'arm m/s':>8}  {'x median':>8}  {'grip /s':>8}")
        for e, f, v, r, g in summary:
            print(f"  {f'{e}:{f}':>14}  {v:>8.3f}  {r:>8.1f}  {g:>8.2f}")


if __name__ == "__main__":
    main()