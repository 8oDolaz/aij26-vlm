"""LeRobot v3 episodes as generators.Episode: instruction, state, gripper closedness and grasp phases.

The gripper scale is the dataset's p1..p99 of the gripper dim over all data/*.parquet, as in tools/fetch_data.py
(Fractal's meta/stats.json has no quantiles).
"""
import glob
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from aijgen import phases as ph
from aijgen.depth import lerobot
from aijgen.generators.base import Episode


def load_states(root: Path) -> dict[int, np.ndarray]:
    """observation.state per episode index, (length, state_dim), in frame order."""
    eps, frames, states = [], [], []
    for f in sorted(glob.glob(str(root / "data" / "*" / "*.parquet"))):
        t = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state"])
        eps.append(t["episode_index"].to_numpy())
        frames.append(t["frame_index"].to_numpy())
        states.append(np.stack(t["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float64))
    ep, fr, st = np.concatenate(eps), np.concatenate(frames), np.concatenate(states)
    order = np.lexsort((fr, ep))
    ep, st = ep[order], st[order]
    cut = np.flatnonzero(np.diff(ep)) + 1
    return {int(e[0]): s for e, s in zip(np.split(ep, cut), np.split(st, cut))}


def evenly(items: list, n: int | None) -> list:
    if n is None or len(items) <= n:
        return items
    return [items[i] for i in np.linspace(0, len(items) - 1, n).round().astype(int)]


def load_episodes(root: Path, cfg: dict) -> list[tuple[Episode, dict[str, lerobot.Episode]]]:
    """(Episode, video per camera) for up to cfg["max_episodes"] episodes, evenly spread over those with a video for
    the first camera on disk and at least cfg["min_length"] frames. Sorted by episode index."""
    cams = cfg["cameras"]
    videos = {c: {e.index: e for e in lerobot.list_episodes(root, c)} for c in cams}
    candidates = [i for i, e in sorted(videos[cams[0]].items()) if e.length >= cfg["min_length"]]
    chosen = evenly(candidates, cfg.get("max_episodes"))
    states = load_states(root)
    dim = cfg["gripper_dim"]
    lo, hi = np.percentile(np.concatenate([s[:, dim] for s in states.values()]), [1, 99])
    out = []
    for i in chosen:
        state = states[i]
        c = ph.closedness(state[:, dim], cfg["closed_high"], hi - lo)
        ep = Episode(dataset=root.name, index=i, task=videos[cams[0]][i].task, length=len(state),
                     fps=videos[cams[0]][i].fps, state=state, closedness=c,
                     labels=ph.label(c) if c is not None else None)
        out.append((ep, {cam: v[i] for cam, v in videos.items() if i in v}))
    return out
