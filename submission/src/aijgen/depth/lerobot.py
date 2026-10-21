"""Episode index and frame decoding for LeRobot v3 datasets, where one mp4 holds many consecutive episodes."""
import glob
import json
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Episode:
    dataset: str
    index: int
    length: int
    task: str
    video_key: str
    video: Path
    t0: float  # episode start, seconds into the mp4
    fps: float


def list_episodes(root: Path, video_key: str) -> list[Episode]:
    """Episodes whose mp4 for video_key is on disk, sorted by episode index."""
    fps = float(json.loads((root / "meta" / "info.json").read_text())["fps"])
    vk = f"videos/{video_key}"
    cols = ["episode_index", "length", "tasks", f"{vk}/chunk_index", f"{vk}/file_index", f"{vk}/from_timestamp"]
    meta = pd.concat(pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(root / "meta/episodes/*/*.parquet"))))
    out = []
    for idx, length, tasks, chunk, file, t0 in zip(*(meta[c] for c in cols)):
        video = root / "videos" / video_key / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"
        if video.exists():
            out.append(Episode(root.name, int(idx), int(length), str(list(tasks)[0]), video_key, video, float(t0), fps))
    return sorted(out, key=lambda e: e.index)


def read_frames(ep: Episode, indices: list[int]) -> dict[int, np.ndarray]:
    """RGB frames of one episode by frame index (0 = episode start). Missing indices are left out."""
    want = set(indices)
    out = {}
    with av.open(str(ep.video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        start = ep.t0 + min(want) / ep.fps
        container.seek(int(max(start - 1.0, 0) / stream.time_base), stream=stream)  # keyframe at or before start
        for frame in container.decode(stream):
            i = round((frame.time - ep.t0) * ep.fps)
            if i > max(want):
                break
            if i in want and i not in out:
                out[i] = frame.to_ndarray(format="rgb24")
    return out
