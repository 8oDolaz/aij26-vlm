#!/usr/bin/env python3
"""Download the Bridge video files that hold every episode with real image_0, image_1 and image_2 frames.

LeRobot v3 packs many consecutive episodes into one ~200 MB mp4 per camera, so the unit is a file, not an episode.
The episode metadata (meta/episodes, fetched in full by tools/fetch_data.py) says which file and time range hold
each episode's camera, and its per-episode image stats say whether the camera is real: Bridge pads cameras missing
from a session with black video, so a camera counts as real when its stats max > 0.05 (as in fetch_data.py).

At the pinned revision: 26,050 of 50,415 episodes have all three cameras (6,994 of them also image_3, not fetched).
They sit in 81 image_0 files and all 71 + 71 image_1/image_2 files: 44.1 GB. image_0 files also hold
single-camera episodes in between; there is no way to skip those bytes.

Files already on disk are skipped, so the download can be interrupted and re-run.

Run in the dev container, from the repo root:
  docker compose run --rm dev python tools/fetch_bridge_multiview.py --dry-run
  docker compose run --rm dev python tools/fetch_bridge_multiview.py
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

REPO, REVISION = "nvidia/BridgeData2_LeRobot_v3", "b96f7216e3cff58007884656a81584c857c185ae"
VIEWS = ["observation.images.image_0", "observation.images.image_1", "observation.images.image_2"]
REAL_MAX = 0.05


def cam_max(v) -> float:
    """Max of a per-episode image stat, stored as nested object arrays (channel x 1 x 1)."""
    if v is None:
        return 0.0
    if isinstance(v, np.ndarray) and v.dtype == object or isinstance(v, (list, tuple)):
        return max((cam_max(x) for x in v), default=0.0)
    return float(np.max(v))


def needed_files(root: Path) -> tuple[list[str], int]:
    """(repo paths of the mp4s holding all multi-view episodes, number of such episodes)."""
    cols = ["episode_index"]
    for k in VIEWS:
        cols += [f"stats/{k}/max", f"videos/{k}/chunk_index", f"videos/{k}/file_index"]
    meta = pd.concat(pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(root / "meta/episodes/*/*.parquet"))))
    real = np.ones(len(meta), bool)
    for k in VIEWS:
        real &= meta[f"stats/{k}/max"].map(cam_max).to_numpy() > REAL_MAX
    files = set()
    for k in VIEWS:
        for c, f in zip(meta[f"videos/{k}/chunk_index"][real], meta[f"videos/{k}/file_index"][real]):
            files.add(f"videos/{k}/chunk-{int(c):03d}/file-{int(f):03d}.mp4")
    return sorted(files), int(real.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/raw/bridge"), help="Bridge dir with meta/ on disk")
    ap.add_argument("--dry-run", action="store_true", help="only print what would be downloaded")
    args = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download

    files, n_eps = needed_files(args.root)
    sizes = {e.path: e.size for e in HfApi().get_paths_info(REPO, files, repo_type="dataset", revision=REVISION)}
    todo = [p for p in files if not (args.root / p).exists()]
    print(f"{n_eps} episodes with {', '.join(v.split('.')[-1] for v in VIEWS)}: {len(files)} files, "
          f"{sum(sizes.values()) / 1e9:.1f} GB; {len(files) - len(todo)} on disk, "
          f"{len(todo)} to download ({sum(sizes[p] for p in todo) / 1e9:.1f} GB)")
    if args.dry_run:
        return
    for n, path in enumerate(todo, 1):
        hf_hub_download(REPO, path, repo_type="dataset", revision=REVISION, local_dir=args.root)
        print(f"[{n}/{len(todo)}] {path} ({sizes[path] / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
