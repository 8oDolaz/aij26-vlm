"""Run the question generators on downloaded Bridge episodes and save a small preview dataset.

python tools/preview_generators.py --n 20
Output: data/gen/preview/annotations.jsonl and data/gen/preview/images/
"""
import argparse
import json
import shutil
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from functools import lru_cache
import logging
from tqdm import tqdm

from explore_episode import gripper_scale, load_meta_episodes, read_frames

REPO = Path(__file__).resolve().parents[1]           # aij26-vlm/
ROOT = REPO / "data" / "raw" / "bridge"              
CONFIG = REPO / "submission" / "config.yaml"         # generator settings
OUT = REPO / "data" / "generators_output" / "test_run"  # output: annotations.jsonl + images/
CAM = "image_0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("run_generators")

sys.path.insert(0, str(REPO / "submission" / "src"))
from aijgen import phases as ph
from aijgen.generators import Episode, build_generators

def data_path(ep: pd.Series) -> Path:
    """Parquet file with this episode's frames."""
    c, f = int(ep["data/chunk_index"]), int(ep["data/file_index"])
    return ROOT / "data" / f"chunk-{c:03d}" / f"file-{f:03d}.parquet"

def video_path(ep: pd.Series) -> Path:
    """mp4 file with this episode's frames from camera CAM."""
    key = f"videos/observation.images.{CAM}"
    c, f = int(ep[f"{key}/chunk_index"]), int(ep[f"{key}/file_index"])
    return ROOT / "videos" / f"observation.images.{CAM}" / f"chunk-{c:03d}" / f"file-{f:03d}.mp4"

@lru_cache(maxsize=2)
def read_table(path: Path) -> pd.DataFrame:
    """One data parquet; cached so it is read once, not once per episode."""
    return pd.read_parquet(path, columns=["episode_index", "frame_index", "observation.state"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="how many downloaded episodes to process")
    args = ap.parse_args()

    config = yaml.safe_load(open(CONFIG))
    generators = build_generators(config) #lsit of all enabled generators
    log.info("generators: %s", [g.name for g in generators])
    scale = gripper_scale(ROOT)

    eps = load_meta_episodes(ROOT) #meta table
    on_disk = [data_path(ep).exists() and video_path(ep).exists() for _, ep in eps.iterrows()]
    log.info("%d of %d episodes are downloaded", sum(on_disk), len(on_disk))
    eps = eps[on_disk].sort_values("episode_index").head(args.n)
    log.info("processing %d episodes -> %s", len(eps), OUT)

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)

    records, stats = [], Counter()
    progress = tqdm(eps.iterrows(), total=len(eps), desc="episodes", unit="ep")
    for _, row in progress: #по эпизодам 
        e = int(row["episode_index"])
        data = read_table(data_path(row)) #episode_inde, frame_index, observation.state
        rows = data[data["episode_index"] == e].sort_values("frame_index")
        state = np.stack(rows["observation.state"].values)
        close_state = ph.closedness(state[:, 6], closed_high=False, scale=scale)

        ep = Episode(dataset="bridge", 
                     index=e, 
                     task=list(row["tasks"])[0],
                     length=len(state), 
                     state=state, 
                     closedness=close_state,
                     labels=ph.label(close_state) if close_state is not None else None)

        questions = []
        for g in generators:          # каждый включённый генератор
            for q in g(ep):           # g(ep) → __call__ → rng → generate → список Question
                questions.append(q)

        stats["episodes with questions" if questions else "episodes without questions"] += 1
        if not questions:
            continue

        frames = read_frames(ROOT, row, CAM, ep.length)
        for q in questions:
            images = []
            for frame in q.frames:
                phase = ph.PHASES[ep.labels[frame]]
                name = f"images/bridge_ep{e:06d}_fr{frame:03d}_{CAM}_{phase}.jpg"
                if not (OUT / name).exists():
                    Image.fromarray(frames[frame]).save(OUT / name, quality=90)

                images.append(name)

            records.append({
                "messages": [
                    {"role": "user", "content": "<image>\n" * len(images) + q.question},
                    {"role": "assistant", "content": q.answer},
                ],
                "images": images,
            })

            stats[f"{q.kind} answer {q.answer}"] += 1 #cколько овтетов с этой буквой
            stats[f"{q.kind} phase {q.meta.get('phase')}"] += 1 #сколько вопросов по каждой фазе
        progress.set_postfix(questions=len(records))

    with open(OUT / "annotations.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log.info("%d questions -> %s", len(records), OUT / "annotations.jsonl")
    for key, value in sorted(stats.items()):
        log.info("  %s: %d", key, value)


if __name__ == "__main__":
    main()