"""Run the enabled per-episode generators (config "generators") over the datasets in config "episodes".

Each Question is written once per camera that has a non-blank frame at all of its frames (Bridge: up to 3 views of
the same moment). Frames go to <out>/frames/<dataset>/; records reference them relative to <out>.
"""
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from aijgen.depth import lerobot
from aijgen.generators import build_generators
from aijgen.generators.episodes import load_episodes


def generate(input_root: Path, out: Path, config: dict) -> tuple[dict[str, list[dict]], dict]:
    """({generator name: ShareGPT records}, stats). Image paths are relative to `out`."""
    generators = build_generators(config)
    ecfg = config["episodes"]
    records, stats = {g.name: [] for g in generators}, Counter()
    if not generators:
        return records, {}
    for ds_name, dcfg in ecfg["datasets"].items():
        root = input_root / ds_name
        if not (root / "meta" / "info.json").exists():
            print(f"skip {ds_name}: no {root}/meta/info.json")
            continue
        episodes = load_episodes(root, dcfg)
        print(f"{ds_name}: {len(episodes)} episodes for {[g.name for g in generators]}", flush=True)
        for n, (ep, videos) in enumerate(episodes):
            questions = [q for g in generators for q in g(ep)]
            stats[f"{ds_name} episodes {'with' if questions else 'without'} questions"] += 1
            if not questions:
                continue
            wanted = sorted({f for q in questions for f in q.frames})
            frames = {cam: lerobot.read_frames(v, wanted) for cam, v in videos.items()}
            for q in questions:
                for cam in dcfg["cameras"]:
                    imgs = [frames.get(cam, {}).get(f) for f in q.frames]
                    if any(im is None or im.std() < ecfg["min_frame_std"] for im in imgs):
                        stats[f"{ds_name} {q.kind} views skipped"] += 1
                        continue
                    paths = []
                    for f, im in zip(q.frames, imgs):
                        rel = Path("frames") / ds_name / f"ep{ep.index:06d}_f{f:04d}_{cam.split('.')[-1]}.jpg"
                        if not (out / rel).exists():
                            (out / rel).parent.mkdir(parents=True, exist_ok=True)
                            Image.fromarray(im).save(out / rel, quality=95)
                        paths.append(str(rel))
                    records[q.kind].append({
                        "messages": [{"role": "user", "content": "<image>\n" * len(paths) + q.question},
                                     {"role": "assistant", "content": q.answer}],
                        "images": paths})
                    stats[f"{ds_name} {q.kind} records"] += 1
                    stats[f"{q.kind} answer {q.answer}"] += 1
                    for k, v in q.meta.items():
                        stats[f"{q.kind} {k} {v}"] += 1
            if (n + 1) % 200 == 0:
                print(f"  {ds_name}: {n + 1}/{len(episodes)} episodes, "
                      f"{sum(map(len, records.values()))} records", flush=True)
    return records, dict(sorted(stats.items()))
