#!/usr/bin/env python3
"""Build a VLM SFT mix from generated datasets: every record of --scene plus samples of the depth_tier1 and phase_qa
records of --old (a generate_dataset.py output where depth images are under depth_tier1/), with image paths re-rooted
at --media-root so one media_dir serves all of them.

Answer letters (docs/vlm_data_priorities.md #0): two-option records (BLINK yes/no and box questions, depth point
pairs) push A and B over the four-option questions' 25%. While A or B is over --max-share of the bare-letter answers,
two-option records are dropped in random order (so their own A/B balance holds), keeping at least --min-keep of them.

Run in the dev container:
  python tools/build_sft_mix.py --scene /data/gen/scene14k/annotations.jsonl \
      --old /data/gen/d2_depth_phase/annotations.jsonl --depth 3000 --phase 1000 \
      --media-root /data/gen --out /data/gen/d4_mix/annotations.jsonl
Writes the JSONL and <out>.stats.json (counts per source, drops, letter shares).
"""
import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np

OPTION = re.compile(r"^\(?([A-J])[).] ", re.M)


def load(path: Path, media_root: Path) -> list[dict]:
    """Records of a JSONL, image paths made relative to media_root (they are relative to the JSONL's dir)."""
    prefix = Path(os.path.relpath(path.parent.resolve(), media_root.resolve()))
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            r["_orig_images"] = r["images"]
            r["images"] = [str(prefix / p) for p in r["images"]]
            out.append(r)
    return out


def letter(r: dict) -> str | None:
    a = r["messages"][1]["content"]
    return a if re.fullmatch(r"[A-J]", a) else None


def n_options(r: dict) -> int:
    return len(set(OPTION.findall(r["messages"][0]["content"])))


def shares(records) -> dict:
    c = Counter(filter(None, map(letter, records)))
    n = sum(c.values())
    return {"letter_records": n, **{k: round(v / max(n, 1), 4) for k, v in sorted(c.items())}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--depth", type=int, default=3000)
    ap.add_argument("--phase", type=int, default=1000)
    ap.add_argument("--media-root", type=Path, default=Path("/data/gen"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-share", type=float, default=0.27)
    ap.add_argument("--min-keep", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    scene = load(args.scene, args.media_root)
    old = load(args.old, args.media_root)
    depth = [r for r in old if r["_orig_images"][0].startswith("depth_tier1/")]
    phase = [r for r in old if not r["_orig_images"][0].startswith("depth_tier1/")]
    pick = lambda recs, n: [recs[i] for i in sorted(rng.choice(len(recs), min(n, len(recs)), replace=False))]
    sources = {"scene": scene, "depth": pick(depth, args.depth), "phase": pick(phase, args.phase)}
    records = [r for recs in sources.values() for r in recs]
    before = shares(records)

    counts = Counter(filter(None, map(letter, records)))
    total = sum(counts.values())
    two = [i for i, r in enumerate(records) if letter(r) and n_options(r) == 2]
    floor = int(np.ceil(args.min_keep * len(two)))
    drop = set()
    for i in rng.permutation(two):
        top, top_n = counts.most_common(1)[0]
        if top not in ("A", "B") or top_n / total <= args.max_share or len(two) - len(drop) <= floor:
            break
        drop.add(int(i))
        counts[letter(records[i])] -= 1
        total -= 1
    records = [r for i, r in enumerate(records) if i not in drop]
    for r in records:
        r.pop("_orig_images", None)
    records = [records[i] for i in rng.permutation(len(records))]

    missing = [p for r in records[:200] for p in r["images"] if not (args.media_root / p).is_file()]
    if missing:
        raise SystemExit(f"images not found under {args.media_root}, e.g. {missing[:3]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"sources": {k: len(v) for k, v in sources.items()}, "available": {"depth": len(depth), "phase": len(phase)},
             "two_option_records": len(two), "dropped_two_option": len(drop), "total": len(records),
             "letters_before": before, "letters": shares(records)}
    args.out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
