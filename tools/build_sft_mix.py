#!/usr/bin/env python3
"""Build a self-contained VLM SFT mix from generated datasets: every record of --scene plus samples of the depth_tier1
and phase_qa records of --old (a generate_dataset.py output where depth images are under depth_tier1/).

The images the mix uses are copied next to it, to <out dir>/images/<source dir name>/<path in the source>, and the
records point there (paths relative to the out dir, which is the media_dir for training). The out dir can then be
moved or used on its own. Every image is checked before anything is written.

Answer letters (docs/vlm_data_priorities.md #0): two-option records (BLINK yes/no and box questions, depth point
pairs) push A and B over the four-option questions' 25%. While A or B is over --max-share of the bare-letter answers,
two-option records are dropped in random order (so their own A/B balance holds), keeping at least --min-keep of them.

Run in the dev container:
  python tools/build_sft_mix.py --scene /data/gen/scene14k/annotations.jsonl \
      --old /data/gen/d2_depth_phase/annotations.jsonl --depth 3000 --phase 1000 \
      --out /data/gen/d4_mix/annotations.jsonl
Writes the JSONL, <out dir>/images/ and <out>.stats.json (counts per source, drops, letter shares, images copied).
"""
import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

OPTION = re.compile(r"^\(?([A-J])[).] ", re.M)


def load(path: Path) -> list[dict]:
    """Records of a JSONL; "_src" holds each image's absolute path (paths in the file are relative to its dir) and
    "images" the path it gets in the mix."""
    base = path.parent.resolve()
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            r["_rel"] = list(r["images"])
            r["_src"] = [str(base / p) for p in r["images"]]
            r["images"] = [f"images/{base.name}/{p}" for p in r["images"]]
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
    ap.add_argument("--out", type=Path, required=True, help="annotations.jsonl of the mix; images go next to it")
    ap.add_argument("--max-share", type=float, default=0.27)
    ap.add_argument("--min-keep", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    scene, old = load(args.scene), load(args.old)
    if args.scene.parent.resolve().name == args.old.parent.resolve().name:
        raise SystemExit("--scene and --old live in directories with the same name; their images would collide")
    depth = [r for r in old if r["_rel"][0].startswith("depth_tier1/")]
    phase = [r for r in old if not r["_rel"][0].startswith("depth_tier1/")]
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
    records = [records[i] for i in rng.permutation(len(records))]

    copies = {dst: src for r in records for src, dst in zip(r["_src"], r["images"])}
    missing = [src for src in copies.values() if not Path(src).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} images not found, e.g. {missing[:3]}")

    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = size = 0
    for dst, src in copies.items():
        target = out_dir / dst
        if not target.is_file() or target.stat().st_size != Path(src).stat().st_size:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            copied += 1
        size += target.stat().st_size
    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}, ensure_ascii=False) + "\n")

    stats = {"sources": {k: len(v) for k, v in sources.items()}, "available": {"depth": len(depth), "phase": len(phase)},
             "two_option_records": len(two), "dropped_two_option": len(drop), "total": len(records),
             "images": {"referenced": len(copies), "copied_now": copied, "gb": round(size / 1e9, 2)},
             "media_dir": str(out_dir.resolve()), "letters_before": before, "letters": shares(records)}
    args.out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
