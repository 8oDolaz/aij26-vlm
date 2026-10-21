#!/usr/bin/env python3
"""Build the ShareGPT JSONL for SmolVLM2 SFT from the raw robot datasets (Bridge, Fractal, ...).

Sources, each switched on in config.yaml:
  depth_tier1  BLINK-style "which point is closer" questions (aijgen.depth.tier1), work dir <out>/depth_tier1/
  generators   per-episode questions from gripper phases etc. (aijgen.generators), frames in <out>/frames/
The records of all sources are capped per source (mix; scene_qa per template), optionally trimmed so no answer
letter dominates (letter_balance), shuffled with the seed and written to one JSONL.
Image paths are relative to the JSONL's directory, which the kit's run_pipeline.py uses as media_dir by default.

  python generate_dataset.py --input /data/raw --output /output/annotations.jsonl --config config.yaml
--output may also be a directory (annotations.jsonl goes inside). Next to the JSONL: annotations.stats.json.
"""
import argparse
import json
import re
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def depth_records(input_root: Path, out: Path, cfg: dict, relabel: bool = False) -> tuple[list[dict], dict]:
    from aijgen.depth import tier1
    work = out / "depth_tier1"
    ann_file = work / "annotations.jsonl"
    report_file = work / "report.json"
    if not relabel and ann_file.exists() and report_file.exists():
        print(f"depth_tier1: reusing existing {ann_file}", flush=True)
        with open(report_file) as f:
            report = json.load(f)
    else:
        report = tier1.generate(argparse.Namespace(input=input_root, output=work, relabel=relabel, max_episodes=None), cfg)
    records = []
    with open(ann_file) as f:
        for line in f:
            r = json.loads(line)
            r["images"] = [str(Path(p).resolve().relative_to(out.resolve())) for p in r["images"]]
            records.append(r)
    return records, {k: report.get(k, {}) for k in ("funnel", "final_by_dataset", "final_by_camera", "timing")}


def letter(r: dict) -> str | None:
    """The answer if it is a bare option letter, else None."""
    a = r["messages"][1]["content"]
    return a if re.fullmatch(r"[A-J]", a) else None


def letter_shares(records: list[dict]) -> dict:
    """Share of each answer letter over all records answered with a bare letter (docs/vlm_data_priorities.md #0:
    a skewed mix teaches a default letter)."""
    letters = Counter(filter(None, map(letter, records)))
    n = sum(letters.values())
    return {"letter_records": n, **{k: round(v / max(n, 1), 4) for k, v in sorted(letters.items())}}


def sample(recs: list, cap: int | None, seed: int, name: str) -> list:
    if cap is None or len(recs) <= cap:
        return recs
    keep = np.random.default_rng([seed, zlib.crc32(name.encode())]).choice(len(recs), cap, replace=False)
    return [recs[i] for i in sorted(keep)]


def apply_mix(sources: dict[str, list], mix: dict, seed: int) -> tuple[dict[str, list], dict]:
    """Cap each source with the seed. Sources named "<generator>/<template>" (scene_qa) are capped per template when
    mix[<generator>] is a mapping {template: cap} (a template left out is kept whole); otherwise mix[<generator>]
    caps all its templates together."""
    groups = {}
    for name, recs in sources.items():
        head, _, sub = name.partition("/")
        groups.setdefault(head, {})[sub] = recs
    kept, stats = {}, {}
    for head, subs in groups.items():
        caps = mix.get(head)
        if isinstance(caps, dict):
            parts = {(f"{head}/{s}" if s else head): (recs, caps.get(s)) for s, recs in subs.items()}
        else:
            parts = {head: ([r for recs in subs.values() for r in recs], caps)}
        for name, (recs, cap) in parts.items():
            kept[name] = sample(recs, cap, seed, name)
            stats[name] = {"generated": len(recs), "kept": len(kept[name])}
            if isinstance(caps, dict) and cap is not None and len(recs) < cap:
                stats[name]["short_of_cap"] = cap - len(recs)
    return kept, stats


def balance_letters(kept: dict[str, list], max_share: float, min_keep: float, seed: int) -> dict:
    """Keep any answer letter from going over max_share of the bare-letter answers.

    Two-option sources (every letter answer A or B: BLINK yes/no, box A/B) are what push A and B above the 4-option
    questions' 25%. While A or B is the most frequent letter and over max_share, records of those sources are dropped
    in random order, so each source keeps its own A/B balance; no source loses more than 1 - min_keep of its records.
    Changes `kept` in place; returns what was dropped per source."""
    counts = Counter(filter(None, (letter(r) for recs in kept.values() for r in recs)))
    total = sum(counts.values())
    two = [n for n, recs in kept.items() if (ls := {letter(r) for r in recs} - {None}) and ls <= {"A", "B"}]
    pool = [(n, i) for n in two for i, r in enumerate(kept[n]) if letter(r)]
    floor = {n: int(np.ceil(min_keep * len(kept[n]))) for n in two}
    left = {n: len(kept[n]) for n in two}
    drop = {n: set() for n in two}
    for k in np.random.default_rng([seed, zlib.crc32(b"letter_balance")]).permutation(len(pool)):
        top, top_n = counts.most_common(1)[0] if counts else ("", 0)
        if top not in ("A", "B") or top_n / total <= max_share:
            break
        n, i = pool[k]
        if left[n] <= floor[n]:
            continue
        drop[n].add(i)
        left[n] -= 1
        counts[letter(kept[n][i])] -= 1
        total -= 1
    for n in two:
        kept[n] = [r for i, r in enumerate(kept[n]) if i not in drop[n]]
    top, top_n = counts.most_common(1)[0] if counts else ("", 0)
    return {"dropped": {n: len(d) for n, d in drop.items() if d}, "max_letter": top,
            "max_share": round(top_n / max(total, 1), 4), "target": max_share}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="root with the raw datasets (bridge/, fractal/, ...)")
    parser.add_argument("--output", type=Path, required=True, help="annotations.jsonl path, or a directory for it")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--relabel", action="store_true", help="force regeneration even if cached files exist")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    jsonl = args.output if args.output.suffix == ".jsonl" else args.output / "annotations.jsonl"
    out = jsonl.parent
    out.mkdir(parents=True, exist_ok=True)

    sources, stats = {}, {}
    if config.get("depth_tier1", {}).get("enabled"):
        sources["depth_tier1"], stats["depth_tier1"] = depth_records(args.input, out, config["depth_tier1"], relabel=args.relabel)
    if any(g.get("enabled") for g in config.get("generators", {}).values()):
        from aijgen.generators.run import generate
        by_generator, stats["generators"] = generate(args.input, out, config, relabel=args.relabel)
        sources.update(by_generator)

    seed = int(config["seed"])
    kept, stats["mix"] = apply_mix(sources, config.get("mix") or {}, seed)
    lb = config.get("letter_balance") or {}
    if lb.get("max_share") is not None:
        stats["letter_balance"] = balance_letters(kept, float(lb["max_share"]), float(lb.get("min_keep", 0.5)), seed)
        for name, recs in kept.items():
            stats["mix"][name]["kept"] = len(recs)
    records = [r for recs in kept.values() for r in recs]
    records = [records[i] for i in np.random.default_rng(seed).permutation(len(records))]
    stats["answer_letters"] = letter_shares(records)

    with open(jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["total"] = len(records)
    jsonl.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats.get("mix", {}), indent=1))
    print("answer letters:", json.dumps(stats["answer_letters"]))
    print(f"wrote {len(records)} records to {jsonl}")


if __name__ == "__main__":
    main()
