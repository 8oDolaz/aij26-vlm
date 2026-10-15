#!/usr/bin/env python3
"""Build the ShareGPT JSONL for SmolVLM2 SFT from the raw robot datasets (Bridge, Fractal, ...).

Sources, each switched on in config.yaml:
  depth_tier1  BLINK-style "which point is closer" questions (aijgen.depth.tier1), work dir <out>/depth_tier1/
  generators   per-episode questions from gripper phases etc. (aijgen.generators), frames in <out>/frames/
The records of all sources are capped per source (mix), shuffled with the seed and written to one JSONL.
Image paths are relative to the JSONL's directory, which the kit's run_pipeline.py uses as media_dir by default.

  python generate_dataset.py --input /data/raw --output /output/annotations.jsonl --config config.yaml
--output may also be a directory (annotations.jsonl goes inside). Next to the JSONL: annotations.stats.json.
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def depth_records(input_root: Path, out: Path, cfg: dict) -> tuple[list[dict], dict]:
    from aijgen.depth import tier1
    work = out / "depth_tier1"
    report = tier1.generate(argparse.Namespace(input=input_root, output=work, relabel=False, max_episodes=None), cfg)
    records = []
    with open(work / "annotations.jsonl") as f:
        for line in f:
            r = json.loads(line)
            r["images"] = [str(Path(p).resolve().relative_to(out.resolve())) for p in r["images"]]
            records.append(r)
    return records, {k: report[k] for k in ("funnel", "final_by_dataset", "final_by_camera", "timing")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="root with the raw datasets (bridge/, fractal/, ...)")
    parser.add_argument("--output", type=Path, required=True, help="annotations.jsonl path, or a directory for it")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    jsonl = args.output if args.output.suffix == ".jsonl" else args.output / "annotations.jsonl"
    out = jsonl.parent
    out.mkdir(parents=True, exist_ok=True)

    sources, stats = {}, {}
    if config.get("depth_tier1", {}).get("enabled"):
        sources["depth_tier1"], stats["depth_tier1"] = depth_records(args.input, out, config["depth_tier1"])
    if any(g.get("enabled") for g in config.get("generators", {}).values()):
        from aijgen.generators.run import generate
        by_generator, stats["generators"] = generate(args.input, out, config)
        sources.update(by_generator)

    seed = int(config["seed"])
    mix = config.get("mix") or {}
    records = []
    for name, recs in sources.items():
        cap = mix.get(name)
        if cap is not None and len(recs) > cap:
            keep = np.random.default_rng([seed, zlib.crc32(name.encode())]).choice(len(recs), cap, replace=False)
            recs = [recs[i] for i in sorted(keep)]
        stats.setdefault("mix", {})[name] = {"generated": len(sources[name]), "kept": len(recs)}
        records += recs
    records = [records[i] for i in np.random.default_rng(seed).permutation(len(records))]

    with open(jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["total"] = len(records)
    jsonl.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats.get("mix", {}), indent=1))
    print(f"wrote {len(records)} records to {jsonl}")


if __name__ == "__main__":
    main()
