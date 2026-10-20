#!/usr/bin/env python3
"""Inspect and validate semantic prompt splitting across LeRobot raw datasets or files.

Usage:
  # Scan a specific dataset (e.g. Bridge):
  python tools/check_prompts.py --dataset data/raw/bridge

  # Scan all datasets in data/raw:
  python tools/check_prompts.py --dataset data/raw

  # Print only warnings / questionable prompts:
  python tools/check_prompts.py --dataset data/raw/bridge --warnings-only

  # Print all parsed prompts to stdout:
  python tools/check_prompts.py --dataset data/raw/bridge --show-all

  # Save detailed report to file:
  python tools/check_prompts.py --dataset data/raw/bridge --out runs/prompts_bridge.txt

  # Test a single prompt directly:
  python tools/check_prompts.py --task "take the glass and insert it into the hole"
"""
import argparse
import glob
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "submission" / "src"))

from aijgen.objects.grounding import parse_task_objects

# ANSI Colors
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"

SUSPICIOUS_TOKENS = {
    "and", "then", "with", "it", "them", "this", "that", "the", "a", "an",
    "put", "place", "take", "pick", "move", "drop", "slide", "insert",
    "push", "pull", "open", "close", "lift", "reach", "grasp", "grab",
    "between", "into", "onto", "from", "inside", "under"
}


def read_tasks_from_parquet(path: Path) -> list[str]:
    """Reads task strings from a parquet file using pyarrow or pandas."""
    tasks = []
    # 1. Try pyarrow
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        col_names = [c for c in ("tasks", "task", "task_name", "instruction") if c in table.column_names]
        if col_names:
            for item in table[col_names[0]].to_pylist():
                if isinstance(item, list):
                    for sub in item:
                        if sub and isinstance(sub, str) and sub.strip():
                            tasks.append(sub.strip())
                elif isinstance(item, str) and item.strip():
                    tasks.append(item.strip())
            return tasks
    except Exception:
        pass

    # 2. Try pandas
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        for col_name in ("tasks", "task", "task_name", "instruction"):
            if col_name in df.columns:
                for item in df[col_name].dropna():
                    if isinstance(item, (list, tuple)):
                        for sub in item:
                            if sub and isinstance(sub, str) and sub.strip():
                                tasks.append(sub.strip())
                    elif isinstance(item, str) and item.strip():
                        tasks.append(item.strip())
                break
    except Exception:
        pass

    return tasks


def extract_dataset_tasks(dataset_dir: Path) -> set[str]:
    """Extracts all unique task prompt strings from a dataset directory."""
    tasks = set()

    # A. tasks.jsonl
    tasks_jsonl = dataset_dir / "meta" / "tasks.jsonl"
    if tasks_jsonl.exists():
        try:
            for line in tasks_jsonl.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                for key in ("task", "task_name", "instruction"):
                    if key in data and isinstance(data[key], str) and data[key].strip():
                        tasks.add(data[key].strip())
                        break
        except Exception:
            pass

    # B. tasks.parquet
    tasks_parquet = dataset_dir / "meta" / "tasks.parquet"
    if tasks_parquet.exists():
        tasks.update(read_tasks_from_parquet(tasks_parquet))

    # C. meta/episodes/*/*.parquet
    episode_files = sorted(glob.glob(str(dataset_dir / "meta/episodes/*/*.parquet")))
    for ep_file in episode_files:
        tasks.update(read_tasks_from_parquet(Path(ep_file)))

    # D. Fallback check: selection.json
    selection_json = dataset_dir / "selection.json"
    if selection_json.exists():
        try:
            data = json.loads(selection_json.read_text(encoding="utf-8"))
            if "instructions" in data:
                tasks.update(str(s).strip() for s in data["instructions"] if str(s).strip())
        except Exception:
            pass

    return tasks


def find_all_datasets(root: Path) -> dict[str, set[str]]:
    """Discovers datasets in root (or root itself) and extracts tasks for each."""
    results = {}

    # Check if root itself is a dataset
    if (root / "meta").exists():
        tasks = extract_dataset_tasks(root)
        if tasks:
            results[root.name] = tasks
        return results

    # Search subdirectories
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and (sub / "meta").exists():
            tasks = extract_dataset_tasks(sub)
            if tasks:
                results[sub.name] = tasks

    # If nothing found, try recursive search for meta directories
    if not results:
        for meta_dir in sorted(root.glob("**/meta")):
            d = meta_dir.parent
            tasks = extract_dataset_tasks(d)
            if tasks:
                results[d.name] = tasks

    return results


def check_candidate_quality(task: str, candidates: list[str]) -> list[str]:
    """Returns warning messages if any candidates contain suspicious words or anomalies."""
    warnings = []
    if not candidates:
        warnings.append("EMPTY_RESULT: No objects extracted from instruction")
        return warnings

    for c in candidates:
        words = c.lower().split()
        for w in words:
            if w in SUSPICIOUS_TOKENS:
                if w == "and" and any(color in c.lower() for color in ("black", "white", "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "gray", "grey")):
                    continue
                warnings.append(f"LEAKY_TOKEN: '{w}' found in candidate '{c}'")
        if len(c) > 35:
            warnings.append(f"OVERLY_LONG: candidate '{c}' has {len(c)} characters")

    return warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, help="Path to raw dataset root (e.g. data/raw/bridge or data/raw)")
    parser.add_argument("--file", type=Path, help="Path to a text or JSONL file containing tasks (one per line)")
    parser.add_argument("--task", type=str, help="Single task instruction string to test directly")
    parser.add_argument("--out", type=Path, help="Optional output file to save the full report")
    parser.add_argument("--warnings-only", action="store_true", help="Display only flagged prompts / warnings")
    parser.add_argument("--show-all", action="store_true", help="Print all parsed prompts to stdout")
    parser.add_argument("--sample", type=int, default=10, help="Number of sample prompts to display in terminal")
    args = parser.parse_args()

    # Single task test mode
    if args.task:
        candidates = parse_task_objects(args.task)
        warnings = check_candidate_quality(args.task, candidates)
        print(f"\n{BOLD}Task:{RESET} \"{args.task}\"")
        print(f"{BOLD}Parsed Candidates:{RESET} {candidates}")
        if warnings:
            print(f"{RED}{BOLD}Warnings:{RESET}")
            for w in warnings:
                print(f"  - {RED}{w}{RESET}")
        else:
            print(f"{GREEN}{BOLD}Status: Clean split!{RESET}")
        return

    # File input mode
    datasets_tasks = {}
    if args.file:
        tasks = set()
        for line in args.file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    d = json.loads(line)
                    t = d.get("task") or d.get("task_name") or d.get("instruction")
                    if t:
                        tasks.add(str(t).strip())
                except Exception:
                    pass
            elif line.startswith('"') and ' -> ' in line:
                import re as re_m
                m = re_m.match(r'^"(.*)"\s+->\s+', line)
                if m:
                    tasks.add(m.group(1))
            elif not line.startswith("=") and not line.startswith("*") and not line.startswith("-") and not line.startswith("[") and not line.startswith("-->"):
                tasks.add(line)
        datasets_tasks[args.file.name] = tasks
    elif args.dataset:
        datasets_tasks = find_all_datasets(args.dataset)
        if not datasets_tasks:
            sys.exit(f"Error: No datasets or metadata found in {args.dataset}")
    else:
        # Default fallback: try data/raw
        default_dir = REPO / "data" / "raw"
        if default_dir.exists():
            datasets_tasks = find_all_datasets(default_dir)
        if not datasets_tasks:
            parser.error("Specify --dataset, --file, or --task.")

    report_lines = []

    for ds_name, tasks in datasets_tasks.items():
        total = len(tasks)
        if total == 0:
            print(f"Dataset '{ds_name}': 0 task instructions found.")
            continue

        cands_map = {}
        warnings_map = {}
        grouped = defaultdict(list)

        for t in sorted(tasks):
            cands = parse_task_objects(t)
            warns = check_candidate_quality(t, cands)
            cands_map[t] = cands

            if warns:
                warnings_map[t] = warns
                grouped["flagged"].append(t)
            elif len(cands) == 0:
                grouped["empty"].append(t)
            elif len(cands) == 1:
                grouped["1_target"].append(t)
            elif len(cands) == 2:
                grouped["2_targets"].append(t)
            else:
                grouped["3plus_targets"].append(t)

        clean_count = len(grouped["2_targets"]) + len(grouped["1_target"]) + len(grouped["3plus_targets"])
        pct_clean = 100.0 * clean_count / total
        pct_2 = 100.0 * len(grouped["2_targets"]) / total
        pct_1 = 100.0 * len(grouped["1_target"]) / total
        pct_3 = 100.0 * len(grouped["3plus_targets"]) / total
        pct_warn = 100.0 * len(grouped["flagged"]) / total
        pct_empty = 100.0 * len(grouped["empty"]) / total

        header = f"\n{'='*70}\n{BOLD}PROMPT SPLITTING REPORT: {CYAN}{ds_name}{RESET} ({total} unique prompts)\n{'='*70}"
        print(header)
        report_lines.append(header)

        summary = (
            f"  * {BOLD}Target + Destination (2 objects):{RESET} {GREEN}{len(grouped['2_targets']):5d}{RESET} ({pct_2:.1f}%)\n"
            f"  * {BOLD}Single Target        (1 object): {RESET} {GREEN}{len(grouped['1_target']):5d}{RESET} ({pct_1:.1f}%)\n"
            f"  * {BOLD}Multi-Target         (3+ objects):{RESET}{CYAN}{len(grouped['3plus_targets']):5d}{RESET} ({pct_3:.1f}%)\n"
            f"  * {BOLD}Flagged / Leaky words:           {RESET} {YELLOW if grouped['flagged'] else GREEN}{len(grouped['flagged']):5d}{RESET} ({pct_warn:.1f}%)\n"
            f"  * {BOLD}Empty / Unparsed     (0 objects):{RESET} {RED if grouped['empty'] else GREEN}{len(grouped['empty']):5d}{RESET} ({pct_empty:.1f}%)\n"
            f"  --> {BOLD}Overall Clean Rate:{RESET} {GREEN if pct_clean >= 95 else YELLOW}{pct_clean:.1f}%{RESET}"
        )
        print(summary)
        report_lines.append(summary)

        # Print warnings / flagged if any
        if grouped["flagged"] or grouped["empty"]:
            warn_hdr = f"\n{RED}{BOLD}=== ISSUES / SUSPICIOUS PROMPTS ({len(grouped['flagged']) + len(grouped['empty'])} total) ==={RESET}"
            print(warn_hdr)
            report_lines.append(warn_hdr)

            for t in grouped["empty"]:
                line = f"  [EMPTY] \"{t}\" -> {cands_map[t]}"
                print(f"{RED}{line}{RESET}")
                report_lines.append(line)

            for t in grouped["flagged"]:
                w_str = "; ".join(warnings_map[t])
                line = f"  [FLAGGED] \"{t}\"\n            Candidates: {cands_map[t]}\n            Reason: {w_str}"
                print(f"{YELLOW}{line}{RESET}")
                report_lines.append(line)
        else:
            no_issues = f"\n{GREEN}{BOLD}✓ Zero flagged prompts! All {total} prompts parsed cleanly.{RESET}"
            print(no_issues)
            report_lines.append(no_issues)

        # Samples
        if not args.warnings_only:
            sample_hdr = f"\n{BOLD}=== SAMPLE CLEAN SPLITS (Target + Destination) ==={RESET}"
            print(sample_hdr)
            report_lines.append(sample_hdr)

            sample_2 = random.sample(grouped["2_targets"], min(args.sample, len(grouped["2_targets"])))
            for t in sample_2:
                line = f"  * \"{t}\"\n    -> {GREEN}{cands_map[t]}{RESET}"
                print(line)
                report_lines.append(f"  * \"{t}\"\n    -> {cands_map[t]}")

            if grouped["1_target"]:
                sample_1_hdr = f"\n{BOLD}=== SAMPLE CLEAN SPLITS (Single Target) ==={RESET}"
                print(sample_1_hdr)
                report_lines.append(sample_1_hdr)
                sample_1 = random.sample(grouped["1_target"], min(5, len(grouped["1_target"])))
                for t in sample_1:
                    line = f"  * \"{t}\"\n    -> {CYAN}{cands_map[t]}{RESET}"
                    print(line)
                    report_lines.append(f"  * \"{t}\"\n    -> {cands_map[t]}")

        # Show all if requested
        if args.show_all:
            show_hdr = f"\n{BOLD}=== ALL PARSED PROMPTS ({total}) ==={RESET}"
            print(show_hdr)
            report_lines.append(show_hdr)
            for t in sorted(tasks):
                line = f"  \"{t}\" -> {cands_map[t]}"
                print(line)
                report_lines.append(line)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # Strip ANSI codes for clean file writing
        clean_text = "\n".join(report_lines)
        import re as re_mod
        clean_text = re_mod.sub(r"\033\[[0-9;]*m", "", clean_text)
        args.out.write_text(clean_text, encoding="utf-8")
        print(f"\n{BOLD}Full report saved to:{RESET} {args.out}")


if __name__ == "__main__":
    main()
