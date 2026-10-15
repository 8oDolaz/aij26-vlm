"""Step through a generated dataset: the image in a window, the question and the correct answer in the terminal.

Depth questions also show DA3's depth map (turbo: red = close, blue = far) with the same A/B points.
Keys in the window: n / space / right = next, p / left = previous, r = random, q / Esc = quit.

.venv/bin/python tools/inspect_annotations.py mount/rel_depth_phase
.venv/bin/python tools/inspect_annotations.py mount/rel_depth_phase --kind depth --shuffle
python3 tools/inspect_annotations.py mount/rel_depth_phase --print --kind phase | less -R   # terminal only, no window
"""
import argparse
import json
import random
import re
from pathlib import Path

BOLD, GREEN, DIM, RESET = "\033[1m", "\033[92m", "\033[2m", "\033[0m"
OPTION = re.compile(r"^\s*\(?([A-D])[.)]\s*(.+)$")
PHASE_IMAGE = re.compile(r"ep(\d+)_f(\d+)_(\w+)\.jpg$")
EPISODE_IN_NAME = re.compile(r"_ep(\d+)_")
DEPTH_IMAGE = re.compile(r"_(\d{6})\.jpg$")


def load(root: Path) -> list[dict]:
    """Records of annotations.jsonl, each with its kind (depth / phase) and, for depth, its line of
    depth_tier1/annotations.meta.jsonl."""
    meta_path = root / "depth_tier1" / "annotations.meta.jsonl"
    depth_meta = {}
    if meta_path.exists():
        depth_meta = {m["id"]: m for m in map(json.loads, meta_path.read_text().splitlines())}
    phase_meta_path = root / "annotations.meta.jsonl"
    phase_meta = {}
    if phase_meta_path.exists():
        phase_meta = {m["id"]: m for m in map(json.loads, phase_meta_path.read_text().splitlines())}
    records = []
    for n, line in enumerate((root / "annotations.jsonl").read_text().splitlines()):
        r = json.loads(line)
        image = r["images"][0]
        r["n"], r["kind"] = n, "depth" if image.startswith("depth_tier1/") else "phase"
        if r["kind"] == "depth":
            r["meta"] = depth_meta.get(int(DEPTH_IMAGE.search(image).group(1)))
        if r["kind"] == "phase":
            r["pmeta"] = phase_meta.get(n)            
        records.append(r)
    return records

def episode_of(r: dict) -> int | None:
    """Episode number of a phase record: from annotations.meta.jsonl, else from the image name."""
    if r.get("pmeta"):
        return r["pmeta"]["episode"]
    m = EPISODE_IN_NAME.search(r["images"][0])
    return int(m.group(1)) if m else None


def describe(r: dict, total: int) -> str:
    question, answer = r["messages"][0]["content"], r["messages"][1]["content"].strip()
    options = dict(m.groups() for m in map(OPTION.match, question.splitlines()) if m)
    lines = [f"{BOLD}[{r['n'] + 1}/{total}] {r['kind']}{RESET}  {DIM}{' '.join(r['images'])}{RESET}"]
    m = r.get("meta")
    if m:
        lines.append(f"{DIM}{m['dataset']} ep{m['episode']} f{m['frame_index']} {m['camera'].split('.')[-1]}  "
                     f"task: {m['task']!r}{RESET}")
        lines.append(f"{DIM}closer: {m['near']}  gap g={m['g']:.2f}  group={m['group']}  "
                     f"A={m['A']['kind']} d={m['A']['d']:.2f}  B={m['B']['kind']} d={m['B']['d']:.2f}  "
                     f"background={m['background']}  prompt={m['kind']}{RESET}")
    elif (pm := r.get("pmeta")):
        lines.append(f"{DIM}{pm['dataset']} ep{pm['episode']} f{pm['frame']}/{pm.get('length', '?')} {pm['camera']}  task: {pm['task']!r}{RESET}")
        lines.append(f"{DIM}gripper state={pm['gripper_state']}  action={pm['gripper_action']}  "
                     f"closedness={pm['closedness']}  around={pm['closedness_around']}{RESET}")
        lines.append(f"{DIM}phase (5) = {pm['phase5']}   question phase = {pm['question_phase']}{RESET}")
    elif (p := PHASE_IMAGE.search(r["images"][0])):
        lines.append(f"{DIM}ep{int(p.group(1))} f{int(p.group(2))} {p.group(3)}{RESET}")
    lines += ["", question.replace("<image>\n", ""), ""]
    lines.append(f"{GREEN}{BOLD}Answer: {answer}{RESET}" +
                 (f"{GREEN}  = {options[answer]}{RESET}" if answer in options else ""))
    return "\n".join(lines)


def view(root: Path, r: dict):
    """The question image, plus the depth map with the same points for depth questions; scaled up 2x."""
    import cv2
    import numpy as np
    img = cv2.imread(str(root / r["images"][0]))
    m = r.get("meta")
    cache = root / "depth_tier1" / "cache" / f"{m['frame']}.npz" if m else None
    if cache and cache.exists():
        disp = np.load(cache)["disp"].astype(np.float32)
        lo, hi = np.percentile(disp, [2, 98])
        depth = cv2.applyColorMap((np.clip((disp - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)
        depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
        for name, (x, y) in m["drawn"].items():
            cv2.circle(depth, (x, y), 9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(depth, name, (x + 11, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        img = np.hstack([img, depth])
    return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="dir with annotations.jsonl (generate_dataset.py --output)")
    ap.add_argument("--kind", choices=["depth", "phase"], help="only this kind of question")
    ap.add_argument("--shuffle", action="store_true", help="random order instead of file order")
    ap.add_argument("--start", type=int, default=1, help="record number to start at (as printed)")
    ap.add_argument("--print", action="store_true", help="print all selected records, no window")
    ap.add_argument("--episode", type=int, nargs="+", help="only these episodes, e.g. --episode 47 13860")
    args = ap.parse_args()

    records = load(args.root)
    total = len(records)
    records = [r for r in records if args.kind in (None, r["kind"])]

    if args.episode:
        wanted = set(args.episode)
        records = [r for r in records if episode_of(r) in wanted]
        if not records:
            print(f"no records for episodes {sorted(wanted)}")
            return

    if args.shuffle:
        random.shuffle(records)
    if args.print:
        try:
            for r in records:
                print(describe(r, total) + "\n" + "-" * 80)
        except BrokenPipeError:  # piped into head / less that quit early
            pass
        return

    import cv2
    i = next((k for k, r in enumerate(records) if r["n"] + 1 >= args.start), 0)
    while True:
        r = records[i]
        print("\033[2J\033[H" + describe(r, total))
        print(f"\n{DIM}n/space: next  p: previous  r: random  q: quit{RESET}")
        cv2.imshow("annotation", view(args.root, r))
        key = cv2.waitKeyEx(0)
        if key in (ord("q"), 27) or cv2.getWindowProperty("annotation", cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in (ord("p"), 65361, 0x1000012):  # p, left arrow (GTK, Qt)
            i = (i - 1) % len(records)
        elif key == ord("r"):
            i = random.randrange(len(records))
        else:
            i = (i + 1) % len(records)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
