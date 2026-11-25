#!/usr/bin/env python3
"""SAM 3 detection probe on LeRobot frames: boxes and names of objects from a closed per-dataset dictionary.

SAM 3 answers one noun phrase per query: a presence probability ("is there any X?") and boxes scored by
box confidence x presence. It names nothing by itself, so every box is named by the dictionary entry it was found with.
Each frame is encoded once and every name is queried against it. The same raw answers are then filtered per vocabulary:
  instr  the episode's instruction nouns, mapped onto the dataset dictionary ("small knife" -> knife,
         "spaghetti laddle" -> ladle); names that map to nothing are listed in summary.txt
  dict   the whole closed dictionary of the dataset (submission/src/aijgen/scene/dicts/<dataset>.txt)
  neg    negative control: 3 entries of another dataset's dictionary that share no word with this dataset's
         (chip bags on a Bridge frame, pots on a Fractal one). Almost surely absent, so their boxes are false positives.

Box filtering, per vocabulary and frame: aijgen.scene.detect.select (the same rules the scene_qa generator uses),
with --threshold, --max-area, --dup-iou, --contain, --conflict-iou and --margin.

Writes to --out:
  sheet_<vocab>_<page>.jpg  kept boxes (name + score, "x2+" if the name is kept more than once on the frame);
                            sheet_instr_* also draws their masks; --show-dropped adds ambiguous boxes in orange
  records.jsonl             every frame with its names and raw answers (boxes >= 0.05), to re-tune thresholds offline
  summary.txt               per-vocabulary counts, present-vs-absent score separation, dictionary coverage, timing

Run in the organizers' image on the GPU box (transformers 5.3.0):
  docker compose exec dev python tools/sam3_probe.py --input /data/raw --out runs/sam3_probe
facebook/sam3 is gated: request access on Hugging Face, then make the token visible inside the container
(`hf auth login` there, or HF_TOKEN). --smoke runs random weights with a public CLIP tokenizer on one frame per dataset
to check the code path without access; its boxes are meaningless.
"""
import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "submission" / "src"))
from aijgen.depth.lerobot import list_episodes, read_frames  # noqa: E402
from aijgen.scene.detect import Sam3, select  # noqa: E402
from aijgen.scene.vocab import (DEFAULT_EXCLUDE, DICT_DIR, Entry, load_dictionary, map_to_dictionary,  # noqa: E402
                                name_ok)

VOCABS = ("instr", "dict", "neg")
SWEEP = (0.3, 0.4, 0.5, 0.6)    # thresholds for the present-vs-absent table
MIN_FRAME_STD = 8.0             # flatter frames are blank cameras
TILE_H, COLS, ROWS = 360, 3, 4

# Instruction parsing: "take the eggplant and put it between the two right burners" -> ["eggplant", "burner"]
VERBS = {
    "pick", "picked", "lift", "lifted", "put", "place", "placed", "set", "move", "moved", "take", "took", "taken",
    "grab", "grabbed", "get", "got", "bring", "brought", "open", "opened", "close", "closed", "fold", "folded",
    "unfold", "unfolded", "push", "pushed", "pull", "pulled", "slide", "slid", "flip", "flipped", "turn", "turned",
    "knock", "knocked", "sweep", "swept", "wipe", "wiped", "stack", "stacked", "remove", "removed", "insert", "inserted",
    "drop", "dropped", "rotate", "rotated", "separate", "separated", "nudge", "nudged", "add", "added", "transfer",
    "transferred", "reposition", "repositioned", "pour", "poured", "hang", "hung", "cover", "covered", "uncover",
    "uncovered"}
PARTICLES = {"up", "down", "over", "out", "off", "back", "away"}
PIECE_OF = re.compile(r"\b(?:piece|pieces|bunch|pair|sheet)\s+of\b")
SPLIT = re.compile(r"[,;.]|\b(?:and|then|into|onto|in|on|to|from|of|at|by|near|beside|between|under|underneath|"
                   r"inside|above|below|along|against|with|toward|towards|across|around|behind|next|so|it|them)\b")
DROP = {"the", "a", "an", "two", "three", "some", "its", "their", "this", "that", "other", "another", "left", "right",
        "top", "bottom", "middle", "center", "centre", "front", "back", "upper", "lower", "side", "edge", "corner",
        "upright", "one", "ones"} | VERBS | PARTICLES
MAX_TASK_WORDS = 20  # longer "instructions" are mostly captions or junk ("Look at the sequence of images...")
IRREGULAR = {"knives": "knife", "shelves": "shelf", "leaves": "leaf", "loaves": "loaf"}


def is_verb_form(word: str) -> bool:
    """folds, pushes, putting, moving: inflections of VERBS that the word list does not spell out."""
    stems = {word.removesuffix("s"), word.removesuffix("es")}
    if word.endswith("ing") and len(word) > 5:
        stem = word[:-3]
        stems |= {stem, stem + "e", stem[:-1] if len(stem) > 2 and stem[-1] == stem[-2] else stem}
    return bool(stems & VERBS)


def singular(word: str) -> str:
    if word in IRREGULAR:
        return IRREGULAR[word]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and re.search(r"(ches|shes|xes|sses)$", word):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def instruction_names(task: str, exclude=DEFAULT_EXCLUDE, max_words: int = 4) -> list[str]:
    """Object noun phrases of an instruction, in order, without verbs, articles, positions, surfaces or `exclude`d
    heads."""
    names = []
    for chunk in SPLIT.split(PIECE_OF.sub(" ", task.lower())):
        words = [w for w in re.sub(r"[^a-z0-9 ]", " ", chunk).split() if w not in DROP and not is_verb_form(w)]
        if not words:
            continue
        words[-1] = singular(words[-1])
        name = " ".join(words[-max_words:])
        if name_ok(name, exclude) and name not in names:
            names.append(name)
    return names


# Closed dictionary (aijgen.scene.vocab)
def mapped_names(task: str, entries: dict[str, Entry], exclude) -> tuple[list[str], list[str]]:
    """(dictionary names of the instruction's objects, instruction names that map to nothing)."""
    mapped, unmapped = [], []
    for nm in instruction_names(task, exclude):
        e = map_to_dictionary(nm, entries)
        if e is None:
            unmapped.append(nm)
        elif e not in mapped:
            mapped.append(e)
    return mapped, unmapped


# Data
def first_camera(root: Path) -> str:
    feats = json.loads((root / "meta" / "info.json").read_text())["features"]
    return sorted(k for k, v in feats.items() if v.get("dtype") == "video")[0]


def usable_episodes(root: Path, cam: str, entries, exclude):
    return [e for e in list_episodes(root, cam)
            if e.length >= 10 and len(e.task.split()) <= MAX_TASK_WORDS and mapped_names(e.task, entries, exclude)[0]]


def pick_frames(eps, n_episodes: int, n_frames: int, span: tuple[float, float], rng: np.random.Generator):
    """Distinct-instruction episodes; n_frames evenly spread over `span` of each."""
    by_task = {}
    for e in eps:
        by_task.setdefault(e.task, e)
    uniq = list(by_task.values())
    chosen = [uniq[i] for i in sorted(rng.choice(len(uniq), size=min(n_episodes, len(uniq)), replace=False))]
    out = []
    for ep in chosen:
        idx = sorted({int(round(x)) for x in np.linspace(span[0] * (ep.length - 1), span[1] * (ep.length - 1), n_frames)})
        frames = read_frames(ep, idx)
        for fi in idx:
            if fi in frames and frames[fi].std() >= MIN_FRAME_STD:
                out.append((ep, fi, frames[fi]))
    return out


def coverage(root: Path, cam: str, entries, exclude) -> tuple[float, Counter]:
    """Share of episodes whose instruction objects all map onto the dictionary, and the unmapped names by count."""
    eps = [e for e in list_episodes(root, cam) if len(e.task.split()) <= MAX_TASK_WORDS]
    unmapped, full = Counter(), 0
    for e in eps:
        m, u = mapped_names(e.task, entries, exclude)
        unmapped.update(set(u))
        full += bool(m) and not u
    return full / max(len(eps), 1), unmapped


# Drawing
FONT = cv2.FONT_HERSHEY_SIMPLEX
PALETTE = [(0, 200, 0), (255, 140, 0), (0, 170, 255), (220, 0, 220), (0, 220, 220), (255, 60, 60), (150, 90, 255),
           (180, 220, 0)]  # RGB


def name_color(name: str):
    return PALETTE[sum(map(ord, name)) % len(PALETTE)]


def banner(img: np.ndarray, lines: list[str]) -> np.ndarray:
    h, w = img.shape[:2]
    scale, lh = 0.45 * w / 640, int(18 * w / 640)
    pad = np.full((lh * len(lines) + 6, w, 3), 20, np.uint8)
    for i, ln in enumerate(lines):
        cv2.putText(pad, ln[:95], (6, lh * (i + 1)), FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([pad, img])


def draw_box(img, box, text, color, thick=2):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)
    scale = 0.5 * img.shape[1] / 640
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    ty = y1 - 4 if y1 - th - 6 > 0 else y1 + th + 4
    cv2.rectangle(img, (x1, ty - th - 3), (x1 + tw + 4, ty + 3), (0, 0, 0), -1)
    cv2.putText(img, text, (x1 + 2, ty), FONT, scale, color, 1, cv2.LINE_AA)


def detection_tile(rgb, rec, vocab, sel, show_dropped: bool):
    img = rgb.copy()
    kept_masks = [(c["mask"], name_color(c["name"])) for c in sel["kept"] if c["mask"] is not None]
    if vocab == "instr" and kept_masks:  # instr names are the only ones queried with masks; drawn on their sheet only
        tint = img.astype(np.float32)
        for m, color in kept_masks:
            tint[m] = 0.5 * tint[m] + 0.5 * np.array(color, np.float32)
        img = tint.astype(np.uint8)
        for m, color in kept_masks:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, color, 1, cv2.LINE_AA)
    if show_dropped:
        for c in sel["ambiguous"]:
            draw_box(img, c["box"], f"?{c['name']} {c['score']:.2f}", (255, 165, 0), 1)
    for c in sel["kept"]:
        draw_box(img, c["box"], f"{c['name']} {c['score']:.2f}" + ("" if c["unique"] else " x2+"), name_color(c["name"]))
    names = rec["names"][vocab]
    shown = ", ".join(names) if len(names) <= 8 else f"{len(names)} names"
    if vocab == "instr" and rec["unmapped"]:
        shown += f"  (not in dictionary: {', '.join(rec['unmapped'])})"
    return banner(img, [f"{rec['dataset']} ep{rec['episode']} f{rec['frame']}: {rec['task']}", f"[{vocab}] {shown}"])


def write_sheets(tiles: list[np.ndarray], stem: Path):
    per = COLS * ROWS
    for p in range(0, len(tiles), per):
        page = [cv2.resize(t, (round(t.shape[1] * TILE_H / t.shape[0]), TILE_H)) for t in tiles[p:p + per]]
        w = max(t.shape[1] for t in page)
        page = [np.pad(t, ((0, 0), (0, w - t.shape[1]), (0, 0))) for t in page]
        page += [np.zeros_like(page[0])] * (-len(page) % COLS)
        rows = [np.hstack(page[r:r + COLS]) for r in range(0, len(page), COLS)]
        cv2.imwrite(f"{stem}_{p // per + 1}.jpg", cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])


# Summary
def summarize(records, selections, cover, timing, args) -> str:
    n = len(records)
    out = [f"frames: {n} ({', '.join(f'{k}={v}' for k, v in Counter(r['dataset'] for r in records).items())}), "
           f"threshold {args.threshold}, max area {args.max_area}, model {args.model}"
           f"{' (SMOKE: random weights)' if args.smoke else ''}", ""]

    out.append("Per vocabulary (means per frame unless noted)")
    out.append(f"  {'vocab':6s} {'names':>6s} {'kept':>6s} {'>=2 obj':>8s} {'non-uniq':>9s} {'dupl':>6s} {'ambig':>6s} "
               f"{'lost':>5s} {'too big':>8s}")
    for v in VOCABS:
        sels = [s[v] for s in selections]
        frames_2 = sum(len({k["name"] for k in s["kept"] if k["unique"]}) >= 2 for s in sels)
        out.append(f"  {v:6s} {np.mean([len(r['names'][v]) for r in records]):6.1f} "
                   f"{np.mean([len(s['kept']) for s in sels]):6.2f} {frames_2:4d}/{n:<3d} "
                   f"{np.mean([sum(not k['unique'] for k in s['kept']) for s in sels]):9.2f} "
                   f"{np.mean([s['duplicate'] for s in sels]):6.2f} {np.mean([len(s['ambiguous']) for s in sels]):6.2f} "
                   f"{np.mean([s['lost'] for s in sels]):5.2f} {np.mean([s['too_big'] for s in sels]):8.2f}")
    out.append("  >=2 obj: frames with at least 2 distinct names each kept exactly once (usable for pair questions)")
    out.append("  dupl: same/similar name on the same region; ambig: near tie with an unrelated name (both dropped);")
    out.append("  lost: clearly beaten by an unrelated name; too big: box over --max-area of the frame")
    out.append("")

    out.append("Present (instruction names) vs probably absent (negative control): share of (frame, name) with a box >= t")
    out.append(f"  {'t':>5s} {'instr hit':>10s} {'neg hit':>8s}")
    groups = {g: [(r, nm) for r in records for nm in r["names"][g]] for g in ("instr", "neg")}
    for t in SWEEP:
        rates = {g: np.mean([max(r["answers"][nm]["scores"], default=0) >= t for r, nm in pairs]) if pairs else float("nan")
                 for g, pairs in groups.items()}
        out.append(f"  {t:5.2f} {rates['instr']:10.2f} {rates['neg']:8.2f}")
    for g, pairs in groups.items():
        pres = [r["answers"][nm]["presence"] for r, nm in pairs]
        if pres:
            out.append(f"  presence prob {g:5s}: median {np.median(pres):.2f}, p10 {np.percentile(pres, 10):.2f}, "
                       f"p90 {np.percentile(pres, 90):.2f} (n={len(pres)})")
    missed = Counter(nm for r, nm in groups["instr"] if max(r["answers"][nm]["scores"], default=0) < args.threshold)
    if missed:
        out.append(f"  instruction names not found at {args.threshold}: "
                   + ", ".join(f"{k} x{c}" for k, c in missed.most_common()))
    out.append("")

    out.append("Dictionary coverage over all episodes (instruction objects that all map onto the dictionary)")
    for ds, (share, unmapped) in cover.items():
        out.append(f"  {ds}: {100 * share:.0f}% of episodes; most frequent unmapped names: "
                   + ", ".join(f"{k} x{c}" for k, c in unmapped.most_common(15)))
    out.append("")

    out.append(f"Timing on {args.device} ({str(args.dtype).replace('torch.', '')}): image encode "
               f"{1000 * np.mean(timing['encode']):.0f} ms/frame, query {1000 * np.mean(timing['query']):.0f} ms/name, "
               f"{np.mean(timing['n_queries']):.0f} names/frame")
    if args.device == "cuda":
        out.append(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=REPO / "data" / "raw", help="root with one LeRobot dataset per folder")
    ap.add_argument("--datasets", nargs="+", default=["bridge", "fractal"])
    ap.add_argument("--dict-dir", type=Path, default=DICT_DIR, help="closed dictionaries, one <dataset>.txt each")
    ap.add_argument("--episodes", type=int, default=6, help="episodes per dataset, distinct instructions")
    ap.add_argument("--frames", type=int, default=2, help="frames per episode")
    ap.add_argument("--span", type=float, nargs=2, default=[0.05, 0.3],
                    help="part of each episode to take frames from (early: objects not yet moved)")
    ap.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE),
                    help="drop names whose last word is one of these, from instructions and dictionaries")
    ap.add_argument("--threshold", type=float, default=0.4, help="min box score (confidence x presence)")
    ap.add_argument("--max-area", type=float, default=0.4, help="drop boxes covering more of the frame (fridge, sink)")
    ap.add_argument("--dup-iou", type=float, default=0.7, help="similar-name boxes overlapping more are one object")
    ap.add_argument("--contain", type=float, default=0.85,
                    help="similar-name boxes this much inside one another are one object (nested sink basins)")
    ap.add_argument("--conflict-iou", type=float, default=0.7, help="unrelated-name boxes overlapping more compete")
    ap.add_argument("--margin", type=float, default=0.15, help="score gap a name needs to win a competing box")
    ap.add_argument("--show-dropped", action="store_true", help="also draw ambiguous boxes (orange, '?')")
    ap.add_argument("--model", default="facebook/sam3")
    ap.add_argument("--fp32", action="store_true", help="run in fp32 (default on GPU: bf16, as it would ship)")
    ap.add_argument("--smoke", action="store_true", help="random weights, 1 frame per dataset, few names: code path only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "sam3_probe")
    args = ap.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.dtype = torch.float32 if args.fp32 or args.device == "cpu" else torch.bfloat16
    if args.smoke:
        args.episodes, args.frames = 1, 1
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    exclude = frozenset(args.exclude)

    found, cover = {}, {}  # dataset -> (dictionary, frames); dataset -> coverage
    for ds in args.datasets:
        root, dict_path = args.input / ds, args.dict_dir / f"{ds}.txt"
        if not (root / "meta" / "info.json").is_file() or not dict_path.is_file():
            print(f"skip {ds}: no {root}/meta/info.json or no {dict_path}")
            continue
        entries = load_dictionary(dict_path, exclude)
        cam = first_camera(root)
        eps = usable_episodes(root, cam, entries, exclude)
        found[ds] = (cam, entries, pick_frames(eps, args.episodes, args.frames, tuple(args.span), rng))
        cover[ds] = coverage(root, cam, entries, exclude)
        print(f"{ds}: {len(found[ds][2])} frames ({cam}), dictionary of {len(entries)} names, "
              f"{100 * cover[ds][0]:.0f}% of episodes fully covered")

    jobs = []  # (dataset, camera, ep, frame index, rgb, names per vocab, unmapped instruction names, entries)
    for ds, (cam, entries, frames) in found.items():
        own_words = {w for n in entries for w in n.split()}
        others = {n: e for o, f in found.items() if o != ds for n, e in f[1].items() if not own_words & set(n.split())}
        dict_names = list(entries)[:3] if args.smoke else list(entries)
        for ep, fi, rgb in frames:
            instr, unmapped = mapped_names(ep.task, entries, exclude)
            neg = [list(others)[i] for i in sorted(rng.choice(len(others), size=min(3, len(others)), replace=False))]
            names = {"instr": instr, "dict": list(dict.fromkeys(instr + dict_names)), "neg": neg}
            jobs.append((ds, cam, ep, fi, rgb, names, unmapped, entries | {n: others[n] for n in neg}))
    if not jobs:
        sys.exit("no frames found")

    print(f"\nloading {args.model} on {args.device} ({args.dtype})...")
    t0 = time.perf_counter()
    sam = Sam3(args.model, args.device, args.dtype, args.smoke)
    print(f"loaded in {time.perf_counter() - t0:.1f} s")

    records, selections, rgbs = [], [], []
    timing = defaultdict(list)
    with open(args.out / "records.jsonl", "w") as jf:
        for j, (ds, cam, ep, fi, rgb, names, unmapped, entries) in enumerate(jobs):
            h, w = rgb.shape[:2]
            t = time.perf_counter()
            vision = sam.encode(rgb)
            timing["encode"].append(time.perf_counter() - t)

            queries = list(dict.fromkeys(n for v in VOCABS for n in names[v]))
            answers = {}
            for nm in queries:
                t = time.perf_counter()
                answers[nm] = sam.query(vision, entries[nm].prompt, (h, w), masks=nm in names["instr"],
                                        mask_min=args.threshold)
                timing["query"].append(time.perf_counter() - t)
            timing["n_queries"].append(len(queries))

            rec = {"dataset": ds, "camera": cam, "episode": ep.index, "frame": fi, "task": ep.task, "names": names,
                   "unmapped": unmapped, "prompts": {n: entries[n].prompt for n in queries if entries[n].prompt != n},
                   "answers": {n: {k: v for k, v in a.items() if k != "masks"} for n, a in answers.items()}}
            jf.write(json.dumps(rec) + "\n")
            sel = {v: select(answers, names[v], entries, h * w, args) for v in VOCABS}
            records.append(rec)
            selections.append(sel)
            rgbs.append(rgb)
            kept = "; ".join(f"{v}: {', '.join(k['name'] for k in sel[v]['kept']) or '-'}" for v in VOCABS)
            print(f"[{j + 1}/{len(jobs)}] {ds} ep{ep.index} f{fi} \"{ep.task}\" -> {kept}")

    for v in VOCABS:
        write_sheets([detection_tile(rgb, r, v, s[v], args.show_dropped) for rgb, r, s in zip(rgbs, records, selections)],
                     args.out / f"sheet_{v}")

    summary = summarize(records, selections, cover, timing, args)
    (args.out / "summary.txt").write_text(summary + "\n")
    print("\n" + summary + f"\n\nwrote {args.out}/")


if __name__ == "__main__":
    main()
