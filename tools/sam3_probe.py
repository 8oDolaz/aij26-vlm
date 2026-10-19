#!/usr/bin/env python3
"""SAM 3 detection probe on LeRobot frames: compares prompt vocabularies and segments the robot arm and gripper.

SAM 3 answers one noun phrase per query: a presence probability ("is there any X?") and boxes scored by
box confidence x presence. It names nothing by itself, so every box is named by the phrase it was found with.
Each frame is encoded once and every name is queried against it. The same raw answers are then filtered per vocabulary:
  instr      nouns parsed from the episode's instruction
  instr+top  instr plus the --top-n most frequent instruction nouns of that dataset
  fixed      a fixed word list (--vocab-file; the old 64-word list by default)
  neg        negative control: 3 common names of the other datasets (Fractal's "redbull can" for a Bridge frame) that
             share no word with this episode's or this dataset's frequent names. These objects are almost surely
             absent, so boxes for them are false positives.
Per vocabulary: boxes >= --threshold, same-name duplicates merged, a box that clearly loses to a different name on the
same region dropped, near-ties between names marked ambiguous, boxes lying on the arm/gripper masks rejected.

Arm test: every --arm-prompts / --gripper-prompts phrase is segmented on every frame (masks + boxes).

Writes to --out:
  sheet_<vocab>_<page>.jpg  kept boxes (coloured, name + score), ambiguous (orange), on the arm (grey)
  sheet_arm_<page>.jpg      arm masks (blue) and gripper masks (red) of the first prompt of each list
  records.jsonl             every frame with its names and raw answers (boxes >= 0.05), to re-tune thresholds offline
  summary.txt               per-vocabulary counts, present-vs-absent score separation, arm prompt stats, timing

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
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "submission" / "src"))
from aijgen.depth.lerobot import list_episodes, read_frames  # noqa: E402

DEFAULT_VOCAB = REPO / "submission" / "src" / "aijgen" / "objects" / "bridge_objects.txt"
VOCABS = ("instr", "instr+top", "fixed", "neg")
RAW_MIN_SCORE = 0.05            # answers kept in records.jsonl
SWEEP = (0.3, 0.4, 0.5, 0.6)    # thresholds for the present-vs-absent table
MIN_FRAME_STD = 8.0             # flatter frames are blank cameras
TILE_H, COLS, ROWS = 360, 3, 4

# ---------------------------------------------------------------------------------------------------------------------
# Instruction parsing: "take the eggplant and put it between the two right burners" -> ["eggplant", "burner"]
# ---------------------------------------------------------------------------------------------------------------------
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
SURFACES = {"table", "tabletop", "table top", "counter", "surface", "floor", "wall", "place", "position",
            "robot", "gripper", "arm", "robot arm", "robot gripper"}  # not objects; the arm has its own test
FIXTURES = {"drawer", "shelf", "fridge", "door", "cabinet", "sink", "stove", "burner", "oven", "microwave", "counter",
            "wall", "window", "table"}  # never used as negative-control names: often in view
MAX_TASK_WORDS = 20  # longer "instructions" are mostly captions or junk ("Look at the sequence of images...")


def is_verb_form(word: str) -> bool:
    """folds, pushes, putting, moving: inflections of VERBS that the word list does not spell out."""
    stems = {word.removesuffix("s"), word.removesuffix("es")}
    if word.endswith("ing") and len(word) > 5:
        stem = word[:-3]
        stems |= {stem, stem + "e", stem[:-1] if len(stem) > 2 and stem[-1] == stem[-2] else stem}
    return bool(stems & VERBS)


def singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and re.search(r"(ches|shes|xes|sses)$", word):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def instruction_names(task: str, max_words: int = 4) -> list[str]:
    """Object noun phrases of an instruction, in order, without verbs, articles, positions or surfaces."""
    names = []
    for chunk in SPLIT.split(PIECE_OF.sub(" ", task.lower())):
        words = [w for w in re.sub(r"[^a-z0-9 ]", " ", chunk).split() if w not in DROP and not is_verb_form(w)]
        if not words:
            continue
        words[-1] = singular(words[-1])
        name = " ".join(words[-max_words:])
        if len(name) >= 3 and name not in SURFACES and name not in names:
            names.append(name)
    return names


def load_word_list(path: Path) -> list[str]:
    if not path.is_file():
        return []
    words = [ln.strip().lower() for ln in path.read_text().splitlines()]
    return list(dict.fromkeys(w for w in words if w and not w.startswith("#")))


# ---------------------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------------------
def first_camera(root: Path) -> str:
    feats = json.loads((root / "meta" / "info.json").read_text())["features"]
    return sorted(k for k, v in feats.items() if v.get("dtype") == "video")[0]


def pick_frames(root: Path, n_episodes: int, n_frames: int, span: tuple[float, float], rng: np.random.Generator):
    """Distinct-instruction episodes with parseable names; n_frames evenly spread over `span` of each."""
    cam = first_camera(root)
    eps = [e for e in list_episodes(root, cam)
           if e.length >= 10 and len(e.task.split()) <= MAX_TASK_WORDS and instruction_names(e.task)]
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
    return cam, eps, out


def top_names(eps, n: int) -> list[str]:
    counts = Counter(name for e in eps for name in set(instruction_names(e.task)))
    return [name for name, _ in counts.most_common(n)]


def negative_names(ep, pool: list[str], top: list[str], rng: np.random.Generator, k: int = 3) -> list[str]:
    """k names from `pool` sharing no word with this episode's names or its dataset's frequent names (those are likely
    in any scene of the dataset, e.g. Bridge's burners), and no fixtures."""
    avoid = {w for name in instruction_names(ep.task) + top for w in name.split()} | FIXTURES
    cands = [n for n in pool if not avoid & set(n.split())]
    return [cands[i] for i in sorted(rng.choice(len(cands), size=min(k, len(cands)), replace=False))]


# ---------------------------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------------------------
class Sam3:
    def __init__(self, model_id: str, device: str, dtype: torch.dtype, smoke: bool):
        from transformers import Sam3Model, Sam3Processor
        if smoke:
            from transformers import CLIPTokenizerFast, Sam3Config, Sam3ImageProcessorFast
            tok = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = Sam3Processor(Sam3ImageProcessorFast(), tok)
            torch.manual_seed(0)
            self.model = Sam3Model(Sam3Config())
        else:
            try:
                self.processor = Sam3Processor.from_pretrained(model_id)
                self.model = Sam3Model.from_pretrained(model_id)
            except OSError as e:
                sys.exit(f"cannot load {model_id}: {e}\nfacebook/sam3 is gated: request access on Hugging Face and log in "
                         "(hf auth login, or HF_TOKEN) inside the container.")
        self.model.to(device=device, dtype=dtype).eval()
        self.device, self.dtype = device, dtype
        self.text = {}

    def _sync(self):
        if self.device == "cuda":
            torch.cuda.synchronize()

    @torch.inference_mode()
    def encode(self, rgb: np.ndarray):
        pv = self.processor(images=Image.fromarray(rgb), return_tensors="pt")["pixel_values"]
        out = self.model.get_vision_features(pixel_values=pv.to(self.device, self.dtype))
        self._sync()
        return out

    @torch.inference_mode()
    def query(self, vision, name: str, size: tuple[int, int], masks: bool = False) -> dict:
        """Presence probability, boxes (xyxy pixels) and scores >= RAW_MIN_SCORE; masks (bool HxW) if asked."""
        if name not in self.text:
            self.text[name] = {k: v.to(self.device) for k, v in self.processor(text=name, return_tensors="pt").items()}
        out = self.model(vision_embeds=vision, **self.text[name])
        presence = 1.0 if out.presence_logits is None else float(out.presence_logits.float().sigmoid().max())
        post = (self.processor.post_process_instance_segmentation if masks
                else self.processor.post_process_object_detection)
        r = post(out, threshold=RAW_MIN_SCORE, target_sizes=[size])[0]
        self._sync()
        ans = {"presence": round(presence, 4),
               "scores": [round(float(s), 4) for s in r["scores"].float().cpu()],
               "boxes": [[round(float(v), 1) for v in b] for b in r["boxes"].float().cpu()]}
        if masks:
            ans["masks"] = [m.bool().cpu().numpy() for m in r["masks"]]
        return ans


# ---------------------------------------------------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------------------------------------------------
def iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def box_on_mask(mask: np.ndarray, box) -> float:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    patch = mask[max(y1, 0):max(y2, y1 + 1), max(x1, 0):max(x2, x1 + 1)]
    return float(patch.mean()) if patch.size else 0.0


def select(answers: dict, names: list[str], arm_mask, args) -> dict:
    """Kept / ambiguous / on-arm boxes for one vocabulary, plus counts of what was dropped."""
    cands = sorted(({"name": n, "box": b, "score": s} for n in names for b, s in
                    zip(answers[n]["boxes"], answers[n]["scores"]) if s >= args.threshold), key=lambda c: -c["score"])
    dedup = []
    for c in cands:
        if not any(d["name"] == c["name"] and iou(d["box"], c["box"]) > args.dup_iou for d in dedup):
            dedup.append(c)
    res = {"kept": [], "ambiguous": [], "on_arm": [], "lost": 0, "duplicates": len(cands) - len(dedup)}
    for c in dedup:
        rivals = [d for d in dedup if d["name"] != c["name"] and iou(d["box"], c["box"]) > args.conflict_iou]
        if any(d["score"] > c["score"] + args.margin for d in rivals):
            res["lost"] += 1
        elif rivals and any(abs(d["score"] - c["score"]) <= args.margin for d in rivals):
            res["ambiguous"].append(c)
        elif arm_mask is not None and box_on_mask(arm_mask, c["box"]) >= args.arm_overlap:
            res["on_arm"].append(c)
        else:
            res["kept"].append(c)
    counts = Counter(k["name"] for k in res["kept"])
    for k in res["kept"]:
        k["unique"] = counts[k["name"]] == 1
    return res


# ---------------------------------------------------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------------------------------------------------
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


def detection_tile(rgb, rec, vocab, sel):
    img = rgb.copy()
    for c in sel["on_arm"]:
        draw_box(img, c["box"], f"{c['name']} {c['score']:.2f} ARM", (140, 140, 140), 1)
    for c in sel["ambiguous"]:
        draw_box(img, c["box"], f"?{c['name']} {c['score']:.2f}", (255, 165, 0), 1)
    for c in sel["kept"]:
        draw_box(img, c["box"], f"{c['name']} {c['score']:.2f}" + ("" if c["unique"] else " x2+"), name_color(c["name"]))
    names = rec["names"][vocab]
    shown = ", ".join(names) if len(names) <= 8 else f"{len(names)} names"
    return banner(img, [f"{rec['dataset']} ep{rec['episode']} f{rec['frame']}: {rec['task']}",
                        f"[{vocab}] {shown}"])


def arm_tile(rgb, arm, args):
    img = rgb.copy().astype(np.float32)
    for prompts, color in ((args.arm_prompts, (40, 90, 255)), (args.gripper_prompts, (255, 40, 40))):
        a = arm[prompts[0]]
        for m, s in zip(a["masks"], a["scores"]):
            if s >= args.threshold:
                img[m] = 0.5 * img[m] + 0.5 * np.array(color, np.float32)
    img = img.astype(np.uint8)
    for prompts, color in ((args.arm_prompts, (40, 90, 255)), (args.gripper_prompts, (255, 40, 40))):
        a = arm[prompts[0]]
        for b, s in zip(a["boxes"], a["scores"]):
            if s >= args.threshold:
                draw_box(img, b, f"{prompts[0]} {s:.2f}", color, 1)
    best = [f"{p}={max(arm[p]['scores'], default=0):.2f}" for p in args.arm_prompts + args.gripper_prompts]
    return banner(img, ["blue: " + args.arm_prompts[0] + " | red: " + args.gripper_prompts[0], "best: " + " ".join(best)])


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


# ---------------------------------------------------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------------------------------------------------
def summarize(records, selections, arm_stats, timing, args) -> str:
    n = len(records)
    out = [f"frames: {n} ({', '.join(f'{k}={v}' for k, v in Counter(r['dataset'] for r in records).items())}), "
           f"threshold {args.threshold}, model {args.model}{' (SMOKE: random weights)' if args.smoke else ''}", ""]

    out.append("Per vocabulary (means per frame unless noted)")
    out.append(f"  {'vocab':10s} {'names':>6s} {'kept':>6s} {'>=2 obj':>8s} {'non-uniq':>9s} {'ambig':>6s} "
               f"{'lost':>5s} {'on arm':>7s}")
    for v in VOCABS:
        sels = [s[v] for s in selections]
        kept = [len(s["kept"]) for s in sels]
        frames_2 = sum(len({k["name"] for k in s["kept"] if k["unique"]}) >= 2 for s in sels)
        out.append(f"  {v:10s} {np.mean([len(r['names'][v]) for r in records]):6.1f} {np.mean(kept):6.2f} "
                   f"{frames_2:4d}/{n:<3d} {np.mean([sum(not k['unique'] for k in s['kept']) for s in sels]):9.2f} "
                   f"{np.mean([len(s['ambiguous']) for s in sels]):6.2f} {np.mean([s['lost'] for s in sels]):5.2f} "
                   f"{np.mean([len(s['on_arm']) for s in sels]):7.2f}")
    out.append("  >=2 obj: frames with at least 2 distinct names each found exactly once (usable for pair questions)")
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
    out.append("  'neg' names can still be in the scene (Bridge kitchens are cluttered): check neg boxes on sheet_neg_*.jpg")
    out.append("")

    out.append(f"Arm / gripper prompts (score >= {args.threshold})")
    out.append(f"  {'prompt':26s} {'found':>8s} {'top score':>10s} {'instances':>10s} {'area %':>7s}")
    for p, st in arm_stats.items():
        found = sum(st["found"])
        out.append(f"  {p:26s} {found:4d}/{n:<3d} {np.median(st['top']):10.2f} {np.mean(st['count']):10.2f} "
                   f"{np.median(st['area']) if st['area'] else 0:7.2f}")
    out.append("  area %: median mask area of found frames, union of instances, as % of the frame")
    out.append("")

    out.append(f"Timing on {args.device} ({str(args.dtype).replace('torch.', '')}): image encode "
               f"{1000 * np.mean(timing['encode']):.0f} ms/frame, query {1000 * np.mean(timing['query']):.0f} ms/name, "
               f"{np.mean(timing['n_queries']):.0f} names/frame")
    if args.device == "cuda":
        out.append(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=REPO / "data" / "raw", help="root with one LeRobot dataset per folder")
    ap.add_argument("--datasets", nargs="+", default=["bridge", "fractal"])
    ap.add_argument("--episodes", type=int, default=6, help="episodes per dataset, distinct instructions")
    ap.add_argument("--frames", type=int, default=2, help="frames per episode")
    ap.add_argument("--span", type=float, nargs=2, default=[0.05, 0.3],
                    help="part of each episode to take frames from (early: objects not yet moved)")
    ap.add_argument("--top-n", type=int, default=20, help="frequent dataset nouns added in instr+top")
    ap.add_argument("--vocab-file", type=Path, default=DEFAULT_VOCAB, help="word list for the 'fixed' vocabulary")
    ap.add_argument("--arm-prompts", nargs="+", default=["robot arm", "robotic arm", "black robot arm"])
    ap.add_argument("--gripper-prompts", nargs="+", default=["robot gripper", "gripper", "robot gripper fingers"])
    ap.add_argument("--threshold", type=float, default=0.4, help="min box score (confidence x presence)")
    ap.add_argument("--dup-iou", type=float, default=0.7, help="same-name boxes overlapping more are one object")
    ap.add_argument("--conflict-iou", type=float, default=0.7, help="different-name boxes overlapping more compete")
    ap.add_argument("--margin", type=float, default=0.15, help="score gap a name needs to win a competing box")
    ap.add_argument("--arm-overlap", type=float, default=0.5, help="box share on arm/gripper masks to reject it")
    ap.add_argument("--model", default="facebook/sam3")
    ap.add_argument("--fp32", action="store_true", help="run in fp32 (default on GPU: bf16, as it would ship)")
    ap.add_argument("--smoke", action="store_true", help="random weights, 1 frame per dataset, few names: code path only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "sam3_probe")
    args = ap.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.dtype = torch.float32 if args.fp32 or args.device == "cpu" else torch.bfloat16
    if args.smoke:
        args.episodes, args.frames, args.top_n = 1, 1, 2
        args.arm_prompts, args.gripper_prompts = args.arm_prompts[:1], args.gripper_prompts[:1]
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    fixed = load_word_list(args.vocab_file)[:3 if args.smoke else None]
    found = {}  # dataset -> (camera, usable episodes, frames, top names)
    for ds in args.datasets:
        root = args.input / ds
        if not (root / "meta" / "info.json").is_file():
            print(f"skip {ds}: no {root}/meta/info.json")
            continue
        cam, eps, frames = pick_frames(root, args.episodes, args.frames, tuple(args.span), rng)
        found[ds] = (cam, eps, frames, top_names(eps, args.top_n))
        print(f"{ds}: {len(frames)} frames from {len({f[0].index for f in frames})} episodes ({cam}); "
              f"top-{args.top_n}: {', '.join(found[ds][3])}")

    jobs = []  # (dataset, camera, ep, frame index, rgb, names per vocab)
    for ds, (cam, eps, frames, top) in found.items():
        # Negative control from the other datasets' common names (Fractal's cans are not in Bridge's toy kitchens);
        # with one dataset only, from this dataset's rarer names, which are less surely absent.
        others = [n for o, f in found.items() if o != ds for n in top_names(f[1], 60)]
        pool = others or top_names(eps, 300)[args.top_n:]
        for ep, fi, rgb in frames:
            instr = instruction_names(ep.task)
            names = {"instr": instr, "instr+top": list(dict.fromkeys(instr + top)), "fixed": fixed,
                     "neg": negative_names(ep, pool, top, rng)}
            jobs.append((ds, cam, ep, fi, rgb, names))
    if not jobs:
        sys.exit("no frames found")

    print(f"\nloading {args.model} on {args.device} ({args.dtype})...")
    t0 = time.perf_counter()
    sam = Sam3(args.model, args.device, args.dtype, args.smoke)
    print(f"loaded in {time.perf_counter() - t0:.1f} s")

    records, selections, rgbs = [], [], []
    arm_stats = {p: {"found": [], "top": [], "count": [], "area": []} for p in args.arm_prompts + args.gripper_prompts}
    timing = defaultdict(list)
    with open(args.out / "records.jsonl", "w") as jf:
        for j, (ds, cam, ep, fi, rgb, names) in enumerate(jobs):
            h, w = rgb.shape[:2]
            t = time.perf_counter()
            vision = sam.encode(rgb)
            timing["encode"].append(time.perf_counter() - t)

            arm, arm_mask = {}, np.zeros((h, w), bool)
            for p in args.arm_prompts + args.gripper_prompts:
                t = time.perf_counter()
                arm[p] = sam.query(vision, p, (h, w), masks=True)
                timing["query"].append(time.perf_counter() - t)
                good = [m for m, s in zip(arm[p]["masks"], arm[p]["scores"]) if s >= args.threshold]
                union = np.any(good, axis=0) if good else np.zeros((h, w), bool)
                arm_mask |= union
                st = arm_stats[p]
                st["found"].append(bool(good))
                st["top"].append(max(arm[p]["scores"], default=0.0))
                st["count"].append(len(good))
                if good:
                    st["area"].append(100.0 * union.mean())

            queries = list(dict.fromkeys(n for v in VOCABS for n in names[v]))
            answers = {}
            for nm in queries:
                t = time.perf_counter()
                answers[nm] = sam.query(vision, nm, (h, w))
                timing["query"].append(time.perf_counter() - t)
            timing["n_queries"].append(len(queries) + len(arm))

            rec = {"dataset": ds, "camera": cam, "episode": ep.index, "frame": fi, "task": ep.task, "names": names,
                   "answers": answers,
                   "arm": {p: {k: v for k, v in a.items() if k != "masks"} | {"mask_area_pct": round(
                       100.0 * float(np.any([m for m, s in zip(a["masks"], a["scores"]) if s >= args.threshold],
                                            axis=0).mean()) if any(s >= args.threshold for s in a["scores"]) else 0.0, 2)}
                           for p, a in arm.items()}}
            jf.write(json.dumps(rec) + "\n")
            sel = {v: select(answers, names[v], arm_mask, args) for v in VOCABS}
            records.append(rec)
            selections.append(sel)
            rgbs.append((rgb, arm))
            kept = "; ".join(f"{v}: {', '.join(k['name'] for k in sel[v]['kept']) or '-'}" for v in VOCABS)
            print(f"[{j + 1}/{len(jobs)}] {ds} ep{ep.index} f{fi} \"{ep.task}\" -> {kept}")

    for v in VOCABS:
        write_sheets([detection_tile(rgb, r, v, s[v]) for (rgb, _), r, s in zip(rgbs, records, selections)],
                     args.out / f"sheet_{v.replace('+', '_')}")
    write_sheets([arm_tile(rgb, arm, args) for rgb, arm in rgbs], args.out / "sheet_arm")

    summary = summarize(records, selections, arm_stats, timing, args)
    (args.out / "summary.txt").write_text(summary + "\n")
    print("\n" + summary + f"\n\nwrote {args.out}/")


if __name__ == "__main__":
    main()
