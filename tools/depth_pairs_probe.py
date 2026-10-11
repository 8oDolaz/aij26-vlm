#!/usr/bin/env python3
"""Check whether "which object is closer to the camera" can be labelled reliably, before building the generator.

Objects on one counter or shelf differ little in depth, where a monocular depth model is least reliable. This probe
detects objects with OWLv2 and orders every pair of them three independent ways:
- contact: the object whose box bottom (its contact with the surface) is lower in the image is closer. Exact
  perspective geometry for objects resting on one flat surface under a camera looking down at it, but it assumes
  that camera pose (no strong roll, not straight down) and objects that are not stacked or held;
- object: median depth-model value inside the box (shrunk by SHRINK on each side, to stay off the edges);
- support: depth-model value on the surface just below the contact point. Follows the surface's real depth
  gradient, so it does not care how the camera is tilted or rolled.
A labeller abstains when the pair's gap is below its threshold (--contact-gap: fraction of image height;
--depth-gap: fraction of the frame's 2nd..98th percentile depth range). Pairwise agreement where both decide shows
which labels hold on this camera: run it on several cameras and sessions to see whether contact breaks with pose.

An object resting on the surface reads about the same depth in its middle as on the surface just below it, so an
object whose object and support depths differ by more than --consistency (same units as --depth-gap) is rejected
and never paired. That drops boxes whose support patch lands on the arm or another object, and thin or hollow
objects whose box is mostly background. Rejection is meant to be aggressive: there is far more data than we need.

Writes to --out (default runs/depth_pairs/<camera>_<file>_<from>-<to>/):
- detections.png: up to --sheet frames, boxes with labels | depth; green kept, red rejected, grey robot
- pairs.png: up to --sheet pairs, half where labellers disagree (to see who is right), half where all agree;
  A is red, B is orange, dots mark contact points
- pairs.jsonl: every pair with its normalised gaps and verdicts
and prints detections per frame, pairwise agreement and the yield of pairs all three agree on.

Run in the dev container, after tools/depth_probe.py works there (same dependencies plus torchvision):
  docker compose exec dev python tools/depth_pairs_probe.py \
      data/raw/bridge/videos/observation.images.image_0/chunk-000/file-086.mp4 --from 600 --to 640
  docker compose exec dev python tools/depth_pairs_probe.py <video> --from 0 --to 60 --queries pot lid "stuffed toy"
"""
import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from transformers import AutoProcessor, Owlv2ForObjectDetection

from depth_probe import REPO, colorize, predict, read_frames

# Common Bridge and Fractal objects; ROBOT is detected only so it can be kept out of the pairs.
QUERIES = ["pot", "pan", "lid", "bowl", "cup", "plate", "spoon", "spatula", "knife", "brush", "cloth", "towel",
           "sponge", "stuffed toy", "fruit", "banana", "apple", "orange", "carrot", "corn", "eggplant", "can",
           "bottle", "chip bag", "toy"]
ROBOT = "robot gripper"
LABELLERS = ("contact", "object", "support")
NMS_IOU = 0.5      # class-agnostic NMS over all queries
MIN_AREA = 0.002   # box area as a fraction of the image: drops specks
MAX_AREA = 0.25    # drops the counter, stove or sink detected as an "object"
PAIR_IOU = 0.3     # boxes overlapping more than this are probably stacked or occluding: not a pair
SHRINK = 0.25      # object depth: inner part of the box only
SUP_OFF, SUP_H, SUP_W = 3, 10, 6  # support patch: starts SUP_OFF px below the box, SUP_H tall, 2*SUP_W+1 wide
TILE_H = 240
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLORS = {"A": (0, 0, 255), "B": (0, 160, 255)}  # BGR


@torch.inference_mode()
def detect(frames, model_id: str, queries: list[str], score: float, batch: int):
    """Per frame, the boxes that pass the score, NMS and area filters, as dicts with label, score and box."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_id)
    model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device, dtype).eval()
    post = getattr(processor, "post_process_grounded_object_detection", None) or processor.post_process_object_detection
    h, w = frames[0].shape[:2]
    side = max(h, w)  # OWLv2 pads to a square at the bottom and right, and its boxes are in that square
    out = []
    for i in range(0, len(frames), batch):
        chunk = frames[i:i + batch]
        inputs = processor(text=[queries] * len(chunk), images=chunk, return_tensors="pt").to(device)
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        results = post(outputs=model(**inputs), threshold=score, target_sizes=[(side, side)] * len(chunk))
        for r in results:
            boxes, scores = r["boxes"].float().cpu(), r["scores"].float().cpu()
            dets = []
            for k in nms(boxes, scores, NMS_IOU).tolist():
                x1, y1, x2, y2 = boxes[k].tolist()
                x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
                if not MIN_AREA <= (x2 - x1) * (y2 - y1) / (h * w) <= MAX_AREA:
                    continue
                dets.append({"label": queries[int(r["labels"][k])], "score": round(float(scores[k]), 3),
                             "box": [round(v, 1) for v in (x1, y1, x2, y2)]})
            out.append(dets)
    return out


def iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def object_depth(depth: np.ndarray, box) -> float:
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * SHRINK, (y2 - y1) * SHRINK
    return float(np.median(depth[int(y1 + my):int(y2 - my) + 1, int(x1 + mx):int(x2 - mx) + 1]))


def support_depth(depth: np.ndarray, box) -> float | None:
    """Depth of the surface just below the box's bottom centre; None when the box touches the image bottom."""
    h, w = depth.shape
    xc, y = int((box[0] + box[2]) / 2), int(box[3]) + SUP_OFF
    patch = depth[y:min(y + SUP_H, h), max(xc - SUP_W, 0):min(xc + SUP_W + 1, w)]
    return float(np.median(patch)) if patch.size else None


def verdict(a, b, gap: float):
    """Which of A, B is closer when larger means closer, or None when undecided."""
    if a is None or b is None or abs(a - b) < gap:
        return None
    return "A" if a > b else "B"


def make_pairs(dets_per_frame, depths, times, contact_gap: float, depth_gap: float, consistency: float):
    pairs = []
    for i, (dets, depth) in enumerate(zip(dets_per_frame, depths)):
        h = depth.shape[0]
        lo, hi = np.percentile(depth, (2, 98))
        rng = max(hi - lo, 1e-6)
        objs = [d for d in dets if d["label"] != ROBOT]
        for o in objs:  # every value normalised so that larger means closer
            o["contact"] = o["box"][3] / h
            o["object"] = (object_depth(depth, o["box"]) - lo) / rng
            sup = support_depth(depth, o["box"])
            o["support"] = None if sup is None else (sup - lo) / rng
            o["rejected"] = bool(o["support"] is None or abs(o["object"] - o["support"]) > consistency)
        for a, b in itertools.combinations([o for o in objs if not o["rejected"]], 2):
            if iou(a["box"], b["box"]) > PAIR_IOU:
                continue
            gaps = {k: verdict(a[k], b[k], contact_gap if k == "contact" else depth_gap) for k in LABELLERS}
            pairs.append({"i": i, "t": round(times[i], 2), "a": a, "b": b, "verdict": gaps,
                          "diff": {k: None if a[k] is None or b[k] is None else round(a[k] - b[k], 3)
                                   for k in LABELLERS}})
    return pairs


def summarize(pairs, dets_per_frame):
    n_frames = len(dets_per_frame)
    n_objs = sum(sum(d["label"] != ROBOT for d in dets) for dets in dets_per_frame)
    n_rejected = sum(sum(d.get("rejected", False) for d in dets) for dets in dets_per_frame)
    print(f"{n_frames} frames, {n_objs / n_frames:.1f} objects/frame, {n_rejected}/{n_objs} rejected, "
          f"{len(pairs)} pairs")
    for k in LABELLERS:
        decided = sum(p["verdict"][k] is not None for p in pairs)
        print(f"  {k:8s} decides {decided}/{len(pairs)}")
    for x, y in itertools.combinations(LABELLERS, 2):
        both = [p for p in pairs if p["verdict"][x] and p["verdict"][y]]
        agree = sum(p["verdict"][x] == p["verdict"][y] for p in both)
        print(f"  {x} vs {y}: agree {agree}/{len(both)}" + (f" = {100 * agree / len(both):.0f}%" if both else ""))
    usable = sum(len(set(p["verdict"].values())) == 1 and p["verdict"]["contact"] is not None for p in pairs)
    print(f"  all three decide and agree: {usable} pairs = {usable / n_frames:.1f}/frame")


def mark(img: np.ndarray, p) -> np.ndarray:
    img = img.copy()
    for tag in ("A", "B"):
        o, color = p[tag.lower()], COLORS[tag]
        x1, y1, x2, y2 = map(int, o["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.circle(img, ((x1 + x2) // 2, y2), 5, color, -1)
        cv2.putText(img, f"{tag}: {o['label']}", (x1, max(y1 - 8, 22)), FONT, 0.9, color, 2)
    return img


def tile(left: np.ndarray, right: np.ndarray, caption: str) -> np.ndarray:
    t = np.hstack([left, right])
    t = cv2.resize(t, (round(t.shape[1] * TILE_H / t.shape[0]), TILE_H), interpolation=cv2.INTER_AREA)
    cv2.rectangle(t, (0, 0), (t.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(t, caption, (6, 16), FONT, 0.5, (255, 255, 255), 1)
    return t


def write_grid(path, tiles, cols: int = 2):
    if not tiles:
        return
    tiles = tiles + [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]))


def spread(items, n: int):
    """Up to n items evenly spaced through the list, deterministic."""
    if len(items) <= n:
        return items
    return [items[i] for i in np.linspace(0, len(items) - 1, n).round().astype(int)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--from", dest="start", type=float, required=True, help="start, seconds into the file")
    ap.add_argument("--to", dest="end", type=float, required=True, help="end, seconds into the file")
    ap.add_argument("--stride", type=int, default=5, help="keep every n-th frame")
    ap.add_argument("--queries", nargs="+", default=QUERIES, help="object names for OWLv2")
    ap.add_argument("--detector", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Base-hf")
    ap.add_argument("--score", type=float, default=0.5, help="OWLv2 score threshold")
    ap.add_argument("--contact-gap", type=float, default=0.03, help="min contact-point gap, fraction of height")
    ap.add_argument("--depth-gap", type=float, default=0.05, help="min depth gap, fraction of the frame's range")
    ap.add_argument("--consistency", type=float, default=0.10,
                    help="max object vs support depth difference, fraction of the frame's range")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--sheet", type=int, default=12, help="frames and pairs on the contact sheets")
    ap.add_argument("--out", type=Path, help="output directory")
    args = ap.parse_args()
    if args.end <= args.start:
        ap.error("--to must be after --from")

    frames, times, _ = read_frames(args.video, args.start, args.end, args.stride)
    if not frames:
        raise SystemExit(f"no frames in [{args.start}, {args.end}] s of {args.video}")
    camera = args.video.parents[1].name.removeprefix("observation.images.")
    out = args.out or REPO / "runs" / "depth_pairs" / f"{camera}_{args.video.stem}_{args.start:g}-{args.end:g}"
    out.mkdir(parents=True, exist_ok=True)

    dets_per_frame = detect(frames, args.detector, list(args.queries) + [ROBOT], args.score, args.batch)
    depths, _, _ = predict(frames, args.depth_model, args.batch)
    pairs = make_pairs(dets_per_frame, depths, times, args.contact_gap, args.depth_gap, args.consistency)
    summarize(pairs, dets_per_frame)

    det_tiles = []
    for i in spread(list(range(len(frames))), args.sheet):
        img = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        for d in dets_per_frame[i]:
            x1, y1, x2, y2 = map(int, d["box"])
            color = (128, 128, 128) if d["label"] == ROBOT else (0, 0, 255) if d.get("rejected") else (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{d['label']} {d['score']:.2f}", (x1, max(y1 - 8, 22)), FONT, 0.8, color, 2)
        det_tiles.append(tile(img, colorize(depths[i]), f"t={times[i]:.1f}s  {len(dets_per_frame[i])} boxes"))
    write_grid(out / "detections.png", det_tiles)

    decided = [p for p in pairs if sum(v is not None for v in p["verdict"].values()) >= 2]
    disagree = [p for p in decided if len({v for v in p["verdict"].values() if v}) > 1]
    agree = [p for p in pairs if len(set(p["verdict"].values())) == 1 and p["verdict"]["contact"]]
    shown = spread(disagree, args.sheet // 2)
    shown += spread(agree, args.sheet - len(shown))
    pair_tiles = []
    for p in shown:
        caption = f"t={p['t']}s  " + "  ".join(f"{k}:{p['verdict'][k] or '-'}" for k in LABELLERS)
        rgb = cv2.cvtColor(frames[p["i"]], cv2.COLOR_RGB2BGR)
        pair_tiles.append(tile(mark(rgb, p), mark(colorize(depths[p["i"]]), p), caption))
    write_grid(out / "pairs.png", pair_tiles)

    with open(out / "pairs.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps({k: v for k, v in p.items() if k != "i"}) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
