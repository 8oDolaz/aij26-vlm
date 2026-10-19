"""SAM 3 detection of dictionary objects on one frame, box filtering, and the per-frame cache.

SAM 3 answers one noun phrase per query: a presence probability ("is there any X?") and boxes scored by
box confidence x presence. It names nothing by itself, so every box is named by the dictionary entry it was found with.
Each frame is encoded once and every name is queried against it.

Raw answers (all boxes >= RAW_MIN_SCORE, masks for boxes >= mask_min, per-mask depth) are cached per frame, so the
filtering below and all question templates can be re-tuned without a GPU.

Box filtering (`select_candidates`), per frame, best score first:
  1. boxes below p.threshold or covering more than p.max_area of the frame are dropped;
  2. a box on the same region (IoU > p.dup_iou, or one p.contain inside the other) as a kept box of the same or a
     similar name is a duplicate and dropped (similar: vocab.similar);
  3. a box overlapping a kept box of an unrelated name (IoU > p.conflict_iou) is dropped if it scores clearly lower
     (by p.margin); on a near tie both boxes are dropped (ambiguous: the name cannot be trusted).
Only numpy at import time; torch, transformers and cv2 are imported where the model and depth are used.
"""
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aijgen.scene.vocab import Entry, similar

RAW_MIN_SCORE = 0.05  # boxes kept in the raw answers and the cache
CACHE_VERSION = 1


# ---------------------------------------------------------------------------------------------------------------------
# Box filtering
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class FilterParams:
    threshold: float = 0.85    # min box score (confidence x presence)
    max_area: float = 0.4      # drop boxes covering more of the frame (fridge, sink)
    dup_iou: float = 0.7       # similar-name boxes overlapping more are one object
    contain: float = 0.85      # similar-name boxes this much inside one another are one object
    conflict_iou: float = 0.7  # unrelated-name boxes overlapping more compete
    margin: float = 0.15       # score gap a name needs to win a competing box
    uncertain: float = 0.3     # boxes between this and threshold are "maybe there": they block counts and absence


def area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def overlaps(a, b) -> tuple[float, float]:
    """(IoU, intersection over the smaller box)."""
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = area(a) + area(b) - inter
    small = min(area(a), area(b))
    return (inter / union if union > 0 else 0.0), (inter / small if small > 0 else 0.0)


def same_region(a, b, p) -> bool:
    """Two boxes of similar names on one object (filter rule 2)."""
    iou, inside = overlaps(a, b)
    return iou > p.dup_iou or inside > p.contain


def select_candidates(cands: list[dict], entries: dict[str, Entry], frame_area: float, p) -> dict:
    """Kept candidates, the ambiguous ones, and counts of what was dropped (module docstring). A candidate is a dict
    with at least name, box (xyxy pixels) and score; other keys are carried along. Kept ones get "unique": their name
    is kept exactly once on the frame."""
    cands = sorted((c for c in cands if c["score"] >= p.threshold), key=lambda c: -c["score"])
    kept, ambiguous = [], []
    res = {"too_big": 0, "duplicate": 0, "lost": 0}
    for c in cands:
        if area(c["box"]) > p.max_area * frame_area:
            res["too_big"] += 1
            continue
        claimed = kept + ambiguous  # regions already taken, by a trusted or a contested box
        if any(similar(o["name"], c["name"], entries) and same_region(o["box"], c["box"], p) for o in claimed):
            res["duplicate"] += 1
            continue
        rivals = [o for o in claimed if not similar(o["name"], c["name"], entries)
                  and overlaps(o["box"], c["box"])[0] > p.conflict_iou]
        is_kept = lambda o: any(o is k for k in kept)  # identity: candidates may hold numpy masks
        if not rivals:
            kept.append(c)
        elif all(is_kept(o) and o["score"] - c["score"] > p.margin for o in rivals):
            res["lost"] += 1
        else:  # near tie with a kept box, or a region already contested: no name here can be trusted
            for o in rivals:
                if is_kept(o) and o["score"] - c["score"] <= p.margin:
                    kept = [k for k in kept if k is not o]
                    ambiguous.append(o)
            ambiguous.append(c)
    counts = Counter(k["name"] for k in kept)
    for k in kept:
        k["unique"] = counts[k["name"]] == 1
    return res | {"kept": kept, "ambiguous": ambiguous}


def select(answers: dict, names: list[str], entries: dict[str, Entry], frame_area: float, p) -> dict:
    """select_candidates over raw answers {name: {"boxes", "scores", "masks"?}} (the probe's format)."""
    cands = [{"name": n, "box": b, "score": s, "mask": m} for n in names for b, s, m in
             zip(answers[n]["boxes"], answers[n]["scores"],
                 answers[n].get("masks") or [None] * len(answers[n]["boxes"]))]
    return select_candidates(cands, entries, frame_area, p)


# ---------------------------------------------------------------------------------------------------------------------
# Per-frame detections and their cache
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class RawBox:
    name: str
    box: list[float]                 # xyxy pixels
    score: float
    mask: np.ndarray | None = None   # bool (h, w); only for score >= mask_min
    disp: float | None = None        # median normalised disparity (larger = closer) in the eroded mask


@dataclass
class FrameDetections:
    camera: str
    size: tuple[int, int]            # (h, w)
    names: list[str]                 # dictionary names queried
    negatives: list[str]             # absent-list names queried
    presence: dict[str, float]       # per queried name
    boxes: list[RawBox] = field(default_factory=list)
    params: dict = field(default_factory=dict)  # what produced it; a cache with other params is recomputed

    def by_name(self, name: str) -> list[RawBox]:
        return [b for b in self.boxes if b.name == name]


def _pack_mask(m: np.ndarray) -> tuple[list[int], np.ndarray]:
    """(crop rectangle y0, y1, x0, x1, packed bits of the crop)."""
    ys, xs = np.nonzero(m)
    if not len(ys):
        return [0, 0, 0, 0], np.zeros(0, np.uint8)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    return [y0, y1, x0, x1], np.packbits(m[y0:y1, x0:x1])


def _unpack_mask(rect: list[int], bits: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    y0, y1, x0, x1 = rect
    m = np.zeros(size, bool)
    n = (y1 - y0) * (x1 - x0)
    if n:
        m[y0:y1, x0:x1] = np.unpackbits(bits, count=n).astype(bool).reshape(y1 - y0, x1 - x0)
    return m


def save(path: Path, det: FrameDetections) -> None:
    """One compressed npz: metadata as JSON, masks as packed crops."""
    boxes, chunks, offset = [], [], 0
    for b in det.boxes:
        rec = {"name": b.name, "box": [round(float(v), 1) for v in b.box], "score": round(float(b.score), 4),
               "disp": None if b.disp is None else round(float(b.disp), 4)}
        if b.mask is not None:
            rect, bits = _pack_mask(b.mask)
            rec |= {"rect": rect, "bits": [offset, offset + len(bits)]}
            chunks.append(bits)
            offset += len(bits)
        boxes.append(rec)
    meta = {"version": CACHE_VERSION, "camera": det.camera, "size": list(det.size), "names": det.names,
            "negatives": det.negatives, "presence": det.presence, "boxes": boxes, "params": det.params}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)),
                        bits=np.concatenate(chunks) if chunks else np.zeros(0, np.uint8))
    tmp.replace(path)


def load(path: Path) -> FrameDetections | None:
    """The cached detections, or None if missing, unreadable or of another cache version."""
    try:
        with np.load(path) as z:
            meta, bits = json.loads(str(z["meta"])), z["bits"]
    except (OSError, ValueError, KeyError):
        return None
    if meta.get("version") != CACHE_VERSION:
        return None
    size = tuple(meta["size"])
    boxes = [RawBox(r["name"], r["box"], r["score"],
                    _unpack_mask(r["rect"], bits[r["bits"][0]:r["bits"][1]], size) if "rect" in r else None,
                    r["disp"]) for r in meta["boxes"]]
    return FrameDetections(meta["camera"], size, meta["names"], meta["negatives"], meta["presence"], boxes,
                           meta["params"])


# ---------------------------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------------------------
class Sam3:
    """facebook/sam3 via transformers (5.3.0 in the organizers' image). model_id may be a local directory."""

    def __init__(self, model_id: str, device: str | None = None, dtype=None, smoke: bool = False):
        import torch
        from transformers import Sam3Model, Sam3Processor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if self.device == "cuda" else torch.float32)
        if smoke:  # random weights and a public tokenizer: checks the code path only, boxes are meaningless
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
                sys.exit(f"cannot load {model_id}: {e}\nfacebook/sam3 is gated: request access on Hugging Face and "
                         "log in (hf auth login, or HF_TOKEN), or point the config at a local copy.")
        self.model.to(device=self.device, dtype=self.dtype).eval()
        self.text = {}

    def _sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def encode(self, rgb: np.ndarray):
        from PIL import Image
        with self.torch.inference_mode():
            pv = self.processor(images=Image.fromarray(rgb), return_tensors="pt")["pixel_values"]
            out = self.model.get_vision_features(pixel_values=pv.to(self.device, self.dtype))
        self._sync()
        return out

    def query(self, vision, prompt: str, size: tuple[int, int], masks: bool = False, mask_min: float = 0.0) -> dict:
        """Presence probability, boxes (xyxy pixels) and scores >= RAW_MIN_SCORE; with masks=True also "masks"
        (bool HxW) aligned with the boxes, None for boxes scoring below mask_min."""
        with self.torch.inference_mode():
            if prompt not in self.text:
                self.text[prompt] = {k: v.to(self.device)
                                     for k, v in self.processor(text=prompt, return_tensors="pt").items()}
            out = self.model(vision_embeds=vision, **self.text[prompt])
            presence = 1.0 if out.presence_logits is None else float(out.presence_logits.float().sigmoid().max())
            r = self.processor.post_process_object_detection(out, threshold=RAW_MIN_SCORE, target_sizes=[size])[0]
            ans = {"presence": round(presence, 4),
                   "scores": [round(float(s), 4) for s in r["scores"].float().cpu()],
                   "boxes": [[round(float(v), 1) for v in b] for b in r["boxes"].float().cpu()]}
            if masks:
                ans["masks"] = [None] * len(ans["boxes"])
                if any(s >= mask_min for s in ans["scores"]):
                    seg = self.processor.post_process_instance_segmentation(out, threshold=mask_min,
                                                                            target_sizes=[size])[0]
                    seg_boxes = [[float(v) for v in b] for b in seg["boxes"].float().cpu()]
                    seg_masks = [m.bool().cpu().numpy() for m in seg["masks"]]
                    for i, (b, s) in enumerate(zip(ans["boxes"], ans["scores"])):
                        if s < mask_min or not seg_boxes:  # same model output: the matching box is identical
                            continue
                        j = max(range(len(seg_boxes)), key=lambda k: overlaps(b, seg_boxes[k])[0])
                        if overlaps(b, seg_boxes[j])[0] > 0.9:
                            ans["masks"][i] = seg_masks[j]
        self._sync()
        return ans


def detect(sam: Sam3, rgb: np.ndarray, camera: str, entries: dict[str, Entry], negatives: list[str],
           mask_min: float, params: dict | None = None) -> FrameDetections:
    """All dictionary names and the given absent-list names queried on one frame. Masks for dictionary boxes only."""
    h, w = rgb.shape[:2]
    vision = sam.encode(rgb)
    det = FrameDetections(camera, (h, w), list(entries), list(negatives), {}, [], dict(params or {}))
    for name in det.names + det.negatives:
        is_dict = name in entries
        a = sam.query(vision, entries[name].prompt if is_dict else name, (h, w), masks=is_dict, mask_min=mask_min)
        det.presence[name] = a["presence"]
        for i, (b, s) in enumerate(zip(a["boxes"], a["scores"])):
            det.boxes.append(RawBox(name, b, s, a["masks"][i] if is_dict else None))
    return det


def add_depth(det: FrameDetections, disp: np.ndarray, erode_frac: float) -> None:
    """Set RawBox.disp: median of the per-frame normalised disparity (aijgen.depth.pairs.normalise; 0 far .. 1 near)
    inside each mask, eroded by erode_frac of the width so depth edges do not leak in (the raw mask if erosion
    empties it)."""
    import cv2
    from aijgen.depth.pairs import normalise
    dn = normalise(disp)
    r = max(1, round(erode_frac * det.size[1]))
    kernel = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    for b in det.boxes:
        if b.mask is None or not b.mask.any():
            continue
        inner = cv2.erode(b.mask.astype(np.uint8), kernel).astype(bool)
        b.disp = float(np.median(dn[inner if inner.any() else b.mask]))
