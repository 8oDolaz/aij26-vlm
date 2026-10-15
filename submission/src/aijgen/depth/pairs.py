"""Tier 1 point pairs: sampling by segment, gap and edge filters, temporal stability, balancing, shortcut audit.

Disparities are normalised per frame, dn = (d - p2) / (p98 - p2), so larger means closer and the gap between two
points is the confidence `g` calibrated on NYU-v2 (docs/relative_depth_plan.md, section 2.2): with both points off
depth edges, g >= 0.15 orders about 98.5% of pairs correctly. Pairs whose closer point is higher in the image
(against the "lower is closer" prior the depth model shares) are only about 97% right at the same gap, so they get a
stricter threshold.

Points and pairs are plain dicts so they serialise straight into the metadata sidecar.
"""
import itertools

import cv2
import numpy as np

GROUPS = ("consistent", "inconsistent", "same_row")  # closer point is lower / higher / on the same row


def normalise(d: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(d, (2, 98))
    return ((d - lo) / max(hi - lo, 1e-6)).astype(np.float32)


def edge_map(dn: np.ndarray, radius: int) -> np.ndarray:
    """Max minus min of dn in a (2r+1)^2 window: large on depth discontinuities."""
    k = np.ones((2 * radius + 1,) * 2, np.uint8)
    return cv2.dilate(dn, k) - cv2.erode(dn, k)


def _point(dn, y, x, seg, area, kind):
    return {"y": int(y), "x": int(x), "seg": int(seg), "area": round(float(area), 4), "kind": kind,
            "d": round(float(dn[y, x]), 4)}


def sample_points(labels: np.ndarray, dn: np.ndarray, cfg: dict, rng: np.random.Generator) -> list[dict]:
    """A few points per segment (not per pixel area), off depth edges and away from segment boundaries.

    SAM2 segments give `object` points, or `background` ones when the segment is huge or far (walls, cabinets).
    Pixels no mask covers (bare table, counter) are split into a grid with one `surface` point per cell, or a
    `background` point when the cell is far. No point lies within `border_frac` of the width (left, right) or height
    (top, bottom) of the image border.
    """
    h, w = dn.shape
    ok = edge_map(dn, cfg["edge_radius"]) < cfg["edge_max"]
    by, bx = round(cfg["border_frac"] * h), round(cfg["border_frac"] * w)
    ok[:by], ok[h - by:], ok[:, :bx], ok[:, w - bx:] = False, False, False, False
    r = max(1, round(cfg["erode_frac"] * w))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    points = []
    ids, counts = np.unique(labels, return_counts=True)
    for sid, count in zip(ids.tolist(), counts.tolist()):
        area = count / labels.size
        if sid == 0 or area < cfg["min_segment_area"]:
            continue
        mask = labels == sid
        ys, xs = np.nonzero(cv2.erode(mask.astype(np.uint8), kernel).astype(bool) & ok)
        if not len(ys):
            continue
        far = float(np.median(dn[mask])) < cfg["background_disp"]
        kind = "background" if far or area > cfg["large_segment_area"] else "object"
        for i in rng.choice(len(ys), min(cfg["points_per_segment"], len(ys)), replace=False):
            points.append(_point(dn, ys[i], xs[i], sid, area, kind))
    free = labels == 0
    inner = cv2.erode(free.astype(np.uint8), kernel).astype(bool) & ok
    rows, cols = cfg["grid"]
    for gr in range(rows):
        for gc in range(cols):
            y0, x0 = gr * h // rows, gc * w // cols
            ys, xs = np.nonzero(inner[y0:(gr + 1) * h // rows, x0:(gc + 1) * w // cols])
            if len(ys) < cfg["min_cell_pixels"]:
                continue
            i = rng.integers(len(ys))
            y, x = ys[i] + y0, xs[i] + x0
            kind = "background" if dn[y, x] < cfg["background_disp"] else "surface"
            points.append(_point(dn, y, x, -(gr * cols + gc + 1), free.mean(), kind))
    return points


def candidate_pairs(points: list[dict], shape: tuple[int, int], cfg: dict) -> tuple[list[dict], int]:
    """Pairs of points from different segments that pass the gap threshold of their group.

    Returns (passing pairs, number of pairs considered before the gap filter).
    """
    h, w = shape
    out, considered = [], 0
    for a, b in itertools.combinations(points, 2):
        if a["seg"] == b["seg"] or np.hypot(a["x"] - b["x"], a["y"] - b["y"]) < cfg["min_dist_frac"] * w:
            continue
        considered += 1
        gap = a["d"] - b["d"]
        near, far = (a, b) if gap > 0 else (b, a)
        if abs(a["y"] - b["y"]) <= cfg["same_row_frac"] * h:
            group, need = "same_row", cfg["gap_same_row"]
        elif near["y"] > far["y"]:
            group, need = "consistent", cfg["gap"]
        else:
            group, need = "inconsistent", cfg["gap_inconsistent"]
        if abs(gap) >= need:
            out.append({"near": near, "far": far, "g": round(abs(gap), 4), "group": group,
                        "background": "background" in (near["kind"], far["kind"])})
    return out, considered


def stable(pair: dict, neighbours: list[np.ndarray], cfg: dict) -> bool:
    """Same order in every nearby frame where neither point changed (static camera; the arm may pass over a point).

    A point whose normalised disparity moved by more than `stability_change` is treated as occluded or moved and
    that frame is skipped for this pair. Needs at least `stability_min_valid` usable frames.
    """
    n, f = pair["near"], pair["far"]
    valid = agree = 0
    for dn in neighbours:
        dn_n, dn_f = float(dn[n["y"], n["x"]]), float(dn[f["y"], f["x"]])
        if abs(dn_n - n["d"]) > cfg["stability_change"] or abs(dn_f - f["d"]) > cfg["stability_change"]:
            continue
        valid += 1
        agree += dn_n > dn_f
    return valid >= cfg["stability_min_valid"] and agree == valid


def per_frame(pairs: list[dict], k: int, rng: np.random.Generator) -> list[dict]:
    """Up to k pairs per group from one frame, never reusing a segment within a group."""
    out = []
    for group in GROUPS:
        used, taken = set(), 0
        for i in rng.permutation(len(pairs)):
            p = pairs[i]
            segs = {p["near"]["seg"], p["far"]["seg"]}
            if p["group"] != group or segs & used:
                continue
            out.append(p)
            used |= segs
            taken += 1
            if taken == k:
                break
    return out


def _capacity(pairs: list[dict], bg_share: float) -> int:
    nonbg = sum(not p["background"] for p in pairs)
    bg = len(pairs) - nonbg
    return nonbg + min(bg, int(nonbg * bg_share / (1 - bg_share)))


def _take(pairs: list[dict], n: int, bg_share: float, rng: np.random.Generator) -> list[dict]:
    """n pairs, with background pairs at most bg_share of them (n must be within _capacity)."""
    shuffled = [pairs[i] for i in rng.permutation(len(pairs))]
    nonbg = [p for p in shuffled if not p["background"]]
    bg = [p for p in shuffled if p["background"]]
    n_bg = min(len(bg), round(bg_share * n))
    n_nonbg = min(len(nonbg), n - n_bg)
    n_bg = n - n_nonbg
    return nonbg[:n_nonbg] + bg[:n_bg]


def balance(pairs: list[dict], bg_share: float, rng: np.random.Generator) -> list[dict]:
    """Equal numbers of shortcut-consistent and -inconsistent pairs, plus all same-row pairs: they carry no
    "lower is closer" signal, so they need no partner (Fractal has few inconsistent pairs, but many same-row ones).
    Background pairs are capped at bg_share in every group."""
    by = {g: [p for p in pairs if p["group"] == g] for g in GROUPS}
    n = min(_capacity(by["consistent"], bg_share), _capacity(by["inconsistent"], bg_share))
    out = _take(by["consistent"], n, bg_share, rng) + _take(by["inconsistent"], n, bg_share, rng)
    out += _take(by["same_row"], _capacity(by["same_row"], bg_share), bg_share, rng)
    return out


def assign_letters(pairs: list[dict], rng: np.random.Generator) -> None:
    """Sets pair["a_is_near"] so that within every group exactly half the pairs have A closer. With the groups
    balanced this also makes "A is the lower point" 50/50."""
    for group in GROUPS:
        members = [p for p in pairs if p["group"] == group]
        flags = np.arange(len(members)) % 2 == 0
        rng.shuffle(flags)
        for p, flag in zip(members, flags):
            p["a_is_near"] = bool(flag)


def points_ab(p: dict) -> tuple[dict, dict]:
    return (p["near"], p["far"]) if p["a_is_near"] else (p["far"], p["near"])


def audit(pairs: list[dict], shape_of) -> dict:
    """Accuracy of label-free shortcuts at predicting which letter is closer. All should be near 0.5.

    shape_of(pair) -> (h, w) of the pair's frame, for the centre baseline.
    """
    if not pairs:
        return {}
    hits = {"lower_is_closer": [], "bigger_segment_is_closer": [], "nearer_centre_is_closer": [], "always_A": []}
    for p in pairs:
        a, b = points_ab(p)
        h, w = shape_of(p)
        truth_a = p["a_is_near"]
        hits["lower_is_closer"].append((a["y"] >= b["y"]) == truth_a)
        hits["bigger_segment_is_closer"].append((a["area"] >= b["area"]) == truth_a)
        ca, cb = (np.hypot(q["x"] - w / 2, q["y"] - h / 2) for q in (a, b))
        hits["nearer_centre_is_closer"].append((ca <= cb) == truth_a)
        hits["always_A"].append(truth_a)
    return {k: round(float(np.mean(v)), 3) for k, v in hits.items()}
