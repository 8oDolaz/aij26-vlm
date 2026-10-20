"""Object-level cross-view matching on two Bridge views of the same moment (docs/sam3_vqa_plan.md, P4).

  crossview_crosspoint  CrossPoint Correspondence-Judgement: a red dot on an object in image 1; green / blue / yellow
                        dots in image 2, one on the same object. Only small objects, so "the same object" is close to
                        "the same 3D point", which is what the bench asks.
  crossview_erqa        ERQA Multi-view: "Which circle in the second image matches the same object as the green
                        circle in the first image?", 4 coloured circles in image 2.

The match is by name, so both ends must be safe to name (build_scene subjects: the only box of their kind on the
frame, no stray "maybe" box of a similar name). Conservative checks on top:
  - the name is a subject in both views, and in every other non-blank view it is either a subject or not seen at all
    (two same-kind objects, one visible per view, would otherwise look like one);
  - a distractor is a kept object of a non-similar name whose box hardly overlaps the target's box in image 2
    (IoU < distractor_iou, containment < distractor_contain), so a renamed copy of the target cannot be a distractor;
  - every marker sits on the most interior pixels of its eroded mask, inside the image border and min_dot_dist from
    the others.
Pure numpy; markers are overlay specs with "image": 0 or 1 (aijgen.scene.render).
"""
import numpy as np

from aijgen.scene import formats as F
from aijgen.scene.detect import overlaps
from aijgen.scene.geometry import erode, far_enough, in_box, inside_border
from aijgen.scene.templates import DOT_COLORS, Item, Obj, Scene
from aijgen.scene.vocab import similar

RING_COLORS = {"red": (230, 30, 30), "green": (40, 180, 60), "blue": (40, 90, 235), "yellow": (250, 215, 0),
               "purple": (150, 60, 200), "orange": (255, 140, 0), "pink": (255, 105, 180)}
CROSSPOINT_COLORS = ("green", "blue", "yellow")  # the bench's three candidate colours; the query dot is red

ERQA_Q = [  # ERQA Multi-view wordings, the object-level one first
    "Which circle in the second image matches the same object as the {c} circle in the first image?",
    "Which circle in the second image marks the same object as the {c} circle in the first image?",
    "In the second image, select the circle that marks the same object as the {c} circle in the first image.",
]
ERQA_OPTION = ["{C} circle", "The {c} circle", "{C}"]  # "Red circle." / "The blue circle." / "Blue."


def interior_point(mask: np.ndarray, border: float = 0.05) -> tuple[int, int] | None:
    """(x, y) on the most interior part of the mask: the pixels left by the largest erosion that keeps any, the one
    nearest their centroid. None if the mask is empty or that point is within `border` of the image edge."""
    if mask is None or not mask.any():
        return None
    ys, xs = np.nonzero(mask)  # work on the mask's box plus a 1 px margin: same erosion, much less area
    y0, x0 = max(int(ys.min()) - 1, 0), max(int(xs.min()) - 1, 0)
    crop = mask[y0:int(ys.max()) + 2, x0:int(xs.max()) + 2]
    lo, hi = 0, 1
    while erode(crop, hi).any():  # find a radius that empties the mask
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if erode(crop, mid).any() else (lo, mid)
    core = erode(crop, lo) if lo else crop
    ys, xs = np.nonzero(core)
    i = int(np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2))
    xy = (int(xs[i]) + x0, int(ys[i]) + y0)
    return xy if inside_border(xy, mask.shape, border) else None


def diag(box) -> float:
    return float(np.hypot(box[2] - box[0], box[3] - box[1]))


def _subject(scene: Scene, name: str) -> Obj | None:
    return next((s for s in scene.subjects if s.name == name), None)


def matches(s1: Scene, s2: Scene, others: list[Scene], entries) -> list[tuple[Obj, Obj]]:
    """(object in view 1, the same object in view 2) for names that are subjects in both views and, in every other
    view, a subject or absent (no kept box of a similar name)."""
    out = []
    for a in s1.subjects:
        b = _subject(s2, a.name)
        if b is None or a.mask is None or b.mask is None:
            continue
        if all(_subject(s, a.name) is not None or not any(similar(o.name, a.name, entries) for o in s.objects)
               for s in others):
            out.append((a, b))
    return out


def distractors(s2: Scene, target: Obj, entries, cfg) -> list[Obj]:
    """Kept objects of view 2 that are certainly not the target (module docstring)."""
    iou_max, contain_max = cfg.get("distractor_iou", 0.1), cfg.get("distractor_contain", 0.3)
    out = []
    for o in s2.objects:
        if o is target or o.mask is None or similar(o.name, target.name, entries):
            continue
        iou, inside = overlaps(o.box, target.box)
        if iou < iou_max and inside < contain_max:
            out.append(o)
    return out


def _points(s2: Scene, target: Obj, cands: list[Obj], n: int, rng, cfg):
    """[target point, n distractor points] in view 2, or None."""
    w = s2.size[1]
    border, min_dist, pad = cfg.get("border", 0.05), cfg.get("min_dot_dist", 0.08) * w, cfg.get("box_pad", 0.02) * w
    first = interior_point(target.mask, border)
    if first is None:
        return None
    pts = [first]
    for i in rng.permutation(len(cands)):
        if len(pts) > n:
            break
        xy = interior_point(cands[i].mask, border)
        if xy is not None and far_enough(xy, pts, min_dist) and not in_box(xy, target.box, pad):
            pts.append(xy)
    return pts if len(pts) == n + 1 else None


def _pairs(s1, s2, others, entries, rng, cfg, n_distractors, small_only):
    """(query point in view 1, [answer point, distractor points] in view 2) for up to per_frame targets."""
    out = []
    w1, w2 = s1.size[1], s2.size[1]
    found = matches(s1, s2, others, entries)
    for i in rng.permutation(len(found)):
        if len(out) >= cfg.get("per_frame", 1):
            break
        a, b = found[i]
        lim = cfg.get("max_target_frac", 0.15)
        if small_only and (diag(a.box) > lim * w1 or diag(b.box) > lim * w2):
            continue
        q = interior_point(a.mask, cfg.get("border", 0.05))
        pts = _points(s2, b, distractors(s2, b, entries, cfg), n_distractors, rng, cfg) if q is not None else None
        if pts is not None:
            out.append((a, q, pts))
    return out


def crossview_crosspoint(s1, s2, others, entries, rng, cfg) -> list[Item]:
    items = []
    for a, q, pts in _pairs(s1, s2, others, entries, rng, cfg, n_distractors=2, small_only=True):
        colors = [CROSSPOINT_COLORS[j] for j in rng.permutation(3)]  # colours[0] marks the answer
        overlays = [{"kind": "dot", "xy": list(q), "color": list(DOT_COLORS["red"]), "label": None, "image": 0}]
        overlays += [{"kind": "dot", "xy": list(p), "color": list(DOT_COLORS[c]), "label": None, "image": 1}
                     for p, c in zip(pts, colors)]
        options, letter = F.shuffle(f"{colors[0].capitalize()} dot", [f"{c.capitalize()} dot" for c in colors[1:]], rng)
        question = F.CROSSPOINT_JUDGE_Q[int(rng.integers(len(F.CROSSPOINT_JUDGE_Q)))]
        items.append(Item("crossview_crosspoint", "crosspoint", F.crosspoint(question, options), letter, overlays))
    return items


def crossview_erqa(s1, s2, others, entries, rng, cfg) -> list[Item]:
    items = []
    names = list(RING_COLORS)
    for a, q, pts in _pairs(s1, s2, others, entries, rng, cfg, n_distractors=3, small_only=False):
        picks = [names[j] for j in rng.permutation(len(names))[:5]]
        query, colors = picks[0], picks[1:]  # the query colour is not among the options
        overlays = [{"kind": "ring", "xy": list(q), "color": list(RING_COLORS[query]), "image": 0}]
        overlays += [{"kind": "ring", "xy": list(p), "color": list(RING_COLORS[c]), "image": 1}
                     for p, c in zip(pts, colors)]
        form = ERQA_OPTION[int(rng.integers(len(ERQA_OPTION)))]
        text = lambda c: form.format(c=c, C=c.capitalize())
        options, letter = F.shuffle(text(colors[0]), [text(c) for c in colors[1:]], rng)
        question = ERQA_Q[int(rng.integers(len(ERQA_Q)))].format(c=query)
        items.append(Item("crossview_erqa", "erqa", F.erqa(question, options), letter, overlays))
    return items


TEMPLATES = {f.__name__: f for f in (crossview_crosspoint, crossview_erqa)}


def make_pair_items(s1: Scene, s2: Scene, others: list[Scene], entries, rng, templates: dict) -> list[Item]:
    """Questions of every enabled cross-view template (config order), at most per_frame each."""
    items = []
    for name, cfg in templates.items():
        if name not in TEMPLATES:
            raise ValueError(f"unknown cross-view template {name!r} (known: {', '.join(TEMPLATES)})")
        cfg = cfg or {}
        if cfg.get("enabled", True) and cfg.get("per_frame", 1) > 0:
            items += TEMPLATES[name](s1, s2, others, entries, rng, cfg)[:cfg.get("per_frame", 1)]
    return items
