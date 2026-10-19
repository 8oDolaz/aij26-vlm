"""Scene questions from one frame's SAM 3 detections (docs/sam3_vqa_plan.md, "Template decisions").

build_scene turns raw detections into what questions may rely on:
  objects   boxes kept by the filter (detect.select_candidates)
  subjects  objects safe to name in a question: the only one of their kind on the frame (no other kept, contested or
            "maybe" box >= filter.uncertain of a similar name elsewhere) and not in scene.exclude_subjects
  counts    names whose count is certain: every box >= filter.uncertain of a similar name is one of the kept boxes
  absent    absent-list names SAM 3 confirms absent on this frame (SAM 3 misses things, so "not detected" is never
            used as "absent")
make_items runs the templates enabled in config, each giving at most `per_frame` questions per frame.
Pure numpy; markers are returned as overlay specs and drawn by aijgen.scene.render.
"""
from dataclasses import dataclass, field

import numpy as np

from aijgen.scene import formats as F
from aijgen.scene.detect import FrameDetections, select_candidates, same_region
from aijgen.scene.geometry import (OPPOSITE, RELATIONS, erode, far_enough, in_box, inside_border, perturbed_box,
                                   relations, sample_background, sample_in_mask)
from aijgen.scene.vocab import Entry, plural, similar


@dataclass
class Obj:
    name: str
    box: list[float]
    score: float
    mask: np.ndarray | None = None
    disp: float | None = None


@dataclass
class Scene:
    size: tuple[int, int]                                  # (h, w)
    objects: list[Obj] = field(default_factory=list)
    subjects: list[Obj] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    absent: list[str] = field(default_factory=list)
    occupied: list[list[float]] = field(default_factory=list)  # every box >= filter.uncertain, any name


@dataclass
class Item:
    template: str
    bench: str
    question: str
    answer: str
    overlays: list[dict] = field(default_factory=list)


def build_scene(det: FrameDetections, entries: dict[str, Entry], p, exclude_subjects=(), absent_presence=0.05,
                max_count=6) -> Scene:
    h, w = det.size
    exclude = set(exclude_subjects)
    dict_boxes = [b for b in det.boxes if b.name in entries]
    sel = select_candidates([{"name": b.name, "box": b.box, "score": b.score, "raw": b} for b in dict_boxes],
                            entries, h * w, p)
    kept, contested = sel["kept"], sel["ambiguous"]
    maybe = [b for b in dict_boxes if b.score >= p.uncertain]
    scene = Scene((h, w), occupied=[list(b.box) for b in det.boxes if b.score >= p.uncertain])
    scene.objects = [Obj(c["name"], list(c["box"]), c["score"], c["raw"].mask, c["raw"].disp) for c in kept]

    for c, obj in zip(kept, scene.objects):
        n = c["name"]
        if n in exclude:
            continue
        rivals = [o for o in kept + contested if o is not c and similar(o["name"], n, entries)]
        stray = [b for b in maybe if similar(b.name, n, entries) and not same_region(b.box, c["box"], p)]
        if not rivals and not stray:
            scene.subjects.append(obj)

    for n in dict.fromkeys(c["name"] for c in kept):
        if n in exclude:
            continue
        mine = [c for c in kept if c["name"] == n]
        others = [o for o in kept + contested if o["name"] != n and similar(o["name"], n, entries)]
        unsure = [o for o in contested if o["name"] == n]
        stray = [b for b in maybe if similar(b.name, n, entries)
                 and not any(same_region(b.box, c["box"], p) for c in mine)]
        if not others and not unsure and not stray and len(mine) <= max_count:
            scene.counts[n] = len(mine)

    for n in det.negatives:
        if det.presence.get(n, 1.0) < absent_presence and max((b.score for b in det.by_name(n)), default=0) < p.uncertain:
            scene.absent.append(n)
    return scene


# ---------------------------------------------------------------------------------------------------------------------
# Relations: EmbSpatial (Q23), BLINK Spatial Relation (Q15), MSMU Relative Position (Q34)
# ---------------------------------------------------------------------------------------------------------------------
EMB_RELATION = {"left": ["left of", "on the left side of", "at the left side of"],
                "right": ["right of", "on the right side of", "at the right side of"],
                "above": ["above"],
                "under": ["under", "beneath"]}
# EmbSpatial's other distractors. "outside" / "out of" are left out: for two separate objects they are arguably true.
EMB_OTHER = ["blocking", "inside", "connected to", "touching"]
EMB_RELATION_Q = [
    "What is the spatial relationship between {a} and {b} in the image?",
    "How are {a} and {b} positioned in relation to each other in the image?",
    "In the image, how do the positions of {a} and {b} interact with each other?",
    "What is the spatial arrangement of {a} and {b} in the image concerning each other?",
    "What is the spatial configuration between {a} and {b} in relation to each other within the image?",
]
BLINK_RELATION = {"left": ["to the left of"], "right": ["to the right of"], "above": ["above"],
                  "under": ["below", "under"]}


def _pairs(scene: Scene, cfg: dict, rng: np.random.Generator, want=None):
    """(a, b, relations of a to b) for ordered subject pairs with at least one relation (in `want` if given), in
    random order."""
    gap = cfg.get("gap_frac", 0.02) * scene.size[1]
    out = []
    for a in scene.subjects:
        for b in scene.subjects:
            if a is not b:
                rel = relations(a.box, b.box, gap)
                if want is not None:
                    rel &= set(want)
                if rel:
                    out.append((a, b, rel))
    return [out[i] for i in rng.permutation(len(out))]


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def relation_embspatial(scene, rng, cfg):
    items = []
    for a, b, rel in _pairs(scene, cfg, rng)[:cfg.get("per_frame", 1)]:
        r = _pick(rng, sorted(rel))
        wrong = [d for d in RELATIONS if d not in rel]  # a relation that also holds is never a distractor
        n_dir = min(len(wrong), int(rng.integers(1, 3)))
        picks = [wrong[i] for i in rng.permutation(len(wrong))[:n_dir]]
        picks += [EMB_OTHER[i] for i in rng.permutation(len(EMB_OTHER))[:3 - n_dir]]
        phrase = lambda d: _pick(rng, EMB_RELATION[d]) if d in EMB_RELATION else d
        sentence = lambda d: f"The {a.name} is {phrase(d)} the {b.name}."
        options, letter = F.shuffle(sentence(r), [sentence(d) for d in picks], rng)
        q = _pick(rng, EMB_RELATION_Q).format(a=a.name, b=b.name)
        items.append(Item("relation_embspatial", "embspatial", F.embspatial(q, options), letter))
    return items


def relation_blink(scene, rng, cfg):
    items = []
    for a, b, rel in _pairs(scene, cfg, rng)[:cfg.get("per_frame", 1)]:
        truth = bool(rng.random() < 0.5)
        r = _pick(rng, sorted(rel))
        asked = r if truth else OPPOSITE[r]  # the opposite of a holding relation cannot hold
        q = f"Is the {a.name} {_pick(rng, BLINK_RELATION[asked])} the {b.name}?"
        items.append(Item("relation_blink", "blink", F.blink(q, ["yes", "no"]), "A" if truth else "B"))
    return items


def relation_msmu(scene, rng, cfg):
    items = []
    for a, b, rel in _pairs(scene, cfg, rng)[:cfg.get("per_frame", 1)]:
        horizontal = sorted(rel & {"left", "right"})
        vertical = sorted(rel & {"above", "under"})
        if horizontal and (not vertical or rng.random() < 0.5):
            side = horizontal[0]
            asked = side if rng.random() < 0.5 else OPPOSITE[side]
            if rng.random() < 0.5:
                q = f"Is the {a.name} to the {asked} of the {b.name} from the viewer's perspective?"
            else:
                q = f"Does the {a.name} appear on the {asked} side of the {b.name}?"
            ans = f"{'Yes' if asked == side else 'No'}, the {a.name} is to the {side} of the {b.name}."
        else:
            high, low = (a, b) if vertical[0] == "above" else (b, a)
            x, y = (a, b) if rng.random() < 0.5 else (b, a)
            if rng.random() < 0.5:
                q, ans = f"Which stands higher in the image, the {x.name} or the {y.name}?", f"The {high.name} stands higher."
            else:
                q, ans = f"Which one is lower in the image, the {x.name} or the {y.name}?", f"The {low.name} is lower."
        items.append(Item("relation_msmu", "msmu", F.msmu(q), ans))
    return items


# ---------------------------------------------------------------------------------------------------------------------
# Closest / farthest among named objects: EmbSpatial Close / Far (Q24, Q25)
# ---------------------------------------------------------------------------------------------------------------------
CLOSE_Q = [
    "Among the listed objects, which one is closest to your current location in the image?",
    "From your viewpoint, which object among the options is the nearest within the image?",
    "Which object from the list is situated at the shortest distance from your point of view within the image?",
    "Which object, in relation to your current position, holds the nearest placement in the image?",
]
FAR_Q = [
    "Among the listed objects, which one is farthest from your current location in the image?",
    "From your viewpoint, which object among the options is the farthest within the image?",
    "Which object from the list is situated at the largest distance from your point of view within the image?",
    "Which object, in relation to your current position, holds the farthest placement in the image?",
]


def _depth_order(scene, rng, cfg, closest: bool):
    """Option sets of n_options subjects whose closest (or farthest) one beats the runner-up by cfg margin in
    normalised disparity; up to per_frame of them, sharing no answer object."""
    n = cfg.get("n_options", 4)
    cands = [s for s in scene.subjects if s.disp is not None]
    out, used = [], set()
    if len(cands) < n:
        return out
    for _ in range(cfg.get("tries", 20)):
        pick = [cands[i] for i in rng.choice(len(cands), n, replace=False)]
        ranked = sorted(pick, key=lambda s: -s.disp if closest else s.disp)  # answer first
        if abs(ranked[0].disp - ranked[1].disp) >= cfg.get("margin", 0.1) and id(ranked[0]) not in used:
            used.add(id(ranked[0]))
            out.append((ranked[0], ranked[1:]))
            if len(out) >= cfg.get("per_frame", 1):
                break
    return out


def close_embspatial(scene, rng, cfg):
    items = []
    for ans, rest in _depth_order(scene, rng, cfg, closest=True):
        options, letter = F.shuffle(ans.name, [o.name for o in rest], rng)
        items.append(Item("close_embspatial", "embspatial", F.embspatial(_pick(rng, CLOSE_Q), options), letter))
    return items


def far_embspatial(scene, rng, cfg):
    items = []
    for ans, rest in _depth_order(scene, rng, cfg, closest=False):
        options, letter = F.shuffle(ans.name, [o.name for o in rest], rng)
        items.append(Item("far_embspatial", "embspatial", F.embspatial(_pick(rng, FAR_Q), options), letter))
    return items


# ---------------------------------------------------------------------------------------------------------------------
# Pointing: ERQA "four dots, which is on X" (Q12)
# ---------------------------------------------------------------------------------------------------------------------
DOT_COLORS = {"red": (230, 30, 30), "green": (40, 180, 60), "blue": (40, 90, 235), "yellow": (250, 215, 0),
              "purple": (150, 60, 200), "cyan": (0, 200, 220)}
POINT_COLOR_Q = ["There are four points marked with colors, which one is on the {name}.",
                 "Four coloured dots are marked on the image. Which one is on the {name}?",
                 "Which of the coloured dots lies on the {name}?"]
POINT_LETTER_Q = ["There are four points marked with letters, which one is on the {name}?",
                  "Identify among points A, B, C, D, which is on the {name}?"]


def _dots(scene, target: Obj, rng, cfg):
    """[correct point, 3 distractor points] or None. The correct point is inside the eroded target mask and on no
    other kept object's mask; distractors are on other objects (eroded masks) or on the background, never inside the
    target's padded box. All points keep min_dot_dist apart and away from the image border."""
    h, w = scene.size
    r = max(1, round(cfg.get("erode_frac", 0.015) * w))
    min_dist = cfg.get("min_dot_dist", 0.08) * w
    border = cfg.get("border", 0.05)
    pad = cfg.get("box_pad", 0.02) * w
    others = [o for o in scene.objects if o is not target and o.mask is not None]
    taken = np.zeros((h, w), bool)
    for o in others:
        taken |= o.mask
    pts = []
    ok = lambda xy: inside_border(xy, (h, w), border) and far_enough(xy, pts, min_dist)
    first = sample_in_mask(erode(target.mask, r) & ~taken, rng, ok)
    if first is None:
        return None
    pts.append(first)
    away = lambda xy: ok(xy) and not in_box(xy, target.box, pad)
    for i in rng.permutation(len(others)):
        if len(pts) >= 3:  # at least one background distractor
            break
        xy = sample_in_mask(erode(others[i].mask, r), rng, away)
        if xy is not None:
            pts.append(xy)
    blocked = [[target.box[0] - pad, target.box[1] - pad, target.box[2] + pad, target.box[3] + pad]]
    while len(pts) < 4:
        xy = sample_background((h, w), blocked, rng, away)
        if xy is None:
            return None
        pts.append(xy)
    return pts


def pointing_erqa(scene, rng, cfg):
    items = []
    targets = [s for s in scene.subjects if s.mask is not None]
    for i in rng.permutation(len(targets))[:cfg.get("per_frame", 1)]:
        t = targets[i]
        pts = _dots(scene, t, rng, cfg)
        if pts is None:
            continue
        if rng.random() < cfg.get("p_letters", 0.3):
            labels = [F.LETTERS[j] for j in rng.permutation(4)]
            red = DOT_COLORS["red"]
            overlays = [{"kind": "dot", "xy": list(p), "color": list(red), "label": lab} for p, lab in zip(pts, labels)]
            q = F.erqa(_pick(rng, POINT_LETTER_Q).format(name=t.name), list("ABCD"))
            items.append(Item("pointing_erqa", "erqa", q, labels[0], overlays))
        else:
            names = list(DOT_COLORS)
            colors = [names[j] for j in rng.permutation(len(names))[:4]]
            overlays = [{"kind": "dot", "xy": list(p), "color": list(DOT_COLORS[c]), "label": None}
                        for p, c in zip(pts, colors)]
            options, letter = F.shuffle(f"{colors[0]} dot", [f"{c} dot" for c in colors[1:]], rng)
            q = F.erqa(_pick(rng, POINT_COLOR_Q).format(name=t.name), options)
            items.append(Item("pointing_erqa", "erqa", q, letter, overlays))
    return items


# ---------------------------------------------------------------------------------------------------------------------
# BLINK Object Localization: "which box encloses X" (Q16)
# ---------------------------------------------------------------------------------------------------------------------
BOX_COLORS = [(230, 30, 30), (40, 180, 60), (40, 90, 235), (250, 215, 0), (150, 60, 200)]


def localization_blink(scene, rng, cfg):
    items = []
    h, w = scene.size
    targets = [s for s in scene.subjects
               if (s.box[2] - s.box[0]) * (s.box[3] - s.box[1]) >= cfg.get("min_area", 0.003) * h * w]
    for i in rng.permutation(len(targets))[:cfg.get("per_frame", 1)]:
        t = targets[i]
        fake = perturbed_box(t.box, (h, w), rng, tuple(cfg.get("iou", (0.3, 0.65))))
        if fake is None:
            continue
        correct = "A" if rng.random() < 0.5 else "B"
        boxes = {correct: t.box, "B" if correct == "A" else "A": fake}
        c = rng.choice(len(BOX_COLORS), 2, replace=False)
        overlays = [{"kind": "box", "xyxy": list(boxes[lab]), "color": list(BOX_COLORS[c[k]]), "label": lab}
                    for k, lab in enumerate("AB")]
        q = F.blink(F.BLINK_LOCALIZATION.format(name=t.name), ["Box A", "Box B"], choices_line=False)
        items.append(Item("localization_blink", "blink", q, correct, overlays))
    return items


# ---------------------------------------------------------------------------------------------------------------------
# MSMU existence (Q31): questions about an absent object, answered "No X in the scene."
# ---------------------------------------------------------------------------------------------------------------------
ABSENT_Q = ["How tall is {n}?", "What is the height of {n}?", "Where is {n}?", "What is the size of {n}?",
            "what's the total number of {pl} in the image?"]
ABSENT_A = ["No {n} in the scene.", "Can not find {n}."]


def region(box, size) -> str:
    """'the top left part' .. 'the center' of the image, from the box centre in a 3x3 grid."""
    h, w = size
    cx, cy = (box[0] + box[2]) / 2 / w, (box[1] + box[3]) / 2 / h
    col = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else ""
    row = "top" if cy < 1 / 3 else "bottom" if cy > 2 / 3 else ""
    where = " ".join(filter(None, (row, col)))
    return f"the {where} part" if where else "the center"


def existence_msmu(scene, rng, cfg):
    """Mostly absent objects; with p_present a "Where is X?" about a present subject, so the question shape alone
    does not give the answer away."""
    items = []
    for _ in range(cfg.get("per_frame", 1)):
        if scene.subjects and rng.random() < cfg.get("p_present", 0.3):
            s = _pick(rng, scene.subjects)
            q, ans = f"Where is the {s.name}?", f"The {s.name} is in {region(s.box, scene.size)} of the image."
        elif scene.absent:
            n = _pick(rng, scene.absent)
            q = _pick(rng, ABSENT_Q).format(n=n, pl=plural(n))
            ans = _pick(rng, ABSENT_A).format(n=n)
        else:
            continue
        items.append(Item("existence_msmu", "msmu", F.msmu(q), ans))
    return items


# ---------------------------------------------------------------------------------------------------------------------
# Counting: BLINK (Q13, 4 numeric options) and MSMU (Q32, free-form number)
# ---------------------------------------------------------------------------------------------------------------------
def _count_names(scene, rng, cfg):
    names = sorted(scene.counts)
    return [names[i] for i in rng.permutation(len(names))[:cfg.get("per_frame", 1)]]


def counting_blink(scene, rng, cfg):
    items = []
    for n in _count_names(scene, rng, cfg):
        c = scene.counts[n]
        pool = [k for k in range(max(0, c - 3), c + 4) if k != c]
        distractors = [str(pool[i]) for i in rng.choice(len(pool), 3, replace=False)]
        options, letter = F.shuffle(str(c), distractors, rng)
        q = _pick(rng, ["How many {pl} are in the image?", "How many {pl} are there in the image?"]).format(pl=plural(n))
        items.append(Item("counting_blink", "blink", F.blink(q, options), letter))
    return items


def counting_msmu(scene, rng, cfg):
    items = []
    for n in _count_names(scene, rng, cfg):
        c, pl = scene.counts[n], plural(n)
        q = _pick(rng, ["How many {pl} are there in the image ?", "what's the total number of {pl} in the image?"])
        ans = _pick(rng, [f"There is only 1 {n}.", "1."] if c == 1 else [f"There are {c} {pl}.", f"{c}."])
        items.append(Item("counting_msmu", "msmu", F.msmu(q.format(pl=pl), numeric=True), ans))
    return items


TEMPLATES = {f.__name__: f for f in (relation_embspatial, relation_blink, relation_msmu, close_embspatial,
                                     far_embspatial, pointing_erqa, localization_blink, existence_msmu,
                                     counting_blink, counting_msmu)}


def make_items(scene: Scene, rng: np.random.Generator, templates: dict) -> list[Item]:
    """Questions of every enabled template (config order), at most per_frame each."""
    items = []
    for name, cfg in templates.items():
        if name not in TEMPLATES:
            raise ValueError(f"unknown scene template {name!r} (known: {', '.join(TEMPLATES)})")
        cfg = cfg or {}
        if cfg.get("enabled", True) and cfg.get("per_frame", 1) > 0:
            items += TEMPLATES[name](scene, rng, cfg)[:cfg.get("per_frame", 1)]
    return items
