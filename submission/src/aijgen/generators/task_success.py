"""ERQA task-success questions with hard negatives (docs/sam3_vqa_plan.md, P4: Q08).

  "Was the task successful: put the spoon on the towel Choices: A. No. B. Yes. ..."

Per eligible episode, one Yes and one No, on the same view:
  Yes    the real instruction on a "done" frame: random in the last done_window (5%) of the episode, and only a
         frame where the outcome is visible (place: gripper open, arm >= release_min_distance from where it let go;
         pick: still holding, lifted);
  No     with p_swap, the same done frame with the instruction's object swapped for another object W that never
         moved (hard negative); otherwise the real instruction on an early frame (first half of the first pre_grasp
         run, before anything is touched).

Conservative on purpose: gripper phases and arm motion cannot place every moment of an episode, so an episode is used
only when nothing hints at a failure (clean_episode), and SAM 3 must confirm the outcome on the image:
  - the instruction parses as pick / place with one dictionary object each (aijgen.scene.instructions);
  - phases exactly pre_grasp -> closing -> contact -> opening -> post_grasp (place) or pre_grasp -> closing ->
    contact (pick): no re-grasp, no aborted grasp, no second closing;
  - contact >= min_contact_s, arm carried >= min_carry, arm went down >= min_descent before the grasp, length <=
    max_length (long episodes are often struggles); place: >= min_post_release_s after the release; pick: the last
    frame >= min_lift above the grasp; Bridge: the gripper did not close fully (full_close_raw: closed on nothing);
  - Yes needs, on the done frame, X (and Y for place) as build_scene subjects with X on Y (inter / area(X) >=
    on_target), and on the early frame X not yet on Y; pick: X higher in the image than at the start;
  - W is a subject on both frames, not similar to X or Y, its box unchanged (IoU >= swap_min_iou) and not on Y. An
    episode without a valid W is skipped even when the early No is drawn, so Yes and No come from the same episodes.
Skipped episodes are counted by reason in self.skips (run.py adds them to the stats as "task_success skip <reason>").
"""
import math
import zlib
from collections import Counter

import numpy as np

from aijgen import phases as ph
from aijgen.depth import lerobot
from aijgen.generators.base import Question, pick_frame
from aijgen.generators.scene_qa import SceneQAGenerator
from aijgen.scene import formats as F
from aijgen.scene.detect import area, overlaps
from aijgen.scene.instructions import Task, parse_task
from aijgen.scene.templates import Obj, Scene
from aijgen.scene.vocab import similar

QUESTIONS = [  # ERQA's task-success wordings; {gerund} only where the instruction has a clean -ing form
    "Was the task successful: {task}",
    "Is this task successful - {task}",
    "Has this task been successfully completed? {Task}",
    "Did the robot complete the task: {task}?",
    "Is the robot successful in {gerund}?",
]
PLACE_PHASES = [ph.PRE_GRASP, ph.IMMOBILIZATION, ph.CONTACT, ph.DETACH, ph.POST_GRASP]
PICK_PHASES = [ph.PRE_GRASP, ph.IMMOBILIZATION, ph.CONTACT]
CLEAN = {"min_contact_s": 1.0, "min_carry": 0.05, "min_descent": 0.02, "min_post_release_s": 0.6, "min_lift": 0.05}
FRAMES = {"done_window": 0.05, "min_done_frames": 2, "release_min_distance": 0.05}
CHECKS = {"on_target": 0.5, "start_on_target": 0.2, "min_lift_px": 0.03, "swap_min_iou": 0.6,
          "swap_max_on_target": 0.1}


# ---------------------------------------------------------------------------------------------------------------------
# Episode filter and frames (proprio only)
# ---------------------------------------------------------------------------------------------------------------------
def clean_episode(ep, kind: str, dcfg: dict, c: dict = CLEAN) -> tuple[dict | None, str]:
    """({"pre": (s, e), "contact": (s, e), "fps": fps}, "ok") for an episode with no sign of failure, else
    (None, reason)."""
    if ep.labels is None or ep.closedness is None:
        return None, "no_gripper_signal"
    if ep.length > dcfg.get("max_length", 10 ** 9):
        return None, "too_long"
    fps = float(dcfg.get("fps") or ep.fps or 10.0)
    runs = ph.runs(ep.labels)
    seq = [p for p, _, _ in runs]
    contacts = [(s, e) for p, s, e in runs if p == ph.CONTACT]
    if len(contacts) != 1:
        return None, "no_grasp" if not contacts else "regrasp"
    if seq != (PLACE_PHASES if kind == "place" else PICK_PHASES):
        return None, "irregular_phases"  # aborted grasp, second closing, not released / released, cut mid-action
    pre = (runs[0][1], runs[0][2])
    cs, ce = contacts[0]
    xyz = ep.state[:, :3]
    if (ce - cs + 1) / fps < c["min_contact_s"]:
        return None, "short_contact"
    if np.linalg.norm(xyz[cs:ce + 1] - xyz[cs], axis=1).max() < c["min_carry"]:
        return None, "not_carried"
    if xyz[0, 2] - xyz[cs, 2] < c["min_descent"]:
        return None, "no_descent"
    thr = dcfg.get("full_close_raw")
    if thr is not None:
        raw = float(np.median(ep.state[cs:ce + 1, dcfg["gripper_dim"]]))
        if (raw >= thr) if dcfg.get("closed_high", False) else (raw <= thr):
            return None, "full_close"
    if kind == "place" and (ep.length - 1 - ce) / fps < c["min_post_release_s"]:
        return None, "release_at_end"
    if kind == "pick" and xyz[-1, 2] - xyz[cs, 2] < c["min_lift"]:
        return None, "not_lifted"
    return {"pre": pre, "contact": (cs, ce), "fps": fps}, "ok"


def done_frames(ep, kind: str, info: dict, f: dict = FRAMES, c: dict = CLEAN) -> list[int]:
    """Frames of the last done_window of the episode (at least min_done_frames) where the outcome is visible."""
    n = max(int(f["min_done_frames"]), math.ceil(f["done_window"] * ep.length))
    cs, ce = info["contact"]
    xyz = ep.state[:, :3]
    out = []
    for t in range(max(0, ep.length - n), ep.length):
        if kind == "place":
            ok = (t > ce and ep.closedness[t] <= ph.GRASP_THRESHOLD
                  and np.linalg.norm(xyz[t] - xyz[ce]) >= f["release_min_distance"])
        else:
            ok = ep.labels[t] == ph.CONTACT and xyz[t, 2] - xyz[cs, 2] >= c["min_lift"]
        if ok:
            out.append(t)
    return out


def early_frame(info: dict, rng: np.random.Generator) -> int:
    """A frame of the first half of the first pre_grasp run: the arm is still on its way, nothing touched."""
    s, e = info["pre"]
    return pick_frame(s, e, rng, (0.0, 0.5))


# ---------------------------------------------------------------------------------------------------------------------
# Image checks (SAM 3 scenes)
# ---------------------------------------------------------------------------------------------------------------------
def find(scene: Scene, name: str, entries) -> Obj | None:
    """The subject of that name or of its dictionary line (subjects are the only box of their kind)."""
    return next((s for s in scene.subjects if s.name == name or entries[s.name].group == entries[name].group), None)


def on(a, b) -> float:
    """Share of box a inside box b."""
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return inter / area(a) if area(a) > 0 else 0.0


def yes_confirmed(task: Task, end: Scene, start: Scene, entries, k: dict = CHECKS) -> str:
    """"ok" if the images show the instruction done on the done frame and not yet done on the early frame, else the
    reason."""
    x1, x0 = find(end, task.obj, entries), find(start, task.obj, entries)
    if x1 is None or x0 is None:
        return "object_not_seen"
    if task.kind == "pick":
        lift = x0.box[3] - x1.box[3]  # image y grows downwards
        return "ok" if lift >= k["min_lift_px"] * end.size[0] else "not_lifted_on_image"
    y1, y0 = find(end, task.target, entries), find(start, task.target, entries)
    if y1 is None or y0 is None:
        return "target_not_seen"
    if on(x1.box, y1.box) < k["on_target"]:
        return "not_on_target_at_end"
    if on(x0.box, y0.box) >= k["start_on_target"]:
        return "on_target_at_start"
    return "ok"


def swap_candidates(task: Task, end: Scene, start: Scene, entries, k: dict = CHECKS) -> list[str]:
    """Names W for which "<instruction with W>" is surely not done on the done frame (module docstring)."""
    y1 = find(end, task.target, entries) if task.target else None
    out = []
    for w in end.subjects:
        if similar(w.name, task.obj, entries) or (task.target and similar(w.name, task.target, entries)):
            continue
        w0 = find(start, w.name, entries)
        if w0 is None or overlaps(w0.box, w.box)[0] < k["swap_min_iou"]:
            continue
        if y1 is not None and on(w.box, y1.box) >= k["swap_max_on_target"]:
            continue
        out.append(w.name)
    return sorted(set(out))


def question(task_text: str, gerund: str | None, answer_yes: bool, rng: np.random.Generator) -> tuple[str, str]:
    """(ERQA prompt, correct letter) with Yes / No in random order."""
    forms = QUESTIONS if gerund else QUESTIONS[:-1]
    q = forms[int(rng.integers(len(forms)))].format(task=task_text, Task=task_text[:1].upper() + task_text[1:],
                                                    gerund=gerund)
    options, letter = F.shuffle("Yes" if answer_yes else "No", ["No" if answer_yes else "Yes"], rng)
    return F.erqa(q, options), letter


# ---------------------------------------------------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------------------------------------------------
class TaskSuccessGenerator(SceneQAGenerator):
    name = "task_success"

    def __init__(self, cfg: dict):
        super().__init__({**cfg, "templates": {}, "crossview": {}})
        self.clean = {**CLEAN, **cfg.get("clean", {})}
        self.frames_cfg = {**FRAMES, **cfg.get("frames", {})}
        self.checks = {**CHECKS, **cfg.get("checks", {})}
        self.p_swap = float(cfg.get("p_swap", 0.7))
        self.skips = Counter()

    def sampled(self, ep, dcfg) -> bool:
        frac = float(dcfg.get("sample_fraction", 1.0))
        return zlib.crc32(f"{self.seed}:{ep.dataset}:{ep.index}".encode()) % 10000 < frac * 10000

    def skip(self, reason: str) -> list:
        self.skips[reason] += 1
        return []

    def generate(self, ep, rng: np.random.Generator, videos: dict | None = None) -> list[Question]:
        dcfg = self.datasets.get(ep.dataset)
        if dcfg is None or not videos or not self.sampled(ep, dcfg):
            return []
        entries, absent = self.vocab(ep.dataset)
        task = parse_task(ep.task, entries)
        if task is None:
            return self.skip("instruction_not_parsed")
        info, reason = clean_episode(ep, task.kind, dcfg, self.clean)
        if info is None:
            return self.skip(reason)
        done = done_frames(ep, task.kind, info, self.frames_cfg, self.clean)
        if not done:
            return self.skip("no_done_frame")
        fd, fe = done[int(rng.integers(len(done)))], early_frame(info, rng)

        views = [v for v in dcfg["views"] if v in videos]
        frames = {v: lerobot.read_frames(videos[v], [fe, fd]) for v in views}
        valid = [v for v in views if all(f in frames[v] and frames[v][f].std() >= self.min_std for f in (fe, fd))]
        if not valid:
            return self.skip("blank_views")
        cam = valid[ep.index % len(valid)]
        end = self.scene(self.detections(ep, fd, cam, valid, frames, entries, absent), entries)
        start = self.scene(self.detections(ep, fe, cam, valid, frames, entries, absent), entries)
        verdict = yes_confirmed(task, end, start, entries, self.checks)
        if verdict != "ok":
            return self.skip(verdict)
        swaps = swap_candidates(task, end, start, entries, self.checks)
        if not swaps:
            return self.skip("no_swap_object")
        self.skips["used"] += 1

        meta = {"camera": cam, "bench": "erqa", "template": "success_erqa", "task_kind": task.kind}
        text, letter = question(task.text, task.gerund(), True, rng)
        out = [Question(text, letter, [fd], self.name, {**meta, "answer_kind": "yes"})]
        if rng.random() < self.p_swap:
            w = swaps[int(rng.integers(len(swaps)))]
            swapped = task.swapped(w)
            text, letter = question(swapped, task.gerund(swapped), False, rng)
            out.append(Question(text, letter, [fd], self.name, {**meta, "answer_kind": "swap", "_swap": w}))
        else:
            text, letter = question(task.text, task.gerund(), False, rng)
            out.append(Question(text, letter, [fe], self.name, {**meta, "answer_kind": "early"}))
        return out
