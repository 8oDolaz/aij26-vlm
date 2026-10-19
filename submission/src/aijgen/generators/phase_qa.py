"""Q1: which action phase is the robot in. One question per phase run."""
import numpy as np

from aijgen import phases as ph
from aijgen.generators.base import Generator, Question, pick_frame

# answer option text for each of the 5 phases of aijgen.phases
PHASE_TEXT = {
    "pre_grasp":      "Approaching the object with the gripper open",
    "immobilization": "Closing the gripper around the object",
    "contact":        "Holding the object in the closed gripper",
    "detach":         "Opening the gripper to release the object",
    "post_grasp":     "Moving away after releasing the object",
}

# merge_phases: true -> the short transitions join their neighbours, 5 phases -> 3
MERGE = {
    "pre_grasp":      "pre_grasp",
    "immobilization": "pre_grasp",
    "contact":        "contact",
    "detach":         "post_grasp",
    "post_grasp":     "post_grasp",
}
MERGED_TEXT = {
    "pre_grasp":  "Approaching the object and closing the gripper on it",
    "contact":    "Holding the object in the closed gripper",
    "post_grasp": "Releasing the object and moving away",
}

QUESTION = 'The robot is tasked to "{task}". What is the robot\'s current action phase?'


class PhaseGenerator(Generator):
    name = "phase_qa"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.frame_range = tuple(cfg.get("frame_range", [0.1, 0.9]))
        self.min_phase_seconds = float(cfg.get("min_phase_seconds", 0.5))
        self.release_min_distance = float(cfg.get("release_min_distance", 0.05))

        if cfg.get("merge_phases", True):
            self.names = ("pre_grasp", "contact", "post_grasp")
            self.text = MERGED_TEXT
            to_name = MERGE
        else:
            self.names = ph.PHASES
            self.text = PHASE_TEXT
            to_name = {p: p for p in ph.PHASES}
        # phase index of aijgen.phases -> index in self.names; merged: [0, 0, 1, 2, 2]
        self.to_index = np.array([self.names.index(to_name[p]) for p in ph.PHASES])

        self.labeled_phases = set(cfg.get("labeled_phases", ["pre_grasp", "contact", "post_grasp"]))
        self.phase_frame_range = {p: tuple(r) for p, r in cfg.get("phase_frame_range", {}).items()}
        unknown = (self.labeled_phases | set(self.phase_frame_range)) - set(self.names)
        if unknown:
            raise ValueError(f"Unknown phases in config: {unknown} (allowed: {self.names})")

    def generate(self, ep, rng):
        # no gripper signal, or no real grasp
        if ep.labels is None or not ph.has_phases(ep.labels, {ph.IMMOBILIZATION, ph.CONTACT}):
            return []

        labels = self.to_index[ep.labels]          # per-frame index in self.names
        question = QUESTION.format(task=ep.task)
        out = []
        for p, start, end in ph.runs(labels):
            phase = self.names[p]
            if phase not in self.labeled_phases:
                continue

            fps = ep.fps if ep.fps > 0 else 10.0
            duration = (end - start + 1) / fps          # seconds
            if duration < self.min_phase_seconds:
                continue
            
            extra = {}
            if phase == "post_grasp":
                # releasing is visible only once the gripper is open and the arm has moved off the object
                picked = self.pick_release_frame(ep, start, end, rng)
                if picked is None:
                    continue
                frame, moved = picked
                extra = {"moved_m": round(moved, 3)}
            else:
                frame_range = self.phase_frame_range.get(phase, self.frame_range)
                frame = pick_frame(start, end, rng, frame_range)

            correct = self.text[phase]
            others = [self.text[name] for name in self.names if name != phase]
            k = min(3, len(others))
            distractors = [others[i] for i in rng.choice(len(others), size=k, replace=False)]

            text, letter = self.mcq(question, correct, distractors, rng)
            out.append(Question(text, letter, [frame], self.name, {"phase": phase, **extra}))
        return out

    def pick_release_frame(self, ep, start, end, rng):
        """A frame of the releasing run where the gripper is open and the arm has moved away from
        where it let go, so the release is visible on the image. (frame, distance in m) or None."""
        held = np.flatnonzero(ep.labels[:start] == ph.CONTACT)
        if len(held) == 0:
            return None
        release_pos = ep.state[held[-1], 0:3]           # arm xyz at the last holding frame
        candidates = []
        for t in range(start, end + 1):
            moved = float(np.linalg.norm(ep.state[t, 0:3] - release_pos))
            if ep.closedness is not None and ep.closedness[t] <= ph.GRASP_THRESHOLD and moved >= self.release_min_distance:
                candidates.append((t, moved))
        if not candidates:
            return None
        return candidates[int(rng.integers(len(candidates)))]
