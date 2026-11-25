"""Grasp phases from a gripper signal, after Robo2VLM's determine_grasp_phases (example/robo2vlm/generation/utils.py).

Each episode's signal is rescaled to 0 (open) .. 1 (closed) over its own range, so a gripper that stops on a thick
object still reaches "contact". Then a state machine with Robo2VLM's thresholds labels every frame:

  pre_grasp -> immobilization (closing) -> contact (closed) -> detach (opening) -> post_grasp -> (next grasp)

Differences from Robo2VLM:
- an episode whose gripper moves less than min_range of the dataset's range gets no phases. Rescaling sensor noise
  to 0..1 invents grasps; that is ~10% of Bridge episodes;
- a brief dip while in contact stays "contact" (Robo2VLM tags it as a one-frame "transition").
"""
import numpy as np

PHASES = ("pre_grasp", "immobilization", "contact", "detach", "post_grasp")
PRE_GRASP, IMMOBILIZATION, CONTACT, DETACH, POST_GRASP = range(len(PHASES))
GRASP_THRESHOLD = 0.2    # above this the gripper is no longer open (Robo2VLM BaseTrajectory default)
CONTACT_THRESHOLD = 0.9  # at or above this it is closed
MIN_RANGE = 0.2          # minimum gripper travel, as a fraction of the dataset's p1..p99 range


def closedness(g: np.ndarray, closed_high: bool, scale: float, min_range: float = MIN_RANGE) -> np.ndarray | None:
    """Raw gripper values of one episode -> 0 (open) .. 1 (closed), or None if it barely moves.

    closed_high: whether larger raw values mean more closed. scale: the dataset's p1..p99 gripper range.
    """
    g = np.asarray(g, dtype=np.float64)
    lo, hi = g.min(), g.max()
    if hi - lo < max(min_range * scale, 1e-6):
        return None
    c = (g - lo) / (hi - lo)
    return c if closed_high else 1.0 - c


def label(c: np.ndarray, grasp: float = GRASP_THRESHOLD, contact: float = CONTACT_THRESHOLD) -> np.ndarray:
    """Per-frame phase index for a closedness signal (0 open .. 1 closed)."""
    out = np.empty(len(c), dtype=np.int8)
    cur, prev = PRE_GRASP, c[0]
    for i, x in enumerate(c.tolist()):
        closing, opening = x > prev, x < prev
        if cur in (PRE_GRASP, POST_GRASP) and x > grasp and closing:
            cur = IMMOBILIZATION
        elif cur == IMMOBILIZATION and x >= contact:
            cur = CONTACT
        elif cur == IMMOBILIZATION and x <= grasp and opening:
            cur = PRE_GRASP  # grasp aborted
        elif cur == CONTACT and x < contact and opening:
            cur = DETACH
        elif cur == DETACH and x <= grasp:
            cur = POST_GRASP
        elif cur == DETACH and x >= contact and closing:
            cur = CONTACT  # re-grasp
        out[i] = cur
        prev = x
    return out


def runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Run-length phases: [(phase, first_frame, last_frame), ...]."""
    cut = np.flatnonzero(np.diff(labels)) + 1
    starts, ends = np.r_[0, cut], np.r_[cut, len(labels)] - 1
    return [(int(labels[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def has_phases(labels: np.ndarray, required: set[int], min_frames: int = 1) -> bool:
    """Whether every required phase occurs as a run of at least min_frames."""
    return required <= {p for p, s, e in runs(labels) if e - s + 1 >= min_frames}
