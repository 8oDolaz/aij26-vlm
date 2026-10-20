"""Shared pieces for all question generators.

A generator turns one Episode into a list of Question items. It never touches files or video:
Question.frames are frame indices, and the caller saves the images.
"""
import inspect
import math
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

LETTERS = "ABCD"


def mcq_suffix(n: int) -> str:
    """ERQA-style instruction for n options: n=4 -> "... one of A, B, C, or D." """
    letters = LETTERS[:n]
    choices = letters[0] if n == 1 else ", ".join(letters[:-1]) + (", or " if n > 2 else " or ") + letters[-1]
    return ("Answer with the single letter corresponding to the correct option. "
            f"Do not explain. Your answer must be one of {choices}.")


@dataclass
class Question:
    question: str
    answer: str
    frames: list[int]                         # frame indices in the episode, one image each, in order
    kind: str                                 # generator name, for statistics
    meta: dict = field(default_factory=dict)  # debug info, not written to the JSONL; "camera" / "cameras" (one per
                                              # image) choose the views (generators.run.image_cameras)
    overlays: list[dict] = field(default_factory=list)  # markers (aijgen.scene.render); "image": k = on image k only


@dataclass
class Episode:
    dataset: str                              # "bridge", "fractal", ...
    index: int                                # episode_index in the dataset
    task: str                                 # instruction
    length: int                               # number of frames
    state: np.ndarray                         # (length, 7)
    fps: float = 0.0                          # frames per second
    closedness: np.ndarray | None = None      # (length,) 0 open .. 1 closed; None = no gripper signal
    labels: np.ndarray | None = None          # (length,) phase index per frame (aijgen.phases)


def stable_id(s: str) -> int:
    """Deterministic integer for a string (seeds for datasets)"""
    return zlib.crc32(s.encode("utf-8"))


def pick_frame(start: int, 
               end: int, 
               rng: np.random.Generator,
               frame_range: tuple[float, float]) -> int:
    """Random frame in the run [start, end] (end inclusive), restricted to frame_range fractions."""
    span = end - start
    low_bound = start + math.ceil(frame_range[0] * span)
    hight_bound = start + math.floor(frame_range[1] * span)
    if low_bound > hight_bound:                                 
        return (start + end) // 2
    return int(rng.integers(low_bound, hight_bound + 1))


class Generator(ABC):
    """Base class. Subclasses set `name` and implement `generate`."""

    name: str = ""                            

    def __init__(self, cfg: dict):
        self.cfg = cfg 
        self.seed = int(cfg["seed"])                       

    def rng(self, ep: Episode) -> np.random.Generator:
        """Own random stream per (seed, dataset, episode, generator)"""
        return np.random.default_rng([self.seed, stable_id(ep.dataset), ep.index, stable_id(self.name)])

    @abstractmethod
    def generate(self, ep: Episode, rng: np.random.Generator, **kwargs) -> list[Question]:
        """Questions for one episode."""

    def __call__(self, ep: Episode, **kwargs) -> list[Question]:
        sig = inspect.signature(self.generate)
        if "videos" in sig.parameters and "videos" in kwargs:
            return self.generate(ep, self.rng(ep), videos=kwargs["videos"])
        return self.generate(ep, self.rng(ep))

    @staticmethod
    def mcq(question: str, 
            correct: str,
            distractors: list[str],
            rng: np.random.Generator) -> tuple[str, str]:
        """Shuffle options, return (question with lettered options + instructions, correct letter)."""
        options = [correct, *distractors]
        options = [options[i] for i in rng.permutation(len(options))]
        letter = LETTERS[options.index(correct)]
        lines = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
        return f"{question}\n{lines}\n{mcq_suffix(len(options))}", letter
