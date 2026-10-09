"""Shared pieces for all question generators.

A generator turns one Episode into a list of QA items. It never touches files or video:
QA.frames are frame indices, and the caller saves the images.
"""
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

LETTERS = "ABCD"

MCQ_SUFFIX = ("Answer with the single letter corresponding to the correct option. "
              "Do not explain. Your answer must be one of A, B, C, or D.")


@dataclass
class Question:
    question: str
    answer: str
    frames: list[int]                         # frame indices in the episode, one image each, in order
    kind: str                                 # generator name, for statistics
    meta: dict = field(default_factory=dict)  # debug info, not written to the JSONL


@dataclass
class Episode:
    dataset: str                              # "bridge", "fractal", ...
    index: int                                # episode_index in the dataset
    task: str                                 # instruction
    length: int                               # number of frames
    state: np.ndarray                         # (length, 7)
    closedness: np.ndarray | None = None      # (length,) 0 open .. 1 closed; None = no gripper signal
    labels: np.ndarray | None = None          # (length,) phase index per frame (aijgen.phases)


def stable_id(s: str) -> int:
    """Deterministic integer for a string (seeds for datasets)"""
    return zlib.crc32(s.encode("utf-8"))


class Generator(ABC):
    """Base class. Subclasses set `name` and implement `generate`."""

    name: str = ""                            

    def __init__(self, cfg: dict, seed: int):
        self.cfg = cfg 
        self.seed = int(cfg["seed"])                       

    def rng(self, ep: Episode) -> np.random.Generator:
        """Own random stream per (seed, dataset, episode, generator)"""
        return np.random.default_rng([self.seed, stable_id(ep.dataset), ep.index, stable_id(self.name)])

    @abstractmethod
    def generate(self, ep: Episode, rng: np.random.Generator) -> list[Question]:
        """Questions for one episode."""

    def __call__(self, ep: Episode) -> list[Question]:
        return self.generate(ep, self.rng(ep))

    # helpers for subclasses

    @staticmethod
    def mcq(question: str, correct: str, distractors: list[str],
            rng: np.random.Generator) -> tuple[str, str]:
        """Shuffle options, return (question with lettered options + MCQ_SUFFIX, correct letter)."""
        options = [correct, *distractors]
        options = [options[i] for i in rng.permutation(len(options))]
        letter = LETTERS[options.index(correct)]
        lines = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
        return f"{question}\n{lines}\n{MCQ_SUFFIX}", letter