"""Phase questions for one episode with a gripper.

Implemented:
  phase -- 'The robot is tasked to "<task>". What is the robot's current action phase?'  (A-D)

One question per phase run, on a random frame inside the run.
Input: instruction and per-frame phase labels (aijgen.phases.label).
Output: QA items that reference frames by index; the caller saves the images.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from aijgen import phases as ph

LETTERS = "ABCD"
MCQ_SUFFIX = "Answer with a single letter: A, B, C, or D."

PHASE_TEXT = {
    ph.PRE_GRASP: "Approaching the object with the gripper open",
    ph.IMMOBILIZATION: "Closing the gripper around the object",
    ph.CONTACT: "Holding the object with the gripper closed",
    ph.DETACH: "Releasing the object by opening the gripper",
    ph.POST_GRASP: "Moving away after releasing the object",
}

