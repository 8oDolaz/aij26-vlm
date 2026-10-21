"""Tests for submission/src/aijgen/phases.py. Run: python -m unittest discover -s tests"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission" / "src"))
from aijgen import phases as ph  # noqa: E402

PRE, IMM, CON, DET, POST = range(5)


class TestClosedness(unittest.TestCase):
    def test_rescales_and_flips(self):
        # Bridge-style: 1 = open, low = closed.
        c = ph.closedness(np.array([1.0, 1.0, 0.2, 0.2, 1.0]), closed_high=False, scale=1.0)
        np.testing.assert_allclose(c, [0, 0, 1, 1, 0])

    def test_barely_moving_gripper_has_no_phases(self):
        self.assertIsNone(ph.closedness(np.array([1.0, 0.999, 1.0]), closed_high=False, scale=1.0))


class TestLabel(unittest.TestCase):
    def test_full_cycle(self):
        c = np.array([0, 0, 0.5, 1, 1, 0.5, 0, 0])
        self.assertEqual(ph.label(c).tolist(), [PRE, PRE, IMM, CON, CON, DET, POST, POST])
        self.assertEqual(ph.runs(ph.label(c)),
                         [(PRE, 0, 1), (IMM, 2, 2), (CON, 3, 4), (DET, 5, 5), (POST, 6, 7)])

    def test_aborted_grasp_goes_back_to_pre_grasp(self):
        self.assertEqual(ph.label(np.array([0, 0.5, 0.1, 0])).tolist(), [PRE, IMM, PRE, PRE])

    def test_ends_holding(self):
        self.assertEqual(ph.label(np.array([0, 0, 0.5, 1, 1])).tolist(), [PRE, PRE, IMM, CON, CON])

    def test_regrasp_during_detach(self):
        self.assertEqual(ph.label(np.array([0, 0.5, 1, 0.5, 1, 0.5, 0])).tolist(),
                         [PRE, IMM, CON, DET, CON, DET, POST])

    def test_one_phase_per_frame(self):
        # A jump from open to closed still passes through immobilization, as in Robo2VLM.
        self.assertEqual(ph.label(np.array([0, 1, 1])).tolist(), [PRE, IMM, CON])


class TestHasPhases(unittest.TestCase):
    def test_min_frames(self):
        lab = ph.label(np.array([0, 0, 0.5, 1, 1, 0.5, 0]))  # post_grasp lasts 1 frame
        self.assertTrue(ph.has_phases(lab, {PRE, CON, POST}, min_frames=1))
        self.assertFalse(ph.has_phases(lab, {PRE, CON, POST}, min_frames=2))
        self.assertTrue(ph.has_phases(lab, {PRE, CON}, min_frames=2))


if __name__ == "__main__":
    unittest.main()
