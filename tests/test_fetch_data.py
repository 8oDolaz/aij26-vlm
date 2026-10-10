"""Tests for the selection logic in tools/fetch_data.py (no network). Run: python -m unittest discover -s tests"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import fetch_data as fd  # noqa: E402


class TestPathLength(unittest.TestCase):
    def test_path_length(self):
        ep = np.array([0, 0, 0, 1, 1])
        x = np.array([[0, 0], [1, 0], [1, 2], [5, 5], [5, 5]], float)
        ids, d = fd.path_length(ep, x)
        self.assertEqual(d.tolist(), [3.0, 0.0])


def scene(files, multiview=()):
    """Episodes from {file: [task, ...]}; episodes in multiview files also need a 2nd-camera file."""
    primary, tasks, mv, extra = [], [], [], []
    for f, ts in files.items():
        for t in ts:
            primary.append(f)
            tasks.append(t)
            mv.append(f in multiview)
            extra.append({f"cam1/{f}"} if f in multiview else set())
    n = len(primary)
    return (np.array(primary, dtype=object), np.array(tasks, dtype=object), np.ones(n, bool),
            np.array(mv), extra)


class TestSelectFiles(unittest.TestCase):
    def test_prefers_new_instructions_and_respects_budget(self):
        # a and b hold the same instruction (sorted dataset), c holds new ones.
        primary, tasks, usable, mv, extra = scene({"a": ["x"] * 4, "b": ["x"] * 4, "c": ["y", "z", "w", "v"]})
        sizes = {"a": 10, "b": 10, "c": 10}
        # Pick order: c first (4 new instructions), then a (lower index than b, same gain).
        self.assertEqual(fd.select_files(primary, tasks, usable, mv, extra, sizes, budget=20), ["c", "a"])
        self.assertEqual(fd.select_files(primary, tasks, usable, mv, extra, sizes, budget=9), [])

    def test_extra_camera_files_count_toward_cost(self):
        primary, tasks, usable, mv, extra = scene({"a": ["x"] * 4, "b": ["y"] * 4}, multiview={"a"})
        sizes = {"a": 10, "b": 10, "cam1/a": 10}
        # a alone would cost 20 with its second camera, so only b fits in 15.
        self.assertEqual(fd.select_files(primary, tasks, usable, mv, extra, sizes, budget=15), ["b"])

    def test_mv_share_buys_multiview_first(self):
        primary, tasks, usable, mv, extra = scene({"a": ["x"] * 4, "b": ["y"] * 4}, multiview={"b"})
        sizes = {"a": 10, "b": 10, "cam1/b": 10}
        self.assertEqual(fd.select_files(primary, tasks, usable, mv, extra, sizes, budget=20), ["a"])
        self.assertEqual(fd.select_files(primary, tasks, usable, mv, extra, sizes, budget=20, mv_share=1.0), ["b"])

    def test_unusable_episodes_do_not_count(self):
        primary, tasks, usable, mv, extra = scene({"a": ["x"] * 4, "b": ["y"] * 2})
        usable[:4] = False
        self.assertEqual(fd.select_files(primary, tasks, usable, mv, extra, {"a": 1, "b": 1}, budget=1), ["b"])

    def test_ties_go_to_lower_file(self):
        primary, tasks, usable, mv, extra = scene({"b": ["x"], "a": ["y"]})
        self.assertEqual(fd.select_files(primary, tasks, usable, mv, extra, {"a": 1, "b": 1}, budget=1), ["a"])


if __name__ == "__main__":
    unittest.main()
