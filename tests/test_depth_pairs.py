"""Tests for submission/src/aijgen/depth/{pairs,render,tier1}.py. Run: python -m unittest discover -s tests"""
import itertools
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission" / "src"))
from aijgen.depth import pairs as P  # noqa: E402
from aijgen.depth import render as R  # noqa: E402
from aijgen.depth import tier1 as T  # noqa: E402
from aijgen.depth.lerobot import Episode  # noqa: E402

POINTS = {"edge_radius": 4, "edge_max": 0.1, "erode_frac": 0.008, "border_frac": 0.05, "min_segment_area": 0.002,
          "large_segment_area": 0.25, "background_disp": 0.2, "points_per_segment": 2, "grid": [4, 6],
          "min_cell_pixels": 50}
PAIRS = {"min_dist_frac": 0.0625, "same_row_frac": 0.03, "gap": 0.15, "gap_inconsistent": 0.25,
         "gap_same_row": 0.2}
STAB = {"stability_change": 0.1, "stability_min_valid": 2}


def pt(y, x, d, seg, kind="object", area=0.01):
    return {"y": y, "x": x, "d": d, "seg": seg, "kind": kind, "area": area}


class TestSampling(unittest.TestCase):
    def setUp(self):
        # 240x320 frame: far wall on top (large segment), a near box (small segment), unmasked floor
        self.dn = np.tile(np.linspace(0.0, 1.0, 240, dtype=np.float32)[:, None], (1, 320))
        self.labels = np.zeros((240, 320), np.int16)
        self.labels[:80] = 1                    # wall: 33% of the frame, far
        self.labels[150:200, 100:160] = 2       # box
        self.dn[150:200, 100:160] = 0.95        # sticks out towards the camera

    def test_points_inside_segments_and_off_edges(self):
        pts = P.sample_points(self.labels, self.dn, POINTS, np.random.default_rng(0))
        edge = P.edge_map(self.dn, 4)
        for q in pts:
            if q["seg"] > 0:
                self.assertEqual(self.labels[q["y"], q["x"]], q["seg"])
            else:
                self.assertEqual(self.labels[q["y"], q["x"]], 0)
            self.assertLess(edge[q["y"], q["x"]], 0.1)
        kinds = {q["seg"]: q["kind"] for q in pts}
        self.assertEqual(kinds[1], "background")  # large and far
        self.assertEqual(kinds[2], "object")
        self.assertLessEqual(sum(q["seg"] == 2 for q in pts), 2)  # per segment, not per pixel

    def test_points_inside_border_rectangle(self):
        self.labels[:, :40] = 3  # a segment hugging the left border: only its part right of the margin is usable
        cfg = dict(POINTS, border_frac=0.15)  # 36 px top/bottom, 48 px left/right
        pts = P.sample_points(self.labels, self.dn, cfg, np.random.default_rng(0))
        self.assertTrue(pts)
        for q in pts:
            self.assertTrue(36 <= q["y"] < 240 - 36 and 48 <= q["x"] < 320 - 48, q)
        self.assertNotIn(3, {q["seg"] for q in pts})  # entirely inside the margin
        self.assertTrue(any(q["y"] < 36 or q["x"] < 48 or q["x"] >= 272 or q["y"] >= 204
                            for q in P.sample_points(self.labels, self.dn, dict(cfg, border_frac=0.0),
                                                     np.random.default_rng(0))))

    def test_deterministic(self):
        a = P.sample_points(self.labels, self.dn, POINTS, np.random.default_rng(3))
        b = P.sample_points(self.labels, self.dn, POINTS, np.random.default_rng(3))
        self.assertEqual(a, b)


class TestPairs(unittest.TestCase):
    def test_groups_and_thresholds(self):
        h, w = 480, 640
        lower_near = [pt(400, 100, 0.8, 1), pt(100, 500, 0.6, 2)]   # gap 0.2, closer is lower -> passes 0.15
        higher_near = [pt(100, 100, 0.8, 3), pt(400, 500, 0.6, 4)]  # gap 0.2, closer is higher -> needs 0.25
        same_row = [pt(200, 100, 0.85, 5), pt(205, 500, 0.6, 6)]    # gap 0.25, same row -> passes 0.2
        out, considered = P.candidate_pairs(lower_near, (h, w), PAIRS)
        self.assertEqual((len(out), out[0]["group"], out[0]["near"]["seg"]), (1, "consistent", 1))
        self.assertEqual(P.candidate_pairs(higher_near, (h, w), PAIRS)[0], [])
        out, _ = P.candidate_pairs(same_row, (h, w), PAIRS)
        self.assertEqual(out[0]["group"], "same_row")

    def test_same_segment_and_close_points_skipped(self):
        # seg1 twice (same segment), and seg1@(10,10) is within 40 px of seg2@(12,30)
        out, considered = P.candidate_pairs([pt(10, 10, 0.9, 1), pt(400, 600, 0.1, 1), pt(12, 30, 0.5, 2)],
                                            (480, 640), PAIRS)
        self.assertEqual(considered, 1)  # only seg1@(400,600) vs seg2
        self.assertEqual(len(out), 1)    # gap 0.4, closer point higher: passes the 0.25 threshold
        self.assertEqual(out[0]["group"], "inconsistent")

    def test_stability(self):
        p = {"near": pt(10, 10, 0.6, 1), "far": pt(50, 50, 0.45, 2)}

        def frame(near, far):
            d = np.full((100, 100), 0.3, np.float32)
            d[10, 10], d[50, 50] = near, far
            return d

        same = frame(0.6, 0.45)
        moved = frame(0.2, 0.45)      # near point changed by 0.4 (arm passed over it): frame skipped
        flipped = frame(0.52, 0.54)   # both within tolerance, order reversed: disagreement
        self.assertTrue(P.stable(p, [same, frame(0.62, 0.44)], STAB))
        self.assertFalse(P.stable(p, [same, moved], STAB))           # only one usable neighbour
        self.assertTrue(P.stable(p, [same, moved, same], STAB))
        self.assertFalse(P.stable(p, [same, same, flipped], STAB))


class TestBalance(unittest.TestCase):
    def make(self, group, n, bg):
        out = []
        for i in range(n):
            near_y, far_y = (300, 100) if group == "consistent" else (100, 300) if group == "inconsistent" else (200, 200)
            out.append({"near": pt(near_y, 10 + i, 0.9, 2 * i), "far": pt(far_y, 600 - i, 0.5, 2 * i + 1),
                        "g": 0.4, "group": group, "background": i < bg})
        return out

    def test_balanced_groups_letters_and_background(self):
        rng = np.random.default_rng(0)
        pairs = self.make("consistent", 100, 50) + self.make("inconsistent", 30, 10) + self.make("same_row", 50, 0)
        out = P.balance(pairs, 0.15, rng)
        groups = [p["group"] for p in out]
        self.assertEqual(groups.count("consistent"), groups.count("inconsistent"))
        self.assertEqual(groups.count("same_row"), 50)  # no background among them: all kept
        for g in P.GROUPS:
            members = [p for p in out if p["group"] == g]
            self.assertLessEqual(sum(p["background"] for p in members), round(0.15 * len(members)) + 1)
        P.assign_letters(out, rng)
        audit = P.audit(out, lambda p: (480, 640))
        self.assertAlmostEqual(audit["always_A"], 0.5, delta=0.03)
        self.assertAlmostEqual(audit["lower_is_closer"], 0.5, delta=0.1)


class TestRender(unittest.TestCase):
    def test_markers_inside_and_drawn(self):
        img = np.full((384, 512, 3), 128, np.uint8)
        pts = {"A": (5, 3), "B": (500, 380)}  # corners: label boxes must be moved inside
        for style in ("ring", "dot"):
            out = R.draw(img, pts, style)
            self.assertEqual(out.shape, img.shape)
            self.assertGreater(np.abs(out.astype(int) - img).sum(), 0)
        ring = R.draw(img, {"A": (100, 100), "B": (300, 300)}, "ring")
        r = max(4, round(0.011 * 512))
        self.assertTrue((ring[100, 100 + r] == R.RED).all())       # on the ring
        self.assertTrue((ring[100, 100] == 128).all())             # hollow centre: the point stays visible

    def test_questions(self):
        cfg = {"p_blink": 0.6, "p_blink_farther": 0.1}
        rng = np.random.default_rng(0)
        for _ in range(200):
            prompt, answer, kind = R.question("B", "ring", cfg, rng)
            if kind == "blink_closer":
                self.assertTrue(prompt.startswith(R.NOTE))
                self.assertEqual(answer, "B")
            elif kind == "blink_farther":
                self.assertEqual(answer, "A")
            elif "yes or no" in prompt:
                self.assertEqual(answer, "No.")
        for _ in range(50):
            self.assertEqual(R.question("A", "dot", cfg, rng)[2], "free")


class TestSets(unittest.TestCase):
    SHAPE = (480, 640)

    def frame_pairs(self):
        # a near box low in the frame, a mid-depth row and a far row; two points share segment 7
        pts = [pt(400, 320, 0.95, 1), pt(250, 80, 0.6, 2), pt(250, 560, 0.55, 3), pt(120, 200, 0.2, 4),
               pt(120, 440, 0.15, 5), pt(330, 600, 0.5, 6), pt(60, 100, 0.05, 7), pt(60, 300, 0.1, 7)]
        return P.candidate_pairs(pts, self.SHAPE, PAIRS)[0]

    def test_target_beats_every_member(self):
        pairs = self.frame_pairs()
        beaten = {(P._key(p["near"]), P._key(p["far"])) for p in pairs}
        for relation in ("closest", "farthest"):
            for k in (3, 4):
                sets = P.candidate_sets(pairs, k, relation, self.SHAPE, PAIRS, np.random.default_rng(0))
                self.assertTrue(sets, (relation, k))
                for s in sets:
                    t, others = s["target"], s["others"]
                    self.assertEqual(len(others), k - 1)
                    for o in others:
                        edge = (P._key(t), P._key(o)) if relation == "closest" else (P._key(o), P._key(t))
                        self.assertIn(edge, beaten)
                    members = [t] + others
                    self.assertEqual(len({q["seg"] for q in members}), k)  # distinct segments
                    for a, b in itertools.combinations(members, 2):
                        self.assertGreaterEqual(np.hypot(a["x"] - b["x"], a["y"] - b["y"]),
                                                PAIRS["min_dist_frac"] * self.SHAPE[1])
        closest = P.candidate_sets(pairs, 4, "closest", self.SHAPE, PAIRS, np.random.default_rng(0))
        targets = {P._key(s["target"]): s for s in closest}
        self.assertIn((400, 320), targets)                                 # the box beats everything
        self.assertFalse({k for k in targets if k[0] <= 120})              # far points never are the closest
        self.assertTrue(targets[(400, 320)]["shortcut"])                   # the box is the lowest point

    def make_sets(self, k, n_hit, n_miss):
        out = []
        for i in range(n_hit + n_miss):
            t = pt(400 if i < n_hit else 100, 10 + i, 0.9, 10 * i)
            others = [pt(200 + j, 300 + j, 0.3, 10 * i + j + 1) for j in range(k - 1)]
            out.append({"target": t, "others": others, "relation": "closest", "k": k, "g": 0.4, "groups": [],
                        "background": False, "shortcut": i < n_hit})
        return out

    def test_balance_and_letters(self):
        rng = np.random.default_rng(0)
        sets = self.make_sets(4, 200, 120) + self.make_sets(3, 10, 100)
        out = P.balance_sets(sets, 0.15, rng)
        for k in (3, 4):
            bucket = [s for s in out if s["k"] == k]
            self.assertAlmostEqual(np.mean([s["shortcut"] for s in bucket]), 1 / k, delta=0.02)
        self.assertEqual(sum(s["k"] == 3 for s in out), 30)  # scarce hits: 10 hits + 20 misses
        P.assign_set_letters(out, rng)
        for k in (3, 4):
            bucket = [s for s in out if s["k"] == k]
            for key in ("slot", "opt_slot"):
                counts = Counter(s[key] for s in bucket)
                self.assertEqual(set(counts), set(range(k)))
                self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
            for s in bucket:
                self.assertIs(s["points"][s["slot"]], s["target"])
                self.assertEqual(len(s["points"]), k)
        audit = P.audit_sets(out, lambda s: self.SHAPE)
        self.assertAlmostEqual(audit["lowest_is_closest_highest_is_farthest"], audit["chance"], delta=0.03)

    def test_subsample_keeps_groups(self):
        pairs = TestBalance().make("consistent", 60, 0) + TestBalance().make("same_row", 30, 0)
        out = P.subsample(pairs, 30, np.random.default_rng(0))
        self.assertEqual(Counter(p["group"] for p in out), {"consistent": 20, "same_row": 10})
        self.assertIs(P.subsample(pairs, 500, np.random.default_rng(0)), pairs)


class TestSetRender(unittest.TestCase):
    def test_note_matches_lmms_eval(self):
        # lmms-eval blink: pre_prompt.format(", ".join(chr(65 + i) for i in range(len(choices))))
        self.assertEqual(R.note("ABCD"), "Note: You only need to respond with A, B, C, D without providing any "
                                         "additional information.\n")
        self.assertEqual(R.NOTE, "Note: You only need to respond with A, B without providing any additional "
                                 "information.\n")

    def test_set_questions(self):
        rng = np.random.default_rng(0)
        kinds, orders = Counter(), set()
        for i in range(300):
            k, slot, opt_slot = (4, i % 4, (i // 4) % 4) if i % 2 else (3, i % 3, (i // 3) % 3)
            prompt, answer, kind, labels = R.set_question(k, slot, opt_slot, "closest", "ring", 0.5, rng)
            kinds[kind.split("_")[1]] += 1
            self.assertEqual(len(labels), k)
            if kind == "set_letter":
                self.assertEqual(answer, labels[slot])
                self.assertTrue(prompt.startswith(R.note(labels)))
                self.assertIn(f"({answer}) {answer} is closest", prompt)
            else:
                self.assertEqual(labels, [str(n + 1) for n in range(k)])
                self.assertEqual(answer, chr(65 + opt_slot))
                lines = [l for l in prompt.split("\n") if "point " in l and l[0] in "(ABCD"]
                self.assertEqual(len(lines), k)
                target_line = next(l for l in lines if l.lstrip("(").startswith(answer))
                self.assertTrue(target_line.endswith(f"point {labels[slot]}"), (target_line, prompt))
                orders.add(tuple(l.split()[-1] for l in lines))
        self.assertGreater(min(kinds.values()), 100)  # both variants used
        self.assertGreater(len(orders), 10)           # option order varies

    def test_draw_four_markers(self):
        img = np.full((384, 512, 3), 128, np.uint8)
        pts = {"1": (5, 3), "2": (60, 40), "3": (500, 380), "4": (256, 200)}
        out = R.draw(img, pts, "ring")
        self.assertEqual(out.shape, img.shape)
        r = max(4, round(0.011 * 512))
        self.assertTrue((out[200, 256 + r] == R.RED).all())
        # a label box over another point's marker is rejected, a clear one is accepted
        self.assertFalse(R._box_ok((50, 30, 70, 50), [(60, 40)], r, 512, 384))
        self.assertTrue(R._box_ok((150, 130, 170, 150), [(60, 40), (500, 380)], r, 512, 384))


class TestViews(unittest.TestCase):
    VIEWS = ["observation.images.image_0", "observation.images.image_1", "observation.images.image_2"]

    def ep(self, index, view):
        return Episode("bridge", index, 30, "task", view, Path(f"{view}.mp4"), 0.0, 5.0)

    def test_episodes_grouped_across_views(self):
        cam0, cam1, cam2 = self.VIEWS
        per_view = {cam0: [self.ep(i, cam0) for i in (1, 2, 3)],  # episode 1 has every camera, 2 two, 3 only cam 0
                    cam1: [self.ep(i, cam1) for i in (1, 2)],
                    cam2: [self.ep(1, cam2)]}
        groups = T.episodes_by_index(per_view, 3)
        self.assertEqual([sorted(g) for g in groups], [self.VIEWS])
        self.assertEqual({e.index for e in groups[0].values()}, {1})
        self.assertEqual(groups[0][cam2].video_key, cam2)  # each view keeps its own video
        self.assertEqual([g[cam0].index for g in T.episodes_by_index(per_view, 2)], [1, 2])
        self.assertEqual(len(T.episodes_by_index(per_view, 1)), 3)

    def test_question_view_rotates_evenly(self):
        counts = Counter(T.question_view(self.VIEWS, turn) for turn in range(300))
        self.assertEqual(set(counts.values()), {100})
        self.assertEqual(T.question_view(["observation.images.image"], 7), "observation.images.image")


if __name__ == "__main__":
    unittest.main()
