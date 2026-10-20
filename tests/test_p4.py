"""Tests for the P4 SAM 3 questions (docs/sam3_vqa_plan.md): cross-view matching (aijgen.scene.crossview) and ERQA
task success with hard negatives (aijgen.generators.task_success, aijgen.scene.instructions).
Run: python -m unittest discover -s tests

Numpy only: no model, cv2 or video is touched.
"""
import importlib.util
import re
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "submission" / "src"))
from aijgen import phases as ph  # noqa: E402
from aijgen.generators import task_success as TS  # noqa: E402
from aijgen.generators.base import Episode, Question  # noqa: E402
from aijgen.generators.run import image_cameras, image_overlays  # noqa: E402
from aijgen.scene import crossview as CV  # noqa: E402
from aijgen.scene import formats as F  # noqa: E402
from aijgen.scene.instructions import parse_task  # noqa: E402
from aijgen.scene.templates import Obj, Scene  # noqa: E402
from aijgen.scene.vocab import load_dictionary  # noqa: E402

H, W = 480, 640
DICT = """
pot / pan
bowl
carrot
spoon
towel / cloth
sponge
drawer
coke can
pepsi can
"""


def entries():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.txt"
        path.write_text(DICT)
        return load_dictionary(path)


ENTRIES = entries()


def rect_mask(box, size=(H, W)):
    m = np.zeros(size, bool)
    x0, y0, x1, y1 = (int(v) for v in box)
    m[y0:y1, x0:x1] = True
    return m


def obj(name, box):
    return Obj(name, list(map(float, box)), 0.95, rect_mask(box))


def scene(subjects=(), others=()):
    """A Scene whose subjects are `subjects` and whose kept objects are subjects + `others`."""
    s = Scene((H, W))
    s.subjects = list(subjects)
    s.objects = list(subjects) + list(others)
    return s


# ---------------------------------------------------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------------------------------------------------
class TestInstructions(unittest.TestCase):
    def test_place(self):
        t = parse_task("Put the spoon on the towel.", ENTRIES)
        self.assertEqual((t.kind, t.obj, t.target), ("place", "spoon", "towel"))
        self.assertEqual(t.swapped("carrot"), "Put the carrot on the towel")
        self.assertEqual(t.gerund(), "putting the spoon on the towel")

    def test_place_variants(self):
        self.assertEqual(parse_task("put the blue spoon on top of the cloth", ENTRIES).target, "cloth")
        t = parse_task("pick up the carrot and put it in the pot", ENTRIES)
        self.assertEqual((t.kind, t.obj, t.target), ("place", "carrot", "pot"))
        self.assertIsNone(t.gerund())  # "picking up the carrot and put it ..." is not English
        self.assertEqual(parse_task("place coke can into top drawer", ENTRIES).obj, "coke can")

    def test_pick(self):
        t = parse_task("pick coke can", ENTRIES)
        self.assertEqual((t.kind, t.obj, t.target), ("pick", "coke can", None))
        self.assertEqual(t.swapped("pepsi can"), "pick pepsi can")  # no article added where there was none
        self.assertEqual(t.gerund(), "picking up coke can")

    def test_rejected(self):
        for task in ["move the pot to the left of the stove", "put the spoon in the pot on the stove",
                     "pick coke can from top drawer", "put the spoon on the table", "put the red object in the pot",
                     "move the spoon near the bowl", "fold the cloth", "put the pan in the pot", "",
                     "put the spoon on the left towel"]:
            self.assertIsNone(parse_task(task, ENTRIES), task)


# ---------------------------------------------------------------------------------------------------------------------
# Clean-episode filter and frames
# ---------------------------------------------------------------------------------------------------------------------
DCFG = {"fps": 5, "max_length": 60, "gripper_dim": 6, "closed_high": False, "full_close_raw": 0.07}


def episode(kind="place", n=40, close=(12, 14), open_at=28, carry=0.15, away=0.10, length_extra=0,
            extra_close=None, abort=None, raw_closed=0.4):
    """Synthetic Bridge-like episode: arm goes down 10 cm to the grasp, closes over close=(start, contact frame),
    carries `carry` m, opens at open_at (place) and moves `away` m up afterwards."""
    n = n + length_extra
    c = np.zeros(n)
    c[close[0]:close[1]] = 0.5
    c[close[1]:] = 1.0
    if kind == "place":
        c[open_at] = 0.5
        c[open_at + 1:] = 0.0
    if extra_close is not None:  # gripper closes again after the release
        c[extra_close] = 0.5
        c[extra_close + 1:] = 1.0
    if abort is not None:  # a closing that opens again before contact
        c[abort] = 0.5
        c[abort + 1] = 0.0
    xyz = np.zeros((n, 3))
    xyz[:, 2] = 0.20
    xyz[:close[1], 2] = np.linspace(0.20, 0.10, close[1])
    xyz[close[1]:, 2] = 0.10
    end = open_at if kind == "place" else n - 1
    xyz[close[1]:end + 1, 0] = np.linspace(0.0, carry, end - close[1] + 1)
    xyz[end + 1:, 0] = carry
    if kind == "pick":
        xyz[close[1]:, 2] = np.linspace(0.10, 0.20, n - close[1])
    else:
        xyz[end + 1:, 2] = 0.10 + np.minimum(away, 0.02 * np.arange(1, n - end))
    state = np.zeros((n, 7))
    state[:, :3] = xyz
    state[:, 6] = 1.0 - c * (1.0 - raw_closed)  # Bridge: low raw = closed
    return Episode("bridge", 0, "put the spoon on the towel", n, state, 5.0, c, ph.label(c))


class TestCleanEpisode(unittest.TestCase):
    def test_textbook_place_and_pick(self):
        info, why = TS.clean_episode(episode(), "place", DCFG)
        self.assertEqual(why, "ok")
        self.assertEqual(info["contact"][0], 14)
        info, why = TS.clean_episode(episode("pick"), "pick", DCFG)
        self.assertEqual(why, "ok")

    def test_failure_signs(self):
        cases = {
            "regrasp": (episode(extra_close=34), "place", DCFG),
            "irregular_phases": (episode(abort=5), "place", DCFG),
            "short_contact": (episode(open_at=17), "place", DCFG),
            "not_carried": (episode(carry=0.0), "place", DCFG),
            "release_at_end": (episode(open_at=38, n=40), "place", DCFG),  # 0.4 s after the release
            "too_long": (episode(), "place", {**DCFG, "max_length": 30}),
            "full_close": (episode(raw_closed=0.05), "place", DCFG),
        }
        for want, (ep, kind, d) in cases.items():
            self.assertEqual(TS.clean_episode(ep, kind, d)[1], want, want)
        # a place task that never lets go, and a pick task that lets go
        self.assertEqual(TS.clean_episode(episode("pick"), "place", DCFG)[1], "irregular_phases")
        self.assertEqual(TS.clean_episode(episode(), "pick", DCFG)[1], "irregular_phases")
        # pick: holding but not lifted
        ep = episode("pick")
        ep.state[14:, 2] = 0.10
        self.assertEqual(TS.clean_episode(ep, "pick", DCFG)[1], "not_lifted")
        # no descent before the grasp
        ep = episode()
        ep.state[:14, 2] = 0.10
        self.assertEqual(TS.clean_episode(ep, "place", DCFG)[1], "no_descent")

    def test_done_frames_window(self):
        ep = episode(n=60)
        info, _ = TS.clean_episode(ep, "place", DCFG)
        done = TS.done_frames(ep, "place", info)
        self.assertTrue(done)
        self.assertTrue(all(t >= 60 - 3 for t in done), done)  # ceil(5% of 60) = 3
        ep = episode(away=0.0)  # arm stays where it let go: the release is not visible
        info, _ = TS.clean_episode(ep, "place", DCFG)
        self.assertEqual(TS.done_frames(ep, "place", info), [])

    def test_early_frame(self):
        ep = episode()
        info, _ = TS.clean_episode(ep, "place", DCFG)
        rng = np.random.default_rng(0)
        s, e = info["pre"]
        for _ in range(50):
            f = TS.early_frame(info, rng)
            self.assertTrue(s <= f <= s + (e - s) / 2)


# ---------------------------------------------------------------------------------------------------------------------
# Image checks for task success
# ---------------------------------------------------------------------------------------------------------------------
TOWEL = (300, 250, 500, 400)
SPOON_START, SPOON_END = (60, 300, 140, 340), (360, 300, 440, 340)
CARROT = (520, 60, 600, 140)


class TestSuccessChecks(unittest.TestCase):
    task = parse_task("put the spoon on the towel", ENTRIES)

    def test_yes_confirmed(self):
        start = scene([obj("spoon", SPOON_START), obj("towel", TOWEL), obj("carrot", CARROT)])
        end = scene([obj("spoon", SPOON_END), obj("cloth", TOWEL), obj("carrot", CARROT)])  # same dictionary line
        self.assertEqual(TS.yes_confirmed(self.task, end, start, ENTRIES), "ok")
        self.assertEqual(TS.yes_confirmed(self.task, start, start, ENTRIES), "not_on_target_at_end")
        self.assertEqual(TS.yes_confirmed(self.task, end, end, ENTRIES), "on_target_at_start")
        self.assertEqual(TS.yes_confirmed(self.task, scene([obj("towel", TOWEL)]), start, ENTRIES), "object_not_seen")

    def test_pick_lift(self):
        task = parse_task("pick coke can", ENTRIES)
        start = scene([obj("coke can", (100, 300, 140, 380))])
        lifted = scene([obj("coke can", (100, 200, 140, 280))])
        self.assertEqual(TS.yes_confirmed(task, lifted, start, ENTRIES), "ok")
        self.assertEqual(TS.yes_confirmed(task, start, start, ENTRIES), "not_lifted_on_image")

    def test_swap_candidates(self):
        start = scene([obj("spoon", SPOON_START), obj("towel", TOWEL), obj("carrot", CARROT),
                       obj("bowl", (20, 20, 100, 100)), obj("sponge", (330, 260, 380, 290)),
                       obj("pan", (150, 150, 250, 230))])
        end = scene([obj("spoon", SPOON_END), obj("towel", TOWEL), obj("carrot", CARROT),
                     obj("bowl", (200, 20, 280, 100)),          # moved: no
                     obj("sponge", (330, 260, 380, 290)),       # already on the towel: no
                     obj("pan", (150, 150, 250, 230))])
        self.assertEqual(TS.swap_candidates(self.task, end, start, ENTRIES), ["carrot", "pan"])
        # never the object itself, the target, or something of the same kind
        task = parse_task("put the carrot in the pot", ENTRIES)
        self.assertNotIn("pan", TS.swap_candidates(task, end, start, ENTRIES))

    def test_question_format_and_balance(self):
        rng = np.random.default_rng(0)
        letters = Counter()
        for _ in range(1000):
            text, letter = TS.question("put the spoon on the towel", None, True, rng)
            letters[letter] += 1
            self.assertNotIn("successful in", text)  # no gerund template without a gerund
            self.assertTrue(text.endswith(F.ERQA_SUFFIX))
            self.assertRegex(text, r" Choices: A\. (Yes|No)\. B\. (Yes|No)\. Please answer")
        self.assertTrue(450 < letters["A"] < 550, letters)


# ---------------------------------------------------------------------------------------------------------------------
# Cross-view
# ---------------------------------------------------------------------------------------------------------------------
CV_CFG = {"per_frame": 1, "max_target_frac": 0.15, "min_dot_dist": 0.08, "border": 0.05}


def views():
    """Carrot (small) seen in both views, with three other objects in view 2."""
    v1 = scene([obj("carrot", (100, 100, 160, 160)), obj("bowl", (300, 100, 400, 200))])
    v2 = scene([obj("carrot", (400, 300, 460, 360)), obj("bowl", (100, 100, 200, 200)),
                obj("spoon", (250, 80, 330, 140)), obj("sponge", (100, 300, 200, 380))])
    return v1, v2


class TestCrossview(unittest.TestCase):
    def test_interior_point(self):
        m = rect_mask((100, 100, 200, 160))
        x, y = CV.interior_point(m)
        self.assertTrue(130 <= x <= 170 and 120 <= y <= 140, (x, y))
        self.assertIsNone(CV.interior_point(rect_mask((0, 0, 20, 20))))  # at the border

    def test_matches_and_third_view_veto(self):
        v1, v2 = views()
        self.assertEqual({a.name for a, _ in CV.matches(v1, v2, [], ENTRIES)}, {"carrot", "bowl"})
        # a third view with two carrots (none a subject) vetoes the carrot; one where it is absent does not
        two = scene([], [obj("carrot", (10, 10, 50, 50)), obj("carrot", (300, 300, 340, 340))])
        self.assertEqual({a.name for a, _ in CV.matches(v1, v2, [two], ENTRIES)}, {"bowl"})
        self.assertEqual({a.name for a, _ in CV.matches(v1, v2, [scene()], ENTRIES)}, {"carrot", "bowl"})

    def test_distractors_never_overlap_target(self):
        _, v2 = views()
        target = v2.subjects[0]
        v2.objects.append(obj("towel", (390, 290, 470, 370)))  # a box on the carrot under another name
        names = {o.name for o in CV.distractors(v2, target, ENTRIES, CV_CFG)}
        self.assertEqual(names, {"bowl", "spoon", "sponge"})

    def test_crosspoint(self):
        v1, v2 = views()
        items = CV.crossview_crosspoint(v1, v2, [], ENTRIES, np.random.default_rng(0), CV_CFG)
        self.assertEqual(len(items), 1)
        it = items[0]
        self.assertTrue(it.question.startswith(F.CROSSPOINT_PRE + " "))
        self.assertRegex(it.question, r"\nA\. (Green|Blue|Yellow) dot  B\. \w+ dot  C\. \w+ dot$")
        first = [o for o in it.overlays if o["image"] == 0]
        second = [o for o in it.overlays if o["image"] == 1]
        self.assertEqual((len(first), len(second)), (1, 3))
        x, y = first[0]["xy"]
        self.assertTrue(100 <= x < 160 and 100 <= y < 160)  # on the carrot in view 1
        answer_color = re.search(rf"{it.answer}\. (\w+) dot", it.question).group(1).lower()
        dot = next(o for o in second if tuple(o["color"]) == CV.DOT_COLORS[answer_color])
        x, y = dot["xy"]
        self.assertTrue(400 <= x < 460 and 300 <= y < 360)  # the answer dot is on the carrot in view 2

    def test_crosspoint_small_targets_only(self):
        v1, v2 = views()
        big = {**CV_CFG, "max_target_frac": 0.05}  # the carrot's diagonal is ~0.13 of the width
        self.assertEqual(CV.crossview_crosspoint(v1, v2, [], ENTRIES, np.random.default_rng(0), big), [])

    def test_erqa(self):
        letters = Counter()
        for seed in range(200):
            v1, v2 = views()
            items = CV.crossview_erqa(v1, v2, [], ENTRIES, np.random.default_rng(seed), CV_CFG)
            for it in items:
                letters[it.answer] += 1
                self.assertTrue(it.question.endswith(F.ERQA_SUFFIX))
                self.assertEqual(len(re.findall(r" [A-D]\. ", it.question)), 4)
                query = re.search(r"as the (\w+) circle in the first image", it.question).group(1)
                self.assertNotRegex(it.question.split("Choices:")[1], rf"(?i)\b{query}\b")  # not among the options
                rings = [o for o in it.overlays if o["kind"] == "ring"]
                self.assertEqual(len(rings), 5)
        self.assertEqual(set(letters), set("ABCD"))
        self.assertTrue(min(letters.values()) > 25, letters)


# ---------------------------------------------------------------------------------------------------------------------
# Plumbing: per-image cameras and overlays, option counts in the mix builder
# ---------------------------------------------------------------------------------------------------------------------
class TestPlumbing(unittest.TestCase):
    def test_image_cameras(self):
        q = Question("q", "A", [5, 5], "scene_qa", {"cameras": ["c0", "c1"]})
        self.assertEqual(image_cameras(q, ["c0", "c1", "c2"]), [["c0", "c1"]])
        q = Question("q", "A", [5], "scene_qa", {"camera": "c2"})
        self.assertEqual(image_cameras(q, ["c0", "c1", "c2"]), [["c2"]])
        q = Question("q", "A", [3, 9], "phase_qa")
        self.assertEqual(image_cameras(q, ["c0", "c1"]), [["c0", "c0"], ["c1", "c1"]])

    def test_image_overlays(self):
        marks = [{"kind": "dot", "image": 0}, {"kind": "dot", "image": 1}, {"kind": "box"}]
        self.assertEqual(image_overlays(marks, 0), [marks[0], marks[2]])
        self.assertEqual(image_overlays(marks, 1), [marks[1], marks[2]])

    def test_mix_option_counts(self):
        spec = importlib.util.spec_from_file_location("build_sft_mix", REPO / "tools" / "build_sft_mix.py")
        mix = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mix)
        rec = lambda text: {"messages": [{"role": "user", "content": text}, {"role": "assistant", "content": "A"}]}
        self.assertEqual(mix.n_options(rec(F.erqa("Was the task successful: put the spoon on the towel",
                                                  ["No", "Yes"]))), 2)
        self.assertEqual(mix.n_options(rec(F.crosspoint(F.CROSSPOINT_JUDGE_Q[0], ["Green dot", "Blue dot",
                                                                                   "Yellow dot"]))), 3)
        self.assertEqual(mix.n_options(rec(F.blink("Is the cup left of the pot?", ["yes", "no"]))), 2)
        self.assertEqual(mix.n_options(rec(F.embspatial("q", ["a", "b", "c", "d"]))), 4)


if __name__ == "__main__":
    unittest.main()
