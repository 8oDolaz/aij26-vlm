"""Tests for submission/src/aijgen/scene (labels of the SAM 3 scene questions). Run: python -m unittest discover -s tests

Numpy only: no model, cv2 or video is touched.
"""
import re
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission" / "src"))
from aijgen.scene import detect as D  # noqa: E402
from aijgen.scene import formats as F  # noqa: E402
from aijgen.scene import templates as T  # noqa: E402
from aijgen.scene.geometry import erode, relations  # noqa: E402
from aijgen.scene.vocab import DICT_DIR, load_dictionary, load_names, plural, similar  # noqa: E402

H, W = 480, 640
P = D.FilterParams()
DICT = """
pot / pan / saucepan
bowl
carrot
spoon
sponge
stove / cooktop
coke can
pepsi can
"""


def entries():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.txt"
        path.write_text(DICT)
        return load_dictionary(path)


ENTRIES = entries()


def rect_mask(box):
    m = np.zeros((H, W), bool)
    x0, y0, x1, y1 = (int(v) for v in box)
    m[y0:y1, x0:x1] = True
    return m


def raw(name, box, score=0.95, disp=None, mask=True):
    return D.RawBox(name, list(map(float, box)), score, rect_mask(box) if mask else None, disp)


def detections(boxes, negatives=(), presence=None):
    pres = {n: 0.99 for n in ENTRIES} | {n: 0.01 for n in negatives} | (presence or {})
    return D.FrameDetections("cam", (H, W), list(ENTRIES), list(negatives), pres, list(boxes))


def scene(boxes, **kw):
    return T.build_scene(detections(boxes, **{k: v for k, v in kw.items() if k in ("negatives", "presence")}),
                         ENTRIES, P, exclude_subjects=kw.get("exclude", ("stove",)))


# A tidy scene: four separated objects with clearly different depths
CARROT = (40, 60, 140, 160)     # top left
BOWL = (300, 60, 420, 160)      # top middle
SPONGE = (40, 300, 160, 420)    # bottom left
SPOON = (480, 300, 600, 420)    # bottom right
TIDY = [raw("carrot", CARROT, disp=0.2), raw("bowl", BOWL, disp=0.4), raw("sponge", SPONGE, disp=0.6),
        raw("spoon", SPOON, disp=0.9)]


class TestVocab(unittest.TestCase):
    def test_groups_and_similarity(self):
        self.assertTrue(similar("pot", "saucepan", ENTRIES))       # same line
        self.assertTrue(similar("coke can", "pepsi can", ENTRIES))  # same last word
        self.assertFalse(similar("pot", "bowl", ENTRIES))

    def test_plural(self):
        self.assertEqual([plural(n) for n in ("coke can", "knife", "dish", "strawberry", "tomato", "grapes")],
                         ["coke cans", "knives", "dishes", "strawberries", "tomatoes", "grapes"])

    def test_absent_lists_never_overlap_their_dictionary(self):
        """An absent name sharing a last word with a dictionary name could be in the scene."""
        for ds in ("bridge", "fractal"):
            words = {n.split()[-1] for n in load_dictionary(DICT_DIR / f"{ds}.txt")}
            absent = load_names(DICT_DIR / f"{ds}_absent.txt")
            self.assertTrue(absent)
            self.assertFalse([n for n in absent if n.split()[-1] in words], ds)


class TestGeometry(unittest.TestCase):
    def test_relations_need_whole_box_separation(self):
        a, b = (0, 0, 100, 100), (150, 200, 250, 300)
        self.assertEqual(relations(a, b, 10), {"left", "above"})
        self.assertEqual(relations(b, a, 10), {"right", "under"})
        self.assertEqual(relations((0, 0, 100, 100), (95, 0, 200, 100), 10), set())  # overlapping
        self.assertEqual(relations((0, 0, 100, 100), (105, 0, 200, 100), 10), set())  # gap below the minimum

    def test_erode(self):
        m = np.zeros((20, 20), bool)
        m[5:15, 5:15] = True
        self.assertEqual(int(erode(m, 1).sum()), 64)
        self.assertTrue(erode(m, 1)[6:14, 6:14].all())


class TestSelect(unittest.TestCase):
    def test_duplicates_ties_and_losers(self):
        cands = [{"name": "pot", "box": [0, 0, 100, 100], "score": 0.95},
                 {"name": "saucepan", "box": [2, 2, 100, 100], "score": 0.9},   # same group, same object
                 {"name": "bowl", "box": [300, 0, 400, 100], "score": 0.9},
                 {"name": "carrot", "box": [301, 1, 400, 100], "score": 0.88},  # near tie with the bowl
                 {"name": "sponge", "box": [0, 300, 100, 400], "score": 0.95},
                 {"name": "spoon", "box": [1, 301, 100, 400], "score": 0.7}]    # clearly loses to the sponge
        # threshold lowered: above the default 0.85 a box can never lose by the 0.15 margin
        sel = D.select_candidates(cands, ENTRIES, H * W, D.FilterParams(threshold=0.5))
        self.assertEqual(sorted(c["name"] for c in sel["kept"]), ["pot", "sponge"])
        self.assertEqual(sorted(c["name"] for c in sel["ambiguous"]), ["bowl", "carrot"])
        self.assertEqual((sel["duplicate"], sel["lost"]), (1, 1))


class TestBuildScene(unittest.TestCase):
    def test_unique_objects_are_subjects(self):
        s = scene(TIDY)
        self.assertEqual(sorted(o.name for o in s.subjects), ["bowl", "carrot", "sponge", "spoon"])
        self.assertEqual(s.counts, {"carrot": 1, "bowl": 1, "sponge": 1, "spoon": 1})

    def test_two_of_a_kind_are_counted_not_named(self):
        s = scene(TIDY + [raw("pot", (200, 250, 300, 350)), raw("pot", (350, 250, 450, 350))])
        self.assertNotIn("pot", [o.name for o in s.subjects])
        self.assertEqual(s.counts["pot"], 2)

    def test_similar_name_elsewhere_blocks_both(self):
        s = scene(TIDY + [raw("pot", (200, 250, 300, 350)), raw("saucepan", (350, 250, 450, 350))])
        self.assertFalse({"pot", "saucepan"} & {o.name for o in s.subjects})
        self.assertFalse({"pot", "saucepan"} & set(s.counts))

    def test_uncertain_box_elsewhere_blocks_subject_and_count(self):
        s = scene(TIDY + [raw("pot", (200, 250, 300, 350)), raw("pot", (350, 250, 450, 350), score=0.5)])
        self.assertNotIn("pot", [o.name for o in s.subjects])
        self.assertNotIn("pot", s.counts)

    def test_duplicate_box_on_same_object_is_fine(self):
        s = scene(TIDY + [raw("pot", (200, 250, 300, 350)), raw("saucepan", (202, 252, 300, 350), score=0.5)])
        self.assertIn("pot", [o.name for o in s.subjects])
        self.assertEqual(s.counts["pot"], 1)

    def test_excluded_names(self):
        s = scene(TIDY + [raw("stove", (200, 250, 300, 350))])
        self.assertNotIn("stove", [o.name for o in s.subjects])
        self.assertNotIn("stove", s.counts)

    def test_absent_needs_low_presence_and_no_maybe_box(self):
        negs = ["piano", "bicycle", "guitar"]
        boxes = TIDY + [D.RawBox("bicycle", [0, 0, 50, 50], 0.4)]
        s = scene(boxes, negatives=negs, presence={"guitar": 0.2})
        self.assertEqual(s.absent, ["piano"])


def letters_of(template, sc, n=1500, cfg=None):
    out = Counter()
    for seed in range(n):
        for it in T.TEMPLATES[template](sc, np.random.default_rng(seed), cfg or {}):
            out[it.answer] += 1
    return out


def options_of(question: str) -> dict:
    return dict(re.findall(r"^([A-D])\) (.*)$", question, re.M))


class TestTemplates(unittest.TestCase):
    def test_letters_are_uniform(self):
        sc = scene(TIDY)
        for name, k in (("relation_embspatial", 4), ("relation_blink", 2), ("close_embspatial", 4),
                        ("far_embspatial", 4), ("pointing_erqa", 4), ("localization_blink", 2),
                        ("counting_blink", 4)):
            c = letters_of(name, sc)
            total = sum(c.values())
            self.assertGreater(total, 1000, name)
            self.assertEqual(set(c), set(F.LETTERS[:k]), name)
            for letter, v in c.items():
                self.assertAlmostEqual(v / total, 1 / k, delta=0.05, msg=f"{name} {letter}")

    def test_embspatial_distractors_are_false(self):
        sc = scene(TIDY)
        name_box = {o.name: o.box for o in sc.subjects}
        for seed in range(300):
            for it in T.relation_embspatial(sc, np.random.default_rng(seed), {}):
                stem = it.question.splitlines()[1]
                a, b = sorted((n for n in name_box if f" {n} " in stem), key=stem.index)
                true = relations(name_box[a], name_box[b], 0.02 * W)
                for letter, text in options_of(it.question).items():
                    said = {r for r, phrases in T.EMB_RELATION.items() if any(f" is {p} the" in text for p in phrases)}
                    if letter == it.answer:
                        self.assertTrue(said and said <= true, text)
                    else:
                        self.assertFalse(said & true, text)

    def test_blink_relation_truth(self):
        sc = scene(TIDY)
        name_box = {o.name: o.box for o in sc.subjects}
        inverse = {p: r for r, ps in T.BLINK_RELATION.items() for p in ps}
        for seed in range(300):
            for it in T.relation_blink(sc, np.random.default_rng(seed), {}):
                a, phrase, b = re.search(r"Is the (\w+) (.+) the (\w+)\?", it.question).groups()
                holds = inverse[phrase] in relations(name_box[a], name_box[b], 0.02 * W)
                self.assertEqual(it.answer, "A" if holds else "B")

    def test_close_and_far_pick_the_extremes(self):
        sc = scene(TIDY)
        for seed in range(50):
            for it in T.close_embspatial(sc, np.random.default_rng(seed), {}):
                self.assertEqual(options_of(it.question)[it.answer], "spoon")
            for it in T.far_embspatial(sc, np.random.default_rng(seed), {}):
                self.assertEqual(options_of(it.question)[it.answer], "carrot")

    def test_depth_margin(self):
        close = [raw("carrot", CARROT, disp=0.2), raw("bowl", BOWL, disp=0.4), raw("sponge", SPONGE, disp=0.85),
                 raw("spoon", SPOON, disp=0.9)]
        self.assertEqual(T.close_embspatial(scene(close), np.random.default_rng(0), {"margin": 0.1}), [])
        self.assertEqual(len(T.far_embspatial(scene(close), np.random.default_rng(0), {"margin": 0.1})), 1)

    def test_pointing_dots(self):
        sc = scene(TIDY)
        for seed in range(100):
            for it in T.pointing_erqa(sc, np.random.default_rng(seed), {}):
                target = next(o for o in sc.subjects if f"on the {o.name}" in it.question)
                dots = it.overlays
                if dots[0]["label"]:  # letters: the answer is the correct dot's label
                    correct = next(d for d in dots if d["label"] == it.answer)
                else:
                    color = re.search(rf"{it.answer}\. (\w+) dot\.", it.question).group(1)
                    correct = next(d for d in dots if tuple(d["color"]) == T.DOT_COLORS[color])
                x, y = correct["xy"]
                self.assertTrue(target.mask[y, x])
                for d in dots:
                    if d is not correct:
                        x, y = d["xy"]
                        self.assertFalse(target.box[0] <= x <= target.box[2] and target.box[1] <= y <= target.box[3])
                pts = np.array([d["xy"] for d in dots], float)
                dist = np.linalg.norm(pts[:, None] - pts[None], axis=-1) + np.eye(4) * 1e9
                self.assertGreaterEqual(dist.min(), 0.08 * W - 1e-6)

    def test_localization_boxes(self):
        sc = scene(TIDY)
        for seed in range(100):
            for it in T.localization_blink(sc, np.random.default_rng(seed), {}):
                boxes = {o["label"]: o["xyxy"] for o in it.overlays}
                target = next(o for o in sc.subjects if f"encloses the {o.name}?" in it.question)
                self.assertEqual(boxes[it.answer], target.box)
                other = boxes["B" if it.answer == "A" else "A"]
                self.assertTrue(0.15 <= D.overlaps(other, target.box)[0] <= 0.45)

    def test_existence_and_counting(self):
        sc = scene(TIDY + [raw("pot", (200, 250, 300, 350)), raw("pot", (350, 250, 450, 350))], negatives=["piano"])
        for seed in range(50):
            for it in T.existence_msmu(sc, np.random.default_rng(seed), {"p_present": 0.0}):
                self.assertIn("piano", it.question)
                self.assertIn(it.answer, ("No piano in the scene.", "Can not find piano."))
            for it in T.counting_msmu(sc, np.random.default_rng(seed), {}):
                n = next(k for k in sc.counts if plural(k) in it.question)
                self.assertEqual(re.findall(r"\d+", it.answer), [str(sc.counts[n])])
            for it in T.counting_blink(sc, np.random.default_rng(seed), {}):
                n = next(k for k in sc.counts if plural(k) in it.question)
                opts = dict(re.findall(r"^\(([A-D])\) (\d+)$", it.question, re.M))
                self.assertEqual(opts[it.answer], str(sc.counts[n]))
                self.assertEqual(len(set(opts.values())), 4)

    def test_no_subjects_no_questions(self):
        sc = scene([raw("pot", (200, 250, 300, 350)), raw("pot", (350, 250, 450, 350))])
        items = T.make_items(sc, np.random.default_rng(0), {n: {} for n in T.TEMPLATES})
        self.assertEqual({it.template for it in items}, {"counting_blink", "counting_msmu"})


class TestFormats(unittest.TestCase):
    def test_erqa(self):
        q = F.erqa("Which one?", ["red dot", "blue dot"])
        self.assertTrue(q.startswith("Which one? Choices: A. red dot. B. blue dot. Please answer directly"))
        self.assertTrue(q.endswith("Your answer must be one of A, B, C, or D."))

    def test_embspatial_and_blink(self):
        self.assertEqual(F.embspatial("Q?", ["x", "y", "z", "w"]),
                         "Note: You only need to respond with A, B, C, D without providing any additional "
                         "information.\nQ?\nA) x\nB) y\nC) z\nD) w")
        self.assertEqual(F.blink("Q?", ["yes", "no"]),
                         "Note: You only need to respond with A, B without providing any additional information."
                         "\nQ?\nSelect from the following choices.\n(A) yes\n(B) no")


class TestCache(unittest.TestCase):
    def test_roundtrip(self):
        det = detections(TIDY + [D.RawBox("piano", [0, 0, 10, 10], 0.06)], negatives=["piano"])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.npz"
            D.save(path, det)
            back = D.load(path)
        self.assertEqual([b.name for b in back.boxes], [b.name for b in det.boxes])
        for a, b in zip(det.boxes, back.boxes):
            self.assertEqual(a.box, b.box)
            self.assertEqual(a.disp, b.disp)
            self.assertEqual(a.mask is None, b.mask is None)
            if a.mask is not None:
                self.assertTrue(np.array_equal(a.mask, b.mask))
        self.assertEqual(back.presence, det.presence)
        self.assertEqual(back.negatives, ["piano"])


if __name__ == "__main__":
    unittest.main()
