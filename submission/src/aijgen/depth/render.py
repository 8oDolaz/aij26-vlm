"""Markers and prompts for point-pair depth questions.

`ring` reproduces BLINK Relative_Depth: a small hollow red circle per point and a white letter in a black box
next to it (above by default). `dot` is a variation (filled coloured dot, outlined letter) so the model does not
tie the skill to one marker style; dots are only used with prompts that do not say "circled".

Point sets (3-4 points) are asked two ways: `set_letter` labels the markers A, B, C, D and lists them as BLINK-style
options, `set_numbered` labels them 1..k and lists them under shuffled option letters in one of the benchmarks' option
formats, so the answer letter is not the marker's label.
"""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
RED = (255, 0, 0)
DOT_COLORS = [(255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

# lmms-eval blink: pre_prompt.format("A, B") + doc["prompt"]; the answer is parsed as a single letter.
PRE_PROMPT = "Note: You only need to respond with {} without providing any additional information.\n"


def note(letters) -> str:
    """lmms-eval BLINK's pre_prompt for these choice letters (EmbSpatial uses the same wording)."""
    return PRE_PROMPT.format(", ".join(letters))


NOTE = note("AB")
BLINK_CLOSER = ("Two points are circled on the image, labeled by A and B beside each circle. Which point is closer "
                "to the camera?\nSelect from the following choices.\n(A) A is closer\n(B) B is closer")
BLINK_FARTHER = ("Two points are circled on the image, labeled by A and B beside each circle. Which point is "
                 "farther from the camera?\nSelect from the following choices.\n(A) A is farther\n(B) B is farther")
FREE = [  # (question, answer); {n} = closer letter, {f} = farther letter
    ("Which of the two marked points, A or B, is closer to the camera?", "Point {n} is closer to the camera than point {f}."),
    ("Which marked point is farther away from the camera, A or B?", "Point {f} is farther from the camera."),
    ("Look at the points marked A and B. Which one is nearer to the viewer?", "{n}"),
    ("Is point A closer to the camera than point B? Answer yes or no.", None),  # yes/no, filled in below
    ("From the camera's point of view, which point lies deeper in the scene: A or B?", "Point {f} is deeper in the scene, so point {n} is closer."),
]


def resize_width(img: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    s = width / img.shape[1]
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (width, round(img.shape[0] * s)), interpolation=interp), s


def _box_ok(box, others, r, w, h):
    x0, y0, x1, y1 = box
    if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
        return False
    # keep clear of the other points' markers
    return not any(x0 - r - 2 <= ox <= x1 + r + 2 and y0 - r - 2 <= oy <= y1 + r + 2 for ox, oy in others)


def draw(img: np.ndarray, pts: dict[str, tuple[int, int]], style: str, color=RED) -> np.ndarray:
    """Copy of img (RGB) with points {letter: (x, y)} marked in the given style."""
    img = img.copy()
    h, w = img.shape[:2]
    r = max(4, round(0.011 * w))
    scale = 0.45 * w / 512
    for letter, (x, y) in pts.items():
        if style == "ring":
            cv2.circle(img, (x, y), r, color, max(1, round(r / 3)), cv2.LINE_AA)
        else:
            cv2.circle(img, (x, y), r, color, -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), r, (0, 0, 0), 1, cv2.LINE_AA)
    for letter, (x, y) in pts.items():
        others = [p for k, p in pts.items() if k != letter]
        (tw, th), _ = cv2.getTextSize(letter, FONT, scale, 1)
        bw, bh = tw + 6, th + 6
        gap = r + 4
        candidates = [(x - bw // 2, y - gap - bh), (x - bw // 2, y + gap), (x + gap, y - bh // 2), (x - gap - bw, y - bh // 2)]
        x0, y0 = next(((cx, cy) for cx, cy in candidates if _box_ok((cx, cy, cx + bw, cy + bh), others, r, w, h)),
                      candidates[0])
        x0, y0 = min(max(x0, 0), w - bw - 1), min(max(y0, 0), h - bh - 1)
        org = (x0 + 3, y0 + bh - 3)
        if style == "ring":
            cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (0, 0, 0), -1)
            cv2.putText(img, letter, org, FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, letter, org, FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, letter, org, FONT, scale, color, 1, cv2.LINE_AA)
    return img


def question(near_letter: str, style: str, cfg: dict, rng: np.random.Generator) -> tuple[str, str, str]:
    """(prompt, answer, kind). Ring style is mostly BLINK's exact prompt; dots only get free-form prompts."""
    far_letter = "B" if near_letter == "A" else "A"
    u = rng.random()
    if style == "ring" and u < cfg["p_blink"]:
        return NOTE + BLINK_CLOSER, near_letter, "blink_closer"
    if style == "ring" and u < cfg["p_blink"] + cfg["p_blink_farther"]:
        return NOTE + BLINK_FARTHER, far_letter, "blink_farther"
    q, a = FREE[rng.integers(len(FREE))]
    if a is None:
        a = "Yes." if near_letter == "A" else "No."
    return q, a.format(n=near_letter, f=far_letter), "free"


COUNT = {3: "Three", 4: "Four"}
SET_STEMS = {  # set_numbered questions, after EmbSpatial's close/far stems (docs/relative_depth_plan.md, section 3)
    "closest": ["Which of the marked points is closest to the camera?",
                "Among the numbered points, which one is nearest to the viewer?",
                "Which marked point is at the shortest distance from the camera?",
                "From the camera's point of view, which numbered point is the closest?"],
    "farthest": ["Which of the marked points is farthest from the camera?",
                 "Among the numbered points, which one is the most distant from the viewer?",
                 "Which marked point lies deepest in the scene?",
                 "From the camera's point of view, which numbered point is the farthest away?"],
}
# Option line formats of the target benches: BLINK "(A) x", EmbSpatial "A) x", MMMU "A. x" (+ its answer suffix).
OPTION_FORMATS = ("blink", "embspatial", "mmmu")
MMMU_SUFFIX = "\nAnswer with the option's letter from the given choices directly."


def _and(items: list[str]) -> str:
    return ", ".join(items[:-1]) + " and " + items[-1]


def set_question(k: int, slot: int, opt_slot: int, relation: str, style: str, p_numbered: float,
                 rng: np.random.Generator) -> tuple[str, str, str, list[str]]:
    """(prompt, answer letter, kind, marker labels in marker order) for a k-point set whose target is marker `slot`.

    set_letter: markers A..k, BLINK's wording and option list, answer = the target's marker letter.
    set_numbered: markers 1..k, options A..k list the points in a shuffled order with the target at `opt_slot`,
    answer = that option's letter.
    """
    letters = [chr(65 + i) for i in range(k)]
    word = "closest to" if relation == "closest" else "farthest from"
    verb = "circled" if style == "ring" else "marked"
    if rng.random() >= p_numbered:
        prompt = (note(letters) + f"{COUNT[k]} points are {verb} on the image, labeled by {_and(letters)} beside each "
                  f"{'circle' if style == 'ring' else 'point'}. Which point is {word} the camera?\n"
                  "Select from the following choices.\n"
                  + "\n".join(f"({L}) {L} is {relation}" for L in letters))
        return prompt, letters[slot], "set_letter", letters
    numbers = [str(i + 1) for i in range(k)]
    rest = [n for i, n in enumerate(numbers) if i != slot]
    rest = [rest[i] for i in rng.permutation(k - 1)]
    order = rest[:opt_slot] + [numbers[slot]] + rest[opt_slot:]  # point number listed under each option letter
    fmt = OPTION_FORMATS[rng.integers(len(OPTION_FORMATS))]
    line = {"blink": "({}) point {}", "embspatial": "{}) point {}", "mmmu": "{}. point {}"}[fmt]
    stem = SET_STEMS[relation][rng.integers(len(SET_STEMS[relation]))]
    body = (f"{COUNT[k]} points are {verb} on the image and labeled {_and(numbers)}. {stem}\n"
            + "\n".join(line.format(L, n) for L, n in zip(letters, order)))
    prompt = body + MMMU_SUFFIX if fmt == "mmmu" else note(letters) + body
    return prompt, letters[opt_slot], f"set_numbered_{fmt}", numbers
