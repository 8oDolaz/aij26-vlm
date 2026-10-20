"""Markers and prompts for point-pair depth questions.

`ring` reproduces BLINK Relative_Depth: a small hollow red circle per point and a white letter in a black box
next to it (above by default). `dot` is a variation (filled coloured dot, outlined letter) so the model does not
tie the skill to one marker style; dots are only used with prompts that do not say "circled".
"""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
RED = (255, 0, 0)
DOT_COLORS = [(255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

# lmms-eval blink: pre_prompt.format("A, B") + doc["prompt"]; the answer is parsed as a single letter.
NOTE = "Note: You only need to respond with A, B without providing any additional information.\n"
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


def _box_ok(box, other, r, w, h):
    x0, y0, x1, y1 = box
    if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
        return False
    ox, oy = other  # keep clear of the other point's ring
    return not (x0 - r - 2 <= ox <= x1 + r + 2 and y0 - r - 2 <= oy <= y1 + r + 2)


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
        other = next(p for k, p in pts.items() if k != letter)
        (tw, th), _ = cv2.getTextSize(letter, FONT, scale, 1)
        bw, bh = tw + 6, th + 6
        gap = r + 4
        candidates = [(x - bw // 2, y - gap - bh), (x - bw // 2, y + gap), (x + gap, y - bh // 2), (x - gap - bw, y - bh // 2)]
        x0, y0 = next(((cx, cy) for cx, cy in candidates if _box_ok((cx, cy, cx + bw, cy + bh), other, r, w, h)),
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
