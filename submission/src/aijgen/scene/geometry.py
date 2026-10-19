"""Image-plane geometry for scene questions: box relations, mask erosion, marker placement, perturbed boxes.

Relations follow EmbSpatial's definition (docs/vlm_benchmark_categories.html): A is left of B when A's whole bounding
box lies to the left of B's, as the viewer sees it; above/under the same way vertically. A small gap is required so
touching boxes count as neither.
"""
import numpy as np

from aijgen.scene.detect import overlaps

RELATIONS = ("left", "right", "above", "under")
OPPOSITE = {"left": "right", "right": "left", "above": "under", "under": "above"}


def relations(a, b, gap: float) -> set[str]:
    """Relations of box a to box b (xyxy pixels) that hold with at least `gap` pixels between the boxes."""
    out = set()
    if a[2] + gap <= b[0]:
        out.add("left")
    if a[0] >= b[2] + gap:
        out.add("right")
    if a[3] + gap <= b[1]:
        out.add("above")
    if a[1] >= b[3] + gap:
        out.add("under")
    return out


def erode(mask: np.ndarray, r: int) -> np.ndarray:
    """Binary erosion with a (2r+1)^2 square, via an integral image (no cv2). Pixels near the border are eroded too."""
    if r <= 0:
        return mask.copy()
    m = np.pad(mask.astype(np.int32), r + 1)
    ii = m.cumsum(0).cumsum(1)
    k = 2 * r + 1
    h, w = mask.shape
    # sum of the window centred on (y, x) of the original image
    s = (ii[k:k + h, k:k + w] - ii[0:h, k:k + w] - ii[k:k + h, 0:w] + ii[0:h, 0:w])
    return s == k * k


def inside_border(xy, size: tuple[int, int], border: float) -> bool:
    h, w = size
    x, y = xy
    return border * w <= x < (1 - border) * w and border * h <= y < (1 - border) * h


def in_box(xy, box, pad: float = 0.0) -> bool:
    x, y = xy
    return box[0] - pad <= x <= box[2] + pad and box[1] - pad <= y <= box[3] + pad


def far_enough(xy, others, min_dist: float) -> bool:
    return all((xy[0] - o[0]) ** 2 + (xy[1] - o[1]) ** 2 >= min_dist ** 2 for o in others)


def sample_in_mask(mask: np.ndarray, rng: np.random.Generator, ok=lambda xy: True, tries: int = 50):
    """A random pixel (x, y) of mask that passes ok(), or None."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    for i in rng.integers(len(ys), size=min(tries, len(ys))):
        xy = (int(xs[i]), int(ys[i]))
        if ok(xy):
            return xy
    return None


def sample_background(size: tuple[int, int], blocked: list, rng: np.random.Generator, ok=lambda xy: True,
                      tries: int = 200):
    """A random pixel (x, y) outside every box in `blocked` (xyxy, already padded) that passes ok(), or None."""
    h, w = size
    for _ in range(tries):
        xy = (int(rng.integers(w)), int(rng.integers(h)))
        if not any(in_box(xy, b) for b in blocked) and ok(xy):
            return xy
    return None


def perturbed_box(box, size: tuple[int, int], rng: np.random.Generator, iou_range: tuple[float, float],
                  max_shift: float = 0.8, tries: int = 200):
    """A box overlapping `box` with IoU inside iou_range (BLINK Object Localization's noisy box): each side moved by up
    to max_shift of the box size, kept inside the image and at least 8 px on each side. None if no try lands in the
    range."""
    h, w = size
    bw, bh = box[2] - box[0], box[3] - box[1]
    for _ in range(tries):
        d = rng.uniform(-max_shift, max_shift, 4) * np.array([bw, bh, bw, bh])
        x0, y0, x1, y1 = (float(np.clip(v, 0, lim)) for v, lim in
                          zip(np.array(box, float) + d, (w - 1, h - 1, w - 1, h - 1)))
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        cand = [x0, y0, x1, y1]
        if iou_range[0] <= overlaps(cand, box)[0] <= iou_range[1]:
            return cand
    return None
