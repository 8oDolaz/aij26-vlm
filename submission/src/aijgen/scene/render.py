"""Draw question markers (overlay specs from aijgen.scene.templates) on an RGB frame. Sizes scale with the width.

  {"kind": "dot", "xy": [x, y], "color": [r, g, b], "label": "A" or None}   filled dot, black outline, optional letter
  {"kind": "box", "xyxy": [x0, y0, x1, y1], "color": [r, g, b], "label": "A"}  box with its letter in a filled tag
"""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(img, text, x, y, scale, fg, bg):
    """Text in a filled box whose top-left corner is near (x, y), kept inside the image."""
    h, w = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    bw, bh = tw + 6, th + 6
    x0, y0 = int(min(max(x, 0), w - bw - 1)), int(min(max(y, 0), h - bh - 1))
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), bg, -1)
    cv2.putText(img, text, (x0 + 3, y0 + bh - 3), FONT, scale, fg, 1, cv2.LINE_AA)


def draw_overlays(rgb: np.ndarray, overlays: list[dict]) -> np.ndarray:
    img = rgb.copy()
    w = img.shape[1]
    r = max(4, round(0.014 * w))
    scale = 0.5 * w / 640
    thick = max(2, round(0.004 * w))
    for o in overlays:
        color = tuple(int(c) for c in o["color"])
        if o["kind"] == "dot":
            x, y = (int(v) for v in o["xy"])
            cv2.circle(img, (x, y), r, color, -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), r, (0, 0, 0), 1, cv2.LINE_AA)
            if o.get("label"):
                _label(img, o["label"], x + r + 2, y - r - 2 - round(18 * scale), scale, (255, 255, 255), (0, 0, 0))
        elif o["kind"] == "box":
            x0, y0, x1, y1 = (int(round(v)) for v in o["xyxy"])
            cv2.rectangle(img, (x0, y0), (x1, y1), color, thick, cv2.LINE_AA)
            if o.get("label"):
                fg = (0, 0, 0) if sum(color) > 450 else (255, 255, 255)  # dark letter on yellow
                _label(img, o["label"], x0, y0 - round(22 * scale), scale, fg, color)
        else:
            raise ValueError(f"unknown overlay kind {o['kind']!r}")
    return img
