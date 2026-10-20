"""Image processing, robot arm suppression, segment extraction, and spatial relation utilities."""
import colorsys

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
DEFAULT_COLORS = [
    (255, 99, 71),    # tomato
    (30, 144, 255),   # dodger blue
    (50, 205, 50),    # lime green
    (255, 215, 0),    # gold
    (147, 112, 219),  # medium purple
    (255, 105, 180),  # hot pink
    (0, 206, 209),    # dark turquoise
    (255, 140, 0),    # dark orange
]


def extract_segments(labels: np.ndarray, min_area_px: int = 50) -> list[dict]:
    """Extracts bounding boxes, centroids, areas, and masks for each unique segment ID in labels.

    labels: (H, W) int array where 0 is unmasked and positive integers represent instance masks.
    """
    h, w = labels.shape[:2]
    total_px = float(h * w)
    segments = []
    unique_ids = [int(i) for i in np.unique(labels) if i > 0]

    for seg_id in sorted(unique_ids):
        mask = labels == seg_id
        area_px = int(mask.sum())
        if area_px < min_area_px:
            continue

        ys, xs = np.where(mask)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())

        m = cv2.moments(mask.astype(np.uint8))
        if m["m00"] > 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
        else:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

        segments.append({
            "seg_id": seg_id,
            "box": [x1, y1, x2, y2],
            "centroid": (cx, cy),
            "area_px": area_px,
            "area_pct": round(100.0 * area_px / total_px, 2),
            "mask": mask,
        })
    return segments


def compute_robot_arm_mask(
    frame: np.ndarray,
    v_thresh: int = 85,
    top_margin: int = 15,
    dilate_px: int = 2,
) -> np.ndarray:
    """Detects the robot arm manipulator via dark metal appearance and top/edge connectivity.

    Returns:
        binary boolean mask (H, W) where True indicates the robot arm.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    # Dark metallic material: low brightness, low saturation
    dark = (v_channel < v_thresh) & (s_channel < 120)

    # Exclude colored or bright objects (e.g. purple eggplant, red handle, green knob)
    colored_or_bright = (s_channel > 95) & (v_channel > 65)
    dark &= (~colored_or_bright)

    # Morphological close to bridge joints and reflections
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close)

    # Connected components
    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    arm_mask = np.zeros((h, w), dtype=bool)
    for lbl in range(1, num_labels):
        x, y, comp_w, comp_h, area = stats[lbl]
        # Must connect to top margin or top side edge and have sufficient size
        if (y <= top_margin or x <= 5 or (x + comp_w) >= w - 5) and comp_h >= 25 and area >= 120:
            arm_mask |= (labels_im == lbl)

    if dilate_px > 0 and arm_mask.any():
        kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        arm_mask = cv2.dilate(arm_mask.astype(np.uint8), kernel_dil).astype(bool)
        arm_mask &= (~colored_or_bright)

    return arm_mask


def filter_robot_arm_segments(
    segments: list[dict],
    arm_mask: np.ndarray,
    min_overlap: float = 0.30,
) -> list[dict]:
    """Filters out and suppresses any segments or detections that fall on the robot arm or gripper."""
    if not arm_mask.any():
        return segments

    kept = []
    robot_names = {"[robot_arm]", "robot_arm", "arm", "gripper", "manipulator", "claw", "robot_hand"}

    for s in segments:
        label = str(s.get("label", "")).lower()
        if label in robot_names or s.get("is_robot_arm"):
            continue

        if s.get("mask") is not None:
            inter = (s["mask"] & arm_mask).sum()
            area = max(1, s["area_px"])
            overlap = inter / area
        else:
            bx1, by1, bx2, by2 = [int(v) for v in s["box"]]
            crop = arm_mask[by1:by2, bx1:bx2]
            overlap = crop.sum() / max(1, (bx2 - bx1) * (by2 - by1))

        if overlap < min_overlap:
            kept.append(s)

    return kept


def get_pairwise_spatial_relations(
    segments: list[dict],
    shape: tuple[int, int],
) -> list[dict]:
    """Computes structured qualitative 2D spatial relationships between pairs of grounded objects.

    Returns:
        List of dicts containing:
          - 'obj_a': str
          - 'obj_b': str
          - 'primary_rel': str (e.g. 'to the left of', 'to the right of', 'above', 'below', 'inside')
          - 'distractor_rels': list[str] (mutually exclusive alternative directions for MCQ)
          - 'h_rel': str
          - 'v_rel': str
          - 'larger_obj': str
          - 'smaller_obj': str
          - 'box_a': list[int]
          - 'box_b': list[int]
    """
    grounded = [
        s for s in segments
        if s.get("label") and not str(s.get("label")).startswith("#") and s.get("label") != "__distractor__"
    ]
    if len(grounded) < 2:
        return []

    h, w = shape[:2]
    pairs = []

    for i in range(len(grounded)):
        for j in range(len(grounded)):
            if i == j:
                continue
            a, b = grounded[i], grounded[j]
            if a["label"] == b["label"]:
                continue

            ax, ay = a.get("centroid", ((a["box"][0] + a["box"][2]) // 2, (a["box"][1] + a["box"][3]) // 2))
            bx, by = b.get("centroid", ((b["box"][0] + b["box"][2]) // 2, (b["box"][1] + b["box"][3]) // 2))
            dx, dy = ax - bx, ay - by

            # Check containment / inside
            abox, bbox = a["box"], b["box"]
            a_in_b = (bbox[0] <= abox[0] and bbox[1] <= abox[1] and bbox[2] >= abox[2] and bbox[3] >= abox[3])

            if a_in_b:
                primary_rel = "inside"
                distractor_rels = ["to the left of", "to the right of", "above", "below"]
            else:
                # Determine primary direction based on largest offset
                norm_dx = abs(dx) / float(w)
                norm_dy = abs(dy) / float(h)

                if norm_dx >= norm_dy:
                    primary_rel = "to the left of" if dx < 0 else "to the right of"
                else:
                    primary_rel = "above" if dy < 0 else "below"

                all_dirs = ["to the left of", "to the right of", "above", "below"]
                distractor_rels = [d for d in all_dirs if d != primary_rel]

            h_rel = "to the left of" if dx < -0.05 * w else ("to the right of" if dx > 0.05 * w else "horizontally aligned with")
            v_rel = "above" if dy < -0.05 * h else ("below" if dy > 0.05 * h else "vertically aligned with")

            # Size comparison
            area_a = a.get("area_px", (abox[2] - abox[0]) * (abox[3] - abox[1]))
            area_b = b.get("area_px", (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

            if area_a > 1.30 * area_b:
                larger_obj, smaller_obj = a["label"], b["label"]
            elif area_b > 1.30 * area_a:
                larger_obj, smaller_obj = b["label"], a["label"]
            else:
                larger_obj, smaller_obj = None, None

            pairs.append({
                "obj_a": a["label"],
                "obj_b": b["label"],
                "primary_rel": primary_rel,
                "distractor_rels": distractor_rels,
                "h_rel": h_rel,
                "v_rel": v_rel,
                "larger_obj": larger_obj,
                "smaller_obj": smaller_obj,
                "box_a": a["box"],
                "box_b": b["box"],
            })

    return pairs


def compute_spatial_relations(
    segments: list[dict], shape: tuple[int, int]
) -> list[str]:
    """Computes qualitative 2D spatial relationship strings between pairs of grounded objects."""
    pairs = get_pairwise_spatial_relations(segments, shape)
    relations = []
    seen = set()

    for p in pairs:
        key = tuple(sorted([p["obj_a"], p["obj_b"]]))
        if key in seen:
            continue
        seen.add(key)

        if p["primary_rel"] == "inside":
            relations.append(f"'{p['obj_a']}' is inside '{p['obj_b']}'")
        else:
            relations.append(f"'{p['obj_a']}' is {p['primary_rel']} '{p['obj_b']}'")

    return relations


def get_distinct_colors(n: int) -> list[tuple[int, int, int]]:
    """Generates n visually distinct RGB colors using golden ratio HSV spacing."""
    colors = []
    golden_ratio = 0.618033988749895
    h = 0.15
    for _ in range(n):
        h = (h + golden_ratio) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def draw_sam_segmentation(
    frame: np.ndarray,
    labels: np.ndarray,
    segments: list[dict] | None = None,
    show_boxes: bool = True,
    show_labels: bool = True,
    alpha: float = 0.45,
    show_bg_labels: bool = False,
) -> np.ndarray:
    """Overlays SAM instance masks, bounding boxes, and high-contrast semantic tags."""
    h, w = frame.shape[:2]
    canvas = frame.copy()
    overlay = frame.copy()

    unique_ids = [int(i) for i in np.unique(labels) if i > 0]
    seg_by_id = {s["seg_id"]: s for s in segments} if segments else {}
    has_semantic = any(s.get("label") and not str(s.get("label")).startswith("#") for s in (segments or []))
    palette = get_distinct_colors(max(len(unique_ids) + 5, 16))

    for seg_id in unique_ids:
        mask = labels == seg_id
        color = palette[seg_id % len(palette)]
        overlay[mask] = color

    blended = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)

    for seg_id in unique_ids:
        mask = labels == seg_id
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = palette[seg_id % len(palette)]
        cv2.drawContours(blended, contours, -1, color, 1, cv2.LINE_AA)

    if show_boxes or show_labels:
        for seg_id in unique_ids:
            mask = labels == seg_id
            area_px = int(mask.sum())
            if area_px < 60:
                continue

            ys, xs = np.where(mask)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            color = palette[seg_id % len(palette)]

            meta = seg_by_id.get(seg_id, {})
            custom_label = meta.get("label")
            is_semantic = custom_label and not str(custom_label).startswith("#")

            if show_boxes and is_semantic:
                cv2.rectangle(blended, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            if show_labels and is_semantic:
                score = meta.get("score")
                score_str = f" {score:.2f}" if score is not None else ""
                tag_text = f" {str(custom_label).upper()}{score_str} "

                scale = max(0.42, 0.48 * w / 640.0)
                (tw, th), _ = cv2.getTextSize(tag_text, FONT, scale, 1)

                tx1 = max(2, min(x1, w - tw - 8))
                ty2 = max(th + 6, y1 - 4) if y1 - th - 8 > 2 else min(h - 4, y1 + th + 10)
                ty1 = ty2 - th - 6
                tx2 = tx1 + tw + 4

                cv2.rectangle(blended, (tx1, ty1), (tx2, ty2), (18, 18, 18), -1)
                cv2.rectangle(blended, (tx1, ty1), (tx2, ty2), color, 2, cv2.LINE_AA)
                cv2.putText(blended, tag_text, (tx1 + 3, ty2 - 5), FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)

    return blended


def create_sam_comparison(
    frame: np.ndarray,
    labels: np.ndarray,
    segments: list[dict] | None = None,
    show_boxes: bool = True,
    show_bg_labels: bool = False,
) -> np.ndarray:
    """Creates a side-by-side image: [Original Frame | SAM Segmentation]."""
    vis = draw_sam_segmentation(
        frame,
        labels,
        segments=segments,
        show_boxes=show_boxes,
        show_bg_labels=show_bg_labels,
    )
    divider = np.full((frame.shape[0], 4, 3), 40, dtype=np.uint8)
    return np.hstack([frame, divider, vis])


def draw_detections(frame: np.ndarray, dets: list[dict], alpha: float = 0.35) -> np.ndarray:
    """Overlays bounding boxes and labels on an RGB frame."""
    canvas = frame.copy()
    overlay = frame.copy()
    h, w = frame.shape[:2]

    for idx, d in enumerate(dets):
        color = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        box = d["box"]
        x1, y1, x2, y2 = map(int, map(round, box))
        label = d.get("label", "")

        if "mask" in d and d["mask"] is not None:
            overlay[d["mask"]] = color

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    blended = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)
    for idx, d in enumerate(dets):
        color = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        box = d["box"]
        x1, y1, x2, y2 = map(int, map(round, box))
        label = d.get("label", "")
        scale = max(0.4, 0.45 * w / 640.0)
        (tw, th), _ = cv2.getTextSize(label, FONT, scale, 1)
        ty0 = max(y1 - th - 6, 0)
        cv2.rectangle(blended, (x1, ty0), (x1 + tw + 6, ty0 + th + 6), (0, 0, 0), -1)
        cv2.putText(blended, label, (x1 + 3, ty0 + th + 2), FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)

    return blended
