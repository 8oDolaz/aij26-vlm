#!/usr/bin/env python3
"""Run SAM 2.1 instance segmentation probe and zero-shot grounding on images, videos, or LeRobot episodes.

Usage:
  # 1. Single frame from a LeRobot dataset episode:
  python tools/sam_probe.py --dataset data/raw/bridge --episode 13612 --cam image_0 --frame 20 --out runs/sam_probe/

  # 2. Entire video / episode:
  python tools/sam_probe.py --dataset data/raw/bridge --episode 13612 --cam image_0 --video --stride 2 --out runs/sam_probe/

  # 3. On a standalone image:
  python tools/sam_probe.py vendor/participant/examples/smoke/images/00000001.jpg --out runs/sam_probe/

  # 4. On a standalone video file:
  python tools/sam_probe.py path/to/video.mp4 --video --candidates "eggplant,burners" --out runs/sam_probe/
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "submission" / "src"))

from aijgen.depth.models import Segmenter
from aijgen.objects.crop_classifier import CropSegmentClassifier
from aijgen.objects.scene_vocabulary import (
    build_scene_queries,
    load_vocabulary_from_file,
    parse_task_objects,
)
from aijgen.objects.processing import (
    annotate_robot_arm_segments,
    compute_robot_arm_mask,
    compute_spatial_relations,
    create_sam_comparison,
    draw_sam_segmentation,
    extract_segments,
    filter_robot_arm_segments,
)


def load_frame_from_file(source: Path, timestamp: float | None = None) -> np.ndarray:
    """Loads an RGB frame from an image file or a video at a specific timestamp (seconds)."""
    if source.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
        bgr = cv2.imread(str(source))
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {source}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    import av

    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        if timestamp is not None:
            container.seek(int(timestamp / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            return frame.to_ndarray(format="rgb24")
    raise RuntimeError(f"No frames decoded from {source}")


def load_video_file_frames(
    source: Path, stride: int = 1, max_frames: int | None = None
) -> tuple[list[tuple[int, float, np.ndarray]], float]:
    """Decodes consecutive frames from a standalone video file."""
    import av

    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate or 10.0)
        frames_list = []
        for idx, frame in enumerate(container.decode(stream)):
            if idx % max(1, stride) == 0:
                t_sec = float(frame.time) if frame.time is not None else idx / fps
                frames_list.append((idx, t_sec, frame.to_ndarray(format="rgb24")))
                if max_frames and len(frames_list) >= max_frames:
                    break
    if not frames_list:
        raise RuntimeError(f"No frames decoded from video {source}")
    return frames_list, fps


def load_frame_from_dataset(root: Path, episode_idx: int, cam_name: str, frame_idx: int) -> tuple[np.ndarray, str]:
    """Loads a single frame directly from a LeRobot dataset structure via aijgen.depth.lerobot."""
    from aijgen.depth.lerobot import list_episodes, read_frames

    cam_key = cam_name if cam_name.startswith("observation.images.") else f"observation.images.{cam_name}"
    eps = list_episodes(root, cam_key)
    if not eps:
        raise RuntimeError(f"No episodes found for camera '{cam_key}' in {root}")

    ep = next((e for e in eps if e.index == episode_idx), None)
    if ep is None:
        avail = [e.index for e in eps[:10]]
        raise ValueError(f"Episode {episode_idx} not found for {cam_key}. First available indices: {avail}...")

    frames = read_frames(ep, [frame_idx])
    if frame_idx not in frames:
        raise ValueError(f"Frame {frame_idx} could not be decoded (episode length: {ep.length})")

    return frames[frame_idx], ep.task


def load_episode_frames(
    root: Path, episode_idx: int, cam_name: str, stride: int = 1, max_frames: int | None = None
) -> tuple[list[tuple[int, float, np.ndarray]], str, float]:
    """Loads a sequence of frames of an episode from a LeRobot dataset."""
    from aijgen.depth.lerobot import list_episodes, read_frames

    cam_key = cam_name if cam_name.startswith("observation.images.") else f"observation.images.{cam_name}"
    eps = list_episodes(root, cam_key)
    if not eps:
        raise RuntimeError(f"No episodes found for camera '{cam_key}' in {root}")

    ep = next((e for e in eps if e.index == episode_idx), None)
    if ep is None:
        avail = [e.index for e in eps[:10]]
        raise ValueError(f"Episode {episode_idx} not found for {cam_key}. First available indices: {avail}...")

    indices = list(range(0, ep.length, max(1, stride)))
    if max_frames:
        indices = indices[:max_frames]

    print(f"Reading {len(indices)} frames from episode {episode_idx} (total len: {ep.length}, fps: {ep.fps:.1f})...")
    frames_dict = read_frames(ep, indices)
    frames_list = []
    for idx in indices:
        if idx in frames_dict:
            t_sec = idx / ep.fps
            frames_list.append((idx, t_sec, frames_dict[idx]))

    return frames_list, ep.task, ep.fps


def add_hud(img: np.ndarray, top_text: str, top_info: str = "", bottom_text: str = "") -> np.ndarray:
    """Adds a top task/frame banner and an optional bottom spatial relations banner."""
    h, w = img.shape[:2]
    banner_h = 32
    if h <= banner_h * 2:
        return img

    canvas = img.copy()
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (15, 15, 15), -1)

    has_bottom = bool(bottom_text and h > 110)
    if has_bottom:
        cv2.rectangle(overlay, (0, h - banner_h), (w, h), (15, 15, 15), -1)

    blended = cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0)

    # Scale font size based on image width
    scale = max(0.40, min(0.55, 0.50 * w / 640.0))
    cv2.putText(blended, top_text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    if top_info:
        (tw, _), _ = cv2.getTextSize(top_info, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.putText(blended, top_info, (w - tw - 8, 22), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 240, 255), 1, cv2.LINE_AA)

    if has_bottom:
        cv2.putText(blended, bottom_text, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, scale, (50, 230, 255), 1, cv2.LINE_AA)

    return blended


def write_contact_sheet(path: Path, tiles_data: list[tuple[np.ndarray, int, float]], cols: int = 2):
    """Saves a multi-row contact sheet of key frames."""
    if not tiles_data:
        return
    tile_h = 240
    resized = []
    for img, idx, t_sec in tiles_data:
        h, w = img.shape[:2]
        new_w = round(w * tile_h / h)
        tile = cv2.resize(img, (new_w, tile_h), interpolation=cv2.INTER_AREA)
        resized.append(tile)

    while len(resized) % cols != 0:
        resized.append(np.zeros_like(resized[0]))

    rows = []
    for r in range(0, len(resized), cols):
        rows.append(np.hstack(resized[r:r + cols]))

    sheet = np.vstack(rows)
    cv2.imwrite(str(path), sheet)


class TemporalObjectTracker:
    """Carries forward and temporally smooths grounded objects across video frames."""

    def __init__(self, max_missed: int = 3):
        self.max_missed = max_missed
        self.tracks: dict[str, dict] = {}

    def update(self, segments: list[dict]) -> list[dict]:
        """Smooths detected objects and restores briefly dropped objects."""
        current_by_label = {s["label"]: s for s in segments if "label" in s and not str(s["label"]).startswith("#")}
        active_labels = set(current_by_label.keys())

        # 1. Update tracks with detections in current frame
        for label, det in current_by_label.items():
            if label in self.tracks:
                old_box = self.tracks[label]["box"]
                new_box = det["box"]
                # Exponential smoothing of bounding box
                smooth_box = [
                    round(0.70 * n + 0.30 * o, 1) for n, o in zip(new_box, old_box)
                ]
                det["box"] = smooth_box
                self.tracks[label] = {
                    "box": smooth_box,
                    "score": max(det.get("score", 0.0), self.tracks[label]["score"] * 0.95),
                    "missed": 0,
                    "label": label,
                    "seg_id": det.get("seg_id", 999),
                }
            else:
                self.tracks[label] = {
                    "box": det["box"],
                    "score": det.get("score", 0.0),
                    "missed": 0,
                    "label": label,
                    "seg_id": det.get("seg_id", 999),
                }

        # 2. Check for tracks that missed detection in this frame
        for label, track in list(self.tracks.items()):
            if label not in active_labels:
                track["missed"] += 1
                if track["missed"] <= self.max_missed:
                    decayed_score = round(track["score"] * 0.88, 2)
                    if decayed_score >= 0.04:
                        sustained_seg = {
                            "seg_id": track["seg_id"],
                            "box": track["box"],
                            "centroid": (
                                int((track["box"][0] + track["box"][2]) / 2),
                                int((track["box"][1] + track["box"][3]) / 2),
                            ),
                            "area_px": int((track["box"][2] - track["box"][0]) * (track["box"][3] - track["box"][1])),
                            "label": label,
                            "score": decayed_score,
                            "mask": None,
                            "is_temporal": True,
                        }
                        segments.append(sustained_seg)
                else:
                    del self.tracks[label]

        return segments


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", type=Path, help="Path to input image or mp4 video (optional if --dataset is given)")
    parser.add_argument("--dataset", type=Path, help="Path to raw dataset root (e.g. data/raw/bridge)")
    parser.add_argument("--episode", type=int, default=0, help="Episode index in the dataset")
    parser.add_argument("--cam", default="image_0", help="Camera key (e.g. image_0 or observation.images.image_0)")
    parser.add_argument("--frame", type=int, default=0, help="Frame index within the episode (for single-frame mode)")
    parser.add_argument("--time", type=float, default=0.0, help="Timestamp in seconds for standalone video inputs")
    parser.add_argument("--video", action="store_true", help="Process the full video / episode and save output mp4")
    parser.add_argument("--stride", type=int, default=2, help="Frame stride for video mode (e.g. 1=all frames, 2=every second frame)")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process in video mode")
    parser.add_argument("--fps", type=float, default=None, help="Output video framerate (defaults to source fps / stride)")
    parser.add_argument("--threshold", type=float, default=0.6, help="Detection score threshold for semantic discovery (default: 0.15)")
    parser.add_argument("--clip-model", default="google/siglip-base-patch16-256", help="Model identifier for SigLIP classifier (default: google/siglip-base-patch16-256)")
    parser.add_argument("--bg-alpha", type=float, default=0.40, help="Background context attenuation for crop classifier (default: 0.40)")
    parser.add_argument("--vocab-file", type=Path, help="Path to custom text file with object names (defaults to bridge_objects.txt)")
    parser.add_argument("--top-k", type=int, default=12, help="Maximum number of objects to detect on scene (default: 12)")
    parser.add_argument("--from-prompt", action="store_true", help="Only look for objects extracted from episode instruction prompt")
    parser.add_argument("--candidates", help="Optional comma-separated candidate object names to search for instead of full list")
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "sam_probe", help="Output directory")
    parser.add_argument("--model", default="facebook/sam2.1-hiera-small", help="SAM 2.1 model identifier")
    parser.add_argument("--min-area", type=int, default=50, help="Minimum mask area in pixels to report")
    parser.add_argument("--max-area-pct", type=float, default=35.0, help="Maximum segment area percent for object grounding (default: 35.0% to reject background/table)")
    parser.add_argument("--no-boxes", action="store_true", help="Do not draw bounding box outlines")
    parser.add_argument("--no-labels", action="store_true", help="Do not draw segment ID tags")
    parser.add_argument("--show-all-labels", action="store_true", help="Draw background segment IDs even when semantic objects are present")
    parser.add_argument("--alpha", type=float, default=0.45, help="Alpha blending transparency for masks")
    parser.add_argument("--no-distractors", action="store_true", help="Disable negative prompt distractor classes (background / robot arm suppression)")
    parser.add_argument(
        "--arm-mode",
        choices=["parts", "arm", "gripper", "suppress", "none"],
        default="suppress",
        help="Robot arm annotation mode: 'suppress' (default: hide/clear robot arm completely), 'none' (leave raw), or 'parts'/'gripper'/'arm'",
    )
    parser.add_argument("--no-arm-mask", action="store_true", help="Alias to completely disable arm processing (--arm-mode none)")
    parser.add_argument("--no-ground", action="store_true", help="Disable zero-shot semantic grounding to task candidates")
    args = parser.parse_args()

    arm_mode = "none" if args.no_arm_mask else args.arm_mode

    args.out.mkdir(parents=True, exist_ok=True)
    task_desc = ""

    # Load input data: Video sequence mode vs Single frame mode
    if args.video:
        if args.dataset:
            frames_seq, task_desc, source_fps = load_episode_frames(
                args.dataset, args.episode, args.cam, stride=args.stride, max_frames=args.max_frames
            )
            print(f"Loaded {len(frames_seq)} frames of episode {args.episode} ({args.cam}) from {args.dataset}")
        elif args.source:
            frames_seq, source_fps = load_video_file_frames(args.source, stride=args.stride, max_frames=args.max_frames)
            print(f"Loaded {len(frames_seq)} frames from video {args.source}")
        else:
            parser.error("Specify --dataset or provide an input video file path.")
    else:
        if args.dataset:
            frame, task_desc = load_frame_from_dataset(args.dataset, args.episode, args.cam, args.frame)
            print(f"Loaded frame {args.frame} of episode {args.episode} ({args.cam}) from {args.dataset}")
        elif args.source:
            frame = load_frame_from_file(args.source, args.time)
            print(f"Loaded frame from file {args.source}")
        else:
            parser.error("Specify either an input file as positional argument, or provide --dataset.")

    if task_desc:
        print(f"Episode instruction: \"{task_desc}\"")

    # Load object list: by default, OWLv2 scans for ALL words from the vocabulary list
    from aijgen.objects.scene_vocabulary import load_vocabulary_from_file
    scene_vocab = load_vocabulary_from_file(args.vocab_file)

    candidates = []
    if not args.no_ground:
        if args.candidates:
            candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
            print(f"\nUsing explicit candidate objects ({len(candidates)}): {candidates}")
        elif args.from_prompt and task_desc:
            candidates = parse_task_objects(task_desc)
            print(f"\nUsing objects parsed from task prompt ({len(candidates)}): {candidates}")
        else:
            candidates = scene_vocab
            print(f"\nLoaded object list from file ({len(candidates)} categories from {args.vocab_file or 'bridge_objects.txt'}):")
            print(f"  {', '.join(candidates[:20])}... (+{len(candidates) - 20} more)")
            print(f"OWLv2 will scan the image for all words from this list (threshold={args.threshold}).")

    # Initialize Model Pipeline: SAM 2.1 for segmentation + SigLIP for semantic classification
    print(f"\nInitializing SAM 2.1 ({args.model})...")
    t0 = time.time()
    segmenter = Segmenter(args.model)
    print(f"SAM 2.1 loaded in {time.time() - t0:.2f} s")

    grounder = None
    if not args.no_ground and candidates:
        print(f"\nInitializing SAM->Crop classifier ({args.clip_model})...")
        t0 = time.time()
        grounder = CropSegmentClassifier(
            args.clip_model,
            bg_alpha=args.bg_alpha,
            max_area_pct=args.max_area_pct,
            arm_mode=arm_mode,
        )
        print(f"Crop classifier loaded in {time.time() - t0:.2f} s")

    # =========================================================================
    # VIDEO PROCESSING MODE
    # =========================================================================
    if args.video:
        h, w = frames_seq[0][2].shape[:2]
        out_fps = max(1.0, (source_fps / max(1, args.stride)) if args.fps is None else args.fps)

        grounded_video_path = args.out / "sam_grounded.mp4"
        comparison_video_path = args.out / "sam_comparison.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_grounded = cv2.VideoWriter(str(grounded_video_path), fourcc, out_fps, (w, h))
        writer_comparison = cv2.VideoWriter(str(comparison_video_path), fourcc, out_fps, (2 * w + 4, h))

        sheet_frames = []
        sheet_indices = set(np.linspace(0, len(frames_seq) - 1, min(8, len(frames_seq))).round().astype(int))

        print(f"\nRendering video at {out_fps:.1f} FPS to {args.out}...")
        t_start = time.time()
        tracker = TemporalObjectTracker(max_missed=2)

        for step, (frame_idx, t_sec, f_rgb) in enumerate(frames_seq):
            labels = segmenter(f_rgb)
            segments = extract_segments(labels, min_area_px=args.min_area)

            if grounder is not None and candidates:
                segments = grounder.ground(
                    f_rgb,
                    segments,
                    vocabulary=candidates,
                    threshold=args.threshold,
                    top_k=args.top_k,
                    include_distractors=not args.no_distractors,
                )
                segments = tracker.update(segments)

            if arm_mode != "none":
                arm_mask = compute_robot_arm_mask(f_rgb)
                segments, labels = annotate_robot_arm_segments(
                    segments, labels, arm_mask, mode=arm_mode
                )

            grounded_items = [
                s for s in segments
                if "label" in s and not str(s["label"]).startswith("#")
            ]
            grounded_summary = ", ".join(f"'{s['label']}' (#{s['seg_id']})" for s in grounded_items)

            overlay = draw_sam_segmentation(
                f_rgb,
                labels,
                segments=segments,
                show_boxes=not args.no_boxes,
                show_labels=not args.no_labels,
                alpha=args.alpha,
                show_bg_labels=args.show_all_labels,
            )
            comparison = create_sam_comparison(
                f_rgb,
                labels,
                segments=segments,
                show_boxes=not args.no_boxes,
                show_bg_labels=args.show_all_labels,
            )

            relations = compute_spatial_relations(segments, (h, w))
            rel_str = " | ".join(relations) if relations else ""
            bottom_hud = f"Spatial: {rel_str}" if rel_str else ""

            # Add stylish HUD
            info_str = f"f={frame_idx:03d} ({t_sec:.1f}s)"
            banner_prefix = "Scene Objects Discovery" if not task_desc else f"Scene Discovery | {task_desc}"
            banner_text = f"{banner_prefix[:55]}..." if len(banner_prefix) > 55 else banner_prefix
            overlay_bgr = add_hud(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), banner_text, info_str, bottom_hud)
            comp_bgr = add_hud(cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR), banner_text, info_str, bottom_hud)

            writer_grounded.write(overlay_bgr)
            writer_comparison.write(comp_bgr)

            if step in sheet_indices:
                sheet_frames.append((comp_bgr, frame_idx, t_sec))

            dt = time.time() - t_f0
            fps_proc = 1.0 / max(dt, 1e-4)
            rel_log = f" | Relations: {rel_str}" if rel_str else ""
            print(f"  [{step + 1:3d}/{len(frames_seq):3d}] Frame {frame_idx:4d} ({t_sec:4.1f}s) | {len(segments):2d} segs | {fps_proc:4.1f} FPS | Grounded: {grounded_summary or '-'}{rel_log}")

        writer_grounded.release()
        writer_comparison.release()
        t_total = time.time() - t_start

        print(f"\nVideo generation complete in {t_total:.1f} s ({len(frames_seq) / t_total:.1f} FPS avg)!")
        print(f"  * Saved grounded video:     {grounded_video_path}")
        print(f"  * Saved side-by-side video: {comparison_video_path}")

        # Save summary keyframe contact sheet
        if sheet_frames:
            sheet_path = args.out / "sam_summary_sheet.png"
            write_contact_sheet(sheet_path, sheet_frames, cols=2)
            print(f"  * Saved keyframe collage:   {sheet_path}")

        print(f"\nDone! You can inspect the video outputs in: {args.out}")
        return

    # =========================================================================
    # SINGLE FRAME MODE
    # =========================================================================
    h, w = frame.shape[:2]
    print(f"Frame resolution: {w}x{h}")

    print("Running automatic mask generation...")
    t1 = time.time()
    labels = segmenter(frame)
    print(f"SAM inference completed in {time.time() - t1:.2f} s")

    segments = extract_segments(labels, min_area_px=args.min_area)
    print(f"\nExtracted {len(segments)} segments (min_area >= {args.min_area} px):")
    for s in segments:
        print(f"  - Segment #{s['seg_id']:2d}: box={s['box']}  area={s['area_px']} px ({s['area_pct']}%)  center={s['centroid']}")

    if grounder is not None and candidates:
        segments = grounder.ground(
            frame,
            segments,
            vocabulary=candidates,
            threshold=args.threshold,
            top_k=args.top_k,
            include_distractors=not args.no_distractors,
        )

    if arm_mode != "none":
        arm_mask = compute_robot_arm_mask(frame)
        segments, labels = annotate_robot_arm_segments(
            segments, labels, arm_mask, mode=arm_mode
        )

    grounded_items = [
        s for s in segments
        if "label" in s and not str(s["label"]).startswith("#")
    ]
    print(f"\nDiscovered {len(grounded_items)} objects on image (threshold={args.threshold}):")
    for s in grounded_items:
        fb = " [fallback box]" if s.get("is_fallback") else ""
        arm_tag = " [robot part]" if s.get("is_robot_arm") else ""
        print(f"  * '{s['label']}' (score {s.get('score', 0):.2f}) -> Segment #{s['seg_id']}  box={s['box']}{fb}{arm_tag}")

    relations = compute_spatial_relations(segments, (h, w))
    rel_str = " | ".join(relations) if relations else ""
    bottom_hud = f"Spatial: {rel_str}" if rel_str else ""
    if relations:
        print("\nSpatial Relations:")
        for r in relations:
            print(f"  * {r}")

    # Render visualizations
    print("\nRendering visualizations...")
    overlay = draw_sam_segmentation(
        frame,
        labels,
        segments=segments,
        show_boxes=not args.no_boxes,
        show_labels=not args.no_labels,
        alpha=args.alpha,
        show_bg_labels=args.show_all_labels,
    )
    comparison = create_sam_comparison(
        frame,
        labels,
        segments=segments,
        show_boxes=not args.no_boxes,
        show_bg_labels=args.show_all_labels,
    )

    info_str = f"frame {args.frame}"
    banner_prefix = "Scene Objects Discovery" if not task_desc else f"Scene Discovery | {task_desc}"
    banner_text = f"{banner_prefix[:55]}..." if len(banner_prefix) > 55 else banner_prefix
    overlay = cv2.cvtColor(add_hud(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), banner_text, info_str, bottom_hud), cv2.COLOR_BGR2RGB)
    comparison = cv2.cvtColor(add_hud(cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR), banner_text, info_str, bottom_hud), cv2.COLOR_BGR2RGB)

    overlay_path = args.out / "sam_overlay.png"
    grounded_path = args.out / "sam_grounded.png"
    comparison_path = args.out / "sam_comparison.png"
    npz_path = args.out / "sam_labels.npz"

    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(grounded_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(comparison_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    np.savez_compressed(npz_path, labels=labels)

    print(f"Saved overlay rendering:    {overlay_path}")
    print(f"Saved grounded targets:     {grounded_path}")
    print(f"Saved side-by-side collage: {comparison_path}")
    print(f"Saved raw label map:        {npz_path}")
    print("\nDone! You can inspect the generated images in the output directory.")


if __name__ == "__main__":
    main()