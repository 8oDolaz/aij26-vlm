#!/usr/bin/env python3
"""Run a monocular depth model on a time range of one video, to judge by eye whether its relative depth is good
enough to label "which point is closer" questions (Bridge's toy-kitchen scenes are shallow).

LeRobot v3 packs many episodes into one mp4 and --from/--to are seconds into that file. An episode's span is in
meta/episodes: videos/<video_key>/from_timestamp and to_timestamp.

Depth Anything V2 relative models output affine-invariant inverse depth: larger means closer, and values are only
comparable within one frame. The visualisation normalises each frame on its own (2nd to 98th percentile);
warm colours are near, dark is far.

Writes to --out (default runs/depth_probe/<camera>_<file>_<from>-<to>_<model>/):
- side_by_side.mp4: RGB | depth, at the video's frame rate divided by --stride
- sheet.png: --sheet evenly spaced frames, each as an RGB | depth pair, with timestamps
- depth.npz: `depth` (frames, H, W) float16 raw model output at video resolution, `t` timestamps in seconds
and prints per-frame inference time and peak VRAM, to estimate the cost over the whole dataset.

Licences: Depth-Anything-V2-Small is Apache-2.0; Base and Large are CC-BY-NC-4.0.

Run in the dev container (torch, transformers, av, opencv, numpy; the HF cache is mounted for the weights):
  docker compose exec dev python tools/depth_probe.py \
      data/raw/bridge/videos/observation.images.image_0/chunk-000/file-028.mp4 --from 120 --to 150
  docker compose exec dev python tools/depth_probe.py <video> --from 0 --to 60 --stride 5 \
      --model depth-anything/Depth-Anything-V2-Small-hf
"""
import argparse
import time
from pathlib import Path

import av
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

REPO = Path(__file__).resolve().parents[1]
TILE_H = 240  # contact-sheet tile height, px
SHEET_COLS = 2  # RGB | depth pairs per contact-sheet row


def read_frames(path: Path, start: float, end: float, stride: int):
    """RGB frames with timestamps in [start, end], every `stride`-th one, and the video's frame rate."""
    frames, times = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        container.seek(int(start / stream.time_base), stream=stream)  # lands on the keyframe at or before start
        i = 0
        for frame in container.decode(stream):
            if frame.time < start - 1e-6:
                continue
            if frame.time > end:
                break
            if i % stride == 0:
                frames.append(frame.to_ndarray(format="rgb24"))
                times.append(frame.time)
            i += 1
        fps = float(stream.average_rate)
    return frames, times, fps


@torch.inference_mode()
def predict(frames, model_id: str, batch: int):
    """Raw depth per frame at the frame's resolution, plus seconds spent in the model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device, dtype).eval()
    h, w = frames[0].shape[:2]
    depths, elapsed = [], 0.0
    for i in range(0, len(frames), batch):
        inputs = processor(images=frames[i:i + batch], return_tensors="pt").to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = model(pixel_values=inputs["pixel_values"].to(dtype)).predicted_depth  # (B, h', w')
        pred = F.interpolate(pred[:, None].float(), size=(h, w), mode="bicubic", align_corners=False)[:, 0]
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - t0
        depths.extend(pred.cpu().numpy())
    return depths, elapsed, device


def colorize(depth: np.ndarray) -> np.ndarray:
    """BGR turbo image of one frame's depth, normalised on its own 2nd..98th percentile."""
    lo, hi = np.percentile(depth, (2, 98))
    norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def pair(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    return np.hstack([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), colorize(depth)])


def write_video(path: Path, frames, depths, fps: float):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * w, h))
    for rgb, depth in zip(frames, depths):
        writer.write(pair(rgb, depth))
    writer.release()


def write_sheet(path: Path, frames, depths, times, n: int):
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).round().astype(int)
    tiles = []
    for i in idx:
        tile = pair(frames[i], depths[i])
        tile = cv2.resize(tile, (round(tile.shape[1] * TILE_H / tile.shape[0]), TILE_H), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, f"t={times[i]:.1f}s", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(tile)
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % SHEET_COLS)
    rows = [np.hstack(tiles[r:r + SHEET_COLS]) for r in range(0, len(tiles), SHEET_COLS)]
    cv2.imwrite(str(path), np.vstack(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--from", dest="start", type=float, required=True, help="start, seconds into the file")
    ap.add_argument("--to", dest="end", type=float, required=True, help="end, seconds into the file")
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Base-hf")
    ap.add_argument("--stride", type=int, default=1, help="keep every n-th frame")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--sheet", type=int, default=8, help="frames on the contact sheet")
    ap.add_argument("--out", type=Path, help="output directory")
    args = ap.parse_args()
    if args.end <= args.start:
        ap.error("--to must be after --from")

    frames, times, fps = read_frames(args.video, args.start, args.end, args.stride)
    if not frames:
        raise SystemExit(f"no frames in [{args.start}, {args.end}] s of {args.video}")
    camera = args.video.parents[1].name.removeprefix("observation.images.")
    out = args.out or REPO / "runs" / "depth_probe" / (
        f"{camera}_{args.video.stem}_{args.start:g}-{args.end:g}_{args.model.split('/')[-1]}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(frames)} frames {frames[0].shape[1]}x{frames[0].shape[0]} from {times[0]:.2f}s to {times[-1]:.2f}s")

    depths, elapsed, device = predict(frames, args.model, args.batch)
    print(f"{args.model} on {device}: {1000 * elapsed / len(frames):.1f} ms/frame", end="")
    if device == "cuda":
        print(f", peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", end="")
    print()

    write_video(out / "side_by_side.mp4", frames, depths, fps / args.stride)
    write_sheet(out / "sheet.png", frames, depths, times, args.sheet)
    np.savez_compressed(out / "depth.npz", depth=np.stack(depths).astype(np.float16), t=np.array(times))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
