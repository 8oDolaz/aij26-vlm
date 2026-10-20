#!/usr/bin/env python3
"""Hand check of the SAM 3 questions (docs/sam3_vqa_plan.md): run the config's enabled SAM 3 generators (scene_qa,
task_success) on a few episodes per dataset and draw one contact sheet per template.

Each tile: the question images as they are saved (markers drawn), then each image's detections (green = subject,
nameable in questions; yellow = kept but not nameable; orange = contested, dropped), with the question, its options
and the answer. Cross-view questions show both views; task-success tiles also say which kind of answer it is (yes,
swap: object swapped, early: pre-grasp frame).
Writes to --out:
  <generator>/cache/            SAM 3 (+ DA3) detections; generate_dataset.py --output <same dir> reuses them
  sheet_<template>_<page>.jpg
  review.jsonl                  one line per question: id, template, dataset, episode, frames, cameras, question, answer
Report wrong labels by id. The plan asks for >= 95% correct on ~50 questions per template before it goes into the mix.

Run on the GPU box (the first run needs the GPU; cached frames load no model):
  docker compose exec dev python tools/scene_qa_sheet.py --input /data/raw --out runs/scene_qa_check --episodes 40
  docker compose exec dev python tools/scene_qa_sheet.py --input /data/raw --out runs/p4_check \
    --config configs/gen_p4.yaml --episodes 60
"""
import argparse
import json
import re
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "submission" / "src"))
from aijgen.depth import lerobot  # noqa: E402
from aijgen.generators.episodes import load_episodes  # noqa: E402
from aijgen.generators.run import image_cameras, image_overlays  # noqa: E402
from aijgen.generators.scene_qa import SceneQAGenerator  # noqa: E402
from aijgen.generators.task_success import TaskSuccessGenerator  # noqa: E402
from aijgen.scene import detect as D  # noqa: E402
from aijgen.scene.render import draw_overlays  # noqa: E402

GENERATORS = {g.name: g for g in (SceneQAGenerator, TaskSuccessGenerator)}

PANEL_H, COLS, ROWS = 300, 2, 4
FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE = re.compile(r"^\(?([A-J])[).]\s+(.+)$")
INLINE = re.compile(r"([A-D])\. (.+?)\.(?= [A-D]\. | Please)")


def options(question: str) -> dict:
    """Letter -> option text, for the EmbSpatial / BLINK (one per line) and ERQA (inline) formats."""
    lines = [part for ln in question.splitlines() for part in re.split(r"  (?=[A-J]\. )", ln)]  # CrossPoint
    opts = {m.group(1): m.group(2) for m in map(LINE.match, lines) if m}
    if not opts and "Choices:" in question:
        opts = dict(INLINE.findall(question.split("Choices:", 1)[1]))
    return opts


def caption(question: str, answer: str) -> list[str]:
    """The question without boilerplate, wrapped, then the answer with its option text."""
    q = question.split(" Please answer directly", 1)[0]
    lines = [ln for ln in q.splitlines() if not ln.startswith(("Note: You only need", "Answer the question based"))]
    out = []
    for ln in lines:
        out += textwrap.wrap(ln, 95) or [""]
    opt = options(question).get(answer)
    out.append(f">> {answer}" + (f" = {opt}" if opt else ""))
    return out


def detection_panel(rgb: np.ndarray, det: D.FrameDetections, scene, entries, p) -> np.ndarray:
    img = rgb.copy()
    subjects = {id(o) for o in scene.subjects}
    h, w = det.size
    contested = D.select_candidates([{"name": b.name, "box": b.box, "score": b.score} for b in det.boxes
                                     if b.name in entries], entries, h * w, p)["ambiguous"]
    for o in scene.objects:
        color = (0, 200, 0) if id(o) in subjects else (230, 200, 0)
        x0, y0, x1, y1 = (int(v) for v in o.box)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        depth = "" if o.disp is None else f" d{o.disp:.2f}"
        cv2.putText(img, f"{o.name} {o.score:.2f}{depth}", (x0 + 2, max(12, y0 - 4)), FONT, 0.4, color, 1, cv2.LINE_AA)
    for c in contested:
        x0, y0, x1, y1 = (int(v) for v in c["box"])
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 140, 0), 1)
    return img


def tile(panels: list[np.ndarray], lines: list[str]) -> np.ndarray:
    panels = [cv2.resize(im, (round(im.shape[1] * PANEL_H / im.shape[0]), PANEL_H)) for im in panels]
    top = np.hstack(panels)
    w = top.shape[1]
    lh = 15
    text = np.full((lh * len(lines) + 8, w, 3), 20, np.uint8)
    for i, ln in enumerate(lines):
        color = (120, 255, 120) if ln.startswith(">>") else (255, 255, 255)
        cv2.putText(text, ln, (6, lh * (i + 1)), FONT, 0.42, color, 1, cv2.LINE_AA)
    return np.vstack([top, text])


def write_sheets(tiles: list[np.ndarray], stem: Path):
    per = COLS * ROWS
    for p in range(0, len(tiles), per):
        page = tiles[p:p + per]
        w = max(t.shape[1] for t in page)
        page = [np.pad(t, ((0, 0), (0, w - t.shape[1]), (0, 0))) for t in page]
        page += [np.zeros_like(page[0])] * (-len(page) % COLS)
        rows = []
        for r in range(0, len(page), COLS):
            row = page[r:r + COLS]
            h = max(t.shape[0] for t in row)
            rows.append(np.hstack([np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0))) for t in row]))
        cv2.imwrite(f"{stem}_{p // per + 1}.jpg", cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=REPO / "data" / "raw", help="root with one LeRobot dataset per folder")
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "scene_qa_check")
    ap.add_argument("--config", type=Path, default=REPO / "submission" / "config.yaml")
    ap.add_argument("--datasets", nargs="+", default=["bridge", "fractal"])
    ap.add_argument("--episodes", type=int, default=40, help="per dataset, evenly spread")
    args = ap.parse_args()

    config = yaml.safe_load(args.config.read_text())
    gens = [GENERATORS[name](dict(gcfg, work_dir=str(args.out / name)))
            for name, gcfg in config["generators"].items() if name in GENERATORS and gcfg.get("enabled", True)]
    by_template, review = defaultdict(list), []
    args.out.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        root = args.input / ds
        mine = [g for g in gens if ds in g.datasets]
        if not (root / "meta" / "info.json").exists() or not mine:
            print(f"skip {ds}")
            continue
        ecfg = dict(config["episodes"]["datasets"][ds], max_episodes=args.episodes)
        for n, (ep, videos) in enumerate(load_episodes(root, ecfg)):
            for gen in mine:
                entries, _ = gen.vocab(ds)
                questions = gen(ep, videos=videos)
                print(f"[{ds} {n + 1}] {gen.name} ep{ep.index} \"{ep.task}\": {len(questions)} questions", flush=True)
                for q in questions:
                    cams = image_cameras(q, [])[0]
                    shown, panels = [], []
                    for k, (fi, cam) in enumerate(zip(q.frames, cams)):
                        rgb = lerobot.read_frames(videos[cam], [fi])[fi]
                        det = D.load(gen.work / "cache" / ds / f"ep{ep.index:06d}_f{fi:04d}_{cam.split('.')[-1]}.npz")
                        scene = gen.scene(det, entries)
                        shown.append(draw_overlays(rgb, image_overlays(q.overlays, k)))
                        panels.append(detection_panel(rgb, det, scene, entries, gen.filter))
                    qid = len(review)
                    t = q.meta["template"]
                    where = ", ".join(f"f{fi} {cam.split('.')[-1]}" for fi, cam in zip(q.frames, cams))
                    lines = [f"#{qid} {t} | {ds} ep{ep.index} {where} | {ep.task}"]
                    if "answer_kind" in q.meta:
                        lines.append(f"answer kind: {q.meta['answer_kind']}"
                                     + (f" (swapped in: {q.meta['_swap']})" if "_swap" in q.meta else ""))
                    lines += caption(q.question, q.answer)
                    if t == "existence_msmu":
                        lines.append(f"absent here: {', '.join(scene.absent)}")
                    by_template[t].append(tile(shown + panels, lines))
                    review.append({"id": qid, "template": t, "dataset": ds, "episode": ep.index, "frames": q.frames,
                                   "cameras": cams, "question": q.question, "answer": q.answer,
                                   "answer_kind": q.meta.get("answer_kind")})
    for gen in gens:
        if getattr(gen, "skips", None):
            print(f"{gen.name} episodes: {dict(gen.skips.most_common())}")
    for t, tiles in by_template.items():
        write_sheets(tiles, args.out / f"sheet_{t}")
    with open(args.out / "review.jsonl", "w") as f:
        for r in review:
            f.write(json.dumps(r) + "\n")
    print({t: len(v) for t, v in sorted(by_template.items())})
    print(f"wrote {args.out}/")


if __name__ == "__main__":
    main()
