#!/usr/bin/env python3
"""Tier 1 of docs/relative_depth_plan.md: BLINK-style "which point is closer" questions from Bridge and Fractal.

Per source frame (one moment of an episode): Depth Anything 3 on all of the dataset's views at once (Bridge: cameras
0, 1 and 2; Fractal: its single camera), and again at a few neighbouring frames for the stability check. One view is
picked for the question (rotating over the views, so cameras are used evenly); SAM2.1 segments only that view, points
are sampled per segment, and pairs are kept only when the normalised gap clears the threshold of their shortcut group
and the order holds in the neighbouring frames; then balanced so "lower is closer" and the other shortcuts score
about 50%.

Writes to --output:
  annotations.jsonl       ShareGPT records ({"messages": [...], "images": [abs path]})
  annotations.meta.jsonl  one line per record, same order: id, source frame, question camera, views given to DA3,
                          points, gap, group, prompt kind
  frames/                 question images with markers drawn
  cache/                  per source frame: RGB jpg of the question view, of the other views (<key>.<camera>.jpg),
                          and npz (inverse depth and DA3 confidence of the question view, its inverse depth at the
                          neighbouring frames, SAM2 labels, question camera, views)
  sheets/sheet_NN.jpg     spot-check sheets: question image | depth | the other views, captioned with id, answer,
                          gap and group
  review.jsonl            the sheet samples by id (to report wrong ones)
  report.json             funnel, shortcut audit, balance, gap quantiles, timing

Run from the repo root in the dev container (DA3 source at depth_tier1.da3_src; it needs `addict`):
  PYTHONPATH=submission/src python -m aijgen.depth.tier1 --input data/raw --output runs/depth_tier1
  PYTHONPATH=submission/src python -m aijgen.depth.tier1 --input data/raw --output runs/depth_tier1 --relabel
--relabel reuses cache/ and loads no model, so thresholds in config.yaml can be re-tuned in seconds.
"""
import argparse
import json
import shutil
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from aijgen.depth import pairs as P
from aijgen.depth import render as R
from aijgen.depth.lerobot import Episode, list_episodes, read_frames

SHEET_SAMPLES, SHEET_ROWS, TILE_W = 48, 6, 380


def rng_for(seed: int, *keys) -> np.random.Generator:
    return np.random.default_rng([seed] + [zlib.crc32(str(k).encode()) for k in keys])


def evenly(items: list, n: int) -> list:
    if len(items) <= n:
        return items
    return [items[i] for i in np.linspace(0, len(items) - 1, n).round().astype(int)]


def episodes_by_index(per_view: dict[str, list[Episode]], min_views: int) -> list[dict[str, Episode]]:
    """Per episode index, its Episode for every view with a video on disk; only episodes with >= min_views views.
    Sorted by episode index."""
    grouped = defaultdict(dict)
    for view, eps in per_view.items():
        for e in eps:
            grouped[e.index][view] = e
    return [grouped[i] for i in sorted(grouped) if len(grouped[i]) >= min_views]


def question_view(valid: list[str], turn: int) -> str:
    """The view questions are asked on: rotates with `turn`, so every camera gets about the same share."""
    return valid[turn % len(valid)]


def short(view: str) -> str:
    return view.split(".")[-1]


class Models:
    """Loaded on first use, so --relabel never touches torch."""

    def __init__(self, cfg):
        self.cfg, self.teacher, self.segmenter = cfg, None, None

    def get(self):
        if self.teacher is None:
            from aijgen.depth.models import DepthTeacher, Segmenter
            self.teacher = DepthTeacher(self.cfg["depth_model"], self.cfg.get("da3_src"), self.cfg["da3_process_res"])
            self.segmenter = Segmenter(self.cfg["segmenter"])
        return self.teacher, self.segmenter


def frame_data(key: str, eps: dict[str, Episode], fi: int, turn: int, ds: dict, cfg: dict, cache: Path,
               models: Models, relabel: bool, funnel: Counter, timing: Counter):
    """(rgb, disparity, labels, neighbour disparities, question view, views) for one source frame, from cache or
    computed. Depth comes from one DA3 call over all non-blank views of the moment, so every view informs the
    question view's depth; the neighbouring frames get the same treatment. None if fewer than ds["min_views"] views
    are present and non-blank (Bridge episodes recorded with fewer cameras)."""
    npz, jpg = cache / f"{key}.npz", cache / f"{key}.jpg"
    if npz.exists() and jpg.exists():
        z = np.load(npz)
        rgb = cv2.cvtColor(cv2.imread(str(jpg)), cv2.COLOR_BGR2RGB)
        return (rgb, z["disp"].astype(np.float32), z["labels"], [d.astype(np.float32) for d in z["neighbours"]],
                str(z["camera"]), [str(v) for v in z["views"]])
    if relabel:
        return None
    length = min(e.length for e in eps.values())
    offsets = cfg["stability"]["offsets"] if cfg["stability"]["enabled"] else []
    wanted = [fi + o for o in offsets if 0 <= fi + o < length]
    frames = {v: read_frames(e, [fi] + wanted) for v, e in eps.items()}
    valid = [v for v in ds["views"] if v in frames and fi in frames[v] and frames[v][fi].std() >= cfg["min_frame_std"]]
    if len(valid) < ds["min_views"]:
        funnel["frames_too_few_views"] += 1
        return None
    cam = question_view(valid, turn)
    q = valid.index(cam)
    teacher, segmenter = models.get()
    t = time.perf_counter()
    disps, confs = teacher([frames[v][fi] for v in valid])
    nbrs = [teacher([frames[v][i] for v in valid])[0][q] for i in wanted if all(i in frames[v] for v in valid)]
    timing["depth_s"] += time.perf_counter() - t
    t = time.perf_counter()
    labels = segmenter(frames[cam][fi])
    timing["segment_s"] += time.perf_counter() - t
    timing["frames_computed"] += 1
    npz.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[cam][fi].shape[:2]
    neighbours = np.stack(nbrs).astype(np.float16) if nbrs else np.zeros((0, h, w), np.float16)
    np.savez_compressed(npz, disp=disps[q].astype(np.float16), conf=confs[q].astype(np.float16),
                        neighbours=neighbours, labels=labels, camera=cam, views=np.array(valid))
    for v in valid:
        out = jpg if v == cam else cache / f"{key}.{short(v)}.jpg"
        cv2.imwrite(str(out), cv2.cvtColor(frames[v][fi], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return frames[cam][fi], disps[q], labels, nbrs, cam, valid


def collect(args, cfg, models: Models, funnel: Counter, timing: Counter):
    """Filtered (not yet balanced) pairs per dataset, and per source frame its shape."""
    by_dataset, shapes = defaultdict(list), {}
    lo, hi = cfg["frames_span"]
    for ds_name, ds in cfg["datasets"].items():
        root = args.input / ds_name
        if not (root / "meta" / "info.json").exists():
            print(f"skip {ds_name}: no {root}/meta/info.json")
            continue
        per_view = {v: [e for e in list_episodes(root, v) if e.length >= ds["min_length"]] for v in ds["views"]}
        groups = episodes_by_index(per_view, ds["min_views"])
        chosen = evenly(groups, args.max_episodes or ds["max_episodes"])
        print(f"{ds_name}: {len(groups)} episodes with >= {ds['min_views']} of {[short(v) for v in ds['views']]} "
              f"on disk, using {len(chosen)}", flush=True)
        for n_ep, eps in enumerate(chosen):
            ep = next(iter(eps.values()))
            length = min(e.length for e in eps.values())
            fis = sorted({int(i) for i in np.linspace(lo * (length - 1), hi * (length - 1),
                                                      ds["frames_per_episode"]).round()})
            for n_fi, fi in enumerate(fis):
                key = f"{ds_name}/ep{ep.index:06d}_f{fi:04d}"
                funnel["frames"] += 1
                data = frame_data(key, eps, fi, n_ep * len(fis) + n_fi, ds, cfg, args.output / "cache", models,
                                  args.relabel, funnel, timing)
                if data is None:
                    funnel["frames_missing_or_blank"] += 1
                    continue
                _, disp, labels, nbr, cam, views = data
                dn = P.normalise(disp)
                rng = rng_for(cfg["seed"], key)
                pts = P.sample_points(labels, dn, cfg["points"], rng)
                found, considered = P.candidate_pairs(pts, dn.shape, cfg["pairs"])
                funnel["points"] += len(pts)
                funnel["pairs_considered"] += considered
                funnel["pairs_pass_gap"] += len(found)
                if cfg["stability"]["enabled"]:
                    nbr_dn = [P.normalise(d) for d in nbr]
                    kept = [p for p in found if P.stable(p, nbr_dn, cfg["stability"])]
                else:
                    kept = found
                funnel["pairs_pass_stability"] += len(kept)
                kept = P.per_frame(kept, cfg["pairs"]["max_per_frame_per_group"], rng)
                funnel["pairs_after_frame_cap"] += len(kept)
                for p in kept:
                    p.update(frame=key, dataset=ds_name, camera=cam, views=views, episode=ep.index, frame_index=fi,
                             task=ep.task)
                by_dataset[ds_name].extend(kept)
                shapes[key] = dn.shape
    return by_dataset, shapes


def colorize(dn: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.applyColorMap((np.clip(dn, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO),
                        cv2.COLOR_BGR2RGB)


def other_views(out: Path, m: dict, height: int) -> np.ndarray:
    """The views DA3 saw besides the question view, stacked in a column `height` tall (black if there are none)."""
    width = round(height * 2 / 3)  # two 4:3 views stacked
    others = [v for v in m["views"] if v != m["camera"]]
    if not others:
        return np.zeros((height, width, 3), np.uint8)
    cells = []
    for v in others:
        img = cv2.cvtColor(cv2.imread(str(out / "cache" / f"{m['frame']}.{short(v)}.jpg")), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (width, height // len(others)), interpolation=cv2.INTER_AREA)
        cv2.putText(img, short(v), (4, 14), R.FONT, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        cells.append(img)
    col = np.vstack(cells)
    return np.vstack([col, np.zeros((height - col.shape[0], width, 3), np.uint8)])


def write_sheets(out: Path, samples: list[dict]):
    """Question image as the model sees it | depth map with the same markers | the other views of the same moment;
    caption with id and answer."""
    sheets = out / "sheets"
    shutil.rmtree(sheets, ignore_errors=True)
    sheets.mkdir(parents=True)
    tiles, review = [], []
    for s in samples:
        m = s["meta"]
        shown = cv2.cvtColor(cv2.imread(s["image"]), cv2.COLOR_BGR2RGB)
        z = np.load(out / "cache" / f"{m['frame']}.npz")
        depth, _ = R.resize_width(colorize(P.normalise(z["disp"].astype(np.float32))), shown.shape[1])
        depth = R.draw(depth, {k: tuple(v) for k, v in m["drawn"].items()}, "ring", (255, 255, 255))
        tile = np.hstack([shown, depth, other_views(out, m, shown.shape[0])])
        tile = cv2.resize(tile, (TILE_W * 5 // 2, round(tile.shape[0] * TILE_W * 5 / 2 / tile.shape[1])),
                          interpolation=cv2.INTER_AREA)
        cap = np.zeros((40, tile.shape[1], 3), np.uint8)
        line1 = f"#{m['id']}  {m['dataset']}/{short(m['camera'])} ep{m['episode']} f{m['frame_index']}  " \
                f"{m['group']} g={m['g']:.2f}{' bg' if m['background'] else ''}  DA3 views: {len(m['views'])}"
        line2 = f"closer: {m['near']}   [{m['kind']}] answer: {s['answer'][:60]}"
        cv2.putText(cap, line1, (6, 16), R.FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(cap, line2, (6, 34), R.FONT, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        tiles.append(np.vstack([cap, tile]))
        review.append({"id": m["id"], "sheet": f"sheet_{(len(tiles) - 1) // (2 * SHEET_ROWS):02d}.jpg",
                       "camera": m["camera"], "closer": m["near"], "g": m["g"], "group": m["group"],
                       "image": s["image"], "frame": m["frame"]})
    hmax = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.zeros((hmax - t.shape[0], t.shape[1], 3), np.uint8)]) for t in tiles]
    per = 2 * SHEET_ROWS
    for n in range(0, len(tiles), per):
        chunk = tiles[n:n + per] + [np.zeros_like(tiles[0])] * (-len(tiles[n:n + per]) % 2)
        rows = [np.hstack(chunk[i:i + 2]) for i in range(0, len(chunk), 2)]
        cv2.imwrite(str(sheets / f"sheet_{n // per:02d}.jpg"), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    with open(out / "review.jsonl", "w") as f:
        for r in review:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="raw data root with bridge/, fractal/")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[3] / "config.yaml")
    ap.add_argument("--relabel", action="store_true", help="use cache/ only, no models")
    ap.add_argument("--max-episodes", type=int, help="override max_episodes per dataset")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())["depth_tier1"]
    args.output.mkdir(parents=True, exist_ok=True)
    funnel, timing = Counter(), Counter()
    t_start = time.perf_counter()

    by_dataset, shapes = collect(args, cfg, Models(cfg), funnel, timing)

    selected = []
    for ds_name, found in by_dataset.items():
        rng = rng_for(cfg["seed"], ds_name, "balance")
        chosen = P.balance(found, cfg["pairs"]["background_share"], rng)
        P.assign_letters(chosen, rng)
        selected += chosen
    funnel["final"] = len(selected)
    order = rng_for(cfg["seed"], "order").permutation(len(selected))
    selected = [selected[i] for i in order]

    frames_dir = args.output / "frames"
    shutil.rmtree(frames_dir, ignore_errors=True)
    rgb_cache, samples = {}, []
    pc = cfg["prompts"]
    with open(args.output / "annotations.jsonl", "w") as fa, open(args.output / "annotations.meta.jsonl", "w") as fm:
        for n, p in enumerate(selected):
            rng = rng_for(cfg["seed"], p["frame"], p["near"]["seg"], p["far"]["seg"], "render")
            if p["frame"] not in rgb_cache:
                rgb_cache = {p["frame"]: cv2.cvtColor(cv2.imread(str(args.output / "cache" / f"{p['frame']}.jpg")),
                                                      cv2.COLOR_BGR2RGB)}
            img, s = R.resize_width(rgb_cache[p["frame"]], cfg["render_width"])
            a, b = P.points_ab(p)
            drawn = {"A": (round(a["x"] * s), round(a["y"] * s)), "B": (round(b["x"] * s), round(b["y"] * s))}
            style = "dot" if rng.random() < pc["p_dot"] else "ring"
            color = R.DOT_COLORS[rng.integers(len(R.DOT_COLORS))] if style == "dot" else R.RED
            near = "A" if p["a_is_near"] else "B"
            prompt, answer, kind = R.question(near, style, pc, rng)
            path = (frames_dir / f"{p['frame']}_{short(p['camera'])}_{n:06d}.jpg").resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), cv2.cvtColor(R.draw(img, drawn, style, color), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            fa.write(json.dumps({"messages": [{"role": "user", "content": "<image>\n" + prompt},
                                              {"role": "assistant", "content": answer}],
                                 "images": [str(path)]}) + "\n")
            meta = {"id": n, "frame": p["frame"], "dataset": p["dataset"], "camera": p["camera"], "views": p["views"],
                    "episode": p["episode"], "frame_index": p["frame_index"], "task": p["task"], "A": a, "B": b,
                    "drawn": drawn, "near": near, "g": p["g"], "group": p["group"], "background": p["background"],
                    "style": style, "kind": kind}
            fm.write(json.dumps(meta) + "\n")
            p["kind"], p["style"] = kind, style
            samples.append({"meta": meta, "image": str(path), "answer": answer})

    sheet_ids = sorted(rng_for(cfg["seed"], "sheets").choice(len(samples), min(SHEET_SAMPLES, len(samples)),
                                                             replace=False)) if samples else []
    if samples:
        write_sheets(args.output, [samples[i] for i in sheet_ids])

    report = {
        "funnel": dict(funnel),
        "candidates_by_dataset_group": {ds: dict(Counter(p["group"] for p in found))
                                        for ds, found in by_dataset.items()},
        "final_by_dataset": dict(Counter(p["dataset"] for p in selected)),
        "final_by_camera": dict(Counter(p["camera"] for p in selected)),
        "final_by_group": dict(Counter(p["group"] for p in selected)),
        "background_share": round(float(np.mean([p["background"] for p in selected])), 3) if selected else None,
        "point_kinds": dict(Counter(q["kind"] for p in selected for q in (p["near"], p["far"]))),
        "prompt_kinds": dict(Counter(p["kind"] for p in selected)),
        "styles": dict(Counter(p["style"] for p in selected)),
        "shortcut_audit": P.audit(selected, lambda p: shapes[p["frame"]]),
        "shortcut_audit_by_dataset": {ds: P.audit([p for p in selected if p["dataset"] == ds],
                                                  lambda p: shapes[p["frame"]]) for ds in by_dataset},
        "g_quantiles_by_group": {g: [round(float(x), 3) for x in np.quantile([p["g"] for p in selected
                                                                               if p["group"] == g], (0.1, 0.5, 0.9))]
                                 for g in P.GROUPS if any(p["group"] == g for p in selected)},
        "pairs_per_100_frames": round(100 * len(selected) / max(funnel["frames"] - funnel["frames_missing_or_blank"],
                                                                1), 1),
        "timing": {"total_s": round(time.perf_counter() - t_start, 1),
                   "frames_computed": timing["frames_computed"],
                   "depth_s_per_frame": round(timing["depth_s"] / max(timing["frames_computed"], 1), 3),
                   "segment_s_per_frame": round(timing["segment_s"] / max(timing["frames_computed"], 1), 3)},
        "config": cfg,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "config"}, indent=1))
    print(f"wrote {len(selected)} samples to {args.output}")


if __name__ == "__main__":
    main()
