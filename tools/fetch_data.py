#!/usr/bin/env python3
"""Download Bridge, Fractal and Language Table (LeRobot v3) at the organizers' pinned commits: all metadata,
plus either every video file (--full) or a deterministic subset picked from that metadata.

--no-video skips video altogether: only meta/ and data/, which hold every episode's state, action and
instruction; selection.json then lists every usable episode.

LeRobot v3 packs hundreds to thousands of episodes into one mp4, so the unit we choose is a video file.
meta/ and data/ (per-episode instruction, length, video offsets, image stats; per-frame state and action)
are ~0.75 GB for all three and are always fetched in full. Selection uses only them:
- an episode is usable if it has an instruction, a sane length and does something. Bridge, Fractal: the measured
  gripper goes through the --phases (default pre_grasp, contact, post_grasp, each >= --min-phase-frames), with
  Robo2VLM's phase segmentation in submission/src/aijgen/phases.py, which the generator shares. Language Table
  has no gripper: effector motion above the dataset's 5th percentile;
- a camera is real for an episode if its image stats max > 0.05. Bridge pads cameras missing from a session
  with black video, and its stored std is 0 everywhere, so max is the usable signal;
- files of the primary camera (first video key in info.json) are picked greedily by gain per byte within the
  budget. Gain counts usable episodes, extra weight for multi-view ones, and instructions not yet covered.
  Bridge is sorted by collection session (its first 2 files hold 16 instructions, 2 random ones ~330), so this
  matters there; Fractal and Language Table are shuffled and any file is as good as another.
  A multi-view episode costs ~3 files, so it never wins on gain per byte: mv_share of the budget is spent first
  counting only multi-view episodes;
- other cameras: only the files holding real frames of the chosen usable episodes.
No randomness; ties go to the lower file index. Video keys missing from info.json are never fetched
(Bridge's videos/observation.images.main is a byte-identical copy of image_0, 30 GB).

Writes <out>/<dataset>/selection.json (repo, revision, files, usable and multi-view episode ids).
generate_dataset.py should take its episodes from it, so a run on the full data picks the same ones.

Run in the dev container (huggingface_hub, pyarrow, numpy; HF token comes from the mounted HF cache):
  docker compose run --rm dev python tools/fetch_data.py --dry-run
  docker compose run --rm dev python tools/fetch_data.py
  docker compose run --rm dev python tools/fetch_data.py --full fractal --budget-gb bridge=10
  docker compose run --rm dev python tools/fetch_data.py --no-video            # proprioception only, all datasets
  docker compose run --rm dev python tools/fetch_data.py --no-video bridge     # video for fractal, language_table only
  docker compose run --rm dev python tools/fetch_data.py --phases pre_grasp contact   # grasps that end holding count too
  docker compose run --rm dev python tools/fetch_data.py --phases none                # no gripper requirement
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "submission" / "src"))
from aijgen import phases  # noqa: E402  (shared with the generator)

GB = 10**9
REAL_MAX = 0.05     # per-episode image max (0..1) above which a camera has real frames
MOTION_PCTL = 5     # Language Table: drop episodes whose effector path is below this percentile
MV_WEIGHT = 1.0     # extra gain per usable episode with 2+ real cameras
TASK_WEIGHT = 1.0   # gain per instruction the selection does not cover yet


@dataclass(frozen=True)
class Source:
    repo: str
    revision: str            # from docs/organizers/data_description.txt
    budget_gb: float         # default subset size, video only
    min_len: int             # frames
    gripper_dim: int | None  # index of the gripper in `observation.state`; None = no gripper, use motion
    closed_high: bool = True # whether larger gripper values mean more closed
    motion_dims: tuple = ()  # `observation.state` dims whose path length measures motion
    mv_share: float = 0.0    # part of the budget reserved for multi-view episodes


SOURCES = {
    "bridge": Source("nvidia/BridgeData2_LeRobot_v3", "b96f7216e3cff58007884656a81584c857c185ae",
                     budget_gb=20.0, min_len=10, gripper_dim=6, closed_high=False, mv_share=0.5),
    "fractal": Source("BrunoM42/fractal20220817_data_lerobot", "fc006b5dc812220645a2e6dd0b68a0a03bd0ac6c",
                      budget_gb=1.1, min_len=10, gripper_dim=7),
    "language_table": Source("tailong-wu/language_table_lerobot_v30", "d54542e0eebf8084edb10e2c67ef5b94bd41fcc4",
                             budget_gb=0.5, min_len=5, gripper_dim=None, motion_dims=(0, 1)),
}


# metadata part

def _flat(arr, n: int) -> np.ndarray:
    """A (possibly nested) list column -> (n, k) float array."""
    import pyarrow as pa
    while pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type) or pa.types.is_fixed_size_list(arr.type):
        arr = arr.flatten()
    return np.asarray(arr, dtype=np.float64).reshape(n, -1)


def load_episodes(root: Path, video_keys: list[str], video_path: str) -> dict:
    """Per-episode arrays from meta/episodes: index, length, task, and for each video key its file and realness."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    files = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    cols = ["episode_index", "length", "tasks"]
    for k in video_keys:
        cols += [f"videos/{k}/chunk_index", f"videos/{k}/file_index", f"stats/{k}/max"]
    t = pa.concat_tables([pq.read_table(f, columns=cols) for f in files]).combine_chunks()
    n = t.num_rows
    ep = {
        "index": np.asarray(t["episode_index"]),
        "length": np.asarray(t["length"]),
        "task": np.array([(x[0] if x else "").strip() for x in t["tasks"].to_pylist()], dtype=object),
    }
    for k in video_keys:
        chunk, file = np.asarray(t[f"videos/{k}/chunk_index"]), np.asarray(t[f"videos/{k}/file_index"])
        ep[f"path/{k}"] = np.array([video_path.format(video_key=k, chunk_index=c, file_index=f)
                                    for c, f in zip(chunk, file)], dtype=object)
        ep[f"real/{k}"] = _flat(t[f"stats/{k}/max"].combine_chunks(), n).max(1) > REAL_MAX
    return ep


def _segments(ep_idx: np.ndarray):
    """For rows sorted by episode: start of each episode, episode id per start, segment id per row."""
    starts = np.flatnonzero(np.r_[True, ep_idx[1:] != ep_idx[:-1]])
    seg = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(ep_idx)]))
    return starts, ep_idx[starts], seg


def path_length(ep_idx: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """L1 path length of x (rows, dims) per episode. Rows sorted by episode."""
    starts, ids, seg = _segments(ep_idx)
    step = np.abs(np.diff(x, axis=0)).sum(1)
    same = seg[1:] == seg[:-1]
    return ids, np.bincount(seg[1:][same], weights=step[same], minlength=len(starts))


def activity(root: Path, src: Source, required: set[int], min_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """(episode ids, active flag) from data/*.parquet: the gripper goes through the required phases,
    or (no gripper) the effector path is above the dataset's MOTION_PCTL percentile."""
    import pyarrow.parquet as pq
    dims = [src.gripper_dim] if src.gripper_dim is not None else list(src.motion_dims)
    eps, vals = [], []
    for f in sorted((root / "data").glob("*/*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "observation.state"]).combine_chunks()
        eps.append(np.asarray(t["episode_index"]))
        vals.append(_flat(t["observation.state"].combine_chunks(), t.num_rows)[:, dims])
    ep_idx, x = np.concatenate(eps), np.concatenate(vals)
    order = np.argsort(ep_idx, kind="stable")
    ep_idx, x = ep_idx[order], x[order]
    if src.gripper_dim is None:
        ids, d = path_length(ep_idx, x)
        return ids, d >= np.percentile(d, MOTION_PCTL)
    starts, ids, _ = _segments(ep_idx)
    if not required:
        return ids, np.ones(len(ids), bool)
    g = x[:, 0]
    lo, hi = np.percentile(g, [1, 99])
    ok = np.zeros(len(ids), bool)
    for j, (s, e) in enumerate(zip(starts, np.r_[starts[1:], len(g)])):
        c = phases.closedness(g[s:e], src.closed_high, hi - lo)
        ok[j] = c is not None and phases.has_phases(phases.label(c), required, min_frames)
    return ids, ok


def usable_mask(ep: dict, act_ids: np.ndarray, active: np.ndarray, src: Source) -> np.ndarray:
    pos = np.searchsorted(act_ids, ep["index"])
    found = (pos < len(act_ids)) & (act_ids[np.minimum(pos, len(act_ids) - 1)] == ep["index"])
    on = np.zeros(len(ep["index"]), bool)
    on[found] = active[pos[found]]
    return (ep["task"] != "") & (ep["length"] >= src.min_len) & on


# files selection

def select_files(primary: np.ndarray, tasks: np.ndarray, usable: np.ndarray, multiview: np.ndarray,
                 extra: list[set], sizes: dict[str, int], budget: int, mv_share: float = 0.0) -> list[str]:
    """Greedy pick of primary-camera files by gain per byte while the total stays within budget.

    primary: (n,) primary video path per episode; extra: per episode, the other-camera paths it needs.
    The cost of a file includes the other-camera files its usable episodes need and that are not chosen yet.
    The first mv_share of the budget goes to files with multi-view episodes, counting only those.
    """
    cands = {}
    for f in sorted(set(primary[usable])):
        m = usable & (primary == f)
        cands[f] = (int(m.sum()), int((m & multiview).sum()), set(tasks[m]),
                    set().union(*(extra[i] for i in np.flatnonzero(m))))
    state = {"chosen": [], "have": set(), "covered": set(), "spent": 0}
    if mv_share > 0:
        _greedy(cands, sizes, int(budget * mv_share), state, w_ep=0.0, w_mv=1.0)
    _greedy(cands, sizes, budget, state, w_ep=1.0, w_mv=MV_WEIGHT)
    return state["chosen"]


def _greedy(cands: dict, sizes: dict[str, int], budget: int, state: dict, w_ep: float, w_mv: float) -> None:
    while True:
        best, best_r, best_c = None, 0.0, 0
        for f, (n, n_mv, f_tasks, need) in cands.items():  # sorted: ties keep the lower file index
            if w_ep == 0 and n_mv == 0:
                continue
            cost = sizes[f] + sum(sizes[p] for p in need - state["have"])
            gain = w_ep * n + w_mv * n_mv + TASK_WEIGHT * len(f_tasks - state["covered"])
            if state["spent"] + cost <= budget and gain > 0 and gain / cost > best_r:
                best, best_r, best_c = f, gain / cost, cost
        if best is None:
            return
        n, n_mv, f_tasks, need = cands.pop(best)
        state["chosen"].append(best)
        state["have"] |= need | {best}
        state["covered"] |= f_tasks
        state["spent"] += best_c


# actual download

def video_sizes(src: Source, video_keys: list[str]) -> dict[str, int]:
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile
    api, sizes = HfApi(), {}
    for k in video_keys:
        for f in api.list_repo_tree(src.repo, repo_type="dataset", revision=src.revision,
                                    path_in_repo=f"videos/{k}", recursive=True):
            if isinstance(f, RepoFile):
                sizes[f.path] = f.size
    return sizes


def download(src: Source, root: Path, patterns: list[str], workers: int) -> None:
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=src.repo, repo_type="dataset", revision=src.revision, local_dir=root,
                      allow_patterns=patterns, max_workers=workers)


def match_owner(path: Path, ref: Path) -> None:
    """The dev container runs as root; give what we wrote to the owner of the data dir on the host."""
    if os.geteuid() != 0 or not ref.exists():
        return
    st = ref.stat()
    for p in [path, *path.rglob("*")]:
        os.chown(p, st.st_uid, st.st_gid, follow_symlinks=False)


def run(name: str, src: Source, out: Path, video: str, budget_gb: float, required: tuple[str, ...],
        min_frames: int, dry_run: bool, workers: int) -> None:
    """video: 'subset' (within budget_gb), 'full' or 'none'. required: grasp phases an episode must go through."""
    root = out / name
    print(f"== {name}: {src.repo}@{src.revision[:8]} -> {root}", flush=True)
    download(src, root, ["README.md", "meta/*", "data/*"], workers)
    info = json.loads((root / "meta" / "info.json").read_text())
    if info.get("codebase_version") != "v3.0":
        sys.exit(f"{name}: expected LeRobot v3.0, got {info.get('codebase_version')}")
    keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    sizes = video_sizes(src, keys) if video != "none" else {}

    ep = load_episodes(root, keys, info["video_path"])
    act_ids, active = activity(root, src, {phases.PHASES.index(p) for p in required}, min_frames)
    usable = usable_mask(ep, act_ids, active, src)
    real = np.stack([ep[f"real/{k}"] for k in keys], 1)
    multiview = real.sum(1) >= 2
    primary = ep[f"path/{keys[0]}"]

    if video == "none":
        files, local = [], np.ones(len(primary), bool)
    elif video == "full":
        files, local = sorted(sizes), np.ones(len(primary), bool)
    else:
        extra = [{ep[f"path/{k}"][i] for k in keys[1:] if ep[f"real/{k}"][i]} for i in range(len(primary))]
        chosen = select_files(primary, ep["task"], usable, multiview, extra, sizes, int(budget_gb * GB), src.mv_share)
        if not chosen:
            sys.exit(f"{name}: no file fits in {budget_gb} GB (smallest primary file: "
                     f"{min(sizes[p] for p in set(primary)) / GB:.2f} GB)")
        local = np.isin(primary, chosen)
        files = sorted(set(chosen).union(*(extra[i] for i in np.flatnonzero(local & usable))))

    sel = local & usable
    total = sum(sizes[f] for f in files)
    what = (f"phases {'+'.join(required)} >= {min_frames}f" if required else "no phase filter") \
        if src.gripper_dim is not None else f"motion >= p{MOTION_PCTL}"
    print(f"   {what}: {active.mean():.0%} of episodes pass")
    print(f"   episodes {len(primary)}, usable {usable.sum()} ({usable.mean():.0%}); "
          f"selected {sel.sum()} usable, {(sel & multiview).sum()} multi-view, "
          f"{len(set(ep['task'][sel]))}/{len(set(ep['task'][usable]))} instructions; "
          + (f"{len(files)} video files ({len(set(primary[local]))} of {keys[0]}), {total / GB:.2f} GB"
             if video != "none" else "no video"))
    manifest = {
        "dataset": name, "repo": src.repo, "revision": src.revision,
        "mode": video, "budget_gb": budget_gb if video == "subset" else None,
        "mv_share": src.mv_share if video == "subset" else None,
        "video_keys": keys, "files": files, "bytes": total,
        "rules": {"min_len": src.min_len, "gripper_dim": src.gripper_dim, "closed_high": src.closed_high,
                  "phases": list(required) if src.gripper_dim is not None else None, "min_phase_frames": min_frames,
                  "grasp_threshold": phases.GRASP_THRESHOLD, "contact_threshold": phases.CONTACT_THRESHOLD,
                  "min_gripper_range": phases.MIN_RANGE, "motion_dims": list(src.motion_dims),
                  "motion_pctl": MOTION_PCTL, "real_max": REAL_MAX, "mv_weight": MV_WEIGHT, "task_weight": TASK_WEIGHT},
        "episodes": {"usable": ep["index"][sel].tolist(), "multiview": ep["index"][sel & multiview].tolist()},
    }
    (root / "selection.json").write_text(json.dumps(manifest, indent=1) + "\n")
    if files and not dry_run:
        download(src, root, files, workers)
    match_owner(root, out.parent)


def default_out() -> Path:
    # In the dev container /data is DATA_DIR on the host; outside it, use the repo's data/.
    return Path("/data/raw") if Path("/data").is_dir() else REPO / "data" / "raw"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", choices=SOURCES, default=list(SOURCES))
    p.add_argument("--full", nargs="+", choices=SOURCES, default=[], help="download these completely")
    p.add_argument("--no-video", nargs="*", choices=SOURCES, metavar="NAME",
                   help="skip video (meta/ and data/ only) for these, or for all datasets if none are named")
    p.add_argument("--budget-gb", nargs="+", default=[], metavar="NAME=GB",
                   help="video budget per subset dataset (defaults: " +
                        ", ".join(f"{k}={v.budget_gb:g}" for k, v in SOURCES.items()) + ")")
    p.add_argument("--phases", nargs="+", choices=phases.PHASES + ("none",), metavar="PHASE",
                   default=["pre_grasp", "contact", "post_grasp"],
                   help=f"grasp phases a gripper episode must go through, from {', '.join(phases.PHASES)}; "
                        "'none' drops the requirement (default: pre_grasp contact post_grasp)")
    p.add_argument("--min-phase-frames", type=int, default=2, help="shortest run that counts as a phase (default 2)")
    p.add_argument("--out", type=Path, default=default_out())
    p.add_argument("--dry-run", action="store_true", help="fetch metadata, write selection.json, print the plan")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()

    budgets = {k: v.budget_gb for k, v in SOURCES.items()}
    for item in a.budget_gb:
        k, _, v = item.partition("=")
        if k not in SOURCES or not v:
            p.error(f"--budget-gb: expected NAME=GB with NAME in {list(SOURCES)}, got {item!r}")
        budgets[k] = float(v)
    if "none" in a.phases and len(a.phases) > 1:
        p.error("--phases none cannot be combined with phase names")
    required = () if a.phases == ["none"] else tuple(a.phases)
    no_video = list(SOURCES) if a.no_video == [] else a.no_video or []
    if both := set(a.full) & set(no_video):
        p.error(f"--full and --no-video both name {sorted(both)}")
    names = a.datasets + a.full + (a.no_video or [])  # naming a dataset in --full / --no-video selects it
    for name in dict.fromkeys(names):
        video = "full" if name in a.full else "none" if name in no_video else "subset"
        run(name, SOURCES[name], a.out, video, budgets[name], required, a.min_phase_frames, a.dry_run, a.workers)


if __name__ == "__main__":
    main()
