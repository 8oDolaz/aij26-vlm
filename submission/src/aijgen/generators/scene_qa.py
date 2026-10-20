"""Scene questions from SAM 3 detections (aijgen.scene; docs/sam3_vqa_plan.md).

Per episode, frames_per_episode source frames evenly spread over frames_span (early: objects not yet moved). At each,
the dataset's non-blank views are found and one of them is the question view, rotating with the episode index and the
frame's turn so cameras are used about evenly. SAM 3 runs on the question view only; DA3 runs on all non-blank views
of the moment together, and the question view's disparity gives each mask its depth.

Detections are cached per source frame in <work_dir>/cache/<dataset>/ (work_dir: <output>/scene_qa by default), so
re-running with other filter, scene or template settings loads no model.

With a `crossview` block (aijgen.scene.crossview), frames with at least two non-blank views also get object-level
cross-view questions: SAM 3 runs on a pair of views (rotating like the question view), plus every other non-blank
view when crossview.check_third_view is on. crossview.pre_grasp_only keeps only frames of the first pre_grasp run of
episodes that have phase labels (nothing in the gripper yet).

Model references (sam3_model, depth_model, da3_src) are Hugging Face ids or paths; a relative path that exists under
the submission root wins, so the packed ZIP can ship the weights next to the code. depth_model: null skips DA3
(templates that need depth then find no candidates); frames cached with depth serve both settings.
"""
import zlib
from pathlib import Path

import numpy as np

from aijgen import phases as ph
from aijgen.depth import lerobot
from aijgen.generators.base import Generator, Question
from aijgen.scene import detect as D
from aijgen.scene.crossview import make_pair_items
from aijgen.scene.templates import Scene, build_scene, make_items
from aijgen.scene.vocab import DICT_DIR, load_dictionary, load_names

SUBMISSION_DIR = Path(__file__).resolve().parents[3]


def resolve(ref: str | None) -> str | None:
    """`ref` as a path under the submission root if one exists there, else unchanged."""
    if not ref or Path(ref).is_absolute():
        return ref
    local = SUBMISSION_DIR / ref
    return str(local) if local.exists() else ref


class SceneQAGenerator(Generator):
    name = "scene_qa"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.work = Path(cfg["work_dir"]) if cfg.get("work_dir") else None
        self.filter = D.FilterParams(**cfg.get("filter", {}))
        self.scene_cfg = cfg.get("scene", {})
        self.templates = cfg.get("templates") or {}
        self.crossview = cfg.get("crossview") or {}
        self.with_depth = bool(cfg.get("depth_model"))
        self.datasets = cfg["datasets"]
        self.span = tuple(cfg.get("frames_span", [0.05, 0.35]))
        self.n_frames = int(cfg.get("frames_per_episode", 2))
        self.min_length = int(cfg.get("min_length", 10))
        self.min_std = float(cfg.get("min_frame_std", 8.0))
        det = cfg.get("detection", {})
        # A cached frame made with other values is recomputed.
        self.params = {"mask_min": float(det.get("mask_min", 0.5)),
                       "negatives_per_frame": int(det.get("negatives_per_frame", 8)),
                       "depth_erode_frac": float(det.get("depth_erode_frac", 0.01))}
        self._vocab = {}
        self._sam = self._depth = None

    def vocab(self, dataset: str):
        """(dictionary entries, absent-list names) of a dataset."""
        if dataset not in self._vocab:
            d = self.datasets[dataset]
            self._vocab[dataset] = (load_dictionary(DICT_DIR / d.get("dictionary", f"{dataset}.txt")),
                                    load_names(DICT_DIR / d.get("absent", f"{dataset}_absent.txt")))
        return self._vocab[dataset]

    def _models(self):
        if self._sam is None:
            self._sam = D.Sam3(resolve(self.cfg["sam3_model"]))
            if self.with_depth:
                from aijgen.depth.models import DepthTeacher
                self._depth = DepthTeacher(resolve(self.cfg["depth_model"]), resolve(self.cfg.get("da3_src")),
                                           int(self.cfg.get("da3_process_res", 504)))
        return self._sam, self._depth

    def cache_ok(self, det: D.FrameDetections | None, cam: str, entries) -> bool:
        """A cached frame is reused if it was made with the same detection params, names and camera, and has depth
        whenever depth is wanted (params["depth"] is False only on frames cached without DA3)."""
        if det is None or det.names != list(entries) or det.camera != cam:
            return False
        base = {k: v for k, v in det.params.items() if k != "depth"}
        return base == self.params and (det.params.get("depth", True) or not self.with_depth)

    def detections(self, ep, fi: int, cam: str, valid: list[str], frames: dict, entries, absent) -> D.FrameDetections:
        """From the cache, or SAM 3 on view `cam` plus DA3 on all valid views (then cached)."""
        path = None
        if self.work is not None:
            path = self.work / "cache" / ep.dataset / f"ep{ep.index:06d}_f{fi:04d}_{cam.split('.')[-1]}.npz"
            det = D.load(path)
            if self.cache_ok(det, cam, entries):
                return det
        neg_rng = np.random.default_rng([self.seed, zlib.crc32(ep.dataset.encode()), ep.index, fi])
        n = min(self.params["negatives_per_frame"], len(absent))
        negatives = [absent[i] for i in sorted(neg_rng.choice(len(absent), n, replace=False))]
        sam, depth = self._models()
        params = self.params if self.with_depth else {**self.params, "depth": False}
        det = D.detect(sam, frames[cam][fi], cam, entries, negatives, self.params["mask_min"], params)
        if self.with_depth:
            disps, _ = depth([frames[v][fi] for v in valid])
            D.add_depth(det, disps[valid.index(cam)], self.params["depth_erode_frac"])
        if path is not None:
            D.save(path, det)
        return det

    def scene(self, det: D.FrameDetections, entries) -> Scene:
        return build_scene(det, entries, self.filter,
                           exclude_subjects=self.scene_cfg.get("exclude_subjects", ()),
                           absent_presence=float(self.scene_cfg.get("absent_presence", 0.05)),
                           max_count=int(self.scene_cfg.get("max_count", 6)))

    def crossview_ok(self, ep, fi: int) -> bool:
        """crossview.pre_grasp_only: the frame is in the episode's first pre_grasp run (episodes without phase
        labels pass: their gripper hardly moves)."""
        if not self.crossview.get("pre_grasp_only", True) or ep.labels is None:
            return True
        p, s, e = ph.runs(ep.labels)[0]
        return p == ph.PRE_GRASP and s <= fi <= e

    def generate(self, ep, rng: np.random.Generator, videos: dict | None = None) -> list[Question]:
        dcfg = self.datasets.get(ep.dataset)
        if dcfg is None or not videos or ep.length < self.min_length:
            return []
        views = [v for v in dcfg["views"] if v in videos]
        if not views:
            return []
        lo, hi = self.span
        fis = sorted({int(round(x)) for x in np.linspace(lo * (ep.length - 1), hi * (ep.length - 1), self.n_frames)})
        frames = {v: lerobot.read_frames(videos[v], fis) for v in views}
        entries, absent = self.vocab(ep.dataset)
        out = []
        for turn, fi in enumerate(fis):
            valid = [v for v in views if fi in frames[v] and frames[v][fi].std() >= self.min_std]
            if not valid:  # blank cameras: the episode was recorded without them
                continue
            cam = valid[(ep.index + turn) % len(valid)]
            if self.templates:
                scene = self.scene(self.detections(ep, fi, cam, valid, frames, entries, absent), entries)
                for it in make_items(scene, rng, self.templates):
                    out.append(Question(it.question, it.answer, [fi], self.name,
                                        meta={"camera": cam, "bench": it.bench, "template": it.template},
                                        overlays=it.overlays))
            if self.crossview.get("templates") and len(valid) >= 2 and self.crossview_ok(ep, fi):
                out += self.crossview_questions(ep, fi, turn, valid, frames, entries, absent, rng)
        return out

    def crossview_questions(self, ep, fi, turn, valid, frames, entries, absent, rng) -> list[Question]:
        """Cross-view questions at one moment: image 1 = view a, image 2 = view b, the pair rotating over the valid
        views; the other valid views only veto names (crossview.matches)."""
        k = (ep.index + turn) % len(valid)
        a, b = valid[k], valid[(k + 1) % len(valid)]
        if rng.random() < 0.5:
            a, b = b, a
        scenes = {v: self.scene(self.detections(ep, fi, v, valid, frames, entries, absent), entries) for v in (a, b)}
        others = []
        if self.crossview.get("check_third_view", True):
            others = [self.scene(self.detections(ep, fi, v, valid, frames, entries, absent), entries)
                      for v in valid if v not in (a, b)]
        return [Question(it.question, it.answer, [fi, fi], self.name,
                         meta={"cameras": [a, b], "bench": it.bench, "template": it.template}, overlays=it.overlays)
                for it in make_pair_items(scenes[a], scenes[b], others, entries, rng, self.crossview["templates"])]
