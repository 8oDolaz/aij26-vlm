"""Model wrappers: Depth Anything 3 (multi-view relative depth) and SAM2.1 automatic masks (class-agnostic segments)."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


class DepthTeacher:
    """Depth Anything 3 any-view model (DA3-BASE: Apache-2.0).

    One call takes the views of one moment (Bridge: cameras 0, 1, 2) and predicts their depth jointly, so the views
    share one scale and each is informed by the others; a single view is plain monocular depth. Output per view:
    inverse depth at the view's own resolution (larger = closer, like Depth Anything V2's disparity, so the per-frame
    normalisation and gap thresholds apply unchanged) and DA3's per-pixel depth confidence.

    `src` is the depth_anything_3 source dir (the DA3 repo's src/). Only the network, config registry and input
    processor are imported, not depth_anything_3.api, whose export code needs pycolmap, open3d, moviepy and more.
    The network needs addict and omegaconf; xformers is optional.
    """

    def __init__(self, model_id: str, src: str | None = None, process_res: int = 504):
        if src and src not in sys.path:
            sys.path.insert(0, src)
        from depth_anything_3.cfg import create_object, load_config
        from depth_anything_3.registry import MODEL_REGISTRY
        from depth_anything_3.utils.io.input_processor import InputProcessor
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        path = Path(snapshot_download(model_id, allow_patterns=["config.json", "model.safetensors"]))
        name = json.loads((path / "config.json").read_text())["model_name"]
        self.net = create_object(load_config(MODEL_REGISTRY[name]))
        state = {k.removeprefix("model."): v for k, v in load_file(path / "model.safetensors").items()}
        # The released checkpoints lack some LayerNorms of the auxiliary ray (pose) head; the official loader is
        # non-strict too. Depth comes from the main head, so anything else missing is an error.
        missing, unexpected = self.net.load_state_dict(state, strict=False)
        bad = [k for k in missing if not k.startswith("head.scratch.output_conv2_aux.")] + unexpected
        if bad:
            raise RuntimeError(f"DA3 checkpoint mismatch: {bad[:10]}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net.to(self.device).eval()
        self.processor = InputProcessor()
        self.process_res = process_res

    @torch.inference_mode()
    def __call__(self, views: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """(inverse depth per view, confidence per view) for the RGB views of one moment, in input order."""
        x, _, _ = self.processor(views, None, None, self.process_res, "upper_bound_resize", sequential=True)
        x = x[None].to(self.device).float()  # (1, N, 3, H, W)
        dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type=self.device, dtype=dtype, enabled=self.device == "cuda"):
            out = self.net(x, None, None, [], False, False, "saddle_balanced")
        depth = out["depth"][0].float().clamp(min=1e-6).cpu().numpy()
        conf = out["depth_conf"][0].float().cpu().numpy()
        disp, confs = [], []
        for v, d, c in zip(views, depth, conf):
            size = (v.shape[1], v.shape[0])
            disp.append(cv2.resize(1.0 / d, size, interpolation=cv2.INTER_LINEAR))
            confs.append(cv2.resize(c, size, interpolation=cv2.INTER_LINEAR))
        return disp, confs


class Segmenter:
    """SAM2.1 automatic mask generation, flattened to one label map where the smallest covering mask wins.

    fp32 only: the transformers pipeline's NMS fails in fp16. About 2.3 s per 640x480 frame on a 5090.
    """

    def __init__(self, model_id: str, points_per_batch: int = 64, pred_iou_thresh: float = 0.86,
                 stability_score_thresh: float = 0.92):
        from transformers import pipeline
        self.pipe = pipeline("mask-generation", model=model_id, device=0 if torch.cuda.is_available() else -1)
        self.kwargs = dict(points_per_batch=points_per_batch, pred_iou_thresh=pred_iou_thresh,
                           stability_score_thresh=stability_score_thresh)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        from PIL import Image
        masks = [np.asarray(m, dtype=bool) for m in self.pipe(Image.fromarray(frame), **self.kwargs)["masks"]]
        labels = np.zeros(frame.shape[:2], np.int16)  # 0 = no mask
        for k in sorted(range(len(masks)), key=lambda k: -masks[k].sum()):  # largest first, smaller ones overwrite
            labels[masks[k]] = k + 1
        return labels
