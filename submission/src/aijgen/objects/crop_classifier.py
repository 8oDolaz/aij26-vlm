"""Zero-shot semantic classification of SAM segment crops using SigLIP (ConceptFusion approach)."""
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import torch
except ImportError:
    torch = None

from aijgen.objects.scene_vocabulary import build_scene_queries, load_vocabulary_from_file


def _box_iou(box1: list[int | float], box2: list[int | float]) -> float:
    """Compute Intersection over Union (IoU) between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(1e-6, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1e-6, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def _box_io_min(box1: list[int | float], box2: list[int | float]) -> float:
    """Compute Intersection over Minimum area between two boxes (measures containment)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    min_area = min(
        max(1e-6, (box1[2] - box1[0]) * (box1[3] - box1[1])),
        max(1e-6, (box2[2] - box2[0]) * (box2[3] - box2[1])),
    )
    return inter / min_area if min_area > 0 else 0.0


class CropSegmentClassifier:
    """Zero-shot classifier that evaluates cropped SAM masks using SigLIP embeddings."""

    def __init__(
        self,
        model_id: str = "google/siglip-base-patch16-256",
        device: str | None = None,
        bg_alpha: float = 0.40,
        max_area_pct: float = 35.0,
    ):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.bg_alpha = bg_alpha
        self.max_area_pct = max_area_pct

        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

        self.is_siglip = "siglip" in model_id.lower()
        self._cached_vocab_hash: int | None = None
        self._cached_text_embeds: Any = None
        self._cached_queries: list[str] = []
        self._cached_map: dict[str, str] = {}

    def _ensure_text_cache(self, vocabulary: list[str] | None = None):
        """Precomputes and caches normalized text embeddings for candidate objects + negatives."""
        import torch

        vocab = vocabulary if vocabulary is not None else load_vocabulary_from_file()
        current_hash = hash(tuple(sorted(vocab)))

        if self._cached_vocab_hash == current_hash and self._cached_text_embeds is not None:
            return

        expanded_queries, query_to_canonical = build_scene_queries(vocab, include_distractors=True)
        all_queries = list(expanded_queries)

        with torch.no_grad():
            inputs = self.processor(
                text=all_queries,
                padding="max_length",
                max_length=64,
                return_tensors="pt",
            ).to(self.device)

            text_features = self.model.get_text_features(**inputs)
            if hasattr(text_features, "pooler_output"):
                text_features = text_features.pooler_output
            elif hasattr(text_features, "last_hidden_state"):
                text_features = text_features.last_hidden_state[:, 0, :]

            text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)

        self._cached_text_embeds = text_features
        self._cached_queries = all_queries
        self._cached_map = query_to_canonical
        self._cached_vocab_hash = current_hash

    def extract_crop(
        self,
        frame: np.ndarray,
        segment: dict,
        padding_pct: float = 0.15,
        min_crop_size: int = 5,
    ) -> Image.Image | None:
        """Extracts a padded crop around the segment with soft background attenuation."""
        h, w = frame.shape[:2]
        bx1, by1, bx2, by2 = [int(round(v)) for v in segment["box"]]

        bw = max(1, bx2 - bx1)
        bh = max(1, by2 - by1)
        pad_x = int(bw * padding_pct)
        pad_y = int(bh * padding_pct)

        cx1 = max(0, min(w - 1, bx1 - pad_x))
        cy1 = max(0, min(h - 1, by1 - pad_y))
        cx2 = max(cx1 + 1, min(w, bx2 + pad_x))
        cy2 = max(cy1 + 1, min(h, by2 + pad_y))

        if (cx2 - cx1) < min_crop_size or (cy2 - cy1) < min_crop_size:
            return None

        crop_rgb = frame[cy1:cy2, cx1:cx2].copy()
        if crop_rgb.size == 0 or crop_rgb.shape[0] < min_crop_size or crop_rgb.shape[1] < min_crop_size:
            return None

        mask = segment.get("mask")
        if mask is not None and self.bg_alpha < 1.0:
            crop_mask = mask[cy1:cy2, cx1:cx2]
            outside = ~crop_mask
            if outside.any():
                crop_rgb[outside] = (
                    crop_rgb[outside].astype(np.float32) * self.bg_alpha
                    + 240.0 * (1.0 - self.bg_alpha)
                ).astype(np.uint8)

        return Image.fromarray(crop_rgb)

    def classify(
        self,
        frame: np.ndarray,
        segments: list[dict],
        vocabulary: list[str] | None = None,
        threshold: float = 0.40,
        iou_thresh: float = 0.40,
    ) -> list[dict]:
        """Classifies each SAM segment crop against the vocabulary and applies NMS deduplication."""
        import torch

        if not segments:
            return segments

        self._ensure_text_cache(vocabulary)

        # 1. Prepare valid crops and discard 0-size / background segments
        valid_crops: list[Image.Image] = []
        valid_segments: list[dict] = []
        for s in segments:
            if self.max_area_pct and s.get("area_pct", 0) > self.max_area_pct:
                continue
            crop = self.extract_crop(frame, s)
            if crop is not None and crop.width >= 5 and crop.height >= 5:
                valid_crops.append(crop)
                valid_segments.append(s)

        if not valid_crops:
            return segments

        # 2. Extract image features in batch
        with torch.no_grad():
            img_inputs = self.processor(images=valid_crops, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**img_inputs)
            if hasattr(image_features, "pooler_output"):
                image_features = image_features.pooler_output
            elif hasattr(image_features, "last_hidden_state"):
                image_features = image_features.last_hidden_state[:, 0, :]

            image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)

            # Cosine similarity matrix: (num_valid_segments, num_queries)
            sim_matrix = torch.matmul(image_features, self._cached_text_embeds.T)

            if self.is_siglip and hasattr(self.model, "logit_scale"):
                scale = self.model.logit_scale.exp()
                bias = self.model.logit_bias if hasattr(self.model, "logit_bias") else 0.0
                scores_matrix = torch.sigmoid(sim_matrix * scale + bias)
            else:
                scores_matrix = torch.softmax(sim_matrix * 10.0, dim=-1)

            scores_np = scores_matrix.cpu().numpy()

        # 3. Candidate assignments
        candidate_detections = []
        for seg, crop_scores in zip(valid_segments, scores_np):
            best_target_score = -1.0
            best_target_canonical = None
            best_distractor_score = -1.0

            for q_idx, score in enumerate(crop_scores):
                q = self._cached_queries[q_idx]
                canonical = self._cached_map.get(q, q)
                score_val = float(score)

                if canonical == "__distractor__":
                    if score_val > best_distractor_score:
                        best_distractor_score = score_val
                else:
                    if score_val > best_target_score:
                        best_target_score = score_val
                        best_target_canonical = canonical

            if (
                best_target_canonical
                and best_target_score >= threshold
                and best_target_score >= best_distractor_score
            ):
                candidate_detections.append({
                    "segment": seg,
                    "label": best_target_canonical,
                    "score": round(best_target_score, 3),
                })

        # 4. NMS / Deduplication: sort by score descending, suppress heavily overlapping boxes
        candidate_detections.sort(key=lambda d: d["score"], reverse=True)
        kept_detections = []

        for cand in candidate_detections:
            c_box = cand["segment"]["box"]
            c_label = cand["label"]
            suppress = False

            for kept in kept_detections:
                k_box = kept["segment"]["box"]
                k_label = kept["label"]

                iou = _box_iou(c_box, k_box)
                io_min = _box_io_min(c_box, k_box)

                if c_label == k_label and (iou > iou_thresh or io_min > 0.65):
                    suppress = True
                    break

                if iou > 0.60 or io_min > 0.85:
                    suppress = True
                    break

            if not suppress:
                kept_detections.append(cand)

        # 5. Attach kept labels to original segments
        for item in kept_detections:
            item["segment"]["label"] = item["label"]
            item["segment"]["score"] = item["score"]

        return segments

    def ground(
        self,
        frame: np.ndarray,
        segments: list[dict],
        vocabulary: list[str] | None = None,
        threshold: float = 0.40,
        **kwargs: Any,
    ) -> list[dict]:
        """Drop-in compatible wrapper matching ground interface."""
        return self.classify(
            frame,
            segments,
            vocabulary=vocabulary,
            threshold=threshold,
        )
