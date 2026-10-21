"""Object and Spatial QA generator targeting EmbSpatial, ERQA, CrossPoint, and MSMU benchmarks.

Uses SAM 2.1 instance segmentation and SigLIP zero-shot crop classification to detect scene objects,
filter out the robot manipulator, and generate multiple-choice visual questions.
"""
from typing import Any

import numpy as np

from aijgen.depth import lerobot
from aijgen.depth.models import Segmenter
from aijgen.generators.base import Generator, Question, pick_frame
from aijgen.objects import (
    CropSegmentClassifier,
    compute_robot_arm_mask,
    extract_segments,
    filter_robot_arm_segments,
    get_pairwise_spatial_relations,
    load_vocabulary_from_file,
    parse_task_objects,
)


class ObjectQAGenerator(Generator):
    name = "object_qa"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.sam_model_id = cfg.get("sam_model", "facebook/sam2.1-hiera-small")
        self.siglip_model_id = cfg.get("siglip_model", "google/siglip-base-patch16-256")
        self.score_threshold = float(cfg.get("score_threshold", 0.40))
        self.frames_span = tuple(cfg.get("frames_span", [0.05, 0.35]))
        self.frames_per_episode = int(cfg.get("frames_per_episode", 2))
        self.min_frame_std = float(cfg.get("min_frame_std", 8.0))

        self.segmenter: Segmenter | None = None
        self.classifier: CropSegmentClassifier | None = None
        self.vocab = load_vocabulary_from_file()

    def _ensure_models(self):
        """Lazy initialization of SAM 2.1 and SigLIP classifiers."""
        if self.segmenter is None:
            from aijgen.paths import resolve
            self.segmenter = Segmenter(resolve(self.sam_model_id))
        if self.classifier is None:
            self.classifier = CropSegmentClassifier(self.siglip_model_id)

    def generate(
        self,
        ep,
        rng: np.random.Generator,
        videos: dict | None = None,
    ) -> list[Question]:
        """Generates benchmark-aligned MCQs for an episode using detected scene objects."""
        if not videos or ep.length < 5:
            return []

        self._ensure_models()

        # Target primary cameras (e.g. image_0 or first available camera)
        preferred_cams = [c for c in ["observation.images.image_0", "observation.images.image"] if c in videos]
        target_cam = preferred_cams[0] if preferred_cams else next(iter(videos.keys()))
        video = videos[target_cam]

        # Select frame indices from initial portion of episode where tabletop objects are undisturbed
        span_start = max(0, int(round(self.frames_span[0] * (ep.length - 1))))
        span_end = min(ep.length - 1, max(span_start + 1, int(round(self.frames_span[1] * (ep.length - 1)))))

        if span_start >= span_end:
            frame_indices = [span_start]
        else:
            candidates = list(range(span_start, span_end + 1))
            k = min(self.frames_per_episode, len(candidates))
            frame_indices = sorted([int(x) for x in rng.choice(candidates, size=k, replace=False)])

        # Decode selected frames
        decoded_frames = lerobot.read_frames(video, frame_indices)
        task_candidates = [t.lower() for t in parse_task_objects(ep.task)]

        questions: list[Question] = []

        for fi in frame_indices:
            frame = decoded_frames.get(fi)
            if frame is None or frame.std() < self.min_frame_std:
                continue

            fi_int = int(fi)

            # 1. Run instance segmentation + robot arm suppression + SigLIP classification
            labels = self.segmenter(frame)
            segments = extract_segments(labels, min_area_px=60)
            arm_mask = compute_robot_arm_mask(frame)
            segments = filter_robot_arm_segments(segments, arm_mask)
            classified = self.classifier.classify(frame, segments, vocabulary=self.vocab, threshold=self.score_threshold)

            detected_objects = [s for s in classified if s.get("label")]
            if not detected_objects:
                continue

            unique_detected_labels = sorted(list({s["label"] for s in detected_objects}))

            # -------------------------------------------------------------
            # BENCHMARK 1: EmbSpatial (Qualitative 2D Spatial Relationships)
            # -------------------------------------------------------------
            spatial_pairs = get_pairwise_spatial_relations(detected_objects, frame.shape)
            if spatial_pairs:
                pair = rng.choice(spatial_pairs)
                obj_a, obj_b = pair["obj_a"], pair["obj_b"]
                primary_rel = pair["primary_rel"]
                distractor_rels = pair["distractor_rels"]

                # EmbSpatial Type A: "Where is X located relative to Y?"
                q_text = f"Where is the {obj_a} located relative to the {obj_b}?"
                correct = f"{primary_rel.capitalize()} the {obj_b}"
                distractors = [f"{d.capitalize()} the {obj_b}" for d in distractor_rels[:3]]
                mcq_q, ans_letter = self.mcq(q_text, correct, distractors, rng)
                questions.append(Question(
                    question=mcq_q,
                    answer=ans_letter,
                    frames=[fi_int],
                    kind=self.name,
                    meta={"camera": target_cam, "benchmark": "embspatial", "subtype": "relative_direction"},
                ))

                # EmbSpatial Type B: "Which object is located [direction] the Y?"
                if primary_rel != "inside":
                    absent_objs = [v for v in self.vocab if v not in unique_detected_labels]
                    if len(absent_objs) >= 3:
                        q_text = f"Which object is located {primary_rel} the {obj_b}?"
                        correct = f"The {obj_a}"
                        distractors = [f"The {o}" for o in rng.choice(absent_objs, size=3, replace=False)]
                        mcq_q, ans_letter = self.mcq(q_text, correct, distractors, rng)
                        questions.append(Question(
                            question=mcq_q,
                            answer=ans_letter,
                            frames=[fi_int],
                            kind=self.name,
                            meta={"camera": target_cam, "benchmark": "embspatial", "subtype": "object_at_direction"},
                        ))

            # -------------------------------------------------------------
            # BENCHMARK 2: ERQA (Object Presence & Task Target Identification)
            # -------------------------------------------------------------
            # ERQA Type A: Object Presence Verification
            target_present = rng.choice(unique_detected_labels)
            absent_objs = [v for v in self.vocab if v not in unique_detected_labels]
            if len(absent_objs) >= 3:
                q_text = "Which of the following objects is present on the tabletop?"
                correct = f"The {target_present}"
                distractors = [f"The {o}" for o in rng.choice(absent_objs, size=3, replace=False)]
                mcq_q, ans_letter = self.mcq(q_text, correct, distractors, rng)
                questions.append(Question(
                    question=mcq_q,
                    answer=ans_letter,
                    frames=[fi_int],
                    kind=self.name,
                    meta={"camera": target_cam, "benchmark": "erqa", "subtype": "presence"},
                ))

            # ERQA Type B: Task-Grounded Object Identification
            matched_targets = [t for t in task_candidates if t in unique_detected_labels]
            if matched_targets and len(absent_objs) >= 3:
                task_target = matched_targets[0]
                q_text = f'The robot is tasked to "{ep.task}". Which object in the scene is being targeted for manipulation?'
                correct = f"The {task_target}"
                distractors = [f"The {o}" for o in rng.choice(absent_objs, size=3, replace=False)]
                mcq_q, ans_letter = self.mcq(q_text, correct, distractors, rng)
                questions.append(Question(
                    question=mcq_q,
                    answer=ans_letter,
                    frames=[fi_int],
                    kind=self.name,
                    meta={"camera": target_cam, "benchmark": "erqa", "subtype": "task_grounding"},
                ))

            # -------------------------------------------------------------
            # BENCHMARK 3: CrossPoint / MSMU (Counting & Relative Size)
            # -------------------------------------------------------------
            # CrossPoint / MSMU Type A: Object Counting
            count_val = len(detected_objects)
            if 1 <= count_val <= 6:
                q_text = "How many distinct target objects are visible on the tabletop?"
                correct = f"{count_val}"
                alt_counts = [c for c in [1, 2, 3, 4, 5, 6] if c != count_val]
                distractors = [f"{c}" for c in rng.choice(alt_counts, size=3, replace=False)]
                mcq_q, ans_letter = self.mcq(q_text, correct, distractors, rng)
                questions.append(Question(
                    question=mcq_q,
                    answer=ans_letter,
                    frames=[fi_int],
                    kind=self.name,
                    meta={"camera": target_cam, "benchmark": "msmu", "subtype": "counting"},
                ))

            # MSMU Type B: Relative Size Comparison
            if spatial_pairs:
                valid_size_pairs = [p for p in spatial_pairs if p.get("larger_obj")]
                if valid_size_pairs:
                    size_pair = rng.choice(valid_size_pairs)
                    obj_a, obj_b = size_pair["obj_a"], size_pair["obj_b"]
                    larger = size_pair["larger_obj"]
                    smaller = size_pair["smaller_obj"]

                    q_text = f"Comparing their visible sizes in the image, which object appears larger: the {obj_a} or the {obj_b}?"
                    correct = f"The {larger}"
                    distractors = [f"The {smaller}", "Both objects are approximately the same size"]
                    mcq_q, ans_letter = self.mcq(q_text, correct, distractors, rng)
                    questions.append(Question(
                        question=mcq_q,
                        answer=ans_letter,
                        frames=[fi_int],
                        kind=self.name,
                        meta={"camera": target_cam, "benchmark": "msmu", "subtype": "relative_size"},
                    ))

        return questions
