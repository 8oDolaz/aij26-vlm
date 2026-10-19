#!/usr/bin/env python3
"""Interactive test tool to inspect detector quality and evaluate VLM answers on generated questions.

Usage:
  # 1. Inspect detections and generated questions on an episode (saves visualization image):
  python tools/test_qa_interactive.py --episode 13612

  # 2. Also run the SmolVLM2 model on the generated questions and compare predictions with Ground Truth:
  python tools/test_qa_interactive.py --episode 13612 --vlm

  # 3. Test multiple random episodes:
  python tools/test_qa_interactive.py --n 5 --vlm --out runs/preview_qa/
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "submission" / "src"))

from aijgen.depth import lerobot
from aijgen.generators.object_qa import ObjectQAGenerator
from aijgen.objects import (
    draw_sam_segmentation,
    extract_segments,
    compute_robot_arm_mask,
    filter_robot_arm_segments,
)


def load_vlm(model_id: str = "HuggingFaceTB/SmolVLM2-500M-Instruct"):
    """Loads SmolVLM2 model and processor from local cache."""
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print(f"\nLoading VLM ({model_id})...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    print("VLM loaded successfully.\n")
    return model, processor


def query_vlm(model, processor, image: Image.Image, prompt_text: str) -> str:
    """Queries SmolVLM2 with an image and text prompt."""
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text.replace("<image>\n", "").strip()},
            ],
        }
    ]

    chat_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=chat_prompt, images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated_texts = processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return generated_texts[0].strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=REPO / "data" / "raw" / "bridge", help="Path to raw dataset")
    parser.add_argument("--episode", type=int, default=13612, help="Episode index to test")
    parser.add_argument("--cam", default="observation.images.image_0", help="Camera view")
    parser.add_argument("--n", type=int, default=1, help="Number of episodes to test (if > 1, samples episodes)")
    parser.add_argument("--threshold", type=float, default=0.40, help="Detection score threshold")
    parser.add_argument("--vlm", action="store_true", help="Run SmolVLM2 model inference on questions")
    parser.add_argument("--vlm-model", default="HuggingFaceTB/SmolVLM2-500M-Instruct", help="VLM model identifier")
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "preview_qa", help="Output directory for visual overlays")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    vlm_model, vlm_processor = (None, None)
    if args.vlm:
        vlm_model, vlm_processor = load_vlm(args.vlm_model)

    episodes = lerobot.list_episodes(args.dataset, args.cam)
    ep_by_index = {e.index: e for e in episodes}

    if args.n > 1:
        test_indices = list(ep_by_index.keys())[:args.n]
    else:
        test_indices = [args.episode] if args.episode in ep_by_index else [episodes[0].index]

    gen_cfg = {
        "seed": 42,
        "sam_model": "facebook/sam2.1-hiera-small",
        "siglip_model": "google/siglip-base-patch16-256",
        "score_threshold": args.threshold,
        "frames_span": [0.05, 0.35],
        "frames_per_episode": 2,
    }
    generator = ObjectQAGenerator(gen_cfg)

    print("=" * 80)
    print("DETECTOR & VLM QUESTION EVALUATION TEST")
    print("=" * 80)

    total_questions = 0
    correct_matches = 0

    for ep_idx in test_indices:
        lep = ep_by_index[ep_idx]
        videos = {args.cam: lep}

        from aijgen.generators.base import Episode
        ep = Episode(
            dataset=args.dataset.name,
            index=lep.index,
            task=lep.task,
            length=lep.length,
            state=np.zeros((lep.length, 7)),
            fps=lep.fps,
        )

        print(f"\nEpisode {ep.index}: \"{ep.task}\" (length={ep.length} frames, fps={ep.fps:.1f})")

        questions = generator.generate(ep, generator.rng(ep), videos=videos)
        print(f"Generated {len(questions)} benchmark questions.")
        if not questions:
            print("  (No questions generated for this episode)")
            continue

        wanted_frames = sorted({f for q in questions for f in q.frames})
        frames_dict = lerobot.read_frames(lep, wanted_frames)

        generator._ensure_models()
        for fi in wanted_frames:
            frame_rgb = frames_dict[fi]
            labels = generator.segmenter(frame_rgb)
            segments = extract_segments(labels, min_area_px=60)
            arm_mask = compute_robot_arm_mask(frame_rgb)
            filtered_segs = filter_robot_arm_segments(segments, arm_mask)
            classified = generator.classifier.classify(
                frame_rgb, filtered_segs, vocabulary=generator.vocab, threshold=args.threshold
            )
            detected = [s for s in classified if s.get("label")]

            vis = draw_sam_segmentation(frame_rgb, labels, segments=classified, show_boxes=True, show_labels=True)
            vis_path = args.out / f"ep{ep.index:06d}_f{fi:04d}_annotated.jpg"
            cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            print(f"  [Saved visualization image: {vis_path}]")
            det_summary = ", ".join(f"{s['label']} ({s['score']:.2f})" for s in detected)
            print(f"  Detections: {det_summary}")

        for q_idx, q in enumerate(questions, 1):
            total_questions += 1
            fi = q.frames[0]
            frame_rgb = frames_dict[fi]
            pil_img = Image.fromarray(frame_rgb)

            benchmark = q.meta.get("benchmark", "qa")
            subtype = q.meta.get("subtype", "")

            print("-" * 80)
            print(f"Q{q_idx} [{benchmark.upper()} | {subtype}] (Frame {fi}):")
            print(q.question.replace("<image>\n", ""))
            print(f"Ground Truth (Detector): {q.answer}")

            if vlm_model is not None:
                prediction = query_vlm(vlm_model, vlm_processor, pil_img, q.question)
                pred_letter = prediction[:1].upper() if prediction else ""
                is_match = (pred_letter == q.answer.strip().upper())
                if is_match:
                    correct_matches += 1
                    status_str = "\033[92mMATCH (YES)\033[0m"
                else:
                    status_str = "\033[91mNO MATCH\033[0m"

                print(f"Model Prediction:       {prediction}  -->  {status_str}")
            print("-" * 80)

    print("\n" + "=" * 80)
    print(f"SUMMARY: Processed {len(test_indices)} episodes, {total_questions} questions.")
    if vlm_model is not None and total_questions > 0:
        acc = 100.0 * correct_matches / total_questions
        print(f"VLM Accuracy (Pre-training / Base SmolVLM2): {correct_matches}/{total_questions} ({acc:.1f}%)")
    print(f"Annotated images saved to: {args.out}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
