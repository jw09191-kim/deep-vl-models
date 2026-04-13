#!/usr/bin/env python3
"""
Video-MME Benchmark Evaluation for VoRA models.

Dataset: https://huggingface.co/datasets/lmms-lab/Video-MME
Videos must be downloaded separately and placed in --video_dir.
Video filenames must match {video_id}.mp4.

Usage:
    python src/tests/eval_video_mme.py \
        --model_path /path/to/checkpoint \
        --model_type vitl \
        --video_dir /path/to/videos \
        --output_path results/video_mme_results.json \
        --num_frames 16
"""

import os
import sys
import json
import argparse
import re
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
from tqdm import tqdm
from my_vora_omni.src.register import *

# Project root (my_vora_omni/../)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────
# Model / Processor registry
# ─────────────────────────────────────────────
REGISTRY = {
    "vitl":     ("Qwen3VLVJepa2LProcessor",  "Qwen3_5VJEPALModel"),
    "vitg":     ("Qwen3VLVJepa2GProcessor",  "Qwen3_5VJEPAGModel"),
    "vjepa21b": ("Qwen3VLVJepa21BProcessor", "Qwen3_5VJEPA21BModel"),
    "vjepa21l": ("Qwen3VLVJepa21LProcessor", "Qwen3_5VJEPA21LModel"),
    "vjepa21g": ("Qwen3VLVJepa21GProcessor", "Qwen3_5VJEPA21GModel"),
}

ANSWER_RE = re.compile(r'\b([A-D])\b')

QUESTION_TEMPLATE = (
    "{question}\n"
    "A. {A}\n"
    "B. {B}\n"
    "C. {C}\n"
    "D. {D}\n"
    "Answer with the option's letter from the given choices directly."
)


# ─────────────────────────────────────────────
# Frame sampling
# ─────────────────────────────────────────────
def sample_frames_decord(video_path: str, num_frames: int) -> np.ndarray:
    """Return float32 [T, H, W, C] array sampled uniformly from the video."""
    import decord
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=2)
    total = len(vr)
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices).numpy()  # [T, H, W, C]  uint8
    frames = frames.astype(np.float32)
    # AV1 decode failures (e.g. missing HW accel) return silent zero frames.
    if frames.max() < 1.0:
        raise ValueError("decord returned blank frames — possible AV1 decode failure")
    return frames


def sample_frames_pyav(video_path: str, num_frames: int) -> np.ndarray:
    """PyAV-based frame extraction; handles AV1 via FFmpeg software decoding."""
    import av

    with av.open(video_path) as container:
        stream = container.streams.video[0]

        # Duration in seconds (AV_TIME_BASE = 1 000 000 µs)
        if container.duration:
            duration_sec = container.duration / 1_000_000
        elif stream.duration and stream.time_base:
            duration_sec = float(stream.duration * stream.time_base)
        else:
            duration_sec = 60.0

        timestamps_sec = np.linspace(0, duration_sec * 0.99, num_frames)
        frames_out = []

        for ts in timestamps_sec:
            container.seek(int(ts * 1_000_000), backward=True, any_frame=False)
            for frame in container.decode(stream):
                frames_out.append(frame.to_ndarray(format="rgb24").astype(np.float32))
                break  # one frame per seek point

    if not frames_out:
        raise ValueError("PyAV extracted no frames")

    # Pad to requested count if seeks came up short
    while len(frames_out) < num_frames:
        frames_out.append(frames_out[-1])

    return np.stack(frames_out[:num_frames])  # [T, H, W, C]


def sample_frames_cv2(video_path: str, num_frames: int) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = max(total, 1)
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        frames.append(frame.astype(np.float32))
    cap.release()
    return np.stack(frames)  # [T, H, W, C]


def sample_frames(video_path: str, num_frames: int) -> np.ndarray:
    """Try decoders in order: decord → PyAV → cv2."""
    for decoder in (sample_frames_decord, sample_frames_pyav, sample_frames_cv2):
        try:
            return decoder(video_path, num_frames)
        except Exception:
            continue
    raise RuntimeError(f"All video decoders failed for: {video_path}")


# ─────────────────────────────────────────────
# Preprocessing: frames → processor input
# ─────────────────────────────────────────────
def frames_to_video_tensor(frames: np.ndarray, num_frames: int) -> torch.Tensor:
    """
    frames: [T, H, W, C] float32  (0–255)
    Returns: [1, T_padded, C, H, W] uint8 tensor  (tubelet-aligned)
    """
    T = frames.shape[0]
    # Ensure T is divisible by tubelet_size=2
    if T % 2 != 0:
        last = frames[-1:]
        frames = np.concatenate([frames, last], axis=0)
        T = frames.shape[0]

    # [T, H, W, C] -> [T, C, H, W]
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)  # uint8-like but float
    return tensor  # will be rescaled inside processor


# ─────────────────────────────────────────────
# Answer parsing
# ─────────────────────────────────────────────
def parse_answer(text: str) -> str:
    """Extract first A/B/C/D from model output."""
    text = text.strip()
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1)
    # fallback: first character if it looks like a letter
    if text and text[0] in "ABCD":
        return text[0]
    return "X"  # invalid / unknown


# ─────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────
def load_videomme_hf(split: str = "test"):
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/Video-MME", split=split, trust_remote_code=True)
    samples = []
    for row in ds:
        options_raw = row["options"]  # ["A. ...", "B. ...", "C. ...", "D. ..."]
        opts = {}
        for ch in options_raw:
            letter, _, text = ch.partition(". ")
            opts[letter.strip()] = text.strip()
        samples.append({
            "video_id":    row["video_id"],
            "duration":    row.get("duration", ""),
            "domain":      row.get("domain", ""),
            "sub_category": row.get("sub_category", ""),
            "task_type":   row.get("task_type", ""),
            "question_id": row.get("question_id", ""),
            "question":    row["question"],
            "A": opts.get("A", ""), "B": opts.get("B", ""),
            "C": opts.get("C", ""), "D": opts.get("D", ""),
            "answer":      row["answer"],
        })
    return samples


def load_videomme_local(json_path: str):
    """
    Supports two formats:
      1. List of flat dicts (each row = one question)
      2. List of dicts with nested 'questions' list (original Video-MME JSON)
    """
    with open(json_path) as f:
        data = json.load(f)

    samples = []
    for item in data:
        if "questions" in item:
            # nested format
            for q in item["questions"]:
                choices_raw = q.get("choices", [])
                opts = {}
                for ch in choices_raw:
                    letter, _, text = ch.partition(". ")
                    opts[letter.strip()] = text.strip()
                samples.append({
                    "video_id":    item["video_id"],
                    "duration":    item.get("duration", ""),
                    "domain":      item.get("domain", ""),
                    "task_type":   q.get("task_type", ""),
                    "question_id": q.get("question_id", ""),
                    "question":    q["question"],
                    "A": opts.get("A", ""), "B": opts.get("B", ""),
                    "C": opts.get("C", ""), "D": opts.get("D", ""),
                    "answer":      q["answer"],
                })
        else:
            # flat format
            choices_raw = item.get("choices", [])
            opts = {}
            for ch in choices_raw:
                letter, _, text = ch.partition(". ")
                opts[letter.strip()] = text.strip()
            samples.append({
                "video_id":    item["video_id"],
                "duration":    item.get("duration", ""),
                "domain":      item.get("domain", ""),
                "task_type":   item.get("task_type", ""),
                "question_id": item.get("question_id", ""),
                "question":    item["question"],
                "A": opts.get("A", ""), "B": opts.get("B", ""),
                "C": opts.get("C", ""), "D": opts.get("D", ""),
                "answer":      item["answer"],
            })
    return samples


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────
def load_model_and_processor(model_path: str, model_type: str, device: str):
    PROC_MAP = {
        "vitl":     Qwen3VLVJepa2LProcessor,
        "vitg":     Qwen3VLVJepa2GProcessor,
        "vjepa21b": Qwen3VLVJepa21BProcessor,
        "vjepa21l": Qwen3VLVJepa21LProcessor,
        "vjepa21g": Qwen3VLVJepa21GProcessor,
    }
    MODEL_MAP = {
        "vitl":     Qwen3_5VJEPALModel,
        "vitg":     Qwen3_5VJEPAGModel,
        "vjepa21b": Qwen3_5VJEPA21BModel,
        "vjepa21l": Qwen3_5VJEPA21LModel,
        "vjepa21g": Qwen3_5VJEPA21GModel,
    }

    proc_cls  = PROC_MAP[model_type]
    model_cls = MODEL_MAP[model_type]

    print(f"Loading processor from {model_path}...")
    processor = proc_cls.from_pretrained(model_path)

    print(f"Loading model ({model_cls.__name__}) from {model_path}...")
    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    return model, processor


# ─────────────────────────────────────────────
# Single-sample inference
# ─────────────────────────────────────────────
@torch.inference_mode()
def run_inference(model, processor, video_tensor: torch.Tensor, question_text: str,
                  device: str, max_new_tokens: int = 16) -> str:
    """
    video_tensor: [T, C, H, W]  float32  (raw pixel values 0-255)
    Returns the raw model output string.
    """
    # Build chat messages
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_tensor,  # [T, C, H, W]
                    "fps": 1.0,
                },
                {"type": "text", "text": question_text},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Process inputs
    inputs = processor(
        text=[text],
        videos=[video_tensor],
        return_tensors="pt",
        padding=True,
    )
    # num_soft_tokens_* are training-only metadata; strip them before generate()
    _TRAINING_ONLY_KEYS = {"num_soft_tokens_per_image", "num_soft_tokens_per_video"}
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
              for k, v in inputs.items() if k not in _TRAINING_ONLY_KEYS}

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    new_ids = generated_ids[:, input_len:]
    output = processor.tokenizer.decode(new_ids[0], skip_special_tokens=True)
    return output.strip()


# ─────────────────────────────────────────────
# Accuracy helpers
# ─────────────────────────────────────────────
def compute_accuracy(results: list) -> dict:
    by_duration = defaultdict(lambda: {"correct": 0, "total": 0})
    by_domain   = defaultdict(lambda: {"correct": 0, "total": 0})
    overall     = {"correct": 0, "total": 0}

    for r in results:
        correct = r["predicted"] == r["answer"]
        dur = r.get("duration", "unknown")
        dom = r.get("domain", "unknown")

        by_duration[dur]["total"] += 1
        by_domain[dom]["total"] += 1
        overall["total"] += 1

        if correct:
            by_duration[dur]["correct"] += 1
            by_domain[dom]["correct"] += 1
            overall["correct"] += 1

    def acc(d):
        return round(d["correct"] / d["total"] * 100, 2) if d["total"] > 0 else 0.0

    return {
        "overall":     {"accuracy": acc(overall), **overall},
        "by_duration": {k: {"accuracy": acc(v), **v} for k, v in sorted(by_duration.items())},
        "by_domain":   {k: {"accuracy": acc(v), **v} for k, v in sorted(by_domain.items())},
    }


def print_summary(metrics: dict):
    print("\n" + "=" * 55)
    print("  Video-MME Results")
    print("=" * 55)
    ov = metrics["overall"]
    print(f"  Overall  : {ov['accuracy']:5.2f}%  ({ov['correct']}/{ov['total']})")
    print("-" * 55)
    for dur, v in metrics["by_duration"].items():
        print(f"  {dur:<10}: {v['accuracy']:5.2f}%  ({v['correct']}/{v['total']})")
    print("=" * 55)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Video-MME evaluation for VoRA")
    p.add_argument("--model_path", required=True, help="Path to trained model checkpoint")
    p.add_argument("--model_type", required=True, choices=list(REGISTRY.keys()),
                   help="Encoder type: vitl | vitg | vjepa21b | vjepa21l | vjepa21g")
    p.add_argument("--video_dir", required=True,
                   help="Directory containing {video_id}.mp4 files")
    p.add_argument("--output_path", default="results/video_mme_results.json",
                   help="Where to save detailed results JSON")
    p.add_argument("--data_path", default=None,
                   help="Local Video-MME JSON (optional; loads from HF if not set)")
    p.add_argument("--num_frames", type=int, default=16,
                   help="Number of frames to sample per video (default: 16)")
    p.add_argument("--max_new_tokens", type=int, default=16,
                   help="Max tokens to generate per answer (default: 16)")
    p.add_argument("--device", default="cuda",
                   help="Device: 'cuda', 'cpu', or 'auto' (default: cuda)")
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only the first N samples (for quick testing)")
    p.add_argument("--skip_missing", action="store_true",
                   help="Skip samples whose video file is not found")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(Path(args.output_path).parent, exist_ok=True)

    # ── Load dataset ──
    if args.data_path:
        print(f"Loading Video-MME from local file: {args.data_path}")
        samples = load_videomme_local(args.data_path)
    else:
        print("Loading Video-MME from HuggingFace (lmms-lab/Video-MME)...")
        samples = load_videomme_hf()

    if args.limit:
        samples = samples[: args.limit]
    print(f"Total questions: {len(samples)}")

    # ── Load model ──
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model_and_processor(args.model_path, args.model_type, device)

    # ── Evaluate ──
    results = []
    video_cache: dict = {}  # cache frames per video_id to avoid re-reading

    start_time = time.time()
    for sample in tqdm(samples, desc="Evaluating"):
        video_id  = sample["video_id"]
        video_path = Path(args.video_dir) / f"{video_id}.mp4"
        
        if not video_path.exists():
            if args.skip_missing:
                continue
            # try without extension
            alts = list(Path(args.video_dir).glob(f"{video_id}.*"))
            if alts:
                video_path = alts[0]
            else:
                print(f"\n[WARN] Video not found: {video_path}  — skipping")
                continue

        # Sample frames (cached per video)
        if video_id not in video_cache:
            try:
                frames = sample_frames(str(video_path), args.num_frames)
            except Exception as e:
                print(f"\n[WARN] Failed to load {video_path}: {e}  — skipping")
                continue
            # [T, H, W, C] float32 -> [T, C, H, W] float32
            video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)
            video_cache[video_id] = video_tensor
        else:
            video_tensor = video_cache[video_id]

        # Build question text
        question_text = QUESTION_TEMPLATE.format(
            question=sample["question"],
            A=sample["A"], B=sample["B"], C=sample["C"], D=sample["D"],
        )

        # Inference
        try:
            raw_output = run_inference(
                model, processor, video_tensor, question_text,
                device=device, max_new_tokens=args.max_new_tokens,
            )
        except Exception as e:
            print(f"\n[WARN] Inference failed for {video_id}: {e}")
            raw_output = ""

        predicted = parse_answer(raw_output)
        correct   = predicted == sample["answer"]

        results.append({
            "video_id":    video_id,
            "question_id": sample.get("question_id", ""),
            "duration":    sample.get("duration", ""),
            "domain":      sample.get("domain", ""),
            "task_type":   sample.get("task_type", ""),
            "question":    sample["question"],
            "answer":      sample["answer"],
            "predicted":   predicted,
            "raw_output":  raw_output,
            "correct":     correct,
        })

    elapsed = time.time() - start_time

    # ── Compute and print metrics ──
    metrics = compute_accuracy(results)
    print_summary(metrics)
    print(f"\nElapsed: {elapsed:.1f}s  ({elapsed / max(len(results), 1):.2f}s/sample)")

    # ── Save results ──
    output = {
        "model_path":  args.model_path,
        "model_type":  args.model_type,
        "num_frames":  args.num_frames,
        "num_samples": len(results),
        "elapsed_sec": round(elapsed, 1),
        "metrics":     metrics,
        "results":     results,
    }
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved → {args.output_path}")


if __name__ == "__main__":
    main()
