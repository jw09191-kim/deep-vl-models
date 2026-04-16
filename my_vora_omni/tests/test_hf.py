"""
VoRA Qwen3.5 — standalone inference test
Usage:
    python test_infer.py <checkpoint_path> [--encoder vitl|vitg]
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath('.'))

import torch
from src.model import (
    Qwen3_5VJEPALModel,
    Qwen3_5VJEPAGModel,
    Qwen3_5VJEPA21BModel,
    Qwen3_5VJEPA21LModel,
    Qwen3_5VJEPA21GModel
)
from src.processor import (
    Qwen3VLVJepa2LProcessor,
    Qwen3VLVJepa2GProcessor,
    Qwen3VLVJepa21BProcessor,
    Qwen3VLVJepa21LProcessor,
    Qwen3VLVJepa21GProcessor
)


def load_model(checkpoint: str, encoder: str, base_model_path: str = "Qwen/Qwen3.5-0.8B", device: str = "cuda"):
    MODEL_MAP = {
        "vitb": (Qwen3_5VJEPA21BModel, Qwen3VLVJepa21BProcessor),
        "vitl": (Qwen3_5VJEPALModel, Qwen3VLVJepa2LProcessor),
        "vitg": (Qwen3_5VJEPAGModel, Qwen3VLVJepa2GProcessor),
        "vjepa21l": (Qwen3_5VJEPA21LModel, Qwen3VLVJepa21LProcessor),
        "vjepa21g": (Qwen3_5VJEPA21GModel, Qwen3VLVJepa21GProcessor),
    }

    if encoder not in MODEL_MAP:
        raise ValueError(f"지원하지 않는 encoder 타입입니다: {encoder}")

    ModelCls, ProcCls = MODEL_MAP[encoder]

    # 주의: Processor는 체크포인트가 아닌 원본 Base 모델 경로에서 불러옵니다.
    # Base 모델 경로를 사용하시는 실제 Qwen 모델 경로로 맞춰주세요.
    processor = ProcCls.from_pretrained(base_model_path, trust_remote_code=True)

    # 모델 가중치는 병합된 체크포인트에서 불러옵니다.
    model = ModelCls.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()
    return model, processor


@torch.no_grad()
def generate(model, processor, messages, max_new_tokens=256):
    for msg in messages:
        if isinstance(msg["content"], str):
            msg["content"] = [{"type": "text", "text": msg["content"]}]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.0
    )

    # Strip input tokens
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to merged checkpoint")
    parser.add_argument("--encoder", default="vitl", choices=["vitb", "vitl", "vitg", "vjepa21l", "vjepa21g"])
    parser.add_argument("--base_model", default="Qwen/Qwen3.5-0.8B", help="Base model path for processor")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint} (encoder={args.encoder})...")
    model, processor = load_model(args.checkpoint, args.encoder, base_model_path=args.base_model, device=args.device)
    print("Model loaded.\n")

    # ── Test 1: Text-only ──────────────────────────────────────
    print("=" * 60)
    print("[Test 1] Text-only")
    print("=" * 60)
    messages = [{"role": "user", "content": "Hi, how have you been?"}]
    response = generate(model, processor, messages, args.max_new_tokens)
    print(f"Response: {response}\n")

    # ── Test 2: Image ──────────────────────────────────────────
    print("=" * 60)
    print("[Test 2] Image")
    print("=" * 60)
    image_path = "/home/jw09191/tmp/video/ai_tf/003_video.mp4"
    if os.path.exists(image_path):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        response = generate(model, processor, messages, args.max_new_tokens)
        print(f"Response: {response}\n")
    else:
        print(f"Skipped (file not found: {image_path})\n")

    # ── Test 3: Video ──────────────────────────────────────────
    print("=" * 60)
    print("[Test 3] Video")
    print("=" * 60)
    video_path = "my_vora_omni/examples/01_dog.mp4"
    if os.path.exists(video_path):
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": "Describe this video."},
            ],
        }]
        response = generate(model, processor, messages, args.max_new_tokens)
        print(f"Response: {response}\n")
    else:
        print(f"Skipped (file not found: {video_path})\n")

    # ── Test 4: Image + Video ──────────────────────────────────
    print("=" * 60)
    print("[Test 4] Image + Video combined")
    print("=" * 60)
    if os.path.exists(image_path) and os.path.exists(video_path):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "video", "video": video_path},
                {"type": "text", "text": "Compare the image and the video."},
            ],
        }]
        response = generate(model, processor, messages, args.max_new_tokens)
        print(f"Response: {response}\n")
    else:
        print("Skipped (files not found)\n")

    print("All tests done.")


if __name__ == "__main__":
    main()
