"""
VoRA Qwen3.5 — standalone inference test
Usage:
    python -m my_vora_omni.tests.test_hf <checkpoint_path> [--encoder vitl|vitg]
"""
import argparse
import os
import re

import torch
from my_vora_omni.src.model import (
    Qwen3_5VJEPALModel,
    Qwen3_5VJEPAGModel,
    Qwen3_5VJEPA21BModel,
    Qwen3_5VJEPA21LModel,
    Qwen3_5VJEPA21GModel
)
from my_vora_omni.src.processor import (
    Qwen3VLVJepa2LProcessor,
    Qwen3VLVJepa2GProcessor,
    Qwen3VLVJepa21BProcessor,
    Qwen3VLVJepa21LProcessor,
    Qwen3VLVJepa21GProcessor
)


def load_model(checkpoint: str, encoder: str, device: str = "cuda"):
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

    processor = ProcCls.from_pretrained(checkpoint, trust_remote_code=True)
    model = ModelCls.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()
    return model, processor


def _post_process(text: str) -> str:
    # <think>...</think> 블록 제거 (enable_thinking=False여도 간혹 누출)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # 앞뒤 공백 제거
    text = text.strip()
    # 3개 이상 연속 줄바꿈 → 2개로 축약
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 문장 내 과도한 공백 정리
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text


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
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return _post_process(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to merged checkpoint")
    parser.add_argument("--encoder", default="vitl", choices=["vitb", "vitl", "vitg", "vjepa21l", "vjepa21g"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--nframes", type=int, default=32, help="Max video frames (must match FPS_MAX_FRAMES used at training)")
    args = parser.parse_args()

    # FPS_MAX_FRAMES는 VJEPAVideoProcessor.__init__에서 self.num_frames를 결정하므로
    # 프로세서 생성 전에 설정해야 합니다.
    os.environ["FPS_MAX_FRAMES"] = str(args.nframes)

    print(f"Loading model from {args.checkpoint} (encoder={args.encoder})...")
    model, processor = load_model(args.checkpoint, args.encoder, device=args.device)
    print("Model loaded.\n")

    # ── Test 3: Video ──────────────────────────────────────────
    print("=" * 60)
    print("[Test 3] Video")
    print("=" * 60)
    video_path = "/home/jw09191/tmp/video/ai_tf/052_2_ybw.mp4"
    if os.path.exists(video_path):
        messages = [
            {
                "role": "system",
                "content": "You are a highly accurate video analysis AI. You must describe only what is explicitly visible in the video. Do not guess, assume, or hallucinate any details that are not present."
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "nframes": args.nframes},
                    {"type": "text", "text": "Please describe the main events and visual details of this video."},
                ],
            }
        ]
        response = generate(model, processor, messages, args.max_new_tokens)
        print(f"Response: {response}\n")
    else:
        print(f"Skipped (file not found: {video_path})\n")

    print("All tests done.")


if __name__ == "__main__":
    main()
