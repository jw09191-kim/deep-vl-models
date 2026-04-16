"""
VoRA Qwen3.5 — standalone inference test
Usage:
    python -m my_vora_omni.tests.test_hf <checkpoint> --video a.mp4
    python -m my_vora_omni.tests.test_hf <checkpoint> --video "*.mp4" --save
"""
import argparse
import glob
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
    # URL 제거 (http/https/ftp/www 형태 모두 포함)
    text = re.sub(r'https?://\S+|ftp://\S+|www\.\S+', '', text)
    # 앞뒤 공백 제거
    text = text.strip()
    # 3개 이상 연속 줄바꿈 → 2개로 축약
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 문장 내 과도한 공백 정리 (URL 제거 후 빈 자리도 처리)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text


def _resolve_videos(pattern: str) -> list[str]:
    """단일 파일 경로 또는 glob 패턴을 받아 존재하는 파일 목록을 반환."""
    if os.path.isfile(pattern):
        return [pattern]
    matched = sorted(glob.glob(pattern, recursive=True))
    if not matched:
        print(f"Warning: no files matched '{pattern}'")
    return matched


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
    parser.add_argument("--video", required=True, help="Video file path or glob pattern (e.g. a.mp4 or '*.mp4')")
    parser.add_argument("--save", action="store_true", help="Save results as <video_name>.txt in an output directory")
    parser.add_argument("--output_dir", default="output_descriptions", help="Output directory when --save is used (default: output_descriptions)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--nframes", type=int, default=32, help="Max video frames (must match FPS_MAX_FRAMES used at training)")
    args = parser.parse_args()

    # FPS_MAX_FRAMES는 VJEPAVideoProcessor.__init__에서 self.num_frames를 결정하므로
    # 프로세서 생성 전에 설정해야 합니다.
    os.environ["FPS_MAX_FRAMES"] = str(args.nframes)

    video_paths = _resolve_videos(args.video)
    if not video_paths:
        return

    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Results will be saved to: {args.output_dir}/\n")

    print(f"Loading model from {args.checkpoint} (encoder={args.encoder})...")
    model, processor = load_model(args.checkpoint, args.encoder, device=args.device)
    print(f"Model loaded. Processing {len(video_paths)} video(s).\n")

    for i, video_path in enumerate(video_paths, 1):
        print(f"[{i}/{len(video_paths)}] {video_path}")
        print("-" * 60)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a highly accurate video analysis AI. "
                    "You must describe only what is explicitly visible in the video. "
                    "Do not guess, assume, or hallucinate any details that are not present."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "nframes": args.nframes},
                    {"type": "text", "text": "What is happening in this video?"},
                ],
            }
        ]

        response = generate(model, processor, messages, args.max_new_tokens)
        print(f"Response: {response}\n")

        if args.save:
            stem = os.path.splitext(os.path.basename(video_path))[0]
            out_path = os.path.join(args.output_dir, f"{stem}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(response)
            print(f"Saved: {out_path}\n")

    print("All done.")


if __name__ == "__main__":
    main()
