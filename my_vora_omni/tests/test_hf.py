"""
VoRA Qwen3.5 — standalone inference test
Usage:
    python -m my_vora_omni.tests.test_hf <checkpoint_path> [--encoder vitl|vitg] [--max_tiles 4]
"""
import os
import re
import argparse

import torch
from my_vora_omni.src.model.model import (
    Qwen3_5VJEPALModel,
    Qwen3_5VJEPAGModel,
    Qwen3_5VJEPA21BModel,
    Qwen3_5VJEPA21LModel,
    Qwen3_5VJEPA21GModel,
    Gemma4VJEPALModel,
    Gemma4VJEPAGModel,
    Gemma4VJEPA21BModel,
    Gemma4VJEPA21LModel,
    Gemma4VJEPA21GModel,
)
from my_vora_omni.src.processor.processor import (
    Qwen3VLVJepa2LProcessor,
    Qwen3VLVJepa2GProcessor,
    Qwen3VLVJepa21BProcessor,
    Qwen3VLVJepa21LProcessor,
    Qwen3VLVJepa21GProcessor,
    Gemma4VJepa2LProcessor,
    Gemma4VJepa2GProcessor,
    Gemma4VJEPA21BProcessor,
    Gemma4VJEPA21LProcessor,
    Gemma4VJEPA21GProcessor,
)


def load_model(model_id: str, encoder: str = "vitl", device: str = "cuda"):
    if 'qwen' in model_id:
        MODEL_MAP = {
            "vitl": (Qwen3_5VJEPALModel, Qwen3VLVJepa2LProcessor),
            "vitg": (Qwen3_5VJEPAGModel, Qwen3VLVJepa2GProcessor),
            "vjepa21b": (Qwen3_5VJEPA21BModel, Qwen3VLVJepa21BProcessor),
            "vjepa21l": (Qwen3_5VJEPA21LModel, Qwen3VLVJepa21LProcessor),
            "vjepa21g": (Qwen3_5VJEPA21GModel, Qwen3VLVJepa21GProcessor),
        }
    elif 'gemma' in model_id:
        MODEL_MAP = {
            "vitl": (Gemma4VJEPALModel, Gemma4VJepa2LProcessor),
            "vitg": (Gemma4VJEPAGModel, Gemma4VJepa2GProcessor),
            "vjepa21b": (Gemma4VJEPA21BModel, Gemma4VJEPA21BProcessor),
            "vjepa21l": (Gemma4VJEPA21LModel, Gemma4VJEPA21LProcessor),
            "vjepa21g": (Gemma4VJEPA21GModel, Gemma4VJEPA21GProcessor),
        }

    if encoder not in MODEL_MAP:
        raise ValueError
    
    ModelCls, ProcCls = MODEL_MAP[encoder]
    processor = ProcCls.from_pretrained(model_id, trust_remote_code=True)
    model = ModelCls.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()
    return model, processor

def _post_process(text: str) -> str:
    # <think>...</think> 블록 제거
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # URL 제거 (http/https/ftp/www 형태 모두 포함)
    text = re.sub(r'https?://\S+|ftp://\S+|www\.\S+', '', text)
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

    from my_vora_omni.src.processor.processor import Gemma4VJEPAProcessor
    is_gemma = isinstance(processor, Gemma4VJEPAProcessor)

    template_kwargs = {} if is_gemma else {"enable_thinking": False}
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        **template_kwargs,
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.1,
    )

    # Strip input tokens
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return _post_process(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to merged checkpoint")
    parser.add_argument("--encoder", default="vitl", choices=["vitl", "vitg", "vjepa21b", "vjepa21l", "vjepa21g"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--nframes", type=int, default=32, help="Max video frames (must match FPS_MAX_FRAMES used at training)")
    parser.add_argument("--max_tiles", type=int, default=4, help="Max image tiles (sets IMAGE_MAX_TILES environment variable)")
    args = parser.parse_args()
    
    os.environ["FPS_MAX_FRAMES"] = str(args.nframes)
    os.environ["IMAGE_MAX_TILES"] = str(args.max_tiles)

    print(f"Loading model from {args.checkpoint} (encoder={args.encoder})...")
    model, processor = load_model(args.checkpoint, args.encoder, device=args.device)
    print("Model loaded.\n")

    print("=" * 60)
    print("[Test 3] Video")
    print("=" * 60)
    video_path = "my_vora_omni/examples/01_dog.mp4"
    if os.path.exists(video_path):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "nframes": args.nframes},
                    {"type": "text", "text": "Describe this video."}
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