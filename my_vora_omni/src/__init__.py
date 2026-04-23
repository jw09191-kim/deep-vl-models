from .model import (
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
from .processor import (
    Qwen3VLVJepa2LProcessor,
    Qwen3VLVJepa2GProcessor,
    Qwen3VLVJepa21BProcessor,
    Qwen3VLVJepa21LProcessor,
    Qwen3VLVJepa21GProcessor,
    Qwen3VJEPAImageProcessor,
    Qwen3VJEPAVideoProcessor,
    Gemma4VJepa2LProcessor,
    Gemma4VJepa2GProcessor,
    Gemma4VJEPA21BProcessor,
    Gemma4VJEPA21LProcessor,
    Gemma4VJEPA21GProcessor,
    Gemma4VJEPAImageProcessor,
    Gemma4VJEPAVideoProcessor,
)
from .template import Qwen3_5VJEPATemplate

# from . import register 

__all__ = [
    "Qwen3_5VJEPALModel",
    "Qwen3_5VJEPAGModel",
    "Qwen3_5VJEPA21BModel",
    "Qwen3_5VJEPA21LModel",
    "Qwen3_5VJEPA21GModel",
    "Qwen3VLVJepa2LProcessor",
    "Qwen3VLVJepa2GProcessor",
    "Qwen3VLVJepa21BProcessor",
    "Qwen3VLVJepa21LProcessor",
    "Qwen3VLVJepa21GProcessor",
    "Qwen3_5VJEPATemplate",
    "Qwen3VJEPAImageProcessor",
    "Qwen3VJEPAVideoProcessor",
    "Gemma4VJEPALModel",
    "Gemma4VJEPAGModel",
    "Gemma4VJEPA21BModel",
    "Gemma4VJEPA21LModel",
    "Gemma4VJEPA21GModel",
    "Gemma4VJepa2LProcessor",
    "Gemma4VJepa2GProcessor",
    "Gemma4VJEPA21BProcessor",
    "Gemma4VJEPA21LProcessor",
    "Gemma4VJEPA21GProcessor",
    "Gemma4VJEPAImageProcessor",
    "Gemma4VJEPAVideoProcessor",
]

from transformers import AutoImageProcessor, AutoVideoProcessor
AutoImageProcessor.register(
    "Qwen3VJEPAImageProcessor",
    fast_image_processor_class=Qwen3VJEPAImageProcessor,
    exist_ok=True,
)
AutoVideoProcessor.register(
    "Qwen3VJEPAVideoProcessor",
    video_processor_class=Qwen3VJEPAVideoProcessor,
    exist_ok=True,
)
AutoImageProcessor.register(
    "Gemma4VJEPAImageProcessor",
    fast_image_processor_class=Gemma4VJEPAImageProcessor,
    exist_ok=True,
)
AutoVideoProcessor.register(
    "Gemma4VJEPAVideoProcessor",
    video_processor_class=Gemma4VJEPAVideoProcessor,
    exist_ok=True,
)