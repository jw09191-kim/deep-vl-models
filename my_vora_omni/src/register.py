from swift.model import (
    Model,
    ModelGroup,
    ModelLoader,
    ModelMeta,
    MultiModelKeys,
    register_model,
    register_model_arch
)
from swift.model.models.qwen import Qwen3_5Loader
from swift.model.models.gemma import Gemma4Loader
from swift.template import register_template
from swift.template.templates.qwen import QwenTemplateMeta
from swift.template.templates.gemma import Gemma4TemplateMeta

from transformers import AutoImageProcessor, AutoVideoProcessor
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
    VJEPAImageProcessor,
    VJEPAVideoProcessor
)

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
from my_vora_omni.src.template.template import Qwen3_5VJEPATemplate, Gemma4VJEPATemplate


AutoImageProcessor.register(
    "VJEPAImageProcessor",
    fast_image_processor_class=VJEPAImageProcessor,
    exist_ok=True,
)
AutoVideoProcessor.register(
    "VJEPAVideoProcessor",
    video_processor_class=VJEPAVideoProcessor,
    exist_ok=True,
)


class Qwen35VJEPALoaderBase(Qwen3_5Loader):
    PROCESSOR_CLS = None
    MODEL_CLS     = None

    def get_processor(self, model_dir, config):
        return self.PROCESSOR_CLS.from_pretrained(model_dir)

    def get_model(self, model_dir, config, processor, model_kwargs):
        self.auto_model_cls = self.MODEL_CLS
        model = super().get_model(model_dir, config, processor, model_kwargs)
        return model


class Qwen35VJEPALLoader(Qwen35VJEPALoaderBase):
    PROCESSOR_CLS = Qwen3VLVJepa2LProcessor
    MODEL_CLS     = Qwen3_5VJEPALModel

class Qwen35VJEPAGLoader(Qwen35VJEPALoaderBase):
    PROCESSOR_CLS = Qwen3VLVJepa2GProcessor
    MODEL_CLS     = Qwen3_5VJEPAGModel

class Qwen35VJEPA21BLoader(Qwen35VJEPALoaderBase):
    PROCESSOR_CLS = Qwen3VLVJepa21BProcessor
    MODEL_CLS     = Qwen3_5VJEPA21BModel

class Qwen35VJEPA21LLoader(Qwen35VJEPALoaderBase):
    PROCESSOR_CLS = Qwen3VLVJepa21LProcessor
    MODEL_CLS     = Qwen3_5VJEPA21LModel

class Qwen35VJEPA21GLoader(Qwen35VJEPALoaderBase):
    PROCESSOR_CLS = Qwen3VLVJepa21GProcessor
    MODEL_CLS     = Qwen3_5VJEPA21GModel


register_model_arch(
    MultiModelKeys(
        'qwen35_vjepa',
        language_model=['model.language_model', 'lm_head'],
        vision_tower=['model.visual.encoder'],
        aligner=['model.visual.merger'],
    )
)

register_template(
    QwenTemplateMeta(
        'vora_qwen35',
        template_cls=Qwen3_5VJEPATemplate,
        default_system='You are a helpful assistant.',
        thinking_prefix='<think>\n',
        non_thinking_prefix='<think>\n\n</think>\n\n',
        agent_template='qwen3_5',
        is_thinking=False,
    )
)

_QWEN35_MODEL_GROUP = [ModelGroup([
    Model('Qwen/Qwen3.5-0.8B', 'Qwen/Qwen3.5-0.8B'),
    Model('Qwen/Qwen3.5-2B',   'Qwen/Qwen3.5-2B'),
])]

_COMMON = dict(
    template='vora_qwen35',
    is_multimodal=True,
    model_arch='qwen35_vjepa',
    requires=['transformers>=5.0.0'],
    tags=['vision', 'video'],
)

register_model(ModelMeta('vora-qwen35-vitl',     _QWEN35_MODEL_GROUP, Qwen35VJEPALLoader,    architectures=['Qwen3_5VJEPALModel'],    **_COMMON))
register_model(ModelMeta('vora-qwen35-vitg',     _QWEN35_MODEL_GROUP, Qwen35VJEPAGLoader,    architectures=['Qwen3_5VJEPAGModel'],    **_COMMON))
register_model(ModelMeta('vora-qwen35-vjepa21b', _QWEN35_MODEL_GROUP, Qwen35VJEPA21BLoader,  architectures=['Qwen3_5VJEPA21BModel'],  **_COMMON))
register_model(ModelMeta('vora-qwen35-vjepa21l', _QWEN35_MODEL_GROUP, Qwen35VJEPA21LLoader,  architectures=['Qwen3_5VJEPA21LModel'],  **_COMMON))
register_model(ModelMeta('vora-qwen35-vjepa21g', _QWEN35_MODEL_GROUP, Qwen35VJEPA21GLoader,  architectures=['Qwen3_5VJEPA21GModel'],  **_COMMON))


# ──────────────────────────────────────────────
# Gemma-4 Registration
# ──────────────────────────────────────────────

class Gemma4VJEPALoaderBase(Gemma4Loader):
    PROCESSOR_CLS = None
    MODEL_CLS     = None

    def get_processor(self, model_dir, config):
        return self.PROCESSOR_CLS.from_pretrained(model_dir)

    def get_model(self, model_dir, config, processor, model_kwargs):
        self.auto_model_cls = self.MODEL_CLS
        return super().get_model(model_dir, config, processor, model_kwargs)


class Gemma4VJEPALLoader(Gemma4VJEPALoaderBase):
    PROCESSOR_CLS = Gemma4VJepa2LProcessor
    MODEL_CLS     = Gemma4VJEPALModel

class Gemma4VJEPAGLoader(Gemma4VJEPALoaderBase):
    PROCESSOR_CLS = Gemma4VJepa2GProcessor
    MODEL_CLS     = Gemma4VJEPAGModel

class Gemma4VJEPA21BLoader(Gemma4VJEPALoaderBase):
    PROCESSOR_CLS = Gemma4VJEPA21BProcessor
    MODEL_CLS     = Gemma4VJEPA21BModel

class Gemma4VJEPA21LLoader(Gemma4VJEPALoaderBase):
    PROCESSOR_CLS = Gemma4VJEPA21LProcessor
    MODEL_CLS     = Gemma4VJEPA21LModel

class Gemma4VJEPA21GLoader(Gemma4VJEPALoaderBase):
    PROCESSOR_CLS = Gemma4VJEPA21GProcessor
    MODEL_CLS     = Gemma4VJEPA21GModel


register_model_arch(
    MultiModelKeys(
        'gemma4_vjepa',
        language_model=['model.language_model', 'lm_head'],
        vision_tower=['visual.encoder'],   # outer class에 직접 부착 — "model." prefix 없음
        aligner=['visual.merger'],
    )
)

register_template(
    Gemma4TemplateMeta(
        'vora_gemma4',
        template_cls=Gemma4VJEPATemplate,
        default_system='You are a helpful assistant.',
    )
)

_GEMMA4_MODEL_GROUP = [ModelGroup([
    Model('google/gemma-4-E2B',    'google/gemma-4-E2B'),
    Model('google/gemma-4-E2B-it', 'google/gemma-4-E2B-it'),
    Model('google/gemma-4-E4B',    'google/gemma-4-E4B'),
    Model('google/gemma-4-E4B-it', 'google/gemma-4-E4B-it'),
])]

_COMMON_GEMMA4 = dict(
    template='vora_gemma4',
    is_multimodal=True,
    model_arch='gemma4_vjepa',
    requires=['transformers>=5.0.0'],
    tags=['vision', 'video'],
)

register_model(ModelMeta('vora-gemma4-vitl',     _GEMMA4_MODEL_GROUP, Gemma4VJEPALLoader,    architectures=['Gemma4VJEPALModel'],    **_COMMON_GEMMA4))
register_model(ModelMeta('vora-gemma4-vitg',     _GEMMA4_MODEL_GROUP, Gemma4VJEPAGLoader,    architectures=['Gemma4VJEPAGModel'],    **_COMMON_GEMMA4))
register_model(ModelMeta('vora-gemma4-vjepa21b', _GEMMA4_MODEL_GROUP, Gemma4VJEPA21BLoader,  architectures=['Gemma4VJEPA21BModel'],  **_COMMON_GEMMA4))
register_model(ModelMeta('vora-gemma4-vjepa21l', _GEMMA4_MODEL_GROUP, Gemma4VJEPA21LLoader,  architectures=['Gemma4VJEPA21LModel'],  **_COMMON_GEMMA4))
register_model(ModelMeta('vora-gemma4-vjepa21g', _GEMMA4_MODEL_GROUP, Gemma4VJEPA21GLoader,  architectures=['Gemma4VJEPA21GModel'],  **_COMMON_GEMMA4))