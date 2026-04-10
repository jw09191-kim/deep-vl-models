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
from swift.template import register_template, TemplateMeta
from swift.template.templates.qwen import QwenTemplateMeta, MLLMTemplateType

from transformers import AutoImageProcessor, AutoVideoProcessor
from my_vora_omni.src.processor.processor import (
    Qwen3VLVJepa2LProcessor,
    Qwen3VLVJepa2GProcessor,
    Qwen3VLVJepa21BProcessor,
    Qwen3VLVJepa21LProcessor,
    Qwen3VLVJepa21GProcessor,
    VJEPAImageProcessor,
    VJEPAVideoProcessor
)

from my_vora_omni.src.model.model import (
    Qwen3_5VJEPALModel,
    Qwen3_5VJEPAGModel,
    Qwen3_5VJEPA21BModel,
    Qwen3_5VJEPA21LModel,
    Qwen3_5VJEPA21GModel,
)
from my_vora_omni.src.template.template import Qwen3_5VJEPATemplate


AutoImageProcessor.register(
    "VJEPAImageProcessor",
    fast_image_processor_class=VJEPAImageProcessor,
    exist_ok=True,
)
AutoVideoProcessor.register(
    "VJEPAVideoProcessor",
    VJEPAVideoProcessor,
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
        default_system=None,
        thinking_prefix='<think>\n',
        non_thinking_prefix='<think>\n\n</think>\n\n',
        agent_template='qwen3_5',
        is_thinking=False,
    )
)

_COMMON = dict(
    template='vora_qwen35',
    is_multimodal=True,
    model_arch='qwen35_vjepa',
    requires=['transformers>=5.0.0'],
    tags=['vision', 'video'],
)

register_model(
    ModelMeta(
        'vora-qwen35-vitl',
        [ModelGroup([
            Model('Qwen/Qwen3.5-0.8B', 'Qwen/Qwen3.5-0.8B'),
            Model('Qwen/Qwen3.5-2B',   'Qwen/Qwen3.5-2B'),
        ])],
        Qwen35VJEPALLoader,
        architectures=['Qwen3_5VJEPALModel'],
        **_COMMON,
    )
)

register_model(
    ModelMeta(
        'vora-qwen35-vitg',
        [ModelGroup([
            Model('Qwen/Qwen3.5-0.8B', 'Qwen/Qwen3.5-0.8B'),
            Model('Qwen/Qwen3.5-2B',   'Qwen/Qwen3.5-2B'),
        ])],
        Qwen35VJEPAGLoader,
        architectures=['Qwen3_5VJEPAGModel'],
        **_COMMON,
    )
)

register_model(
    ModelMeta(
        'vora-qwen35-vjepa21b',
        [ModelGroup([
            Model('Qwen/Qwen3.5-0.8B', 'Qwen/Qwen3.5-0.8B'),
            Model('Qwen/Qwen3.5-2B',   'Qwen/Qwen3.5-2B'),
        ])],
        Qwen35VJEPA21BLoader,
        architectures=['Qwen3_5VJEPA21BModel'],
        **_COMMON,
    )
)

register_model(
    ModelMeta(
        'vora-qwen35-vjepa21l',
        [ModelGroup([
            Model('Qwen/Qwen3.5-0.8B', 'Qwen/Qwen3.5-0.8B'),
            Model('Qwen/Qwen3.5-2B',   'Qwen/Qwen3.5-2B'),
        ])],
        Qwen35VJEPA21LLoader,
        architectures=['Qwen3_5VJEPA21LModel'],
        **_COMMON,
    )
)

register_model(
    ModelMeta(
        'vora-qwen35-vjepa21g',
        [ModelGroup([
            Model('Qwen/Qwen3.5-0.8B', 'Qwen/Qwen3.5-0.8B'),
            Model('Qwen/Qwen3.5-2B',   'Qwen/Qwen3.5-2B'),
        ])],
        Qwen35VJEPA21GLoader,
        architectures=['Qwen3_5VJEPA21GModel'],
        **_COMMON,
    )
)