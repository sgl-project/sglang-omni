"""Register SenseNova with the native diffusion runtime."""

from sglang.multimodal_gen.registry import register_pipeline

from sglang_omni.models.sensenova_u1.native.options import is_sensenova_u1_model
from sglang_omni.models.sensenova_u1.native.pipeline import SenseNovaU1Pipeline
from sglang_omni.models.sensenova_u1.native.pipeline_config import (
    SenseNovaU1PipelineConfig,
)
from sglang_omni.models.sensenova_u1.native.sampling import SenseNovaU1SamplingParams
from sglang_omni.vendor.sensenova_u1 import register

register_pipeline(
    SenseNovaU1Pipeline,
    sampling_param_cls=SenseNovaU1SamplingParams,
    pipeline_config_cls=SenseNovaU1PipelineConfig,
    hf_model_paths=["sensenova/SenseNova-U1.5-8B-MoT"],
    model_detectors=[is_sensenova_u1_model],
    overwrite=True,
)
register()
