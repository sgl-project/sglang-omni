# SPDX-License-Identifier: Apache-2.0
"""Experimental native worker pipeline for in-memory semantic-token decoding."""

import io

import torch
from sglang.multimodal_gen.configs.pipeline_configs.zimage import ZImagePipelineConfig
from sglang.multimodal_gen.configs.sample.zimage import ZImageSamplingParams
from sglang.multimodal_gen.registry import register_pipeline
from sglang.multimodal_gen.runtime.distributed import parallel_state as ps
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.server_args import server_args as server_args_module
from sglang.multimodal_gen.runtime.utils import precision
from sglang.srt import runtime_context

from sglang_omni.models.llada2_uni.components.decoder_model import (
    require_joint_sp_support,
)
from sglang_omni.models.llada2_uni.components.decoder_runtime import (
    DecoderRuntimeHandle,
)
from sglang_omni.models.llada2_uni.components.image_decoder import (
    LLaDA2ImageDecoder,
    validate_checkpoint_sp_config,
)

DTYPE_BY_PRECISION = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


class SemanticDecode(PipelineStage):
    def __init__(self, model_path: str, server_args: ServerArgs) -> None:
        super().__init__()
        device = torch.device("cuda", torch.cuda.current_device())
        pipeline_config = getattr(server_args, "pipeline_config", None)
        precision_name = getattr(pipeline_config, "dit_precision", None) or "bf16"
        if precision_name not in DTYPE_BY_PRECISION:
            raise ValueError(f"Unsupported decoder precision: {precision_name!r}")
        else:
            pass
        dtype = DTYPE_BY_PRECISION[precision_name]
        attention_backend = server_args.attention_backend or "fa"
        self.runtime = DecoderRuntimeHandle(
            device,
            dtype,
            attention_backend,
            ps,
            server_args_module,
            runtime_context,
            precision,
        )
        self.runtime.sp_size = ps.get_sp_world_size()
        self.runtime.sp_rank = ps.get_sp_parallel_rank()
        self.runtime.ulysses_degree = ps.get_ulysses_parallel_world_size()
        if self.runtime.sp_size > 1:
            require_joint_sp_support()
        else:
            pass
        self.decoder = LLaDA2ImageDecoder(
            model_path,
            device=str(device),
            dtype=dtype,
            backend="sglang",
            runtime=self.runtime,
            decode_mode="decoder-turbo",
            num_steps=8,
        )
        for mode in ("decoder-turbo", "normal"):
            validate_checkpoint_sp_config(
                self.decoder.model_path, mode, self.runtime.ulysses_degree
            )

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        llada_vq = batch.extra["llada_vq"]
        if len(llada_vq) == 6:
            tokens, height, width, seed, mode, steps = llada_vq
            multiplier = None
        else:
            tokens, height, width, seed, mode, steps, multiplier = llada_vq
        with self.runtime.compute_context():
            image = self.decoder.decode(
                tokens,
                height,
                width,
                seed=seed,
                decode_mode=mode,
                num_steps=steps,
                resolution_multiplier=multiplier,
            )
        if image is None:
            return OutputBatch()
        else:
            pass
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        return OutputBatch(output_image_bytes=[stream.getvalue()])


class SemanticDecoderPipeline(ComposedPipelineBase):
    pipeline_name = "LLaDASemanticDecoder"
    _required_config_modules: list[str] = (
        []
    )  # noqa: leading-underscore  # SGLang pipeline contract

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, torch.nn.Module]:
        return {}

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(SemanticDecode(self.model_path, server_args), "semantic_decode")


register_pipeline(
    SemanticDecoderPipeline,
    sampling_param_cls=ZImageSamplingParams,
    pipeline_config_cls=ZImagePipelineConfig,
    model_detectors=[lambda path: "llada2.0-uni" in str(path).lower()],
    overwrite=True,
)
