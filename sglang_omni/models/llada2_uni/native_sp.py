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

from sglang_omni.models.llada2_uni.components.decoder_runtime import (
    DecoderRuntimeHandle,
)
from sglang_omni.models.llada2_uni.components.image_decoder import LLaDA2ImageDecoder


class SemanticDecode(PipelineStage):
    def __init__(self, model_path: str) -> None:
        super().__init__()
        device = torch.device("cuda", torch.cuda.current_device())
        self.runtime = DecoderRuntimeHandle(
            device,
            torch.bfloat16,
            "fa",
            ps,
            server_args_module,
            runtime_context,
            precision,
        )
        self.runtime.sp_size = ps.get_sp_world_size()
        self.runtime.sp_rank = ps.get_sp_parallel_rank()
        self.runtime.ulysses_degree = ps.get_ulysses_parallel_world_size()
        self.decoder = LLaDA2ImageDecoder(
            model_path,
            device=str(device),
            backend="sglang",
            runtime=self.runtime,
            decode_mode="decoder-turbo",
            num_steps=8,
        )

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        tokens, height, width, seed, mode, steps = batch.extra["llada_vq"]
        with self.runtime.compute_context():
            image = self.decoder.decode(
                tokens, height, width, seed=seed, decode_mode=mode, num_steps=steps
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
        self.add_stage(SemanticDecode(self.model_path), "semantic_decode")


register_pipeline(
    SemanticDecoderPipeline,
    sampling_param_cls=ZImageSamplingParams,
    pipeline_config_cls=ZImagePipelineConfig,
    model_detectors=[lambda path: "llada2.0-uni" in str(path).lower()],
    overwrite=True,
)
