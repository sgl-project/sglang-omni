# SPDX-License-Identifier: Apache-2.0
"""The Drama 3 pipeline served over HTTP from a synthetic checkpoint; its engine never
samples end-of-speech, so every request generates exactly max_new_tokens frames."""

from __future__ import annotations

import argparse
from collections.abc import Mapping

import torch
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.registry import ModelRegistry
from torch import nn

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.schema import StageConfig
from sglang_omni.models.fishaudio_drama3.config import (
    DEFAULT_ASYNC_DECODE_MIN_BATCH_SIZE,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_ENABLE_ASYNC_DECODE,
    DEFAULT_FIRST_FRAME_CUDA_GRAPH_MAX_REQUESTS,
    DEFAULT_FIRST_FRAME_CUDA_GRAPHS,
    DEFAULT_MAX_NEW_TOKENS_CAP,
    DEFAULT_MAX_PROMPT_ROWS,
    DEFAULT_MAX_RUNNING_REQUESTS,
    DEFAULT_STREAM_FOLLOWUP_STRIDE,
    DEFAULT_STREAM_FOLLOWUP_STRIDE_RAMP,
    DEFAULT_STREAM_STRIDE,
    DEFAULT_TOWER_FUSED_LAYERS,
    DEFAULT_TRITON_SAMPLING,
    Drama3PipelineConfig,
)
from sglang_omni.models.fishaudio_drama3.engine_builder import (
    FISH_OMNI_ARCHITECTURE,
    Drama3EngineBuilder,
)
from sglang_omni.models.fishaudio_drama3.hf_config import FishOmniConfig
from sglang_omni.models.fishaudio_drama3.request_builders import Drama3RequestData
from sglang_omni.models.fishaudio_drama3.sglang_model import Drama3ForCausalLM
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.serve.launcher import launch_server

MODULE = "tests.unit_test.fishaudio_drama3.http_server"
MODEL_NAME = "drama3"
# The synthetic model needs small KV pools, and the vocoder shares the GPU.
ENGINE_MEMORY_FRACTION = "0.3"


class EosMaskedLogits(nn.Module):
    """Wraps the logits processor so end-of-speech never wins a sample.

    The lm_head scores the main-token candidates only, so end-of-speech is a candidate column, not its token id.
    """

    def __init__(self, logits_processor: LogitsProcessor, *, eos_column: int) -> None:
        super().__init__()
        self.logits_processor = logits_processor
        self.eos_column = eos_column

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        lm_head: ParallelLMHead,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        output = self.logits_processor(input_ids, hidden_states, lm_head, forward_batch)
        output.next_token_logits[:, self.eos_column] = float("-inf")
        return output


class EosMaskedDrama3Model(Drama3ForCausalLM):
    def __init__(
        self,
        config: FishOmniConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        candidate_token_ids = self.audio_decoder.main_token_candidate_ids.tolist()
        self.logits_processor = EosMaskedLogits(
            self.logits_processor,
            eos_column=candidate_token_ids.index(config.eos_token_id),
        )


class EosMaskedEngineBuilder(Drama3EngineBuilder):
    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        super().pre_infra_setup(checkpoint_dir)
        ModelRegistry.models[FISH_OMNI_ARCHITECTURE] = EosMaskedDrama3Model


def create_eos_masked_tts_engine_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    max_running_requests: int = DEFAULT_MAX_RUNNING_REQUESTS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_CAP,
    max_prompt_rows: int = DEFAULT_MAX_PROMPT_ROWS,
    server_args_overrides: Mapping[str, object] | None = None,
    stream_stride: int = DEFAULT_STREAM_STRIDE,
    stream_followup_stride_ramp: tuple[int, ...] = DEFAULT_STREAM_FOLLOWUP_STRIDE_RAMP,
    stream_followup_stride: int = DEFAULT_STREAM_FOLLOWUP_STRIDE,
    first_frame_cuda_graphs: bool = DEFAULT_FIRST_FRAME_CUDA_GRAPHS,
    first_frame_cuda_graph_max_requests: int = (
        DEFAULT_FIRST_FRAME_CUDA_GRAPH_MAX_REQUESTS
    ),
    triton_sampling: bool = DEFAULT_TRITON_SAMPLING,
    tower_fused_layers: bool = DEFAULT_TOWER_FUSED_LAYERS,
    enable_async_decode: bool = DEFAULT_ENABLE_ASYNC_DECODE,
    async_decode_min_batch_size: int = DEFAULT_ASYNC_DECODE_MIN_BATCH_SIZE,
) -> OmniScheduler[Drama3RequestData]:
    return EosMaskedEngineBuilder(
        context_length=context_length,
        max_running_requests=max_running_requests,
        stream_stride=stream_stride,
        stream_followup_stride_ramp=stream_followup_stride_ramp,
        stream_followup_stride=stream_followup_stride,
        max_prompt_rows=max_prompt_rows,
        max_new_tokens_cap=max_new_tokens,
        first_frame_cuda_graphs=first_frame_cuda_graphs,
        first_frame_cuda_graph_max_requests=first_frame_cuda_graph_max_requests,
        triton_sampling=triton_sampling,
        tower_fused_layers=tower_fused_layers,
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=async_decode_min_batch_size,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        server_args_overrides=server_args_overrides,
    )


class EosMaskedPipelineConfig(Drama3PipelineConfig):
    """The Drama 3 pipeline with its engine built by the factory above."""

    stages: list[StageConfig] = [
        (
            stage.model_copy(
                update={
                    "factory_path": f"{MODULE}.create_eos_masked_tts_engine_executor"
                }
            )
            if stage.name == "tts_engine"
            else stage
        )
        for stage in Drama3PipelineConfig.model_fields["stages"].default
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, required=True)
    arguments = parser.parse_args()
    config = ConfigManager(
        EosMaskedPipelineConfig(model_path=arguments.checkpoint)
    ).merge_config([("tts_engine.engine.mem_fraction_static", ENGINE_MEMORY_FRACTION)])
    launch_server(config, host="127.0.0.1", port=arguments.port, model_name=MODEL_NAME)
