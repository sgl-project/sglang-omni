# SPDX-License-Identifier: Apache-2.0
"""Backbone configurations for Tiny and dense Ming 3 TTS."""

from __future__ import annotations

from typing import ClassVar

import torch
from pydantic import JsonValue
from sglang.srt.configs.bailing_hybrid import BailingHybridConfig
from sglang.srt.runtime_context import get_exec, get_schedule
from torch import nn
from transformers import Qwen2Config

from sglang_omni.models.ming3_tts.acoustic import FLOW_STEPS
from sglang_omni.modules.ming_audio.flow.aggregator import Aggregator
from sglang_omni.modules.ming_audio.flow.cfm import get_epss_timesteps
from sglang_omni.modules.ming_audio.flow.flow_loss import FlowLoss


class Ming3TTSConfig(BailingHybridConfig):
    model_type: ClassVar[str] = "lingv3_tts"

    def __init__(
        self,
        llm_config: dict[str, JsonValue] | None = None,
        tts_config: dict[str, JsonValue] | None = None,
        flow_config: dict[str, JsonValue] | None = None,
        aggregator_config: dict[str, JsonValue] | None = None,
        **kwargs: JsonValue,
    ) -> None:
        backbone_fields = dict(llm_config or {})
        backbone_fields.update(kwargs)
        backbone_fields.pop("model_type", None)
        backbone_fields.pop("auto_map", None)
        super().__init__(**backbone_fields)
        self.tts_config: dict[str, JsonValue] = dict(tts_config or {})
        self.flow_config: dict[str, JsonValue] = dict(flow_config or {})
        self.aggregator_config: dict[str, JsonValue] = dict(aggregator_config or {})


class Ming3DenseTTSConfig(Qwen2Config):
    """Qwen2 backbone config for the dense Ming 3 TTS checkpoint."""

    model_type: ClassVar[str] = "dense"

    def __init__(
        self,
        llm_config: dict[str, JsonValue] | None = None,
        ditar_config: dict[str, JsonValue] | None = None,
        aggregator_config: dict[str, JsonValue] | None = None,
        audio_tokenizer_config: dict[str, JsonValue] | None = None,
        **kwargs: JsonValue,
    ) -> None:
        backbone_fields = dict(llm_config or {})
        for unused_name in (
            "model_type",
            "architectures",
            "auto_map",
            "torch_dtype",
            "transformers_version",
            "mlp_depth",
        ):
            kwargs.pop(unused_name, None)
            backbone_fields.pop(unused_name, None)
        super().__init__(**backbone_fields)
        # note (chenyang): AutoConfig matches the class attribute "dense",
        # while the Qwen2 runtime reads the instance model type.
        self.model_type = "qwen2"
        encoder_config = dict(audio_tokenizer_config or {}).get("enc_kwargs") or {}
        self.tts_config: dict[str, JsonValue] = {
            "latent_dim": int(encoder_config.get("latent_dim", 64)),
            "speaker_embedding_dim": 192,
        }
        self.flow_config: dict[str, JsonValue] = dict(ditar_config or {})
        self.aggregator_config: dict[str, JsonValue] = dict(aggregator_config or {})
        self.audio_tokenizer_config: dict[str, JsonValue] = dict(
            audio_tokenizer_config or {}
        )


def build_audio_aggregator(
    hidden_size: int, latent_dim: int, aggregator_config: dict[str, JsonValue]
) -> Aggregator:
    """Build the aggregator that projects one latent into one backbone embedding."""
    aggregator_kwargs = dict(aggregator_config)
    aggregator_kwargs.setdefault("in_channels", latent_dim)
    # The 0.5B checkpoint omits this key. Tiny checkpoints set it explicitly.
    aggregator_kwargs.setdefault("attn_backend", "torch")
    return Aggregator(llm_input_dim=hidden_size, **aggregator_kwargs)


def install_speech_heads(
    module: nn.Module, config: Ming3TTSConfig | Ming3DenseTTSConfig
) -> None:
    """Attach the flow, stop, and speaker heads shared by both backbones."""
    latent_dim = int(config.tts_config["latent_dim"])
    flow_config = dict(config.flow_config)
    flow_config.pop("z_channels", None)
    module.flowloss = FlowLoss(
        z_channels=latent_dim, llm_cond_dim=config.hidden_size, **flow_config
    )
    module.linear_proj_audio = build_audio_aggregator(
        config.hidden_size, latent_dim, config.aggregator_config
    )
    module.stop_head = nn.Linear(config.hidden_size, 2)
    module.spk_head = nn.Linear(
        int(config.tts_config["speaker_embedding_dim"]), config.hidden_size
    )
    module.flow_timesteps = get_epss_timesteps(
        FLOW_STEPS, module.stop_head.weight.device, torch.float32
    )
    module.acoustic_graphs = {}
    module.enable_fused_flow_head = False
    module.flow_head_path_logged = False
    module.weights_loaded = False
    module.flow_head_projections = None
    module.native_capture = None
    # note (chenyang): decode reads feedback rows from a fixed table by
    # row id, so the backbone CUDA graph replays them through input_ids.
    graph_config = get_exec().graph
    decode_batch_capacity = get_schedule().max_running_requests
    if not graph_config.disable_cuda_graph:
        decode_batch_capacity = max(
            decode_batch_capacity, graph_config.cuda_graph_config.decode.max_bs
        )
    else:
        pass
    module.decode_feedback = nn.Embedding(decode_batch_capacity, config.hidden_size)
    module.decode_feedback.weight.requires_grad_(False)
    module.decode_feedback_rows = torch.arange(
        decode_batch_capacity, device=module.decode_feedback.weight.device
    )
    # note (rsi): prefill reads its input rows from a fixed table by row id
    # too, so a prefill CUDA graph that embeds input_ids replays the rows the
    # runner staged; token embeddings are taken from the weight directly.
    schedule = get_schedule()
    prefill_capacity = (
        max(schedule.max_prefill_tokens, schedule.chunked_prefill_size)
        + decode_batch_capacity
    )
    module.prefill_inputs = nn.Embedding(prefill_capacity, config.hidden_size)
    module.prefill_inputs.weight.requires_grad_(False)
    module.prefill_input_rows = torch.arange(
        prefill_capacity, device=module.prefill_inputs.weight.device
    )
    module.get_input_embeddings().forward = module.prefill_inputs.forward
