# SPDX-License-Identifier: Apache-2.0
"""SGLang-native Qwen3-TTS talker wrapper."""

from __future__ import annotations

import gc
import logging
import math
import os
import time
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterable, Literal, Optional, Tuple, TypeAlias

import torch
from sglang.kernels.fused_op import get_fused_op_backend
from sglang.kernels.spec import KernelBackend
from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.unquant import (
    UnquantizedLinearMethod,
    get_bf16_gemm_backend,
)
from sglang.srt.layers.sampler import multinomial_with_seed
from sglang.srt.runtime_context import get_context, get_exec, get_parallel, get_schedule
from sglang.srt.utils import add_prefix
from sglang.srt.utils.common import is_pin_memory_available
from torch import nn

from sglang_omni.models.qwen3_omni.components.talker import (  # noqa: E501
    Qwen3OmniMoeTalkerDenseMLP,
    ResizeMLP,
    bind_default_weight_loaders,
)
from sglang_omni.models.qwen3_omni.components.thinker_model import (
    Qwen3OmniMoeThinkerTextAttention,
)
from sglang_omni.models.qwen3_tts.compat import (
    apply_qwen_tts_transformers_compatibility_patches,
)
from sglang_omni.models.qwen3_tts.predictor_kernels import (
    gather_codec_embedding_and_add,
)
from sglang_omni.models.qwen3_tts.sampling_kernels import (
    sample_from_logits_with_seed_top_k_top_p,
    sample_from_logprobs_with_seed_npu,
    sample_from_sorted_logprobs_with_seed_small_k,
)
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import ReplayableGraph
from sglang_omni.scheduling.types import SchedulerRequest
from sglang_omni.vendor.sglang.core import ForwardBatch
from sglang_omni.vendor.sglang.layers import ReplicatedLinear, RMSNorm
from sglang_omni.vendor.sglang.models import FusedSetKVBufferArg, apply_qk_norm

if TYPE_CHECKING:
    from qwen_tts import Qwen3TTSTokenizer
    from qwen_tts.core.models.configuration_qwen3_tts import (
        Qwen3TTSConfig,
        Qwen3TTSTalkerCodePredictorConfig,
        Qwen3TTSTalkerConfig,
    )

    from sglang_omni.models.qwen3_tts.request_builders import VoicePrompt
else:
    pass

logger = logging.getLogger(__name__)

PredictorGraphSignature: TypeAlias = tuple[
    Literal["argmax", "sampled"], int, bool, bool, bool
]
PredictorGraphKey: TypeAlias = tuple[
    int, Literal["argmax", "sampled"], int, bool, bool, bool
]

QTTS_PREDICTOR_GRAPH_ENV = "SGLANG_OMNI_QTTS_PREDICTOR_GRAPH"
_PREDICTOR_GRAPH_MAX_LAZY_KEYS = 32
_PREDICTOR_GRAPH_MAX_FAILURES = 8
_PREDICTOR_GRAPH_WARMUP_PASSES = 2
# Note: (Jiaxin Deng) 50 is on the ladder because it is the family checkpoint
# default, keeping the dominant signature's kernel width exactly as before.
_PREDICTOR_TOP_K_LADDER = (4, 8, 16, 32, 50, 64, 128, 256, 512, 1024)


def predictor_graph_env_override() -> bool | None:
    """The operator's explicit choice, or None to leave it to the platform."""
    value = os.environ.get(QTTS_PREDICTOR_GRAPH_ENV)
    if value is None:
        return None
    else:
        pass
    return value.strip().lower() not in ("0", "false", "off", "no")


def predictor_gqa_attention(
    q: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_heads: int,
    num_key_value_heads: int,
    is_causal: bool,
) -> torch.Tensor:
    """Run Predictor GQA, preferring Ascend's inference kernel on NPU for one query."""
    if q.device.type == "npu" and not is_causal:
        fused_attention = getattr(
            getattr(torch.ops, "npu", None),
            "npu_fused_infer_attention_score",
            None,
        )
        if fused_attention is not None:
            output, _ = fused_attention(
                q.transpose(1, 2).contiguous(),
                key.transpose(1, 2).contiguous(),
                value.transpose(1, 2).contiguous(),
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                input_layout="BSND",
                scale=1.0 / math.sqrt(q.shape[-1]),
            )
            return output.transpose(1, 2)
        else:
            pass
    else:
        pass

    return torch.nn.functional.scaled_dot_product_attention(
        q,
        key,
        value,
        is_causal=is_causal,
        enable_gqa=True,
    )


def quantize_predictor_top_k(max_top_k: int, vocab_size: int) -> int | None:
    """Smallest ladder width covering max_top_k, or None to use the full sort."""
    for step in _PREDICTOR_TOP_K_LADDER:
        if step >= max_top_k:
            return step if step < vocab_size else None
        else:
            pass
    return None


def predictor_signature_terms(
    sampled_top_ks: list[int],
    sampled_top_ps: list[float],
    vocab_size: int,
) -> tuple[int, bool, bool]:
    """The one signature rule, shared by the batch path and the startup capture."""
    bounded_top_ks = [
        int(top_k) for top_k in sampled_top_ks if 0 < int(top_k) < vocab_size
    ]
    has_top_p = any(0.0 < float(top_p) < 1.0 for top_p in sampled_top_ps)
    has_unbounded_top_k = len(bounded_top_ks) != len(sampled_top_ks)
    max_top_k = 0
    max_bounded_top_k = max(bounded_top_ks, default=0)
    if max_bounded_top_k > 0 and not has_unbounded_top_k:
        # Note: (Jiaxin Deng) ladder-quantized so predictor-graph keys are
        # shared across request top_k values; per-row masks keep true k.
        quantized = quantize_predictor_top_k(max_bounded_top_k, vocab_size)
        if quantized is None:
            has_unbounded_top_k = True
        else:
            max_top_k = quantized
    else:
        pass
    return max_top_k, has_top_p, has_unbounded_top_k


def sample_seeded_categorical(
    logprobs: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    if logprobs.device.type == "npu":
        return sample_from_logprobs_with_seed_npu(logprobs, seeds, positions)
    else:
        pass
    return multinomial_with_seed(logprobs, seeds, positions).view(-1)


class PredictorDecodeGraph:
    """Device graph over the full per-token predictor chain for one batch bucket.

    One graph per (bucket, sampling signature): the signature pins the host
    branches of the sampling path (argmax vs sampled, top-k bound, top-p
    presence), so replay reproduces the bits of the eager pass.
    Per-step inputs reach the captured region through persistent device
    buffers written with device-side copies before replay. Holds no reference
    to the talker: a cycle would put the graph's finalizer behind the
    cyclic collector.
    """

    def __init__(
        self,
        batch_size: int,
        signature: PredictorGraphSignature,
        *,
        device: torch.device,
        hidden_size: int,
        hidden_dtype: torch.dtype,
    ) -> None:
        self.batch_size = batch_size
        self.signature = signature
        self.device = device
        self.device_module = torch.get_device_module(device)
        self.layer0_codes = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        self.talker_hidden = torch.zeros(
            batch_size, 1, hidden_size, dtype=hidden_dtype, device=device
        )
        self.semantic_positions = torch.zeros(
            batch_size, dtype=torch.long, device=device
        )
        self.graph: ReplayableGraph | None = None
        self.result_codes: torch.Tensor | None = None
        self.summed_embeddings: torch.Tensor | None = None

    @torch.no_grad()
    def replay(
        self,
        layer0_codes: torch.Tensor,
        talker_hidden: torch.Tensor,
        semantic_positions: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        live = layer0_codes.shape[0]
        if live > self.batch_size:
            raise ValueError(
                "Qwen3-TTS predictor graph bucket is too small: "
                f"bucket={self.batch_size}, live={live}"
            )
        else:
            pass
        with self.device_module.device(self.device):
            self.layer0_codes[:live].copy_(layer0_codes)
            self.talker_hidden[:live].copy_(talker_hidden)
            if semantic_positions is None:
                self.semantic_positions.zero_()
            else:
                self.semantic_positions[:live].copy_(semantic_positions.reshape(live))
            if live < self.batch_size:
                self.layer0_codes[live:].zero_()
                self.talker_hidden[live:].zero_()
                if semantic_positions is not None:
                    self.semantic_positions[live:].zero_()
                else:
                    pass
            else:
                pass
            self.graph.replay()
        assert self.result_codes is not None
        assert self.summed_embeddings is not None
        return self.result_codes[:live], self.summed_embeddings[:live]


class Qwen3TTSTalkerDecoderLayer(nn.Module):
    def __init__(
        self,
        config: "Qwen3TTSTalkerConfig | Qwen3TTSTalkerCodePredictorConfig",
        layer_id: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3OmniMoeThinkerTextAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            config=config,
            prefix=add_prefix("self_attn", prefix),
            dual_chunk_attention_config=None,
            alt_stream=None,
        )
        self.mlp = Qwen3OmniMoeTalkerDenseMLP(
            config.hidden_size,
            config.intermediate_size,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3TTSTalkerTextModel(nn.Module):
    def __init__(self, config: "Qwen3TTSTalkerConfig", prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.text_embedding = nn.Embedding(
            config.text_vocab_size, config.text_hidden_size
        )
        self.layers = nn.ModuleList(
            [
                Qwen3TTSTalkerDecoderLayer(
                    config,
                    idx,
                    prefix=add_prefix(f"layers.{idx}", prefix),
                )
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.start_layer = 0
        self.end_layer = config.num_hidden_layers
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        max_batch_size = get_schedule().max_running_requests
        self.feedback_buffer = torch.zeros(
            max_batch_size,
            config.hidden_size,
            device=self.codec_embedding.weight.device,
            dtype=self.codec_embedding.weight.dtype,
        )
        self.feedback_mask = torch.zeros(
            max_batch_size,
            dtype=torch.bool,
            device=self.codec_embedding.weight.device,
        )
        self.decode_feedback_embedding = nn.Embedding(
            max_batch_size,
            config.hidden_size,
            device=self.codec_embedding.weight.device,
            dtype=self.codec_embedding.weight.dtype,
        )
        self.decode_feedback_embedding.weight.requires_grad_(False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.codec_embedding

    def get_text_embeddings(self) -> nn.Embedding:
        return self.text_embedding

    def build_input_hidden_states(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.codec_embedding(input_ids)
        bs = hidden_states.shape[0]
        feedback_mask = self.feedback_mask[:bs]
        return torch.where(
            feedback_mask.unsqueeze(-1),
            self.feedback_buffer[:bs].to(hidden_states.dtype),
            hidden_states,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        forward_mode = getattr(forward_batch, "forward_mode", None)
        is_decode = forward_mode is not None and forward_mode.is_decode()
        if input_embeds is None:
            if is_decode:
                hidden_states = self.decode_feedback_embedding(input_ids)
            else:
                hidden_states = self.build_input_hidden_states(input_ids)
        else:
            hidden_states = input_embeds

        residual = None
        layers = self.layers
        for idx in range(self.start_layer, self.end_layer):
            hidden_states, residual = layers[idx](
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=residual,
            )
        if residual is None:
            return self.norm(hidden_states)
        else:
            pass
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3TTSCodePredictor(nn.Module):
    def __init__(self, config: "Qwen3TTSTalkerConfig", prefix: str = "") -> None:
        super().__init__()
        self.config = config
        cp_config = config.code_predictor_config
        self.model = nn.Module()
        self.model.codec_embedding = nn.ModuleList(
            [
                nn.Embedding(cp_config.vocab_size, config.hidden_size)
                for _ in range(config.num_code_groups - 1)
            ]
        )
        self.model.layers = nn.ModuleList(
            [
                Qwen3TTSTalkerDecoderLayer(
                    cp_config,
                    idx,
                    prefix=add_prefix(f"model.layers.{idx}", prefix),
                )
                for idx in range(cp_config.num_hidden_layers)
            ]
        )
        self.model.norm = RMSNorm(cp_config.hidden_size, eps=cp_config.rms_norm_eps)
        self.lm_head = nn.ModuleList(
            [
                ReplicatedLinear(
                    cp_config.hidden_size,
                    cp_config.vocab_size,
                    bias=False,
                    prefix=add_prefix(f"lm_head.{idx}", prefix),
                )
                for idx in range(config.num_code_groups - 1)
            ]
        )
        if cp_config.hidden_size != config.hidden_size:
            self.small_to_mtp_projection = nn.Linear(
                config.hidden_size, cp_config.hidden_size, bias=True
            )
        else:
            self.small_to_mtp_projection = None

    def project_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.small_to_mtp_projection is None:
            return hidden_states
        else:
            pass
        return self.small_to_mtp_projection(hidden_states)


class Qwen3TTSPromptBuilderMixin:
    """Prompt construction shared by the talker and the standalone prompt frontend.

    Expects ``model`` (embedding tables and ``_feedback_buffer``), ``text_projection``,
    ``code_predictor.model.codec_embedding``, ``speaker_encoder``, ``speech_tokenizer``,
    ``config``/``root_config`` and ``speaker_encoder_sample_rate`` on the instance.
    """

    @property
    def device(self) -> torch.device:
        return self.model.codec_embedding.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.model.codec_embedding.weight.dtype

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def get_text_embeddings(self) -> nn.Embedding:
        return self.model.get_text_embeddings()

    def load_speech_tokenizer(
        self, speech_tokenizer: "Qwen3TTSTokenizer | None"
    ) -> None:
        self.speech_tokenizer = speech_tokenizer

    def get_supported_languages(self):
        return ["auto", *list(self.config.codec_language_id.keys())]

    def get_supported_speakers(self):
        return (getattr(self.config, "spk_id", None) or {}).keys()

    @torch.inference_mode()
    def extract_speaker_embedding(self, audio, sr):
        apply_qwen_tts_transformers_compatibility_patches()
        from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

        if sr != self.speaker_encoder_sample_rate:
            raise ValueError(
                f"Expected {self.speaker_encoder_sample_rate}Hz reference audio"
            )
        else:
            pass
        if self.speaker_encoder is None:
            raise RuntimeError("Qwen3-TTS speaker encoder is not loaded")
        else:
            pass
        mels = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=self.speaker_encoder_sample_rate,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)
        return self.speaker_encoder(mels.to(self.device).to(self.dtype))[0]

    @torch.inference_mode()
    def generate_speaker_prompt(
        self, voice_clone_prompt: Mapping[str, Sequence[object]] | VoicePrompt
    ):
        return [
            emb.to(self.device).to(self.dtype)
            for emb in voice_clone_prompt["ref_spk_embedding"]
        ]

    def build_instruct_embed(
        self,
        instruct_id: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if instruct_id is None:
            return None
        else:
            pass
        return self.text_projection(self.get_text_embeddings()(instruct_id))

    def build_tts_special_embeds(
        self,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = torch.tensor(
            [
                [
                    self.root_config.tts_bos_token_id,
                    self.root_config.tts_eos_token_id,
                    self.root_config.tts_pad_token_id,
                ]
            ],
            device=self.device,
            dtype=dtype,
        )
        return self.text_projection(self.get_text_embeddings()(ids)).chunk(3, dim=1)

    def resolve_language_id(
        self,
        *,
        language: str,
        voice: str | None = None,
    ) -> int | None:
        # Note(yzxiao): QwenLM/Qwen3-TTS (qwen-tts 0.1.1) also applies speaker
        # dialects to Chinese. Keep explicit languages unchanged here so existing
        # Chinese requests do not switch to a dialect when using Eric or Dylan.
        if language.lower() != "auto":
            return self.config.codec_language_id[language.lower()]
        else:
            pass
        if voice is None:
            return None
        else:
            pass
        spk_is_dialect = getattr(self.config, "spk_is_dialect", None) or {}
        dialect = spk_is_dialect.get(voice.lower())
        if isinstance(dialect, str) and dialect:
            return self.config.codec_language_id.get(dialect)
        else:
            pass
        return None

    def build_codec_prefill(
        self,
        *,
        language: str,
        dtype: torch.dtype,
        voice: str | None = None,
    ) -> torch.Tensor:
        language_id = self.resolve_language_id(language=language, voice=voice)
        if language_id is None:
            codec_prefill = [
                self.config.codec_nothink_id,
                self.config.codec_think_bos_id,
                self.config.codec_think_eos_id,
            ]
        else:
            codec_prefill = [
                self.config.codec_think_id,
                self.config.codec_think_bos_id,
                language_id,
                self.config.codec_think_eos_id,
            ]
        return self.get_input_embeddings()(
            torch.tensor([codec_prefill], device=self.device, dtype=dtype)
        )

    def finish_text_prompt(
        self,
        *,
        talker_input_embed: torch.Tensor,
        input_id: torch.Tensor,
        codec_last_embed: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        tts_eos_embed: torch.Tensor,
        non_streaming_mode: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if non_streaming_mode:
            text_all = self.text_projection(
                self.get_text_embeddings()(input_id[:, 3:-5])
            )
            text_all = torch.cat([text_all, tts_eos_embed], dim=1)
            pad_ids = torch.full(
                (1, int(text_all.shape[1])),
                int(self.config.codec_pad_id),
                device=self.device,
                dtype=input_id.dtype,
            )
            talker_input_embed = torch.cat(
                [
                    talker_input_embed,
                    text_all + self.get_input_embeddings()(pad_ids),
                    tts_pad_embed
                    + self.get_input_embeddings()(
                        torch.tensor(
                            [[self.config.codec_bos_id]],
                            device=self.device,
                            dtype=input_id.dtype,
                        )
                    ),
                ],
                dim=1,
            )
            return talker_input_embed, tts_pad_embed
        else:
            pass

        first_text = (
            self.text_projection(self.get_text_embeddings()(input_id[:, 3:4]))
            + codec_last_embed
        )
        talker_input_embed = torch.cat([talker_input_embed, first_text], dim=1)
        trailing_text_hidden = torch.cat(
            [
                self.text_projection(self.get_text_embeddings()(input_id[:, 4:-5])),
                tts_eos_embed,
            ],
            dim=1,
        )
        return talker_input_embed, trailing_text_hidden

    def apply_instruct_prefix(
        self,
        talker_input_embed: torch.Tensor,
        instruct_id: torch.Tensor | None,
    ) -> torch.Tensor:
        instruct_embed = self.build_instruct_embed(instruct_id)
        if instruct_embed is None:
            return talker_input_embed
        else:
            pass
        return torch.cat([instruct_embed, talker_input_embed], dim=1)

    def build_conditioned_prompt_prefix(
        self,
        *,
        input_id: torch.Tensor,
        codec_input: torch.Tensor,
        tts_bos_embed: torch.Tensor,
        tts_pad_embed: torch.Tensor,
    ) -> torch.Tensor:
        role_embed = self.text_projection(self.get_text_embeddings()(input_id[:, :3]))
        prompt_embed = (
            torch.cat(
                [tts_pad_embed.expand(-1, codec_input.shape[1] - 2, -1), tts_bos_embed],
                dim=1,
            )
            + codec_input[:, :-1]
        )
        return torch.cat([role_embed, prompt_embed], dim=1)

    def build_voice_clone_inputs(
        self,
        *,
        input_id: torch.Tensor,
        ref_id: torch.Tensor | None,
        voice_clone_prompt: Mapping[str, Sequence[object]] | VoicePrompt,
        language: str,
        non_streaming_mode: bool,
        instruct_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        voice_clone_spk_embeds = self.generate_speaker_prompt(voice_clone_prompt)
        speaker_embed = voice_clone_spk_embeds[0]

        tts_bos_embed, tts_eos_embed, tts_pad_embed = self.build_tts_special_embeds(
            dtype=input_id.dtype
        )
        codec_input_0 = self.build_codec_prefill(
            language=language,
            dtype=input_id.dtype,
        )
        codec_input_1 = self.get_input_embeddings()(
            torch.tensor(
                [[self.config.codec_pad_id, self.config.codec_bos_id]],
                device=self.device,
                dtype=input_id.dtype,
            )
        )
        codec_input = torch.cat(
            [codec_input_0, speaker_embed.view(1, 1, -1), codec_input_1], dim=1
        )
        talker_input_embed = self.build_conditioned_prompt_prefix(
            input_id=input_id,
            codec_input=codec_input,
            tts_bos_embed=tts_bos_embed,
            tts_pad_embed=tts_pad_embed,
        )

        ref_code = None
        ref_codes = voice_clone_prompt.get("ref_code")
        if ref_codes is not None:
            ref_code = ref_codes[0]
        else:
            pass

        if ref_code is not None and voice_clone_prompt["icl_mode"][0]:
            if ref_id is None:
                raise ValueError("Qwen3-TTS ICL mode requires ref_text tokens")
            else:
                pass
            icl_embed, trailing_text_hidden = self.generate_icl_prompt(
                text_id=input_id[:, 3:-5],
                ref_id=ref_id[:, 3:-2],
                ref_code=ref_code.to(self.device),
                tts_pad_embed=tts_pad_embed,
                tts_eos_embed=tts_eos_embed,
                non_streaming_mode=non_streaming_mode,
            )
            talker_input_embed = torch.cat([talker_input_embed, icl_embed], dim=1)
        else:
            talker_input_embed, trailing_text_hidden = self.finish_text_prompt(
                talker_input_embed=talker_input_embed,
                input_id=input_id,
                codec_last_embed=codec_input[:, -1:],
                tts_pad_embed=tts_pad_embed,
                tts_eos_embed=tts_eos_embed,
                non_streaming_mode=non_streaming_mode,
            )

        talker_input_embed = self.apply_instruct_prefix(
            talker_input_embed,
            instruct_id,
        )
        attention_mask = torch.ones(
            (1, talker_input_embed.shape[1]), device=self.device, dtype=torch.long
        )
        return talker_input_embed, attention_mask, trailing_text_hidden, ref_code

    def build_custom_voice_inputs(
        self,
        *,
        input_id: torch.Tensor,
        voice: str,
        language: str,
        non_streaming_mode: bool,
        instruct_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        spk_id = getattr(self.config, "spk_id", None) or {}
        if not spk_id:
            raise ValueError(
                "Qwen3-TTS CustomVoice requires a checkpoint with configured spk_id"
            )
        else:
            pass
        speaker_key = voice.lower()
        spk_id_map = {str(key).lower(): value for key, value in spk_id.items()}
        if speaker_key not in spk_id_map:
            supported = ", ".join(sorted(str(key) for key in spk_id))
            raise ValueError(
                f"Unsupported Qwen3-TTS CustomVoice speaker {voice!r}. "
                f"Supported speakers: {supported}"
            )
        else:
            pass

        tts_bos_embed, tts_eos_embed, tts_pad_embed = self.build_tts_special_embeds(
            dtype=input_id.dtype
        )
        codec_input_0 = self.build_codec_prefill(
            language=language,
            dtype=input_id.dtype,
            voice=speaker_key,
        )
        speaker_embed = self.get_input_embeddings()(
            torch.tensor(
                [spk_id_map[speaker_key]], device=self.device, dtype=input_id.dtype
            )
        ).view(1, 1, -1)
        codec_input_1 = self.get_input_embeddings()(
            torch.tensor(
                [[self.config.codec_pad_id, self.config.codec_bos_id]],
                device=self.device,
                dtype=input_id.dtype,
            )
        )
        codec_input = torch.cat([codec_input_0, speaker_embed, codec_input_1], dim=1)
        talker_input_embed = self.build_conditioned_prompt_prefix(
            input_id=input_id,
            codec_input=codec_input,
            tts_bos_embed=tts_bos_embed,
            tts_pad_embed=tts_pad_embed,
        )
        talker_input_embed, trailing_text_hidden = self.finish_text_prompt(
            talker_input_embed=talker_input_embed,
            input_id=input_id,
            codec_last_embed=codec_input[:, -1:],
            tts_pad_embed=tts_pad_embed,
            tts_eos_embed=tts_eos_embed,
            non_streaming_mode=non_streaming_mode,
        )
        talker_input_embed = self.apply_instruct_prefix(
            talker_input_embed,
            instruct_id,
        )
        attention_mask = torch.ones(
            (1, talker_input_embed.shape[1]), device=self.device, dtype=torch.long
        )
        return talker_input_embed, attention_mask, trailing_text_hidden, None

    def build_voice_design_inputs(
        self,
        *,
        input_id: torch.Tensor,
        language: str,
        non_streaming_mode: bool,
        instruct_id: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        if instruct_id is None:
            raise ValueError("Qwen3-TTS VoiceDesign requires instructions")
        else:
            pass

        tts_bos_embed, tts_eos_embed, tts_pad_embed = self.build_tts_special_embeds(
            dtype=input_id.dtype
        )
        codec_input_0 = self.build_codec_prefill(
            language=language,
            dtype=input_id.dtype,
        )
        codec_input_1 = self.get_input_embeddings()(
            torch.tensor(
                [[self.config.codec_pad_id, self.config.codec_bos_id]],
                device=self.device,
                dtype=input_id.dtype,
            )
        )
        codec_input = torch.cat([codec_input_0, codec_input_1], dim=1)
        talker_input_embed = self.build_conditioned_prompt_prefix(
            input_id=input_id,
            codec_input=codec_input,
            tts_bos_embed=tts_bos_embed,
            tts_pad_embed=tts_pad_embed,
        )
        talker_input_embed, trailing_text_hidden = self.finish_text_prompt(
            talker_input_embed=talker_input_embed,
            input_id=input_id,
            codec_last_embed=codec_input[:, -1:],
            tts_pad_embed=tts_pad_embed,
            tts_eos_embed=tts_eos_embed,
            non_streaming_mode=non_streaming_mode,
        )
        talker_input_embed = self.apply_instruct_prefix(
            talker_input_embed,
            instruct_id,
        )
        attention_mask = torch.ones(
            (1, talker_input_embed.shape[1]), device=self.device, dtype=torch.long
        )
        return talker_input_embed, attention_mask, trailing_text_hidden, None

    def generate_icl_prompt(
        self,
        text_id: torch.Tensor,
        ref_id: torch.Tensor,
        ref_code: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        tts_eos_embed: torch.Tensor,
        non_streaming_mode: bool,
    ):
        text_embed = self.text_projection(
            self.get_text_embeddings()(torch.cat([ref_id, text_id], dim=-1))
        )
        text_embed = torch.cat([text_embed, tts_eos_embed], dim=1)
        codec_embed = []
        for idx in range(self.config.num_code_groups):
            if idx == 0:
                codec_embed.append(self.get_input_embeddings()(ref_code[:, :1]))
            else:
                codec_embed.append(
                    self.code_predictor.model.codec_embedding[idx - 1](
                        ref_code[:, idx : idx + 1]
                    )
                )
        codec_embed = torch.cat(codec_embed, dim=1).sum(1).unsqueeze(0)
        codec_embed = torch.cat(
            [
                self.get_input_embeddings()(
                    torch.tensor(
                        [[self.config.codec_bos_id]],
                        device=self.device,
                        dtype=text_id.dtype,
                    )
                ),
                codec_embed,
            ],
            dim=1,
        )
        text_lens = text_embed.shape[1]
        codec_lens = codec_embed.shape[1]
        if non_streaming_mode:
            icl_input_embed = text_embed + self.get_input_embeddings()(
                torch.tensor(
                    [[self.config.codec_pad_id] * text_lens],
                    device=self.device,
                    dtype=text_id.dtype,
                )
            )
            icl_input_embed = torch.cat(
                [icl_input_embed, codec_embed + tts_pad_embed], dim=1
            )
            return icl_input_embed, tts_pad_embed
        else:
            pass
        if text_lens > codec_lens:
            return text_embed[:, :codec_lens] + codec_embed, text_embed[:, codec_lens:]
        else:
            pass
        text_embed = torch.cat(
            [text_embed] + [tts_pad_embed] * (codec_lens - text_lens), dim=1
        )
        return text_embed + codec_embed, tts_pad_embed


class Qwen3TTSTalker(Qwen3TTSPromptBuilderMixin, nn.Module):
    """Qwen3-TTS Base talker with SGLang-managed KV cache for the main AR loop."""

    # The outer forward substitutes ForwardBatch.mrope_positions whenever they
    # are present. Prefill CUDA graph runners capture the inner text model
    # directly and use this marker to preserve that position contract.
    is_mrope_enabled = True

    def __init__(
        self,
        config: "Qwen3TTSConfig | Qwen3TTSTalkerConfig",
        quant_config: object = None,
        prefix: str = "",
    ) -> None:
        del quant_config
        super().__init__()
        if hasattr(config, "talker_config"):
            root_config = config
            config = config.talker_config
        else:
            root_config = None
        self.root_config = root_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.tts_model_type = getattr(root_config, "tts_model_type", "base")
        self.tokenizer_type = getattr(root_config, "tokenizer_type", "")
        self.tts_model_size = getattr(root_config, "tts_model_size", "")
        self.speaker_encoder_sample_rate = getattr(
            getattr(root_config, "speaker_encoder_config", None),
            "sample_rate",
            24000,
        )

        self.text_projection = ResizeMLP(
            config.text_hidden_size,
            config.text_hidden_size,
            config.hidden_size,
            prefix=add_prefix("text_projection", prefix),
        )
        self.model = Qwen3TTSTalkerTextModel(config, prefix=add_prefix("model", prefix))
        self.codec_head = ReplicatedLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            prefix=add_prefix("codec_head", prefix),
        )
        self.code_predictor = Qwen3TTSCodePredictor(
            config,
            prefix=add_prefix("code_predictor", prefix),
        )

        if root_config is not None and self.tts_model_type == "base":
            apply_qwen_tts_transformers_compatibility_patches()
            from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder

            self.speaker_encoder = Qwen3TTSSpeakerEncoder(
                root_config.speaker_encoder_config
            )
        else:
            self.speaker_encoder = None
        self.speech_tokenizer = None

        server_args = get_context().server_args
        max_batch_size = get_schedule().max_running_requests
        hidden_size = config.hidden_size
        predictor_len = config.num_code_groups + 1
        device = self.model.codec_embedding.weight.device
        dtype = self.model.codec_embedding.weight.dtype
        self.feedback_buffer = self.model.feedback_buffer
        self.feedback_mask = self.model.feedback_mask
        self.decode_feedback_embedding = self.model.decode_feedback_embedding
        cp_layers = self.code_predictor.model.layers
        cp_attn = cp_layers[0].self_attn
        self.predictor_positions = torch.arange(
            predictor_len, device=device, dtype=torch.long
        )
        self.predictor_position_rows = (
            self.predictor_positions[:, None]
            .expand(predictor_len, max_batch_size)
            .contiguous()
        )
        self.predictor_device = device
        self.predictor_device_module = torch.get_device_module(device)
        # note(ratish): slot major, so the rope kernel stores k and v as one
        # row per (batch row, slot) and the attention reads a transposed view.
        self.predictor_k_cache = torch.zeros(
            len(cp_layers),
            max_batch_size,
            predictor_len,
            cp_attn.num_kv_heads,
            cp_attn.head_dim,
            device=device,
            dtype=dtype,
        )
        self.predictor_v_cache = torch.zeros_like(self.predictor_k_cache)
        self.predictor_k_rows = [
            layer_cache.view(max_batch_size * predictor_len, -1)
            for layer_cache in self.predictor_k_cache
        ]
        self.predictor_v_rows = [
            layer_cache.view(max_batch_size * predictor_len, -1)
            for layer_cache in self.predictor_v_cache
        ]
        self.predictor_cache_slots = (
            torch.arange(max_batch_size, device=device, dtype=torch.long)[None, :]
            * predictor_len
            + self.predictor_positions[:, None]
        ).contiguous()
        # note(ratish): the first pass carries two tokens per request, request by
        # request, so a batch's positions and cache slots are a prefix of these.
        self.predictor_pair_positions = self.predictor_positions[:2].repeat(
            max_batch_size
        )
        self.predictor_pair_cache_slots = self.predictor_cache_slots[:2].t().reshape(-1)
        self.predictor_rope_stores_kv = self.resolve_predictor_rope_store(
            cp_attn, device=device
        )
        self.sampled_token_ids = torch.zeros(
            max_batch_size, dtype=torch.long, device=device
        )
        self.output_codes = torch.zeros(
            max_batch_size,
            config.num_code_groups,
            dtype=torch.long,
            device=device,
        )
        self.output_embeds = torch.zeros(
            max_batch_size, hidden_size, device=device, dtype=dtype
        )
        self.predictor_embedding_buffer = torch.empty(
            max_batch_size, hidden_size, device=device, dtype=dtype
        )
        self.sub_batch_size = 0
        self.sub_temperature_tensor = torch.full(
            (max_batch_size,), 0.9, device=device, dtype=torch.float32
        )
        self.sub_top_p_tensor = torch.ones(
            max_batch_size, device=device, dtype=torch.float32
        )
        self.sub_top_k_tensor = torch.full(
            (max_batch_size,), 50, device=device, dtype=torch.long
        )
        self.semantic_sampling_seed_tensor = torch.zeros(
            max_batch_size, device=device, dtype=torch.long
        )
        self.sub_sampling_seed_tensor = torch.zeros(
            max_batch_size, device=device, dtype=torch.long
        )
        self.sub_do_sample_tensor = torch.zeros(
            max_batch_size, device=device, dtype=torch.bool
        )
        self.sub_seed_offsets = torch.arange(
            1, config.num_code_groups, device=device, dtype=torch.long
        )
        self.sub_has_sampled_rows = False
        self.sub_has_argmax_rows = False
        self.sub_sampled_has_top_p = False
        self.sub_sampled_max_top_k = 0
        self.sub_sampled_has_unbounded_top_k = False
        self.decode_prep_rids: list[tuple[str, int]] | None = None
        self.predictor_graph_batch_sizes = self.normalize_predictor_graph_batch_sizes(
            server_args,
            max_batch_size=max_batch_size,
        )
        self.predictor_graphs: dict[PredictorGraphKey, PredictorDecodeGraph] = {}
        self.predictor_graph_disabled: set[PredictorGraphKey] = set()
        # note(ratish): None until the startup capture, which runs before the
        # KV pool is sized, or the first decode resolves it.
        self.predictor_graph_enabled: bool | None = None
        self.predictor_graph_failure_count = 0
        self.predictor_graph_capacity_fallback_count = 0
        self.predictor_graph_capacity_warned = False
        self.predictor_graph_capture_count = 0
        self.predictor_graph_startup_count = 0
        self.predictor_graph_pool = None
        self.predictor_capture_stream: torch.Stream | None = None
        bind_default_weight_loaders(self)
        self.cached_params_dict = dict(self.named_parameters())
        self.sampler = None

    def prepare_decode_buffers(self, requests: list[SchedulerRequest]) -> None:
        batch_size = len(requests)
        if batch_size > self.sub_temperature_tensor.shape[0]:
            raise RuntimeError("Qwen3-TTS sampling buffers are too small")
        else:
            pass

        # Note: (Jiaxin Deng) every staged value here is static per request, so
        # an unchanged batch composition can reuse the previous staging wholesale.
        # Request ids alone are reusable across request lifetimes, so identity is
        # (request_id, per-data epoch); test doubles without request_id restage.
        rids: list[tuple[str, int]] | None = []
        for sched_req in requests:
            rid = getattr(sched_req, "request_id", None)
            if rid is None:
                rids = None
                break
            else:
                pass
            data = sched_req.data
            epoch = getattr(
                data, "_qwen3_tts_prep_epoch", None
            )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            if epoch is None:
                epoch = self.decode_prep_epoch = (
                    getattr(self, "decode_prep_epoch", 0) + 1
                )
                data._qwen3_tts_prep_epoch = epoch  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            else:
                pass
            rids.append((rid, epoch))
        if rids is not None and rids == getattr(self, "decode_prep_rids", None):
            return
        else:
            pass
        self.decode_prep_rids = None

        semantic_seeds: list[int] = []
        sub_temperatures: list[float] = []
        sub_top_ps: list[float] = []
        sub_top_ks: list[int] = []
        sub_seeds: list[int] = []
        sub_do_samples: list[bool] = []
        sample_rows: list[int] = []
        for row_idx, sched_req in enumerate(requests):
            data = sched_req.data
            try:
                semantic_seed = int(data.semantic_sampling_seed)
                do_sample = bool(data.subtalker_dosample)
                subtalker_temperature = float(data.subtalker_temperature)
                subtalker_top_p = float(data.subtalker_top_p)
                subtalker_top_k = int(data.subtalker_top_k)
                subtalker_seed = int(data.subtalker_sampling_seed)
            except AttributeError as exc:
                raise TypeError(
                    "Qwen3-TTS decode buffers require request data with semantic "
                    "and subtalker sampling fields"
                ) from exc
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "Qwen3-TTS decode buffers require numeric semantic and "
                    "subtalker sampling fields"
                ) from exc
            semantic_seeds.append(semantic_seed)
            sub_do_samples.append(do_sample)
            # note(ratish): the sampler divides by the temperature, staged clamped
            # so the sub-steps read it without a kernel.
            sub_temperatures.append(
                max(subtalker_temperature, 1e-5) if do_sample else 1.0
            )
            sub_top_ps.append(subtalker_top_p if do_sample else 1.0)
            # Note (Shulei He): a greedy row's original top_k can be 0 or -1,
            # which would otherwise hit the full-sort branch.
            sub_top_ks.append(subtalker_top_k if do_sample else 1)
            sub_seeds.append(subtalker_seed)
            if do_sample:
                sample_rows.append(row_idx)
            else:
                pass

        predictor_vocab_size = int(self.config.code_predictor_config.vocab_size)
        max_top_k, has_top_p, has_unbounded_top_k = predictor_signature_terms(
            [sub_top_ks[row_idx] for row_idx in sample_rows],
            [sub_top_ps[row_idx] for row_idx in sample_rows],
            predictor_vocab_size,
        )
        self.sub_batch_size = batch_size
        self.sub_has_sampled_rows = bool(sample_rows)
        self.sub_has_argmax_rows = len(sample_rows) < batch_size
        self.sub_sampled_has_top_p = has_top_p
        self.sub_sampled_max_top_k = max_top_k
        self.sub_sampled_has_unbounded_top_k = has_unbounded_top_k

        if batch_size == 0:
            self.decode_prep_rids = rids
            return
        else:
            pass

        # note(ratish): a pageable source waits for the last step's predictor.
        pin_memory = is_pin_memory_available(self.sub_temperature_tensor.device)
        for buffer, values in (
            (self.semantic_sampling_seed_tensor, semantic_seeds),
            (self.sub_temperature_tensor, sub_temperatures),
            (self.sub_top_p_tensor, sub_top_ps),
            (self.sub_top_k_tensor, sub_top_ks),
            (self.sub_sampling_seed_tensor, sub_seeds),
            (self.sub_do_sample_tensor, sub_do_samples),
        ):
            buffer[:batch_size].copy_(
                torch.tensor(values, dtype=buffer.dtype, pin_memory=pin_memory),
                non_blocking=True,
            )
        self.decode_prep_rids = rids

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        input_embeds_are_projected: bool = False,
        omni_prefill_rids: list[str] | None = None,
    ) -> LogitsProcessorOutput:
        del input_embeds_are_projected, omni_prefill_rids
        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions
        else:
            pass

        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )
        if forward_batch.forward_mode.is_extend():
            last_index = self.extend_last_index(forward_batch, hidden_states.device)
            hidden_states = hidden_states[last_index]
        else:
            pass
        logits, _ = self.codec_head(hidden_states)
        logits_output = LogitsProcessorOutput(
            next_token_logits=logits,
            hidden_states=hidden_states,
        )
        return logits_output

    def extend_last_index(
        self,
        forward_batch: ForwardBatch,
        device: torch.device,
    ) -> torch.Tensor:
        extend_seq_lens = forward_batch.extend_seq_lens
        if extend_seq_lens is None:
            return torch.tensor([forward_batch.input_ids.shape[0] - 1], device=device)
        else:
            pass
        return torch.cumsum(extend_seq_lens.to(device=device), dim=0) - 1

    @torch.no_grad()
    def code_predictor_forward(
        self,
        layer0_codes: torch.Tensor,
        talker_hidden: torch.Tensor,
        semantic_positions: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer0_codes.ndim == 1:
            layer0_codes = layer0_codes.unsqueeze(1)
        else:
            pass
        if talker_hidden.ndim == 2:
            talker_hidden = talker_hidden.unsqueeze(1)
        else:
            pass
        graph_result = self.predictor_forward_graphed(
            layer0_codes,
            talker_hidden,
            semantic_positions,
        )
        if graph_result is not None:
            return graph_result
        else:
            pass
        result_codes, summed_embeddings = self.code_predictor_forward_incremental(
            layer0_codes=layer0_codes,
            talker_hidden=talker_hidden,
            semantic_positions=semantic_positions,
        )
        return result_codes, summed_embeddings

    @staticmethod
    def normalize_predictor_graph_batch_sizes(
        server_args: object,
        *,
        max_batch_size: int,
    ) -> tuple[int, ...]:
        from sglang_omni.scheduling.generation_batch_policy import (
            get_decode_cuda_graph_bs,
        )

        raw_batch_sizes = get_decode_cuda_graph_bs(server_args)
        if raw_batch_sizes is None:
            # Note: (Jiaxin Deng) mirrors the backbone's default capture list
            # ([1, 2, 4, 8, 12, 16] at max_running_requests=16).
            raw_batch_sizes = (1, 2, 4, *range(8, int(max_batch_size) + 1, 4))
        else:
            pass
        normalized = sorted(
            {
                int(batch_size)
                for batch_size in raw_batch_sizes
                if 1 <= int(batch_size) <= int(max_batch_size)
            }
        )
        if not normalized or normalized[-1] < int(max_batch_size):
            normalized.append(int(max_batch_size))
        else:
            pass
        return tuple(normalized)

    def predictor_graph_bucket_size(self, batch_size: int) -> int | None:
        for bucket_size in self.predictor_graph_batch_sizes:
            if bucket_size >= batch_size:
                return bucket_size
            else:
                pass
        return None

    def predictor_graph_signature(
        self,
        batch_size: int,
        semantic_positions: torch.Tensor | None,
    ) -> PredictorGraphSignature | None:
        if semantic_positions is not None:
            if semantic_positions.device != self.predictor_device:
                return None
            else:
                pass
            if (
                semantic_positions.ndim not in (1, 2)
                or semantic_positions.shape[0] != batch_size
            ):
                return None
            else:
                pass
            if semantic_positions.ndim == 2 and semantic_positions.shape[1] != 1:
                return None
            else:
                pass
        else:
            pass
        if not self.sub_has_sampled_rows:
            return ("argmax", 0, False, False, False)
        else:
            pass
        if semantic_positions is None:
            return None
        else:
            pass
        return (
            "sampled",
            int(self.sub_sampled_max_top_k),
            bool(self.sub_sampled_has_top_p),
            bool(self.sub_sampled_has_unbounded_top_k),
            bool(self.sub_has_argmax_rows),
        )

    @contextmanager
    def predictor_graph_capture_state(
        self, bucket_size: int, signature: PredictorGraphSignature
    ) -> Generator[None, None, None]:
        saved = (
            self.sub_batch_size,
            self.sub_has_sampled_rows,
            self.sub_sampled_max_top_k,
            self.sub_sampled_has_top_p,
            self.sub_sampled_has_unbounded_top_k,
            self.sub_has_argmax_rows,
        )
        try:
            self.sub_batch_size = bucket_size
            self.sub_has_sampled_rows = signature[0] == "sampled"
            (
                _,
                self.sub_sampled_max_top_k,
                self.sub_sampled_has_top_p,
                self.sub_sampled_has_unbounded_top_k,
                self.sub_has_argmax_rows,
            ) = signature
            yield
        finally:
            (
                self.sub_batch_size,
                self.sub_has_sampled_rows,
                self.sub_sampled_max_top_k,
                self.sub_sampled_has_top_p,
                self.sub_sampled_has_unbounded_top_k,
                self.sub_has_argmax_rows,
            ) = saved

    def predictor_graph_memory_pool(self):
        # Note: (Jiaxin Deng) one shared pool across keys; private per-graph
        # pools would retain intermediates per key and scale with diversity.
        if self.predictor_graph_pool is None:
            self.predictor_graph_pool = self.predictor_device_module.graph_pool_handle()
        else:
            pass
        return self.predictor_graph_pool

    def resolve_predictor_graph_enabled(self) -> bool:
        # Device first: a device that cannot record answers without published config.
        if current_platform.get_device_graph_backend(self.predictor_device) is None:
            return False
        else:
            pass
        # Note: (Jiaxin Deng) capture under TP would record collectives; the
        # graphed chain is only validated single-rank, so TP stays eager.
        if int(get_parallel().tp_size) != 1:
            return False
        else:
            pass
        override = predictor_graph_env_override()
        if override is None:
            should_capture = current_platform.enable_tts_predictor_graph()
        else:
            should_capture = override
        if not should_capture:
            return False
        else:
            pass
        return not bool(get_exec().graph.disable_cuda_graph)

    def capture_predictor_graphs(
        self,
        *,
        do_sample: bool,
        top_k: int,
        top_p: float,
    ) -> int:
        """Capture the bucket ladder of the signatures a batch gets when its rows
        sample with these values, with and without argmax rows mixed in. Buckets
        go in descending order so the smaller ones reuse the pool of the larger
        ones. The set is captured whole and the lazy capture budget counts only
        keys beyond it. The mixed signature skips bucket 1: a mixed batch holds
        a sampled row and an argmax row."""
        if self.predictor_graph_enabled is None:
            self.predictor_graph_enabled = self.resolve_predictor_graph_enabled()
        else:
            pass
        if not self.predictor_graph_enabled:
            return 0
        else:
            pass
        signatures: list[PredictorGraphSignature]
        if do_sample:
            max_top_k, has_top_p, has_unbounded_top_k = predictor_signature_terms(
                [int(top_k)],
                [float(top_p)],
                int(self.config.code_predictor_config.vocab_size),
            )
            signatures = [
                ("sampled", max_top_k, has_top_p, has_unbounded_top_k, has_argmax_rows)
                for has_argmax_rows in (False, True)
            ]
        else:
            signatures = [("argmax", 0, False, False, False)]
        started = time.perf_counter()
        captured_before = len(self.predictor_graphs)
        for signature in signatures:
            for bucket_size in reversed(self.predictor_graph_batch_sizes):
                if signature[4] and bucket_size < 2:
                    continue
                else:
                    pass
                key = (bucket_size, *signature)
                if key in self.predictor_graphs:
                    continue
                else:
                    pass
                self.predictor_graphs[key] = self.capture_predictor_graph(
                    bucket_size, signature
                )
                self.predictor_graph_capture_count += 1
                self.predictor_graph_startup_count += 1
        captured = len(self.predictor_graphs) - captured_before
        elapsed_s = time.perf_counter() - started
        logger.info(
            f"Captured {captured} Qwen3-TTS predictor graphs for "
            f"signatures={signatures} in {elapsed_s:.1f} s"
        )
        return captured

    @torch.no_grad()
    def capture_predictor_graph(
        self,
        bucket_size: int,
        signature: PredictorGraphSignature,
    ) -> PredictorDecodeGraph:
        """One stream per talker for warmups and captures: the allocator only
        reuses a pool block on the stream that freed it. Automatic collection
        is off for the capture because a graph finalizer reached by the
        cyclic collector while a stream is capturing destroys its pool inside
        the capture."""
        device = self.predictor_device
        module = self.predictor_device_module
        backend = current_platform.get_device_graph_backend(device)
        if self.predictor_capture_stream is None:
            self.predictor_capture_stream = module.Stream(device=device)
        else:
            pass
        capture_stream = self.predictor_capture_stream
        current_stream = module.current_stream(device)
        graph = PredictorDecodeGraph(
            bucket_size,
            signature,
            device=device,
            hidden_size=int(self.output_embeds.shape[-1]),
            hidden_dtype=self.output_embeds.dtype,
        )
        # note(ratish): the buffers are zero filled on the current stream and
        # layer0_codes is an embedding index, so the capture stream waits here.
        capture_stream.wait_stream(current_stream)

        def run_once() -> Tuple[torch.Tensor, torch.Tensor]:
            return self.code_predictor_forward_incremental(
                graph.layer0_codes,
                graph.talker_hidden,
                semantic_positions=graph.semantic_positions,
            )

        gc_was_enabled = gc.isenabled()
        gc.disable()
        captured = None
        try:
            # note(ratish): the outer stream context restores the current stream
            # when a failed capture raises from capture_end before the graph
            # context restores it.
            with (
                module.device(device),
                current_platform.graph_capture_attention(),
                self.predictor_graph_capture_state(bucket_size, signature),
                module.stream(capture_stream),
            ):
                for _ in range(_PREDICTOR_GRAPH_WARMUP_PASSES):
                    run_once()
                with backend.capture(
                    pool=self.predictor_graph_memory_pool(),
                    stream=capture_stream,
                    thread_local_errors=True,
                ) as captured:
                    graph.result_codes, graph.summed_embeddings = run_once()
        except Exception:
            # Note: (Jiaxin Deng) release the graph's private memory pool
            # eagerly; the raising object may linger on traceback frames.
            if captured is not None:
                try:
                    captured.reset()
                except Exception:
                    pass
            else:
                pass
            raise
        finally:
            current_stream.wait_stream(capture_stream)
            if gc_was_enabled:
                gc.enable()
            else:
                pass
        graph.graph = captured
        if graph.result_codes is None or graph.summed_embeddings is None:
            raise RuntimeError("Qwen3-TTS predictor graph captured no outputs")
        else:
            pass
        return graph

    def predictor_forward_graphed(
        self,
        layer0_codes: torch.Tensor,
        talker_hidden: torch.Tensor,
        semantic_positions: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor] | None:
        if self.predictor_graph_enabled is None:
            self.predictor_graph_enabled = self.resolve_predictor_graph_enabled()
        else:
            pass
        if not self.predictor_graph_enabled:
            return None
        else:
            pass
        batch_size, seq_len = layer0_codes.shape
        if seq_len != 1 or batch_size == 0:
            return None
        else:
            pass
        if layer0_codes.dtype not in (torch.int, torch.long):
            return None
        else:
            pass
        graph_device = self.predictor_device
        if layer0_codes.device != graph_device or talker_hidden.device != graph_device:
            return None
        else:
            pass
        if batch_size != self.sub_batch_size:
            return None
        else:
            pass
        if self.predictor_device_module.is_current_stream_capturing():
            return None
        else:
            pass
        signature = self.predictor_graph_signature(batch_size, semantic_positions)
        if signature is None:
            return None
        else:
            pass
        bucket_size = self.predictor_graph_bucket_size(batch_size)
        if bucket_size is None:
            return None
        else:
            pass
        key = (bucket_size, *signature)
        if key in self.predictor_graph_disabled:
            return None
        else:
            pass
        graph = self.predictor_graphs.get(key)
        if graph is None:
            lazy_keys = len(self.predictor_graphs) - self.predictor_graph_startup_count
            if lazy_keys >= _PREDICTOR_GRAPH_MAX_LAZY_KEYS:
                self.predictor_graph_capacity_fallback_count += 1
                if not self.predictor_graph_capacity_warned:
                    self.predictor_graph_capacity_warned = True
                    logger.warning(
                        "Qwen3-TTS predictor graph cache holds %d keys beyond "
                        "the startup set; falling back to eager execution for "
                        "uncached key=%s",
                        lazy_keys,
                        key,
                    )
                else:
                    pass
                return None
            else:
                pass
            try:
                graph = self.capture_predictor_graph(bucket_size, signature)
            except Exception:
                self.predictor_graph_disabled.add(key)
                self.predictor_graph_failure_count += 1
                logger.warning(
                    "Disabling Qwen3-TTS predictor graph for key=%s",
                    key,
                    exc_info=True,
                )
                if self.predictor_graph_failure_count >= _PREDICTOR_GRAPH_MAX_FAILURES:
                    self.predictor_graph_enabled = False
                    logger.warning(
                        "Disabling Qwen3-TTS predictor graphs entirely "
                        "after %d capture failures",
                        self.predictor_graph_failure_count,
                    )
                else:
                    pass
                return None
            self.predictor_graphs[key] = graph
            self.predictor_graph_capture_count += 1
            logger.info("Captured Qwen3-TTS predictor graph for key=%s", key)
        else:
            pass
        result = graph.replay(layer0_codes, talker_hidden, semantic_positions)
        return result

    def code_predictor_forward_incremental(
        self,
        layer0_codes: torch.Tensor,
        talker_hidden: torch.Tensor,
        semantic_positions: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer0_codes.ndim == 1:
            layer0_codes = layer0_codes.unsqueeze(1)
        else:
            pass
        if talker_hidden.ndim == 2:
            talker_hidden = talker_hidden.unsqueeze(1)
        else:
            pass

        batch_size, seq_len = layer0_codes.shape
        semantic_positions = self.normalize_semantic_positions(
            semantic_positions,
            batch_size=batch_size,
            seq_len=seq_len,
            device=layer0_codes.device,
        )
        predictor_dtype = self.predictor_k_cache.dtype
        num_groups = self.config.num_code_groups
        result_codes = self.output_codes[:batch_size].unsqueeze(-1)
        summed_embeddings = self.output_embeds[:batch_size].unsqueeze(1)
        result_codes.zero_()
        summed_embeddings.zero_()
        embedding_buffer = getattr(self, "predictor_embedding_buffer", None)
        if embedding_buffer is not None:
            embedding_buffer = embedding_buffer[:batch_size]
        else:
            pass
        use_fused_embedding = embedding_buffer is not None and layer0_codes.is_cuda

        for pos in range(seq_len):
            layer0_code = layer0_codes[:, pos : pos + 1]
            layer0_embed = self.get_input_embeddings()(layer0_code).to(
                dtype=predictor_dtype
            )
            pos_codes = result_codes[:, :, pos]
            pos_summed = summed_embeddings[:, pos, :]
            pos_summed.zero_()
            pos_codes[:, 0].copy_(layer0_code[:, 0])
            pos_summed.add_(layer0_embed[:, 0, :])

            # note(ratish): the talker hidden and the layer 0 embedding are the
            # first two tokens; one causal pass reads the predictor weights once.
            pair_embeds = self.code_predictor.project_input(
                torch.cat(
                    (
                        talker_hidden[:, pos : pos + 1, :].to(dtype=predictor_dtype),
                        layer0_embed,
                    ),
                    dim=1,
                )
            )
            # note(ratish): dense rows, the gemv and cutedsl lm_head GEMMs view
            # their input as 2D and read it with a fixed row stride.
            last_hidden = self.predictor_forward_tokens(
                token_embeds=pair_embeds,
                batch_size=batch_size,
                cache_len=0,
            )[:, 1:, :].contiguous()
            cache_len = pair_embeds.shape[1]

            sub_positions = (
                self.sub_seed_positions(semantic_positions[:, pos])
                if self.sub_has_sampled_rows
                else None
            )
            for layer_idx in range(num_groups - 1):
                logits, _ = self.code_predictor.lm_head[layer_idx](last_hidden)
                next_code = self.sample_subtalker_token(
                    logits[:, -1, :],
                    sub_positions=(
                        None if sub_positions is None else sub_positions[layer_idx]
                    ),
                )
                pos_codes[:, layer_idx + 1].copy_(next_code)
                codec_embedding = self.code_predictor.model.codec_embedding[layer_idx]
                fused_embedding = (
                    use_fused_embedding
                    and embedding_buffer.dtype == predictor_dtype
                    and gather_codec_embedding_and_add(
                        next_code,
                        codec_embedding.weight,
                        embedding_buffer,
                        pos_summed,
                    )
                )
                if fused_embedding:
                    new_embed = embedding_buffer.unsqueeze(1)
                else:
                    new_embed = codec_embedding(next_code.unsqueeze(1)).to(
                        dtype=predictor_dtype
                    )
                    pos_summed.add_(new_embed[:, 0, :])
                new_predictor_embed = self.code_predictor.project_input(new_embed)
                if layer_idx < num_groups - 2:
                    last_hidden = self.predictor_forward_tokens(
                        token_embeds=new_predictor_embed,
                        batch_size=batch_size,
                        cache_len=cache_len,
                    )
                    cache_len += 1
                else:
                    pass
        return result_codes, summed_embeddings

    def normalize_semantic_positions(
        self,
        semantic_positions: torch.Tensor | None,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if semantic_positions is None:
            base = torch.zeros(batch_size, device=device, dtype=torch.long)
        else:
            base = semantic_positions.to(device=device, dtype=torch.long)
            if base.ndim == 2:
                if base.shape != (batch_size, seq_len):
                    raise ValueError("Qwen3-TTS subtalker positions shape mismatch")
                else:
                    pass
                return base
            else:
                pass
            if base.ndim != 1 or base.shape[0] != batch_size:
                raise ValueError("Qwen3-TTS subtalker positions shape mismatch")
            else:
                pass
        offsets = torch.arange(seq_len, device=device, dtype=torch.long)
        return base.unsqueeze(1) + offsets.unsqueeze(0)

    def sub_seed_positions(self, semantic_positions: torch.Tensor) -> torch.Tensor:
        """Seed positions of every sub-step of one decode position, sub-step
        first, so each sub-step reads its row without a kernel."""
        group_stride = max(int(self.config.num_code_groups) - 1, 1)
        return torch.add(
            self.sub_seed_offsets.unsqueeze(1),
            semantic_positions.unsqueeze(0),
            alpha=group_stride,
        )

    def sample_subtalker_token(
        self,
        logits: torch.Tensor,
        *,
        sub_positions: torch.Tensor | None,
    ) -> torch.Tensor:
        if logits.shape[0] == 0:
            return torch.empty((0,), device=logits.device, dtype=torch.long)
        else:
            pass
        batch_size = int(logits.shape[0])
        if batch_size > self.sub_batch_size:
            raise RuntimeError("Qwen3-TTS subtalker sampling buffers are too small")
        else:
            pass

        if not self.sub_has_sampled_rows:
            return torch.argmax(logits, dim=-1).to(dtype=torch.long)
        else:
            pass

        sampled_tokens = self.sample_subtalker_token_seeded(
            logits,
            sub_positions=sub_positions,
        )
        if not self.sub_has_argmax_rows:
            return sampled_tokens
        else:
            pass
        argmax_tokens = torch.argmax(logits, dim=-1).to(dtype=torch.long)
        return torch.where(
            self.sub_do_sample_tensor[:batch_size],
            sampled_tokens,
            argmax_tokens,
        )

    def sample_subtalker_token_seeded(
        self,
        logits: torch.Tensor,
        *,
        sub_positions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(logits.shape[0])
        vocab_size = int(logits.shape[-1])
        temperatures = self.sub_temperature_tensor[:batch_size]
        top_ks = self.sub_top_k_tensor[:batch_size]
        top_ps = self.sub_top_p_tensor[:batch_size]
        seeds = self.sub_sampling_seed_tensor[:batch_size]
        max_top_k = int(self.sub_sampled_max_top_k)
        has_unbounded_top_k = bool(self.sub_sampled_has_unbounded_top_k)

        if logits.is_cuda:
            fused_sampled = sample_from_logits_with_seed_top_k_top_p(
                logits,
                temperatures,
                top_ks,
                top_ps,
                seeds,
                sub_positions,
                max_top_k=max_top_k,
                has_top_p=bool(self.sub_sampled_has_top_p),
            )
            if fused_sampled is not None:
                return fused_sampled.to(torch.long)
            else:
                pass
        else:
            pass

        scores = logits.float() / temperatures.unsqueeze(1)
        if max_top_k > 0 and max_top_k < vocab_size and not has_unbounded_top_k:
            sorted_scores, sorted_idx = torch.topk(scores, max_top_k, dim=-1)
            rank = torch.arange(max_top_k, device=logits.device).unsqueeze(0)
            keep_top_k = rank < top_ks.unsqueeze(1)
            sorted_scores = sorted_scores.masked_fill(~keep_top_k, -float("inf"))
        else:
            sorted_scores, sorted_idx = torch.sort(scores, dim=-1, descending=True)
            rank = torch.arange(vocab_size, device=logits.device).unsqueeze(0)
            keep_all = (top_ks <= 0) | (top_ks >= vocab_size)
            keep_top_k = keep_all.unsqueeze(1) | (rank < top_ks.unsqueeze(1))
            sorted_scores = sorted_scores.masked_fill(~keep_top_k, -float("inf"))

        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        if self.sub_sampled_has_top_p:
            active_top_p = (top_ps > 0.0) & (top_ps < 1.0)
            cdf = torch.cumsum(sorted_probs, dim=-1)
            remove = (
                cdf - sorted_probs >= top_ps.unsqueeze(1)
            ) & active_top_p.unsqueeze(1)
            remove[:, 0] = False
            sorted_probs = sorted_probs.masked_fill(remove, -float("inf"))
        else:
            pass
        sorted_probs = sorted_probs.masked_fill(~keep_top_k, -float("inf"))
        sorted_logprobs = torch.where(
            sorted_probs > 0,
            torch.log(sorted_probs),
            torch.full_like(sorted_probs, -float("inf")),
        )

        sampled = sample_from_sorted_logprobs_with_seed_small_k(
            sorted_logprobs,
            sorted_idx,
            seeds,
            sub_positions,
        )
        if sampled is not None:
            return sampled.to(torch.long)
        else:
            pass

        sampled_rank = sample_seeded_categorical(
            sorted_logprobs,
            seeds,
            sub_positions,
        ).to(device=logits.device, dtype=torch.long)
        return sorted_idx.gather(1, sampled_rank.unsqueeze(1)).view(-1).to(torch.long)

    def predictor_forward_tokens(
        self,
        *,
        token_embeds: torch.Tensor,
        batch_size: int,
        cache_len: int,
    ) -> torch.Tensor:
        """Run the predictor stack on token_embeds [batch, tokens, hidden] at slots
        cache_len onward: one token at any slot, or the pair at slots 0 and 1."""
        num_tokens, hidden_size = token_embeds.shape[1:]
        if num_tokens == 1:
            positions = self.predictor_position_rows[cache_len, :batch_size]
            cache_slots = self.predictor_cache_slots[cache_len, :batch_size]
        else:
            assert (num_tokens, cache_len) == (2, 0), "pair tables hold slots 0 and 1"
            positions = self.predictor_pair_positions[: 2 * batch_size]
            cache_slots = self.predictor_pair_cache_slots[: 2 * batch_size]
        num_rows = batch_size * num_tokens
        # note(ratish): 2D rows for the fused add and norm, which every
        # backend's kernel expects and which writes both operands in place.
        residual = token_embeds.reshape(num_rows, 1, hidden_size)
        mlp_out: torch.Tensor | None = None
        for layer_idx, layer in enumerate(self.code_predictor.model.layers):
            if mlp_out is None:
                normed = layer.input_layernorm(residual.reshape(-1, hidden_size))
            else:
                normed, residual = layer.input_layernorm(
                    mlp_out, residual.reshape(-1, hidden_size)
                )
                residual = residual.reshape(num_rows, 1, hidden_size)
            attn_input = self.predictor_cached_self_attention(
                layer_idx=layer_idx,
                attn=layer.self_attn,
                hidden_states=normed.reshape(batch_size, num_tokens, hidden_size),
                positions=positions,
                cache_slots=cache_slots,
                cache_len=cache_len,
            )
            residual = self.predictor_o_proj_add_residual(
                layer.self_attn.o_proj,
                attn_input,
                residual,
            )
            normed = layer.post_attention_layernorm(residual.reshape(-1, hidden_size))
            mlp_out = layer.mlp(normed)
        normed, _ = self.code_predictor.model.norm(
            mlp_out, residual.reshape(-1, hidden_size)
        )
        return normed.reshape(batch_size, num_tokens, hidden_size)

    @staticmethod
    def predictor_o_proj_add_residual(
        o_proj: nn.Module,
        attn_input: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Run the Predictor attention output projection and residual add."""

        weight = getattr(o_proj, "weight", None)
        # note(ratish): the fusion stands in for sglang's linear only where that
        # linear is torch's GEMM: batch invariant mode overrides aten::addmm but
        # not the out variant, and the cutedsl backend replaces F.linear on sm100.
        use_fused_addmm = (
            attn_input.is_cuda
            and not is_batch_invariant_mode_enabled()
            and not get_bf16_gemm_backend().is_cutedsl()
            and not torch.is_grad_enabled()
            and isinstance(
                getattr(o_proj, "quant_method", None), UnquantizedLinearMethod
            )
            and getattr(o_proj, "tp_size", None) == 1
            and getattr(o_proj, "bias", None) is None
            and isinstance(weight, torch.Tensor)
            and weight.is_cuda
            and weight.dtype == torch.bfloat16
            and weight.ndim == 2
            and weight.is_contiguous()
            and attn_input.dtype == torch.bfloat16
            and attn_input.ndim == 2
            and attn_input.is_contiguous()
            and attn_input.device == weight.device
            and attn_input.shape[1] == weight.shape[1]
            and residual.is_cuda
            and residual.dtype == torch.bfloat16
            and residual.ndim == 3
            and residual.is_contiguous()
            and residual.device == weight.device
            and residual.shape == (attn_input.shape[0], 1, weight.shape[0])
        )
        if use_fused_addmm:
            residual_2d = residual.reshape(attn_input.shape[0], weight.shape[0])
            # Note (Jun Liu): The caller replaces ``residual`` immediately after
            # this call. Reuse its storage so addmm's beta term needs no copy.
            torch.addmm(
                residual_2d,
                attn_input,
                weight.t(),
                out=residual_2d,
            )
            return residual
        else:
            pass

        attn_output, _ = o_proj(attn_input)
        return residual + attn_output.reshape(
            attn_input.shape[0], 1, residual.shape[-1]
        )

    @staticmethod
    def resolve_predictor_rope_store(
        attn: Qwen3OmniMoeThinkerTextAttention, *, device: torch.device
    ) -> bool:
        """Resolve store support before capture: a supported CUDA head size
        does not imply CUDA dispatch when the backend override selects Torch."""
        return (
            device.type == "cuda"
            and attn.compatible_with_fused_kv_buffer
            and not attn.rotary_emb.use_fallback_kernel
            and get_fused_op_backend()
            not in (KernelBackend.TORCH, KernelBackend.TORCH_COMPILE)
        )

    def predictor_cached_self_attention(
        self,
        *,
        layer_idx: int,
        attn: Qwen3OmniMoeThinkerTextAttention,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cache_slots: torch.Tensor,
        cache_len: int,
    ) -> torch.Tensor:
        batch_size, num_tokens, hidden_size = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, hidden_size)
        qkv, _ = attn.qkv_proj(flat_hidden)
        q_linear, k_linear, v = qkv.split(
            [attn.q_size, attn.kv_size, attn.kv_size], dim=-1
        )
        q, k = apply_qk_norm(
            q=q_linear,
            k=k_linear,
            q_norm=attn.q_norm,
            k_norm=attn.k_norm,
            head_dim=attn.head_dim,
            alt_stream=attn.alt_stream,
        )
        if self.predictor_rope_stores_kv:
            store = FusedSetKVBufferArg(
                value=v,
                k_buffer=self.predictor_k_rows[layer_idx],
                v_buffer=self.predictor_v_rows[layer_idx],
                cache_loc=cache_slots,
            )
        else:
            store = None
        q, k = attn.rotary_emb(
            positions.to(device=flat_hidden.device, dtype=torch.long),
            q,
            k,
            fused_set_kv_buffer_arg=store,
        )
        end = cache_len + num_tokens
        if store is None:
            self.predictor_k_cache[layer_idx, :batch_size, cache_len:end].copy_(
                k.view(batch_size, num_tokens, attn.num_kv_heads, attn.head_dim)
            )
            self.predictor_v_cache[layer_idx, :batch_size, cache_len:end].copy_(
                v.view(batch_size, num_tokens, attn.num_kv_heads, attn.head_dim)
            )
        else:
            pass
        q = q.reshape(batch_size, num_tokens, attn.num_heads, attn.head_dim).transpose(
            1, 2
        )
        cached_k = self.predictor_k_cache[layer_idx, :batch_size, :end].transpose(1, 2)
        cached_v = self.predictor_v_cache[layer_idx, :batch_size, :end].transpose(1, 2)
        attn_output = predictor_gqa_attention(
            q,
            cached_k,
            cached_v,
            num_heads=attn.num_heads,
            num_key_value_heads=attn.num_kv_heads,
            is_causal=num_tokens > 1,
        )
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size * num_tokens, attn.num_heads * attn.head_dim
        )
        return attn_output

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = self.cached_params_dict
        stacked_params = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        for name, loaded_weight in weights:
            if name.startswith("talker."):
                target = name[len("talker.") :]
            elif name.startswith("speaker_encoder."):
                target = name
            else:
                continue

            handled = False
            for param_name, weight_name, shard_id in stacked_params:
                if weight_name in target:
                    param = params_dict.get(target.replace(weight_name, param_name))
                    if param is not None:
                        param.weight_loader(param, loaded_weight, shard_id)
                        handled = True
                        break
                    else:
                        pass
                else:
                    pass
            if handled:
                continue
            else:
                pass
            param = params_dict.get(target)
            if param is not None:
                weight_loader = getattr(param, "weight_loader", None)
                if weight_loader is None:
                    param.data.copy_(loaded_weight)
                else:
                    weight_loader(param, loaded_weight)
            else:
                pass


EntryClass = Qwen3TTSTalker
