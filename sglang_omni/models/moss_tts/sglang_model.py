# SPDX-License-Identifier: Apache-2.0
"""SGLang-native MOSS-TTS Delay model wrapper."""

from __future__ import annotations

import logging
import os
from copy import copy
from typing import Any, Iterable, Optional, Tuple

import torch
from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    LogitsProcessorOutput,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3Model
from sglang.srt.runtime_context import get_schedule
from sglang.srt.utils import add_prefix

from sglang_omni.models.moss_tts.payload_types import moss_tts_special_token_defaults
from sglang_omni.models.moss_tts.sampler import (
    DelayGraphBatch,
    DelaySamplingOutput,
    MossTTSDelayAudioGraphSampler,
    matches_graph_profile,
)
from sglang_omni.models.moss_tts.sampling_cuda_graph import (
    MossTTSDelaySamplingCudaGraphRunner,
)
from sglang_omni.platforms import current_platform

logger = logging.getLogger(__name__)


class ChannelLogitsList(list):
    """Per-channel logits; ``fused_audio`` carries the [B, n_vq, vocab] fp32
    tensor the audio entries are views of, so consumers can skip re-stacking."""

    fused_audio: torch.Tensor | None = None


def as_qwen3_config(config: Any) -> Any:
    from transformers import Qwen3Config

    if isinstance(config, Qwen3Config):
        return config
    else:
        pass
    if isinstance(config, dict):
        return Qwen3Config(**config)
    else:
        pass
    if hasattr(config, "to_dict"):
        return Qwen3Config(**config.to_dict())
    else:
        pass
    return config


class MossTTSDelaySGLangModel(torch.nn.Module):
    """MOSS-TTS Delay AR backbone with one text channel and N RVQ channels."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]

    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = self.normalize_config(config)
        self.quant_config = quant_config
        self.hidden_size = int(self.config.hidden_size)
        self.delay_graph_sampler = MossTTSDelayAudioGraphSampler(self.config)
        self.sampling_graph_runner: MossTTSDelaySamplingCudaGraphRunner | None = None

        self.embedding_list = torch.nn.ModuleList()
        if self.pp_group.is_first_rank or (
            bool(getattr(self.config, "tie_word_embeddings", False))
            and self.pp_group.is_last_rank
        ):
            for idx in range(self.config.channels):
                self.embedding_list.append(
                    VocabParallelEmbedding(
                        int(self.config.vocab_size_list[idx]),
                        self.hidden_size,
                        quant_config=quant_config,
                        prefix=add_prefix(f"embedding_list.{idx}", prefix),
                    )
                )
        else:
            for _ in range(self.config.channels):
                self.embedding_list.append(PPMissingLayer())

        self.model = Qwen3Model(
            config=self.config.language_config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        self.lm_heads = torch.nn.ModuleList()
        if self.pp_group.is_last_rank:
            for idx in range(self.config.channels):
                self.lm_heads.append(
                    ParallelLMHead(
                        num_embeddings=int(self.config.vocab_size_list[idx]),
                        embedding_dim=self.hidden_size,
                        prefix=add_prefix(f"lm_heads.{idx}", prefix),
                    )
                )
        else:
            for _ in range(self.config.channels):
                self.lm_heads.append(PPMissingLayer())

        self.logits_processors = torch.nn.ModuleList(
            [
                self.make_logits_processor(self.config, idx)
                for idx in range(self.config.channels)
            ]
        )
        self.pad_token_per_channel = self.compute_pad_token_per_channel()
        self.stacked_audio_head_weight: torch.Tensor | None = None
        self.audio_head_padded_vocab = 0
        self.audio_head_expected_ptrs: list[int] = []
        self.fused_audio_heads_enabled: bool | None = None
        self.register_buffer(
            "_text_control_token_ids",
            torch.tensor(
                [
                    int(self.config.audio_assistant_gen_slot_token_id),
                    int(self.config.audio_assistant_delay_slot_token_id),
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )

        weight = self.first_embedding_weight()
        self.decode_input_embedding = torch.nn.Embedding(
            get_schedule().max_running_requests,
            self.hidden_size,
            device=weight.device,
            dtype=weight.dtype,
        )
        self.decode_input_embedding.weight.requires_grad_(False)

    @staticmethod
    def normalize_config(config: Any) -> Any:
        language_config = as_qwen3_config(getattr(config, "language_config", None))
        config.language_config = language_config
        config.hidden_size = int(
            getattr(config, "hidden_size", language_config.hidden_size)
        )
        config.vocab_size = int(
            getattr(config, "vocab_size", language_config.vocab_size)
        )
        config.n_vq = int(getattr(config, "n_vq", 32))
        config.channels = int(getattr(config, "channels", config.n_vq + 1))
        audio_vocab_size = int(getattr(config, "audio_vocab_size", 1024))
        if not getattr(config, "vocab_size_list", None):
            config.vocab_size_list = [config.vocab_size] + [audio_vocab_size + 1] * (
                config.channels - 1
            )
        else:
            pass
        if not getattr(config, "pad_token", None):
            text_pad = int(getattr(config, "pad_token_id", 0) or 0)
            audio_pad = int(getattr(config, "audio_pad_code", audio_vocab_size))
            config.pad_token = [text_pad] + [audio_pad] * (config.channels - 1)
        else:
            pass
        for attr, default in moss_tts_special_token_defaults(audio_vocab_size):
            if getattr(config, attr, None) is None:
                setattr(config, attr, default)
            else:
                pass
        config.language_config.channels = config.channels
        config.language_config.vocab_size_list = list(config.vocab_size_list)
        config.language_config.pad_token = list(config.pad_token)
        return config

    def first_embedding_weight(self) -> torch.Tensor:
        for layer in self.embedding_list:
            weight = getattr(layer, "weight", None)
            if isinstance(weight, torch.Tensor):
                return weight
            else:
                pass
        return torch.empty((), dtype=torch.float32)

    @property
    def start_layer(self) -> int:
        return self.model.start_layer

    @property
    def end_layer(self) -> int:
        return self.model.end_layer

    @property
    def device(self) -> torch.device:
        return self.first_embedding_weight().device

    @property
    def dtype(self) -> torch.dtype:
        return self.first_embedding_weight().dtype

    def compute_pad_token_per_channel(self) -> list[int]:
        pad = getattr(self.config, "pad_token", None)
        if isinstance(pad, (list, tuple)) and pad:
            pad_ids = [int(value) if value is not None else 0 for value in pad]
            if len(pad_ids) < self.config.channels:
                pad_ids.extend([pad_ids[-1]] * (self.config.channels - len(pad_ids)))
            else:
                pass
            return pad_ids[: self.config.channels]
        else:
            pass
        return [int(getattr(self.config, "pad_token_id", 0) or 0)] + [
            int(getattr(self.config, "audio_pad_code", 0) or 0)
        ] * (self.config.channels - 1)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.prepare_multi_modal_inputs(input_ids)

    def prepare_multi_modal_inputs(self, input_ids: torch.LongTensor) -> torch.Tensor:
        if input_ids.dim() == 1:
            channels = int(self.config.channels)
            total_tokens = int(input_ids.shape[0])
            if total_tokens % channels == 0:
                input_ids_2d = input_ids.view(total_tokens // channels, channels)
            else:
                input_ids_2d = torch.empty(
                    (total_tokens, channels),
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                for idx, pad_id in enumerate(self.pad_token_per_channel):
                    input_ids_2d[:, idx].fill_(int(pad_id))
                input_ids_2d[:, 0] = input_ids
        elif input_ids.dim() == 2:
            input_ids_2d = input_ids
        else:
            raise ValueError(
                "MOSS-TTS input_ids must be rank-1 flattened rows or rank-2 "
                f"multi-channel rows, got shape {tuple(input_ids.shape)}"
            )

        if int(input_ids_2d.shape[-1]) != int(self.config.channels):
            raise ValueError(
                f"MOSS-TTS expected {self.config.channels} channels, "
                f"got {input_ids_2d.shape[-1]}"
            )
        else:
            pass

        weight = self.first_embedding_weight()
        embeds = torch.zeros(
            input_ids_2d.shape[0],
            self.hidden_size,
            device=input_ids_2d.device,
            dtype=weight.dtype,
        )
        for idx, embed_layer in enumerate(self.embedding_list):
            embeds = embeds + embed_layer(input_ids_2d[:, idx])
        return embeds

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        omni_prefill_rids: list[str] | None = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        input_embeds_are_projected: bool = False,
    ) -> LogitsProcessorOutput:
        del omni_prefill_rids
        del input_embeds_are_projected
        if input_embeds is None:
            forward_mode = getattr(forward_batch, "forward_mode", None)
            is_decode = (
                forward_mode is not None
                and hasattr(forward_mode, "is_decode")
                and bool(forward_mode.is_decode())
            )
            if is_decode:
                input_embeds = self.decode_input_embedding(input_ids)
            elif self.pp_group.is_first_rank:
                input_embeds = self.prepare_multi_modal_inputs(input_ids)
            else:
                input_embeds = None
        else:
            pass

        hidden_states = self.model(
            input_ids=None,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if not self.pp_group.is_last_rank:
            return hidden_states
        else:
            pass

        sample_hidden_states = self.select_sample_hidden_states(
            hidden_states,
            forward_batch,
        )
        # The MOSS runner samples 33 channels from hidden states after the
        # graph-captured backbone returns. Keeping logits outside model.forward
        # avoids SGLang graph replay dropping customized_info and avoids reusing
        # the text-vocab graph logits buffer for small audio heads.
        dummy_logits = sample_hidden_states.new_empty(
            (sample_hidden_states.shape[0], 1)
        )
        return LogitsProcessorOutput(
            next_token_logits=dummy_logits,
            hidden_states=sample_hidden_states,
        )

    @staticmethod
    def make_logits_processor(config: Any, channel: int) -> LogitsProcessor:
        """Per-channel LogitsProcessor sized to that channel's own vocab.

        sglang's ``_get_logits`` slices the head output to ``config.vocab_size``
        to strip the ParallelLMHead vocab padding (1025 -> 1088 for the audio
        heads). Sharing the text config (vocab 152k) for the audio heads leaves
        the padding columns 1025..1087 in the audio logits; with the padding
        weights ~0 they get sampled at the audio temperature as out-of-range
        codes, which corrupt frames and drive the high-WER failure tail.

        Per-channel sizing referenced from
        https://github.com/sgl-project/sglang-omni/pull/608.
        """
        channel_config = copy(config)
        channel_config.vocab_size = int(config.vocab_size_list[channel])
        return LogitsProcessor(channel_config)

    def compute_channel_outputs(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> list[LogitsProcessorOutput]:
        logits_metadata = LogitsMetadata.from_forward_batch(forward_batch)
        logits_metadata.next_token_logits_buffer = None
        logits_metadata.forward_mode = ForwardMode.DECODE
        outputs = [
            self.logits_processors[0](
                None,
                hidden_states=hidden_states,
                lm_head=self.lm_heads[0],
                logits_metadata=logits_metadata,
            )
        ]
        outputs.extend(
            processor(
                None,
                hidden_states=hidden_states,
                lm_head=self.lm_heads[idx],
                logits_metadata=logits_metadata,
            )
            for idx, processor in enumerate(self.logits_processors[1:], start=1)
        )
        return outputs

    def audio_heads_use_plain_lm_head(self) -> bool:
        # Note (Jiaxin Deng): mirror of the pinned _compute_lm_head branch
        # order; the fused GEMM reproduces only its final plain-matmul arm.
        try:
            from sglang.srt.layers.logits_processor import (
                should_apply_lm_head_quant_method,
            )
            from sglang.srt.utils.common import use_intel_amx_backend
        except ImportError:
            return False
        for head in self.lm_heads[1:]:
            if hasattr(head, "set_lora") and hasattr(head, "apply_lora"):
                return False
            else:
                pass
            if should_apply_lm_head_quant_method(
                head, getattr(head, "quant_method", None)
            ):
                return False
            else:
                pass
            if use_intel_amx_backend(head):
                return False
            else:
                pass
        for processor in self.logits_processors[1:]:
            if getattr(processor, "use_fp32_lm_head", False):
                return False
            else:
                pass
            if getattr(processor, "rl_on_policy_target", None) is not None:
                return False
            else:
                pass
        return True

    def fused_audio_heads_requested(self) -> bool:
        return (
            os.environ.get("MOSS_DELAY_FUSED_AUDIO_HEADS", "1") != "0"
            and self.pp_group.is_last_rank
        )

    def fused_audio_heads_eligible(self, weights: list[Any]) -> bool:
        # Note (Jiaxin Deng): the fused path bypasses LogitsProcessor, so it is
        # gated to the plain configuration it reproduces: TP1, unquantized
        # same-shape ParallelLMHead weights, one audio vocab, no softcapping.
        first = weights[0] if weights else None
        return bool(
            first is not None
            and first.ndim == 2
            and first.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and get_tensor_model_parallel_world_size() == 1
            and getattr(self.config, "final_logit_softcapping", None) in (None, 0)
            and all(
                type(head).__name__ == "ParallelLMHead" for head in self.lm_heads[1:]
            )
            and all(
                w is not None and w.shape == first.shape and w.dtype == first.dtype
                for w in weights
            )
            and len({int(v) for v in self.config.vocab_size_list[1:]}) == 1
            and self.audio_heads_use_plain_lm_head()
        )

    def record_stacked_audio_heads(self, stacked: torch.Tensor, rows: int) -> None:
        self.stacked_audio_head_weight = stacked
        self.audio_head_padded_vocab = rows
        self.audio_head_expected_ptrs = [
            stacked[index * rows : (index + 1) * rows].data_ptr()
            for index in range(len(self.lm_heads) - 1)
        ]

    def ensure_stacked_audio_heads(self) -> bool:
        if self.stacked_audio_head_weight is not None:
            return True
        else:
            pass
        if self.fused_audio_heads_enabled is False:
            return False
        else:
            pass
        weights = [getattr(head, "weight", None) for head in self.lm_heads[1:]]
        enabled = self.fused_audio_heads_requested() and (
            self.fused_audio_heads_eligible(weights)
        )
        self.fused_audio_heads_enabled = enabled
        if not enabled:
            logger.info(
                "MOSS-TTS fused audio heads disabled (unsupported configuration)"
            )
            return False
        else:
            pass
        # Note (Jiaxin Deng): re-point each head's weight at a slice of one
        # stacked buffer so the fused GEMM adds no steady-state memory; the
        # transient duplicate lives only until the originals are released.
        stacked = torch.cat([w.data for w in weights], dim=0).contiguous()
        rows = int(weights[0].shape[0])
        for index, head in enumerate(self.lm_heads[1:]):
            head.weight.data = stacked[index * rows : (index + 1) * rows]
        self.record_stacked_audio_heads(stacked, rows)
        logger.info(
            "MOSS-TTS fused audio heads enabled (stacked %s)", tuple(stacked.shape)
        )
        return True

    @staticmethod
    def stacked_view_over_heads(weights: list[torch.Tensor]) -> torch.Tensor | None:
        """Return one 2D view spanning the heads, or None if they are not one block."""

        first = weights[0]
        rows, hidden = int(first.shape[0]), int(first.shape[1])
        step = rows * hidden * first.element_size()
        for index, weight in enumerate(weights):
            if (
                not weight.is_contiguous()
                or weight.data_ptr() != first.data_ptr() + index * step
            ):
                return None
            else:
                pass
        stacked = first.new_empty(0)
        try:
            stacked.set_(
                first.untyped_storage(),
                first.storage_offset(),
                (len(weights) * rows, hidden),
            )
        except RuntimeError:
            return None
        return stacked

    def on_weight_share_attached(self) -> None:
        """Re-derive the fused audio-head view from the leader's storage.

        A follower aliases every head onto the leader's storage after load, so
        any local stack is stale. The leader stacks before it exports, which
        leaves the shared audio heads one contiguous block: viewing them keeps
        both replicas on the same GEMM, which the same-GPU weight-share
        byte-identity contract requires.
        """

        self.stacked_audio_head_weight = None
        self.audio_head_expected_ptrs = []
        self.fused_audio_heads_enabled = None
        weights = [getattr(head, "weight", None) for head in self.lm_heads[1:]]
        if not self.fused_audio_heads_requested() or not (
            self.fused_audio_heads_eligible(weights)
        ):
            self.fused_audio_heads_enabled = False
            logger.info(
                "MOSS-TTS fused audio heads disabled (unsupported configuration)"
            )
            return
        else:
            pass
        stacked = self.stacked_view_over_heads(weights)
        if stacked is None:
            # Note (Jiaxin Deng): failing closed keeps the contract, since a
            # follower on the per-head path diverges from the leader's GEMM.
            raise RuntimeError(
                "MOSS-TTS fused audio heads: the shared audio-head storages are "
                "not one contiguous block, so this replica cannot reproduce the "
                "leader's fused GEMM; rerun every replica with "
                "MOSS_DELAY_FUSED_AUDIO_HEADS=0"
            )
        else:
            pass
        self.fused_audio_heads_enabled = True
        self.record_stacked_audio_heads(stacked, int(weights[0].shape[0]))
        logger.info(
            "MOSS-TTS fused audio heads adopted from shared storage (stacked %s)",
            tuple(stacked.shape),
        )

    def fused_audio_heads_ready(self) -> bool:
        # Note (Jiaxin Deng): stacking happens once at load time, before the
        # weight-share IPC export and before KV profiling; the request path
        # only observes the result.
        if self.stacked_audio_head_weight is None:
            return False
        else:
            pass
        # Note (Jiaxin Deng): heads can be replaced independently
        # (set_embed_and_head, assign-loads), so every audio slice must still
        # alias its stacked-buffer offset before the fused GEMM may run.
        for index, head in enumerate(self.lm_heads[1:]):
            weight = getattr(head, "weight", None)
            if (
                weight is None
                or weight.data_ptr() != self.audio_head_expected_ptrs[index]
            ):
                logger.warning(
                    "MOSS-TTS fused audio heads disabled: head %d weight was replaced",
                    index + 1,
                )
                self.stacked_audio_head_weight = None
                self.fused_audio_heads_enabled = False
                return False
            else:
                pass
        return True

    def compute_fused_audio_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        n_audio = int(self.config.channels) - 1
        audio_vocab = int(self.config.vocab_size_list[1])
        flat = torch.nn.functional.linear(hidden_states, self.stacked_audio_head_weight)
        return flat.view(hidden_states.shape[0], n_audio, self.audio_head_padded_vocab)[
            ..., :audio_vocab
        ].to(torch.float32)

    def compute_channel_logits(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        *,
        is_audio: bool = False,
    ) -> list[torch.Tensor]:
        if self.fused_audio_heads_ready():
            logits_metadata = LogitsMetadata.from_forward_batch(forward_batch)
            logits_metadata.next_token_logits_buffer = None
            logits_metadata.forward_mode = ForwardMode.DECODE
            text = self.logits_processors[0](
                None,
                hidden_states=hidden_states,
                lm_head=self.lm_heads[0],
                logits_metadata=logits_metadata,
            ).next_token_logits
            fused = self.compute_fused_audio_logits(hidden_states)
            logits = ChannelLogitsList([text, *fused.unbind(dim=1)])
            logits.fused_audio = fused
        else:
            logits = ChannelLogitsList(
                output.next_token_logits
                for output in self.compute_channel_outputs(hidden_states, forward_batch)
            )
        if is_audio:
            token_ids = self._text_control_token_ids.to(
                device=logits[0].device
            )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            logits[0] = logits[0].index_select(-1, token_ids)
        else:
            pass
        return logits

    @property
    def text_control_token_ids(self) -> torch.Tensor:
        return (
            self._text_control_token_ids
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken

    @staticmethod
    def is_sampling_cuda_graph_compatible(data: Any) -> bool:
        """Return whether one request uses the captured sampling profile."""

        return matches_graph_profile(data)

    def sampling_graph_support_reason(self) -> str | None:
        if not (current_platform.is_cuda() or current_platform.is_musa()):
            return f"{self.device.type.upper()} graph is unavailable"
        else:
            pass
        if int(self.config.channels) != int(self.config.n_vq) + 1:
            return (
                "sampling CUDA graph requires channels == n_vq + 1 "
                f"(got n_vq={self.config.n_vq}, channels={self.config.channels})"
            )
        else:
            pass
        audio_vocabs = {int(size) for size in self.config.vocab_size_list[1:]}
        if len(audio_vocabs) != 1:
            return "all audio codebooks must use the same vocabulary size"
        else:
            pass
        if int(self.config.audio_pad_code) < 0:
            return "audio_pad_code must be non-negative"
        else:
            pass
        if not bool(self.pp_group.is_first_rank and self.pp_group.is_last_rank):
            return "pipeline parallelism is not supported"
        else:
            pass
        return None

    @torch.no_grad()
    def sample_delay_fixed_shape(
        self,
        control_logits: torch.Tensor,
        audio_logits: torch.Tensor,
        batch: DelayGraphBatch,
        *,
        audio_sample_output: torch.Tensor | None = None,
    ) -> DelaySamplingOutput:
        """Run one fixed-shape all-audio sampling/FSM step."""

        return self.delay_graph_sampler(
            control_logits,
            audio_logits,
            batch,
            audio_sample_output=audio_sample_output,
        )

    @torch.no_grad()
    def init_sampling_graphs(
        self,
        batch_sizes: list[int],
        *,
        disable_padding: bool = False,
    ) -> None:
        """Capture sampling/FSM graphs in the backbone batch envelope."""

        buckets = tuple(sorted({int(batch_size) for batch_size in batch_sizes}))
        if not buckets:
            return
        else:
            pass
        if any(batch_size < 1 for batch_size in buckets):
            raise ValueError("MOSS-TTS Delay sampling CUDA graph bs must be >= 1")
        else:
            pass
        reason = self.sampling_graph_support_reason()
        if reason is not None:
            logger.warning(
                "MOSS-TTS Delay sampling CUDA graph disabled: %s. "
                "Falling back to eager sampling.",
                reason,
            )
            return
        else:
            pass
        runner = MossTTSDelaySamplingCudaGraphRunner.capture(
            model=self,
            capture_bs=buckets,
            disable_padding=disable_padding,
        )
        if not runner.graphs:
            logger.warning(
                "MOSS-TTS Delay sampling CUDA graph captured no buckets; "
                "using eager sampling"
            )
            self.sampling_graph_runner = None
            return
        else:
            pass
        self.sampling_graph_runner = runner

    def sampling_graph_available(
        self,
        batch_size: int,
    ) -> bool:
        runner = self.sampling_graph_runner
        return runner is not None and runner.can_replay(batch_size)

    @torch.no_grad()
    def sample_delay_graphed(
        self,
        control_logits: torch.Tensor,
        audio_logits: torch.Tensor,
        batch: DelayGraphBatch,
    ) -> DelaySamplingOutput:
        """Replay one fixed-shape sampling/FSM CUDA graph."""

        runner = self.sampling_graph_runner
        if runner is None:
            raise RuntimeError("MOSS-TTS Delay sampling CUDA graph is not configured")
        else:
            pass
        return runner.replay(
            control_logits,
            audio_logits,
            batch,
        )

    @staticmethod
    def select_sample_hidden_states(
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        forward_mode = getattr(forward_batch, "forward_mode", None)
        is_extend = (
            forward_mode is not None
            and hasattr(forward_mode, "is_extend")
            and bool(forward_mode.is_extend())
        )
        if not is_extend:
            return hidden_states
        else:
            pass
        extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
        if extend_seq_lens is None:
            return hidden_states[-1:].contiguous()
        else:
            pass
        last_index = (
            torch.cumsum(
                extend_seq_lens.to(device=hidden_states.device, dtype=torch.long), dim=0
            )
            - 1
        )
        return hidden_states[last_index]

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())

        for original_name, loaded_weight in weights:
            name = original_name
            if name.startswith("language_model."):
                name = "model." + name[len("language_model.") :]
            else:
                pass

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue
            else:
                pass
            if (
                "rotary_emb.inv_freq" in name
                or "rotary_emb.cos_cached" in name
                or "rotary_emb.sin_cached" in name
                or "projector" in name
            ):
                continue
            else:
                pass

            if name.startswith("emb_ext.") and name.endswith(".weight"):
                mapped = self.map_audio_embedding_name(name)
                if mapped is not None and mapped in params_dict:
                    self.load_param(params_dict[mapped], loaded_weight)
                else:
                    pass
                continue
            else:
                pass

            if name == "model.embed_tokens.weight":
                mapped = "embedding_list.0.weight"
                if mapped in params_dict:
                    self.load_param(params_dict[mapped], loaded_weight)
                else:
                    pass
            else:
                pass

            if name.startswith("lm_heads.") and name.endswith(".weight"):
                if name in params_dict:
                    self.load_param(params_dict[name], loaded_weight)
                else:
                    pass
                continue
            else:
                pass

            mapped_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                else:
                    pass
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                    mapped_stacked = True
                    break
                else:
                    pass
                param = params_dict.get(mapped_name)
                if param is None:
                    break
                else:
                    pass
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                mapped_stacked = True
                break
            if mapped_stacked:
                continue
            else:
                pass

            if name.endswith(".bias") and name not in params_dict:
                continue
            else:
                pass
            param = params_dict.get(name)
            if param is not None:
                self.load_param(param, loaded_weight)
            else:
                logger.warning(f"MOSS-TTS parameter {original_name} not found")

        # Note (Jiaxin Deng): stack here, not on the first request: the
        # weight-share IPC export and KV profiling both run after load, so
        # they must see the final head storage layout and the transient copy.
        self.ensure_stacked_audio_heads()

    @staticmethod
    def map_audio_embedding_name(name: str) -> str | None:
        try:
            idx = int(name.split(".")[1]) + 1
        except (IndexError, ValueError):
            return None
        return f"embedding_list.{idx}.weight"

    @staticmethod
    def load_param(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)

    def get_embed_and_head(self) -> tuple[list[Any], list[Any]]:
        embed_weights = [
            getattr(layer, "weight", None) for layer in self.embedding_list
        ]
        head_weights = [getattr(head, "weight", None) for head in self.lm_heads]
        return embed_weights, head_weights

    def set_embed_and_head(self, embed_list: list[Any], head_list: list[Any]) -> None:
        if embed_list is not None:
            for idx, embed in enumerate(embed_list[: len(self.embedding_list)]):
                if embed is not None and hasattr(self.embedding_list[idx], "weight"):
                    self.embedding_list[idx].weight = embed
                else:
                    pass
        else:
            pass
        if head_list is not None:
            for idx, head in enumerate(head_list[: len(self.lm_heads)]):
                if head is not None and hasattr(self.lm_heads[idx], "weight"):
                    self.lm_heads[idx].weight = head
                else:
                    pass
        else:
            pass
        if current_platform.is_cuda() or current_platform.is_musa():
            torch.cuda.empty_cache()
        else:
            pass

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)


EntryClass = MossTTSDelaySGLangModel
