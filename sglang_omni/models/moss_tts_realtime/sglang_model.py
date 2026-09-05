# SPDX-License-Identifier: Apache-2.0
"""SGLang-native global/local model for MOSS-TTS-Realtime."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from typing import Any, Optional

import torch
from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3Model
from sglang.srt.utils import add_prefix

from sglang_omni.models.moss_tts_realtime.local_transformer import (
    MossTTSRealtimeLocalTransformerForCausalLM,
)
from sglang_omni.models.moss_tts_realtime.state_pool import (
    MossTTSRealtimeDecodeStatePool,
)

logger = logging.getLogger(__name__)

_LANGUAGE_LAYER_RE = re.compile(r"^language_model\.layers\.(\d+)\.")
_QWEN_LAYER_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)
_STACKED_PARAMS_MAPPING = (
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
)


def _normalize_config(config: Any) -> Any:
    """Populate the multi-channel fields consumed by the SGLang model."""

    language_config = config.language_config
    rvq = int(config.rvq)
    config.hidden_size = int(language_config.hidden_size)
    config.vocab_size = int(language_config.vocab_size)
    config.n_vq = rvq
    config.channels = rvq + 1
    config.vocab_size_list = [config.vocab_size] + [int(config.audio_vocab_size)] * rvq
    config.pad_token = [int(config.text_pad)] + [int(config.audio_pad_token)] * rvq
    config.text_pad_token_id = int(config.text_pad)

    language_config.channels = config.channels
    language_config.vocab_size_list = list(config.vocab_size_list)
    language_config.pad_token = list(config.pad_token)
    return config


def expected_moss_tts_realtime_checkpoint_keys(config: Any) -> frozenset[str]:
    """Return the checkpoint keys consumed by this model implementation."""

    config = _normalize_config(config)
    keys: set[str] = {
        *(f"embed_tokens.{index}.weight" for index in range(config.channels)),
        "language_model.embed_tokens.weight",
        "language_model.norm.weight",
        "local_transformer.model.norm.weight",
    }
    for layer_index in range(int(config.language_config.num_hidden_layers)):
        keys.update(
            f"language_model.layers.{layer_index}.{suffix}"
            for suffix in _QWEN_LAYER_SUFFIXES
        )
    local_config = config.local_config
    for layer_index in range(int(local_config.num_hidden_layers)):
        keys.update(
            f"local_transformer.model.layers.{layer_index}.{suffix}"
            for suffix in _QWEN_LAYER_SUFFIXES
        )
    keys.update(
        f"local_transformer.model.embed_tokens.{index}.weight"
        for index in range(int(local_config.rvq) - 1)
    )
    keys.update(
        f"local_transformer.local_lm_heads.{index}.weight"
        for index in range(int(local_config.rvq))
    )
    return frozenset(keys)


class MossTTSRealtimeSGLangModel(torch.nn.Module):
    """17-channel embeddings + Qwen3 backbone + local 16-codebook decoder."""

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
        self.config = _normalize_config(config)
        self.quant_config = quant_config
        self.hidden_size = int(self.config.hidden_size)

        self.embed_tokens = torch.nn.ModuleList()
        if self.pp_group.is_first_rank:
            for channel, vocab_size in enumerate(self.config.vocab_size_list):
                self.embed_tokens.append(
                    VocabParallelEmbedding(
                        int(vocab_size),
                        self.hidden_size,
                        quant_config=quant_config,
                        prefix=add_prefix(f"embed_tokens.{channel}", prefix),
                    )
                )
        else:
            for _ in range(self.config.channels):
                self.embed_tokens.append(PPMissingLayer())

        self.language_model = Qwen3Model(
            config=self.config.language_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        if self.pp_group.is_last_rank:
            self.local_transformer = MossTTSRealtimeLocalTransformerForCausalLM(
                self.config.local_config
            )
        else:
            self.local_transformer = PPMissingLayer()

        # Match MOSS-TTS Local's CUDA-graph-safe decode input path. The model
        # runner refreshes these stable rows before every decode and rewrites
        # decode input_ids to row indices, so SGLang's generic backbone graph
        # does not need to carry a dynamically allocated input_embeds tensor.
        max_batch_size = None
        try:
            from sglang.srt.server_args import get_global_server_args

            max_batch_size = get_global_server_args().max_running_requests
        except Exception:
            max_batch_size = None
        weight = self._first_embedding_weight()
        self._decode_input_embedding = torch.nn.Embedding(
            int(max_batch_size or 1),
            self.hidden_size,
            device=weight.device,
            dtype=weight.dtype,
        )
        self._decode_input_embedding.weight.requires_grad_(False)

        self._state_pool: MossTTSRealtimeDecodeStatePool | None = None
        self._local_cuda_graph_runner: Any | None = None

    def init_frame_decode_graphs(self, batch_sizes: list[int]) -> None:
        """Capture frame-local graphs using the generic SGLang graph buckets."""

        if not self.pp_group.is_last_rank:
            return
        from sglang_omni.models.moss_tts_realtime.local_cuda_graph import (
            MossTTSRealtimeLocalCudaGraphRunner,
        )

        normalized = sorted({int(batch_size) for batch_size in batch_sizes})
        if not normalized:
            return
        runner = MossTTSRealtimeLocalCudaGraphRunner(
            self.local_transformer,
            batch_sizes=normalized,
            max_batch_size=max(normalized),
        )
        runner.warmup()
        self._local_cuda_graph_runner = runner

    def local_cuda_graph_resource_snapshot(self) -> dict[str, int]:
        runner = self._local_cuda_graph_runner
        if runner is None:
            return {
                "local_cuda_graph_captured_batch_count": 0,
                "local_cuda_graph_max_batch_size": 0,
                "local_cuda_graph_replay_total": 0,
                "local_cuda_graph_fallback_total": 0,
                "local_cuda_graph_failure_total": 0,
                "local_cuda_graph_disabled": 0,
            }
        return runner.resource_snapshot()

    def init_decode_state_pool(
        self,
        *,
        max_running_requests: int,
        max_history_frames: int = 1000,
    ) -> MossTTSRealtimeDecodeStatePool:
        """Create the fixed eager decode state before the first request."""

        if not self.pp_group.is_last_rank:
            raise RuntimeError(
                "MOSS-TTS-Realtime decode state is only available on the last PP rank"
            )
        existing = self._state_pool
        if existing is not None:
            if existing.max_running_requests != int(max_running_requests):
                raise RuntimeError(
                    "MOSS-TTS-Realtime state-pool capacity cannot change after "
                    "initialization"
                )
            if existing.max_history_frames != int(max_history_frames):
                raise RuntimeError(
                    "MOSS-TTS-Realtime history capacity cannot change after "
                    "initialization"
                )
            return existing
        self._state_pool = MossTTSRealtimeDecodeStatePool(
            self,
            max_running_requests=max_running_requests,
            max_history_frames=max_history_frames,
        )
        return self._state_pool

    @property
    def state_pool(self) -> MossTTSRealtimeDecodeStatePool:
        pool = self._state_pool
        if pool is None:
            raise RuntimeError("MOSS-TTS-Realtime decode state pool is not initialized")
        return pool

    def reset_request(self, rid: str, turn_state: Any | None = None) -> int | None:
        """Release one request's model-side state, if the pool exists."""

        if self._state_pool is None:
            return None
        return self._state_pool.release_row(rid, turn_state)

    def row_for(self, rid: str) -> int | None:
        if self._state_pool is None:
            return None
        return self._state_pool.row_for(rid)

    def _first_embedding_weight(self) -> torch.Tensor:
        for layer in self.embed_tokens:
            weight = getattr(layer, "weight", None)
            if isinstance(weight, torch.Tensor):
                return weight
        return torch.empty((), dtype=torch.float32)

    @property
    def start_layer(self) -> int:
        return self.language_model.start_layer

    @property
    def end_layer(self) -> int:
        return self.language_model.end_layer

    @property
    def device(self) -> torch.device:
        return self._first_embedding_weight().device

    @property
    def dtype(self) -> torch.dtype:
        return self._first_embedding_weight().dtype

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._prepare_multi_modal_inputs(input_ids)

    def _prepare_multi_modal_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Sum one text embedding and all 16 ordered audio embeddings."""

        if input_ids.ndim == 1:
            rows = torch.full(
                (input_ids.shape[0], self.config.channels),
                int(self.config.audio_pad_token),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            rows[:, 0] = input_ids
        elif input_ids.ndim == 2:
            rows = input_ids
        else:
            raise ValueError(
                "MOSS-TTS-Realtime input ids must be rank 1 or rank 2, got "
                f"shape {tuple(input_ids.shape)}"
            )
        if int(rows.shape[1]) != int(self.config.channels):
            raise ValueError(
                f"MOSS-TTS-Realtime expected {self.config.channels} columns, "
                f"got {rows.shape[1]}"
            )
        if (
            rows.dtype == torch.bool
            or torch.is_floating_point(rows)
            or torch.is_complex(rows)
        ):
            raise TypeError("MOSS-TTS-Realtime input rows must be an integer tensor")
        if torch.any(rows[:, 0] < 0) or torch.any(
            rows[:, 0] >= int(self.config.vocab_size)
        ):
            raise ValueError("MOSS-TTS-Realtime text token id is out of range")
        if torch.any(rows[:, 1:] < 0) or torch.any(
            rows[:, 1:] >= int(self.config.audio_vocab_size)
        ):
            raise ValueError("MOSS-TTS-Realtime audio token id is out of range")

        embeddings = self.embed_tokens[0](rows[:, 0])
        for channel in range(1, self.config.channels):
            embeddings = embeddings + self.embed_tokens[channel](rows[:, channel])
        return embeddings

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        input_embeds_are_projected: bool = False,
    ) -> LogitsProcessorOutput:
        del input_embeds_are_projected
        if input_embeds is None:
            forward_mode = getattr(forward_batch, "forward_mode", None)
            is_decode = (
                forward_mode is not None
                and hasattr(forward_mode, "is_decode")
                and bool(forward_mode.is_decode())
            )
            if is_decode:
                input_embeds = self._decode_input_embedding(input_ids)
            elif self.pp_group.is_first_rank:
                input_embeds = self._prepare_multi_modal_inputs(input_ids)

        hidden_states = self.language_model(
            input_ids=None,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if not self.pp_group.is_last_rank:
            return hidden_states

        sampled_hidden = self._select_sample_hidden_states(hidden_states, forward_batch)
        dummy_logits = sampled_hidden.new_empty((sampled_hidden.shape[0], 1))
        return LogitsProcessorOutput(
            next_token_logits=dummy_logits,
            hidden_states=sampled_hidden,
        )

    @staticmethod
    def _select_sample_hidden_states(
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
        extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
        if extend_seq_lens is None:
            return hidden_states[-1:].contiguous()
        last_indices = (
            torch.cumsum(
                extend_seq_lens.to(device=hidden_states.device, dtype=torch.long),
                dim=0,
            )
            - 1
        )
        return hidden_states[last_indices]

    @torch.no_grad()
    def decode_local_frame(
        self,
        hidden_states: torch.Tensor,
        *,
        sample_audio: Callable[[torch.Tensor, int], torch.Tensor],
    ) -> torch.Tensor:
        if not self.pp_group.is_last_rank:
            raise RuntimeError(
                "local frame decode is only available on the last PP rank"
            )
        graph_runner = self._local_cuda_graph_runner
        if graph_runner is not None:
            batch_size = int(hidden_states.shape[0])
            if graph_runner.supports(batch_size):
                return self.local_transformer.decode_frame(
                    hidden_states,
                    sample_audio=sample_audio,
                    compute_logits=graph_runner.compute,
                )
            graph_runner.record_fallback()
        return self.local_transformer.decode_frame(
            hidden_states,
            sample_audio=sample_audio,
        )

    @torch.no_grad()
    def teacher_forced_local_logits(
        self,
        hidden_states: torch.Tensor,
        prefix_codes: torch.Tensor,
    ) -> torch.Tensor:
        if not self.pp_group.is_last_rank:
            raise RuntimeError("local logits are only available on the last PP rank")
        return self.local_transformer.teacher_forced_logits(
            hidden_states,
            prefix_codes,
        )

    def _skip_checkpoint_key_for_rank(self, name: str) -> bool:
        if name.startswith("embed_tokens."):
            return not self.pp_group.is_first_rank
        if name.startswith("local_transformer."):
            return not self.pp_group.is_last_rank
        if name == "language_model.embed_tokens.weight":
            return not self.pp_group.is_first_rank
        if name == "language_model.norm.weight":
            return not self.pp_group.is_last_rank
        match = _LANGUAGE_LAYER_RE.match(name)
        if match is None:
            return False
        layer_index = int(match.group(1))
        return not self.start_layer <= layer_index < self.end_layer

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        strict: bool = True,
    ) -> set[str]:
        """Load and account for every tensor in the pinned checkpoint."""

        expected = expected_moss_tts_realtime_checkpoint_keys(self.config)
        params = dict(self.named_parameters())
        seen: set[str] = set()

        for name, loaded_weight in weights:
            if name in seen:
                raise ValueError(f"duplicate MOSS-TTS-Realtime checkpoint key {name!r}")
            if name not in expected:
                raise ValueError(
                    f"unexpected MOSS-TTS-Realtime checkpoint key {name!r}"
                )
            seen.add(name)
            if self._skip_checkpoint_key_for_rank(name):
                continue

            if name.startswith("language_model."):
                if self._load_stacked_language_weight(name, loaded_weight, params):
                    continue
            param = params.get(name)
            if param is None:
                raise KeyError(
                    f"MOSS-TTS-Realtime parameter for checkpoint key {name!r} "
                    "was not found"
                )
            self._load_param(name, param, loaded_weight)

        if strict:
            missing = sorted(expected - seen)
            if missing:
                preview = ", ".join(missing[:8])
                suffix = "" if len(missing) <= 8 else f" ... ({len(missing)} total)"
                raise ValueError(
                    "missing MOSS-TTS-Realtime checkpoint keys: " + preview + suffix
                )
        return seen

    @staticmethod
    def _load_stacked_language_weight(
        name: str,
        loaded_weight: torch.Tensor,
        params: dict[str, torch.nn.Parameter],
    ) -> bool:
        for packed_name, shard_name, shard_id in _STACKED_PARAMS_MAPPING:
            needle = f".{shard_name}."
            if needle not in name:
                continue
            mapped_name = name.replace(needle, f".{packed_name}.")
            param = params.get(mapped_name)
            if param is None:
                raise KeyError(
                    f"packed MOSS-TTS-Realtime parameter {mapped_name!r} "
                    f"for {name!r} was not found"
                )
            loader = getattr(param, "weight_loader", default_weight_loader)
            try:
                loader(param, loaded_weight, shard_id)
            except Exception as exc:
                raise ValueError(
                    f"failed loading {name!r} into {mapped_name!r}: {exc}"
                ) from exc
            return True
        return False

    @staticmethod
    def _load_param(
        name: str,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        loader = getattr(param, "weight_loader", default_weight_loader)
        try:
            loader(param, loaded_weight)
        except Exception as exc:
            raise ValueError(f"failed loading {name!r}: {exc}") from exc

    def get_embed_and_head(self) -> tuple[list[Any], list[Any]]:
        embeddings = [getattr(layer, "weight", None) for layer in self.embed_tokens]
        if not self.pp_group.is_last_rank:
            return embeddings, []
        heads = [head.weight for head in self.local_transformer.local_lm_heads]
        return embeddings, heads

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.language_model.load_kv_cache_scales(quantization_param_path)


EntryClass = MossTTSRealtimeSGLangModel
