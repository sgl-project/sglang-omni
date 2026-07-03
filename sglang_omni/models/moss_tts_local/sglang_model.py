# SPDX-License-Identifier: Apache-2.0
"""SGLang-native MOSS-TTS Local (v1.5) model wrapper.

Architecture: a 36-layer Qwen3 global backbone consumes one summed embedding
per audio frame (text channel + 12 RVQ code channels); a 1-layer frame-local
transformer then decodes the next frame from the backbone's last hidden state
— a binary continue/stop decision followed by 12 sequentially sampled RVQ
codes, each fed back as the next local-position embedding.

The backbone runs under SGLang's scheduler (radix cache + CUDA graph); the
local transformer micro-loop runs batched in :meth:`decode_frame`, driven by
the model runner after each backbone step.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3Model
from sglang.srt.utils import add_prefix

from sglang_omni.models.moss_tts_local.local_transformer import (
    MossTTSLocalTransformer,
    sample_seeded_branchless,
)
from sglang_omni.models.moss_tts_local.payload_types import (
    moss_tts_local_special_token_defaults,
)
from sglang_omni.models.moss_tts_local.state_pool import MossTTSLocalDecodeStatePool

logger = logging.getLogger(__name__)


def _as_qwen3_config(config: Any) -> Any:
    from transformers import Qwen3Config

    if isinstance(config, Qwen3Config):
        return config
    if isinstance(config, dict):
        return Qwen3Config(**config)
    if hasattr(config, "to_dict"):
        return Qwen3Config(**config.to_dict())
    return config


class MossTTSLocalSGLangModel(torch.nn.Module):
    """MOSS-TTS Local AR model: Qwen3 backbone + 1-layer local transformer."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = self._normalize_config(config)
        self.quant_config = quant_config
        self.hidden_size = int(self.config.hidden_size)
        self.n_vq = int(self.config.n_vq)

        # Channel 0: text vocab; channels 1..n_vq: per-codebook tables with one
        # extra row for audio_pad_code so prompt-row embedding sums need no
        # masking — the pad row is zeroed after weight loading.
        self.embedding_list = torch.nn.ModuleList()
        if self.pp_group.is_first_rank:
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

        gpt2_cfg = self.config.gpt2_config
        self.local_transformer = MossTTSLocalTransformer(
            hidden_size=self.hidden_size,
            num_heads=int(self._cfg_get(gpt2_cfg, "n_head", 32)),
            inner_size=int(self._cfg_get(gpt2_cfg, "n_inner", 4 * self.hidden_size)),
            num_layers=int(getattr(self.config, "local_transformer_layers", 1)),
            max_positions=self.n_vq + 1,
            rope_base=float(self._cfg_get(gpt2_cfg, "rope_base", 1_000_000.0)),
            layer_norm_eps=float(self._cfg_get(gpt2_cfg, "layer_norm_epsilon", 1e-6)),
        )
        # Binary continue/stop head over the local position-0 hidden state:
        # index 0 -> audio_assistant_slot (emit a frame), 1 -> audio_end (stop).
        self.local_text_lm_head = torch.nn.Linear(self.hidden_size, 2, bias=False)

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

        # Row-indexed decode-state pool: next-step-critical per-request state
        # (next-frame feedback embedding, sampling params/seed, generation step)
        # lives in process-lifetime GPU buffers sized off the staging table
        # above. Allocated here, before any frame/backbone graph capture, so
        # its addresses are fixed for the process lifetime.
        self._state_pool = MossTTSLocalDecodeStatePool(self)
        self._compiled_frame_sampler: Callable[..., torch.Tensor] | None = None
        self._frame_compile_configured = False

    def acquire_row(self, rid: str) -> int:
        """Assign (or return the existing) decode-state pool row for ``rid``."""
        return self._state_pool.acquire_row(rid)

    def release_row(self, rid: str) -> None:
        """Return ``rid``'s pool row to the free list. No-op if unheld."""
        self._state_pool.release_row(rid)

    def reset_request(self, rid: str) -> None:
        """Release pool state for a finished or aborted request (idempotent)."""
        self._state_pool.release_row(rid)

    def row_for(self, rid: str) -> int | None:
        """Return ``rid``'s pool row, or ``None`` if it holds none."""
        return self._state_pool.row_for(rid)

    @staticmethod
    def _cfg_get(config: Any, name: str, default: Any) -> Any:
        if isinstance(config, dict):
            value = config.get(name, default)
        else:
            value = getattr(config, name, default)
        return default if value is None else value

    @staticmethod
    def _normalize_config(config: Any) -> Any:
        qwen3_config = getattr(config, "qwen3_config", None)
        if qwen3_config is None:
            qwen3_config = getattr(config, "language_config", None)
        language_config = _as_qwen3_config(qwen3_config)
        try:
            config.language_config = language_config
        except AttributeError:
            # MossTTSLocalConfig.language_config is a read-only property
            # mirroring qwen3_config; normalize through the backing field.
            config.qwen3_config = language_config
        config.hidden_size = int(
            getattr(config, "hidden_size", None) or language_config.hidden_size
        )
        config.vocab_size = int(
            getattr(config, "vocab_size", None) or language_config.vocab_size
        )
        config.n_vq = int(getattr(config, "n_vq", 12))
        config.channels = int(getattr(config, "channels", config.n_vq + 1))
        audio_vocab_size = int(getattr(config, "audio_vocab_size", 1024) or 1024)
        config.audio_vocab_size = audio_vocab_size
        if not getattr(config, "vocab_size_list", None):
            config.vocab_size_list = [config.vocab_size] + [audio_vocab_size + 1] * (
                config.channels - 1
            )
        for attr, default in moss_tts_local_special_token_defaults(audio_vocab_size):
            if getattr(config, attr, None) is None:
                setattr(config, attr, default)
        if not getattr(config, "pad_token", None):
            text_pad = int(getattr(config, "pad_token_id", 0) or 0)
            audio_pad = int(config.audio_pad_code)
            config.pad_token = [text_pad] + [audio_pad] * (config.channels - 1)
        config.language_config.channels = config.channels
        config.language_config.vocab_size_list = list(config.vocab_size_list)
        config.language_config.pad_token = list(config.pad_token)
        return config

    def _first_embedding_weight(self) -> torch.Tensor:
        for layer in self.embedding_list:
            weight = getattr(layer, "weight", None)
            if isinstance(weight, torch.Tensor):
                return weight
        return torch.empty((), dtype=torch.float32)

    @property
    def start_layer(self) -> int:
        return self.model.start_layer

    @property
    def end_layer(self) -> int:
        return self.model.end_layer

    @property
    def device(self) -> torch.device:
        return self._first_embedding_weight().device

    @property
    def dtype(self) -> torch.dtype:
        return self._first_embedding_weight().dtype

    def _audio_embedding_weight(self, channel: int) -> torch.Tensor:
        """Rows 0..audio_vocab_size-1 of codebook ``channel``'s table."""
        weight = self.embedding_list[channel + 1].weight
        return weight[: int(self.config.audio_vocab_size)]

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._prepare_multi_modal_inputs(input_ids)

    def _prepare_multi_modal_inputs(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Sum text + per-codebook embeddings for ``[T, channels]`` rows.

        Pad codes (``audio_pad_code``) hit the zeroed extra row of each audio
        table, matching the upstream zero-mask semantics without masking.
        """
        if input_ids.dim() == 1:
            channels = int(self.config.channels)
            total_tokens = int(input_ids.shape[0])
            if total_tokens % channels == 0:
                input_ids_2d = input_ids.view(total_tokens // channels, channels)
            else:
                # Profiling/warmup passes flat dummy ids of arbitrary length:
                # treat each id as a text token over pad audio rows.
                input_ids_2d = torch.empty(
                    (total_tokens, channels),
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                pad_token = self.config.pad_token
                for idx in range(channels):
                    input_ids_2d[:, idx].fill_(int(pad_token[idx]))
                input_ids_2d[:, 0] = input_ids
        elif input_ids.dim() == 2:
            input_ids_2d = input_ids
        else:
            raise ValueError(
                "MOSS-TTS Local input_ids must be rank-1 flattened rows or "
                f"rank-2 multi-channel rows, got shape {tuple(input_ids.shape)}"
            )

        if int(input_ids_2d.shape[-1]) != int(self.config.channels):
            raise ValueError(
                f"MOSS-TTS Local expected {self.config.channels} channels, "
                f"got {input_ids_2d.shape[-1]}"
            )

        weight = self._first_embedding_weight()
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
            else:
                input_embeds = None

        hidden_states = self.model(
            input_ids=None,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if not self.pp_group.is_last_rank:
            return hidden_states

        sample_hidden_states = self._select_sample_hidden_states(
            hidden_states,
            forward_batch,
        )
        # The local-transformer frame decode (binary stop head + 12 sequential
        # codebook samples) runs in the model runner after the graph-captured
        # backbone returns; emitting hidden states with dummy logits keeps the
        # backbone CUDA-graph replay free of model-specific outputs.
        dummy_logits = sample_hidden_states.new_empty(
            (sample_hidden_states.shape[0], 1)
        )
        return LogitsProcessorOutput(
            next_token_logits=dummy_logits,
            hidden_states=sample_hidden_states,
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
        last_index = (
            torch.cumsum(
                extend_seq_lens.to(device=hidden_states.device, dtype=torch.long), dim=0
            )
            - 1
        )
        return hidden_states[last_index]

    # ------------------------------------------------------------------
    # Frame decode: eager path (callback-driven) and CUDA-graphed path
    # ------------------------------------------------------------------

    _sample_seeded_branchless = staticmethod(sample_seeded_branchless)

    def _ensure_frame_compile_config(self) -> None:
        if self._frame_compile_configured:
            return
        from sglang.srt.model_executor.cuda_graph_runner import set_torch_compile_config

        set_torch_compile_config()
        self._frame_compile_configured = True

    def _ensure_frame_sampler_compile(self) -> None:
        if self._compiled_frame_sampler is None:
            compile_mode = os.environ.get(
                "SGLANG_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs"
            )
            self._ensure_frame_compile_config()
            self._compiled_frame_sampler = torch.compile(
                sample_seeded_branchless,
                mode=compile_mode,
            )
            self._sample_seeded_branchless = self._compiled_frame_sampler
            logger.info(f"Compiled MOSS-TTS Local frame sampler (mode={compile_mode})")

    @torch.no_grad()
    def _decode_frame_graphable(
        self,
        hidden_states: torch.Tensor,
        text_temperature: torch.Tensor,
        text_top_p: torch.Tensor,
        text_top_k: torch.Tensor,
        audio_temperature: torch.Tensor,
        audio_top_p: torch.Tensor,
        audio_top_k: torch.Tensor,
        seeds: torch.Tensor,
        base_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Branchless frame decode used both eagerly and under graph capture.

        ``base_positions`` is ``generation_steps * (n_vq + 1)``; channel
        ``c``'s sampling position is ``base_positions + c + 1`` and the binary
        text decision samples at ``base_positions``, matching the eager path.
        Repetition penalty is not supported here (the runner falls back to
        the eager path when a request enables it).

        Returns ``(stop_choice, codes, feedback_embeds)`` where
        ``feedback_embeds`` is the next backbone input embedding for a
        continuing row — the assistant-slot text embedding plus all 12 code
        embeddings, summed in the same channel order as
        ``_prepare_multi_modal_inputs``.
        """
        local_hidden = self.local_transformer.step(
            hidden_states.to(dtype=self.dtype), 0
        )
        text_logits = F.linear(local_hidden, self.local_text_lm_head.weight).float()
        stop_choice = self._sample_seeded_branchless(
            text_logits,
            temperature=text_temperature,
            top_p=text_top_p,
            top_k=text_top_k,
            seeds=seeds,
            positions=base_positions,
        )

        slot_ids = torch.full_like(
            seeds, int(self.config.audio_assistant_slot_token_id)
        )
        feedback = self.embedding_list[0](slot_ids)
        codes = []
        current = local_hidden
        for channel in range(self.n_vq):
            head_weight = self._audio_embedding_weight(channel)
            logits = F.linear(current, head_weight).float()
            code = self._sample_seeded_branchless(
                logits,
                temperature=audio_temperature,
                top_p=audio_top_p,
                top_k=audio_top_k,
                seeds=seeds,
                positions=base_positions + channel + 1,
            )
            codes.append(code)
            code_embed = F.embedding(code, head_weight)
            feedback = feedback + code_embed
            if channel + 1 < self.n_vq:
                current = self.local_transformer.step(
                    code_embed.to(dtype=self.dtype), channel + 1
                )
        return stop_choice, torch.stack(codes, dim=-1), feedback

    @torch.no_grad()
    def init_frame_decode_graphs(self, batch_sizes: list[int]) -> None:
        """Capture the per-frame local decode (1 + n_vq micro-steps plus all
        13 seeded sampling passes) into one CUDA graph per batch-size bucket.

        Eager execution launches ~500 kernels per frame (~22 ms regardless of
        batch size); replay collapses that to a few milliseconds. Must run
        during engine init while the device is otherwise idle.
        """
        buckets = sorted({int(bs) for bs in batch_sizes})
        if not buckets:
            return
        device = self.device
        # The captured graphs hold raw pointers into the local KV buffers, so
        # size them for the largest batch any path (graphed or eager fallback)
        # can see and freeze them against reallocation.
        max_eager_bs = int(self._decode_input_embedding.weight.shape[0])
        self.local_transformer._ensure_kv_cache(
            max(max(buckets), max_eager_bs), device, self.dtype
        )
        self.local_transformer.freeze_kv_cache()
        self._ensure_frame_sampler_compile()
        frame_decode = self._decode_frame_graphable
        self._frame_graphs: dict[
            int,
            tuple[
                Any, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor
            ],
        ] = {}

        for bucket in buckets:
            static_inputs = {
                "hidden_states": torch.zeros(
                    bucket, self.hidden_size, device=device, dtype=self.dtype
                ),
                "text_temperature": torch.ones(
                    bucket, device=device, dtype=torch.float32
                ),
                "text_top_p": torch.ones(bucket, device=device, dtype=torch.float32),
                "text_top_k": torch.full(
                    (bucket,), 50, device=device, dtype=torch.long
                ),
                "audio_temperature": torch.ones(
                    bucket, device=device, dtype=torch.float32
                ),
                "audio_top_p": torch.ones(bucket, device=device, dtype=torch.float32),
                "audio_top_k": torch.full(
                    (bucket,), 25, device=device, dtype=torch.long
                ),
                "seeds": torch.zeros(bucket, device=device, dtype=torch.long),
                "base_positions": torch.zeros(bucket, device=device, dtype=torch.long),
            }
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(2):
                    frame_decode(**static_inputs)
            torch.cuda.current_stream().wait_stream(warmup_stream)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                stop_choice, codes, feedback = frame_decode(**static_inputs)
            self._frame_graphs[bucket] = (
                graph,
                static_inputs,
                stop_choice,
                codes,
                feedback,
            )
        logger.info(
            f"MOSS-TTS Local frame-decode CUDA graphs captured for bs={buckets}"
        )

    @property
    def frame_graph_max_bs(self) -> int:
        graphs = getattr(self, "_frame_graphs", None)
        return max(graphs) if graphs else 0

    @torch.no_grad()
    def decode_frame_graphed(
        self,
        hidden_states: torch.Tensor,
        *,
        text_temperature: torch.Tensor,
        text_top_p: torch.Tensor,
        text_top_k: torch.Tensor,
        audio_temperature: torch.Tensor,
        audio_top_p: torch.Tensor,
        audio_top_k: torch.Tensor,
        seeds: torch.Tensor,
        base_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replay the captured frame decode for this batch (padded up to the
        nearest bucket; padding rows sample garbage that the caller discards).

        The returned tensors are slices of the graph's static output buffers:
        the caller must copy anything it keeps before the next replay (any
        later prefill or decode step replays these graphs).
        """
        batch_size = hidden_states.shape[0]
        bucket = min(b for b in self._frame_graphs if b >= batch_size)
        graph, static_inputs, stop_choice, codes, feedback = self._frame_graphs[bucket]

        static_inputs["hidden_states"][:batch_size].copy_(
            hidden_states.to(dtype=self.dtype)
        )
        if batch_size < bucket:
            static_inputs["hidden_states"][batch_size:].zero_()
        for key, value in (
            ("text_temperature", text_temperature),
            ("text_top_p", text_top_p),
            ("text_top_k", text_top_k),
            ("audio_temperature", audio_temperature),
            ("audio_top_p", audio_top_p),
            ("audio_top_k", audio_top_k),
            ("seeds", seeds),
            ("base_positions", base_positions),
        ):
            buf = static_inputs[key]
            buf[:batch_size].copy_(value)
            if batch_size < bucket:
                buf[batch_size:].fill_(1 if buf.dtype.is_floating_point else 1)
        graph.replay()
        return stop_choice[:batch_size], codes[:batch_size], feedback[:batch_size]

    @torch.no_grad()
    def decode_frame(
        self,
        hidden_states: torch.Tensor,
        *,
        sample_text: Callable[[torch.Tensor], torch.Tensor],
        sample_audio: Callable[[torch.Tensor, int], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode one audio frame for the whole batch.

        Args:
            hidden_states: ``[batch, hidden]`` backbone last hidden states.
            sample_text: maps ``[batch, 2]`` fp32 binary-head logits to a
                ``[batch]`` long tensor of indices (0=continue, 1=stop).
            sample_audio: maps (``[batch, audio_vocab]`` fp32 logits, channel)
                to a ``[batch]`` long tensor of codes.

        Returns:
            ``(stop_choice [batch], codes [batch, n_vq])``. Codes are sampled
            for every row regardless of the stop choice; the runner discards
            frames of rows that chose stop, mirroring the upstream loop which
            never emits a frame alongside ``audio_end``.
        """
        local_hidden = self.local_transformer.step(
            hidden_states.to(dtype=self.dtype), 0
        )
        text_logits = F.linear(local_hidden, self.local_text_lm_head.weight)
        stop_choice = sample_text(text_logits.float())

        codes = []
        current = local_hidden
        for channel in range(self.n_vq):
            head_weight = self._audio_embedding_weight(channel)
            logits = F.linear(current, head_weight)
            code = sample_audio(logits.float(), channel)
            codes.append(code)
            if channel + 1 < self.n_vq:
                next_embed = F.embedding(code, head_weight)
                current = self.local_transformer.step(
                    next_embed.to(dtype=self.dtype), channel + 1
                )
        return stop_choice, torch.stack(codes, dim=-1)

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
            if name.startswith("transformer."):
                name = "model." + name[len("transformer.") :]

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
            if "rotary_emb.inv_freq" in name:
                continue

            # Tied heads: the checkpoint may carry text_lm_head /
            # audio_lm_heads tensors that alias embed_tokens /
            # audio_embeddings; the embedding tables are authoritative here.
            if name.startswith("text_lm_head.") or name.startswith("audio_lm_heads."):
                continue

            if name.startswith("audio_embeddings.") and name.endswith(".weight"):
                mapped = self._map_audio_embedding_name(name)
                if mapped is not None and mapped in params_dict:
                    # The checkpoint table has audio_vocab_size rows while the
                    # module reserves an extra (zeroed) pad row, so copy the
                    # real rows directly instead of using the vocab loader.
                    param = params_dict[mapped]
                    rows = int(loaded_weight.shape[0])
                    with torch.no_grad():
                        param.data[:rows].copy_(
                            loaded_weight.to(device=param.device, dtype=param.dtype)
                        )
                continue

            if name.startswith("local_transformer.") or name.startswith(
                "local_text_lm_head."
            ):
                param = params_dict.get(name)
                if param is not None:
                    self._load_param(param, loaded_weight)
                else:
                    logger.warning(
                        f"MOSS-TTS Local parameter {original_name} not found"
                    )
                continue

            if name == "model.embed_tokens.weight":
                mapped = "embedding_list.0.weight"
                if mapped in params_dict:
                    self._load_param(params_dict[mapped], loaded_weight)
                # Fall through: the backbone's own embed_tokens also loads.

            mapped_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                    mapped_stacked = True
                    break
                param = params_dict.get(mapped_name)
                if param is None:
                    break
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                mapped_stacked = True
                break
            if mapped_stacked:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue
            param = params_dict.get(name)
            if param is not None:
                self._load_param(param, loaded_weight)
            else:
                logger.warning(f"MOSS-TTS Local parameter {original_name} not found")

        self._zero_audio_pad_rows()

    def _zero_audio_pad_rows(self) -> None:
        """Zero rows >= audio_vocab_size so pad codes embed to exactly zero."""
        audio_vocab_size = int(self.config.audio_vocab_size)
        with torch.no_grad():
            for layer in self.embedding_list[1:]:
                weight = getattr(layer, "weight", None)
                if (
                    isinstance(weight, torch.Tensor)
                    and weight.shape[0] > audio_vocab_size
                ):
                    weight[audio_vocab_size:].zero_()

    @staticmethod
    def _map_audio_embedding_name(name: str) -> str | None:
        try:
            idx = int(name.split(".")[1]) + 1
        except (IndexError, ValueError):
            return None
        return f"embedding_list.{idx}.weight"

    @staticmethod
    def _load_param(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)

    def get_embed_and_head(self) -> tuple[list[Any], list[Any]]:
        embed_weights = [
            getattr(layer, "weight", None) for layer in self.embedding_list
        ]
        return embed_weights, [self.local_text_lm_head.weight]

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)


EntryClass = MossTTSLocalSGLangModel
