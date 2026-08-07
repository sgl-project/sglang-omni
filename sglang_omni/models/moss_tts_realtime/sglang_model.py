# SPDX-License-Identifier: Apache-2.0
"""SGLang-native MOSS-TTS-Realtime model."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.nn.functional as F
from sglang.srt.layers.quantization.base_config import QuantizationConfig

from sglang_omni.models.moss_tts_local.sglang_model import (
    MossTTSLocalSGLangModel,
    _as_qwen3_config,
)
from sglang_omni.models.moss_tts_realtime.local_transformer import (
    MossTTSRealtimeLocalTransformer,
)
from sglang_omni.models.moss_tts_realtime.payload_types import (
    AUDIO_EOS_TOKEN,
    AUDIO_PAD_TOKEN,
    N_CODEBOOKS,
    REFERENCE_AUDIO_PAD_TOKEN,
    TEXT_PAD_TOKEN,
)
from sglang_omni.models.moss_tts_realtime.state_pool import (
    MossTTSRealtimeDecodeStatePool,
)

logger = logging.getLogger(__name__)


class MossTTSRealtimeSGLangModel(MossTTSLocalSGLangModel):
    """Qwen3 backbone plus a 16-position Qwen-style local decoder."""

    frame_graph_supports_repetition_penalty = True
    rebuild_frame_feedback_from_rows = True

    def __init__(
        self,
        config: Any,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        del self.local_transformer
        del self.local_text_lm_head
        self.local_transformer = MossTTSRealtimeLocalTransformer(
            self.config.local_config
        )
        self._state_pool = MossTTSRealtimeDecodeStatePool(self)

    @staticmethod
    def _normalize_config(config: Any) -> Any:
        language_config = _as_qwen3_config(config.language_config)
        config.language_config = language_config
        config.hidden_size = int(language_config.hidden_size)
        config.vocab_size = int(language_config.vocab_size)
        config.n_vq = int(getattr(config, "rvq", N_CODEBOOKS))
        config.channels = config.n_vq + 1
        config.audio_vocab_size = int(getattr(config, "audio_vocab_size", 1027))
        config.vocab_size_list = [config.vocab_size] + [
            config.audio_vocab_size
        ] * config.n_vq
        config.audio_pad_code = int(getattr(config, "audio_pad_token", AUDIO_PAD_TOKEN))
        config.pad_token_id = int(
            getattr(language_config, "pad_token_id", None)
            or getattr(language_config, "bos_token_id", 151643)
        )
        config.pad_token = [config.pad_token_id] + [config.audio_pad_code] * config.n_vq
        config.audio_assistant_slot_token_id = TEXT_PAD_TOKEN
        config.audio_end_token_id = int(
            getattr(language_config, "eos_token_id", 151645)
        )
        config.reference_audio_pad = int(
            getattr(config, "reference_audio_pad", REFERENCE_AUDIO_PAD_TOKEN)
        )
        config.text_pad = int(getattr(config, "text_pad", TEXT_PAD_TOKEN))
        config.repetition_window = 50
        config.local_transformer_layers = 1
        config.gpt2_config = {}
        language_config.channels = config.channels
        language_config.vocab_size_list = list(config.vocab_size_list)
        language_config.pad_token = list(config.pad_token)
        local_config = config.local_config
        local_config.rvq = config.n_vq
        local_config.audio_vocab_size = config.audio_vocab_size
        local_config.audio_pad_token = config.audio_pad_code
        return config

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        token_presence: torch.Tensor,
        penalties: torch.Tensor,
    ) -> torch.Tensor:
        active = token_presence & penalties.ne(1.0).unsqueeze(1)
        penalties = penalties.to(logits.dtype).unsqueeze(1)
        penalized = torch.where(logits < 0, logits * penalties, logits / penalties)
        return torch.where(active, penalized, logits)

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
        audio_token_presence: torch.Tensor,
        audio_repetition_penalty: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del text_temperature, text_top_p, text_top_k
        current = hidden_states.to(dtype=self.dtype)
        codes = []
        for channel in range(self.n_vq):
            current = self.local_transformer.step(current, channel)
            logits = F.linear(
                current,
                self.local_transformer.local_lm_heads[channel].weight,
            ).float()
            logits = self._apply_repetition_penalty(
                logits,
                audio_token_presence[:, channel],
                audio_repetition_penalty,
            )
            code = self._sample_seeded_branchless(
                logits,
                temperature=audio_temperature,
                top_p=audio_top_p,
                top_k=audio_top_k,
                seeds=seeds,
                positions=base_positions + channel + 1,
            )
            codes.append(code)
            if channel + 1 < self.n_vq:
                current = F.embedding(
                    code,
                    self.local_transformer.model.embed_tokens[channel].weight,
                ).to(dtype=self.dtype)
        code_rows = torch.stack(codes, dim=-1)
        stop_choice = code_rows[:, 0].eq(AUDIO_EOS_TOKEN).long()
        feedback = hidden_states.new_zeros(hidden_states.shape)
        return stop_choice, code_rows, feedback

    @torch.no_grad()
    def init_frame_decode_graphs(self, batch_sizes: list[int]) -> None:
        capacity = int(self._decode_input_embedding.weight.shape[0])
        buckets = sorted({int(bs) for bs in batch_sizes if 0 < int(bs) <= capacity})
        if not buckets:
            return
        device = self.device
        self.local_transformer.ensure_kv_cache(capacity, device, self.dtype)
        self.local_transformer.freeze_kv_cache()
        self._ensure_frame_sampler_compile()
        self._frame_graphs = {}
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
                    (bucket,), 30, device=device, dtype=torch.long
                ),
                "audio_temperature": torch.full(
                    (bucket,), 0.8, device=device, dtype=torch.float32
                ),
                "audio_top_p": torch.full(
                    (bucket,), 0.6, device=device, dtype=torch.float32
                ),
                "audio_top_k": torch.full(
                    (bucket,), 30, device=device, dtype=torch.long
                ),
                "seeds": torch.zeros(bucket, device=device, dtype=torch.long),
                "base_positions": torch.zeros(bucket, device=device, dtype=torch.long),
                "audio_token_presence": torch.zeros(
                    bucket,
                    self.n_vq,
                    int(self.config.audio_vocab_size),
                    device=device,
                    dtype=torch.bool,
                ),
                "audio_repetition_penalty": torch.ones(
                    bucket, device=device, dtype=torch.float32
                ),
            }
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(2):
                    self._decode_frame_graphable(**static_inputs)
            torch.cuda.current_stream().wait_stream(warmup_stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = self._decode_frame_graphable(**static_inputs)
            self._frame_graphs[bucket] = (graph, static_inputs, *outputs)
        logger.info("MOSS-TTS-Realtime frame CUDA graphs captured for bs=%s", buckets)

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
        audio_token_presence: torch.Tensor,
        audio_repetition_penalty: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(hidden_states.shape[0])
        bucket = min(size for size in self._frame_graphs if size >= batch_size)
        graph, static, stop_choice, codes, feedback = self._frame_graphs[bucket]
        values = {
            "hidden_states": hidden_states,
            "text_temperature": text_temperature,
            "text_top_p": text_top_p,
            "text_top_k": text_top_k,
            "audio_temperature": audio_temperature,
            "audio_top_p": audio_top_p,
            "audio_top_k": audio_top_k,
            "seeds": seeds,
            "base_positions": base_positions,
            "audio_token_presence": audio_token_presence,
            "audio_repetition_penalty": audio_repetition_penalty,
        }
        for name, value in values.items():
            static[name][:batch_size].copy_(value)
            if batch_size < bucket:
                static[name][batch_size:].zero_()
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
        del sample_text
        current = hidden_states.to(dtype=self.dtype)
        codes = []
        for channel in range(self.n_vq):
            current = self.local_transformer.step(current, channel)
            logits = F.linear(
                current,
                self.local_transformer.local_lm_heads[channel].weight,
            )
            code = sample_audio(logits.float(), channel)
            codes.append(code)
            if channel + 1 < self.n_vq:
                current = F.embedding(
                    code,
                    self.local_transformer.model.embed_tokens[channel].weight,
                ).to(dtype=self.dtype)
        code_rows = torch.stack(codes, dim=-1)
        return code_rows[:, 0].eq(AUDIO_EOS_TOKEN).long(), code_rows

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        def translated() -> Iterable[tuple[str, torch.Tensor]]:
            for name, weight in weights:
                if name.startswith("embed_tokens."):
                    yield "embedding_list." + name[len("embed_tokens.") :], weight
                elif name.startswith("language_model."):
                    yield "model." + name[len("language_model.") :], weight
                else:
                    yield name, weight

        super().load_weights(translated())

    def _zero_audio_pad_rows(self) -> None:
        with torch.no_grad():
            for layer in self.embedding_list[1:]:
                weight = getattr(layer, "weight", None)
                if isinstance(weight, torch.Tensor):
                    weight[AUDIO_PAD_TOKEN].zero_()
            for layer in self.local_transformer.model.embed_tokens:
                layer.weight[AUDIO_PAD_TOKEN].zero_()

    def get_embed_and_head(self) -> tuple[list[Any], list[Any]]:
        embeds = [getattr(layer, "weight", None) for layer in self.embedding_list]
        embeds.extend(
            layer.weight for layer in self.local_transformer.model.embed_tokens
        )
        heads = [layer.weight for layer in self.local_transformer.local_lm_heads]
        return embeds, heads


EntryClass = MossTTSRealtimeSGLangModel
