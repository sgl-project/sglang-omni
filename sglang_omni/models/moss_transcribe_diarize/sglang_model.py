# SPDX-License-Identifier: Apache-2.0
"""SGLang-native MOSS-Transcribe-Diarize model."""

from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.cuda_graph_runner import set_torch_compile_config
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from sglang.srt.models.whisper import WhisperEncoder
from sglang.srt.utils import add_prefix

from sglang_omni.models.moss_transcribe_diarize.encoder_cuda_graph import (
    WhisperEncoderCudaGraphRunner,
)
from sglang_omni.models.moss_transcribe_diarize.hf_config import (
    MossTranscribeDiarizeConfig,
)
from sglang_omni.scheduling.stage_cache import StageOutputCache

logger = logging.getLogger(__name__)

_ENCODER_CACHE_MAX_ENTRIES = 64


class VQAdaptor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, norm_eps: float = 1e-6):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.LayerNorm(hidden_size, eps=norm_eps, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MossTranscribeDiarizeForConditionalGeneration(nn.Module):
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: MossTranscribeDiarizeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.whisper_encoder = WhisperEncoder(config.audio_config, quant_config)
        self.vq_adaptor = VQAdaptor(
            input_dim=config.adaptor_input_dim,
            hidden_size=config.text_config.hidden_size,
            norm_eps=config.text_config.rms_norm_eps,
        )
        self.language_model = Qwen3ForCausalLM(
            config.text_config,
            quant_config,
            prefix=add_prefix("model.language_model", prefix),
        )
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        self._encoder_cache: Optional[StageOutputCache] = None
        self._encoder_graph_runner = None
        self._compiled_encoder = None
        self._compiled_chunk_buckets: frozenset[int] = frozenset()
        self._compiled_input_feature_len = 0

    def init_encoder_cache(self, max_bytes: int) -> None:
        self._encoder_cache = (
            StageOutputCache(
                max_size=_ENCODER_CACHE_MAX_ENTRIES,
                max_bytes=max_bytes,
                cache_device="cpu",
            )
            if max_bytes and max_bytes > 0
            else None
        )

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    def init_encoder_graphs(self, chunk_buckets, input_feature_len: int) -> None:
        """Capture per-chunk-count CUDA graphs for the Whisper encoder.

        Called from the stage factory after the model is on-device and CUDA
        graphs are enabled. input_feature_len is the fixed length of the
        encoder's input_features time axis for one 30s window
        (WhisperFeatureExtractor.nb_max_frames).
        """
        buckets = [int(b) for b in (chunk_buckets or []) if int(b) >= 1]
        if not buckets:
            return
        runner = WhisperEncoderCudaGraphRunner(
            self.whisper_encoder,
            num_mel_bins=int(self.config.audio_config.num_mel_bins),
            input_feature_len=int(input_feature_len),
        )
        runner.capture(buckets)
        self._encoder_graph_runner = runner

    def compile_encoder(self, chunk_buckets, input_feature_len: int) -> None:
        """torch.compile the Whisper encoder, warming one specialization per
        chunk-count bucket.

        Mutually exclusive with ``init_encoder_graphs``. ``dynamic=False``
        matches shape exactly, so an off-bucket chunk count or frame length --
        or a bucket whose warmup fails -- falls back to eager. Default mode
        only: reduce-overhead's cudagraph trees corrupt memory alongside the
        decode CUDA graphs that always run in this process.

        TODO(yichi): investigate whether reduce-overhead can coexist with the
        decode CUDA graphs (e.g. cudagraph pool isolation or capture ordering)
        and reclaim its kernel-launch savings.
        """
        buckets = sorted({int(b) for b in (chunk_buckets or []) if int(b) >= 1})
        if not buckets:
            return
        set_torch_compile_config()
        self._compiled_encoder = torch.compile(self.whisper_encoder, dynamic=False)
        self._compiled_input_feature_len = int(input_feature_len)
        p = next(self.whisper_encoder.parameters())
        frames = int(input_feature_len)
        num_mel_bins = int(self.config.audio_config.num_mel_bins)
        pos = torch.arange((frames - 1) // 2 + 1, device=p.device, dtype=torch.long)
        warmed: list[int] = []
        with torch.no_grad():
            for n in buckets:
                feats = torch.zeros(
                    n, num_mel_bins, frames, device=p.device, dtype=p.dtype
                )
                try:
                    for _ in range(3):
                        self._compiled_encoder(feats, pos, None)
                except Exception as exc:
                    logger.warning(
                        f"MOSS-TD encoder torch.compile warmup failed for "
                        f"chunks={n}: {exc}; that chunk count will run eager"
                    )
                    continue
                warmed.append(n)
        self._compiled_chunk_buckets = frozenset(warmed)
        logger.info(f"MOSS-TD encoder torch.compile warmed buckets={warmed}")

    def time_merge(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = features.shape
        merge_size = int(self.config.audio_merge_size)
        trimmed_len = (seq_len // merge_size) * merge_size
        return features[:, :trimmed_len, :].reshape(
            batch_size, trimmed_len // merge_size, hidden_size * merge_size
        )

    def get_audio_feature(
        self,
        items: List[MultimodalDataItem],
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        cache = self._encoder_cache
        key = getattr(items[0], "hash", None) if len(items) == 1 else None
        if cache is not None and key is not None:
            cached = cache.get(str(key))
            if cached is not None:
                device = next(self.vq_adaptor.parameters()).device
                return cached.to(device, non_blocking=True)
            output = self._get_audio_feature_uncached(items, forward_batch)
            cache.put(str(key), output)
            return output
        return self._get_audio_feature_uncached(items, forward_batch)

    def _get_audio_feature_uncached(
        self,
        items: List[MultimodalDataItem],
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # NOTE: sglang's mm dispatch calls this per request, so len(items) is always 1 today.
        merge_size = int(self.config.audio_merge_size)
        device = next(self.whisper_encoder.parameters()).device
        encoder_dtype = next(self.whisper_encoder.parameters()).dtype

        chunks: list[torch.Tensor] = []
        token_lens: list[int] = []
        audio_spans: list[list[int]] = []
        for item in items:
            if item.feature is None:
                raise ValueError(
                    "MOSS-Transcribe-Diarize audio item is missing input_features."
                )
            input_features = item.feature
            num_chunks = input_features.shape[0]

            feature_lengths = getattr(item, "audio_feature_lengths", None)
            if feature_lengths is None:
                raise ValueError(
                    "MOSS-Transcribe-Diarize audio item is missing audio_feature_lengths."
                )
            feature_lengths = feature_lengths.to(device="cpu", dtype=torch.long)
            if feature_lengths.numel() != num_chunks:
                raise ValueError(
                    "audio_feature_lengths must contain one length per input_features "
                    f"chunk: got {feature_lengths.numel()} lengths for {num_chunks} chunks."
                )

            chunk_mapping = getattr(item, "audio_chunk_mapping", None)
            if chunk_mapping is None:
                chunk_mapping = torch.zeros(num_chunks, dtype=torch.long)
            else:
                chunk_mapping = chunk_mapping.to(device="cpu", dtype=torch.long)
            if chunk_mapping.numel() != num_chunks:
                raise ValueError(
                    "audio_chunk_mapping must contain one sample index per input_features "
                    f"chunk: got {chunk_mapping.numel()} indices for {num_chunks} chunks."
                )
            feature_lengths = feature_lengths.tolist()
            chunk_mapping = chunk_mapping.tolist()

            num_audios = max(chunk_mapping) + 1 if chunk_mapping else 0
            per_audio: list[list[int]] = [[] for _ in range(num_audios)]
            for chunk_idx, token_len in enumerate(feature_lengths):
                per_audio[chunk_mapping[chunk_idx]].append(len(chunks))
                chunks.append(input_features[chunk_idx])
                token_lens.append(int(token_len))
            audio_spans.extend(ids for ids in per_audio if ids)

        if not chunks:
            hidden_size = self.config.text_config.hidden_size
            adaptor_param = next(self.vq_adaptor.parameters())
            return torch.empty(
                (0, hidden_size), device=adaptor_param.device, dtype=adaptor_param.dtype
            )

        with torch.no_grad():
            batched_features = torch.stack(chunks).to(
                device=device, dtype=encoder_dtype
            )
            encoder_len = (batched_features.shape[-1] - 1) // 2 + 1
            encoder_position_ids = torch.arange(
                encoder_len, device=device, dtype=torch.long
            )
            if (
                self._compiled_encoder is not None
                and batched_features.shape[0] in self._compiled_chunk_buckets
                and batched_features.shape[-1] == self._compiled_input_feature_len
            ):
                features = self._compiled_encoder(
                    batched_features, encoder_position_ids, forward_batch
                )
            elif self._encoder_graph_runner is not None:
                features = self._encoder_graph_runner.run(
                    batched_features, encoder_position_ids, forward_batch
                )
            else:
                features = self.whisper_encoder(
                    batched_features, encoder_position_ids, forward_batch
                )

            adaptor_dtype = next(self.vq_adaptor.parameters()).dtype
            merged = [
                self.time_merge(
                    torch.cat(
                        [
                            features[i : i + 1, : token_lens[i] * merge_size]
                            for i in ids
                        ],
                        dim=1,
                    ).to(dtype=adaptor_dtype)
                ).squeeze(0)
                for ids in audio_spans
            ]
            return self.vq_adaptor(torch.cat(merged, dim=0))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> torch.Tensor:
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.AUDIO: lambda items: self.get_audio_feature(
                    items,
                    forward_batch,
                ),
            },
            positions=positions,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        whisper_stacked_params_mapping = [
            ("self_attn.qkv_proj", "self_attn.q_proj", "q"),
            ("self_attn.qkv_proj", "self_attn.k_proj", "k"),
            ("self_attn.qkv_proj", "self_attn.v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        def load_one(name: str, loaded_weight: torch.Tensor):
            original_name = name
            if "rotary_emb.inv_freq" in name:
                return
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                return

            if name == "lm_head.weight":
                name = "language_model.lm_head.weight"
            elif name.startswith("model.language_model."):
                name = "language_model.model." + name[len("model.language_model.") :]
            elif name.startswith("model.whisper_encoder."):
                name = "whisper_encoder." + name[len("model.whisper_encoder.") :]
            elif name.startswith("model.vq_adaptor."):
                name = "vq_adaptor." + name[len("model.vq_adaptor.") :]

            if (
                name == "language_model.model.embed_tokens.weight"
                and self.config.text_config.tie_word_embeddings
                and "language_model.lm_head.weight" in params_dict
            ):
                param = params_dict["language_model.lm_head.weight"]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            handled = False
            if name.startswith("whisper_encoder."):
                for param_name, weight_name, shard_id in whisper_stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    mapped_name = name.replace(weight_name, param_name)
                    if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                        handled = True
                        break
                    if mapped_name in params_dict:
                        param = params_dict[mapped_name]
                        param.weight_loader(param, loaded_weight, shard_id)
                        handled = True
                    break

            if name.startswith("language_model."):
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    mapped_name = name.replace(weight_name, param_name)
                    if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                        handled = True
                        break
                    if mapped_name in params_dict:
                        param = params_dict[mapped_name]
                        param.weight_loader(param, loaded_weight, shard_id)
                        handled = True
                    break

            if handled:
                return

            if name.endswith(".bias") and name not in params_dict:
                return

            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                logger.debug(f"Skipping weight: {original_name} -> {name}")

        for name, loaded_weight in weights:
            load_one(name, loaded_weight)
            if (
                name.startswith("model.whisper_encoder.layers.")
                and ".self_attn.k_proj.weight" in name
            ):
                load_one(
                    name.replace(".k_proj.weight", ".k_proj.bias"),
                    torch.zeros(
                        loaded_weight.shape[0],
                        dtype=loaded_weight.dtype,
                        device=loaded_weight.device,
                    ),
                )


EntryClass = MossTranscribeDiarizeForConditionalGeneration
