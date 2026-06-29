# SPDX-License-Identifier: Apache-2.0
"""SGLang-native Qwen3-TTS talker wrapper."""

from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import multinomial_with_seed
from sglang.srt.utils import add_prefix
from torch import nn

from sglang_omni.models.qwen3_omni.components.talker import (  # noqa: E501
    Qwen3OmniMoeTalkerDenseMLP,
    ResizeMLP,
    _bind_default_weight_loaders,
)
from sglang_omni.models.qwen3_omni.components.thinker_model import (
    Qwen3OmniMoeThinkerTextAttention,
)
from sglang_omni.models.qwen3_tts.compat import (
    apply_qwen_tts_transformers_compatibility_patches,
)
from sglang_omni.models.qwen3_tts.sampling_kernels import (
    sample_from_sorted_probs_with_seed_small_k,
)
from sglang_omni.vendor.sglang.core import ForwardBatch
from sglang_omni.vendor.sglang.layers import ReplicatedLinear, RMSNorm
from sglang_omni.vendor.sglang.models import apply_qk_norm
from sglang_omni.vendor.sglang.server_args import get_global_server_args


def _sample_seeded_categorical(
    weights: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    return multinomial_with_seed(weights, seeds, positions).view(-1)


class Qwen3TTSTalkerDecoderLayer(nn.Module):
    def __init__(self, config: Any, layer_id: int, prefix: str = "") -> None:
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
    def __init__(self, config: Any, prefix: str = "") -> None:
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

        max_batch_size = get_global_server_args().max_running_requests
        self._feedback_buffer = torch.zeros(
            max_batch_size,
            config.hidden_size,
            device=self.codec_embedding.weight.device,
            dtype=self.codec_embedding.weight.dtype,
        )
        self._feedback_mask = torch.zeros(
            max_batch_size,
            dtype=torch.bool,
            device=self.codec_embedding.weight.device,
        )
        self._decode_feedback_embedding = nn.Embedding(
            max_batch_size,
            config.hidden_size,
            device=self.codec_embedding.weight.device,
            dtype=self.codec_embedding.weight.dtype,
        )
        self._decode_feedback_embedding.weight.requires_grad_(False)

    def get_input_embeddings(self):
        return self.codec_embedding

    def get_text_embeddings(self):
        return self.text_embedding

    def _build_input_hidden_states(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.codec_embedding(input_ids)
        bs = hidden_states.shape[0]
        feedback_mask = self._feedback_mask[:bs]
        return torch.where(
            feedback_mask.unsqueeze(-1),
            self._feedback_buffer[:bs].to(hidden_states.dtype),
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
                hidden_states = self._decode_feedback_embedding(input_ids)
            else:
                hidden_states = self._build_input_hidden_states(input_ids)
        else:
            hidden_states = input_embeds

        residual = None
        layers = self.layers
        if is_decode:
            layers = getattr(self, "_compiled_decode_layers", self.layers)
        for idx in range(self.start_layer, self.end_layer):
            hidden_states, residual = layers[idx](
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=residual,
            )
        if residual is None:
            return self.norm(hidden_states)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3TTSCodePredictor(nn.Module):
    def __init__(self, config: Any, prefix: str = "") -> None:
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
        return self.small_to_mtp_projection(hidden_states)


class Qwen3TTSTalker(nn.Module):
    """Qwen3-TTS Base talker with SGLang-managed KV cache for the main AR loop."""

    def __init__(self, config: Any, quant_config: Any = None, prefix: str = "") -> None:
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

        max_batch_size = get_global_server_args().max_running_requests
        hidden_size = config.hidden_size
        predictor_hidden_size = config.code_predictor_config.hidden_size
        predictor_len = config.num_code_groups + 1
        device = self.model.codec_embedding.weight.device
        dtype = self.model.codec_embedding.weight.dtype
        self._feedback_buffer = self.model._feedback_buffer
        self._feedback_mask = self.model._feedback_mask
        self._decode_feedback_embedding = self.model._decode_feedback_embedding
        self._predictor_input_buffer = torch.zeros(
            max_batch_size,
            predictor_len,
            predictor_hidden_size,
            device=device,
            dtype=dtype,
        )
        cp_layers = self.code_predictor.model.layers
        cp_attn = cp_layers[0].self_attn
        self._predictor_positions = torch.arange(
            predictor_len, device=device, dtype=torch.long
        )
        self._predictor_position_rows = (
            self._predictor_positions[:, None]
            .expand(predictor_len, max_batch_size)
            .contiguous()
        )
        self._predictor_k_cache = torch.zeros(
            len(cp_layers),
            max_batch_size,
            cp_attn.num_kv_heads,
            predictor_len,
            cp_attn.head_dim,
            device=device,
            dtype=dtype,
        )
        self._predictor_v_cache = torch.zeros_like(self._predictor_k_cache)
        self._sampled_token_ids = torch.zeros(
            max_batch_size, dtype=torch.long, device=device
        )
        self._output_codes = torch.zeros(
            max_batch_size,
            config.num_code_groups,
            dtype=torch.long,
            device=device,
        )
        self._output_embeds = torch.zeros(
            max_batch_size, hidden_size, device=device, dtype=dtype
        )
        self._sub_batch_size = 0
        self._sub_temperature_tensor = torch.full(
            (max_batch_size,), 0.9, device=device, dtype=torch.float32
        )
        self._sub_top_p_tensor = torch.ones(
            max_batch_size, device=device, dtype=torch.float32
        )
        self._sub_top_k_tensor = torch.full(
            (max_batch_size,), 50, device=device, dtype=torch.long
        )
        self._semantic_sampling_seed_tensor = torch.zeros(
            max_batch_size, device=device, dtype=torch.long
        )
        self._sub_sampling_seed_tensor = torch.zeros(
            max_batch_size, device=device, dtype=torch.long
        )
        self._sub_sample_rows: list[int] = []
        self._sub_sample_row_indices_tensor = torch.empty(
            max_batch_size, device=device, dtype=torch.long
        )
        self._sub_sample_count = 0
        self._sub_has_sampled_rows = False
        self._sub_sampled_has_top_p = False
        self._sub_sampled_max_top_k = 0
        self._sub_sampled_has_unbounded_top_k = False
        _bind_default_weight_loaders(self)
        self._cached_params_dict = dict(self.named_parameters())
        self._sampler = None

    @property
    def device(self) -> torch.device:
        return self.model.codec_embedding.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.model.codec_embedding.weight.dtype

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_text_embeddings(self):
        return self.model.get_text_embeddings()

    def load_speech_tokenizer(self, speech_tokenizer: Any) -> None:
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
        if self.speaker_encoder is None:
            raise RuntimeError("Qwen3-TTS speaker encoder is not loaded")
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
    def generate_speaker_prompt(self, voice_clone_prompt: dict[str, Any]):
        return [
            emb.to(self.device).to(self.dtype)
            for emb in voice_clone_prompt["ref_spk_embedding"]
        ]

    def _build_instruct_embed(
        self,
        instruct_id: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if instruct_id is None:
            return None
        return self.text_projection(self.get_text_embeddings()(instruct_id))

    def _build_tts_special_embeds(
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

    def _resolve_language_id(
        self,
        *,
        language: str,
        voice: str | None = None,
    ) -> int | None:
        if language.lower() != "auto":
            return self.config.codec_language_id[language.lower()]
        if voice is None:
            return None
        spk_is_dialect = getattr(self.config, "spk_is_dialect", None) or {}
        dialect = spk_is_dialect.get(voice.lower())
        if isinstance(dialect, str) and dialect:
            return self.config.codec_language_id.get(dialect)
        return None

    def _build_codec_prefill(
        self,
        *,
        language: str,
        dtype: torch.dtype,
        voice: str | None = None,
    ) -> torch.Tensor:
        language_id = self._resolve_language_id(language=language, voice=voice)
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

    def _finish_text_prompt(
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

    def _apply_instruct_prefix(
        self,
        talker_input_embed: torch.Tensor,
        instruct_id: torch.Tensor | None,
    ) -> torch.Tensor:
        instruct_embed = self._build_instruct_embed(instruct_id)
        if instruct_embed is None:
            return talker_input_embed
        return torch.cat([instruct_embed, talker_input_embed], dim=1)

    def _build_conditioned_prompt_prefix(
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
        voice_clone_prompt: dict[str, Any],
        language: str,
        non_streaming_mode: bool,
        instruct_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        voice_clone_spk_embeds = self.generate_speaker_prompt(voice_clone_prompt)
        speaker_embed = voice_clone_spk_embeds[0]

        tts_bos_embed, tts_eos_embed, tts_pad_embed = self._build_tts_special_embeds(
            dtype=input_id.dtype
        )
        codec_input_0 = self._build_codec_prefill(
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
        talker_input_embed = self._build_conditioned_prompt_prefix(
            input_id=input_id,
            codec_input=codec_input,
            tts_bos_embed=tts_bos_embed,
            tts_pad_embed=tts_pad_embed,
        )

        ref_code = None
        ref_codes = voice_clone_prompt.get("ref_code")
        if ref_codes is not None:
            ref_code = ref_codes[0]

        if ref_code is not None and voice_clone_prompt["icl_mode"][0]:
            if ref_id is None:
                raise ValueError("Qwen3-TTS ICL mode requires ref_text tokens")
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
            talker_input_embed, trailing_text_hidden = self._finish_text_prompt(
                talker_input_embed=talker_input_embed,
                input_id=input_id,
                codec_last_embed=codec_input[:, -1:],
                tts_pad_embed=tts_pad_embed,
                tts_eos_embed=tts_eos_embed,
                non_streaming_mode=non_streaming_mode,
            )

        talker_input_embed = self._apply_instruct_prefix(
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
        speaker_key = voice.lower()
        spk_id_map = {str(key).lower(): value for key, value in spk_id.items()}
        if speaker_key not in spk_id_map:
            supported = ", ".join(sorted(str(key) for key in spk_id))
            raise ValueError(
                f"Unsupported Qwen3-TTS CustomVoice speaker {voice!r}. "
                f"Supported speakers: {supported}"
            )

        tts_bos_embed, tts_eos_embed, tts_pad_embed = self._build_tts_special_embeds(
            dtype=input_id.dtype
        )
        codec_input_0 = self._build_codec_prefill(
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
        talker_input_embed = self._build_conditioned_prompt_prefix(
            input_id=input_id,
            codec_input=codec_input,
            tts_bos_embed=tts_bos_embed,
            tts_pad_embed=tts_pad_embed,
        )
        talker_input_embed, trailing_text_hidden = self._finish_text_prompt(
            talker_input_embed=talker_input_embed,
            input_id=input_id,
            codec_last_embed=codec_input[:, -1:],
            tts_pad_embed=tts_pad_embed,
            tts_eos_embed=tts_eos_embed,
            non_streaming_mode=non_streaming_mode,
        )
        talker_input_embed = self._apply_instruct_prefix(
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

        tts_bos_embed, tts_eos_embed, tts_pad_embed = self._build_tts_special_embeds(
            dtype=input_id.dtype
        )
        codec_input_0 = self._build_codec_prefill(
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
        talker_input_embed = self._build_conditioned_prompt_prefix(
            input_id=input_id,
            codec_input=codec_input,
            tts_bos_embed=tts_bos_embed,
            tts_pad_embed=tts_pad_embed,
        )
        talker_input_embed, trailing_text_hidden = self._finish_text_prompt(
            talker_input_embed=talker_input_embed,
            input_id=input_id,
            codec_last_embed=codec_input[:, -1:],
            tts_pad_embed=tts_pad_embed,
            tts_eos_embed=tts_eos_embed,
            non_streaming_mode=non_streaming_mode,
        )
        talker_input_embed = self._apply_instruct_prefix(
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
        if text_lens > codec_lens:
            return text_embed[:, :codec_lens] + codec_embed, text_embed[:, codec_lens:]
        text_embed = torch.cat(
            [text_embed] + [tts_pad_embed] * (codec_lens - text_lens), dim=1
        )
        return text_embed + codec_embed, tts_pad_embed

    def prepare_decode_buffers(self, requests: list[Any]) -> None:
        batch_size = len(requests)
        if batch_size > self._sub_temperature_tensor.shape[0]:
            raise RuntimeError("Qwen3-TTS sampling buffers are too small")

        semantic_seeds: list[int] = []
        sub_temperatures: list[float] = []
        sub_top_ps: list[float] = []
        sub_top_ks: list[int] = []
        sub_seeds: list[int] = []
        self._sub_sample_rows = []
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
            sub_temperatures.append(subtalker_temperature)
            sub_top_ps.append(subtalker_top_p)
            sub_top_ks.append(subtalker_top_k)
            sub_seeds.append(subtalker_seed)
            if do_sample:
                self._sub_sample_rows.append(row_idx)

        predictor_vocab_size = int(self.config.code_predictor_config.vocab_size)
        sampled_top_ks = [sub_top_ks[row_idx] for row_idx in self._sub_sample_rows]
        bounded_top_ks = [
            top_k for top_k in sampled_top_ks if 0 < int(top_k) < predictor_vocab_size
        ]
        self._sub_batch_size = batch_size
        self._sub_sample_count = len(self._sub_sample_rows)
        self._sub_has_sampled_rows = bool(self._sub_sample_rows)
        self._sub_sampled_has_top_p = any(
            0.0 < float(sub_top_ps[row_idx]) < 1.0 for row_idx in self._sub_sample_rows
        )
        self._sub_sampled_max_top_k = max(bounded_top_ks, default=0)
        self._sub_sampled_has_unbounded_top_k = len(bounded_top_ks) != len(
            sampled_top_ks
        )

        if batch_size == 0:
            return

        device = self._sub_temperature_tensor.device
        self._semantic_sampling_seed_tensor[:batch_size] = torch.tensor(
            semantic_seeds,
            device=device,
            dtype=self._semantic_sampling_seed_tensor.dtype,
        )
        self._sub_temperature_tensor[:batch_size] = torch.tensor(
            sub_temperatures, device=device, dtype=self._sub_temperature_tensor.dtype
        )
        self._sub_top_p_tensor[:batch_size] = torch.tensor(
            sub_top_ps, device=device, dtype=self._sub_top_p_tensor.dtype
        )
        self._sub_top_k_tensor[:batch_size] = torch.tensor(
            sub_top_ks, device=device, dtype=self._sub_top_k_tensor.dtype
        )
        self._sub_sampling_seed_tensor[:batch_size] = torch.tensor(
            sub_seeds, device=device, dtype=self._sub_sampling_seed_tensor.dtype
        )
        if self._sub_sample_count:
            self._sub_sample_row_indices_tensor[: self._sub_sample_count] = (
                torch.tensor(self._sub_sample_rows, device=device, dtype=torch.long)
            )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        input_embeds_are_projected: bool = False,
    ) -> LogitsProcessorOutput:
        del input_embeds_are_projected
        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions

        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )
        if forward_batch.forward_mode.is_extend():
            last_index = self._extend_last_index(forward_batch, hidden_states.device)
            hidden_states = hidden_states[last_index]
        logits, _ = self.codec_head(hidden_states)
        logits_output = LogitsProcessorOutput(
            next_token_logits=logits,
            hidden_states=hidden_states,
        )
        return logits_output

    def _extend_last_index(
        self,
        forward_batch: ForwardBatch,
        device: torch.device,
    ) -> torch.Tensor:
        extend_seq_lens = forward_batch.extend_seq_lens
        if extend_seq_lens is None:
            return torch.tensor([forward_batch.input_ids.shape[0] - 1], device=device)
        return torch.cumsum(extend_seq_lens.to(device=device), dim=0) - 1

    @torch.no_grad()
    def code_predictor_forward(
        self,
        layer0_codes: torch.Tensor,
        talker_hidden: torch.Tensor,
        semantic_positions: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        result_codes, summed_embeddings = self._code_predictor_forward_incremental(
            layer0_codes=layer0_codes,
            talker_hidden=talker_hidden,
            semantic_positions=semantic_positions,
        )
        return result_codes, summed_embeddings

    def _code_predictor_forward_incremental(
        self,
        layer0_codes: torch.Tensor,
        talker_hidden: torch.Tensor,
        semantic_positions: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer0_codes.ndim == 1:
            layer0_codes = layer0_codes.unsqueeze(1)
        if talker_hidden.ndim == 2:
            talker_hidden = talker_hidden.unsqueeze(1)

        batch_size, seq_len = layer0_codes.shape
        semantic_positions = self._normalize_semantic_positions(
            semantic_positions,
            batch_size=batch_size,
            seq_len=seq_len,
            device=layer0_codes.device,
        )
        predictor_input = self._predictor_input_buffer[:batch_size]
        predictor_input.zero_()
        num_groups = self.config.num_code_groups
        result_codes = self._output_codes[:batch_size].unsqueeze(-1)
        summed_embeddings = self._output_embeds[:batch_size].unsqueeze(1)
        result_codes.zero_()
        summed_embeddings.zero_()

        for pos in range(seq_len):
            layer0_code = layer0_codes[:, pos : pos + 1]
            layer0_embed = self.get_input_embeddings()(layer0_code).to(
                dtype=predictor_input.dtype
            )
            layer0_predictor_embed = self.code_predictor.project_input(layer0_embed)
            pos_codes = result_codes[:, :, pos]
            pos_summed = summed_embeddings[:, pos, :]
            pos_summed.zero_()
            predictor_input[:, 0, :] = self.code_predictor.project_input(
                talker_hidden[:, pos : pos + 1, :]
            )[:, 0, :].to(dtype=predictor_input.dtype)
            predictor_input[:, 1, :] = layer0_predictor_embed[:, 0, :]
            pos_codes[:, 0].copy_(layer0_code[:, 0])
            pos_summed.add_(layer0_embed[:, 0, :])

            cache_len = 0
            self._predictor_forward_one_token(
                token_embeds=predictor_input[:, 0:1, :],
                batch_size=batch_size,
                cache_len=cache_len,
            )
            cache_len += 1
            last_hidden = self._predictor_forward_one_token(
                token_embeds=predictor_input[:, 1:2, :],
                batch_size=batch_size,
                cache_len=cache_len,
            )
            cache_len += 1

            for layer_idx in range(num_groups - 1):
                logits, _ = self.code_predictor.lm_head[layer_idx](last_hidden)
                next_code = self._sample_subtalker_token(
                    logits[:, -1, :],
                    layer_idx,
                    semantic_positions=semantic_positions[:, pos],
                )
                pos_codes[:, layer_idx + 1].copy_(next_code)
                new_embed = self.code_predictor.model.codec_embedding[layer_idx](
                    next_code.unsqueeze(1)
                ).to(dtype=predictor_input.dtype)
                new_predictor_embed = self.code_predictor.project_input(new_embed)
                predictor_input[:, layer_idx + 2, :] = new_predictor_embed[:, 0, :]
                pos_summed.add_(new_embed[:, 0, :])
                if layer_idx < num_groups - 2:
                    last_hidden = self._predictor_forward_one_token(
                        token_embeds=new_predictor_embed,
                        batch_size=batch_size,
                        cache_len=cache_len,
                    )
                    cache_len += 1
        return result_codes, summed_embeddings

    def _normalize_semantic_positions(
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
                return base
            if base.ndim != 1 or base.shape[0] != batch_size:
                raise ValueError("Qwen3-TTS subtalker positions shape mismatch")
        offsets = torch.arange(seq_len, device=device, dtype=torch.long)
        return base.unsqueeze(1) + offsets.unsqueeze(0)

    def _sample_subtalker_token(
        self,
        logits: torch.Tensor,
        layer_idx: int = 0,
        *,
        semantic_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits.shape[0] == 0:
            return torch.empty((0,), device=logits.device, dtype=torch.long)
        batch_size = int(logits.shape[0])
        if batch_size > self._sub_batch_size:
            raise RuntimeError("Qwen3-TTS subtalker sampling buffers are too small")

        if not self._sub_has_sampled_rows:
            return torch.argmax(logits, dim=-1).to(dtype=torch.long)

        if self._sub_sample_rows[-1] >= batch_size:
            raise RuntimeError("Qwen3-TTS sampled row index exceeds batch size")
        sampled_rows = self._sub_sample_row_indices_tensor[: self._sub_sample_count]
        sampled_positions = self._select_semantic_positions(
            semantic_positions,
            batch_size,
            logits.device,
        )
        if self._sub_sample_count == batch_size:
            return self._sample_subtalker_token_seeded(
                logits,
                layer_idx,
                row_indices=sampled_rows,
                semantic_positions=sampled_positions,
            )

        tokens = torch.argmax(logits, dim=-1).to(dtype=torch.long)
        sampled_logits = logits.index_select(0, sampled_rows)
        tokens[sampled_rows] = self._sample_subtalker_token_seeded(
            sampled_logits,
            layer_idx,
            row_indices=sampled_rows,
            semantic_positions=sampled_positions,
        )
        return tokens

    def _select_semantic_positions(
        self,
        semantic_positions: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if semantic_positions is None:
            raise RuntimeError("Qwen3-TTS sampled subtalker rows require positions")
        semantic_positions = semantic_positions.to(device=device, dtype=torch.long)
        if semantic_positions.ndim != 1 or semantic_positions.shape[0] != batch_size:
            raise ValueError("Qwen3-TTS subtalker positions shape mismatch")
        sample_rows = self._sub_sample_row_indices_tensor[: self._sub_sample_count]
        return semantic_positions.index_select(0, sample_rows)

    def _sample_subtalker_token_seeded(
        self,
        logits: torch.Tensor,
        layer_idx: int,
        *,
        row_indices: torch.Tensor,
        semantic_positions: torch.Tensor,
    ) -> torch.Tensor:
        row_indices = row_indices.to(device=logits.device, dtype=torch.long)
        scores = logits.float()
        temperatures = self._sub_temperature_tensor.index_select(
            0, row_indices
        ).clamp_min(1e-5)
        scores = scores / temperatures.unsqueeze(1)

        vocab_size = int(logits.shape[-1])
        top_ks = self._sub_top_k_tensor.index_select(0, row_indices)
        max_top_k = int(self._sub_sampled_max_top_k)
        has_unbounded_top_k = bool(self._sub_sampled_has_unbounded_top_k)
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

        top_ps = self._sub_top_p_tensor.index_select(0, row_indices)
        seeds = self._sub_sampling_seed_tensor.index_select(0, row_indices)
        sub_positions = (
            semantic_positions.to(device=logits.device, dtype=torch.long)
            * max(int(self.config.num_code_groups) - 1, 1)
            + int(layer_idx)
            + 1
        )

        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        if self._sub_sampled_has_top_p:
            active_top_p = (top_ps > 0.0) & (top_ps < 1.0)
            cdf = torch.cumsum(sorted_probs, dim=-1)
            remove = (cdf > top_ps.unsqueeze(1)) & active_top_p.unsqueeze(1)
            remove[:, 0] = False
            sorted_probs = sorted_probs.masked_fill(remove, 0.0)

        sampled = sample_from_sorted_probs_with_seed_small_k(
            sorted_probs,
            sorted_idx,
            seeds,
            sub_positions,
        )
        if sampled is not None:
            return sampled.to(torch.long)

        sampled_rank = _sample_seeded_categorical(
            sorted_probs,
            seeds,
            sub_positions,
        ).to(device=logits.device, dtype=torch.long)
        return sorted_idx.gather(1, sampled_rank.unsqueeze(1)).view(-1).to(torch.long)

    def _predictor_forward_one_token(
        self,
        *,
        token_embeds: torch.Tensor,
        batch_size: int,
        cache_len: int,
    ) -> torch.Tensor:
        hidden_states = token_embeds
        hidden_size = hidden_states.shape[-1]
        positions = self._predictor_position_rows[cache_len, :batch_size]
        for layer_idx, layer in enumerate(self.code_predictor.model.layers):
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states.reshape(-1, hidden_size))
            normed = normed.reshape(batch_size, 1, hidden_size)
            attn_out = self._predictor_cached_self_attention(
                layer_idx=layer_idx,
                attn=layer.self_attn,
                hidden_states=normed,
                positions=positions,
                batch_size=batch_size,
                cache_len=cache_len,
            )
            hidden_states = residual + attn_out
            residual = hidden_states
            normed = layer.post_attention_layernorm(
                hidden_states.reshape(-1, hidden_size)
            )
            mlp_out = layer.mlp(normed).reshape(batch_size, 1, hidden_size)
            hidden_states = residual + mlp_out
        hidden_states = self.code_predictor.model.norm(
            hidden_states.reshape(-1, hidden_size)
        )
        return hidden_states.reshape(batch_size, 1, hidden_size)

    def _predictor_cached_self_attention(
        self,
        *,
        layer_idx: int,
        attn: Qwen3OmniMoeThinkerTextAttention,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        batch_size: int,
        cache_len: int,
    ) -> torch.Tensor:
        _, seq_len, hidden_size = hidden_states.shape
        if seq_len != 1:
            raise ValueError("Qwen3-TTS predictor cache expects one token")
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
        q, k = attn.rotary_emb(
            positions.to(device=flat_hidden.device, dtype=torch.long),
            q,
            k,
            fused_set_kv_buffer_arg=None,
        )
        q = q.reshape(batch_size, 1, attn.num_heads, attn.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, 1, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, 1, attn.num_kv_heads, attn.head_dim).transpose(1, 2)

        layer_k_cache = self._predictor_k_cache[layer_idx, :batch_size]
        layer_v_cache = self._predictor_v_cache[layer_idx, :batch_size]
        layer_k_cache[:, :, cache_len : cache_len + 1, :].copy_(k)
        layer_v_cache[:, :, cache_len : cache_len + 1, :].copy_(v)
        cached_k = layer_k_cache[:, :, : cache_len + 1, :]
        cached_v = layer_v_cache[:, :, : cache_len + 1, :]
        num_kv_groups = attn.num_heads // attn.num_kv_heads
        if num_kv_groups == 1:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q,
                cached_k,
                cached_v,
                is_causal=False,
            )
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q,
                cached_k,
                cached_v,
                is_causal=False,
                enable_gqa=True,
            )
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, attn.num_heads * attn.head_dim
        )
        attn_output, _ = attn.o_proj(attn_output)
        return attn_output.reshape(batch_size, 1, hidden_size)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = self._cached_params_dict
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
            if handled:
                continue
            param = params_dict.get(target)
            if param is not None:
                weight_loader = getattr(param, "weight_loader", None)
                if weight_loader is None:
                    param.data.copy_(loaded_weight)
                else:
                    weight_loader(param, loaded_weight)


EntryClass = Qwen3TTSTalker
