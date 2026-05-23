# SPDX-License-Identifier: Apache-2.0
"""SGLang-native Qwen3-TTS talker wrapper."""

from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.utils import add_prefix
from torch import nn

from sglang_omni.models.qwen3_omni.components.talker import (  # noqa: E501
    Qwen3OmniMoeTalkerDenseMLP,
    ResizeMLP,
    _bind_default_weight_loaders,
    _repeat_kv,
)
from sglang_omni.models.qwen3_omni.components.thinker_model import (
    Qwen3OmniMoeThinkerTextAttention,
)
from sglang_omni.vendor.sglang.core import ForwardBatch
from sglang_omni.vendor.sglang.layers import ReplicatedLinear, RMSNorm
from sglang_omni.vendor.sglang.models import apply_qk_norm
from sglang_omni.vendor.sglang.server_args import get_global_server_args


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

    def get_input_embeddings(self):
        return self.codec_embedding

    def get_text_embeddings(self):
        return self.text_embedding

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.codec_embedding(input_ids)
            bs = hidden_states.shape[0]
            feedback_mask = self._feedback_mask[:bs]
            hidden_states = torch.where(
                feedback_mask.unsqueeze(-1),
                self._feedback_buffer[:bs].to(hidden_states.dtype),
                hidden_states,
            )
            self._feedback_mask[:bs] = False
        else:
            hidden_states = input_embeds

        residual = None
        for idx in range(self.start_layer, self.end_layer):
            hidden_states, residual = self.layers[idx](
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
        self._sub_dosample: list[bool] = []
        self._sub_temperature: list[float] = []
        self._sub_top_p: list[float] = []
        self._sub_top_k: list[int] = []
        self._sub_generators: list[torch.Generator | None] = []
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
        return self.config.spk_id.keys()

    @torch.inference_mode()
    def extract_speaker_embedding(self, audio, sr):
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

    def build_voice_clone_inputs(
        self,
        *,
        input_id: torch.Tensor,
        ref_id: torch.Tensor | None,
        voice_clone_prompt: dict[str, Any],
        language: str,
        non_streaming_mode: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        voice_clone_spk_embeds = self.generate_speaker_prompt(voice_clone_prompt)
        speaker_embed = voice_clone_spk_embeds[0]

        language_id = None
        if language.lower() != "auto":
            language_id = self.config.codec_language_id[language.lower()]

        ids = torch.tensor(
            [
                [
                    self.root_config.tts_bos_token_id,
                    self.root_config.tts_eos_token_id,
                    self.root_config.tts_pad_token_id,
                ]
            ],
            device=self.device,
            dtype=input_id.dtype,
        )
        tts_bos_embed, tts_eos_embed, tts_pad_embed = self.text_projection(
            self.get_text_embeddings()(ids)
        ).chunk(3, dim=1)

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

        codec_input_0 = self.get_input_embeddings()(
            torch.tensor([codec_prefill], device=self.device, dtype=input_id.dtype)
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

        role_embed = self.text_projection(self.get_text_embeddings()(input_id[:, :3]))
        prompt_embed = (
            torch.cat(
                [tts_pad_embed.expand(-1, codec_input.shape[1] - 2, -1), tts_bos_embed],
                dim=1,
            )
            + codec_input[:, :-1]
        )
        talker_input_embed = torch.cat([role_embed, prompt_embed], dim=1)

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
            first_text = (
                self.text_projection(self.get_text_embeddings()(input_id[:, 3:4]))
                + codec_input[:, -1:]
            )
            talker_input_embed = torch.cat([talker_input_embed, first_text], dim=1)
            if non_streaming_mode:
                talker_input_embed = torch.cat(
                    [
                        talker_input_embed[:, :-1],
                        torch.cat(
                            [
                                self.text_projection(
                                    self.get_text_embeddings()(input_id[:, 3:-5])
                                ),
                                tts_eos_embed,
                            ],
                            dim=1,
                        )
                        + self.get_input_embeddings()(
                            torch.tensor(
                                [
                                    [self.config.codec_pad_id]
                                    * (input_id[:, 3:-5].shape[1] + 1)
                                ],
                                device=self.device,
                                dtype=input_id.dtype,
                            )
                        ),
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
                trailing_text_hidden = tts_pad_embed
            else:
                trailing_text_hidden = torch.cat(
                    [
                        self.text_projection(
                            self.get_text_embeddings()(input_id[:, 4:-5])
                        ),
                        tts_eos_embed,
                    ],
                    dim=1,
                )

        attention_mask = torch.ones(
            (1, talker_input_embed.shape[1]), device=self.device, dtype=torch.long
        )
        return talker_input_embed, attention_mask, trailing_text_hidden, ref_code

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
        self._sub_dosample = []
        self._sub_temperature = []
        self._sub_top_p = []
        self._sub_top_k = []
        self._sub_generators = []
        generator_device = self.device
        for sched_req in requests:
            data = sched_req.data
            self._sub_dosample.append(bool(getattr(data, "subtalker_dosample", True)))
            self._sub_temperature.append(
                float(getattr(data, "subtalker_temperature", 0.9))
            )
            self._sub_top_p.append(float(getattr(data, "subtalker_top_p", 1.0)))
            self._sub_top_k.append(int(getattr(data, "subtalker_top_k", 50)))
            seed = getattr(data.req.sampling_params, "sampling_seed", None)
            if seed is None:
                self._sub_generators.append(None)
                continue

            seed = int(seed)
            generator = getattr(data, "_subtalker_generator", None)
            if (
                not isinstance(generator, torch.Generator)
                or getattr(data, "_subtalker_generator_seed", None) != seed
                or getattr(data, "_subtalker_generator_device", None)
                != generator_device
            ):
                generator = torch.Generator(device=generator_device)
                generator.manual_seed(seed)
                data._subtalker_generator = generator
                data._subtalker_generator_seed = seed
                data._subtalker_generator_device = generator_device
            self._sub_generators.append(generator)

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        result_codes, summed_embeddings = self._code_predictor_forward_incremental(
            layer0_codes=layer0_codes,
            talker_hidden=talker_hidden,
        )
        return result_codes, summed_embeddings

    def _code_predictor_forward_incremental(
        self,
        layer0_codes: torch.Tensor,
        talker_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer0_codes.ndim == 1:
            layer0_codes = layer0_codes.unsqueeze(1)
        if talker_hidden.ndim == 2:
            talker_hidden = talker_hidden.unsqueeze(1)

        batch_size, seq_len = layer0_codes.shape
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
                next_code = self._sample_subtalker_token(logits[:, -1, :], layer_idx)
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

    def _sample_subtalker_token(
        self,
        logits: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        del layer_idx
        tokens = []
        for row_idx, row in enumerate(logits):
            if row_idx >= len(self._sub_dosample) or not self._sub_dosample[row_idx]:
                tokens.append(torch.argmax(row))
                continue
            scores = row.float()
            temperature = max(float(self._sub_temperature[row_idx]), 1e-5)
            scores = scores / temperature
            top_k = int(self._sub_top_k[row_idx])
            if top_k > 0 and top_k < scores.numel():
                keep = torch.topk(scores, top_k).indices
                mask = torch.full_like(scores, -float("inf"))
                mask[keep] = scores[keep]
                scores = mask
            probs = torch.softmax(scores, dim=-1)
            generator = (
                self._sub_generators[row_idx]
                if row_idx < len(self._sub_generators)
                else None
            )
            top_p = float(self._sub_top_p[row_idx])
            if 0.0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cdf = torch.cumsum(sorted_probs, dim=-1)
                remove = cdf > top_p
                remove[0] = False
                sorted_probs = sorted_probs.masked_fill(remove, 0)
                sorted_probs = sorted_probs / sorted_probs.sum().clamp_min(1e-12)
                sample = torch.multinomial(sorted_probs, 1, generator=generator)[0]
                tokens.append(sorted_idx[sample])
            else:
                tokens.append(torch.multinomial(probs, 1, generator=generator)[0])
        return torch.stack(tokens).to(dtype=torch.long)

    def _predictor_forward_one_token(
        self,
        *,
        token_embeds: torch.Tensor,
        batch_size: int,
        cache_len: int,
    ) -> torch.Tensor:
        hidden_states = token_embeds
        hidden_size = hidden_states.shape[-1]
        positions = self._predictor_positions[cache_len : cache_len + 1].repeat(
            batch_size
        )
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
        groups = attn.num_heads // attn.num_kv_heads
        cached_k = _repeat_kv(cached_k, groups)
        cached_v = _repeat_kv(cached_v, groups)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            cached_k,
            cached_v,
            is_causal=False,
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
