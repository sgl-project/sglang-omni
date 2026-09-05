# SPDX-License-Identifier: Apache-2.0
"""MLX-native Qwen3-TTS prompt construction.

Ports the prompt assembly from ``qwen3_tts/sglang_model.py`` so a request can be
prepared without a Torch copy of the model. On Apple Silicon that is the
difference between one resident model and two.

All three task types share one skeleton:

    [instructions] role(3 text tokens) | codec control prefix | text + codec

The codec control prefix carries think/no-think, an optional language id, the
speaker conditioning, and codec BOS. Each position sums one text-side and one
codec-side embedding, so the two streams stay aligned; whichever stream has
nothing to say at a position contributes its pad embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

# The chat template is fixed, and the slice offsets below follow from it:
# 3 leading role tokens, and 5 trailing tokens on the target template.
ASSISTANT_TEMPLATE = "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
REF_TEMPLATE = "<|im_start|>assistant\n{text}<|im_end|>\n"
INSTRUCT_TEMPLATE = "<|im_start|>user\n{text}<|im_end|>\n"
ROLE_TOKENS = 3
TARGET_TAIL_TOKENS = 5
REF_TAIL_TOKENS = 2


@dataclass
class PromptInputs:
    """What the talker needs to start generating."""

    input_embeds: mx.array
    trailing_text_embeds: mx.array
    pad_embed: mx.array
    ref_codes: mx.array | None = None


class Qwen3TTSMlxPromptBuilder:
    """Builds Qwen3-TTS prompts from text, entirely in MLX."""

    def __init__(self, talker: Any, model_config: Any, tokenizer: Any) -> None:
        self.talker = talker
        self.model_config = model_config
        self.config = model_config.talker_config
        self.tokenizer = tokenizer
        self._check_token_ids()

    def _check_token_ids(self) -> None:
        """Fail loudly on ids that fall outside their embedding table.

        An out-of-range MLX embedding lookup segfaults rather than raising, so a
        mismatched config would otherwise take down the worker with no message.
        """
        config = self.config
        codec_ids = {
            "codec_eos_token_id": config.codec_eos_token_id,
            "codec_think_id": config.codec_think_id,
            "codec_nothink_id": config.codec_nothink_id,
            "codec_think_bos_id": config.codec_think_bos_id,
            "codec_think_eos_id": config.codec_think_eos_id,
            "codec_pad_id": config.codec_pad_id,
            "codec_bos_id": config.codec_bos_id,
        }
        for name, value in (config.codec_language_id or {}).items():
            codec_ids[f"codec_language_id[{name!r}]"] = value
        for name, value in (config.spk_id or {}).items():
            codec_ids[f"spk_id[{name!r}]"] = (
                value[0] if isinstance(value, (list, tuple)) else value
            )
        self._check_range(codec_ids, config.vocab_size, "codec vocab_size")

        self._check_range(
            {
                "tts_bos_token_id": self.model_config.tts_bos_token_id,
                "tts_eos_token_id": self.model_config.tts_eos_token_id,
                "tts_pad_token_id": self.model_config.tts_pad_token_id,
            },
            config.text_vocab_size,
            "text_vocab_size",
        )

    @staticmethod
    def _check_range(ids: dict[str, Any], limit: int, label: str) -> None:
        bad = {
            name: int(value)
            for name, value in ids.items()
            if value is not None and not 0 <= int(value) < int(limit)
        }
        if bad:
            listed = ", ".join(f"{name}={value}" for name, value in sorted(bad.items()))
            raise ValueError(
                f"Qwen3-TTS config has token ids outside {label}={limit}: {listed}"
            )

    # -- token / embedding primitives ------------------------------------

    def _tokenize(self, text: str) -> mx.array:
        return mx.array(self.tokenizer.encode(text), dtype=mx.int32)[None]

    def _text(self, token_ids: mx.array) -> mx.array:
        return self.talker.text_projection(self.talker.get_text_embeddings()(token_ids))

    def _codec(self, token_ids: list[int]) -> mx.array:
        return self.talker.get_input_embeddings()(mx.array([token_ids], dtype=mx.int32))

    def _special_embeds(self) -> tuple[mx.array, mx.array, mx.array]:
        """The TTS bos / eos / pad embeddings, in that order."""
        embeds = self._text(
            mx.array(
                [
                    [
                        self.model_config.tts_bos_token_id,
                        self.model_config.tts_eos_token_id,
                        self.model_config.tts_pad_token_id,
                    ]
                ],
                dtype=mx.int32,
            )
        )
        return embeds[:, 0:1], embeds[:, 1:2], embeds[:, 2:3]

    # -- control prefix ---------------------------------------------------

    def _language_id(self, language: str, voice: str | None = None) -> int | None:
        language_map = self.config.codec_language_id or {}
        if language.lower() != "auto":
            if language.lower() not in language_map:
                supported = ", ".join(sorted(language_map))
                raise ValueError(
                    f"Unsupported Qwen3-TTS language {language!r}. "
                    f"Supported: {supported}"
                )
            return language_map[language.lower()]
        if voice is None:
            return None
        # A dialect voice implies its language even when the caller said auto.
        dialects = self.config.spk_is_dialect or {}
        dialect = dialects.get(voice.lower())
        if isinstance(dialect, str) and dialect:
            return language_map.get(dialect)
        return None

    def _codec_prefill(self, language: str, voice: str | None = None) -> mx.array:
        language_id = self._language_id(language, voice)
        config = self.config
        if language_id is None:
            ids = [
                config.codec_nothink_id,
                config.codec_think_bos_id,
                config.codec_think_eos_id,
            ]
        else:
            ids = [
                config.codec_think_id,
                config.codec_think_bos_id,
                language_id,
                config.codec_think_eos_id,
            ]
        return self._codec(ids)

    def _speaker_suffix(self) -> mx.array:
        return self._codec([self.config.codec_pad_id, self.config.codec_bos_id])

    # -- skeleton ---------------------------------------------------------

    def _conditioned_prefix(
        self,
        input_ids: mx.array,
        codec_input: mx.array,
        bos_embed: mx.array,
        pad_embed: mx.array,
    ) -> mx.array:
        """Role tokens, then the control prefix summed with text padding."""
        role_embed = self._text(input_ids[:, :ROLE_TOKENS])
        pad_run = mx.broadcast_to(
            pad_embed,
            (1, codec_input.shape[1] - 2, pad_embed.shape[-1]),
        )
        text_side = mx.concatenate([pad_run, bos_embed], axis=1)
        return mx.concatenate([role_embed, text_side + codec_input[:, :-1]], axis=1)

    def _finish_text(
        self,
        prefix: mx.array,
        input_ids: mx.array,
        codec_last_embed: mx.array,
        pad_embed: mx.array,
        eos_embed: mx.array,
        non_streaming_mode: bool,
    ) -> tuple[mx.array, mx.array]:
        """Attach the text stream; returns (prefill, trailing text).

        Non-streaming puts the whole text in the prefill, which keeps full text
        context when the codec side is long. Streaming leaves all but the first
        text position for the decode loop to consume one per frame.
        """
        body = input_ids[:, ROLE_TOKENS:-TARGET_TAIL_TOKENS]
        if non_streaming_mode:
            text_all = mx.concatenate([self._text(body), eos_embed], axis=1)
            codec_pad = self._codec([self.config.codec_pad_id])
            return (
                mx.concatenate(
                    [
                        prefix,
                        text_all + codec_pad,
                        pad_embed + self._codec([self.config.codec_bos_id]),
                    ],
                    axis=1,
                ),
                pad_embed,
            )

        first_text = self._text(input_ids[:, ROLE_TOKENS : ROLE_TOKENS + 1])
        prefill = mx.concatenate([prefix, first_text + codec_last_embed], axis=1)
        trailing = mx.concatenate(
            [
                self._text(input_ids[:, ROLE_TOKENS + 1 : -TARGET_TAIL_TOKENS]),
                eos_embed,
            ],
            axis=1,
        )
        return prefill, trailing

    def _apply_instructions(
        self, prefill: mx.array, instructions: str | None
    ) -> mx.array:
        if not instructions:
            return prefill
        instruct_embed = self._text(
            self._tokenize(INSTRUCT_TEMPLATE.format(text=instructions))
        )
        return mx.concatenate([instruct_embed, prefill], axis=1)

    def _assemble(
        self,
        *,
        text: str,
        codec_input: mx.array,
        language: str,
        non_streaming_mode: bool,
        instructions: str | None,
        ref_codes: mx.array | None = None,
    ) -> PromptInputs:
        del language
        input_ids = self._tokenize(ASSISTANT_TEMPLATE.format(text=text))
        bos_embed, eos_embed, pad_embed = self._special_embeds()
        prefix = self._conditioned_prefix(input_ids, codec_input, bos_embed, pad_embed)
        prefill, trailing = self._finish_text(
            prefix,
            input_ids,
            codec_input[:, -1:],
            pad_embed,
            eos_embed,
            non_streaming_mode,
        )
        return PromptInputs(
            input_embeds=self._apply_instructions(prefill, instructions),
            trailing_text_embeds=trailing,
            pad_embed=pad_embed,
            ref_codes=ref_codes,
        )

    # -- task types -------------------------------------------------------

    def build_custom_voice(
        self,
        *,
        text: str,
        voice: str,
        language: str = "auto",
        non_streaming_mode: bool = False,
        instructions: str | None = None,
    ) -> PromptInputs:
        """Prompt for a preset speaker."""
        speaker_ids = self.config.spk_id or {}
        if not speaker_ids:
            raise ValueError(
                "Qwen3-TTS CustomVoice requires a checkpoint with configured spk_id"
            )
        by_lowered = {str(key).lower(): value for key, value in speaker_ids.items()}
        key = voice.lower()
        if key not in by_lowered:
            supported = ", ".join(sorted(str(k) for k in speaker_ids))
            raise ValueError(
                f"Unsupported Qwen3-TTS CustomVoice speaker {voice!r}. "
                f"Supported speakers: {supported}"
            )

        speaker_token = by_lowered[key]
        if isinstance(speaker_token, (list, tuple)):
            speaker_token = speaker_token[0]
        codec_input = mx.concatenate(
            [
                self._codec_prefill(language, voice=key),
                self._codec([int(speaker_token)]),
                self._speaker_suffix(),
            ],
            axis=1,
        )
        return self._assemble(
            text=text,
            codec_input=codec_input,
            language=language,
            non_streaming_mode=non_streaming_mode,
            instructions=instructions,
        )

    def build_voice_design(
        self,
        *,
        text: str,
        instructions: str,
        language: str = "auto",
        non_streaming_mode: bool = False,
    ) -> PromptInputs:
        """Prompt for a voice described in words; instructions are required."""
        if not instructions:
            raise ValueError("Qwen3-TTS VoiceDesign requires instructions")
        codec_input = mx.concatenate(
            [self._codec_prefill(language), self._speaker_suffix()], axis=1
        )
        return self._assemble(
            text=text,
            codec_input=codec_input,
            language=language,
            non_streaming_mode=non_streaming_mode,
            instructions=instructions,
        )

    def build_voice_clone(
        self,
        *,
        text: str,
        ref_codes: mx.array,
        ref_text: str,
        speaker_embed: mx.array | None,
        language: str = "auto",
        instructions: str | None = None,
    ) -> PromptInputs:
        """Prompt for in-context voice cloning.

        Always non-streaming: the reference codec stream is usually longer than
        the text, and interleaving would drop text context.
        """
        config = self.config
        codec_prefix = self._codec_prefill(language)
        if speaker_embed is not None:
            # The speaker encoder runs in float32; without this cast the whole
            # prompt and the talker's KV are promoted with it.
            speaker = speaker_embed.astype(codec_prefix.dtype).reshape(1, 1, -1)
            codec_prefix = mx.concatenate([codec_prefix, speaker], axis=1)
        codec_input = mx.concatenate([codec_prefix, self._speaker_suffix()], axis=1)

        input_ids = self._tokenize(ASSISTANT_TEMPLATE.format(text=text))
        ref_ids = self._tokenize(REF_TEMPLATE.format(text=ref_text))[
            :, ROLE_TOKENS:-REF_TAIL_TOKENS
        ]
        bos_embed, eos_embed, pad_embed = self._special_embeds()

        text_embed = mx.concatenate(
            [
                self._text(
                    mx.concatenate(
                        [ref_ids, input_ids[:, ROLE_TOKENS:-TARGET_TAIL_TOKENS]],
                        axis=1,
                    )
                ),
                eos_embed,
            ],
            axis=1,
        )

        codec_embed = self.talker.get_input_embeddings()(ref_codes[:, 0, :])
        for group in range(config.num_code_groups - 1):
            codec_embed = codec_embed + self.talker.code_predictor.codec_embedding[
                group
            ](ref_codes[:, group + 1, :])
        codec_embed = mx.concatenate(
            [self._codec([config.codec_bos_id]), codec_embed], axis=1
        )

        icl_embed = mx.concatenate(
            [
                text_embed + self._codec([config.codec_pad_id]),
                codec_embed + pad_embed,
            ],
            axis=1,
        )
        prefix = self._conditioned_prefix(input_ids, codec_input, bos_embed, pad_embed)
        prefill = mx.concatenate([prefix, icl_embed], axis=1)
        return PromptInputs(
            input_embeds=self._apply_instructions(prefill, instructions),
            trailing_text_embeds=pad_embed,
            pad_embed=pad_embed,
            ref_codes=ref_codes,
        )
