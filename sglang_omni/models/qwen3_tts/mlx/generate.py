# SPDX-License-Identifier: Apache-2.0
"""Autoregressive MLX speech generation for Qwen3-TTS CustomVoice."""

from __future__ import annotations

from collections.abc import Generator
from itertools import islice
from pathlib import Path

import mlx.core as mx
from sglang.srt.hardware_backend.mlx.kv_cache import ContiguousAttentionKVCache
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from sglang_omni.models.qwen3_tts.mlx.decoder import (
    Qwen3TTSMlxSpeechDecoder,
    load_qwen3_tts_mlx_decoder,
)
from sglang_omni.models.qwen3_tts.mlx.decoder_stream import Qwen3TTSMlxDecoderStream
from sglang_omni.models.qwen3_tts.mlx.model import (
    Qwen3TTSMlxCodePredictor,
    Qwen3TTSMlxTalker,
    load_qwen3_tts_mlx_talker,
)

CODEC_CONTROL_TOKEN_COUNT = 1024


def sample_codec_token(
    logits: mx.array,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float = 1.0,
    seen_tokens: list[int] | None = None,
    eos_token_id: int | None = None,
    key: mx.array | None = None,
) -> mx.array:
    """Sample one token from the final-position logits."""
    row = logits[0, -1, :].astype(mx.float32)
    if eos_token_id is not None:
        eos_logit = row[eos_token_id : eos_token_id + 1]
        row = mx.concatenate(
            [
                row[:-CODEC_CONTROL_TOKEN_COUNT],
                mx.full((CODEC_CONTROL_TOKEN_COUNT,), -float("inf"), dtype=row.dtype),
            ]
        )
        row = mx.put_along_axis(
            row, mx.array([eos_token_id], dtype=mx.int32), eos_logit, axis=0
        )
    else:
        pass
    if seen_tokens and repetition_penalty != 1.0:
        indices = mx.array(sorted(set(seen_tokens)), dtype=mx.int32)
        selected = mx.take(row, indices)
        adjusted = mx.where(
            selected < 0,
            selected * repetition_penalty,
            selected / repetition_penalty,
        )
        row = mx.put_along_axis(row, indices, adjusted, axis=0)
    else:
        pass
    if temperature <= 0:
        return mx.argmax(row).reshape(1, 1).astype(mx.int32)
    else:
        pass
    row = row / temperature
    if top_k > 0 and top_k < row.shape[0]:
        cutoff = mx.sort(row)[-top_k]
        row = mx.where(row >= cutoff, row, -float("inf"))
    else:
        pass
    if eos_token_id is not None:
        row = mx.put_along_axis(
            row,
            mx.array([eos_token_id], dtype=mx.int32),
            eos_logit / temperature,
            axis=0,
        )
    else:
        pass
    if top_p < 1.0:
        order = mx.argsort(-row)
        sorted_logits = mx.take(row, order)
        probabilities = mx.softmax(sorted_logits)
        cumulative = mx.cumsum(probabilities)
        sorted_logits = mx.where(
            cumulative - probabilities < top_p,
            sorted_logits,
            -float("inf"),
        )
        if eos_token_id is not None:
            eos_rank = mx.argmax(order == eos_token_id)
            sorted_logits = mx.put_along_axis(
                sorted_logits,
                eos_rank.reshape(1),
                row[eos_token_id : eos_token_id + 1],
                axis=0,
            )
        else:
            pass
        rank = mx.random.categorical(sorted_logits[None, :], key=key)[0]
        token = order[rank]
    else:
        token = mx.random.categorical(row[None, :], key=key)[0]
    return token.reshape(1, 1).astype(mx.int32)


class Qwen3TTSMlxCodeGenerator:
    """Share talker weights while keeping generation state per request."""

    def __init__(self, model_dir: Path) -> None:
        self.talker: Qwen3TTSMlxTalker
        self.predictor: Qwen3TTSMlxCodePredictor
        self.talker, self.predictor = load_qwen3_tts_mlx_talker(model_dir)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            str(model_dir)
        )

    def generate_codes(
        self,
        *,
        text: str,
        voice: str,
        language: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        seed: int | None = None,
    ) -> Generator[mx.array, None, None]:
        """Yield one complete codec frame at a time."""
        prompt, trailing_text, pad_embedding = self.talker.build_prompt_embeddings(
            self.tokenizer, text=text, voice=voice, language=language
        )
        cache = [
            ContiguousAttentionKVCache(max_seq_len=prompt.shape[1] + max_new_tokens)
            for _ in self.talker.model.layers
        ]
        code_cache = [
            ContiguousAttentionKVCache(
                max_seq_len=self.talker.artifact.talker_config.num_code_groups
            )
            for _ in self.predictor.model.layers
        ]
        seen_tokens: list[int] = []
        random_key = (
            mx.random.key(seed) if seed is not None and temperature > 0 else None
        )
        code_group_count = self.talker.artifact.talker_config.num_code_groups
        eos_token_id = self.talker.artifact.talker_config.codec_eos_token_id
        for step in range(max_new_tokens):
            if random_key is not None:
                sample_keys = mx.random.split(random_key, num=code_group_count + 1)
                random_key = sample_keys[0]
            else:
                sample_keys = None
            logits, hidden = self.talker.forward_embeddings(prompt, cache)
            first_token = sample_codec_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seen_tokens=seen_tokens,
                eos_token_id=eos_token_id,
                key=sample_keys[1] if sample_keys is not None else None,
            )
            mx.eval(first_token)
            token_id = int(first_token.item())
            if token_id == eos_token_id:
                break
            else:
                pass
            seen_tokens.append(token_id)
            code_tokens = [first_token]
            for layer_cache in code_cache:
                layer_cache.reset()
            for code_group in range(
                self.talker.artifact.talker_config.num_code_groups - 1
            ):
                if code_group == 0:
                    code_input = mx.concatenate(
                        [hidden, self.talker.model.embed_tokens(first_token)], axis=1
                    )
                else:
                    code_input = self.predictor.codec_embedding[code_group - 1](
                        code_tokens[-1]
                    )
                code_logits = self.predictor.forward_embeddings(
                    code_input, cache=code_cache, code_group=code_group
                )
                code_tokens.append(
                    sample_codec_token(
                        code_logits,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        key=(
                            sample_keys[code_group + 2]
                            if sample_keys is not None
                            else None
                        ),
                    )
                )
            frame = mx.concatenate(code_tokens, axis=1)
            codec_embedding = self.talker.model.embed_tokens(first_token)
            for code_group, code_token in enumerate(code_tokens[1:]):
                codec_embedding = codec_embedding + self.predictor.codec_embedding[
                    code_group
                ](code_token)
            if step < trailing_text.shape[1]:
                text_embedding = trailing_text[:, step : step + 1, :]
            else:
                text_embedding = pad_embedding
            prompt = text_embedding + codec_embedding
            mx.async_eval(prompt)
            yield frame
        if not seen_tokens:
            raise RuntimeError("Qwen3-TTS MLX generated no speech tokens")
        else:
            pass


class Qwen3TTSMlxGenerator(Qwen3TTSMlxCodeGenerator):
    """Run codec generation and speech decoding for local callers."""

    def __init__(self, model_dir: Path) -> None:
        super().__init__(model_dir)
        self.decoder: Qwen3TTSMlxSpeechDecoder = load_qwen3_tts_mlx_decoder(model_dir)

    def generate(
        self,
        *,
        text: str,
        voice: str,
        language: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ) -> tuple[mx.array, int]:
        """Return a mono waveform and its semantic-token count."""
        frames = list(
            self.generate_codes(
                text=text,
                voice=voice,
                language=language,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        )
        codes = mx.stack(frames, axis=1)
        waveform, lengths = self.decoder.decode(codes)
        mx.eval(waveform, lengths)
        return waveform[0, : int(lengths[0].item())], len(frames)

    def generate_stream(
        self,
        *,
        text: str,
        voice: str,
        language: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        chunk_frames: int,
    ) -> Generator[tuple[mx.array, int], None, None]:
        """Yield waveform chunks and the cumulative semantic-token count."""
        if chunk_frames <= 0:
            raise ValueError("Qwen3-TTS MLX chunk_frames must be positive")
        else:
            pass
        frames = self.generate_codes(
            text=text,
            voice=voice,
            language=language,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        decoder_stream = Qwen3TTSMlxDecoderStream(self.decoder)
        token_count = 0
        # note (Codex): Match offline prefix trimming when zero semantic codes shorten audio.
        pending_waveform: mx.array = mx.zeros((0,))
        while chunk := list(islice(frames, chunk_frames)):
            token_count += len(chunk)
            waveform, lengths = decoder_stream.decode(mx.stack(chunk, axis=1))
            pending_waveform = mx.concatenate([pending_waveform, waveform[0]])
            valid_samples = int(lengths[0].item())
            yield pending_waveform[:valid_samples], token_count
            pending_waveform = pending_waveform[valid_samples:]
