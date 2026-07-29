# SPDX-License-Identifier: Apache-2.0
"""In-process MiniCPM-o 4.5 speech-generation side component.

This module contains only the model-side pieces required after the SGLang
Qwen3 runner has produced text token ids and their last hidden states:

* the small Llama-based MiniCPM TTS model stored under ``tts.``;
* per-duplex-session TTS KV and position state; and
* the stateful ``assets/token2wav`` streaming decoder.

It intentionally has no subprocess, RPC, socket, or dependency on the demo
repository.  The implementation follows the Apache-2.0 checkpoint runtime's
condition construction and chunk-generation arithmetic, while keeping state
ownership explicit. The current Token2wav provider's voice-prompt cache is
still process-global, so the native stage deliberately admits one session.
"""

from __future__ import annotations

import json
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from transformers import LlamaConfig, LlamaModel

from sglang_omni.models.weight_loader import (
    load_module,
    resolve_dtype,
    resolve_model_path,
)

logger = logging.getLogger(__name__)

_TOKEN2WAV_ASSET_SUBDIR = Path("assets") / "token2wav"
_DEFAULT_SILENCE_TOKEN_ID = 4218
_DEFAULT_SILENCE_PREFIX_LENGTH = 3
_DEFAULT_CODEC_CHUNK_SIZE = 25


@dataclass(frozen=True)
class MiniCPMTTSArchitectureConfig:
    """The checkpoint fields needed to construct the local TTS network."""

    llm_dim: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    max_position_embeddings: int
    num_audio_tokens: int
    num_text_tokens: int
    num_vq: int
    audio_bos_token_id: int
    projector_type: str = "mlp"
    hidden_act: str = "silu"
    llm_intermediate_size: int = 768
    attn_implementation: str = "sdpa"

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "MiniCPMTTSArchitectureConfig":
        required = (
            "llm_dim",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "max_position_embeddings",
            "num_audio_tokens",
            "num_text_tokens",
            "num_vq",
            "audio_bos_token_id",
        )
        missing = [name for name in required if name not in config]
        if missing:
            raise ValueError(
                "MiniCPM-o checkpoint tts_config is missing required fields: "
                + ", ".join(missing)
            )
        return cls(
            **{name: int(config[name]) for name in required},
            projector_type=str(config.get("projector_type", "mlp")),
            hidden_act=str(config.get("hidden_act", "silu")),
            llm_intermediate_size=int(config.get("llm_intermediate_size", 768)),
            attn_implementation=str(config.get("attn_implementation", "sdpa")),
        )


class _MLPProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_dim, out_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(hidden_states)))


class _MiniCPMProjector(nn.Module):
    def __init__(self, config: MiniCPMTTSArchitectureConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.llm_dim, config.llm_intermediate_size, bias=True
        )
        self.up_proj = nn.Linear(
            config.llm_dim, config.llm_intermediate_size, bias=True
        )
        self.down_proj = nn.Linear(
            config.llm_intermediate_size, config.hidden_size, bias=True
        )
        if config.hidden_act != "silu":
            raise ValueError(
                "MiniCPM-o TTS minicpm projector currently supports hidden_act='silu' "
                f"only, got {config.hidden_act!r}"
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


def _create_semantic_projector(
    config: MiniCPMTTSArchitectureConfig,
) -> nn.Module:
    if config.projector_type == "mlp":
        return _MLPProjector(config.llm_dim, config.hidden_size)
    if config.projector_type == "minicpm":
        return _MiniCPMProjector(config)
    if config.projector_type == "default":
        return nn.Linear(config.llm_dim, config.hidden_size, bias=False)
    raise ValueError(
        f"Unsupported MiniCPM-o TTS projector_type: {config.projector_type!r}"
    )


class MiniCPMTTS(nn.Module):
    """Minimal checkpoint-compatible MiniCPM-o 4.5 TTS network.

    Speaker projection and non-streaming training helpers are deliberately not
    constructed.  Duplex inference conditions the TTS model on the main LLM's
    token embedding plus projected hidden state, then feeds audio codes through
    the first codebook embedding.  ``load_module(..., strict=False)`` therefore
    ignores only the unused ``projector_spk`` checkpoint tensors.
    """

    def __init__(self, config: MiniCPMTTSArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        llama_config = LlamaConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=config.num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            attn_implementation=config.attn_implementation,
        )
        self.emb_text = nn.Embedding(config.num_text_tokens, config.hidden_size)
        self.model = LlamaModel(llama_config)
        self.projector_semantic = _create_semantic_projector(config)
        self.emb_code = nn.ModuleList(
            [
                nn.Embedding(config.num_audio_tokens, config.hidden_size)
                for _ in range(config.num_vq)
            ]
        )
        self.head_code = nn.ModuleList(
            [
                weight_norm(
                    nn.Linear(
                        config.hidden_size,
                        config.num_audio_tokens,
                        bias=False,
                    ),
                    name="weight",
                )
                for _ in range(config.num_vq)
            ]
        )

    @property
    def device(self) -> torch.device:
        return self.emb_text.weight.device

    def build_condition(
        self,
        token_ids: Sequence[int],
        hidden_states: torch.Tensor | Sequence[torch.Tensor] | None,
    ) -> torch.Tensor:
        """Build ``token_embed + normalized(projected_hidden) + audio_bos``.

        The number and order of hidden-state rows must exactly match the token
        ids.  This assertion is important in the duplex path: using a hidden
        state from the terminator for a different text token silently corrupts
        the speech condition.
        """

        token_ids = [int(token_id) for token_id in token_ids]
        if not token_ids:
            if hidden_states is not None:
                hidden = _flatten_hidden_states(hidden_states)
                if hidden.shape[0] != 0:
                    raise ValueError(
                        "hidden_states must be empty when token_ids is empty"
                    )
            audio_bos = self.emb_text(
                torch.tensor(
                    [self.config.audio_bos_token_id],
                    dtype=torch.long,
                    device=self.device,
                )
            )
            return audio_bos.unsqueeze(0)

        if hidden_states is None:
            raise ValueError("hidden_states are required when token_ids is non-empty")
        hidden = _flatten_hidden_states(hidden_states)
        if hidden.shape[0] != len(token_ids):
            raise ValueError(
                "MiniCPM-o TTS condition alignment mismatch: "
                f"{len(token_ids)} token ids but {hidden.shape[0]} hidden rows"
            )

        token_tensor = torch.tensor(
            token_ids,
            dtype=torch.long,
            device=self.device,
        )
        token_embeds = self.emb_text(token_tensor)
        projector_parameter = next(self.projector_semantic.parameters())
        hidden = hidden.to(
            device=projector_parameter.device,
            dtype=projector_parameter.dtype,
        )
        semantic = self.projector_semantic(hidden)
        semantic = F.normalize(semantic, p=2, dim=-1)
        condition = token_embeds + semantic.to(dtype=token_embeds.dtype)

        audio_bos = self.emb_text(
            torch.tensor(
                [self.config.audio_bos_token_id],
                dtype=torch.long,
                device=self.device,
            )
        )
        return torch.cat((condition, audio_bos), dim=0).unsqueeze(0)

    @torch.inference_mode()
    def generate_chunk(
        self,
        *,
        inputs_embeds: torch.Tensor,
        temperature: float | torch.Tensor,
        repetition_penalty: float,
        eos_token: int | torch.Tensor,
        force_no_stop: bool = False,
        max_new_tokens: int = 26,
        min_new_tokens: int = 0,
        past_key_values: Any = None,
        text_start_pos: int = 0,
    ) -> tuple[torch.Tensor, Any]:
        """Autoregressively sample one streaming audio-code chunk.

        The returned KV ends immediately after the returned codes.  As in the
        official implementation, the newest sampled candidate is deliberately
        omitted: EOS is not emitted, while on a length stop the un-prefilled
        candidate is not carried into the next chunk.
        """

        if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
            raise ValueError(
                "MiniCPM-o TTS chunk generation requires inputs_embeds [1, T, H]"
            )
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if min_new_tokens < 0:
            raise ValueError("min_new_tokens must be non-negative")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")

        device = inputs_embeds.device
        temperature_tensor = torch.as_tensor(
            temperature,
            dtype=torch.float32,
            device=device,
        ).reshape(-1)
        if temperature_tensor.numel() == 1:
            temperature_tensor = temperature_tensor.expand(self.config.num_vq)
        elif temperature_tensor.numel() != self.config.num_vq:
            raise ValueError(
                "temperature must be scalar or contain one value per codebook"
            )
        if torch.any(temperature_tensor <= 0):
            raise ValueError("temperature must be positive")
        temperature_tensor = temperature_tensor.reshape(-1, 1)

        eos_tensor = torch.as_tensor(eos_token, dtype=torch.long, device=device)
        eos_ids = eos_tensor.reshape(-1)
        finish = torch.zeros(1, dtype=torch.bool, device=device)
        condition_length = int(inputs_embeds.shape[1])
        sampled = torch.zeros(
            (1, max_new_tokens, self.config.num_vq),
            dtype=torch.long,
            device=device,
        )

        last_step = 0
        for step in range(max_new_tokens):
            last_step = step
            if step == 0:
                step_embeds = inputs_embeds
                position_ids = torch.arange(
                    text_start_pos,
                    text_start_pos + condition_length,
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)
            else:
                # The released 4.5 checkpoint has one codebook and conditions
                # subsequent steps on codebook zero.  Keep that behavior for
                # checkpoint compatibility even though heads are represented
                # as a ModuleList.
                step_embeds = self.emb_code[0](sampled[:, step - 1 : step, 0])
                position_ids = torch.tensor(
                    [text_start_pos + condition_length + step - 1],
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)

            output = self.model(
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=step_embeds,
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
            past_key_values = output.past_key_values
            last_hidden = output.last_hidden_state[:, -1:]
            logits = torch.stack(
                [head(last_hidden) for head in self.head_code],
                dim=-1,
            )
            # [batch, seq=1, vocab, vq] -> [batch*vq, vocab]
            logits = logits[:, -1].float().permute(0, 2, 1)
            logits = logits.reshape(-1, self.config.num_audio_tokens)
            logits = logits / temperature_tensor

            if step > 0 and repetition_penalty != 1.0:
                history = sampled[:, :step].permute(0, 2, 1)
                history = history.reshape(-1, step)
                logits = _apply_repetition_penalty(
                    history,
                    logits,
                    penalty=float(repetition_penalty),
                    past_window=16,
                )

            if force_no_stop or step < min_new_tokens:
                logits[:, eos_ids] = -torch.inf

            probabilities = F.softmax(logits, dim=-1)
            next_ids = torch.multinomial(probabilities, num_samples=1)
            next_ids = next_ids.view(1, self.config.num_vq)
            finish.logical_or_(next_ids.eq(eos_ids).any(dim=1))
            sampled[:, step] = next_ids
            if finish.all():
                break

        return sampled[:, :last_step], past_key_values


def _flatten_hidden_states(
    hidden_states: torch.Tensor | Sequence[torch.Tensor],
) -> torch.Tensor:
    if isinstance(hidden_states, torch.Tensor):
        hidden = hidden_states
        while hidden.ndim > 2 and hidden.shape[0] == 1:
            hidden = hidden.squeeze(0)
        if hidden.ndim == 1:
            hidden = hidden.unsqueeze(0)
        if hidden.ndim != 2:
            raise ValueError(
                "hidden_states tensor must reduce to [tokens, hidden_size], "
                f"got shape {tuple(hidden_states.shape)}"
            )
        return hidden

    rows: list[torch.Tensor] = []
    for item in hidden_states:
        if not isinstance(item, torch.Tensor):
            raise TypeError("hidden_states entries must be torch.Tensor values")
        row = item
        while row.ndim > 2 and row.shape[0] == 1:
            row = row.squeeze(0)
        if row.ndim == 1:
            row = row.unsqueeze(0)
        if row.ndim != 2:
            raise ValueError(
                "each hidden-state entry must reduce to [tokens, hidden_size], "
                f"got shape {tuple(item.shape)}"
            )
        rows.append(row)
    if not rows:
        # The hidden width is irrelevant for an empty sequence; callers only
        # inspect the row count before producing the audio BOS embedding.
        return torch.empty((0, 0))
    return torch.cat(rows, dim=0)


def _apply_repetition_penalty(
    input_ids: torch.Tensor,
    scores: torch.Tensor,
    *,
    penalty: float,
    past_window: int,
) -> torch.Tensor:
    if input_ids.shape[1] > past_window:
        input_ids = input_ids[:, -past_window:]
    frequency = F.one_hot(input_ids, scores.shape[1]).sum(dim=1)
    alpha = torch.pow(
        torch.as_tensor(penalty, dtype=scores.dtype, device=scores.device),
        frequency,
    )
    return torch.where(scores < 0, scores * alpha, scores / alpha)


def _clone_cache(value: Any) -> Any:
    """Clone nested Token2wav cache structures without importing demo utils."""

    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_cache(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_cache(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_cache(item) for item in value)
    return value


@dataclass
class MiniCPMOTTSSessionState:
    session_id: str
    prompt_wav_path: str
    past_key_values: Any = None
    text_start_pos: int = 0
    token_buffer: list[int] = field(default_factory=list)
    flow_cache_base: Any = None
    hift_cache_base: Any = None
    flow_cache: Any = None
    hift_cache: Any = None
    pre_lookahead: int = 3
    closed: bool = False

    def release(self) -> None:
        self.past_key_values = None
        self.text_start_pos = 0
        self.token_buffer.clear()
        self.flow_cache_base = None
        self.hift_cache_base = None
        self.flow_cache = None
        self.hift_cache = None
        self.closed = True


@dataclass(frozen=True)
class MiniCPMOTTSChunk:
    audio_tokens: torch.Tensor
    waveform: np.ndarray | None
    sample_rate: int
    end_of_turn: bool


class MiniCPMO45TTSRuntime:
    """Session-aware TTS and Token2wav owner for one SGLang stage process."""

    def __init__(
        self,
        tts_model: MiniCPMTTS,
        token2wav: Any,
        *,
        temperature: float = 0.8,
        repetition_penalty: float = 1.05,
        max_new_tokens: int = 26,
        steady_min_new_tokens: int = 26,
        sample_rate: int = 24_000,
        codec_chunk_size: int = _DEFAULT_CODEC_CHUNK_SIZE,
        silence_token_id: int = _DEFAULT_SILENCE_TOKEN_ID,
        silence_prefix_length: int = _DEFAULT_SILENCE_PREFIX_LENGTH,
        default_prompt_wav_path: str | None = None,
        owns_token2wav: bool = False,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not 0 <= steady_min_new_tokens <= max_new_tokens:
            raise ValueError(
                "steady_min_new_tokens must be between zero and max_new_tokens"
            )
        if codec_chunk_size <= 0:
            raise ValueError("codec_chunk_size must be positive")
        if silence_prefix_length < 0:
            raise ValueError("silence_prefix_length must be non-negative")

        self.tts_model: MiniCPMTTS | None = tts_model
        self.token2wav: Any | None = token2wav
        self.temperature = float(temperature)
        self.repetition_penalty = float(repetition_penalty)
        self.max_new_tokens = int(max_new_tokens)
        self.steady_min_new_tokens = int(steady_min_new_tokens)
        self.sample_rate = int(sample_rate)
        self.codec_chunk_size = int(codec_chunk_size)
        self.silence_token_id = int(silence_token_id)
        self.silence_prefix_length = int(silence_prefix_length)
        self.default_prompt_wav_path = default_prompt_wav_path
        self._owns_token2wav = bool(owns_token2wav)
        self._sessions: dict[str, MiniCPMOTTSSessionState] = {}
        # The Llama side model and Token2wav both use mutable device state.
        # SGLang invokes this owner from one stage process; the lock also makes
        # accidental cross-thread calls deterministic and cache-safe.
        self._model_lock = threading.RLock()
        self._closed = False

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        revision: str | None = None,
        device: str | torch.device = "cuda",
        dtype: str | torch.dtype | None = None,
        local_files_only: bool = False,
        enable_float16: bool = False,
        n_timesteps: int = 10,
        temperature: float = 0.8,
        repetition_penalty: float = 1.05,
        max_new_tokens: int = 26,
        steady_min_new_tokens: int = 26,
        sample_rate: int = 24_000,
        token2wav_factory: Callable[..., Any] | None = None,
    ) -> "MiniCPMO45TTSRuntime":
        """Load only ``tts.`` weights and the checkpoint's Token2wav assets."""

        resolved_path = resolve_model_path(
            model_path,
            local_files_only=local_files_only,
            revision=revision,
        )
        config_path = resolved_path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"MiniCPM-o checkpoint config.json not found under {resolved_path}"
            )
        with config_path.open("r", encoding="utf-8") as config_file:
            checkpoint_config = json.load(config_file)
        tts_config_dict = checkpoint_config.get("tts_config")
        if not isinstance(tts_config_dict, Mapping):
            raise ValueError(
                f"MiniCPM-o checkpoint {config_path} has no object-valued tts_config"
            )
        architecture = MiniCPMTTSArchitectureConfig.from_mapping(tts_config_dict)

        # Avoid first constructing a full FP32 copy before assigning checkpoint
        # tensors.  The project already depends on accelerate for this pattern.
        from accelerate import init_empty_weights

        with init_empty_weights():
            tts_model = MiniCPMTTS(architecture)
        tts_model = load_module(
            tts_model,
            str(resolved_path),
            prefix="tts.",
            dtype=resolve_dtype(dtype),
            device=device,
            # The minimal duplex model intentionally omits projector_spk.
            strict=False,
            require_all_module_keys=True,
            allowed_unexpected_prefixes=("projector_spk.",),
        )

        asset_dir = _resolve_token2wav_asset_dir(
            model_path,
            resolved_path=resolved_path,
            local_files_only=local_files_only,
            revision=revision,
        )
        if token2wav_factory is None:
            token2wav_factory = _import_token2wav()
        token2wav = token2wav_factory(
            str(asset_dir),
            float16=bool(enable_float16),
            n_timesteps=int(n_timesteps),
        )
        default_prompt = resolved_path / "assets" / "HT_ref_audio.wav"
        return cls(
            tts_model,
            token2wav,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            steady_min_new_tokens=steady_min_new_tokens,
            sample_rate=sample_rate,
            default_prompt_wav_path=(
                str(default_prompt) if default_prompt.is_file() else None
            ),
            owns_token2wav=True,
        )

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    def has_session(self, session_id: str) -> bool:
        return session_id in self._sessions

    def open_session(
        self,
        session_id: str,
        *,
        prompt_wav_path: str | None = None,
    ) -> MiniCPMOTTSSessionState:
        """Create isolated Token2wav cache and TTS KV state for a session."""

        self._ensure_open()
        if not session_id:
            raise ValueError("session_id must be non-empty")
        prompt_wav_path = prompt_wav_path or self.default_prompt_wav_path
        if not prompt_wav_path:
            raise ValueError(
                "prompt_wav_path is required to initialize MiniCPM-o Token2wav; "
                "provide one explicitly or keep assets/HT_ref_audio.wav in the "
                "checkpoint snapshot"
            )
        if not Path(prompt_wav_path).is_file():
            raise FileNotFoundError(
                f"MiniCPM-o TTS prompt WAV not found: {prompt_wav_path}"
            )
        with self._model_lock:
            if session_id in self._sessions:
                raise ValueError(f"MiniCPM-o TTS session already exists: {session_id}")
            if self._sessions:
                raise RuntimeError(
                    "stepaudio2-minicpmo 0.1.1 keeps its voice-prompt cache "
                    "process-global; MiniCPM-o TTS currently supports one "
                    "active session"
                )
            token2wav = self._require_token2wav()
            token2wav.cache = None
            try:
                flow_cache, hift_cache = token2wav.set_stream_cache(prompt_wav_path)
                pre_lookahead = int(
                    getattr(
                        getattr(token2wav, "flow", None),
                        "pre_lookahead_len",
                        3,
                    )
                )
                flow_base = _clone_cache(flow_cache)
                hift_base = _clone_cache(hift_cache)
            finally:
                # Also detach partial state if voice-prompt preparation fails.
                token2wav.stream_cache = None
                token2wav.hift_cache_dict = None
            state = MiniCPMOTTSSessionState(
                session_id=session_id,
                prompt_wav_path=prompt_wav_path,
                token_buffer=self._silence_prefix(),
                flow_cache_base=flow_base,
                hift_cache_base=hift_base,
                flow_cache=_clone_cache(flow_base),
                hift_cache=_clone_cache(hift_base),
                pre_lookahead=pre_lookahead,
            )
            self._sessions[session_id] = state
            return state

    def build_condition(
        self,
        token_ids: Sequence[int],
        hidden_states: torch.Tensor | Sequence[torch.Tensor] | None,
    ) -> torch.Tensor:
        self._ensure_open()
        return self._require_tts_model().build_condition(token_ids, hidden_states)

    def synthesize(
        self,
        session_id: str,
        token_ids: Sequence[int],
        hidden_states: torch.Tensor | Sequence[torch.Tensor] | None,
        *,
        end_of_turn: bool = False,
    ) -> MiniCPMOTTSChunk:
        """Generate and decode one duplex unit for ``session_id``.

        On a turn boundary, Token2wav is flushed before either its cache or the
        TTS KV is reset.  This ordering is part of the model contract: resetting
        first discards the lookahead codes buffered by the previous unit.
        """

        self._ensure_open()
        with self._model_lock:
            state = self._get_session(session_id)
            tts_model = self._require_tts_model()
            condition = tts_model.build_condition(token_ids, hidden_states)
            first_chunk = state.text_start_pos == 0
            min_new_tokens = (
                0 if first_chunk or end_of_turn else self.steady_min_new_tokens
            )
            try:
                audio_tokens, next_kv = tts_model.generate_chunk(
                    inputs_embeds=condition,
                    temperature=self.temperature,
                    repetition_penalty=self.repetition_penalty,
                    eos_token=tts_model.config.num_audio_tokens - 1,
                    force_no_stop=False,
                    max_new_tokens=self.max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    past_key_values=state.past_key_values,
                    text_start_pos=state.text_start_pos,
                )
                waveform = self._decode_audio_tokens(
                    state,
                    audio_tokens,
                    is_last_chunk=end_of_turn,
                    force_flush=first_chunk,
                )
            except Exception:
                # Generation can update a DynamicCache in place and Token2wav
                # can consume only part of a buffer before failing.  Neither is
                # retryable from the old position, so fail closed at a clean
                # turn boundary.
                self._reset_turn_state(state)
                raise

            if end_of_turn:
                # Keep this after _decode_audio_tokens.  Resetting first loses
                # codes held for Token2wav lookahead.
                self._reset_turn_state(state)
            else:
                state.past_key_values = next_kv
                state.text_start_pos += int(condition.shape[1]) + int(
                    audio_tokens.shape[1]
                )

            return MiniCPMOTTSChunk(
                audio_tokens=audio_tokens,
                waveform=waveform,
                sample_rate=self.sample_rate,
                end_of_turn=bool(end_of_turn),
            )

    def interrupt_session(
        self,
        session_id: str,
        *,
        flush: bool = True,
    ) -> np.ndarray | None:
        """End the active speech turn and reset its side state.

        ``flush=True`` preserves already-generated lookahead audio.  A caller
        that needs a hard, low-latency cut may pass ``flush=False``; in either
        case the next unit starts with empty TTS KV and pristine Token2wav
        caches.  An untouched session does not emit its silence prefix.
        """

        self._ensure_open()
        with self._model_lock:
            state = self._get_session(session_id)
            has_generated_audio = (
                state.text_start_pos > 0
                or len(state.token_buffer) > self.silence_prefix_length
            )
            waveform = None
            try:
                if flush and has_generated_audio:
                    waveform = self._decode_audio_tokens(
                        state,
                        torch.empty(0, dtype=torch.long),
                        is_last_chunk=True,
                        force_flush=False,
                    )
            finally:
                self._reset_turn_state(state)
            return waveform

    def close_session(self, session_id: str) -> None:
        with self._model_lock:
            state = self._sessions.pop(session_id, None)
            if state is not None:
                state.release()
            if not self._sessions and self.token2wav is not None:
                # stepAudio2 0.1.1 keeps the prepared voice prompt on the
                # provider object. The first release admits one model session,
                # so detach it at the ownership boundary as well as resetting
                # it before the next open.
                self.token2wav.cache = None
                self.token2wav.stream_cache = None
                self.token2wav.hift_cache_dict = None

    def close(self) -> None:
        with self._model_lock:
            if self._closed:
                return
            for state in self._sessions.values():
                state.release()
            self._sessions.clear()
            token2wav = self.token2wav
            self.token2wav = None
            self.tts_model = None
            self._closed = True
            if token2wav is not None:
                token2wav.cache = None
                token2wav.stream_cache = None
                token2wav.hift_cache_dict = None
            if self._owns_token2wav and token2wav is not None:
                close = getattr(token2wav, "close", None)
                if callable(close):
                    close()

    shutdown = close

    def __enter__(self) -> "MiniCPMO45TTSRuntime":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()

    def _decode_audio_tokens(
        self,
        state: MiniCPMOTTSSessionState,
        audio_tokens: torch.Tensor,
        *,
        is_last_chunk: bool,
        force_flush: bool,
    ) -> np.ndarray | None:
        token_ids = (
            audio_tokens.detach()
            .reshape(-1)
            .to(device="cpu", dtype=torch.long)
            .tolist()
        )
        state.token_buffer.extend(int(token_id) for token_id in token_ids)
        pcm_chunks: list[bytes] = []

        with self._activated_token2wav_state(state) as token2wav:
            if force_flush:
                minimum = state.pre_lookahead + 5
                while len(state.token_buffer) >= minimum:
                    amount = min(
                        self.codec_chunk_size + state.pre_lookahead,
                        len(state.token_buffer),
                    )
                    pcm_chunks.append(
                        _require_pcm_bytes(
                            token2wav.stream(
                                state.token_buffer[:amount],
                                prompt_wav=state.prompt_wav_path,
                            )
                        )
                    )
                    consumed = min(
                        self.codec_chunk_size,
                        amount - state.pre_lookahead,
                    )
                    del state.token_buffer[:consumed]
            else:
                amount = self.codec_chunk_size + state.pre_lookahead
                while len(state.token_buffer) >= amount:
                    pcm_chunks.append(
                        _require_pcm_bytes(
                            token2wav.stream(
                                state.token_buffer[:amount],
                                prompt_wav=state.prompt_wav_path,
                            )
                        )
                    )
                    del state.token_buffer[: self.codec_chunk_size]

            if is_last_chunk and state.token_buffer:
                pcm_chunks.append(
                    _require_pcm_bytes(
                        token2wav.stream(
                            state.token_buffer,
                            prompt_wav=state.prompt_wav_path,
                            last_chunk=True,
                        )
                    )
                )
                state.token_buffer.clear()

        if not pcm_chunks:
            return None
        pcm = b"".join(pcm_chunks)
        if not pcm:
            return None
        waveform = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if not is_last_chunk and waveform.shape[0] < self.sample_rate:
            waveform = np.pad(
                waveform,
                (self.sample_rate - waveform.shape[0], 0),
                mode="constant",
            )
        return waveform

    @contextmanager
    def _activated_token2wav_state(
        self, state: MiniCPMOTTSSessionState
    ) -> Iterator[Any]:
        token2wav = self._require_token2wav()
        token2wav.stream_cache = state.flow_cache
        token2wav.hift_cache_dict = state.hift_cache
        try:
            yield token2wav
        finally:
            state.flow_cache = token2wav.stream_cache
            state.hift_cache = token2wav.hift_cache_dict
            token2wav.stream_cache = None
            token2wav.hift_cache_dict = None

    def _reset_turn_state(self, state: MiniCPMOTTSSessionState) -> None:
        state.past_key_values = None
        state.text_start_pos = 0
        state.flow_cache = _clone_cache(state.flow_cache_base)
        state.hift_cache = _clone_cache(state.hift_cache_base)
        state.token_buffer = self._silence_prefix()

    def _silence_prefix(self) -> list[int]:
        return [self.silence_token_id] * self.silence_prefix_length

    def _get_session(self, session_id: str) -> MiniCPMOTTSSessionState:
        try:
            state = self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"Unknown MiniCPM-o TTS session: {session_id}") from exc
        if state.closed:
            raise RuntimeError(f"MiniCPM-o TTS session is closed: {session_id}")
        return state

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("MiniCPM-o TTS runtime is closed")

    def _require_tts_model(self) -> MiniCPMTTS:
        if self.tts_model is None:
            raise RuntimeError("MiniCPM-o TTS runtime has no live TTS model")
        return self.tts_model

    def _require_token2wav(self) -> Any:
        if self.token2wav is None:
            raise RuntimeError("MiniCPM-o TTS runtime has no live Token2wav model")
        return self.token2wav


def _require_pcm_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    raise TypeError(
        "Token2wav.stream must return little-endian PCM16 bytes, "
        f"got {type(value).__name__}"
    )


def _import_token2wav() -> Callable[..., Any]:
    try:
        from stepaudio2 import Token2wav
    except ImportError as exc:
        raise ImportError(
            "MiniCPM-o 4.5 audio output requires Token2wav. Install this "
            "project with the compatible extra: `pip install "
            "'sglang-omni[minicpmo-o]'`."
        ) from exc
    return Token2wav


def _resolve_token2wav_asset_dir(
    model_path: str,
    *,
    resolved_path: Path,
    local_files_only: bool,
    revision: str | None,
) -> Path:
    asset_dir = resolved_path / _TOKEN2WAV_ASSET_SUBDIR
    if asset_dir.is_dir():
        return asset_dir

    if Path(model_path).exists() or local_files_only:
        raise FileNotFoundError(
            f"MiniCPM-o Token2wav assets are missing: expected directory {asset_dir}"
        )

    from huggingface_hub import snapshot_download

    refreshed = Path(
        snapshot_download(
            repo_id=model_path,
            revision=revision,
            allow_patterns=[f"{_TOKEN2WAV_ASSET_SUBDIR.as_posix()}/**"],
            local_files_only=False,
        )
    )
    asset_dir = refreshed / _TOKEN2WAV_ASSET_SUBDIR
    if not asset_dir.is_dir():
        raise FileNotFoundError(
            f"MiniCPM-o Token2wav assets were not found in checkpoint {model_path!r}"
        )
    return asset_dir


__all__ = [
    "MiniCPMO45TTSRuntime",
    "MiniCPMTTS",
    "MiniCPMTTSArchitectureConfig",
    "MiniCPMOTTSChunk",
    "MiniCPMOTTSSessionState",
]
