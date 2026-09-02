# SPDX-License-Identifier: Apache-2.0
"""Fast decode loop for the MiniCPM-o talker (MiniCPMTTS).

The remote code's non-streaming ``MiniCPMTTS.generate`` runs one eager HF
llama step per codec token (~97 tok/s on H100; ~85% of the speech path's
per-request compute). This module reimplements the same loop over the same
weights with a preallocated ``StaticCache`` and, when available, a CUDA graph
capturing the per-step backbone chain (code embedding -> llama decode step ->
code head). Sampling stays eager and reuses the remote code's own
``gen_logits`` processors/warpers, so sampling semantics are unchanged.

Modes:
- ``compat``: op-for-op the remote loop (DynamicCache, HF-built masks).
  Reference for parity tests.
- ``fast``: StaticCache + an explicitly maintained 4D additive mask (HF mask
  construction happens on the host and is not capture-safe; a device-resident
  mask buffer is) + CUDA graph replay. ``SGLANG_OMNI_MINICPMO_TALKER_GRAPH=0``
  keeps the fast loop but replays nothing (eager static-cache steps).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

MINICPMO_TALKER_GRAPH_ENV = "SGLANG_OMNI_MINICPMO_TALKER_GRAPH"


def _graph_env_enabled() -> bool:
    value = os.environ.get(MINICPMO_TALKER_GRAPH_ENV, "1").strip().lower()
    return value not in ("0", "false", "off", "no")


def _mask_min(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).min


class _SamplingState:
    """Per-generate sampling state mirroring the remote loop's semantics."""

    def __init__(
        self,
        *,
        num_vq: int,
        num_audio_tokens: int,
        temperature: float,
        gen_logits_fn: Callable[..., tuple],
        repetition_penalty: float,
        top_p: float,
        top_k: int,
        device: torch.device,
    ) -> None:
        self.temperature = torch.tensor(
            [temperature] * num_vq, dtype=torch.float, device=device
        ).view(-1, 1)
        self.logits_warpers, self.logits_processors = gen_logits_fn(
            num_code=num_audio_tokens,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
        )
        self.num_vq = num_vq

    def sample(
        self,
        logits: torch.Tensor,
        *,
        step: int,
        new_tokens: torch.Tensor,
        min_new_token: int,
        eos_token: torch.Tensor,
    ) -> torch.Tensor:
        """One sampling step: (1, num_audio, num_vq) logits -> (1, num_vq) ids."""
        logits = logits.permute(0, 2, 1).reshape(-1, logits.size(1))
        logits = logits / self.temperature
        if step > 0:
            history = new_tokens[:, 0:step].permute(0, 2, 1)
            logits_token = history.reshape(history.size(0) * history.size(1), -1)
            for processor in self.logits_processors:
                logits = processor(logits_token, logits)
            for warper in self.logits_warpers:
                logits = warper(logits_token, logits)
        if step < min_new_token:
            logits[:, eos_token] = -torch.inf
        scores = F.softmax(logits, dim=-1)
        return torch.multinomial(scores, num_samples=1).view(-1, self.num_vq)


class _GraphedTalkerStep:
    """CUDA graph over one talker decode step for a fixed cache length.

    Per-step inputs reach the captured region through persistent device
    buffers (token ids, position ids, cache position, additive mask row);
    the captured logits tensor is read in place after each replay.
    """

    def __init__(self, backend: "_FastBackend") -> None:
        self.backend = backend
        self.graph = torch.cuda.CUDAGraph()
        self.logits: torch.Tensor | None = None
        try:
            self._capture()
        except Exception:
            # Release the graph's private pool eagerly; the raising object
            # may linger on traceback frames.
            try:
                self.graph.reset()
            except Exception:
                pass
            raise

    @torch.no_grad()
    def _capture(self) -> None:
        backend = self.backend
        device = backend.device
        with torch.cuda.device(device):
            current_stream = torch.cuda.current_stream(device=device)
            warmup_stream = torch.cuda.Stream(device=device)
            warmup_stream.wait_stream(current_stream)
            with torch.cuda.stream(warmup_stream):
                for _ in range(2):
                    backend.step_forward()
            current_stream.wait_stream(warmup_stream)

            capture_stream = torch.cuda.Stream(device=device)
            capture_stream.wait_stream(current_stream)
            with torch.cuda.graph(
                self.graph,
                stream=capture_stream,
                capture_error_mode="thread_local",
            ):
                self.logits = backend.step_forward()
            current_stream.wait_stream(capture_stream)
        if self.logits is None:
            raise RuntimeError("MiniCPM-o talker CUDA graph captured no outputs")

    @torch.no_grad()
    def replay(self) -> torch.Tensor:
        self.graph.replay()
        assert self.logits is not None
        return self.logits


class _FastBackend:
    """StaticCache-backed step executor with an explicit additive mask."""

    def __init__(self, tts: Any, *, max_cache_len: int) -> None:
        from transformers import StaticCache

        self.tts = tts
        self.device = next(tts.model.parameters()).device
        self.dtype = next(tts.model.parameters()).dtype
        self.max_cache_len = max_cache_len
        self.cache = StaticCache(config=tts.model.config, max_cache_len=max_cache_len)
        num_vq = tts.config.num_vq
        self.token_buf = torch.zeros(1, 1, num_vq, dtype=torch.long, device=self.device)
        self.pos_buf = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self.cache_pos_buf = torch.zeros(1, dtype=torch.long, device=self.device)
        self.mask_buf = torch.full(
            (1, 1, 1, max_cache_len),
            _mask_min(self.dtype),
            dtype=self.dtype,
            device=self.device,
        )
        self._graph: _GraphedTalkerStep | None = None
        self._graph_failed = False
        self._used = False

    @torch.no_grad()
    def prefill(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Run the condition through the backbone; returns (1, num_audio, num_vq)."""
        cond_len = inputs_embeds.shape[1]
        if cond_len + 1 >= self.max_cache_len:
            raise ValueError(
                f"talker condition ({cond_len}) does not fit the static cache "
                f"({self.max_cache_len})"
            )
        # transformers v5's StaticLayer.update ignores the cache_position we
        # pass and always writes at arange + cumulative_length, a device
        # tensor the captured graph keeps advancing across replays. Without a
        # per-request reset the write positions climb monotonically: request
        # 2+ conditions on request 1's stale KV, and after ~4 requests the
        # writes walk off the cache end (index_copy_ device assert).
        if self._used:
            self.cache.reset()
        self._used = True
        positions = torch.arange(cond_len, dtype=torch.long, device=self.device)
        self.mask_buf.fill_(_mask_min(self.dtype))
        self.mask_buf[..., :cond_len] = 0.0
        outputs = self.tts.model(
            inputs_embeds=inputs_embeds,
            position_ids=positions.unsqueeze(0),
            cache_position=positions,
            past_key_values=self.cache,
            use_cache=True,
        )
        return self._head_logits(outputs.last_hidden_state[:, -1:])

    def stage_step(self, tokens: torch.Tensor, position: int) -> None:
        """Stage device inputs for the step at absolute cache slot ``position``."""
        self.token_buf.copy_(tokens.view(1, 1, -1))
        self.pos_buf.fill_(position)
        self.cache_pos_buf.fill_(position)
        self.mask_buf[..., position] = 0.0

    @torch.no_grad()
    def step_forward(self) -> torch.Tensor:
        """One decode step from the staged buffers; returns float logits."""
        tts = self.tts
        code_emb = [
            tts.emb_code[q](self.token_buf[:, :, q]) for q in range(tts.config.num_vq)
        ]
        inputs_embeds = torch.stack(code_emb, 3).sum(3)
        outputs = tts.model(
            inputs_embeds=inputs_embeds,
            position_ids=self.pos_buf,
            cache_position=self.cache_pos_buf,
            past_key_values=self.cache,
            attention_mask=self.mask_buf,
            use_cache=True,
        )
        return self._head_logits(outputs.last_hidden_state)

    def _head_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        tts = self.tts
        logits = torch.stack(
            [tts.head_code[q](hidden) for q in range(tts.config.num_vq)], dim=3
        )
        return logits[:, -1].float()

    def step(
        self, tokens: torch.Tensor, position: int, *, use_graph: bool
    ) -> torch.Tensor:
        self.stage_step(tokens, position)
        if use_graph and not self._graph_failed:
            if self._graph is None:
                try:
                    self._graph = _GraphedTalkerStep(self)
                    logger.info("MiniCPM-o talker decode CUDA graph captured")
                except Exception:
                    self._graph_failed = True
                    logger.warning(
                        "MiniCPM-o talker CUDA graph capture failed; "
                        "falling back to eager static-cache decode",
                        exc_info=True,
                    )
            if self._graph is not None:
                return self._graph.replay()
        return self.step_forward()


class TalkerDecodeLoop:
    """Drop-in replacement for the remote ``MiniCPMTTS.generate`` loop."""

    def __init__(
        self,
        tts: Any,
        *,
        gen_logits_fn: Callable[..., tuple],
        max_cache_len: int | None = None,
    ) -> None:
        self._tts = tts
        self._gen_logits = gen_logits_fn
        self._max_cache_len = int(max_cache_len or tts.config.max_position_embeddings)
        self._fast: _FastBackend | None = None

    def _fast_backend(self) -> _FastBackend:
        if self._fast is None:
            self._fast = _FastBackend(self._tts, max_cache_len=self._max_cache_len)
        return self._fast

    @torch.inference_mode()
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        eos_token: torch.Tensor,
        *,
        min_new_token: int,
        max_new_token: int,
        sampling_params: Any,
        mode: str = "fast",
    ) -> torch.Tensor:
        """Generate codec ids for one utterance; returns (1, N, num_vq).

        Loop semantics (ordering of processors/warpers, min-token EOS mask,
        EOS-step exclusion from the returned ids, final-step trim on
        max_new_token) mirror the remote non-streaming ``generate``.
        """
        assert inputs_embeds.shape[0] == 1, "talker decode is batch-1"
        tts = self._tts
        device = inputs_embeds.device
        num_vq = tts.config.num_vq
        eos_token = eos_token.to(device)
        sampling = _SamplingState(
            num_vq=num_vq,
            num_audio_tokens=tts.config.num_audio_tokens,
            temperature=sampling_params.temperature,
            gen_logits_fn=self._gen_logits,
            repetition_penalty=sampling_params.repetition_penalty,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
            device=device,
        )
        new_tokens = torch.zeros(
            1, max_new_token, num_vq, dtype=torch.long, device=device
        )
        finish = torch.zeros(1, dtype=torch.bool, device=device)
        cond_len = inputs_embeds.shape[1]
        use_graph = mode == "fast" and _graph_env_enabled()

        if mode == "fast":
            backend = self._fast_backend()
            logits = backend.prefill(inputs_embeds)
        else:
            backend = None
            past_key_values = None
            logits, past_key_values = self._compat_step(
                inputs_embeds,
                torch.arange(cond_len, dtype=torch.long, device=device).unsqueeze(0),
                past_key_values,
            )

        t = 0
        for t in range(max_new_token):
            if t > 0:
                position = cond_len + t - 1
                if mode == "fast":
                    logits = backend.step(
                        new_tokens[:, t - 1], position, use_graph=use_graph
                    )
                else:
                    code_emb = [
                        tts.emb_code[q](new_tokens[:, t - 1 : t, q])
                        for q in range(num_vq)
                    ]
                    step_embeds = torch.stack(code_emb, 3).sum(3)
                    pos = torch.tensor([[position]], dtype=torch.long, device=device)
                    logits, past_key_values = self._compat_step(
                        step_embeds, pos, past_key_values
                    )
            idx_next = sampling.sample(
                logits,
                step=t,
                new_tokens=new_tokens,
                min_new_token=min_new_token,
                eos_token=eos_token,
            )
            finish.logical_or_(idx_next.eq(eos_token).any(1))
            new_tokens[:, t] = idx_next
            if finish.all():
                break

        if not bool(finish.all()):
            logger.warning(
                "MiniCPM-o talker hit max_new_token=%d without EOS", max_new_token
            )
        return new_tokens[:, 0:t, :]

    def _compat_step(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Any,
    ) -> tuple[torch.Tensor, Any]:
        tts = self._tts
        outputs = tts.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            cache_position=position_ids.view(-1),
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = torch.stack(
            [
                tts.head_code[q](outputs.last_hidden_state)
                for q in range(tts.config.num_vq)
            ],
            dim=3,
        )
        return logits[:, -1].float(), outputs.past_key_values
