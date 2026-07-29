# SPDX-License-Identifier: Apache-2.0
"""SGLang model-runner hooks for MiniCPM-o 4.5 duplex units."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.minicpmo_4_5.state import (
    EmbeddingSpan,
    MiniCPMOSpecialTokens,
    MiniCPMOUnitRequestData,
)
from sglang_omni.scheduling.sglang_backend import SGLangOutputProcessor
from sglang_omni.scheduling.types import RequestOutput

logger = logging.getLogger(__name__)


class MiniCPMO45OutputProcessor(SGLangOutputProcessor):
    """Capture the last LLM hidden state needed by the duplex TTS conditioner."""

    def __init__(self, *, capture_hidden: bool = True, model: Any = None) -> None:
        super().__init__(capture_hidden=capture_hidden, model=model)


class MiniCPMO45ModelRunner(ModelRunner):
    """Run one finite unit while SGLang owns the long-lived decoder KV.

    Media encoders produce already projected Qwen embeddings.  The request
    keeps an absolute-position ledger so a retracted/chunked prefill can rebuild
    any part of the append-only session correctly.  Decode uses ordinary token
    embeddings and the ordinary SGLang paged-KV path.

    MiniCPM-o's duplex sampler is deliberately model-specific.  In particular
    it samples ``chunk_eos`` once from the raw distribution and, when that draw
    misses, masks ``chunk_eos`` before applying the text sampling policy.  This
    is not expressible as a static ``SamplingParams`` configuration.
    """

    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)
        tokenizer = getattr(tp_worker, "tokenizer", None)
        self.tokenizer = tokenizer
        self.special_tokens: MiniCPMOSpecialTokens | None = None

    def set_tokenizer(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.special_tokens = MiniCPMOSpecialTokens.from_tokenizer(tokenizer)

    def custom_prefill_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del schedule_batch
        forward_batch.input_embeds = self._build_prefill_input_embeds(
            forward_batch, requests
        )
        # MiniCPMO45ForCausalLM consumes this marker and bypasses its generic
        # multimodal routine.  That routine would otherwise overwrite the
        # already projected streaming audio/image embeddings.
        forward_batch.minicpmo_projected_input_embeds = True
        return None

    def _build_prefill_input_embeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> torch.Tensor:
        input_ids = forward_batch.input_ids
        embedding_layer = self.model.get_input_embeddings()
        pieces: list[torch.Tensor] = []
        batch_offset = 0

        for sched_req in requests:
            data = _unit_data(sched_req.data)
            req = data.req
            extend_len = int(req.extend_input_len)
            token_slice = input_ids[batch_offset : batch_offset + extend_len]
            batch_offset += extend_len
            embeds = embedding_layer(token_slice)

            extend_range = getattr(req, "extend_range", None)
            if extend_range is None:
                absolute_start = len(req.prefix_indices)
            else:
                absolute_start = int(extend_range.start)
            absolute_end = absolute_start + extend_len

            spans = data.absolute_embedding_spans
            if data.session_state is not None:
                spans = data.session_state.embedding_spans
            _apply_embedding_spans(
                embeds,
                absolute_start=absolute_start,
                absolute_end=absolute_end,
                spans=spans,
            )
            pieces.append(embeds)

        if batch_offset != int(input_ids.numel()):
            raise RuntimeError(
                "MiniCPM-o prefill input partition mismatch: consumed "
                f"{batch_offset}, received {int(input_ids.numel())}"
            )
        if not pieces:
            return embedding_layer(input_ids)
        return torch.cat(pieces, dim=0)

    def sample_before_post_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        return True

    def sample_before_post_decode(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        return True

    def _sample_next_token_ids(
        self,
        logits_output: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> torch.Tensor:
        del forward_batch, schedule_batch
        if len(requests) != 1:
            raise RuntimeError(
                "MiniCPM-o 4.5 native duplex currently requires batch size 1"
            )
        data = _unit_data(requests[0].data)
        special = self._require_special_tokens()
        logits = logits_output.next_token_logits
        if logits is None or logits.ndim != 2 or logits.shape[0] != 1:
            raise RuntimeError("MiniCPM-o duplex sampler requires logits shaped [1, V]")

        cfg = data.extra_model_outputs.get("duplex_sampling", {})
        max_steps = int(cfg.get("max_new_speak_tokens_per_chunk", 20))
        if data.forced_listen:
            return torch.tensor(
                [special.listen], dtype=torch.long, device=logits.device
            )
        if int(data.generation_steps) >= max_steps - 1:
            return torch.tensor(
                [special.chunk_eos], dtype=torch.long, device=logits.device
            )

        candidate = self._duplex_sample(logits[0], data, cfg, special)
        state = data.session_state
        if (
            candidate == special.listen
            and state is not None
            and not state.current_turn_ended
        ):
            candidate = special.tts_bos

        # The unified Demo rejects the token that crosses the character cap and
        # closes the unit with chunk_eos instead.  The first decision marker is
        # intentionally excluded, matching its j != 0 condition.
        if (
            int(data.generation_steps) > 0
            and candidate not in special.chunk_terminators
        ):
            candidate_ids = [*data.generated_unit_ids, candidate]
            text = self.tokenizer.decode(candidate_ids, skip_special_tokens=True)
            if len(text) >= 28:
                candidate = special.chunk_eos

        return torch.tensor([candidate], dtype=torch.long, device=logits.device)

    def _duplex_sample(
        self,
        row: torch.Tensor,
        data: MiniCPMOUnitRequestData,
        cfg: dict[str, Any],
        special: MiniCPMOSpecialTokens,
    ) -> int:
        mode = str(cfg.get("decode_mode", "sampling"))
        logits = row.float().clone()

        # Stage 1: preserve the checkpoint's raw chunk boundary decision.
        if mode == "greedy":
            raw_candidate = int(torch.argmax(logits).item())
        elif mode == "sampling":
            raw_probs = F.softmax(logits, dim=-1)
            _validate_probs(raw_probs, "raw chunk_eos draw")
            raw_candidate = int(torch.multinomial(raw_probs, 1).item())
        else:
            raise ValueError(f"unsupported MiniCPM-o decode_mode {mode!r}")
        if raw_candidate == special.chunk_eos:
            return raw_candidate

        # Stage 2: text/listen sampling with chunk_eos masked.
        # Match the landed unified Demo exactly. Although nearby comments
        # describe conditional consecutive-chunk suppression, its decoder is
        # initialized with tts_pad permanently forbidden.
        generic_stop_ids: set[int] = set()
        for token_ids in (
            getattr(data.req, "eos_token_ids", None),
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(self.tokenizer, "additional_stop_token_ids", None),
        ):
            if token_ids is None:
                continue
            if isinstance(token_ids, int):
                generic_stop_ids.add(token_ids)
            else:
                generic_stop_ids.update(int(token_id) for token_id in token_ids)
        generic_stop_ids.difference_update(special.chunk_terminators)
        forbidden = {
            special.chunk_eos,
            special.tts_pad,
            *special.bad_token_ids,
            *generic_stop_ids,
        }
        valid_forbidden = [token for token in forbidden if 0 <= token < logits.numel()]
        if valid_forbidden:
            logits[valid_forbidden] = -torch.inf

        penalty = float(cfg.get("text_repetition_penalty", 1.05))
        window = int(cfg.get("text_repetition_window_size", 512))
        state = data.session_state
        history = [] if state is None else state.generated_text_ids[-window:]
        if penalty != 1.0:
            for token_id in set(history):
                if 0 <= token_id < logits.numel():
                    # Preserve MiniCPM-o's model-specific rule rather than the
                    # sign-aware Hugging Face repetition penalty.  The official
                    # duplex decoder scales repeated logits identically for
                    # positive and negative values.
                    if penalty > 1.0:
                        logits[token_id] /= penalty
                    else:
                        logits[token_id] *= 1.0 / penalty

        length_penalty = float(cfg.get("length_penalty", 1.1))
        if length_penalty != 1.0 and 0 <= special.turn_eos < logits.numel():
            value = logits[special.turn_eos]
            logits[special.turn_eos] = (
                value / length_penalty if value > 0 else value * length_penalty
            )

        listen_scale = float(cfg.get("listen_prob_scale", 1.0))
        if listen_scale != 1.0:
            logits[special.listen] *= listen_scale
        listen_top_k = cfg.get("listen_top_k")
        if listen_top_k is not None:
            listen_rank = int((logits > logits[special.listen]).sum().item())
            if listen_rank < int(listen_top_k):
                return special.listen

        if mode == "greedy":
            return int(torch.argmax(logits).item())

        temperature = float(cfg.get("temperature", 0.7))
        if temperature <= 0:
            return int(torch.argmax(logits).item())
        filtered = _top_k_top_p_filter(
            logits / temperature,
            top_k=int(cfg.get("top_k", 20)),
            top_p=float(cfg.get("top_p", 0.8)),
        )
        probs = F.softmax(filtered, dim=-1)
        _validate_probs(probs, "filtered text draw")
        return int(torch.multinomial(probs, 1).item())

    def post_process_outputs(
        self,
        result: Any,
        scheduler_output: Any,
        outputs: dict[str, RequestOutput],
    ) -> None:
        del result
        special = self._require_special_tokens()
        for sched_req in scheduler_output.requests:
            data = _unit_data(sched_req.data)
            req_output = outputs[sched_req.request_id]
            if req_output.data is None:
                continue
            sampled = int(req_output.data)
            hidden = _last_hidden(req_output.extra)

            # At step N the captured hidden belongs to the token sampled at
            # step N-1.  The Demo intentionally skips its first decision marker.
            pending = data.pending_tts_token_id
            if pending is not None and int(data.generation_steps) >= 2:
                if hidden is None:
                    raise RuntimeError(
                        "MiniCPM-o TTS conditioning requires captured LLM hidden state"
                    )
                end_of_turn = pending == special.turn_eos
                data.tts_pairs.append((pending, hidden, end_of_turn))

            if sampled in special.chunk_terminators:
                data.pending_tts_token_id = None
                return

            # generation_steps==0 is the listen/speak decision marker and is
            # excluded from generated text/TTS, exactly as the Demo's j != 0.
            if int(data.generation_steps) > 0:
                data.generated_unit_ids.append(sampled)
            data.pending_tts_token_id = sampled

            state = data.session_state
            if state is not None:
                state.current_turn_ended = sampled == special.turn_eos
                if sampled not in {
                    special.speak,
                    special.tts_bos,
                    special.tts_eos,
                    special.tts_pad,
                    special.turn_eos,
                }:
                    state.generated_text_ids.append(sampled)

    def _require_special_tokens(self) -> MiniCPMOSpecialTokens:
        if self.special_tokens is None:
            if self.tokenizer is None:
                raise RuntimeError("MiniCPM-o model runner tokenizer is not configured")
            self.special_tokens = MiniCPMOSpecialTokens.from_tokenizer(self.tokenizer)
        return self.special_tokens


def _unit_data(value: Any) -> MiniCPMOUnitRequestData:
    if not isinstance(value, MiniCPMOUnitRequestData):
        raise TypeError(
            "MiniCPMO45ModelRunner requires MiniCPMOUnitRequestData, got "
            f"{type(value).__name__}"
        )
    return value


def _apply_embedding_spans(
    embeds: torch.Tensor,
    *,
    absolute_start: int,
    absolute_end: int,
    spans: list[EmbeddingSpan],
) -> None:
    for span in spans:
        overlap_start = max(absolute_start, span.start)
        overlap_end = min(absolute_end, span.end)
        if overlap_start >= overlap_end:
            continue
        dst_start = overlap_start - absolute_start
        dst_end = overlap_end - absolute_start
        src_start = overlap_start - span.start
        src_end = overlap_end - span.start
        embeds[dst_start:dst_end] = span.embedding[src_start:src_end].to(
            device=embeds.device,
            dtype=embeds.dtype,
        )


def _last_hidden(extra: Any) -> torch.Tensor | None:
    if not isinstance(extra, dict):
        return None
    hidden = extra.get("hidden_states")
    if isinstance(hidden, dict):
        # MiniCPM-o captures only the stream hidden by default.  Be defensive
        # when a caller enables auxiliary-layer capture.
        hidden = next(
            (
                value
                for value in reversed(list(hidden.values()))
                if torch.is_tensor(value)
            ),
            None,
        )
    if not torch.is_tensor(hidden):
        return None
    while hidden.ndim > 1 and hidden.shape[0] == 1:
        hidden = hidden[0]
    if hidden.ndim == 2:
        hidden = hidden[-1]
    return hidden.clone()


def _top_k_top_p_filter(
    logits: torch.Tensor, *, top_k: int, top_p: float
) -> torch.Tensor:
    out = logits.clone()
    if top_k > 0 and top_k < out.numel():
        threshold = torch.topk(out, top_k).values[-1]
        out[out < threshold] = -torch.inf
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(out, descending=True)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        out[sorted_indices[remove]] = -torch.inf
    return out


def _validate_probs(probs: torch.Tensor, context: str) -> None:
    if not torch.isfinite(probs).all() or float(probs.sum().item()) <= 0:
        raise RuntimeError(
            f"MiniCPM-o sampler produced invalid probabilities ({context})"
        )


__all__ = ["MiniCPMO45ModelRunner", "MiniCPMO45OutputProcessor"]
