# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o thinker model runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.minicpm_o.special_tokens import (
    MiniCPMOSpecialTokenIds,
    resolve_special_token_ids,
)
from sglang_omni.models.minicpm_o.thinker_model_runner import (
    MiniCPMOThinkerModelRunner as OfflineThinkerModelRunner,
)
from sglang_omni.models.minicpm_o.thinker_state import DuplexUnitRequestData
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.sglang_backend.request_data import session_prefill_rows


@dataclass
class DuplexSamplerState:
    """Small mutable view consumed by :func:`duplex_sample`."""

    special_tokens: MiniCPMOSpecialTokenIds
    generation_step: int = 0
    force_listen_count: int = 0
    force_listen_counter: int = 0
    generated_history: list[int] = field(default_factory=list)
    current_turn_ended: bool = True
    forbidden_token_ids: set[int] = field(default_factory=set)
    temperature: float = 0.7
    top_k: int = 100
    top_p: float = 0.8
    repetition_penalty: float = 1.05
    listen_prob_scale: float = 1.0
    greedy: bool = False


def top_k_top_p(logits: torch.Tensor, *, top_k: int, top_p: float) -> torch.Tensor:
    filtered = logits.clone()
    if 0 < top_k < filtered.numel():
        threshold = torch.topk(filtered, top_k).values[-1]
        filtered[filtered < threshold] = -torch.inf
    else:
        pass
    if 0.0 < top_p < 1.0:
        values, indices = torch.sort(filtered, descending=True)
        cumulative = torch.cumsum(F.softmax(values, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        filtered[indices[remove]] = -torch.inf
    else:
        pass
    return filtered


def draw(logits: torch.Tensor, *, greedy: bool) -> int:
    if greedy:
        return int(torch.argmax(logits).item())
    else:
        probabilities = F.softmax(logits, dim=-1)
        if not torch.isfinite(probabilities).all() or float(probabilities.sum()) <= 0:
            raise RuntimeError(
                "MiniCPM-o duplex sampler produced invalid probabilities"
            )
        else:
            pass
        return int(torch.multinomial(probabilities, 1).item())


def duplex_sample(logits: torch.Tensor, state: DuplexSamplerState) -> int:
    """Apply MiniCPM-o's two-stage unit sampler to one vocabulary row.

    The official HF utility accidentally points listen_id at tokenizer EOS.
    This hook intentionally scales the actual <|listen|> logit resolved from
    the checkpoint vocabulary.
    """

    if logits.ndim == 2:
        if logits.shape[0] != 1:
            raise ValueError("duplex_sample expects one logits row")
        else:
            pass
        logits = logits[0]
    else:
        pass
    if logits.ndim != 1:
        raise ValueError("duplex_sample expects logits shaped [V] or [1, V]")
    else:
        pass
    special = state.special_tokens

    if state.generation_step >= 19:
        return special.chunk_eos
    else:
        if (
            state.generation_step == 0
            and state.force_listen_counter < state.force_listen_count
        ):
            state.force_listen_counter += 1
            return special.listen
        else:
            row = logits.float().clone()
            # Note (Junnan Li): Stage 1 is the raw model distribution: notably, temperature is not used.
            if draw(row, greedy=state.greedy) == special.chunk_eos:
                return special.chunk_eos
            else:
                forbidden = {
                    special.chunk_eos,
                    *special.forbidden,
                    *state.forbidden_token_ids,
                }
                valid_forbidden = [
                    token for token in forbidden if 0 <= token < row.numel()
                ]
                if valid_forbidden:
                    row[valid_forbidden] = -torch.inf
                else:
                    pass

                penalty = float(state.repetition_penalty)
                if penalty <= 0:
                    raise ValueError("repetition_penalty must be positive")
                else:
                    pass
                if penalty != 1.0:
                    for token_id in set(state.generated_history[-512:]):
                        if 0 <= int(token_id) < row.numel():
                            # Note (Junnan Li): MiniCPM-o is deliberately sign-insensitive here: for >1 it
                            # Note (Junnan Li): divides both positive and negative logits.
                            if penalty > 1.0:
                                row[int(token_id)] /= penalty
                            else:
                                row[int(token_id)] *= 1.0 / penalty
                        else:
                            pass
                else:
                    pass

                if state.listen_prob_scale != 1.0 and 0 <= special.listen < row.numel():
                    row[special.listen] *= float(state.listen_prob_scale)
                else:
                    pass

                if state.greedy or state.temperature <= 0:
                    candidate = int(torch.argmax(row).item())
                else:
                    filtered = top_k_top_p(
                        row / float(state.temperature),
                        top_k=int(state.top_k),
                        top_p=float(state.top_p),
                    )
                    candidate = draw(filtered, greedy=False)

                # Note (Junnan Li): HF constructs StreamDecoder without a special-token exclusion list.
                # Note (Junnan Li): It remembers every decoder-selected token, including controls, before
                # Note (Junnan Li): the duplex loop rewrites a mid-turn listen to tts_bos. Forced listening
                # Note (Junnan Li): and the raw chunk-EOS shortcut above bypass this history update.
                state.generated_history.append(candidate)
                del state.generated_history[:-512]
                if candidate == special.listen and not state.current_turn_ended:
                    candidate = special.tts_bos
                else:
                    pass
                if candidate == special.turn_eos:
                    state.current_turn_ended = True
                elif candidate not in special.chunk_terminators:
                    state.current_turn_ended = False
                else:
                    pass
                return candidate


class MiniCPMOThinkerModelRunner(OfflineThinkerModelRunner):
    """Extend the shared MiniCPM-o runner with duplex sampling and media history."""

    def __init__(
        self, tp_worker: ModelWorker, output_processor: SGLangOutputProcessor
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self.special_tokens: MiniCPMOSpecialTokenIds | None = None

    @staticmethod
    def is_duplex_request(request: Any) -> bool:
        return isinstance(request.data, DuplexUnitRequestData)

    def custom_prefill_forward(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> Any:
        duplex = [request for request in requests if self.is_duplex_request(request)]
        if not duplex:
            return super().custom_prefill_forward(
                forward_batch, schedule_batch, requests
            )
        else:
            if len(duplex) != len(requests):
                raise RuntimeError(
                    "offline and duplex thinker requests cannot share prefill"
                )
            else:
                pass

            embed_tokens = self.embed_tokens
            last_token_id = embed_tokens.num_embeddings - 1
            rows = [
                session_prefill_rows(
                    request.data,
                    lambda token_ids: embed_tokens(token_ids.clamp(0, last_token_id)),
                    forward_batch.input_ids.device,
                )
                for request in requests
            ]
            attach_omni_prefill_inputs(
                forward_batch,
                # note (Junnan Li): MiniCPMOThinkerForCausalLM is already the text-only wrapper, so its ordinary input_embeds argument is the projected-space API.
                OmniPrefillInputs(input_embeds=torch.cat(rows, dim=0)),
            )
            return None

    def sample_before_post_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        if any(self.is_duplex_request(request) for request in requests):
            return True
        else:
            return super().sample_before_post_prefill(
                forward_batch, schedule_batch, requests
            )

    def sample_before_post_decode(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        if any(self.is_duplex_request(request) for request in requests):
            return True
        else:
            return super().sample_before_post_decode(
                forward_batch, schedule_batch, requests
            )

    def lookahead_eligible(self, batch: Any) -> bool:
        # Note (Junnan Li): The custom repetition window is model-local and advances at resolve;
        # Note (Junnan Li): one-step lookahead would therefore sample from stale history.
        return False

    def special_for_data(self, data: DuplexUnitRequestData) -> MiniCPMOSpecialTokenIds:
        if self.special_tokens is None:
            self.special_tokens = resolve_special_token_ids(data.req.tokenizer)
        else:
            pass
        return self.special_tokens

    def sample_next_token_ids(
        self,
        logits_output: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> torch.Tensor:
        duplex_indices = [
            index
            for index, request in enumerate(requests)
            if self.is_duplex_request(request)
        ]
        if not duplex_indices:
            return super().sample_next_token_ids(
                logits_output, forward_batch, schedule_batch, requests
            )
        else:
            logits = logits_output.next_token_logits
            if logits is None or logits.ndim != 2 or logits.shape[0] != len(requests):
                raise RuntimeError(
                    "duplex thinker requires logits shaped [batch, vocab]"
                )
            else:
                pass
            original_logits = logits.clone()
            if len(duplex_indices) == len(requests):
                result = torch.empty(
                    len(requests), dtype=torch.long, device=logits.device
                )
            else:
                result = (
                    super()
                    .sample_next_token_ids(
                        logits_output, forward_batch, schedule_batch, requests
                    )
                    .clone()
                )

            for index in duplex_indices:
                data = requests[index].data
                session = data.thinker_state
                if session is None:
                    raise RuntimeError("duplex request lost its thinker session state")
                else:
                    pass
                cfg = data.sampling_config
                sampler_state = DuplexSamplerState(
                    special_tokens=self.special_for_data(data),
                    generation_step=int(data.generation_steps),
                    force_listen_count=1 if data.forced_listen else 0,
                    force_listen_counter=0,
                    generated_history=session.generated_history,
                    current_turn_ended=session.current_turn_ended,
                    temperature=float(cfg.get("temperature", 0.7)),
                    top_k=int(cfg.get("top_k", 100)),
                    top_p=float(cfg.get("top_p", 0.8)),
                    repetition_penalty=float(cfg.get("repetition_penalty", 1.05)),
                    listen_prob_scale=float(cfg.get("listen_prob_scale", 1.0)),
                    greedy=bool(cfg.get("greedy", False))
                    or str(cfg.get("decode_mode", "")) == "greedy",
                )
                token = duplex_sample(original_logits[index], sampler_state)
                session.current_turn_ended = sampler_state.current_turn_ended
                if data.forced_listen and int(data.generation_steps) == 0:
                    session.force_listen_counter += 1
                else:
                    pass
                result[index] = token
            return result

    # Note (Junnan Li): The base ThinkerModelRunner pins both hooks to NULL (qwen3_omni captures
    # Note (Junnan Li): hidden states via forward hooks instead). MiniCPM-o's talker consumes the
    # Note (Junnan Li): per-step last-layer hidden state through the output processor, so request
    # Note (Junnan Li): capture here. FULL rather than LAST: decode CUDA graphs are captured with
    # Note (Junnan Li): FULL (enable_return_hidden_states) and their can_run gate requires an
    # Note (Junnan Li): exact hidden-mode match; for decode both modes return the same rows, and
    # Note (Junnan Li): post_process_outputs keeps only the last row per request anyway.
    def requested_capture_hidden_mode_prefill(
        self, schedule_batch: Any, requests: list
    ) -> Any:
        del schedule_batch, requests

        return CaptureHiddenMode.FULL

    def requested_capture_hidden_mode_decode(
        self, schedule_batch: Any, requests: list
    ) -> Any:
        del schedule_batch, requests

        return CaptureHiddenMode.FULL

    def post_process_outputs(
        self,
        result: Any,
        scheduler_output: Any,
        outputs: dict[str, Any],
    ) -> None:
        """Accumulate per-step last-layer hidden states for the talker.

        finalize merges extra into extra_model_outputs with a
        plain update, which would keep only the final step's hidden. The
        talker needs the whole sequence, so collect each step's vector into
        hidden_states_seq: entry 0 is the last prompt position (prefill),
        entry i>0 is the position of output token i-1 (its decode-step input).
        """
        offline_outputs = {
            request.request_id: outputs[request.request_id]
            for request in scheduler_output.requests
            if not self.is_duplex_request(request) and request.request_id in outputs
        }
        if offline_outputs:
            super().post_process_outputs(result, scheduler_output, offline_outputs)
        else:
            pass
        for sched_req in scheduler_output.requests:
            req_output = outputs.get(sched_req.request_id)
            data = sched_req.data
            if isinstance(data, DuplexUnitRequestData):
                if req_output is None or req_output.data is None:
                    continue
                else:
                    pass
                sampled = int(req_output.data)
                special = self.special_for_data(data)
                hidden = last_hidden(req_output.extra)
                pending = data.pending_unit_token
                if pending is not None and int(data.generation_steps) >= 2:
                    if hidden is None:
                        raise RuntimeError(
                            "duplex speech conditioning requires thinker hidden state"
                        )
                    else:
                        pass
                    data.unit_pairs.append(
                        (pending, hidden, pending == special.turn_eos)
                    )
                    if len(data.unit_pairs) > 20:
                        del data.unit_pairs[:-20]
                    else:
                        pass
                else:
                    pass
                if sampled in special.chunk_terminators:
                    data.pending_unit_token = None
                    continue
                else:
                    pass
                if int(data.generation_steps) > 0:
                    data.generated_unit_ids.append(sampled)
                else:
                    pass
                data.pending_unit_token = sampled
                state = data.thinker_state
                if state is not None:
                    if sampled == special.turn_eos:
                        state.current_turn_ended = True
                    elif sampled not in special.chunk_terminators:
                        state.current_turn_ended = False
                    else:
                        pass
                else:
                    pass
                continue
            else:
                pass


def last_hidden(extra: Any) -> torch.Tensor | None:
    if not isinstance(extra, dict):
        return None
    else:
        hidden = extra.get("hidden_states")
        if isinstance(hidden, dict):
            hidden = next(
                (
                    value
                    for value in reversed(list(hidden.values()))
                    if torch.is_tensor(value)
                ),
                None,
            )
        else:
            pass
        if not torch.is_tensor(hidden):
            return None
        else:
            while hidden.ndim > 1 and hidden.shape[0] == 1:
                hidden = hidden[0]
            if hidden.ndim == 2:
                hidden = hidden[-1]
            else:
                pass
            return hidden.detach().clone().to("cpu")


__all__ = [
    "DuplexSamplerState",
    "MiniCPMOThinkerModelRunner",
    "duplex_sample",
]
