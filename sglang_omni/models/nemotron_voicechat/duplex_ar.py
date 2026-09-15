# SPDX-License-Identifier: Apache-2.0
"""Frame fusion over scheduler-owned streaming KV sessions.

A completed unit leaves its sampled token unprocessed. The next unit appends no
IDs and supplies that position's next acoustic/text/code fusion row. Historical
fusion rows are retained for a correct prefill if core replays any prefix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.nemotron_voicechat.model_runner import (
    NemotronVoiceChatModelRunner,
)
from sglang_omni.models.nemotron_voicechat.request_builders import _ar_request
from sglang_omni.models.nemotron_voicechat.talker_model_runner import (
    NemotronVoiceChatTalkerModelRunner,
)
from sglang_omni.scheduling.sglang_backend.ar_session import ARSessionAdapter

logger = logging.getLogger(__name__)


@dataclass
class FrameHistory:
    rows: list = field(default_factory=list)
    positions: int = 0
    text_token: int | None = None
    function_token: int | None = None
    codes: object = None
    spoken: list[int] = field(default_factory=list)
    text: str = ""
    forwarded: int = 0
    reused: int = 0
    units: int = 0


def attach_rows(forward_batch, requests):
    pieces = []
    for i, request in enumerate(requests):
        history = request.data.talker_model_inputs["duplex_history"]
        prefix = int(forward_batch.extend_prefix_lens_cpu[i])
        length = int(forward_batch.extend_seq_lens_cpu[i])
        if prefix + length != history.positions:
            raise RuntimeError(
                "VoiceChat fusion history is not aligned with session KV"
            )
        # The common path needs just the final row; never concatenate the
        # entire conversation merely to slice off its last position.
        history.forwarded += length
        history.reused += prefix
        history.units += 1
        remaining = length
        suffix = []
        for block in reversed(history.rows):
            take = min(remaining, block.shape[0])
            suffix.append(block[-take:])
            remaining -= take
            if remaining == 0:
                break
        if remaining:
            raise RuntimeError("VoiceChat fusion history is incomplete")
        pieces.extend(reversed(suffix))
    attach_omni_prefill_inputs(
        forward_batch,
        OmniPrefillInputs(
            input_embeds=torch.cat(pieces), input_embeds_are_projected=True
        ),
    )


class DuplexThinkerRunner(NemotronVoiceChatModelRunner):
    def before_prefill(self, forward_batch, schedule_batch, requests):
        attach_rows(forward_batch, requests)

    def before_decode(self, *args, **kwargs):
        raise RuntimeError("VoiceChat session units must perform exactly one forward")


class DuplexTalkerRunner(NemotronVoiceChatTalkerModelRunner):
    def before_prefill(self, forward_batch, schedule_batch, requests):
        attach_rows(forward_batch, requests)

    def post_prefill(self, result, forward_batch, schedule_batch, requests):
        for index, request in enumerate(requests):
            request.data.talker_model_inputs["duplex_codes"] = self._generate_codes(
                index
            )

    def before_decode(self, *args, **kwargs):
        raise RuntimeError("VoiceChat session units must perform exactly one forward")


class FrameAdapter(ARSessionAdapter):
    def __init__(self, runner, *, context_length):
        self.runner = runner
        self.context_length = context_length
        self.states = {}

    def open(self, ref, request):
        if request.params.get("instructions"):
            raise ValueError("VoiceChat currently uses the checkpoint system prompt")
        self.states[ref.session_id] = FrameHistory()

    def close(self, ref):
        state = self.states.pop(ref.session_id, None)
        if state is not None:
            logger.info(
                "VoiceChat %s session closed: units=%d forwarded_positions=%d reused_prefix_positions=%d",
                type(self).__name__,
                state.units,
                state.forwarded,
                state.reused,
            )

    def finish_input(self, ref, payload):
        if payload.data.get("eos") and payload.data.get("acoustic") is None:
            return payload
        return None

    def request(self, ref, payload, opening, rows, vocab_size):
        state = self.states[ref.session_id]
        length = rows.shape[0]
        if state.positions + length + 1 > self.context_length:
            raise ValueError(
                "VoiceChat session context limit reached; start a new session"
            )
        ids = opening if state.positions == 0 else []
        state.rows.append(rows.detach())
        state.positions += length
        data = _ar_request(
            payload, input_ids=ids, max_new_tokens=1, vocab_size=vocab_size
        )
        data.talker_model_inputs["duplex_history"] = state
        data.pending_stream_tokens = []
        return data


class ThinkerAdapter(FrameAdapter):
    def __init__(self, runner, *, prompt_ids, pad_id, tokenizer, context_length):
        super().__init__(runner, context_length=context_length)
        self.prompt_ids, self.pad_id, self.tokenizer = prompt_ids, pad_id, tokenizer
        self.silent = set(tokenizer.all_special_ids) | {
            tokenizer.convert_tokens_to_ids(t) for t in ("<s>", "</s>")
        }

    @torch.inference_mode()
    def build(self, ref, chunk, payload):
        state = self.states[ref.session_id]
        model = self.runner.model
        emb = model.llm.get_input_embeddings()
        acoustic = payload.data["acoustic"].to(emb.weight).reshape(1, -1)
        if state.positions == 0:
            opening = [*self.prompt_ids, self.pad_id]
            ids = torch.tensor(opening, device=emb.weight.device)
            pad = emb(torch.full_like(ids, self.pad_id))
            heard = torch.cat([emb(ids[:-1]), acoustic])
            rows = model.fusion(heard, pad, pad)
        else:
            opening = []
            text, function = emb(
                torch.tensor(
                    [state.text_token, state.function_token], device=emb.weight.device
                )
            )
            rows = model.fusion(acoustic, text.reshape(1, -1), function.reshape(1, -1))
        return self.request(ref, payload, opening, rows, model.llm.config.vocab_size)

    def result(self, ref, data):
        state = self.states[ref.session_id]
        state.text_token = int(data.output_ids[-1])
        state.function_token = int(data.extra_model_outputs["function_ids"][-1])
        delta = ""
        if state.text_token not in self.silent:
            state.spoken.append(state.text_token)
            decoded = self.tokenizer.decode(state.spoken)
            # Do not publish incomplete UTF-8 byte fallback tokens.
            if not decoded.endswith("\ufffd"):
                if not decoded.startswith(state.text):
                    raise RuntimeError(
                        "VoiceChat detokenization revised committed text"
                    )
                delta = decoded[len(state.text) :]
                state.text = decoded
        payload = data.stage_payload
        payload.data.update(
            text_token=state.text_token, function_token=state.function_token, text=delta
        )
        return payload


class TalkerAdapter(FrameAdapter):
    @torch.inference_mode()
    def build(self, ref, chunk, payload):
        state = self.states[ref.session_id]
        runner = self.runner
        previous = runner._pad_codes() if state.codes is None else state.codes
        row = runner._step_row(previous, int(payload.data["text_token"]))
        if state.positions == 0:
            rows = torch.cat([runner._warmup(), row])
            opening = [0] * rows.shape[0]
        else:
            rows, opening = row, []
        return self.request(
            ref,
            payload,
            opening,
            rows.to(runner.model._fusion_buffer.dtype),
            runner.model.config.vocab_size,
        )

    def result(self, ref, data):
        state = self.states[ref.session_id]
        state.codes = data.talker_model_inputs["duplex_codes"]
        payload = data.stage_payload
        payload.data.update(codes=state.codes.cpu())
        return payload
