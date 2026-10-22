# SPDX-License-Identifier: Apache-2.0
"""One Qwen thinker request per turn over native session KV, with the heard part of an interrupted reply refilled."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from sglang.srt.managers.schedule_batch import MultimodalInputs
from transformers import PreTrainedTokenizerBase
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerConfig,
)

from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.models.qwen3_omni.request_builders import (
    build_sglang_thinker_request,
    compute_mrope_positions,
    extract_thinker_model_inputs,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling.sglang_backend.ar_session import (
    ARSessionAdapter,
    ARSessionPreparation,
    SessionUnit,
)
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData

TRUNCATE_FORMAT = "qwen.truncate"
# One codec frame is 1920 samples at 24 kHz: the code2wav upsample ratios 8·5·4·3 and 2·2 of the checkpoint.
CODEC_FRAME_MS = 80
# The talker prefill template takes the reply's first three text rows; each later codec frame reads one more.
PREFILL_TEXT_ROWS = 3

ModelInputs = dict[str, torch.Tensor | list[torch.Tensor] | bool]


@dataclass(kw_only=True)
class ThinkerUnit:
    """What a unit adds after the retained history; the turn's own prompt starts at prompt_start, after a separator or a refilled turn."""

    turn: int
    ids: list[int]
    model_inputs: ModelInputs
    prompt_start: int
    prefix_model_inputs: ModelInputs


@dataclass(kw_only=True)
class PendingTurn:
    """An aborted unit: it goes back in front of the next turn with its reply cut to what was heard."""

    unit: ThinkerUnit
    generated: list[int]
    heard: int | None = None


@dataclass(kw_only=True)
class ThinkerHistory:
    ids: list[int] = field(default_factory=list)
    model_inputs: ModelInputs = field(default_factory=dict)
    last_turn: int | None = None
    # The last committed turn's prompt and reply start in ids, and the media rows before its prompt.
    last_prompt_start: int = 0
    last_reply_start: int = 0
    model_inputs_before_last_prompt: ModelInputs = field(default_factory=dict)
    unit: ThinkerUnit | None = None
    pending: PendingTurn | None = None
    # Note (Junnan Li): A truncate of a committed reply cuts KV the native session cannot cut, so units re-prefill the whole history until one commits.
    rebuild: bool = False


def heard_tokens(audio_end_ms: float) -> int:
    return PREFILL_TEXT_ROWS + int(audio_end_ms // CODEC_FRAME_MS)


def merge_model_inputs(retained: ModelInputs, incoming: ModelInputs) -> ModelInputs:
    """Append the incoming rows to the retained ones in host memory.

    Retained rows serve every later prefill of the session: the audio sidecar
    indexes them by position, and a rebuild after a truncate prefills them all
    again. Between turns they cost no device memory; the session runner moves
    a prefill's rows to the device.
    """
    merged: ModelInputs = dict(retained)
    for name, value in incoming.items():
        previous = merged.get(name)
        if value is None or name == "attention_mask":
            continue
        elif isinstance(value, torch.Tensor):
            merged[name] = (
                torch.cat([previous, value.cpu()], dim=0)
                if isinstance(previous, torch.Tensor)
                else value.cpu()
            )
        elif isinstance(value, list) and all(
            isinstance(layer, torch.Tensor) for layer in value
        ):
            if previous is None:
                merged[name] = [layer.cpu() for layer in value]
            elif isinstance(previous, list) and len(previous) == len(value):
                merged[name] = [
                    torch.cat([old_layer, new_layer.cpu()], dim=0)
                    for old_layer, new_layer in zip(previous, value)
                ]
            else:
                raise ValueError(f"incompatible retained Qwen {name}")
        elif isinstance(value, bool) and (previous is None or previous == value):
            merged[name] = value
        else:
            raise ValueError(f"unsupported retained Qwen model input {name}")
    return merged


class QwenThinkerSessionAdapter(ARSessionAdapter):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        vocab_size: int,
        thinker_config: Qwen3OmniMoeThinkerConfig,
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.vocab_size: int = vocab_size
        self.thinker_config: Qwen3OmniMoeThinkerConfig = thinker_config
        self.end_token_id: int = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.newline_ids: list[int] = tokenizer.encode("\n", add_special_tokens=False)
        self.states: dict[SessionIdentity, ThinkerHistory] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.states[session_identity] = ThinkerHistory()

    def close(self, session_identity: SessionIdentity) -> None:
        self.states.pop(session_identity, None)

    def control(self, session_identity: SessionIdentity, event: TimedChunk) -> None:
        history = self.states[session_identity]
        if event.format == TRUNCATE_FORMAT:
            # Note (Junnan Li): The talker's one-text-row-per-codec-frame cadence converts the heard audio time into heard reply tokens.
            self.truncate(
                history,
                int(event.payload["turn"]),
                heard_tokens(float(event.payload["t_ms"])),
            )
        else:
            pass

    def truncate(self, history: ThinkerHistory, turn: int | None, heard: int) -> None:
        pending = history.pending
        if pending is not None and pending.unit.turn == turn:
            pending.heard = min(
                heard,
                len(pending.generated) if pending.heard is None else pending.heard,
            )
        elif history.last_turn == turn:
            reply = history.ids[history.last_reply_start :]
            # Note (Junnan Li): A reply already cut ends at its first <|im_end|>, so a later truncate only cuts further.
            generated = (
                reply[: reply.index(self.end_token_id)]
                if self.end_token_id in reply
                else reply
            )
            if heard < len(generated):
                history.ids = [
                    *history.ids[: history.last_reply_start],
                    *generated[:heard],
                    self.end_token_id,
                    *self.newline_ids,
                ]
                history.rebuild = True
            else:
                pass
        else:
            pass

    def prepare_unit(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> ARSessionPreparation:
        history = self.states[session_identity]
        if history.last_turn in chunk.payload["carried_turns"]:
            # Note (Junnan Li): The dialog asks again a turn whose committed reply never reached the client, so that turn leaves the history.
            history.ids = history.ids[: history.last_prompt_start]
            history.model_inputs = history.model_inputs_before_last_prompt
            history.last_turn = None
            history.rebuild = True
        else:
            pass
        return ARSessionPreparation(reset_history=history.rebuild)

    def build(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> SGLangARRequestData:
        history = self.states[session_identity]
        pipeline_state = Qwen3OmniPipelineState.from_dict(payload.data)
        pending = history.pending
        if pending is not None and pending.unit.turn in chunk.payload["carried_turns"]:
            # Note (Junnan Li): The dialog asks the interrupted utterance again in this unit, so only what preceded its prompt is refilled.
            unit_ids = pending.unit.ids[: pending.unit.prompt_start]
            unit_model_inputs = dict(pending.unit.prefix_model_inputs)
        elif pending is not None:
            heard = len(pending.generated) if pending.heard is None else pending.heard
            unit_ids = [
                *pending.unit.ids,
                *pending.generated[:heard],
                self.end_token_id,
                *self.newline_ids,
            ]
            unit_model_inputs = dict(pending.unit.model_inputs)
        else:
            unit_ids = []
            unit_model_inputs = {}
        prompt = dict(pipeline_state.prompt)
        prompt_ids: list[int] = prompt["input_ids"].tolist()
        if history.ids or unit_ids:
            prompt_ids = self.strip_system_prefix(prompt, prompt_ids)
        else:
            pass
        if not unit_ids:
            unit_ids = self.separator(history.ids)
        else:
            pass
        model_inputs = extract_thinker_model_inputs(pipeline_state.thinker_inputs)
        combined_deepstack = model_inputs.pop("deepstack_visual_embeds", None)
        if combined_deepstack is None:
            pass
        elif "image_embeds" in model_inputs and "video_embeds" not in model_inputs:
            model_inputs["image_deepstack_visual_embeds"] = combined_deepstack
        elif "video_embeds" in model_inputs and "image_embeds" not in model_inputs:
            model_inputs["video_deepstack_visual_embeds"] = combined_deepstack
        else:
            raise ValueError(
                "retained mixed visual inputs require separate deepstack rows"
            )
        # Note (Junnan Li): Encoder lengths replace variable-width attention masks after audio encoding.
        model_inputs.pop("feature_attention_mask", None)
        prompt_start = len(unit_ids)
        prefix_model_inputs = unit_model_inputs
        unit_ids = [*unit_ids, *prompt_ids]
        unit_model_inputs = merge_model_inputs(unit_model_inputs, model_inputs)
        native_ids = [*history.ids, *unit_ids] if history.rebuild else unit_ids
        pipeline_state.prompt = {
            "input_ids": torch.tensor(native_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(native_ids), dtype=torch.long),
        }
        # Note (Junnan Li): Native sessions identify retained rows; media hash remapping would hide their modality positions.
        pipeline_state.thinker_inputs = {"model_inputs": {}}
        request_data = build_sglang_thinker_request(
            pipeline_state,
            params=payload.request.params,
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            request_id=payload.request_id,
        )
        request_data.stage_payload = payload
        # Note (Junnan Li): The preprocessor sees one turn; the scheduler checks the context once the retained turns precede it.
        request_data.enforce_request_limits = True
        history.unit = ThinkerUnit(
            turn=chunk.turn,
            ids=unit_ids,
            model_inputs=unit_model_inputs,
            prompt_start=prompt_start,
            prefix_model_inputs=prefix_model_inputs,
        )
        return request_data

    def strip_system_prefix(self, prompt: dict, prompt_ids: list[int]) -> list[int]:
        prompt_text = prompt["prompt_text"]
        user_start = prompt_text.index("<|im_start|>user\n")
        prefix_ids = self.tokenizer.encode(
            prompt_text[:user_start], add_special_tokens=False
        )
        if prompt_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError("Qwen system prefix does not match tokenized input")
        else:
            return prompt_ids[len(prefix_ids) :]

    def separator(self, ids: list[int]) -> list[int]:
        end_line = [self.end_token_id, *self.newline_ids]
        if not ids or ids[-len(end_line) :] == end_line:
            return []
        elif ids[-1] == self.end_token_id:
            return list(self.newline_ids)
        else:
            return end_line

    def bind_request(
        self,
        session_identity: SessionIdentity,
        request_data: SGLangARRequestData,
        unit: SessionUnit,
    ) -> None:
        history = self.states[session_identity]
        assert history.unit is not None
        model_inputs = merge_model_inputs(
            history.model_inputs, history.unit.model_inputs
        )
        request = request_data.req
        request.omni_model_inputs = dict(model_inputs) if model_inputs else None
        request._omni_consumed = (
            None  # noqa: leading-underscore - model runner contract
        )
        request._omni_mm_positions = (
            None  # noqa: leading-underscore - model runner contract
        )
        request._codec_suppress_tokens = (
            None  # noqa: leading-underscore - model runner contract
        )
        request.skip_radix_cache_insert = True
        request_data.model_inputs = dict(model_inputs)
        if model_inputs:
            positions, delta = compute_mrope_positions(
                torch.tensor(request.origin_input_ids, dtype=torch.long),
                model_inputs,
                self.thinker_config,
            )
            request.multimodal_inputs = MultimodalInputs(mm_items=[])
            request.multimodal_inputs.mrope_positions = positions
            request.multimodal_inputs.mrope_position_delta = delta
        else:
            pass

    def on_abort(self, session_identity: SessionIdentity, unit: SessionUnit) -> None:
        history = self.states[session_identity]
        inflight = history.unit
        if inflight is None:
            return
        else:
            history.unit = None
        generated = (
            [int(token_id) for token_id in unit.session_request.output_ids]
            if unit.session_request is not None
            else []
        )
        history.pending = PendingTurn(unit=inflight, generated=generated)

    def result(
        self, session_identity: SessionIdentity, request_data: SGLangARRequestData
    ) -> StagePayload:
        history = self.states[session_identity]
        inflight = history.unit
        assert inflight is not None
        output_ids = [int(token_id) for token_id in request_data.output_ids]
        history.rebuild = False
        history.last_prompt_start = len(history.ids) + inflight.prompt_start
        history.model_inputs_before_last_prompt = merge_model_inputs(
            history.model_inputs, inflight.prefix_model_inputs
        )
        history.last_reply_start = len(history.ids) + len(inflight.ids)
        history.ids = [*history.ids, *inflight.ids, *output_ids]
        history.model_inputs = merge_model_inputs(
            history.model_inputs, inflight.model_inputs
        )
        history.last_turn = inflight.turn
        history.unit = None
        history.pending = None
        payload = request_data.stage_payload
        payload.data["thinker_out"] = {"output_ids": output_ids, "is_final": True}
        return payload
