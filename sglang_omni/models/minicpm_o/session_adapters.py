# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o unit conversion for the shared session runtime."""

from array import array
from dataclasses import dataclass

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from transformers import PreTrainedTokenizerBase

from sglang_omni.client.client import Client
from sglang_omni.models.minicpm_o.components.streaming_perception import (
    SAMPLE_RATE,
    UNIT_MS,
    PerceptionStepPlan,
)
from sglang_omni.models.minicpm_o.native_config import (
    MiniCPMODuplexPipelineConfig,
    MiniCPMODuplexSampling,
)
from sglang_omni.models.minicpm_o.special_tokens import (
    MiniCPMOSpecialTokenIds,
    resolve_special_token_ids,
)
from sglang_omni.models.minicpm_o.thinker_state import (
    DuplexUnitRequestData,
    MiniCPMOThinkerSessionState,
    SpokenUnit,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import OutputChunk, SessionIdentity, TimedChunk
from sglang_omni.proto.streams import CONTROL, INTERRUPT, SESSION_CONTROL, control_type
from sglang_omni.scheduling.sglang_backend.ar_session import (
    ARSessionAdapter,
    ARSessionPreparation,
)
from sglang_omni.scheduling.sglang_backend.request_data import EmbeddingSpan
from sglang_omni.serve.realtime.adapters import CoordinatorAdapter
from sglang_omni.serve.realtime.manager import (
    RealtimeDeployment,
    runtime_limits_for_pipeline,
)
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    AudioFinished,
    OutputEvent,
    ResponseFinished,
    ResponseStarted,
    ResponseStatus,
    TextDelta,
    TextFinished,
)
from sglang_omni.serve.realtime.types import Capabilities


class ThinkerAdapter(ARSessionAdapter):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, vocab_size: int) -> None:
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.vocab_size: int = vocab_size
        self.special: MiniCPMOSpecialTokenIds = resolve_special_token_ids(
            tokenizer, bad_token_ids=tuple(tokenizer.bad_token_ids)
        )
        self.states: dict[SessionIdentity, MiniCPMOThinkerSessionState] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.states[session_identity] = MiniCPMOThinkerSessionState(
            sampling=MiniCPMODuplexSampling.model_validate(
                {
                    key: request.params[key]
                    for key in MiniCPMODuplexSampling.model_fields
                }
            )
        )

    def close(self, session_identity: SessionIdentity) -> None:
        self.states.pop(session_identity, None)

    def control(self, session_identity: SessionIdentity, event: TimedChunk) -> None:
        if control_type(event) == INTERRUPT:
            state = self.states[session_identity]
            state.is_interrupt_pending = True
            spoken_unit = state.latest_spoken_unit
            last_delivered_unit = event.payload.get("last_delivered_unit")
            if (
                spoken_unit is not None
                and isinstance(last_delivered_unit, int)
                and spoken_unit.unit_index > last_delivered_unit
            ):
                # note (ruoyu): The client never got this unit's text and audio, so the session forgets it and the next unit refills it as the official forced listen, turn_eos first when the turn was open before it.
                state.kept_token_count = spoken_unit.kept_token_count
                state.refill_token_ids = (
                    [self.special.listen]
                    if spoken_unit.is_turn_ended_at_start
                    else [self.special.turn_eos, self.special.listen]
                )
                state.is_turn_ended = True
                state.generated_history = spoken_unit.generated_history_at_start
                state.latest_spoken_unit = None
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
        state = self.states[session_identity]
        kept_token_count = state.kept_token_count
        state.kept_token_count = None
        return ARSessionPreparation(kept_token_count=kept_token_count)

    def finish_input(
        self, session_identity: SessionIdentity, payload: StagePayload
    ) -> StagePayload:
        payload.data = dict(
            talker_conditions=[], text="", is_listen=True, end_of_turn=True
        )
        return payload

    def build(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> DuplexUnitRequestData:
        state = self.states[session_identity]
        plan: PerceptionStepPlan = payload.data
        is_interrupted = state.is_interrupt_pending
        state.is_interrupt_pending = False
        prefix = (
            []
            if state.is_prefix_pending
            else [*state.refill_token_ids, self.special.unit_end]
        )
        state.refill_token_ids = []
        if is_interrupted and not state.is_turn_ended:
            # note (ruoyu): Without turn_eos the model reads the stop as a pause in its own turn.
            suffix = [self.special.turn_eos]
        else:
            suffix = []
        token_ids = [*prefix, *plan["token_ids"], *suffix]
        spans = [
            EmbeddingSpan(
                start=span["token_start"] + len(prefix),
                end=span["token_end"] + len(prefix),
                input_embeds=plan["input_embeds"][
                    span["embedding_start"] : span["embedding_end"]
                ],
            )
            for span in plan["embedding_spans"]
        ]
        sampling = state.sampling
        sampling_params = SamplingParams(
            max_new_tokens=sampling.max_new_tokens_per_unit,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            repetition_penalty=1.0,
            stop_token_ids=set(self.special.chunk_terminators),
            no_stop_trim=True,
            skip_special_tokens=False,
        )
        sampling_params.normalize(self.tokenizer)
        adapter_request = Req(
            payload.request_id,
            "",
            array("q", token_ids),
            sampling_params,
            vocab_size=self.vocab_size,
        )
        adapter_request.tokenizer = self.tokenizer
        adapter_request.return_hidden_states = True
        return DuplexUnitRequestData(
            req=adapter_request,
            input_ids=torch.tensor(token_ids),
            max_new_tokens=sampling.max_new_tokens_per_unit,
            stage_payload=payload,
            thinker_state=state,
            unit_embedding_spans=spans,
            is_listen_forced=is_interrupted
            or state.force_listen_counter < sampling.force_listen_count,
            is_interrupted=is_interrupted,
            unit_index=chunk.seq,
            is_turn_ended_at_start=state.is_turn_ended,
            generated_history_at_start=list(state.generated_history),
        )

    def result(
        self, session_identity: SessionIdentity, data: DuplexUnitRequestData
    ) -> StagePayload:
        state = self.states[session_identity]
        state.is_prefix_pending = False
        if data.is_interrupted:
            # note (ruoyu): Without the ended turn, the next sampled listen is rewritten into speech and the stopped reply resumes.
            state.is_turn_ended = True
        else:
            pass
        if data.generated_unit_ids:
            # note (ruoyu): The native request's prompt is the retained turns plus this unit's ids, so dropping the unit keeps exactly that many tokens.
            state.latest_spoken_unit = SpokenUnit(
                unit_index=data.unit_index,
                kept_token_count=len(data.req.origin_input_ids),
                is_turn_ended_at_start=data.is_turn_ended_at_start,
                generated_history_at_start=data.generated_history_at_start,
            )
        else:
            state.latest_spoken_unit = None
        output_token_ids = [int(token_id) for token_id in data.output_ids]
        data.stage_payload.data = dict(
            talker_conditions=data.talker_conditions,
            text=self.tokenizer.decode(
                data.generated_unit_ids, skip_special_tokens=True
            ),
            is_listen=bool(
                output_token_ids and output_token_ids[0] == self.special.listen
            ),
            end_of_turn=data.is_interrupted
            or self.special.turn_eos in output_token_ids
            or any(ends_turn for _, _, ends_turn in data.talker_conditions),
        )
        return data.stage_payload


@dataclass(frozen=True, kw_only=True)
class CancelledReply:
    """The reply a client stop ends: its turn, and the last unit whose output the client received."""

    turn: int | None
    last_delivered_unit: int


class OutputConverter:
    def __init__(self) -> None:
        self.response_id: str | None = None
        self.item_id: str | None = None
        self.turn: int | None = None
        self.cancelled_turn: int | None = None
        self.last_delivered_unit: int | None = None
        self.text: str = ""
        self.audio: bool = False
        self.serial: int = 0

    def cancel(self) -> CancelledReply | None:
        """Mark the open reply cancelled; None when no reply is open or it is already cancelled."""
        if self.response_id is None or self.cancelled_turn is not None:
            return None
        else:
            assert self.last_delivered_unit is not None
            self.cancelled_turn = self.turn
            return CancelledReply(
                turn=self.turn, last_delivered_unit=self.last_delivered_unit
            )

    def finish(self, is_speech_cut: bool) -> list[OutputEvent]:
        """End the open response: cancelled when the client stopped it, incomplete when the talker's budget cut its speech."""
        if self.response_id is None:
            return []
        else:
            if self.cancelled_turn is not None:
                status: ResponseStatus = "cancelled"
                reason = "interrupted"
            elif is_speech_cut:
                status = "incomplete"
                reason = "max_output_tokens"
            else:
                status = "completed"
                reason = "stop"
            events: list[OutputEvent] = [
                TextFinished(self.response_id, self.item_id, self.text)
            ]
            if self.audio:
                events.append(AudioFinished(self.response_id, self.item_id))
            else:
                pass
            events.append(
                ResponseFinished(
                    self.response_id,
                    self.item_id,
                    self.text,
                    self.audio,
                    status,
                    reason,
                )
            )
            self.response_id = None
            self.cancelled_turn = None
            self.last_delivered_unit = None
            self.text = ""
            self.audio = False
            return events

    def __call__(self, output: OutputChunk) -> list[OutputEvent]:
        value = output.payload
        events: list[OutputEvent] = []
        if self.cancelled_turn is not None and output.turn == self.cancelled_turn:
            # note (ruoyu): The client stopped this reply, so what the pipeline still produces for its turn is not played.
            pass
        elif value["text"] or value["pcm"]:
            if self.response_id is None:
                self.response_id = (
                    f"{output.session_identity.id}-response-{self.serial}"
                )
                self.item_id = self.response_id + "-message"
                self.turn = output.turn
                self.serial += 1
                events.append(ResponseStarted(self.response_id))
            else:
                pass
            self.last_delivered_unit = output.input_seq
            if value["text"]:
                self.text += value["text"]
                events.append(TextDelta(self.response_id, self.item_id, value["text"]))
            else:
                pass
            if value["pcm"]:
                self.audio = True
                events.append(AudioDelta(self.response_id, self.item_id, value["pcm"]))
            else:
                pass
        else:
            pass
        if value["end_of_turn"] or output.eos:
            events.extend(self.finish(value["is_speech_cut"]))
        else:
            pass
        return events


class RealtimeAdapter(CoordinatorAdapter):
    """The coordinator adapter with the client stop of a native duplex reply."""

    output_converter: OutputConverter

    async def cancel(self) -> None:
        """Interrupt the open reply's turn: the turn's later audio is dropped, and the thinker forgets the unit the client did not receive before it listens for the next unit."""
        assert self.session_identity is not None
        cancelled_reply = self.output_converter.cancel()
        if cancelled_reply is None:
            pass
        else:
            await self.client.control_session(
                self.session_identity,
                TimedChunk(
                    CONTROL,
                    0,
                    0,
                    0,
                    {
                        "type": INTERRUPT,
                        "last_delivered_unit": cancelled_reply.last_delivered_unit,
                    },
                    format=SESSION_CONTROL,
                    turn=cancelled_reply.turn,
                ),
            )


def build_realtime_deployment(
    client: Client, config: MiniCPMODuplexPipelineConfig
) -> RealtimeDeployment:

    def factory() -> RealtimeAdapter:
        return RealtimeAdapter(
            client,
            request_builder=lambda session: OmniRequest(
                None,
                params={
                    "instructions": session.get("instructions")
                    or "Streaming Omni Conversation.",
                    **config.sampling.model_dump(),
                    **session.get("sglang", {}).get("sampling", {}),
                    **{
                        field: reference.data
                        for field in ("reference_audio", "tts_reference_audio")
                        if (reference := session.get("sglang", {}).get(field))
                        is not None
                    },
                    "max_slice_nums": session.get("sglang", {}).get(
                        "max_slice_nums", config.vision.max_slice_nums
                    ),
                },
            ),
            output_converter=OutputConverter(),
            input_sample_rate_hz=SAMPLE_RATE,
            atomic_consumption=True,
        )

    return RealtimeDeployment(
        Capabilities(
            native_unit_ms=UNIT_MS,
            output_sample_rate_hz=24000,
            output_modalities=("audio", "text"),
            input_modalities=("audio", "image"),
            image_frames_per_unit=tuple(
                min(
                    config.vision.max_frames_per_unit,
                    config.vision.max_tiles_per_unit
                    // (1 if slices == 1 else slices + 1),
                )
                for slices in range(1, config.vision.max_slice_nums_limit + 1)
            ),
            default_max_slice_nums=config.vision.max_slice_nums,
            tail_policy="pad",
            supports_reference_audio=True,
            sampling_parameters=tuple(MiniCPMODuplexSampling.model_fields),
        ),
        factory,
        limits=runtime_limits_for_pipeline(config),
        max_connections=config.max_sessions,
    )
