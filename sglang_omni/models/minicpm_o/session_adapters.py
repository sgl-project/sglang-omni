# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o unit conversion for the shared session runtime."""

from array import array

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from transformers import PreTrainedTokenizerBase

from sglang_omni.client.client import Client
from sglang_omni.models.minicpm_o.components.streaming_perception import (
    IMAGE_PREFETCH_CLEAR_MODALITY,
    IMAGE_PREFETCH_MODALITY,
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
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import OutputChunk, SessionIdentity, TimedChunk
from sglang_omni.scheduling.sglang_backend.ar_session import ARSessionAdapter
from sglang_omni.scheduling.sglang_backend.request_data import EmbeddingSpan
from sglang_omni.serve.realtime.adapters import CoordinatorAdapter
from sglang_omni.serve.realtime.manager import RealtimeDeployment
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    AudioFinished,
    OutputEvent,
    ResponseFinished,
    ResponseStarted,
    TextDelta,
    TextFinished,
)
from sglang_omni.serve.realtime.types import Capabilities


class MiniCPMORealtimeAdapter(CoordinatorAdapter):
    async def prefetch_image(self, unit_index: int, t_ms: float, image: bytes) -> None:
        assert self.session_identity is not None
        await self.client.append_session_stage(
            self.session_identity,
            TimedChunk(IMAGE_PREFETCH_MODALITY, t_ms, 0, unit_index, image),
            stage="perception",
        )

    async def clear(self) -> int:
        assert self.session_identity is not None
        await self.client.append_session_stage(
            self.session_identity,
            TimedChunk(IMAGE_PREFETCH_CLEAR_MODALITY, 0, 0, 0, None),
            stage="perception",
        )
        return 0


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
        prefix = [] if state.is_prefix_pending else [self.special.unit_end]
        token_ids = [*prefix, *plan["token_ids"]]
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
            is_listen_forced=state.force_listen_counter < sampling.force_listen_count,
        )

    def result(
        self, session_identity: SessionIdentity, data: DuplexUnitRequestData
    ) -> StagePayload:
        state = self.states[session_identity]
        state.is_prefix_pending = False
        output_token_ids = [int(token_id) for token_id in data.output_ids]
        data.stage_payload.data = dict(
            talker_conditions=data.talker_conditions,
            text=self.tokenizer.decode(
                data.generated_unit_ids, skip_special_tokens=True
            ),
            is_listen=bool(
                output_token_ids and output_token_ids[0] == self.special.listen
            ),
            end_of_turn=self.special.turn_eos in output_token_ids
            or any(ends_turn for _, _, ends_turn in data.talker_conditions),
        )
        return data.stage_payload


class OutputConverter:
    def __init__(self) -> None:
        self.response_id: str | None = None
        self.item_id: str | None = None
        self.text: str = ""
        self.audio: bool = False
        self.serial: int = 0

    def finish(self) -> list[OutputEvent]:
        if self.response_id is None:
            return []
        else:
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
                    "completed",
                    "stop",
                )
            )
            self.response_id = None
            self.text = ""
            self.audio = False
            return events

    def __call__(self, output: OutputChunk) -> list[OutputEvent]:
        value = output.payload
        events: list[OutputEvent] = []
        if value["text"] or value["pcm"]:
            if self.response_id is None:
                self.response_id = (
                    f"{output.session_identity.id}-response-{self.serial}"
                )
                self.item_id = self.response_id + "-message"
                self.serial += 1
                events.append(ResponseStarted(self.response_id))
            else:
                pass
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
            events.extend(self.finish())
        else:
            pass
        return events


def build_realtime_deployment(
    client: Client, config: MiniCPMODuplexPipelineConfig
) -> RealtimeDeployment:

    def factory() -> MiniCPMORealtimeAdapter:
        return MiniCPMORealtimeAdapter(
            client,
            stages=["perception", "thinker", "talker", "speech"],
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
        max_connections=config.max_sessions,
    )
