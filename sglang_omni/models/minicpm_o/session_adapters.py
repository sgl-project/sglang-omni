# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o unit conversion for the shared session runtime."""

from array import array

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.client.client import Client
from sglang_omni.models.minicpm_o.special_tokens import resolve_special_token_ids
from sglang_omni.models.minicpm_o.thinker_state import (
    DuplexUnitRequestData,
    MiniCPMOThinkerSessionState,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, SessionLimits, TimedChunk
from sglang_omni.scheduling.sglang_backend.ar_session import ARSessionAdapter
from sglang_omni.scheduling.sglang_backend.request_data import EmbeddingSpan
from sglang_omni.serve.realtime.adapters import CoordinatorAdapter
from sglang_omni.serve.realtime.manager import RealtimeDeployment
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    AudioFinished,
    ResponseFinished,
    ResponseStarted,
    TextDelta,
    TextFinished,
)
from sglang_omni.serve.realtime.types import Capabilities


class ThinkerAdapter(ARSessionAdapter):
    def __init__(self, tokenizer, vocab_size):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.special = resolve_special_token_ids(tokenizer)
        self.states: dict[SessionIdentity, MiniCPMOThinkerSessionState] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.states[session_identity] = MiniCPMOThinkerSessionState()

    def close(self, session_identity: SessionIdentity) -> None:
        self.states.pop(session_identity, None)

    def finish_input(
        self, session_identity: SessionIdentity, payload: StagePayload
    ) -> StagePayload:
        payload.data = dict(pairs=[], text="", is_listen=True, end_of_turn=True)
        return payload

    def build(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> DuplexUnitRequestData:
        state = self.states[session_identity]
        plan = payload.data
        prefix = [] if state.prefix_pending else [self.special.unit_end]
        ids = [*prefix, *plan["token_ids"]]
        spans = [
            EmbeddingSpan(
                start=span["token_start"] + len(prefix),
                end=span["token_end"] + len(prefix),
                input_embeds=plan["input_embeds"][
                    span["embed_start"] : span["embed_end"]
                ],
            )
            for span in plan["embedding_spans"]
        ]
        params = SamplingParams(
            max_new_tokens=plan["decode_budget"],
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            repetition_penalty=1.0,
            stop_token_ids=set(self.special.chunk_terminators),
            no_stop_trim=True,
            skip_special_tokens=False,
        )
        params.normalize(self.tokenizer)
        req = Req(
            payload.request_id, "", array("q", ids), params, vocab_size=self.vocab_size
        )
        req.tokenizer = self.tokenizer
        req.return_hidden_states = True
        cfg = dict(payload.request.params)
        return DuplexUnitRequestData(
            req=req,
            input_ids=torch.tensor(ids),
            max_new_tokens=plan["decode_budget"],
            stage_payload=payload,
            thinker_state=state,
            unit_embedding_spans=spans,
            sampling_config=cfg,
            forced_listen=state.force_listen_counter < cfg.get("force_listen_count", 0),
            prefill_schema=plan["prefill_schema"],
        )

    def result(
        self, session_identity: SessionIdentity, data: DuplexUnitRequestData
    ) -> StagePayload:
        state = self.states[session_identity]
        state.prefix_pending = False
        ids = [int(i) for i in data.output_ids]
        data.stage_payload.data = dict(
            pairs=data.unit_pairs,
            prefill_schema=data.prefill_schema,
            text=self.tokenizer.decode(
                data.generated_unit_ids, skip_special_tokens=True
            ),
            is_listen=bool(ids and ids[0] == self.special.listen),
            end_of_turn=self.special.turn_eos in ids
            or any(pair[2] for pair in data.unit_pairs),
        )
        return data.stage_payload


class OutputConverter:
    def __init__(self):
        self.response_id = None
        self.item_id = None
        self.text = ""
        self.audio = False
        self.serial = 0

    def finish(self):
        if self.response_id is None:
            return []
        else:
            events = [TextFinished(self.response_id, self.item_id, self.text)]
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

    def __call__(self, output):
        value = output.payload
        events = []
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


def build_realtime_deployment(client: Client) -> RealtimeDeployment:

    def factory():
        return CoordinatorAdapter(
            client,
            stages=["perception", "thinker", "talker", "speech"],
            request_builder=lambda config: OmniRequest(
                None,
                params={
                    "instructions": config.get("instructions")
                    or "Streaming Omni Conversation.",
                    "greedy": True,
                    "force_listen_count": 3,
                },
            ),
            output_converter=OutputConverter(),
            input_sample_rate_hz=16000,
            atomic_consumption=True,
            limits=SessionLimits(operation_timeout_s=120),
        )

    return RealtimeDeployment(
        Capabilities(
            native_unit_ms=1000,
            output_sample_rate_hz=24000,
            output_modalities=("audio", "text"),
            tail_policy="pad",
        ),
        factory,
        max_connections=2,
    )
