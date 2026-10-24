# SPDX-License-Identifier: Apache-2.0
"""One Qwen talker request per reply, fed by the thinker's token stream; its text reaches the client through code2wav."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from queue import Queue

import torch
from sglang.srt.managers.schedule_batch import MultimodalInputs
from transformers import PreTrainedTokenizerBase
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerConfig,
)

from sglang_omni.models.qwen3_omni.components.talker_prefill import TalkerPrefillBuilder
from sglang_omni.models.qwen3_omni.mrope_positions import talker_can_use_linear_mrope
from sglang_omni.models.qwen3_omni.request_builders import (
    TalkerSamplingConfig,
    build_talker_request_data,
    compute_mrope_positions,
    resolve_seed,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.request import StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.proto.streams import TEXT, TEXT_DELTA
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.sglang_backend.ar_session import (
    ARSessionAdapter,
    ARSessionPreparation,
    SessionUnit,
)
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData

CODE2WAV_STAGE = "code2wav"


def code2wav_payload(payload: StagePayload) -> StagePayload:
    # Note (Junnan Li): The reply's codec rows stream to code2wav; the thinker's leg keeps the session params.
    return replace(
        payload,
        request=replace(
            payload.request, params={**payload.request.params, "stream": True}
        ),
    )


@dataclass(kw_only=True)
class ReplyText:
    """The thinker tokens of one reply and the text already relayed for them."""

    chunk: TimedChunk
    token_ids: list[int] = field(default_factory=list)
    emitted: str = ""


class QwenTalkerSessionAdapter(ARSessionAdapter):
    def __init__(
        self,
        prefill_builder: TalkerPrefillBuilder,
        tokenizer: PreTrainedTokenizerBase,
        codec_vocab_size: int,
        codec_eos_id: int,
        thinker_config: Qwen3OmniMoeThinkerConfig,
    ) -> None:
        self.prefill_builder: TalkerPrefillBuilder = prefill_builder
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.codec_vocab_size: int = codec_vocab_size
        self.codec_eos_id: int = codec_eos_id
        self.thinker_config: Qwen3OmniMoeThinkerConfig = thinker_config
        # The talker scheduler's outbox, attached once the scheduler exists.
        self.outbox: Queue[OutgoingMessage] | None = None
        self.replies: dict[str, ReplyText] = {}

    def prepare_unit(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> ARSessionPreparation:
        # Note (Junnan Li): Each reply is one talker request over a fresh native session.
        return ARSessionPreparation(reset_history=True)

    def sampling_config(self, params: dict[str, object]) -> TalkerSamplingConfig:
        return {
            "max_new_tokens": int(params["talker_max_new_tokens"]),
            "temperature": float(params["talker_temperature"]),
            "top_k": int(params["talker_top_k"]),
            "top_p": float(params["talker_top_p"]),
            "repetition_penalty": float(params["talker_repetition_penalty"]),
            "codec_eos_id": self.codec_eos_id,
            "suppress_tokens": [
                token_id
                for token_id in range(
                    max(self.codec_vocab_size - 1024, 0), self.codec_vocab_size
                )
                if token_id != self.codec_eos_id
            ],
            "seed": resolve_seed(params),
        }

    def build(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> SGLangARRequestData:
        request_data = build_talker_request_data(
            payload,
            prefill_builder=self.prefill_builder,
            tokenizer=self.tokenizer,
            codec_vocab_size=self.codec_vocab_size,
            codec_bos_id=self.prefill_builder.codec_bos_id,
            audio_token_id=self.thinker_config.audio_token_id,
            image_token_id=self.thinker_config.image_token_id,
            video_token_id=self.thinker_config.video_token_id,
            # Note (Junnan Li): M-RoPE positions go on the native request once retained ids precede the prompt.
            thinker_config=None,
            resolve_sampling_config=self.sampling_config,
        )
        request_data.stage_payload = code2wav_payload(payload)
        reply = ReplyText(chunk=chunk)
        self.replies[payload.request_id] = reply
        for item in payload.prefetched_chunks:
            self.relay_text(payload.request_id, reply, item)
        return request_data

    def complete_without_generation(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> StagePayload:
        reply = ReplyText(chunk=chunk)
        for item in payload.prefetched_chunks:
            self.relay_text(payload.request_id, reply, item)
        return code2wav_payload(payload)

    def bind_request(
        self,
        session_identity: SessionIdentity,
        request_data: SGLangARRequestData,
        unit: SessionUnit,
    ) -> None:
        request = request_data.req
        input_ids = torch.tensor(request.origin_input_ids, dtype=torch.long)
        prompt_model_inputs = request_data.talker_model_inputs
        if talker_can_use_linear_mrope(
            input_ids, prompt_model_inputs, self.thinker_config
        ):
            return
        else:
            positions, delta = compute_mrope_positions(
                input_ids, prompt_model_inputs, self.thinker_config
            )
            request.multimodal_inputs = MultimodalInputs(mm_items=[])
            request.multimodal_inputs.mrope_positions = positions
            request.multimodal_inputs.mrope_position_delta = delta

    def append_text_chunk(
        self, request_data: SGLangARRequestData, item: StreamItem
    ) -> None:
        self.prefill_builder.append_text_chunk(request_data, item)
        request_id = request_data.stage_payload.request_id
        reply = self.replies.get(request_id)
        if reply is None:
            return
        else:
            self.relay_text(request_id, reply, item)

    def mark_thinker_done(self, request_data: SGLangARRequestData) -> None:
        self.prefill_builder.mark_thinker_done(request_data)
        self.replies.pop(request_data.stage_payload.request_id, None)

    def relay_text(self, request_id: str, reply: ReplyText, item: StreamItem) -> None:
        """Decode the reply's text incrementally and hand each increment to code2wav, which publishes it in order with the audio."""
        token_id = int(item.metadata["token_id"])
        reply.token_ids.append(token_id)
        decoded = self.tokenizer.decode(reply.token_ids, skip_special_tokens=True)
        if decoded.endswith("�") or len(decoded) <= len(reply.emitted):
            return
        else:
            delta = decoded[len(reply.emitted) :]
            reply.emitted = decoded
        chunk = TimedChunk(
            TEXT,
            0,
            0,
            reply.chunk.seq,
            {"text": delta},
            format=TEXT_DELTA,
            turn=reply.chunk.turn,
        )
        assert self.outbox is not None
        # Note (Junnan Li): Stage streams carry tensors; the timed chunk rides the item's metadata.
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=torch.empty(0, dtype=torch.long),
                target=CODE2WAV_STAGE,
                metadata={"modality": TEXT, "stream": True, "chunk": chunk.to_dict()},
            )
        )

    def on_abort(self, session_identity: SessionIdentity, unit: SessionUnit) -> None:
        self.replies.pop(unit.request_id, None)

    def result(
        self, session_identity: SessionIdentity, request_data: SGLangARRequestData
    ) -> StagePayload:
        self.replies.pop(request_data.stage_payload.request_id, None)
        return request_data.stage_payload
