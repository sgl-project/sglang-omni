# SPDX-License-Identifier: Apache-2.0
"""Project the D2 playback clock and its learned interrupts onto /v1/realtime."""

from collections.abc import Iterator

from sglang_omni.client.client import Client
from sglang_omni.models.qwen3_omni_d2.config import (
    INPUT_SAMPLE_RATE_HZ,
    MAX_SESSIONS,
    OUTPUT_SAMPLE_RATE_HZ,
    Qwen3OmniD2PipelineConfig,
)
from sglang_omni.models.qwen3_omni_d2.runtime import D2Frame
from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import OutputChunk
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
from sglang_omni.utils.hf import qwen_d2_interval_ms


class D2OutputConverter:
    def __init__(self) -> None:
        self.response_id: str | None = None
        self.transcript = ""

    def finish_response(
        self, status: ResponseStatus, reason: str
    ) -> Iterator[OutputEvent]:
        if self.response_id is not None:
            response_id = self.response_id
            item_id = f"{response_id}-message"
            yield TextFinished(response_id, item_id, self.transcript)
            yield AudioFinished(response_id, item_id)
            yield ResponseFinished(
                response_id, item_id, self.transcript, True, status, reason
            )
            self.response_id = None
            self.transcript = ""
        else:
            pass

    def __call__(self, output: OutputChunk) -> Iterator[OutputEvent]:
        if output.payload is None:
            assert output.eos
        else:
            frame = D2Frame.model_validate(output.payload)
            if frame.is_interruption:
                yield from self.finish_response("cancelled", "interrupted")
            else:
                pass
            if self.response_id is None:
                identity = output.session_identity
                self.response_id = (
                    f"{identity.id}-o{identity.open_index}-f{frame.sequence}"
                )
                yield ResponseStarted(self.response_id)
            else:
                pass
            item_id = f"{self.response_id}-message"
            if frame.text_delta:
                self.transcript += frame.text_delta
                yield TextDelta(self.response_id, item_id, frame.text_delta)
            else:
                pass
            yield AudioDelta(self.response_id, item_id, frame.pcm16)
        if output.eos:
            yield from self.finish_response("completed", "input_audio_ended")
        else:
            pass


def create_realtime_deployment(
    client: Client, pipeline_config: Qwen3OmniD2PipelineConfig
) -> RealtimeDeployment:
    interval_ms = qwen_d2_interval_ms(pipeline_config.model_path)
    if interval_ms is None:
        raise ValueError("Expected a Qwen3-Omni D2 release with a d2.json manifest")
    else:
        return RealtimeDeployment(
            capabilities=Capabilities(
                input_sample_rate_hz=INPUT_SAMPLE_RATE_HZ,
                output_sample_rate_hz=OUTPUT_SAMPLE_RATE_HZ,
                output_modalities=("audio", "text"),
                max_granted_modalities=2,
                native_unit_ms=interval_ms,
                tail_policy="pad",
                supports_custom_instructions=False,
                supports_server_interrupt=True,
            ),
            adapter_factory=lambda: CoordinatorAdapter(
                client,
                request_builder=lambda session: OmniRequest(inputs=None),
                output_converter=D2OutputConverter(),
                atomic_consumption=True,
            ),
            limits=runtime_limits_for_pipeline(pipeline_config),
            max_connections=MAX_SESSIONS,
        )
