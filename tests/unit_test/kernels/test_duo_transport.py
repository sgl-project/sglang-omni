"""The Realization rank boundary transports conditions rather than input media."""

import threading
from dataclasses import replace
from multiprocessing import Process

import pytest
import torch

pytest.importorskip("torchtitan.models.duo_realization.modules.model")

from sglang.multimodal_gen.runtime.models.dits.duo.model import RealizationSegment
from sglang.multimodal_gen.runtime.models.dits.duo.sequence import AttentionSpan

from sglang_omni.models.duo.reference_parallel import (
    ParallelDCPRealizationSessionHooks,
    RealizationRankReply,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import (
    ResourceUsage,
    SessionIdentity,
    SessionOperation,
    TimedChunk,
)
from sglang_omni.scheduling.session import SessionContext


def test_parallel_realization_preserves_clock_and_conditions_without_input_media(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hooks = object.__new__(ParallelDCPRealizationSessionHooks)
    hooks.workers = [Process(), Process()]
    hooks.usages = {}
    hooks.tp_size = 1
    hooks.sp_mode = "ulysses"
    identity = SessionIdentity("duplex", 2)
    request = OmniRequest(inputs=None)
    chunk = TimedChunk(
        "realization_block",
        708.3,
        708.3,
        1,
        {"frames": b"rgb", "audio": b"pcm"},
        eos=True,
        turn=3,
    )
    segment = RealizationSegment(
        span=AttentionSpan(kind="shared", token_count=2),
        modality=0,
        tokens=torch.arange(6).reshape(2, 3),
        positions=torch.zeros(2, 3),
        roles=torch.zeros(2, dtype=torch.long),
    )
    payload = StagePayload(
        "duplex",
        request,
        {name: segment for name in ("reference", "shared", "plan", "video", "audio")},
    )
    context = SessionContext(
        session_identity=identity, cancelled=threading.Event(), emit=lambda output: None
    )

    def exchange(
        operation: SessionOperation, transported: StagePayload | None
    ) -> list[RealizationRankReply]:
        assert operation.session_identity == identity
        assert operation.chunk == replace(chunk, payload=None)
        assert transported is not None
        for name in payload.data:
            torch.testing.assert_close(
                transported.data[name].tokens, segment.tokens, rtol=0, atol=0
            )
        return [
            RealizationRankReply(
                payload=StagePayload("duplex", request, {"metrics": {}}),
                usage=ResourceUsage(kv_tokens=600, bytes=1000),
                stage_seconds={"worker_seconds": 0.4, "denoise_seconds": 0.3},
            ),
            RealizationRankReply(
                usage=ResourceUsage(kv_tokens=600, bytes=1000),
                stage_seconds={"worker_seconds": 0.5, "denoise_seconds": 0.35},
            ),
        ]

    monkeypatch.setattr(hooks, "exchange", exchange)
    result = hooks.append(chunk, payload, context)
    assert chunk.payload == {"frames": b"rgb", "audio": b"pcm"}
    assert result.data["metrics"]["denoise_seconds"] == 0.35
    assert result.data["metrics"]["rank_worker_seconds"] == [0.4, 0.5]
    assert hooks.usage(identity) == ResourceUsage(kv_tokens=600, bytes=2000)
    context.cancelled.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        hooks.append(chunk, payload, context)
