# SPDX-License-Identifier: Apache-2.0
"""Native MLX seed isolation and streaming stage contracts."""

import threading
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from sglang.srt.hardware_backend.mlx.kv_cache import (  # noqa: E402 - MLX is optional outside Apple Silicon.
    ContiguousAttentionKVCache,
)

from sglang_omni.models.qwen3_tts.mlx import (  # noqa: E402 - MLX is optional outside Apple Silicon.
    vocoder as mlx_vocoder,
)
from sglang_omni.models.qwen3_tts.mlx.decoder import (  # noqa: E402 - MLX is optional outside Apple Silicon.
    Qwen3TTSMlxSpeechDecoder,
    Qwen3TTSMlxTokenizerConfig,
)
from sglang_omni.models.qwen3_tts.mlx.generate import (  # noqa: E402 - MLX is optional outside Apple Silicon.
    Qwen3TTSMlxGenerator,
)
from sglang_omni.models.qwen3_tts.payload_types import Qwen3TTSState
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from tests.unit_test.qwen3_tts.test_mlx_decoder import (  # noqa: E402 - Shared optional MLX fixture.
    decoder_config as decoder_config,
)
from tests.unit_test.qwen3_tts.test_mlx_streaming import (  # noqa: E402 - Shared optional MLX fixture.
    code_generator as code_generator,
)


@pytest.mark.parametrize("top_p", [0.8, 1.0])
def test_interleaved_generation_preserves_each_request_seed(
    code_generator: Qwen3TTSMlxGenerator,
    monkeypatch: pytest.MonkeyPatch,
    top_p: float,
) -> None:
    forward = code_generator.talker.forward_embeddings

    def forward_without_eos(
        embeddings: mx.array, cache: list[ContiguousAttentionKVCache] | None = None
    ) -> tuple[mx.array, mx.array]:
        logits, hidden = forward(embeddings, cache)
        logits[
            :, :, code_generator.talker.artifact.talker_config.codec_eos_token_id
        ] = -float("inf")
        return logits, hidden

    monkeypatch.setattr(
        code_generator.talker, "forward_embeddings", forward_without_eos
    )
    generation = {
        "text": "Hello",
        "voice": "Ryan",
        "language": "English",
        "max_new_tokens": 7,
        "temperature": 0.8,
        "top_k": 0,
        "top_p": top_p,
        "repetition_penalty": 1.05,
    }
    expected = [
        np.asarray(
            mx.stack(list(code_generator.generate_codes(**generation, seed=seed)))
        )
        for seed in (42, 43)
    ]
    requests = [
        code_generator.generate_codes(**generation, seed=seed) for seed in (42, 43)
    ]
    actual: list[list[np.ndarray]] = [[], []]
    for _ in range(7):
        for index, request in enumerate(requests):
            actual[index].append(np.asarray(next(request)))
    for index, request in enumerate(requests):
        request.close()
        np.testing.assert_array_equal(np.stack(actual[index]), expected[index])
    assert not np.array_equal(expected[0], expected[1])


@pytest.mark.parametrize("payload_before_chunks", [False, True])
@pytest.mark.parametrize("first_streams_audio", [False, True])
def test_vocoder_interleaves_buffered_and_streaming_requests(
    monkeypatch: pytest.MonkeyPatch,
    decoder_config: Qwen3TTSMlxTokenizerConfig,
    payload_before_chunks: bool,
    first_streams_audio: bool,
) -> None:
    decoder = Qwen3TTSMlxSpeechDecoder(decoder_config)
    mx.eval(decoder.parameters())
    monkeypatch.setattr(mlx_vocoder, "load_qwen3_tts_mlx_decoder", lambda path: decoder)
    scheduler = mlx_vocoder.Qwen3TTSMlxVocoder(Path("unused"))
    frame_count = 7
    codes = [
        np.arange(frame_count * 3, dtype=np.int32).reshape(1, frame_count, 3) % 15 + 1,
        np.arange(frame_count * 3, dtype=np.int32).reshape(1, frame_count, 3) % 13 + 2,
    ]
    codes[0][0, 0, 0] = 0
    codes[0][0, -1, 0] = 0
    payloads = [
        StagePayload(
            request_id=f"request-{index}",
            request=OmniRequest(inputs="Hello", params={"stream": is_streaming}),
            data=Qwen3TTSState(completion_tokens=frame_count).to_dict(),
        )
        for index, is_streaming in enumerate(
            (first_streams_audio, not first_streams_audio)
        )
    ]
    if payload_before_chunks:
        scheduler.handle_new_request_batch(
            [
                IncomingMessage(
                    request_id=payload.request_id, type="new_request", data=payload
                )
                for payload in payloads
            ]
        )
    else:
        pass
    for start in (0, 4):
        for payload, request_codes in zip(payloads, codes, strict=True):
            scheduler.handle_stream_chunk(
                payload.request_id,
                StreamItem(
                    chunk_id=start // 4,
                    from_stage="tts_engine",
                    data=request_codes[:, start : start + 4],
                    metadata={
                        "modality": "audio_codes",
                        "is_streaming": payload.request.params["stream"],
                    },
                ),
            )
    for payload in payloads:
        scheduler.handle_stream_done(payload.request_id)
        if not payload_before_chunks:
            scheduler.handle_new_request_batch(
                [
                    IncomingMessage(
                        request_id=payload.request_id, type="new_request", data=payload
                    )
                ]
            )
        else:
            pass
    messages: list[OutgoingMessage] = [
        scheduler.outbox.get_nowait() for _ in range(scheduler.outbox.qsize())
    ]
    for payload, request_codes in zip(payloads, codes, strict=True):
        outgoing = [
            message for message in messages if message.request_id == payload.request_id
        ]
        assert outgoing[-1].type == "result"
        result = outgoing[-1].data.data
        assert result["usage"]["completion_tokens"] == frame_count
        if payload.request.params["stream"]:
            assert [message.type for message in outgoing] == [
                "stream",
                "stream",
                "result",
            ]
            assert "audio_waveform" not in result
            audio = b"".join(
                message.data["audio_waveform"] for message in outgoing[:-1]
            )
        else:
            assert len(outgoing) == 1
            audio = result["audio_waveform"]
        expected, lengths = decoder.decode(mx.array(request_codes))
        mx.eval(expected, lengths)
        np.testing.assert_allclose(
            np.frombuffer(audio, np.float32),
            np.asarray(expected[0, : int(lengths[0].item())]),
            atol=1e-6,
            rtol=1e-4,
        )


@pytest.mark.parametrize("abort_first", [False, True])
def test_vocoder_drops_late_chunks_after_completion_or_abort(
    monkeypatch: pytest.MonkeyPatch,
    decoder_config: Qwen3TTSMlxTokenizerConfig,
    abort_first: bool,
) -> None:
    decoder = Qwen3TTSMlxSpeechDecoder(decoder_config)
    mx.eval(decoder.parameters())
    monkeypatch.setattr(mlx_vocoder, "load_qwen3_tts_mlx_decoder", lambda path: decoder)
    scheduler = mlx_vocoder.Qwen3TTSMlxVocoder(Path("unused"))
    codes = np.ones((1, 4, 3), dtype=np.int32)
    worker = threading.Thread(target=scheduler.start)
    worker.start()
    try:
        for request_id in ("first", "second"):
            scheduler.inbox.put(
                IncomingMessage(
                    request_id=request_id,
                    type="stream_chunk",
                    data=StreamItem(
                        chunk_id=0,
                        from_stage="tts_engine",
                        data=codes,
                        metadata={"modality": "audio_codes", "is_streaming": True},
                    ),
                )
            )
            audio = scheduler.outbox.get(timeout=5)
            assert (audio.request_id, audio.type) == (request_id, "stream")
            if request_id == "first" and abort_first:
                scheduler.abort(request_id)
            else:
                payload = StagePayload(
                    request_id=request_id,
                    request=OmniRequest(inputs="Hello", params={"stream": True}),
                    data=Qwen3TTSState(completion_tokens=4).to_dict(),
                )
                scheduler.inbox.put(
                    IncomingMessage(request_id=request_id, type="stream_done")
                )
                scheduler.inbox.put(
                    IncomingMessage(
                        request_id=request_id, type="new_request", data=payload
                    )
                )
                terminal = scheduler.outbox.get(timeout=5)
                assert (terminal.request_id, terminal.type) == (request_id, "result")
            if request_id == "first":
                scheduler.inbox.put(
                    IncomingMessage(
                        request_id=request_id,
                        type="stream_chunk",
                        data=StreamItem(
                            chunk_id=1,
                            from_stage="tts_engine",
                            data=codes,
                            metadata={"modality": "audio_codes", "is_streaming": True},
                        ),
                    )
                )
            else:
                pass
    finally:
        scheduler.stop()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert scheduler.outbox.empty()
