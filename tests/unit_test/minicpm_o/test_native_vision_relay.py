# SPDX-License-Identifier: Apache-2.0
"""Native relay of unit images and audio from perception into the thinker session."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image

from sglang_omni.models.minicpm_o.components.streaming_perception import (
    MiniCPMOPerceptionState,
    PerceptionStepPlan,
)
from sglang_omni.models.minicpm_o.duplex_sampler import (
    build_forbidden_token_index,
    duplex_sample,
)
from sglang_omni.models.minicpm_o.native_config import MiniCPMODuplexSampling
from sglang_omni.models.minicpm_o.native_stages import PerceptionHooks
from sglang_omni.models.minicpm_o.native_thinker_model_runner import (
    MiniCPMOThinkerModelRunner,
)
from sglang_omni.models.minicpm_o.session_adapters import ThinkerAdapter
from sglang_omni.models.minicpm_o.special_tokens import REQUIRED_SPECIAL_TOKENS
from sglang_omni.models.minicpm_o.thinker_state import MiniCPMOThinkerSessionState
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import SessionAppend
from sglang_omni.scheduling.sglang_backend.ar_session import ARSessionBridge
from sglang_omni.scheduling.sglang_backend.request_data import (
    EmbeddingSpan,
    splice_embedding_spans,
)

IDENTITY = SessionIdentity("vision")


@pytest.fixture
def perception() -> MiniCPMOPerceptionState:
    tokenizer = Mock(unk_token_id=0, bad_token_ids=[])
    tokenizer.convert_tokens_to_ids.side_effect = dict(
        zip(REQUIRED_SPECIAL_TOKENS, range(1, 17))
    ).__getitem__
    audio_encoder = Mock()
    audio_encoder.forward_streaming_batch.side_effect = lambda chunks: [
        (torch.full((10, 4), 9.0), None) for _ in chunks
    ]
    state = MiniCPMOPerceptionState(
        tokenizer=tokenizer,
        processor=Mock(),
        audio_encoder=audio_encoder,
        image_encoder=Mock(),
        max_slice_nums=1,
        mel_filter_bank=Mock(),
    )
    state.prepare_audio = Mock(return_value=np.zeros(4, dtype=np.float32))
    state.mel_chunk = Mock(return_value=Mock(batch_key=Mock(return_value=())))
    state.finish_audio = Mock()
    state.encode_image = Mock(return_value=torch.full((64, 4), 6.0))
    return state


@pytest.fixture
def hooks(perception: MiniCPMOPerceptionState) -> PerceptionHooks:
    hooks = PerceptionHooks(
        perception.tokenizer,
        Mock(),
        perception.audio_encoder,
        reference_audio=b"",
        image_encoder=Mock(),
        mel_filter_bank=Mock(
            log_mel=Mock(side_effect=lambda windows: torch.zeros(len(windows), 80, 1))
        ),
        reference_cache_capacity=1,
    )
    hooks.states[IDENTITY] = perception
    return hooks


def append_unit(
    hooks: PerceptionHooks, chunk: TimedChunk, payload: StagePayload
) -> StagePayload:
    [result] = hooks.append_batch(
        [
            SessionAppend(
                chunk=chunk,
                payload=payload,
                context=SimpleNamespace(session_identity=IDENTITY),
            )
        ]
    )
    return result


def unit_payload(data: PerceptionStepPlan | None = None) -> StagePayload:
    return StagePayload(
        "unit", OmniRequest(None, params=MiniCPMODuplexSampling().model_dump()), data
    )


@pytest.mark.parametrize(
    ("use_runner", "max_new_tokens", "generation_step", "closes"),
    [
        (True, 20, 0, False),
        (False, 20, 0, False),
        (False, 5, 4, True),
        (False, 6, 4, False),
    ],
)
def test_duplex_sample_masks_bad_tokens_and_closes_at_budget(
    use_runner: bool, max_new_tokens: int, generation_step: int, closes: bool
) -> None:
    """Both init paths keep the checkpoint mask; the token budget closes the chunk."""
    tokenizer = Mock(unk_token_id=0, bad_token_ids=[7, 8, 94])
    tokenizer.convert_tokens_to_ids.side_effect = dict(
        zip(REQUIRED_SPECIAL_TOKENS, range(100, 116))
    ).__getitem__
    if use_runner:
        runner = MiniCPMOThinkerModelRunner.__new__(MiniCPMOThinkerModelRunner)
        runner.special_tokens = None
        runner.eos_token_ids = []
        special = runner.resolve_special_tokens(
            SimpleNamespace(req=Mock(tokenizer=tokenizer))
        )
    else:
        special = ThinkerAdapter(tokenizer, 128).special
    logits = torch.full((128,), -torch.inf)
    logits[7] = 100.0
    logits[special.tts_pad] = 99.0
    logits[42] = 0.0
    state = MiniCPMOThinkerSessionState(
        sampling=MiniCPMODuplexSampling(
            greedy=True,
            top_k=1,
            repetition_penalty=1.0,
            max_new_tokens_per_unit=max_new_tokens,
        )
    )
    token_id = duplex_sample(
        logits,
        state,
        special_tokens=special,
        forbidden_token_index=build_forbidden_token_index(
            special, 128, torch.device("cpu")
        ),
        generation_step=generation_step,
        is_listen_forced=False,
    )
    assert token_id == (special.chunk_eos if closes else 42)


@pytest.mark.parametrize(
    ("has_image", "first_unit"), [(False, False), (True, False), (True, True)]
)
def test_append_and_thinker_splice(
    perception: MiniCPMOPerceptionState,
    hooks: PerceptionHooks,
    has_image: bool,
    first_unit: bool,
) -> None:
    pcm = np.arange(16000, dtype="<i2").tobytes()
    chunk = TimedChunk(
        "audio", 0, 1000, 0, {"pcm": pcm, "images": [b"frame"]} if has_image else pcm
    )
    payload = unit_payload()
    result = append_unit(hooks, chunk, payload)
    assert result is payload
    np.testing.assert_array_equal(
        perception.prepare_audio.call_args.args[0],
        np.arange(16000, dtype=np.float32) / 32768,
    )
    if has_image:
        perception.encode_image.assert_called_once_with(b"frame")
    else:
        perception.encode_image.assert_not_called()
    adapter = ThinkerAdapter(perception.tokenizer, 100)
    adapter.open(IDENTITY, payload.request)
    adapter.states[IDENTITY].is_prefix_pending = first_unit
    request = adapter.build(IDENTITY, chunk, payload)
    assert len(request.unit_embedding_spans) == (2 if has_image else 1)
    prefix_length = 0 if first_unit else 1
    for actual, planned in zip(
        request.unit_embedding_spans, payload.data["embedding_spans"]
    ):
        assert (actual.start, actual.end) == (
            planned["token_start"] + prefix_length,
            planned["token_end"] + prefix_length,
        )
    rows = splice_embedding_spans(
        torch.full((len(request.input_ids), 4), -1.0), 0, request.unit_embedding_spans
    )
    assert torch.equal(rows[-10:], torch.full((10, 4), 9.0))
    if has_image:
        assert torch.equal(
            rows[prefix_length + 2 : prefix_length + 66], torch.full((64, 4), 6.0)
        )
        assert torch.equal(rows[prefix_length + 66], torch.full((4,), -1.0))
    else:
        pass


@pytest.mark.parametrize("finish", ["complete", "abort"])
def test_image_audio_commit_atomically(
    perception: MiniCPMOPerceptionState, finish: str
) -> None:
    payload = unit_payload(
        perception.build_step_plan(
            torch.full((10, 4), 9.0), (torch.full((64, 4), 6.0),)
        )
    )
    adapter = ThinkerAdapter(perception.tokenizer, 100)
    adapter.open(IDENTITY, payload.request)
    request = adapter.build(IDENTITY, TimedChunk("audio", 0, 1000, 0, b""), payload)
    history = EmbeddingSpan(start=1, end=3, input_embeds=torch.ones(2, 4))
    unit = SimpleNamespace(
        session_identity=IDENTITY, session_request=None, embedding_spans=[]
    )
    session = SimpleNamespace(unit=unit, embedding_spans=[history])
    native = Mock(session_id=IDENTITY.id)
    native.create_req.return_value = Mock(
        origin_input_ids=[7] * 12 + list(request.req.origin_input_ids), to_finish=None
    )
    bridge = ARSessionBridge.__new__(ARSessionBridge)
    bridge.drain = Mock()
    bridge.sessions = {IDENTITY.id: session}
    bridge.units_by_request_id = {"unit": unit}
    bridge.bridge_scheduler = Mock()
    bridge.bridge_scheduler.session_controller.get.return_value = native
    bridge.create_session_request(payload, request)
    assert session.embedding_spans == [history]
    assert [(span.start, span.end) for span in unit.embedding_spans] == [
        (14, 78),
        (79, 89),
    ]
    assert request.session_embedding_spans == [history, *unit.embedding_spans]
    if finish == "complete":
        bridge.complete("unit")
        assert session.embedding_spans == [history, *unit.embedding_spans]
        native.abort_req.assert_not_called()
    else:
        bridge.release_append_unit("unit")
        assert session.embedding_spans == [history]
        native.abort_req.assert_called_once_with()
    assert session.unit is None


def test_empty_eos_does_not_encode(
    perception: MiniCPMOPerceptionState, hooks: PerceptionHooks
) -> None:
    payload = unit_payload()
    append_unit(hooks, TimedChunk("audio", 0, 0, 1, None, eos=True), payload)
    assert payload.data is None
    perception.prepare_audio.assert_not_called()
    perception.encode_image.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        ValueError("bad frame"),
        Image.DecompressionBombError("big"),
    ],
)
def test_undecodable_frame_is_dropped_and_siblings_kept(
    perception: MiniCPMOPerceptionState, hooks: PerceptionHooks, error: Exception
) -> None:
    first = torch.full((64, 4), 3.0)
    last = torch.full((64, 4), 7.0)
    perception.encode_image.side_effect = [first, error, last]
    payload = unit_payload()
    append_unit(
        hooks,
        TimedChunk(
            "audio", 0, 1000, 0, {"pcm": b"\0\0", "images": [b"first", b"bad", b"last"]}
        ),
        payload,
    )
    spans = payload.data["embedding_spans"]
    assert [span["modality"] for span in spans] == ["image", "image", "audio"]
    assert torch.equal(payload.data["input_embeds"][:128], torch.cat([first, last]))
