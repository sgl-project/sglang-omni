# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the Qwen3-Omni breakable-prefill adopter."""

from __future__ import annotations

from array import array
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sglang")

import sglang_omni.models.qwen3_omni.thinker_model_runner as qwen_thinker_runner_module
from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni.models.qwen3_omni.thinker_model_runner import (
    Qwen3OmniThinkerModelRunner,
)

VOCAB = 64
HIDDEN = 4
IMAGE_ID = 51
VIDEO_ID = 52
AUDIO_ID = 53


def _runner() -> Qwen3OmniThinkerModelRunner:
    runner = object.__new__(Qwen3OmniThinkerModelRunner)
    runner.tp_worker = SimpleNamespace(record_custom_prefill_eager=lambda: None)
    torch.manual_seed(0)
    runner._embed_tokens = torch.nn.Embedding(VOCAB, HIDDEN)
    runner._image_token_id = IMAGE_ID
    runner._video_token_id = VIDEO_ID
    runner._audio_token_id = AUDIO_ID
    return runner


def _positions(
    *,
    image: tuple[int, ...] = (),
    video: tuple[int, ...] = (),
    audio: tuple[int, ...] = (),
) -> dict[str, torch.Tensor]:
    return {
        "image": torch.tensor(image, dtype=torch.long),
        "video": torch.tensor(video, dtype=torch.long),
        "audio": torch.tensor(audio, dtype=torch.long),
    }


def _request(
    input_ids: list[int],
    model_inputs: dict | None,
    *,
    positions: dict[str, torch.Tensor] | None = None,
    inflight_middle_chunks: int = 0,
):
    has_model_inputs = isinstance(model_inputs, dict) and bool(model_inputs)
    return SimpleNamespace(
        origin_input_ids=list(input_ids),
        omni_model_inputs=model_inputs if has_model_inputs else None,
        _omni_consumed=None,
        _omni_mm_positions=positions,
        inflight_middle_chunks=inflight_middle_chunks,
        multimodal_inputs=(
            SimpleNamespace(mrope_position_delta=object()) if has_model_inputs else None
        ),
    )


def _batch(
    requests: list,
    chunks: list[list[int]] | None = None,
    *,
    prefix_lens: list[int] | None = None,
    input_embeds=None,
    replace_embeds=None,
):
    chunks = chunks or [request.origin_input_ids for request in requests]
    prefix_lens = prefix_lens or [0] * len(requests)
    flat_ids = [token for chunk in chunks for token in chunk]
    forward_batch = SimpleNamespace(
        input_ids=torch.tensor(flat_ids, dtype=torch.long),
        batch_size=len(requests),
        extend_seq_lens_cpu=[len(chunk) for chunk in chunks],
        extend_prefix_lens_cpu=list(prefix_lens),
        input_embeds=input_embeds,
        replace_embeds=replace_embeds,
        mm_inputs=object(),
        positions=torch.arange(len(flat_ids), dtype=torch.long),
        mrope_positions=torch.arange(len(flat_ids), dtype=torch.long).repeat(3, 1),
    )
    schedule_batch = SimpleNamespace(
        reqs=list(requests),
        forward_mode=SimpleNamespace(is_extend=lambda: True),
        is_prefill_only=True,
    )
    return forward_batch, schedule_batch


def test_custom_prefill_forward_records_eager_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = object.__new__(Qwen3OmniThinkerModelRunner)
    calls: list[str] = []
    runner.tp_worker = SimpleNamespace(
        record_custom_prefill_eager=lambda: calls.append("recorded")
    )
    runner._classify_prefill = lambda *_args: SimpleNamespace(kind="custom")
    expected = object()
    monkeypatch.setattr(
        ThinkerModelRunner,
        "custom_prefill_forward",
        lambda *_args: expected,
    )

    result = runner.custom_prefill_forward(SimpleNamespace(), object(), [])

    assert result is expected
    assert calls == ["recorded"]


def test_text_only_prefill_attaches_live_embeddings_without_official_batch_mutation():
    runner = _runner()
    request = _request([7, 8, 9], None)
    forward_batch, schedule_batch = _batch([request])
    live_embeds = torch.full((3, HIDDEN), 31.0)

    assert request.omni_model_inputs is None
    assert request.multimodal_inputs is None
    assert request._omni_mm_positions is None

    class LiveEmbedding:
        num_embeddings = VOCAB

        def __call__(self, input_ids):
            assert input_ids is forward_batch.input_ids
            return live_embeds

    runner._embed_tokens = LiveEmbedding()
    official_mm_inputs = forward_batch.mm_inputs
    official_mrope_positions = forward_batch.mrope_positions

    runner.before_prefill(forward_batch, schedule_batch, [request])

    sidecar = get_omni_prefill_inputs(forward_batch)
    assert sidecar is not None
    assert sidecar.input_embeds is live_embeds
    assert forward_batch.input_embeds is None
    assert forward_batch.replace_embeds is None
    assert forward_batch.mm_inputs is official_mm_inputs
    assert forward_batch.mrope_positions is official_mrope_positions
    assert request.multimodal_inputs is None
    assert request.omni_model_inputs is None
    assert request._omni_consumed is None
    assert request._omni_mm_positions is None
    assert (
        runner.custom_prefill_forward(forward_batch, schedule_batch, [request]) is None
    )


def test_text_invalid_token_ids_are_not_silently_clamped():
    runner = _runner()
    request = _request([VOCAB], None)
    forward_batch, schedule_batch = _batch([request])

    with pytest.raises(IndexError):
        runner.before_prefill(forward_batch, schedule_batch, [request])

    assert get_omni_prefill_inputs(forward_batch) is None


def test_text_only_prefill_skips_chunk_span_normalization(monkeypatch):
    runner = _runner()
    request = _request([7, 8, 9], None)
    forward_batch, schedule_batch = _batch([request])

    def unexpected_chunk_span_normalization(*_args):
        pytest.fail("text-only prefill should not normalize multimodal spans")

    monkeypatch.setattr(
        runner, "_batch_chunk_spans", unexpected_chunk_span_normalization
    )

    runner.before_prefill(forward_batch, schedule_batch, [request])

    assert get_omni_prefill_inputs(forward_batch) is not None


def test_explicit_positions_skip_redundant_origin_prompt_tensorization(monkeypatch):
    runner = _runner()
    request = _request(
        [7, AUDIO_ID, 8],
        {"audio_embeds": torch.ones(1, HIDDEN)},
        positions=_positions(audio=(1,)),
    )
    forward_batch, schedule_batch = _batch([request])

    def unexpected_prompt_tensorization(*_args, **_kwargs):
        raise AssertionError("explicit positions must not rebuild origin_input_ids")

    monkeypatch.setattr(
        qwen_thinker_runner_module.torch,
        "as_tensor",
        unexpected_prompt_tensorization,
    )

    runner.before_prefill(forward_batch, schedule_batch, [request])

    assert get_omni_prefill_inputs(forward_batch) is not None


def test_multi_audio_batch_normalizes_chunk_spans_once(monkeypatch):
    runner = _runner()
    requests = [
        _request(
            [7, AUDIO_ID],
            {"audio_embeds": torch.ones(1, HIDDEN)},
            positions=_positions(audio=(1,)),
        ),
        _request(
            [8, AUDIO_ID],
            {"audio_embeds": torch.full((1, HIDDEN), 2.0)},
            positions=_positions(audio=(1,)),
        ),
    ]
    forward_batch, schedule_batch = _batch(requests)
    calls = []
    original = runner._batch_chunk_spans

    def wrapped_batch_chunk_spans(batch, expected_batch_size):
        calls.append(expected_batch_size)
        return original(batch, expected_batch_size)

    monkeypatch.setattr(runner, "_batch_chunk_spans", wrapped_batch_chunk_spans)

    runner.before_prefill(forward_batch, schedule_batch, requests)

    assert calls == [2]
    assert get_omni_prefill_inputs(forward_batch) is not None


def test_audio_prefill_composes_embeddings_into_the_private_sidecar():
    runner = _runner()
    audio_embeds = torch.arange(HIDDEN, dtype=torch.float32).reshape(1, HIDDEN)
    request = _request(
        [7, AUDIO_ID, 8],
        {"audio_embeds": audio_embeds},
        positions=_positions(audio=(1,)),
    )
    forward_batch, schedule_batch = _batch([request])
    official_mm_inputs = forward_batch.mm_inputs
    official_mrope_positions = forward_batch.mrope_positions
    official_request_mm_inputs = request.multimodal_inputs
    official_mrope_delta = request.multimodal_inputs.mrope_position_delta

    runner.before_prefill(forward_batch, schedule_batch, [request])

    sidecar = get_omni_prefill_inputs(forward_batch)
    assert sidecar is not None
    expected = runner._embed_tokens(forward_batch.input_ids).detach().clone()
    expected[1] = audio_embeds[0]
    torch.testing.assert_close(sidecar.input_embeds, expected)
    assert forward_batch.input_embeds is None
    assert forward_batch.replace_embeds is None
    assert forward_batch.mm_inputs is official_mm_inputs
    assert forward_batch.mrope_positions is official_mrope_positions
    assert request.multimodal_inputs is official_request_mm_inputs
    assert request.multimodal_inputs.mrope_position_delta is official_mrope_delta
    assert request.omni_model_inputs is None
    assert request._omni_consumed is None
    assert request._omni_mm_positions is None


def test_array_backed_origin_input_ids_are_sidecar_eligible():
    runner = _runner()
    request = _request(
        [7, AUDIO_ID, 8],
        {"audio_embeds": torch.ones(1, HIDDEN)},
        positions=_positions(audio=(1,)),
    )
    request.origin_input_ids = array("q", request.origin_input_ids)
    forward_batch, schedule_batch = _batch([request])

    runner.before_prefill(forward_batch, schedule_batch, [request])

    assert get_omni_prefill_inputs(forward_batch) is not None


def test_audio_sidecar_preserves_composed_embedding_identity(monkeypatch):
    runner = _runner()
    request = _request(
        [7, AUDIO_ID, 8],
        {"audio_embeds": torch.ones(1, HIDDEN)},
        positions=_positions(audio=(1,)),
    )
    forward_batch, schedule_batch = _batch([request])
    composed = torch.full((3, HIDDEN), 29.0)
    monkeypatch.setattr(
        runner,
        "_inject_multimodal_embeds",
        lambda *_args: (composed, None, None),
    )

    runner.before_prefill(forward_batch, schedule_batch, [request])

    sidecar = get_omni_prefill_inputs(forward_batch)
    assert sidecar is not None
    assert sidecar.input_embeds is composed


def test_mixed_text_and_audio_batch_uses_one_live_sidecar():
    runner = _runner()
    audio_embeds = torch.full((1, HIDDEN), 17.0)
    text_request = _request([7, 8], None)
    audio_request = _request(
        [9, AUDIO_ID, 10],
        {"audio_embeds": audio_embeds},
        positions=_positions(audio=(1,)),
    )
    forward_batch, schedule_batch = _batch(
        [text_request, audio_request],
        chunks=[[7, 8], [9, AUDIO_ID, 10]],
    )

    runner.before_prefill(
        forward_batch,
        schedule_batch,
        [text_request, audio_request],
    )

    sidecar = get_omni_prefill_inputs(forward_batch)
    assert sidecar is not None
    expected = runner._embed_tokens(forward_batch.input_ids).detach().clone()
    expected[3] = audio_embeds[0]
    torch.testing.assert_close(sidecar.input_embeds, expected)
    assert forward_batch.input_embeds is None


def test_chunked_audio_prefill_attaches_live_text_and_audio_chunks():
    runner = _runner()
    audio_embeds = torch.tensor([[1.0] * HIDDEN, [2.0] * HIDDEN], dtype=torch.float32)
    audio_inputs = {"audio_embeds": audio_embeds}
    request = _request(
        [AUDIO_ID, 7, AUDIO_ID],
        audio_inputs,
        positions=_positions(audio=(0, 2)),
        inflight_middle_chunks=1,
    )
    positions = request._omni_mm_positions
    official_request_mm_inputs = request.multimodal_inputs
    official_mrope_delta = official_request_mm_inputs.mrope_position_delta

    first_batch, first_schedule = _batch(
        [request], chunks=[[AUDIO_ID, 7]], prefix_lens=[0]
    )
    runner.before_prefill(first_batch, first_schedule, [request])
    first_sidecar = get_omni_prefill_inputs(first_batch)
    assert first_sidecar is not None
    first_expected = runner._embed_tokens(first_batch.input_ids).detach().clone()
    first_expected[0] = audio_embeds[0]
    torch.testing.assert_close(
        first_sidecar.input_embeds,
        first_expected,
    )
    assert request._omni_consumed == {"audio": 1}
    assert request.omni_model_inputs is audio_inputs
    assert request._omni_mm_positions is positions
    assert request.multimodal_inputs is official_request_mm_inputs
    assert request.multimodal_inputs.mrope_position_delta is official_mrope_delta

    request.inflight_middle_chunks = 0
    second_batch, second_schedule = _batch(
        [request], chunks=[[AUDIO_ID]], prefix_lens=[2]
    )
    runner.before_prefill(second_batch, second_schedule, [request])
    second_sidecar = get_omni_prefill_inputs(second_batch)
    assert second_sidecar is not None
    second_expected = runner._embed_tokens(second_batch.input_ids).detach().clone()
    second_expected[0] = audio_embeds[1]
    torch.testing.assert_close(second_sidecar.input_embeds, second_expected)
    assert request.omni_model_inputs is None
    assert request._omni_consumed is None
    assert request._omni_mm_positions is None
    assert request.multimodal_inputs is official_request_mm_inputs
    assert request.multimodal_inputs.mrope_position_delta is official_mrope_delta


def test_fresh_cached_prefix_with_only_live_audio_is_sidecar_eligible():
    runner = _runner()
    audio_embeds = torch.arange(HIDDEN, dtype=torch.float32).reshape(1, HIDDEN)
    request = _request(
        [7, 8, AUDIO_ID, 9],
        {"audio_embeds": audio_embeds},
        positions=_positions(audio=(2,)),
    )
    forward_batch, schedule_batch = _batch(
        [request], chunks=[[AUDIO_ID, 9]], prefix_lens=[2]
    )

    assert request._omni_consumed is None
    runner.before_prefill(forward_batch, schedule_batch, [request])

    sidecar = get_omni_prefill_inputs(forward_batch)
    assert sidecar is not None
    expected = runner._embed_tokens(forward_batch.input_ids).detach().clone()
    expected[0] = audio_embeds[0]
    torch.testing.assert_close(sidecar.input_embeds, expected)


def test_cached_prefix_mixed_audio_image_uses_live_rows_in_inherited_eager_path():
    runner = _runner()
    prompt = [AUDIO_ID, IMAGE_ID, AUDIO_ID, IMAGE_ID]
    audio_embeds = torch.tensor([[1.0] * HIDDEN, [2.0] * HIDDEN], dtype=torch.float32)
    image_embeds = torch.tensor([[3.0] * HIDDEN, [4.0] * HIDDEN], dtype=torch.float32)
    request = _request(
        prompt,
        {"audio_embeds": audio_embeds, "image_embeds": image_embeds},
        positions=_positions(image=(1, 3), audio=(0, 2)),
    )
    forward_batch, schedule_batch = _batch(
        [request], chunks=[prompt[2:]], prefix_lens=[2]
    )

    runner.before_prefill(forward_batch, schedule_batch, [request])

    assert get_omni_prefill_inputs(forward_batch) is None
    assert (
        runner.custom_prefill_forward(forward_batch, schedule_batch, [request]) is None
    )
    torch.testing.assert_close(forward_batch.input_embeds[0], audio_embeds[1])
    torch.testing.assert_close(forward_batch.input_embeds[1], image_embeds[1])


@pytest.mark.parametrize(
    ("input_ids", "audio_positions", "prefix", "live_chunk", "expected_audio_row"),
    [
        ([AUDIO_ID, 7, AUDIO_ID, 8], (0, 2), 2, [AUDIO_ID, 8], 1),
        (
            [AUDIO_ID, 7, AUDIO_ID, 8, AUDIO_ID, 9],
            (0, 2, 4),
            4,
            [AUDIO_ID, 9],
            2,
        ),
    ],
)
def test_fresh_cached_audio_prefix_uses_correct_inherited_eager_embedding(
    input_ids: list[int],
    audio_positions: tuple[int, ...],
    prefix: int,
    live_chunk: list[int],
    expected_audio_row: int,
):
    """Cached audio rows must seed the real inherited eager merge at the correct offset."""
    runner = _runner()
    audio_inputs = {
        "audio_embeds": torch.tensor(
            [[float(row + 1)] * HIDDEN for row in range(len(audio_positions))],
            dtype=torch.float32,
        )
    }
    request = _request(
        input_ids,
        audio_inputs,
        positions=_positions(audio=audio_positions),
    )
    forward_batch, schedule_batch = _batch(
        [request], chunks=[live_chunk], prefix_lens=[prefix]
    )
    official_mm_inputs = forward_batch.mm_inputs
    official_request_mm_inputs = request.multimodal_inputs
    official_mrope_delta = official_request_mm_inputs.mrope_position_delta
    requests = [request]

    runner.before_prefill(forward_batch, schedule_batch, requests)

    assert get_omni_prefill_inputs(forward_batch) is None
    result = runner.custom_prefill_forward(forward_batch, schedule_batch, requests)

    assert result is None
    expected = runner._embed_tokens(forward_batch.input_ids).detach().clone()
    expected[0] = audio_inputs["audio_embeds"][expected_audio_row]
    torch.testing.assert_close(
        forward_batch.input_embeds[0],
        audio_inputs["audio_embeds"][expected_audio_row],
    )
    torch.testing.assert_close(
        forward_batch.input_embeds[1],
        runner._embed_tokens(torch.tensor([live_chunk[1]], dtype=torch.long))[0],
    )
    torch.testing.assert_close(forward_batch.input_embeds, expected)
    assert forward_batch.mm_inputs is official_mm_inputs
    assert request.omni_model_inputs is None
    assert request._omni_consumed is None
    assert request._omni_mm_positions is None
    assert request.multimodal_inputs is official_request_mm_inputs
    assert request.multimodal_inputs.mrope_position_delta is official_mrope_delta


def test_cached_audio_eager_cursor_survives_text_only_middle_chunk():
    """The cached audio cursor must survive an intermediate eager chunk with no live audio."""
    runner = _runner()
    audio_embeds = torch.tensor([[1.0] * HIDDEN, [2.0] * HIDDEN], dtype=torch.float32)
    audio_inputs = {"audio_embeds": audio_embeds}
    request = _request(
        [AUDIO_ID, 7, 8, AUDIO_ID, 9],
        audio_inputs,
        positions=_positions(audio=(0, 3)),
        inflight_middle_chunks=1,
    )

    first_batch, first_schedule = _batch([request], chunks=[[7, 8]], prefix_lens=[1])
    runner.before_prefill(first_batch, first_schedule, [request])

    assert get_omni_prefill_inputs(first_batch) is None
    assert runner.custom_prefill_forward(first_batch, first_schedule, [request]) is None
    first_expected = runner._embed_tokens(first_batch.input_ids).detach().clone()
    torch.testing.assert_close(first_batch.input_embeds, first_expected)
    assert request._omni_consumed == {"audio": 1}
    assert request.omni_model_inputs is audio_inputs

    request.inflight_middle_chunks = 0
    second_batch, second_schedule = _batch(
        [request], chunks=[[AUDIO_ID, 9]], prefix_lens=[3]
    )
    runner.before_prefill(second_batch, second_schedule, [request])

    second_sidecar = get_omni_prefill_inputs(second_batch)
    assert second_sidecar is not None
    assert (
        runner.custom_prefill_forward(second_batch, second_schedule, [request]) is None
    )
    second_expected = runner._embed_tokens(second_batch.input_ids).detach().clone()
    second_expected[0] = audio_embeds[1]
    torch.testing.assert_close(second_sidecar.input_embeds, second_expected)
    assert request.omni_model_inputs is None
    assert request._omni_consumed is None
    assert request._omni_mm_positions is None


def test_cached_audio_eager_cursor_survives_unsupported_image_sibling():
    """An unsupported sibling must not prevent cached-audio cursor reconstruction."""
    runner = _runner()
    audio_embeds = torch.tensor([[1.0] * HIDDEN, [2.0] * HIDDEN], dtype=torch.float32)
    image_embeds = torch.full((1, HIDDEN), 3.0)
    audio_request = _request(
        [AUDIO_ID, 7, AUDIO_ID, 8],
        {"audio_embeds": audio_embeds},
        positions=_positions(audio=(0, 2)),
    )
    image_request = _request(
        [IMAGE_ID],
        {"image_embeds": image_embeds},
        positions=_positions(image=(0,)),
    )
    forward_batch, schedule_batch = _batch(
        [audio_request, image_request],
        chunks=[[AUDIO_ID, 8], [IMAGE_ID]],
        prefix_lens=[2, 0],
    )

    runner.before_prefill(
        forward_batch,
        schedule_batch,
        [audio_request, image_request],
    )

    assert get_omni_prefill_inputs(forward_batch) is None
    assert (
        runner.custom_prefill_forward(
            forward_batch,
            schedule_batch,
            [audio_request, image_request],
        )
        is None
    )
    torch.testing.assert_close(forward_batch.input_embeds[0], audio_embeds[1])
    torch.testing.assert_close(
        forward_batch.input_embeds[1],
        runner._embed_tokens(torch.tensor([8], dtype=torch.long))[0],
    )
    torch.testing.assert_close(forward_batch.input_embeds[2], image_embeds[0])
    assert audio_request.omni_model_inputs is None
    assert audio_request._omni_consumed is None
    assert audio_request._omni_mm_positions is None
    assert image_request.omni_model_inputs is None
    assert image_request._omni_consumed is None
    assert image_request._omni_mm_positions is None


def test_cached_audio_eager_cursor_preserves_existing_cursor():
    """A valid existing audio cursor remains authoritative during eager fallback."""
    runner = _runner()
    audio_embeds = torch.tensor([[1.0] * HIDDEN, [2.0] * HIDDEN], dtype=torch.float32)
    audio_request = _request(
        [AUDIO_ID, 7, AUDIO_ID, 8],
        {"audio_embeds": audio_embeds},
        positions=_positions(audio=(0, 2)),
        inflight_middle_chunks=1,
    )
    image_embeds = torch.full((1, HIDDEN), 3.0)
    image_request = _request(
        [IMAGE_ID],
        {"image_embeds": image_embeds},
        positions=_positions(image=(0,)),
    )
    existing_cursor = {"audio": 1}
    audio_request._omni_consumed = existing_cursor
    forward_batch, schedule_batch = _batch(
        [audio_request, image_request],
        chunks=[[AUDIO_ID, 8], [IMAGE_ID]],
        prefix_lens=[2, 0],
    )

    runner.before_prefill(
        forward_batch,
        schedule_batch,
        [audio_request, image_request],
    )

    assert get_omni_prefill_inputs(forward_batch) is None
    assert (
        runner.custom_prefill_forward(
            forward_batch,
            schedule_batch,
            [audio_request, image_request],
        )
        is None
    )
    torch.testing.assert_close(forward_batch.input_embeds[0], audio_embeds[1])
    torch.testing.assert_close(
        forward_batch.input_embeds[1],
        runner._embed_tokens(torch.tensor([8], dtype=torch.long))[0],
    )
    torch.testing.assert_close(forward_batch.input_embeds[2], image_embeds[0])
    # The inherited merge must not replace shared cursor ownership.
    assert audio_request._omni_consumed is existing_cursor
    assert audio_request._omni_consumed == {"audio": 2}
    assert audio_request.omni_model_inputs is not None
    assert image_request.omni_model_inputs is None


@pytest.mark.parametrize(
    "case",
    [
        "image",
        "video",
        "deepstack",
        "image_audio",
        "video_audio",
        "audio_in_video",
        "unknown",
        "malformed_audio",
    ],
)
def test_unsupported_payloads_delegate_to_the_inherited_eager_path(
    monkeypatch: pytest.MonkeyPatch, case: str
):
    runner = _runner()
    if case == "image":
        input_ids = [7, IMAGE_ID, 8]
        model_inputs = {"image_embeds": torch.ones(1, HIDDEN)}
        positions = _positions(image=(1,))
    elif case == "video":
        input_ids = [7, VIDEO_ID, 8]
        model_inputs = {"video_embeds": torch.ones(1, HIDDEN)}
        positions = _positions(video=(1,))
    elif case == "deepstack":
        input_ids = [7, AUDIO_ID, 8]
        model_inputs = {
            "audio_embeds": torch.ones(1, HIDDEN),
            "deepstack_visual_embeds": [torch.ones(1, HIDDEN)],
        }
        positions = _positions(audio=(1,))
    elif case == "image_audio":
        input_ids = [IMAGE_ID, AUDIO_ID]
        model_inputs = {
            "image_embeds": torch.ones(1, HIDDEN),
            "audio_embeds": torch.ones(1, HIDDEN),
        }
        positions = _positions(image=(0,), audio=(1,))
    elif case == "video_audio":
        input_ids = [VIDEO_ID, AUDIO_ID]
        model_inputs = {
            "video_embeds": torch.ones(1, HIDDEN),
            "audio_embeds": torch.ones(1, HIDDEN),
        }
        positions = _positions(video=(0,), audio=(1,))
    elif case == "audio_in_video":
        input_ids = [7, AUDIO_ID, 8]
        model_inputs = {
            "audio_embeds": torch.ones(1, HIDDEN),
            "use_audio_in_video": True,
        }
        positions = _positions(audio=(1,))
    elif case == "unknown":
        input_ids = [7, AUDIO_ID, 8]
        model_inputs = {"audio_embeds": torch.ones(1, HIDDEN), "future_aux": object()}
        positions = _positions(audio=(1,))
    else:
        input_ids = [7, AUDIO_ID, 8]
        model_inputs = {"audio_embeds": torch.ones(HIDDEN)}
        positions = _positions(audio=(1,))

    request = _request(input_ids, model_inputs, positions=positions)
    forward_batch, schedule_batch = _batch([request])
    seen = []

    def inherited_eager(self, forward_batch, schedule_batch, requests):
        seen.append((forward_batch, schedule_batch, requests))
        return "eager"

    monkeypatch.setattr(ThinkerModelRunner, "custom_prefill_forward", inherited_eager)

    runner.before_prefill(forward_batch, schedule_batch, [request])
    assert get_omni_prefill_inputs(forward_batch) is None
    assert (
        runner.custom_prefill_forward(forward_batch, schedule_batch, [request])
        == "eager"
    )
    assert len(seen) == 1
    assert forward_batch.input_embeds is None


def test_mixed_supported_and_unsupported_batch_falls_back_as_one_batch(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _runner()
    text_request = _request([7], None)
    audio_inputs = {"audio_embeds": torch.ones(2, HIDDEN)}
    audio_request = _request(
        [AUDIO_ID, 8, AUDIO_ID],
        audio_inputs,
        positions=_positions(audio=(0, 2)),
    )
    image_request = _request(
        [IMAGE_ID],
        {"image_embeds": torch.ones(1, HIDDEN)},
        positions=_positions(image=(0,)),
    )
    forward_batch, schedule_batch = _batch(
        [text_request, audio_request, image_request],
        chunks=[[7], [8], [IMAGE_ID]],
        prefix_lens=[0, 1, 0],
    )

    monkeypatch.setattr(
        ThinkerModelRunner,
        "custom_prefill_forward",
        lambda *_args: "eager",
    )
    runner.before_prefill(
        forward_batch,
        schedule_batch,
        [text_request, audio_request, image_request],
    )

    assert get_omni_prefill_inputs(forward_batch) is None
    assert audio_request.omni_model_inputs is audio_inputs
    assert audio_request._omni_consumed is None
    assert (
        runner.custom_prefill_forward(
            forward_batch,
            schedule_batch,
            [text_request, audio_request, image_request],
        )
        == "eager"
    )


def test_forward_batch_cardinality_mismatch_falls_back_without_sidecar(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _runner()
    request = _request([7], None)
    forward_batch, schedule_batch = _batch([request])
    forward_batch.batch_size = 2
    monkeypatch.setattr(
        ThinkerModelRunner,
        "custom_prefill_forward",
        lambda *_args: "eager",
    )

    runner.before_prefill(forward_batch, schedule_batch, [request])

    assert get_omni_prefill_inputs(forward_batch) is None
    assert (
        runner.custom_prefill_forward(forward_batch, schedule_batch, [request])
        == "eager"
    )


def test_malformed_consumed_audio_offset_falls_back_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _runner()
    audio_inputs = {"audio_embeds": torch.ones(1, HIDDEN)}
    request = _request(
        [7, AUDIO_ID, 8],
        audio_inputs,
        positions=_positions(audio=(1,)),
    )
    request._omni_consumed = {"audio": 2}
    forward_batch, schedule_batch = _batch([request])
    monkeypatch.setattr(
        ThinkerModelRunner,
        "custom_prefill_forward",
        lambda *_args: "eager",
    )

    runner.before_prefill(forward_batch, schedule_batch, [request])

    assert get_omni_prefill_inputs(forward_batch) is None
    assert request.omni_model_inputs is audio_inputs
    assert request._omni_consumed == {"audio": 2}
    assert (
        runner.custom_prefill_forward(forward_batch, schedule_batch, [request])
        == "eager"
    )


@pytest.mark.parametrize("field", ["input_embeds", "replace_embeds"])
def test_existing_official_embedding_fields_never_get_overwritten(
    monkeypatch: pytest.MonkeyPatch, field: str
):
    runner = _runner()
    request = _request([7, 8], None)
    official_value = torch.ones(2, HIDDEN) if field == "input_embeds" else object()
    kwargs = {field: official_value}
    forward_batch, schedule_batch = _batch([request], **kwargs)
    monkeypatch.setattr(
        ThinkerModelRunner,
        "custom_prefill_forward",
        lambda *_args: "eager",
    )

    runner.before_prefill(forward_batch, schedule_batch, [request])

    assert get_omni_prefill_inputs(forward_batch) is None
    assert getattr(forward_batch, field) is official_value
    assert (
        runner.custom_prefill_forward(forward_batch, schedule_batch, [request])
        == "eager"
    )
