# SPDX-License-Identifier: Apache-2.0
"""The AURA request builder applies the caller's sampling, a per-request frame budget and the stage's device."""

from collections.abc import Mapping
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.aura_omni import engine_builder
from sglang_omni.models.aura_omni.engine_builder import (
    SILENT_TOKEN,
    AuraRequestData,
    make_aura_scheduler_adapters,
    validate_silent_token,
)
from sglang_omni.models.aura_omni.request_builders import AuraTurn
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.serve.openai_errors import is_bad_request_error

STAGE_DEVICE = torch.device("cpu")
FIRST_SPECIAL_TOKEN_ID = 151643
SILENT_TOKEN_ID = 151669
SHARED_VIDEO_CONFIG = {"fps": 2.0}


class FakeTokenizer:
    """Whitespace tokenizer; ids from FIRST_SPECIAL_TOKEN_ID up are special tokens, as in Qwen."""

    def __len__(self) -> int:
        return 151936

    def __call__(self, text: str, add_special_tokens: bool) -> SimpleNamespace:
        return SimpleNamespace(input_ids=list(range(len(text.split()))))

    def decode(self, token_ids: list[int], skip_special_tokens: bool) -> str:
        return " ".join(
            f"w{token_id}"
            for token_id in token_ids
            if not (skip_special_tokens and token_id >= FIRST_SPECIAL_TOKEN_ID)
        )


class SilentTokenTokenizer(FakeTokenizer):
    """Maps AURA's silence marker to its one special id, as AURA's tokenizer does."""

    def __call__(self, text: str, add_special_tokens: bool) -> SimpleNamespace:
        if text == SILENT_TOKEN:
            return SimpleNamespace(input_ids=[SILENT_TOKEN_ID])
        else:
            return super().__call__(text, add_special_tokens)


class FakePlatform:
    def __init__(self) -> None:
        self.current_device: torch.device | None = None

    def set_device(self, device: torch.device) -> None:
        self.current_device = device


class FakeProcessor:
    """Records the frame budget and the thread's device each multimodal call ran with."""

    def __init__(self, platform: FakePlatform) -> None:
        self.video_config = dict(SHARED_VIDEO_CONFIG)
        self.platform = platform
        self.calls: list[tuple[dict[str, float], torch.device | None]] = []

    async def process_mm_data_async(
        self,
        image_data: list[str | bytes],
        input_text: str,
        request_obj: SimpleNamespace,
    ) -> SimpleNamespace:
        self.calls.append((dict(self.video_config), self.platform.current_device))
        return SimpleNamespace(input_ids=[7] * 12)


class FakeMultimodalInputs:
    @staticmethod
    def from_processor_output(processed: SimpleNamespace) -> SimpleNamespace:
        return SimpleNamespace(padded_input_ids=list(processed.input_ids))


@pytest.fixture
def adapters(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    platform = FakePlatform()
    processor = FakeProcessor(platform)
    monkeypatch.setattr(engine_builder, "current_platform", platform)
    monkeypatch.setattr(engine_builder, "MultimodalInputs", FakeMultimodalInputs)
    request_builder, result_adapter = make_aura_scheduler_adapters(
        tokenizer=FakeTokenizer(),
        mm_processor=processor,
        model=SimpleNamespace(),
        device=STAGE_DEVICE,
        sampling_defaults={
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.0,
        },
        stop_token_ids=[151645],
        max_new_tokens=256,
        context_length=2048,
    )
    return SimpleNamespace(
        request_builder=request_builder,
        result_adapter=result_adapter,
        processor=processor,
    )


def turn_payload(
    *,
    params: Mapping[str, object],
    videos: list[str] | None = None,
    video_sampling: dict[str, float] | None = None,
    transcript: str = "what color is the ball",
) -> StagePayload:
    turn: AuraTurn = {
        "system_prompt": "Answer.",
        "history": [],
        "user_text": "",
        "images": [],
        "videos": videos or [],
        "video_sampling": video_sampling or {},
        "audio": None,
        "has_audio": True,
        "transcript": transcript,
    }
    return StagePayload(
        "r1", request=OmniRequest(inputs=None, params=dict(params)), data=turn
    )


def test_caller_sampling_overrides_only_the_fields_it_sets(
    adapters: SimpleNamespace,
) -> None:
    greedy: AuraRequestData = adapters.request_builder(
        turn_payload(params={"temperature": 0.0, "stream": False})
    )
    default: AuraRequestData = adapters.request_builder(
        turn_payload(params={"stream": False})
    )

    assert greedy.top_k == 1
    assert greedy.top_p == pytest.approx(0.8)
    assert (default.temperature, default.top_p, default.top_k) == (
        pytest.approx(0.7),
        pytest.approx(0.8),
        20,
    )


def test_frame_budget_applies_to_its_own_request_on_the_stage_device(
    adapters: SimpleNamespace,
) -> None:
    """Concurrent request builds share one processor, so one request's video_fps must not leak into the next."""
    adapters.request_builder(
        turn_payload(params={}, videos=["clip.mp4"], video_sampling={"fps": 1.0})
    )
    adapters.request_builder(turn_payload(params={}, videos=["clip.mp4"]))

    assert adapters.processor.calls == [
        ({"fps": 1.0}, STAGE_DEVICE),
        (SHARED_VIDEO_CONFIG, STAGE_DEVICE),
    ]
    assert adapters.processor.video_config == SHARED_VIDEO_CONFIG


def test_prompt_longer_than_the_context_is_a_bad_request(
    adapters: SimpleNamespace,
) -> None:
    with pytest.raises(
        ValueError, match="more than the .* the aura stage's context holds"
    ) as error:
        adapters.request_builder(
            turn_payload(params={"max_new_tokens": 2047, "stream": False})
        )
    assert is_bad_request_error(error.value)


@pytest.mark.parametrize(
    "output_ids,text,is_silent",
    [([SILENT_TOKEN_ID, 151645], "", True), ([3, 4, 151645], "w3 w4", False)],
)
def test_silent_token_alone_is_a_silent_answer(
    adapters: SimpleNamespace, output_ids: list[int], text: str, is_silent: bool
) -> None:
    data: AuraRequestData = adapters.request_builder(turn_payload(params={}))
    data.output_ids = output_ids
    data.finish_reason = "stop"

    answer = adapters.result_adapter(data).data

    assert (answer["text"], answer["is_silent"]) == (text, is_silent)
    assert answer["prompt_tokens"] == data.prompt_token_count


def test_checkpoint_without_the_silent_token_is_refused() -> None:
    """A stock Qwen3-VL tokenizer spells the silence marker as plain text, so its silent turns would be spoken."""
    validate_silent_token(SilentTokenTokenizer(), "aurateam/AURA")

    with pytest.raises(ValueError, match="serves AURA checkpoints"):
        validate_silent_token(FakeTokenizer(), "Qwen/Qwen3-VL-8B-Instruct")
