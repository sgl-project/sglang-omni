# SPDX-License-Identifier: Apache-2.0
"""The AURA request builder applies the caller's sampling, a per-request frame budget and the stage's device."""

import threading
from array import array
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.aura_omni import engine_builder
from sglang_omni.models.aura_omni.engine_builder import (
    SILENT_TOKEN,
    AuraRequestData,
    build_aura_stream_output,
    make_aura_scheduler_adapters,
    validate_silent_token,
)
from sglang_omni.models.aura_omni.request_builders import DECODE_STAGE, AuraTurn
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

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        return list(range(len(text.split())))

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
        self.state = threading.local()

    @property
    def current_device(self) -> torch.device | None:
        return getattr(self.state, "device", None)

    def set_device(self, device: torch.device) -> None:
        self.state.device = device


class FakeProcessor:
    """Records the frame budget and the thread's device each multimodal call ran with."""

    def __init__(self, platform: FakePlatform) -> None:
        self.video_config = dict(SHARED_VIDEO_CONFIG)
        self.platform = platform
        self.calls: list[tuple[dict[str, float], torch.device | None]] = []
        self.barrier: threading.Barrier | None = None

    async def process_mm_data_async(
        self,
        image_data: list[str | bytes],
        input_text: str,
        request_obj: SimpleNamespace,
    ) -> SimpleNamespace:
        if self.barrier is not None:
            self.barrier.wait()
        else:
            pass
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
            "min_p": 0.0,
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
        "vision_order": ["video"] * len(videos or []),
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


def test_aura_stage_params_outrank_the_top_level(
    adapters: SimpleNamespace,
) -> None:
    data: AuraRequestData = adapters.request_builder(
        turn_payload(
            params={
                "temperature": 0.9,
                "stage_params": {"aura": {"top_p": 0.3}},
                "stage_sampling": {"aura": {"temperature": 0.2}},
                "stream": False,
            }
        )
    )

    assert data.temperature == pytest.approx(0.2)
    assert data.top_p == pytest.approx(0.3)


def test_caller_stop_ids_reach_the_sampler(adapters: SimpleNamespace) -> None:
    data: AuraRequestData = adapters.request_builder(
        turn_payload(params={"stop_token_ids": [42], "stream": False})
    )

    assert set(data.req.sampling_params.stop_token_ids) == {42, 151645}


def test_stop_strings_are_a_bad_request(adapters: SimpleNamespace) -> None:
    with pytest.raises(ValueError, match="stage_sampling.aura.stop_token_ids"):
        adapters.request_builder(
            turn_payload(params={"stop": ["THE END"], "stream": False})
        )


def test_an_unrelated_stage_override_keeps_the_caller_stops(
    adapters: SimpleNamespace,
) -> None:
    data: AuraRequestData = adapters.request_builder(
        turn_payload(
            params={
                "stop_token_ids": [42],
                "stop": [],
                "stage_sampling": {
                    "aura": {"temperature": 0.2, "stop": [], "stop_token_ids": []}
                },
                "stream": False,
            }
        )
    )

    assert set(data.req.sampling_params.stop_token_ids) == {42, 151645}
    assert data.temperature == pytest.approx(0.2)


def test_a_stage_override_cannot_hide_a_stop_string(
    adapters: SimpleNamespace,
) -> None:
    with pytest.raises(ValueError, match="stage_sampling.aura.stop_token_ids"):
        adapters.request_builder(
            turn_payload(
                params={
                    "stop": ["THE END"],
                    "stage_sampling": {"aura": {"temperature": 0.2, "stop": []}},
                    "stream": False,
                }
            )
        )


def test_caller_seed_reaches_the_sampler(adapters: SimpleNamespace) -> None:
    data: AuraRequestData = adapters.request_builder(
        turn_payload(params={"seed": 7, "stream": False})
    )

    assert data.req.sampling_params.sampling_seed == 7


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("top_p", 0.0),
        ("top_k", -2),
        ("temperature", -1.0),
        ("top_k", float("inf")),
        ("temperature", float("nan")),
    ],
)
def test_sampling_outside_its_range_is_a_bad_request(
    adapters: SimpleNamespace, field: str, value: float
) -> None:
    with pytest.raises(ValueError, match="AURA Omni expects valid sampling"):
        adapters.request_builder(turn_payload(params={field: value, "stream": False}))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_new_tokens", 1.9),
        ("max_new_tokens", "4"),
        ("top_k", -1.9),
        ("top_k", 20.5),
    ],
)
def test_a_fractional_generation_override_is_a_bad_request(
    adapters: SimpleNamespace, field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=f"{field} must be an integer"):
        adapters.request_builder(
            turn_payload(
                params={"stage_sampling": {"aura": {field: value}}, "stream": False}
            )
        )


@pytest.mark.parametrize(("field", "value"), [("max_new_tokens", 8.0), ("top_k", 20.0)])
def test_a_whole_float_override_is_accepted(
    adapters: SimpleNamespace, field: str, value: float
) -> None:
    data: AuraRequestData = adapters.request_builder(
        turn_payload(
            params={"stage_sampling": {"aura": {field: value}}, "stream": False}
        )
    )

    assert getattr(data, field) == int(value)


@pytest.mark.parametrize("bad", ["invalid", None])
def test_malformed_stage_sampling_is_a_bad_request(
    adapters: SimpleNamespace, bad: object
) -> None:
    with pytest.raises(ValueError, match="AURA Omni expects valid sampling"):
        adapters.request_builder(
            turn_payload(
                params={
                    "stage_params": {"aura": {"temperature": bad}},
                    "stream": False,
                }
            )
        )


def test_min_p_is_a_bad_request(adapters: SimpleNamespace) -> None:
    with pytest.raises(ValueError, match="AURA Omni expects min_p=0"):
        adapters.request_builder(turn_payload(params={"min_p": 0.1, "stream": False}))


def test_a_non_integer_seed_is_a_bad_request(adapters: SimpleNamespace) -> None:
    with pytest.raises(ValueError, match="AURA Omni expects valid sampling"):
        adapters.request_builder(
            turn_payload(
                params={"stage_sampling": {"aura": {"seed": "7"}}, "stream": False}
            )
        )


def test_concurrent_frame_budgets_stay_isolated_on_the_stage_device(
    adapters: SimpleNamespace,
) -> None:
    adapters.processor.barrier = threading.Barrier(2, timeout=30)
    budgets = [{"fps": 1.0}, {"fps": 4.0}]

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(
            pool.map(
                lambda sampling: adapters.request_builder(
                    turn_payload(
                        params={}, videos=["clip.mp4"], video_sampling=sampling
                    )
                ),
                budgets,
            )
        )

    assert sorted(config["fps"] for config, _ in adapters.processor.calls) == [1.0, 4.0]
    assert [device for _, device in adapters.processor.calls] == [
        STAGE_DEVICE,
        STAGE_DEVICE,
    ]
    assert adapters.processor.video_config == SHARED_VIDEO_CONFIG


def test_a_request_without_a_budget_uses_the_shared_one(
    adapters: SimpleNamespace,
) -> None:
    adapters.request_builder(turn_payload(params={}, videos=["clip.mp4"]))

    assert adapters.processor.calls == [(SHARED_VIDEO_CONFIG, STAGE_DEVICE)]


def stream_request_data(
    adapters: SimpleNamespace, *, stream: bool, stop_token_ids: list[int]
) -> AuraRequestData:
    data: AuraRequestData = adapters.request_builder(
        turn_payload(params={"stream": stream, "stop_token_ids": stop_token_ids})
    )
    return data


@pytest.mark.parametrize(
    ("stream", "stop_token_ids", "middle_chunks", "output", "expected_tokens"),
    [
        (True, [], 0, 5, [5]),
        (True, [5], 0, 5, []),
        (True, [], 0, 151645, []),
        (True, [], 1, 5, []),
        (True, [], 0, None, []),
        (False, [], 0, 5, []),
    ],
)
def test_the_stream_adapter_emits_only_speakable_tokens(
    adapters: SimpleNamespace,
    stream: bool,
    stop_token_ids: list[int],
    middle_chunks: int,
    output: int | None,
    expected_tokens: list[int],
) -> None:
    data = stream_request_data(adapters, stream=stream, stop_token_ids=stop_token_ids)
    data.req.inflight_middle_chunks = middle_chunks

    messages = list(build_aura_stream_output("r1", data, SimpleNamespace(data=output)))

    assert [message.metadata["token_id"] for message in messages] == expected_tokens


def test_the_stream_adapter_delivers_one_token_to_decode(
    adapters: SimpleNamespace,
) -> None:
    data = stream_request_data(adapters, stream=True, stop_token_ids=[])

    messages = list(build_aura_stream_output("r1", data, SimpleNamespace(data=5)))

    assert len(messages) == 1
    message = messages[0]
    assert message.request_id == "r1"
    assert message.type == "stream"
    assert message.target == DECODE_STAGE
    assert message.metadata == {"token_id": 5}
    assert message.data.tolist() == [5]
    assert message.data.dtype is torch.long


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


def test_a_matched_stop_id_is_not_spoken(adapters: SimpleNamespace) -> None:
    data: AuraRequestData = adapters.request_builder(
        turn_payload(params={"stop_token_ids": [7], "stream": False})
    )
    data.output_ids = [3, 4, 7]
    data.finish_reason = "stop"

    answer = adapters.result_adapter(data).data

    assert answer["text"] == "w3 w4"
    assert answer["output_ids"] == [3, 4, 7]


def test_an_unmatched_trailing_id_is_kept(adapters: SimpleNamespace) -> None:
    data: AuraRequestData = adapters.request_builder(
        turn_payload(params={"stop_token_ids": [7], "stream": False})
    )
    data.output_ids = [3, 4, 5]
    data.finish_reason = "length"

    assert adapters.result_adapter(data).data["text"] == "w3 w4 w5"


def test_processor_padded_ids_keep_an_int64_array(
    adapters: SimpleNamespace,
) -> None:
    data: AuraRequestData = adapters.request_builder(
        turn_payload(params={}, videos=["clip.mp4"])
    )

    assert isinstance(data.req.origin_input_ids, array)
    assert data.req.origin_input_ids.typecode == "q"


@pytest.mark.parametrize("padded_input_ids", [None, [1, 2]])
def test_the_padding_fallback_keeps_an_int64_array(
    monkeypatch: pytest.MonkeyPatch, padded_input_ids: list[int] | None
) -> None:
    class Unpadded:
        @staticmethod
        def from_processor_output(processed: SimpleNamespace) -> SimpleNamespace:
            return SimpleNamespace(padded_input_ids=padded_input_ids)

    class PaddingModel:
        @staticmethod
        def pad_input_ids(input_ids: array, mm_inputs: object) -> array:
            assert isinstance(input_ids, array), type(input_ids)
            assert input_ids.typecode == "q", input_ids.typecode
            return array("q", [*input_ids, 99])

    platform = FakePlatform()
    monkeypatch.setattr(engine_builder, "current_platform", platform)
    monkeypatch.setattr(engine_builder, "MultimodalInputs", Unpadded)
    request_builder, _ = make_aura_scheduler_adapters(
        tokenizer=FakeTokenizer(),
        mm_processor=FakeProcessor(platform),
        model=PaddingModel(),
        device=STAGE_DEVICE,
        sampling_defaults={
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
        },
        stop_token_ids=[151645],
        max_new_tokens=256,
        context_length=2048,
    )

    data: AuraRequestData = request_builder(
        turn_payload(params={}, videos=["clip.mp4"])
    )

    assert isinstance(data.req.origin_input_ids, array)
    assert data.req.origin_input_ids.typecode == "q"
    assert data.req.origin_input_ids[-1] == 99
    assert data.input_ids.dtype is torch.long


def test_checkpoint_without_the_silent_token_is_refused() -> None:
    """A stock Qwen3-VL tokenizer spells the silence marker as plain text, so its silent turns would be spoken."""
    validate_silent_token(SilentTokenTokenizer(), "aurateam/AURA")

    with pytest.raises(ValueError, match="serves AURA checkpoints"):
        validate_silent_token(FakeTokenizer(), "Qwen/Qwen3-VL-8B-Instruct")
