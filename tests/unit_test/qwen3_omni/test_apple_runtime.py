# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_omni import request_builders as qwen_request_builders
from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.models.qwen3_omni.request_builders import build_sglang_thinker_request
from tests.unit_test.fixtures.qwen_fakes import FakeQwenTokenizer, make_qwen_state


def set_mps(monkeypatch: pytest.MonkeyPatch, enabled: bool = True) -> None:
    monkeypatch.setattr(
        qwen_request_builders,
        "use_mlx",
        lambda: enabled,
    )


def test_fresh_apple_import_initializes_inductor_before_sglang_stubs() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sglang_omni.platforms;"
                "import sglang_omni.models.qwen3_omni.request_builders"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_importing_sglang_first_still_lets_sglang_omni_platforms_import() -> None:
    """SGLang installs a stub ``triton`` package on Apple, after which importing"""

    completed = subprocess.run(
        [sys.executable, "-c", "import sglang; import sglang_omni.platforms"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_scheduling_backend_import_never_fails_inside_the_apple_prewarm() -> None:
    """The scheduling backend package runs the same prewarm at import time."""

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sglang; import sglang_omni.scheduling.sglang_backend",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert "apple_torch_compat" not in completed.stderr, completed.stderr


def test_scheduling_backend_import_prewarms_inductor_before_sglang_stubs() -> None:
    """The supported order -- ``sglang_omni`` first -- must keep working, so the
    prewarm still wins the race for every real entry point."""

    completed = subprocess.run(
        [sys.executable, "-c", "import sglang_omni.scheduling.sglang_backend"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_inductor_prewarm_swallows_only_its_own_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard must cover exactly the prewarm import, and must be"""

    from sglang_omni.utils import apple_torch_compat

    calls: list[str] = []

    def _boom(name: str):
        calls.append(name)
        raise TypeError("unsupported operand type(s) for |: 'module' and 'type'")

    monkeypatch.setattr(apple_torch_compat, "platform", DarwinArm64Platform())
    monkeypatch.setattr(
        apple_torch_compat.torch.backends.mps, "is_available", lambda: True
    )
    monkeypatch.setattr(apple_torch_compat, "import_module", _boom)

    apple_torch_compat.prepare_torch_inductor_for_sglang()

    assert calls == ["torch._inductor.runtime.triton_heuristics"]
    with pytest.raises(TypeError):
        apple_torch_compat.import_module("torch._inductor.runtime.triton_heuristics")


def test_inductor_prewarm_is_skipped_entirely_off_apple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive pairing for the guard above: the prewarm must stay Apple-only,
    so a non-Darwin host never imports Torch Inductor at all."""

    from sglang_omni.utils import apple_torch_compat

    calls: list[str] = []
    monkeypatch.setattr(apple_torch_compat, "platform", LinuxX86Platform())
    monkeypatch.setattr(
        apple_torch_compat, "import_module", lambda name: calls.append(name)
    )

    apple_torch_compat.prepare_torch_inductor_for_sglang()

    assert calls == []


class DarwinArm64Platform:
    @staticmethod
    def system() -> str:
        return "Darwin"

    @staticmethod
    def machine() -> str:
        return "arm64"


class LinuxX86Platform:
    @staticmethod
    def system() -> str:
        return "Linux"

    @staticmethod
    def machine() -> str:
        return "x86_64"


@pytest.mark.parametrize(
    ("modality", "token_id", "model_input_key"),
    [
        ("image", 55, "image_embeds"),
        ("video", 66, "video_embeds"),
        ("audio", 77, "audio_embeds"),
    ],
)
def test_thinker_request_accepts_several_runs_of_one_modality(
    monkeypatch: pytest.MonkeyPatch,
    modality: str,
    token_id: int,
    model_input_key: str,
) -> None:
    """Two images (or two audio clips, or two videos) put two *separate* runs of"""

    data = build_apple_thinker_request(
        monkeypatch,
        input_ids=torch.tensor([token_id, token_id, 10, 11, token_id, token_id]),
        model_inputs={model_input_key: torch.ones((4, 4))},
        request_id="rid-multi",
    )

    positions = data.req._omni_mm_positions[
        modality
    ]  # noqa: leading-underscore - external runtime contract
    assert positions.tolist() == [0, 1, 4, 5]


def test_thinker_request_accepts_two_images_and_two_audio_clips_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interleaved modalities, each with more than one run, stay separable."""

    data = build_apple_thinker_request(
        monkeypatch,
        input_ids=torch.tensor([55, 10, 77, 11, 55, 12, 77]),
        model_inputs={
            "image_embeds": torch.ones((2, 4)),
            "audio_embeds": torch.ones((2, 4)),
        },
        request_id="rid-mixed",
    )

    positions = (
        data.req._omni_mm_positions
    )  # noqa: leading-underscore - external runtime contract
    assert positions["image"].tolist() == [0, 4]
    assert positions["audio"].tolist() == [2, 6]
    assert positions["video"].tolist() == []


@pytest.mark.parametrize(
    ("modality", "token_id", "model_input_key"),
    [
        ("image", 55, "image_embeds"),
        ("video", 66, "video_embeds"),
        ("audio", 77, "audio_embeds"),
    ],
)
def test_thinker_request_rejects_a_modality_row_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    modality: str,
    token_id: int,
    model_input_key: str,
) -> None:
    """Placement is by position, so a row/placeholder mismatch would silently
    drop or misplace encoder rows. It has to fail loudly instead."""

    with pytest.raises(
        ValueError,
        match=rf"{modality}.*rid-mismatch.*placeholders=2 encoder rows=3",
    ):
        build_apple_thinker_request(
            monkeypatch,
            input_ids=torch.tensor([token_id, 10, token_id]),
            model_inputs={model_input_key: torch.ones((3, 4))},
            request_id="rid-mismatch",
        )


@pytest.mark.parametrize(
    ("modality", "token_id", "model_input_key"),
    [
        ("image", 55, "image_embeds"),
        ("video", 66, "video_embeds"),
        ("audio", 77, "audio_embeds"),
    ],
)
def test_thinker_request_rejects_placeholders_without_encoder_rows(
    monkeypatch: pytest.MonkeyPatch,
    modality: str,
    token_id: int,
    model_input_key: str,
) -> None:
    """Without encoder rows the merge leaves the placeholder's *text* embedding
    in place and the model silently generates from noise."""

    del model_input_key
    with pytest.raises(
        ValueError,
        match=rf"{modality}.*rid-missing.*placeholders=1 encoder rows=0",
    ):
        build_apple_thinker_request(
            monkeypatch,
            input_ids=torch.tensor([token_id, 10]),
            model_inputs={},
            request_id="rid-missing",
        )


@pytest.mark.parametrize(
    ("modality", "token_id", "model_input_key"),
    [
        ("image", 55, "image_embeds"),
        ("video", 66, "video_embeds"),
        ("audio", 77, "audio_embeds"),
    ],
)
def test_thinker_request_rejects_encoder_rows_without_placeholders(
    monkeypatch: pytest.MonkeyPatch,
    modality: str,
    token_id: int,
    model_input_key: str,
) -> None:
    del token_id
    with pytest.raises(
        ValueError,
        match=rf"{modality}.*rid-orphan.*placeholders=0 encoder rows=2",
    ):
        build_apple_thinker_request(
            monkeypatch,
            input_ids=torch.tensor([10, 11]),
            model_inputs={model_input_key: torch.ones((2, 4))},
            request_id="rid-orphan",
        )


def test_thinker_request_counts_chunked_encoder_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Encoder merges may concatenate list-valued rows, so the count
    check has to sum the chunks rather than look at the first one."""

    data = build_apple_thinker_request(
        monkeypatch,
        input_ids=torch.tensor([55, 10, 55, 55]),
        model_inputs={"image_embeds": [torch.ones((1, 4)), torch.ones((2, 4))]},
        request_id="rid-chunked",
    )

    assert data.req._omni_mm_positions["image"].tolist() == [
        0,
        2,
        3,
    ]  # noqa: leading-underscore - external runtime contract


def test_non_apple_thinker_request_skips_modality_count_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive pairing for the guards above: off Apple the CUDA thinker owns
    this contract, so the request builder must stay a pass-through."""

    data = build_apple_thinker_request(
        monkeypatch,
        input_ids=torch.tensor([55, 10, 55]),
        model_inputs={"image_embeds": torch.ones((3, 4))},
        request_id="rid-cuda",
        apple=False,
    )

    assert data.req._omni_mm_positions["image"].tolist() == [
        0,
        2,
    ]  # noqa: leading-underscore - external runtime contract


def build_apple_thinker_request(
    monkeypatch: pytest.MonkeyPatch,
    *,
    input_ids: torch.Tensor,
    model_inputs: dict[str, object],
    request_id: str,
    apple: bool = True,
):
    set_mps(monkeypatch, apple)
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.normalize",
        lambda self, tokenizer: None,
    )
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.verify",
        lambda self, vocab_size: None,
    )
    monkeypatch.setattr(
        "sglang_omni.models.qwen3_omni.request_builders.compute_mrope_positions",
        lambda input_ids, model_inputs, thinker_config: (
            torch.zeros((3, input_ids.numel()), dtype=torch.long),
            torch.tensor(0),
        ),
    )
    state: Qwen3OmniPipelineState = make_qwen_state(
        prompt={"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)},
        thinker_inputs={"model_inputs": model_inputs},
    )
    return build_sglang_thinker_request(
        state,
        params={"max_new_tokens": 1},
        tokenizer=FakeQwenTokenizer(),
        vocab_size=256,
        request_id=request_id,
        thinker_config=SimpleNamespace(
            image_token_id=55,
            video_token_id=66,
            audio_token_id=77,
        ),
    )
