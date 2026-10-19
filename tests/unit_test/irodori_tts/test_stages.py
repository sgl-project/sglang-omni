# SPDX-License-Identifier: Apache-2.0
"""Stage batching, error isolation, and tensor invariant tests for Irodori-TTS."""

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.irodori_tts.batching import synthesize_batch
from sglang_omni.models.irodori_tts.model_loader import precision_dtype
from sglang_omni.models.irodori_tts.synthesis import (
    find_flattening_points,
    prepare_reference,
)
from sglang_omni.proto import OmniRequest, StagePayload


class Tokenizer:
    def batch_encode(self, texts: list[str], *, max_length: int):
        lengths = [min(len(text), max_length) for text in texts]
        width = max(lengths, default=1)
        token_ids = torch.zeros((len(texts), width), dtype=torch.long)
        failure_text = chr(22833) + chr(25943)
        for row, (text, length) in enumerate(zip(texts, lengths, strict=True)):
            if length:
                token_ids[row, :length] = 2 if text == failure_text else 1
        return token_ids, token_ids.ne(0)


class Codec:
    sample_rate = 10
    model = SimpleNamespace(hop_length=1)

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            latent.shape[0],
            1,
            latent.shape[1],
            dtype=latent.dtype,
            device=latent.device,
        )


def make_payload(
    text: str | None,
    *,
    seconds: float,
    speed: float = 1.0,
    reference_path: str | None = None,
) -> StagePayload:
    inputs = {} if text is None else {"text": text}
    if reference_path is not None:
        inputs["ref_audio"] = reference_path
    else:
        pass
    return StagePayload(
        request_id=f"irodori-{text}",
        request=OmniRequest(
            inputs=inputs,
            metadata={
                "tts_params": {"seconds": seconds, "speed": speed, "num_steps": 4}
            },
        ),
        data={},
    )


def model_configuration(*, use_speaker_condition: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        use_caption_condition=False,
        use_speaker_condition_resolved=use_speaker_condition,
        use_duration_predictor=False,
        latent_patch_size=1,
        latent_dim=2,
        patched_latent_dim=2,
    )


def model() -> SimpleNamespace:
    return SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32)


def test_batched_stage_keeps_item_errors_and_applies_speed(monkeypatch) -> None:
    sampled_batch_sizes: list[int] = []

    def sample(**kwargs):
        batch_size = kwargs["text_input_ids"].shape[0]
        sampled_batch_sizes.append(batch_size)
        return torch.zeros(batch_size, kwargs["sequence_length"], 2)

    monkeypatch.setattr(
        "sglang_omni.models.irodori_tts.batching.sample_euler_rf_cfg", sample
    )
    payloads = [
        make_payload(None, seconds=1.0),
        make_payload("こんにちは", seconds=2.0, speed=2.0),
        make_payload("おはようございます", seconds=2.0),
    ]

    results = synthesize_batch(
        payloads,
        model_configuration=model_configuration(),
        model=model(),
        tokenizer=Tokenizer(),
        caption_tokenizer=None,
        codec=Codec(),
        max_text_len=32,
        max_caption_len=32,
        max_ref_seconds=5.0,
        max_seconds=30.0,
        max_batch_size=8,
        max_batch_tokens=4096,
    )

    assert sampled_batch_sizes == [2]
    assert isinstance(results[0], ValueError)
    assert results[1] is payloads[1]
    assert results[2] is payloads[2]
    assert payloads[1].data["audio_waveform_shape"] == [10]
    assert payloads[2].data["audio_waveform_shape"] == [20]


@pytest.mark.parametrize(
    ("text_lengths", "batch_size"),
    [
        ([1], 1),
        ([1, 13], 2),
        ([1, 7, 24, 48], 4),
        ([1, 4, 16, 32, 64, 3, 24, 80], 8),
    ],
)
def test_batched_sampler_handles_varying_text_and_audio_lengths(
    monkeypatch, text_lengths: list[int], batch_size: int
) -> None:
    observed: list[dict[str, object]] = []

    def sample(**kwargs):
        observed.append(
            {
                "batch_shape": tuple(kwargs["text_input_ids"].shape),
                "text_lengths": kwargs["text_mask"].sum(dim=1).tolist(),
                "sequence_length": kwargs["sequence_length"],
                "latent_lengths": kwargs["latent_mask"].sum(dim=1).tolist(),
            }
        )
        return torch.zeros(
            kwargs["text_input_ids"].shape[0], kwargs["sequence_length"], 2
        )

    monkeypatch.setattr(
        "sglang_omni.models.irodori_tts.batching.sample_euler_rf_cfg", sample
    )
    payloads = [
        make_payload(chr(12354) * text_length, seconds=0.5 + index * 0.05)
        for index, text_length in enumerate(text_lengths)
    ]

    results = synthesize_batch(
        payloads,
        model_configuration=model_configuration(),
        model=model(),
        tokenizer=Tokenizer(),
        caption_tokenizer=None,
        codec=Codec(),
        max_text_len=256,
        max_caption_len=256,
        max_ref_seconds=5.0,
        max_seconds=30.0,
        max_batch_size=batch_size,
        max_batch_tokens=100_000,
    )

    requested_latent_lengths = [
        max(
            1,
            int(
                max(0.5, payload.request.metadata["tts_params"]["seconds"])
                * Codec.sample_rate
            ),
        )
        for payload in payloads
    ]
    assert len(observed) == 1
    assert observed[0]["batch_shape"] == (batch_size, max(text_lengths))
    assert observed[0]["text_lengths"] == text_lengths
    assert observed[0]["latent_lengths"] == requested_latent_lengths
    assert observed[0]["sequence_length"] == max(requested_latent_lengths)
    assert all(
        result is payload for result, payload in zip(results, payloads, strict=True)
    )


def test_batch_failure_returns_per_item_errors(monkeypatch) -> None:
    def sample(**kwargs):
        token_ids = kwargs["text_input_ids"]
        if token_ids.shape[0] > 1:
            raise RuntimeError("simulated batched kernel failure")
        if int(token_ids[0, 0]) == 2:
            raise ValueError("simulated request failure")
        return torch.zeros(1, kwargs["sequence_length"], 2)

    monkeypatch.setattr(
        "sglang_omni.models.irodori_tts.batching.sample_euler_rf_cfg", sample
    )
    monkeypatch.setattr(
        "sglang_omni.models.irodori_tts.synthesis.sample_euler_rf_cfg", sample
    )
    payloads = [
        make_payload("こんにちは", seconds=0.5),
        make_payload("失敗", seconds=0.5),
    ]

    results = synthesize_batch(
        payloads,
        model_configuration=model_configuration(),
        model=model(),
        tokenizer=Tokenizer(),
        caption_tokenizer=None,
        codec=Codec(),
        max_text_len=32,
        max_caption_len=32,
        max_ref_seconds=5.0,
        max_seconds=30.0,
        max_batch_size=8,
        max_batch_tokens=4096,
    )

    assert results[0] is payloads[0]
    assert isinstance(results[1], ValueError)
    assert "simulated request failure" in str(results[1])


def test_batched_sampler_crops_reference_padding_per_subbatch(monkeypatch) -> None:
    sampled_reference_shapes: list[tuple[tuple[int, int], tuple[int, int]]] = []

    def prepare_reference(audio_paths, *_args):
        length = 4 if audio_paths[0].startswith("short") else 400
        return (
            torch.ones(1, length, 2),
            torch.ones(1, length, dtype=torch.bool),
        )

    def sample(**kwargs):
        sampled_reference_shapes.append(
            (
                tuple(kwargs["ref_latent"].shape[:2]),
                tuple(kwargs["ref_mask"].shape),
            )
        )
        return torch.zeros(
            kwargs["text_input_ids"].shape[0], kwargs["sequence_length"], 2
        )

    monkeypatch.setattr(
        "sglang_omni.models.irodori_tts.synthesis.prepare_reference",
        prepare_reference,
    )
    monkeypatch.setattr(
        "sglang_omni.models.irodori_tts.batching.sample_euler_rf_cfg", sample
    )
    payloads = [
        make_payload(
            f"request-{index}", seconds=1.0, reference_path=f"short-{index}.wav"
        )
        for index in range(5)
    ]
    payloads.append(make_payload("long", seconds=1.0, reference_path="long.wav"))

    results = synthesize_batch(
        payloads,
        model_configuration=model_configuration(use_speaker_condition=True),
        model=model(),
        tokenizer=Tokenizer(),
        caption_tokenizer=None,
        codec=Codec(),
        max_text_len=32,
        max_caption_len=32,
        max_ref_seconds=50.0,
        max_seconds=30.0,
        max_batch_size=8,
        max_batch_tokens=100,
    )

    assert sampled_reference_shapes == [
        ((5, 4), (5, 4)),
        ((1, 400), (1, 400)),
    ]
    assert all(
        result is payload for result, payload in zip(results, payloads, strict=True)
    )


def test_flattening_windows_are_vectorized_across_padded_batch() -> None:
    latent = torch.ones(3, 12, 4)
    latent[0, 7:] = 0
    latent[1] = 1
    latent[2] = 0
    lengths = [12, 10, 3]

    assert find_flattening_points(latent, lengths, window_size=4) == [7, 10, 0]


def test_partial_tail_window_uses_its_actual_length() -> None:
    latent = torch.full((1, 100, 2), 0.105)

    assert find_flattening_points(latent, [100]) == [100]


@pytest.mark.parametrize(
    ("precision", "device", "expected"),
    [
        ("fp32", torch.device("cpu"), torch.float32),
        ("bf16", torch.device("cuda"), torch.bfloat16),
        ("bf16", torch.device("xpu"), torch.bfloat16),
    ],
)
def test_precision_dtype_matches_supported_device(
    precision: str, device: torch.device, expected: torch.dtype
) -> None:
    assert precision_dtype(precision, device) is expected


def test_bfloat16_precision_is_rejected_on_cpu() -> None:
    with pytest.raises(ValueError, match="requires a CUDA or XPU device"):
        precision_dtype("bf16", torch.device("cpu"))


def test_reference_tensors_followmodel_device_and_dtype() -> None:
    configuration = SimpleNamespace(
        use_speaker_condition_resolved=True,
        speaker_patch_size=2,
        patched_latent_dim=64,
    )
    model = SimpleNamespace(device=torch.device("cpu"), dtype=torch.bfloat16)

    latent, mask = prepare_reference([], configuration, model, None, 5.0)

    assert latent.device == model.device
    assert latent.dtype == model.dtype
    assert mask.device == model.device
    assert mask.dtype is torch.bool
    assert not mask.any()
