# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.parakeet.model_runner import (
    ParakeetModelRunner,
    resolve_parakeet_architecture,
)

transformers = pytest.importorskip("transformers")
parakeet = pytest.importorskip("transformers.models.parakeet")

TINY_ENCODER = {
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "num_mel_bins": 80,
    "subsampling_conv_channels": 8,
}


def test_resolve_architecture_picks_the_parakeet_head() -> None:
    assert resolve_parakeet_architecture(["ParakeetForTDT"]) == "ParakeetForTDT"
    assert (
        resolve_parakeet_architecture(["SomethingElse", "ParakeetForCTC"])
        == "ParakeetForCTC"
    )


@pytest.mark.parametrize("architectures", [None, [], ["ParakeetEncoder"]])
def test_resolve_architecture_rejects_non_parakeet_checkpoints(
    architectures: list[str] | None,
) -> None:
    with pytest.raises(ValueError, match="Hugging Face format"):
        resolve_parakeet_architecture(architectures)


class RecordingProcessor:
    """Real Parakeet features; decoding records the ids instead of detokenizing."""

    def __init__(self) -> None:
        self.feature_extractor = parakeet.ParakeetFeatureExtractor()
        self.decoded: list[list[int]] = []

    def __call__(self, audio, **kwargs):
        return self.feature_extractor(audio, **kwargs)

    def batch_decode(self, sequences, *, skip_special_tokens: bool) -> list[str]:
        assert skip_special_tokens is True
        assert sequences.device.type == "cpu"
        self.decoded = sequences.tolist()
        return [f" row-{index} " for index in range(len(self.decoded))]


def tiny_runner(model: torch.nn.Module) -> ParakeetModelRunner:
    # note: skip __init__ so the test needs no checkpoint download.
    runner = ParakeetModelRunner.__new__(ParakeetModelRunner)
    runner.processor = RecordingProcessor()
    runner.sample_rate = 16000
    runner.min_samples = 512
    runner.device = torch.device("cpu")
    runner.dtype = torch.float32
    runner.model = model.eval()
    return runner


def tiny_ctc_model() -> torch.nn.Module:
    torch.manual_seed(0)
    config = parakeet.ParakeetCTCConfig(vocab_size=16, encoder_config=TINY_ENCODER)
    return transformers.ParakeetForCTC(config)


def tiny_tdt_model() -> torch.nn.Module:
    torch.manual_seed(0)
    config = parakeet.ParakeetTDTConfig(
        vocab_size=16,
        blank_token_id=15,
        pad_token_id=2,
        decoder_hidden_size=16,
        encoder_config=TINY_ENCODER,
    )
    model = transformers.ParakeetForTDT(config)
    # Hub checkpoints ship these in generation_config.json: decoding starts from
    # blank, and the duration logits appended after the vocabulary never win
    # the token argmax.
    model.generation_config.decoder_start_token_id = config.blank_token_id
    model.generation_config.suppress_tokens = list(
        range(config.vocab_size, config.vocab_size + len(config.durations))
    )
    return model


@pytest.mark.parametrize("make_model", [tiny_ctc_model, tiny_tdt_model])
def test_transcribe_returns_one_stripped_text_per_waveform(make_model) -> None:
    runner = tiny_runner(make_model())
    waveforms = [
        np.random.default_rng(0).standard_normal(16000).astype(np.float32),
        np.zeros(4000, dtype=np.float32),
    ]

    assert runner.transcribe(waveforms) == ["row-0", "row-1"]
    assert len(runner.processor.decoded) == 2


def test_transcribe_pads_clips_shorter_than_one_stft_window() -> None:
    runner = tiny_runner(tiny_ctc_model())

    assert runner.pad_short_waveform(np.ones(10, dtype=np.float32)).shape == (512,)
    assert runner.pad_short_waveform(np.ones(600, dtype=np.float32)).shape == (600,)
    assert runner.transcribe([np.ones(10, dtype=np.float32)]) == ["row-0"]


def test_transcribe_empty_batch_skips_the_model() -> None:
    runner = tiny_runner(SimpleNamespace(eval=lambda: None))

    assert runner.transcribe([]) == []


def test_batched_transducer_drops_emissions_on_padding_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A short clip decoded in a padded batch matches decoding it alone."""
    import sglang_omni.models.parakeet.model_runner as model_runner

    model = tiny_tdt_model()
    runner = tiny_runner(model)
    rng = np.random.default_rng(1)
    long_clip = rng.standard_normal(32000).astype(np.float32)
    short_clip = rng.standard_normal(9000).astype(np.float32)
    spoken = {model.config.pad_token_id, model.config.blank_token_id}

    def short_row(waveforms: list[np.ndarray]) -> list[int]:
        runner.transcribe(waveforms)
        return [token for token in runner.processor.decoded[-1] if token not in spoken]

    alone = short_row([short_clip])
    batched = short_row([long_clip, short_clip])
    monkeypatch.setattr(
        model_runner,
        "drop_tokens_past_valid_frames",
        lambda sequences, durations, valid_frames, pad_token_id: sequences,
    )
    unfiltered = short_row([long_clip, short_clip])

    assert alone and batched == alone
    # Without the filter, Transformers keeps emitting on the padding frames.
    assert len(unfiltered) > len(alone)


def test_drop_tokens_past_valid_frames_uses_step_start_frames() -> None:
    from sglang_omni.models.parakeet.model_runner import drop_tokens_past_valid_frames

    sequences = torch.tensor([[11, 8, 8, 8, 8], [11, 8, 8, 8, 8]])
    durations = torch.tensor([[0, 2, 2, 2, 2], [0, 1, 3, 2, 2]])

    kept = drop_tokens_past_valid_frames(
        sequences, durations, torch.tensor([10, 4]), pad_token_id=2
    )

    assert kept.tolist() == [[11, 8, 8, 8, 8], [11, 8, 8, 2, 2]]
