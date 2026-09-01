# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from sglang_omni.config import AudioChunkingConfig, ResolvedAudioChunking
from sglang_omni.serve import transcription_chunking
from sglang_omni.serve.transcription_chunking import (
    RMSSplitter,
    Span,
    check_total_duration,
    encode_wav,
    needs_chunking,
    plan_audio_chunks,
)
from sglang_omni.utils.audio import load_audio

_ENABLED = ResolvedAudioChunking(allow_audio_chunking=True, max_audio_clip_s=60.0)

_SAMPLE_RATE = 16000
_ENERGY_WINDOW = 1600  # 100ms at 16 kHz


def validate_spans(
    spans: list[Span], total_samples: int, max_chunk_samples: int
) -> None:
    """Assert the rules every splitter's output has to follow:

    1. spans are no gap and no overlap (cover [0, total_samples) exactly);
    2. no span is longer than max_chunk_samples
    3. no span is empty;
    4. same input, same output (can't be checked from one call -- see
       test_split_is_deterministic).

    Why so paranoid about gaps: a gap means that stretch of audio just
    vanishes from the transcript, silently. That's the exact bug chunking
    exists to kill, so every splitter test funnels its output through here.
    """
    expected_start = 0
    for index, (start, end) in enumerate(spans):
        if start != expected_start:
            raise AssertionError(
                f"chunk {index} starts at sample {start}, expected {expected_start}"
            )
        if end <= start:
            raise AssertionError(f"chunk {index} is empty: [{start}, {end})")
        if end - start > max_chunk_samples:
            raise AssertionError(
                f"chunk {index} spans {end - start} samples, over the "
                f"{max_chunk_samples} limit"
            )
        expected_start = end
    if expected_start != total_samples:
        raise AssertionError(
            f"chunks cover {expected_start} of {total_samples} samples"
        )


def _speech(num_samples: int, seed: int = 0) -> np.ndarray:
    """Loud, non-silent audio."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.5, 0.5, num_samples).astype(np.float32)


def _silence(num_samples: int) -> np.ndarray:
    return np.zeros(num_samples, dtype=np.float32)


def test_pipeline_config_leaves_chunking_off_by_default() -> None:
    from sglang_omni.config import PipelineConfig, StageConfig

    config = PipelineConfig(
        model_path="dummy",
        stages=[
            StageConfig(
                name="asr", factory_path="pkg.make_stage", process="asr", terminal=True
            )
        ],
    )
    assert config.resolved_audio_chunking.allow_audio_chunking is False


def test_qwen3_asr_pipeline_declaresResolvedAudioChunking(monkeypatch) -> None:
    from sglang_omni.platforms import current_platform

    # Keep this declaration test independent of the host backend. The Qwen
    # Torch-MPS override is covered explicitly below.
    monkeypatch.setattr(current_platform, "is_mps", lambda: False)

    from sglang_omni.models.qwen3_asr.config import Qwen3ASRPipelineConfig

    declared = Qwen3ASRPipelineConfig(model_path="dummy").resolved_audio_chunking
    assert declared.allow_audio_chunking is True
    # 30s is a scheduling choice, not a context limit (the context is sized
    # for the model's native 1,200s): short chunks batch well and keep one
    # long upload from monopolizing the engine.
    assert declared.max_audio_clip_s == 30.0
    # The native limit feeds the streaming path, which cannot chunk and so
    # accepts single requests up to what the engine context fits.
    assert declared.max_native_clip_s == 1200.0


def test_qwen3_asr_torch_mps_limits_runtime_audio_policy(monkeypatch) -> None:
    import sglang.srt.utils.tensor_bridge as tensor_bridge

    from sglang_omni.models.qwen3_asr.config import Qwen3ASRPipelineConfig
    from sglang_omni.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_mps", lambda: True)
    monkeypatch.setattr(tensor_bridge, "use_mlx", lambda: False)

    resolved = Qwen3ASRPipelineConfig(
        model_path="Qwen/Qwen3-ASR-0.6B"
    ).resolved_audio_chunking

    assert resolved.max_audio_clip_s == 30.0
    assert resolved.max_native_clip_s == 60.0
    assert resolved.max_total_audio_s == 60.0


def test_qwen3_asr_torch_mps_rejects_unsafe_chunk_length(monkeypatch) -> None:
    import sglang.srt.utils.tensor_bridge as tensor_bridge

    from sglang_omni.models.qwen3_asr.config import Qwen3ASRPipelineConfig
    from sglang_omni.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_mps", lambda: True)
    monkeypatch.setattr(tensor_bridge, "use_mlx", lambda: False)

    with pytest.raises(ValueError, match="up to 60s"):
        Qwen3ASRPipelineConfig(
            model_path="Qwen/Qwen3-ASR-0.6B",
            audio_chunking=AudioChunkingConfig(max_audio_clip_s=60.1),
        ).resolved_audio_chunking


@pytest.mark.parametrize(("is_mps", "use_mlx"), [(False, False), (True, True)])
def test_qwen3_asr_non_torch_mps_keeps_native_audio_policy(
    monkeypatch, is_mps: bool, use_mlx: bool
) -> None:
    import sglang.srt.utils.tensor_bridge as tensor_bridge

    from sglang_omni.models.qwen3_asr.config import Qwen3ASRPipelineConfig
    from sglang_omni.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_mps", lambda: is_mps)
    monkeypatch.setattr(tensor_bridge, "use_mlx", lambda: use_mlx)

    resolved = Qwen3ASRPipelineConfig(
        model_path="Qwen/Qwen3-ASR-0.6B"
    ).resolved_audio_chunking

    assert resolved.max_audio_clip_s == 30.0
    assert resolved.max_native_clip_s == 1200.0
    assert resolved.max_total_audio_s == 3600.0


def test_whisper_asr_pipeline_declares_chunking() -> None:
    from sglang_omni.models.whisper_asr.config import WhisperASRPipelineConfig

    declared = WhisperASRPipelineConfig(model_path="dummy").resolved_audio_chunking
    assert declared.allow_audio_chunking is True
    # For Whisper the chunk length IS the native limit: the feature extractor
    # truncates everything past its 30s mel window, so without chunking a
    # longer upload silently loses all audio past the first 30 seconds.
    assert declared.max_audio_clip_s == 30.0
    assert declared.max_native_clip_s == 30.0
    # Whisper hallucinates on very short clips; keep the tail at least 1s.
    assert declared.min_tail_s == 1.0
    assert declared.condition_on_previous_text is False


def test_disabled_config_never_chunks() -> None:
    config = ResolvedAudioChunking()

    assert config.allow_audio_chunking is False
    assert needs_chunking(600.0, config) is False


def test_unknown_duration_is_left_alone() -> None:
    # 0.0 is what _probe_audio_duration returns when it cannot read the header.
    assert needs_chunking(0.0, _ENABLED) is False
    assert needs_chunking(-1.0, _ENABLED) is False


def test_duration_exactly_at_the_clip_limit_is_not_chunked() -> None:
    assert needs_chunking(60.0, _ENABLED) is False


def test_duration_past_the_clip_limit_is_chunked() -> None:
    assert needs_chunking(60.1, _ENABLED) is True


def test_chunk_samples_floors_and_stays_positive() -> None:
    assert _ENABLED.chunk_samples(16000) == 960_000
    assert ResolvedAudioChunking(max_audio_clip_s=0.5).chunk_samples(16000) == 8000
    # Rates so low that the product rounds to zero still yield one sample.
    assert ResolvedAudioChunking(max_audio_clip_s=0.5).chunk_samples(1) == 1


def test_total_duration_limit_can_be_disabled() -> None:
    config = ResolvedAudioChunking(allow_audio_chunking=True, max_total_audio_s=None)

    check_total_duration(100_000.0, config)


def test_total_duration_at_the_limit_is_accepted() -> None:
    config = ResolvedAudioChunking(allow_audio_chunking=True, max_total_audio_s=3600.0)

    check_total_duration(3600.0, config)


def test_total_duration_past_the_limit_is_rejected() -> None:
    config = ResolvedAudioChunking(allow_audio_chunking=True, max_total_audio_s=3600.0)

    with pytest.raises(ValueError) as exc_info:
        check_total_duration(3600.5, config)

    message = str(exc_info.value)
    assert "3600" in message
    assert "3600.500" in message
    # Shared wording with the engine-side limit keeps both mapped to HTTP 400.
    assert "accepts audio up to" in message


def test_total_limit_below_clip_limit_is_rejected_at_config_time() -> None:
    with pytest.raises(ValueError, match="max_total_audio_s"):
        AudioChunkingConfig(max_audio_clip_s=60.0, max_total_audio_s=30.0)


def test_clip_length_past_the_native_limit_is_rejected_at_config_time() -> None:
    from sglang_omni.models.whisper_asr.config import WhisperASRPipelineConfig

    with pytest.raises(ValueError, match="native clip limit"):
        WhisperASRPipelineConfig(
            model_path="dummy",
            audio_chunking=AudioChunkingConfig(max_audio_clip_s=60.0),
        )


def test_stream_clip_limit_falls_back_to_the_chunk_length() -> None:
    assert ResolvedAudioChunking(max_audio_clip_s=60.0).stream_clip_limit_s == 60.0
    assert (
        ResolvedAudioChunking(
            max_audio_clip_s=60.0, max_native_clip_s=1200.0
        ).stream_clip_limit_s
        == 1200.0
    )


@pytest.mark.parametrize(
    "field, value",
    [
        ("max_audio_clip_s", 0.0),
        ("max_audio_clip_s", -1.0),
        ("max_total_audio_s", 0.0),
    ],
)
def test_out_of_range_config_values_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        AudioChunkingConfig(**{field: value})


def test_audio_shorter_than_one_chunk_is_not_split() -> None:
    waveform = _speech(_SAMPLE_RATE)

    spans = RMSSplitter().split(waveform, _SAMPLE_RATE, 2 * _SAMPLE_RATE)

    assert spans == [(0, _SAMPLE_RATE)]


def test_audio_exactly_one_chunk_is_not_split() -> None:
    waveform = _speech(2 * _SAMPLE_RATE)

    spans = RMSSplitter().split(waveform, _SAMPLE_RATE, 2 * _SAMPLE_RATE)

    assert spans == [(0, 2 * _SAMPLE_RATE)]


def test_split_lands_on_the_quiet_window() -> None:
    # 2.5s of speech with a silent stretch at [13600, 15200), which is the
    # second of the two energy windows inside the search region [12000, 16000).
    waveform = _speech(40_000)
    waveform[13_600:15_200] = _silence(1600)

    spans = RMSSplitter(
        search_window_s=0.25, energy_window_samples=_ENERGY_WINDOW
    ).split(waveform, _SAMPLE_RATE, _SAMPLE_RATE)

    cut = spans[0][1]
    assert 13_600 <= cut < 15_200
    validate_spans(spans, 40_000, _SAMPLE_RATE)


def test_split_stays_inside_the_search_window_without_any_pause() -> None:
    # Uniformly loud audio: there is no real pause to find, so the cut may land
    # anywhere in the search region but must not leave it.
    waveform = _speech(40_000)

    spans = RMSSplitter(search_window_s=0.25).split(
        waveform, _SAMPLE_RATE, _SAMPLE_RATE
    )

    assert 12_000 <= spans[0][1] <= _SAMPLE_RATE


def test_split_is_deterministic() -> None:
    waveform = _speech(40_000)
    splitter = RMSSplitter(search_window_s=0.25)

    first = splitter.split(waveform, _SAMPLE_RATE, _SAMPLE_RATE)
    second = splitter.split(waveform, _SAMPLE_RATE, _SAMPLE_RATE)

    assert first == second


def test_search_window_wider_than_the_clip_still_advances() -> None:
    waveform = _speech(40_000)

    spans = RMSSplitter(search_window_s=100.0).split(
        waveform, _SAMPLE_RATE, _SAMPLE_RATE
    )

    # The search floor is start + 1, so the first cut cannot collapse to 0.
    assert spans[0][1] >= 1
    validate_spans(spans, 40_000, _SAMPLE_RATE)


def test_single_sample_chunks_terminate() -> None:
    # Degenerate but reachable through misconfiguration: without the "must
    # advance" floor this loops forever.
    waveform = _speech(10)

    spans = RMSSplitter().split(waveform, _SAMPLE_RATE, 1)

    assert spans == [(index, index + 1) for index in range(10)]


@pytest.mark.parametrize(
    "splitter",
    [
        RMSSplitter(),
        RMSSplitter(search_window_s=0.0),
        RMSSplitter(search_window_s=0.25, energy_window_samples=1),
    ],
    ids=["rms-default", "rms-no-search", "rms-tiny-window"],
)
@pytest.mark.parametrize(
    "waveform",
    [
        np.zeros(0, dtype=np.float32),
        _speech(1),
        _silence(40_000),
        _speech(40_000),
        _speech(_SAMPLE_RATE * 10, seed=7),
    ],
    ids=["empty", "one-sample", "all-silence", "loud", "ten-chunks"],
)
def test_split_contract_holds(splitter, waveform: np.ndarray) -> None:
    max_chunk_samples = _SAMPLE_RATE

    spans = splitter.split(waveform, _SAMPLE_RATE, max_chunk_samples)

    validate_spans(spans, int(waveform.shape[-1]), max_chunk_samples)


@pytest.mark.parametrize(
    "spans, message",
    [
        ([(0, 10), (12, 20)], "expected 10"),
        ([(0, 10), (10, 10), (10, 20)], "is empty"),
        ([(0, 30)], "over the"),
        ([(0, 10)], "cover 10 of 20"),
    ],
    ids=["gap", "empty-span", "oversized", "short-coverage"],
)
def test_validate_spans_rejects_broken_output(spans: list[Span], message: str) -> None:
    # The helper is the assertion every other splitter test leans on, so it
    # needs its own coverage: a lenient checker would make them all vacuous.
    with pytest.raises(AssertionError, match=message):
        validate_spans(spans, 20, 20)


@pytest.mark.parametrize(
    "total_samples",
    [_SAMPLE_RATE + 1, _SAMPLE_RATE + 800, 2 * _SAMPLE_RATE - 1],
    ids=["one-sample-tail", "half-window-tail", "near-full-tail"],
)
def test_sub_minimum_tail_is_avoided_by_shifting_the_cut(total_samples: int) -> None:
    # Audio ending just past a boundary would leave a stub final chunk; a
    # 0.1s clip transcribes to garbage. The splitter pulls the previous cut
    # earlier instead, so the tail is worth transcribing and every span
    # still fits the limit.
    waveform = _speech(total_samples)

    spans = RMSSplitter(search_window_s=0.25).split(
        waveform, _SAMPLE_RATE, _SAMPLE_RATE, min_tail_s=0.5
    )

    validate_spans(spans, total_samples, _SAMPLE_RATE)
    min_tail = _SAMPLE_RATE // 2  # 0.5s
    last_start, last_end = spans[-1]
    assert last_end - last_start >= min_tail


def test_min_tail_shift_skips_degenerate_chunk_sizes() -> None:
    # When the chunk limit itself is below the tail minimum, shifting would
    # produce over-limit spans; the splitter leaves the cuts alone instead.
    waveform = _speech(10)

    spans = RMSSplitter().split(waveform, _SAMPLE_RATE, 1, min_tail_s=0.5)

    assert spans == [(index, index + 1) for index in range(10)]


_PLAN_CONFIG = ResolvedAudioChunking(allow_audio_chunking=True, max_audio_clip_s=1.0)
_PLAN_SPLITTER = RMSSplitter(search_window_s=0.25)


def _wav_bytes(waveform: np.ndarray, sample_rate: int = _SAMPLE_RATE) -> bytes:
    return encode_wav(waveform, sample_rate)


def _plan(waveform: np.ndarray, sample_rate: int = _SAMPLE_RATE):
    return plan_audio_chunks(
        _wav_bytes(waveform, sample_rate),
        _PLAN_CONFIG,
        splitter=_PLAN_SPLITTER,
    )


def test_plan_returns_none_when_the_decoded_audio_fits() -> None:
    # needs_chunking decides from the probed header duration; the decoded
    # waveform is the authority, and a clip that fits falls back to the
    # untouched single-request path.
    assert _plan(_speech(_SAMPLE_RATE)) is None


def test_plan_splits_long_audio_into_timed_spans() -> None:
    plan = _plan(_speech(40_000))

    assert plan is not None
    assert plan.sample_rate == _SAMPLE_RATE
    assert plan.duration_s == pytest.approx(2.5)
    assert [span.index for span in plan.spans] == list(range(len(plan.spans)))
    for span in plan.spans:
        assert span.start_s == pytest.approx(span.start_sample / _SAMPLE_RATE)
        assert span.end_s == pytest.approx(span.end_sample / _SAMPLE_RATE)


def test_plan_spans_cover_the_whole_upload() -> None:
    plan = _plan(_speech(40_000))

    assert plan is not None
    validate_spans(
        [(span.start_sample, span.end_sample) for span in plan.spans],
        40_000,
        _SAMPLE_RATE,
    )
    assert plan.spans[0].start_s == 0.0
    assert plan.spans[-1].end_s == pytest.approx(plan.duration_s)


def test_plan_marks_noise_floor_chunks_as_speechless() -> None:
    # A TTS-runaway-shaped clip: a short sentence, then a long ~-70 dBFS
    # noise floor whose chunks must never reach the engine (they hallucinate).
    rng = np.random.default_rng(0)
    speech = _speech(_SAMPLE_RATE // 2)
    noise_floor = rng.uniform(-3e-4, 3e-4, 2 * _SAMPLE_RATE).astype(np.float32)
    plan = _plan(np.concatenate([speech, noise_floor]))

    assert plan is not None
    assert plan.spans[0].has_speech is True
    # The speech ends at 0.5s and every cut lands past 0.75s, so all later
    # spans hold nothing but the noise floor.
    for span in plan.spans[1:]:
        assert span.has_speech is False


def test_plan_keeps_quiet_speech() -> None:
    # -40 dBFS: very quiet but real speech, far above the skip threshold.
    plan = _plan(_speech(40_000) * 0.01)

    assert plan is not None
    for span in plan.spans:
        assert span.has_speech is True


def test_encoded_chunk_round_trips_sample_for_sample() -> None:
    # The core correctness claim of planning: the bytes handed to the engine
    # are exactly the slice of audio the span describes.
    waveform = _speech(40_000)
    plan = _plan(waveform)

    assert plan is not None
    for span in plan.spans:
        decoded = load_audio(plan.encode(span), target_sample_rate=_SAMPLE_RATE)
        np.testing.assert_array_equal(
            decoded, waveform[span.start_sample : span.end_sample]
        )


def test_encoded_chunk_is_riff_wav() -> None:
    plan = _plan(_speech(40_000))

    assert plan is not None
    chunk = plan.encode(plan.spans[0])
    # RIFF/WAVE is what load_audio's dependency-free fast path looks for.
    assert chunk[:4] == b"RIFF"
    assert chunk[8:12] == b"WAVE"


def test_chunk_audio_is_encoded_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    real_encode_wav = transcription_chunking.encode_wav

    def counting_encode_wav(waveform, sample_rate):
        nonlocal calls
        calls += 1
        return real_encode_wav(waveform, sample_rate)

    monkeypatch.setattr(transcription_chunking, "encode_wav", counting_encode_wav)
    plan = _plan(_speech(40_000))

    assert plan is not None
    assert len(plan.spans) > 1
    assert calls == 0

    plan.encode(plan.spans[0])
    assert calls == 1


def test_plan_resamples_to_the_target_rate() -> None:
    # 2.5s at 8 kHz stays 2.5s of audio but becomes 40k samples at 16 kHz, so
    # the spans must be expressed in the resampled timeline.
    plan = _plan(_speech(20_000), sample_rate=8000)

    assert plan is not None
    assert plan.sample_rate == _SAMPLE_RATE
    assert plan.duration_s == pytest.approx(2.5)
    assert plan.spans[-1].end_sample == 40_000


@pytest.mark.parametrize(
    "parts, expected",
    [
        # Spaced scripts get one space at the seam.
        (["hello", "world"], "hello world"),
        # CJK joins directly; a space here would be a CER insertion error.
        (["我们今天讨论的是", "长音频的切分"], "我们今天讨论的是长音频的切分"),
        # Mixed seam: CJK on either side means no space.
        (["...的是", "OpenAI 的方案"], "...的是OpenAI 的方案"),
        (["the next is", "中文内容"], "the next is中文内容"),
        # Full-width punctuation counts as CJK; half-width does not.
        (["转写结束。", "下面继续"], "转写结束。下面继续"),
        (["that is done.", "The next part"], "that is done. The next part"),
        # Digits behave like spaced script.
        (["chapter 1", "2 comes after"], "chapter 1 2 comes after"),
        # Korean is wide in east_asian_width terms but its orthography DOES
        # separate words with spaces -- a seam must not glue two words.
        (["안녕하세요", "만나서 반갑습니다"], "안녕하세요 만나서 반갑습니다"),
        # Thai is narrow but writes without word spaces -- a seam must not
        # invent one.
        (["สวัสดีครับ", "ผมชื่อมาก"], "สวัสดีครับผมชื่อมาก"),
        # Whitespace inside parts is stripped so the rule owns the seam.
        (["  hello  ", "  world  "], "hello world"),
        # Empty / all-whitespace parts (an all-silence chunk) are skipped.
        (["hello", "   ", "world"], "hello world"),
        (["hello", "", "world"], "hello world"),
        (["  only one  "], "only one"),
        ([], ""),
    ],
    ids=[
        "english",
        "chinese",
        "cjk-then-latin",
        "latin-then-cjk",
        "fullwidth-period",
        "halfwidth-period",
        "digits",
        "korean-spaced",
        "thai-unspaced",
        "padded",
        "blank-part",
        "empty-part",
        "single",
        "empty-list",
    ],
)
def test_join_transcript_parts(parts: list[str], expected: str) -> None:
    from sglang_omni.serve.transcription_chunking import join_transcript_parts

    assert join_transcript_parts(parts) == expected
