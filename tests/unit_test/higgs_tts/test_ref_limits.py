# SPDX-License-Identifier: Apache-2.0
"""Reference-input limits: trimming, prompt budget, download cap, fingerprints.

Covers the review findings on the reference-handling optimizations: the
best-window trim (selection, determinism, disable switch, exact-length
guarantee), the engine-context prompt budget (boundaries, max_new_tokens
clamping, constant drift vs the engine builder), the capped URL download,
the vectorized fingerprint's byte equivalence with the legacy loops, the
anchor-F0 decode truncation, and the banded DTW's agreement with the full
matrix on calibration-like drift.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from sglang_omni.models.higgs_tts import fusion_reference as fr
from sglang_omni.models.higgs_tts import utils as higgs_utils

# --- to_codes_TN value-range validation --------------------------------------


def test_to_codes_tn_rejects_out_of_range_values():
    with pytest.raises(ValueError, match=r"within \[0, 1025\]"):
        higgs_utils.to_codes_TN([[-1] * 8, [0] * 8], 8)
    with pytest.raises(ValueError, match=r"within \[0, 1025\]"):
        higgs_utils.to_codes_TN([[70000] * 8], 8)
    # Boundary values (data max 1023 + BOC/EOC 1024/1025) are legal.
    ok = higgs_utils.to_codes_TN([[0, 1023, 1024, 1025, 1, 2, 3, 4]], 8)
    assert ok is not None and ok.shape == (1, 8)


# --- fingerprint byte equivalence with the legacy loops ----------------------


_BOUNDARY_ROWS = [
    [0, 1, 2, 1023, 1024, 1025, 513, 999],
    [7, 8, 9, 10, 11, 12, 13, 14],
] * 25


def test_fusion_ref_fingerprint_matches_legacy_loop():
    h = hashlib.blake2b(digest_size=16)
    for row in _BOUNDARY_ROWS:
        for c in row:
            h.update(int(c).to_bytes(2, "little"))
    assert fr.ref_fingerprint({"codes_delayed": _BOUNDARY_ROWS}) == h.hexdigest()


def test_request_builders_fingerprint_matches_legacy_loop():
    pytest.importorskip("sglang")
    from sglang_omni.models.higgs_tts.request_builders import _ref_audio_fingerprint

    buf = bytearray(2 * sum(len(r) for r in _BOUNDARY_ROWS))
    i = 0
    for row in _BOUNDARY_ROWS:
        for c in row:
            buf[i] = c & 0xFF
            buf[i + 1] = (c >> 8) & 0xFF
            i += 2
    legacy = hashlib.blake2b(bytes(buf), digest_size=16).hexdigest()
    assert _ref_audio_fingerprint(_BOUNDARY_ROWS) == legacy
    assert _ref_audio_fingerprint(None) is None
    assert _ref_audio_fingerprint([]) is None


# --- anchor-F0 decode truncation ---------------------------------------------


def test_anchor_f0_truncates_long_reference_decode(monkeypatch):
    orch = fr.FusionReferenceOrchestrator()
    seen: list[int] = []

    def fake_delayed_to_wav(delayed):
        seen.append(int(delayed.shape[0]))
        return np.zeros(24000)

    monkeypatch.setattr(orch, "_delayed_to_wav", fake_delayed_to_wav)
    monkeypatch.setattr(fr, "median_f0", lambda wav, fs=24000: 120.0)

    long_ref = {"codes_delayed": [[i % 1024] * 8 for i in range(900)]}
    short_ref = {"codes_delayed": [[i % 1024] * 8 for i in range(100)]}
    assert orch._anchor_f0("fp-long", long_ref) == 120.0
    assert orch._anchor_f0("fp-short", short_ref) == 120.0
    assert seen == [fr._ANCHOR_MAX_DELAYED_ROWS, 100]


# --- banded DTW agrees with the full matrix on calibration-like drift --------


def test_dtw_band_default_matches_full_matrix_for_local_pause():
    rng = np.random.default_rng(20260825)
    feats_a = rng.standard_normal((200, 32))
    # Same-text reading with a local ~30-frame stall then catch-up: within
    # the 64-frame band floor, so the constrained DP must find the exact
    # same optimum as the unconstrained one.
    stalled = np.concatenate(
        [feats_a[:80], np.repeat(feats_a[80:81], 30, axis=0), feats_a[80:170]],
        axis=0,
    )
    banded = fr.dtw_map(feats_a, stalled)
    full = fr.dtw_map(feats_a, stalled, band=max(len(feats_a), len(stalled)))
    assert np.array_equal(banded, full)


def test_dtw_tiny_band_falls_back_without_error():
    rng = np.random.default_rng(7)
    feats_a = rng.standard_normal((60, 8))
    feats_b = rng.standard_normal((90, 8))
    out = fr.dtw_map(feats_a, feats_b, band=1)
    assert out.shape == (60,)
    assert np.all(np.diff(out) >= 0)  # monotone alignment


# --- world_extract(f0_t=...) short-circuit equivalence -----------------------


def test_world_extract_f0_t_shortcut_is_bit_identical():
    pytest.importorskip("pyworld")
    t = np.arange(int(1.2 * 24000)) / 24000
    x = np.zeros_like(t)
    for k in range(1, 5):
        x += (0.5 / k) * np.sin(2 * np.pi * 160.0 * k * t)
    x = (x / np.abs(x).max() * 0.8).astype(np.float64)

    recomputed = fr.world_extract(x)
    shortcut = fr.world_extract(x, f0_t=fr.world_f0(x))
    for a, b in zip(recomputed, shortcut):
        assert np.array_equal(a, b)


# --- capped URL download -----------------------------------------------------


class _StubConnection:
    def __init__(self, client):
        self._client = client

    def get_sync_client(self):
        return self._client


def _capped_client(monkeypatch, handler, cap):
    httpx = pytest.importorskip("httpx")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(higgs_utils, "global_http_connection", _StubConnection(client))
    monkeypatch.setattr(higgs_utils, "_MAX_REF_DOWNLOAD_BYTES", cap)


def test_download_capped_rejects_oversized_content_length(monkeypatch):
    httpx = pytest.importorskip("httpx")

    def handler(request):
        return httpx.Response(
            200,
            headers={"content-length": "5000"},
            stream=httpx.ByteStream(b""),
        )

    _capped_client(monkeypatch, handler, cap=1000)
    with pytest.raises(ValueError, match="over the 1000-byte limit"):
        higgs_utils._download_capped("https://example.com/ref.wav")


def test_download_capped_aborts_mid_stream(monkeypatch):
    httpx = pytest.importorskip("httpx")

    def handler(request):
        # Understated content-length passes the pre-check; the streamed body
        # must still trip the running-total guard.
        return httpx.Response(
            200,
            headers={"content-length": "500"},
            stream=httpx.ByteStream(b"a" * 2000),
        )

    _capped_client(monkeypatch, handler, cap=1000)
    with pytest.raises(ValueError, match="exceeded the 1000-byte limit"):
        higgs_utils._download_capped("https://example.com/ref.wav")


def test_download_capped_returns_body_within_limit(monkeypatch):
    httpx = pytest.importorskip("httpx")
    body = b"b" * 300

    def handler(request):
        return httpx.Response(200, content=body)

    _capped_client(monkeypatch, handler, cap=1000)
    assert higgs_utils._download_capped("https://example.com/ref.wav") == body


# --- reference trimming (needs the stages module → sglang) -------------------


def _stages():
    pytest.importorskip("sglang")
    from sglang_omni.models.higgs_tts import stages

    return stages


def _wav_with_active_segment(
    total_sec: float, active_start_sec: float, active_end_sec: float
) -> torch.Tensor:
    t = np.arange(int(total_sec * 24000)) / 24000
    x = np.zeros_like(t, dtype=np.float64)
    active = (t >= active_start_sec) & (t < active_end_sec)
    x[active] = 0.3 * np.sin(2 * np.pi * 220.0 * t[active])
    return torch.from_numpy(x.astype(np.float32))


def test_trim_picks_most_voiced_window():
    stages = _stages()
    wav = _wav_with_active_segment(60.0, 20.0, 50.0)
    out = stages._trim_reference_waveform(wav)
    assert out.shape[-1] == 30 * 24000
    # The only 30 s window fully covering the active segment starts at 20 s.
    assert torch.equal(out, wav[480000 : 480000 + 720000])


def test_trim_is_deterministic():
    stages = _stages()
    wav = _wav_with_active_segment(60.0, 12.0, 47.0)
    assert torch.equal(
        stages._trim_reference_waveform(wav),
        stages._trim_reference_waveform(wav),
    )


def test_trim_disabled_returns_input_unchanged(monkeypatch):
    stages = _stages()
    monkeypatch.setattr(stages, "_REF_TRIM_SECONDS", 0.0)
    wav = _wav_with_active_segment(60.0, 20.0, 50.0)
    out = stages._trim_reference_waveform(wav)
    assert out is wav


def test_trim_returns_exact_target_when_not_frame_multiple(monkeypatch):
    stages = _stages()
    # 30.03 s is not a multiple of the 50 ms analysis frame; an active tail
    # makes the last window win, which used to silently under-slice.
    monkeypatch.setattr(stages, "_REF_TRIM_SECONDS", 30.03)
    wav = _wav_with_active_segment(40.0, 30.0, 40.0)
    out = stages._trim_reference_waveform(wav)
    assert out.shape[-1] == int(30.03 * 24000)


# --- prompt budget -----------------------------------------------------------


def test_prompt_budget_boundary_and_clamp():
    stages = _stages()
    budget = stages._ENGINE_CONTEXT_BUDGET
    stages._check_prompt_budget(budget - 2048, 2048, what="probe")  # no raise
    with pytest.raises(ValueError, match="probe"):
        stages._check_prompt_budget(budget - 2048 + 1, 2048, what="probe")
    # Values over the engine cap must be clamped before budgeting, otherwise
    # requests the cap would have saved get falsely rejected.
    assert stages._effective_max_new_tokens(6000) == stages._MAX_NEW_TOKENS_CAP
    assert stages._effective_max_new_tokens(100) == 100


# --- long-reference split fusion ---------------------------------------------


def test_split_reference_produces_equal_deterministic_segments():
    stages = _stages()
    wav = _wav_with_active_segment(80.0, 0.0, 80.0)
    segments = stages._split_reference_for_fusion(wav)
    assert segments is not None and len(segments) == 3
    seg_len = 80 * 24000 // 3
    assert all(int(s.shape[-1]) == seg_len for s in segments)
    again = stages._split_reference_for_fusion(wav)
    assert all(torch.equal(a, b) for a, b in zip(segments, again))


def test_split_reference_drops_silent_segments():
    stages = _stages()
    # Speech only in the first 20 s of an 80 s clip: both silent segments
    # are dropped, and the lone speech-bearing segment is returned alone
    # (callers fall back to ordinary single-reference cloning with it).
    wav = _wav_with_active_segment(80.0, 0.0, 20.0)
    segments = stages._split_reference_for_fusion(wav)
    assert segments is not None and len(segments) == 1
    assert torch.equal(segments[0], wav[: 80 * 24000 // 3])


def test_split_reference_short_clip_returns_none():
    stages = _stages()
    wav = _wav_with_active_segment(20.0, 0.0, 20.0)
    assert stages._split_reference_for_fusion(wav) is None


def test_long_reference_mode_defaults_to_encoding_the_whole_clip(monkeypatch):
    """The default serves a long reference as one prompt, not a fusion build.

    split_fuse's build is only worth its cost if later requests for the same
    voice reuse it, and nothing routes them to the pod that holds it.
    """
    stages = _stages()
    monkeypatch.delenv("HIGGS_FUSION_MODE", raising=False)
    assert stages._REF_LONG_MODE == "whole"
    assert stages._long_reference_mode() == "whole"


def test_long_reference_mode_falls_back_to_trim_for_logits_fusion(monkeypatch):
    stages = _stages()
    monkeypatch.delenv("HIGGS_FUSION_MODE", raising=False)
    monkeypatch.setattr(stages, "_REF_LONG_MODE", "split_fuse")
    assert stages._long_reference_mode() == "split_fuse"
    monkeypatch.setenv("HIGGS_FUSION_MODE", "logits")
    assert stages._long_reference_mode() == "trim"


def test_audio_encoder_batches_equal_length_fusion_refs(monkeypatch):
    stages = _stages()
    from types import SimpleNamespace

    from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
    from sglang_omni.proto import OmniRequest, StagePayload

    monkeypatch.setattr(stages, "resolve_checkpoint", lambda p: p)
    monkeypatch.setattr(stages.Tokenizer, "from_file", lambda _p: object())
    monkeypatch.setattr(
        stages, "PreTrainedTokenizerFast", lambda tokenizer_object: object()
    )

    class FakeAdapter:
        def __init__(self, _tokenizer) -> None:
            pass

        def build_prompt(
            self, text: str, *, num_ref_tokens: int, reference_text: str | None
        ) -> list[int]:
            return [len(text), num_ref_tokens, len(reference_text or "")]

    class FakeCodec:
        SAMPLE_RATE = 24000

        def __init__(self) -> None:
            self.single_calls = 0
            self.batch_calls = 0
            self.model = SimpleNamespace(
                acoustic_encoder=torch.nn.Identity(),
                semantic_model=torch.nn.Identity(),
            )

        def encode_reference(self, waveform, sample_rate) -> torch.Tensor:
            self.single_calls += 1
            return torch.tensor([[1, 2], [3, 4]], dtype=torch.long)

        def encode_batch(self, waveforms) -> list[torch.Tensor]:
            self.batch_calls += 1
            return [
                torch.tensor([[10 + i, 20 + i]], dtype=torch.long)
                for i in range(len(waveforms))
            ]

    fake_codec = FakeCodec()
    monkeypatch.setattr(stages, "HiggsTokenizerAdapter", FakeAdapter)
    monkeypatch.setattr(stages, "get_or_load_codec", lambda *a, **k: fake_codec)

    scheduler = stages.create_audio_encoder_executor(
        "ckpt", device="cuda:0", num_codebooks=2
    )
    fake_codec.single_calls = 0  # ignore construction-time warm-up

    def _fusion_payload(specs: list[tuple[int, float]]) -> StagePayload:
        state = HiggsTtsState(
            fusion_refs=[
                {
                    "weight": 1.0,
                    "reference_text": None,
                    "codes_delayed": None,
                    "prompt_token_ids": None,
                    "cal_prompt_token_ids": None,
                    "waveform": torch.full((1, 1, n), fill),
                }
                for n, fill in specs
            ],
            fusion_build={
                "cal_text": "cal",
                "final_prompt_prefix": [1],
                "final_prompt_suffix": [2],
            },
            target_text="hi",
            num_codebooks=2,
        )
        return StagePayload(
            request_id="r",
            request=OmniRequest(inputs={}),
            data=state.to_dict(),
        )

    equal = [(4800, 0.1), (4800, 0.2), (4800, 0.3)]
    out = scheduler._fn(_fusion_payload(equal))
    assert fake_codec.batch_calls == 1
    assert fake_codec.single_calls == 0
    st = HiggsTtsState.from_dict(out.data)
    assert all(r["codes_delayed"] for r in st.fusion_refs)
    assert all(r["cal_prompt_token_ids"] for r in st.fusion_refs)

    # Same content again: served from the content-keyed code cache, no
    # further codec calls of either kind.
    scheduler._fn(_fusion_payload(equal))
    assert fake_codec.batch_calls == 1
    assert fake_codec.single_calls == 0

    # Mixed lengths cannot batch: falls back to per-ref encoding.
    scheduler._fn(_fusion_payload([(4800, 0.4), (9600, 0.5)]))
    assert fake_codec.batch_calls == 1
    assert fake_codec.single_calls == 2


def test_budget_constants_track_engine_config():
    stages = _stages()
    from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig
    from sglang_omni.models.higgs_tts.engine_builder import HiggsTtsEngineBuilder

    assert stages._ENGINE_CONTEXT_BUDGET == HiggsTtsEngineBuilder.context_length - 1
    tts_engine = next(
        stage
        for stage in HiggsTtsPipelineConfig.model_fields["stages"].default
        if stage.name == "tts_engine"
    )
    assert stages._MAX_NEW_TOKENS_CAP == tts_engine.factory.max_new_tokens
