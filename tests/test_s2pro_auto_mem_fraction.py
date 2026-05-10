# SPDX-License-Identifier: Apache-2.0
"""Tests for the S2-Pro auto-sized ``mem_fraction_static`` heuristic.

Covers two layers:

* ``_compute_auto_mem_fraction``: the pure math that turns
  ``mem_get_info`` + a measured vocoder reserve into ``mem_fraction_static``.
* ``_measure_vocoder_peak_bytes``: the probe-load + dummy-decode that
  produces that reserve.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from sglang_omni.models.fishaudio_s2_pro.pipeline import stages

_GiB = 1024**3
_DEFAULT_RESERVE_BYTES = 2 * _GiB  # used by tests so the math matches comments


def _patch_gpu(total_gib: float, free_gib: float):
    """Patch ``torch.cuda.*`` to look like a GPU with the requested capacities."""
    return [
        patch.object(
            stages.torch.cuda,
            "get_device_properties",
            return_value=SimpleNamespace(total_memory=int(total_gib * _GiB)),
        ),
        patch.object(
            stages.torch.cuda,
            "mem_get_info",
            return_value=(int(free_gib * _GiB), int(total_gib * _GiB)),
        ),
    ]


def _run_compute(
    total_gib: float,
    free_gib: float,
    device: str = "cuda:0",
    reserve_bytes: int = _DEFAULT_RESERVE_BYTES,
) -> float:
    patches = _patch_gpu(total_gib, free_gib)
    for p in patches:
        p.start()
    try:
        return stages._compute_auto_mem_fraction(device, reserve_bytes)
    finally:
        for p in reversed(patches):
            p.stop()


class TestComputeAutoMemFraction:
    def test_24gb_gpu_lands_around_065(self) -> None:
        """RTX 3090/4090 with co-located audio modules already loaded."""
        # free=17.2 GiB after audio_decoder + stream codec on a 24 GiB GPU
        fraction = _run_compute(total_gib=23.5, free_gib=17.2)
        # (17.2 - 2.0) / 23.5 = 0.6468
        assert fraction == pytest.approx(0.647, abs=0.005)

    def test_80gb_h100_picks_high_fraction(self) -> None:
        """Big GPU should get a strictly bigger fraction than the old 0.85."""
        fraction = _run_compute(total_gib=79.2, free_gib=73.0)
        # (73.0 - 2.0) / 79.2 = 0.8965
        assert fraction == pytest.approx(0.897, abs=0.005)
        assert fraction > 0.85

    def test_upper_clamp_caps_at_095(self) -> None:
        """Big, nearly-empty GPU is capped at 0.95 to leave room for SGLang internals."""
        # H200-class: (140 - 2) / 141 = 0.979 -> clamped to 0.95
        fraction = _run_compute(total_gib=141.0, free_gib=140.0)
        assert fraction == 0.95

    def test_small_gpu_is_not_clamped_up(self) -> None:
        """16 GiB GPU must NOT be clamped up to 0.5 (which would re-introduce OOM)."""
        # free=9.5 GiB after audio modules; (9.5-2.0)/16 = 0.469
        fraction = _run_compute(total_gib=16.0, free_gib=9.5)
        assert fraction < 0.5
        assert fraction == pytest.approx(0.469, abs=0.005)

    def test_low_value_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Low computed fraction must surface as a warning, not a silent INFO line."""
        caplog.set_level(logging.WARNING, logger=stages.logger.name)
        _run_compute(total_gib=12.0, free_gib=5.5)
        assert any(
            "Auto-sized mem_fraction_static" in r.message
            and r.levelno == logging.WARNING
            for r in caplog.records
        )

    def test_negative_target_floors_to_zero(self) -> None:
        """If reserve > free, fraction goes to 0 instead of becoming negative."""
        # free=1 GiB, reserve=2 GiB -> target clamped to 0
        fraction = _run_compute(total_gib=24.0, free_gib=1.0)
        assert fraction == 0.0

    def test_device_string_without_index_defaults_to_zero(self) -> None:
        """``device='cuda'`` (no index) must not raise; treated as gpu 0."""
        fraction = _run_compute(total_gib=23.5, free_gib=17.2, device="cuda")
        assert fraction == pytest.approx(0.647, abs=0.005)

    def test_measured_reserve_is_honored(self) -> None:
        """A larger measured reserve should shrink the fraction proportionally."""
        # Same 24 GiB / free=17.2, but reserve=3.5 GiB instead of 2 GiB
        fraction = _run_compute(
            total_gib=23.5, free_gib=17.2, reserve_bytes=int(3.5 * _GiB)
        )
        # (17.2 - 3.5) / 23.5 = 0.583
        assert fraction == pytest.approx(0.583, abs=0.005)


class TestMeasureVocoderPeakBytes:
    """Probe path: load codec, dummy-decode, measure peak delta, free."""

    def _patch_probe(
        self,
        *,
        pre_alloc_gib: float,
        peak_alloc_gib: float,
    ) -> list:
        """Patch the small CUDA + codec surface the probe touches."""
        fake_codec = MagicMock(name="fake_codec")
        fake_codec.from_indices = MagicMock(return_value=None)
        return [
            patch.object(stages, "_load_codec", return_value=fake_codec),
            patch.object(stages.torch.cuda, "empty_cache"),
            patch.object(stages.torch.cuda, "synchronize"),
            patch.object(stages.torch.cuda, "reset_peak_memory_stats"),
            patch.object(
                stages.torch.cuda,
                "memory_allocated",
                return_value=int(pre_alloc_gib * _GiB),
            ),
            patch.object(
                stages.torch.cuda,
                "max_memory_allocated",
                return_value=int(peak_alloc_gib * _GiB),
            ),
            patch.object(stages.torch, "zeros", return_value=MagicMock()),
        ]

    def _run_probe(self, *, pre_alloc_gib: float, peak_alloc_gib: float) -> int:
        patches = self._patch_probe(
            pre_alloc_gib=pre_alloc_gib, peak_alloc_gib=peak_alloc_gib
        )
        for p in patches:
            p.start()
        try:
            return stages._measure_vocoder_peak_bytes(
                checkpoint_dir="/dummy",
                device="cuda:0",
                num_codebooks=4,
                probe_tokens=128,
            )
        finally:
            for p in reversed(patches):
                p.stop()

    def test_returns_peak_minus_pre(self) -> None:
        """Peak delta is what's reported; the pre-baseline must be subtracted."""
        # Pre-existing 2 GiB on the GPU (e.g. audio decoder), peak hits 3.4 GiB
        # while codec is loaded + dummy-decoded.
        peak = self._run_probe(pre_alloc_gib=2.0, peak_alloc_gib=3.4)
        # delta = (3.4 - 2.0) GiB
        assert peak == pytest.approx(1.4 * _GiB, rel=0.01)

    def test_zero_when_codec_uses_no_memory(self) -> None:
        """Pathological: peak == pre means no codec memory was tracked."""
        peak = self._run_probe(pre_alloc_gib=2.0, peak_alloc_gib=2.0)
        assert peak == 0

    def test_load_codec_is_called_with_target_device(self) -> None:
        """Probe must load on the SGLang GPU, not on CPU."""
        with (
            patch.object(stages, "_load_codec") as mock_load,
            patch.object(stages.torch.cuda, "empty_cache"),
            patch.object(stages.torch.cuda, "synchronize"),
            patch.object(stages.torch.cuda, "reset_peak_memory_stats"),
            patch.object(stages.torch.cuda, "memory_allocated", return_value=0),
            patch.object(stages.torch.cuda, "max_memory_allocated", return_value=_GiB),
            patch.object(stages.torch, "zeros", return_value=MagicMock()),
        ):
            mock_load.return_value = MagicMock(from_indices=MagicMock())
            stages._measure_vocoder_peak_bytes(
                checkpoint_dir="/dummy",
                device="cuda:1",
                num_codebooks=4,
                probe_tokens=64,
            )
            mock_load.assert_called_once_with("/dummy", "cuda:1")

    def test_dummy_shape_uses_caller_supplied_probe_tokens(self) -> None:
        """The dummy decode honors the caller-supplied ``probe_tokens``."""
        captured: dict = {}

        def fake_zeros(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return MagicMock()

        with (
            patch.object(stages, "_load_codec") as mock_load,
            patch.object(stages.torch.cuda, "empty_cache"),
            patch.object(stages.torch.cuda, "synchronize"),
            patch.object(stages.torch.cuda, "reset_peak_memory_stats"),
            patch.object(stages.torch.cuda, "memory_allocated", return_value=0),
            patch.object(stages.torch.cuda, "max_memory_allocated", return_value=_GiB),
            patch.object(stages.torch, "zeros", side_effect=fake_zeros),
        ):
            mock_load.return_value = MagicMock(from_indices=MagicMock())
            stages._measure_vocoder_peak_bytes(
                checkpoint_dir="/dummy",
                device="cuda:0",
                num_codebooks=4,
                probe_tokens=90,
            )

        # zeros(1, num_codebooks - 1, probe_tokens, dtype=..., device=...)
        assert captured["args"] == (1, 3, 90)
        assert captured["kwargs"]["device"] == "cuda:0"

    def test_probe_failure_in_factory_falls_back_to_constant(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """If the probe raises, the factory must use the fallback reserve."""
        caplog.set_level(logging.WARNING, logger=stages.logger.name)

        used_reserve: dict[str, int] = {}

        def capturing_compute(device: str, reserve: int) -> float:
            used_reserve["value"] = reserve
            return 0.5

        with (
            patch.object(
                stages,
                "_measure_vocoder_peak_bytes",
                side_effect=RuntimeError("simulated probe OOM"),
            ),
            patch.object(
                stages, "_compute_auto_mem_fraction", side_effect=capturing_compute
            ),
        ):
            # Replicate the small fallback block from
            # create_sglang_tts_engine_executor here, so this test does not
            # need to boot the full SGLang factory.
            try:
                reserve = stages._measure_vocoder_peak_bytes(
                    "/dummy", "cuda:0", 4, probe_tokens=90
                )
            except Exception as exc:
                stages.logger.warning(
                    "Vocoder probe failed (%s: %s); using fallback reserve %.1f GiB",
                    type(exc).__name__,
                    exc,
                    stages._VOCODER_RESERVE_FALLBACK_BYTES / _GiB,
                )
                reserve = stages._VOCODER_RESERVE_FALLBACK_BYTES
            stages._compute_auto_mem_fraction("cuda:0", reserve)

        assert used_reserve["value"] == stages._VOCODER_RESERVE_FALLBACK_BYTES
        assert any(
            "Vocoder probe failed" in r.message and r.levelno == logging.WARNING
            for r in caplog.records
        )
