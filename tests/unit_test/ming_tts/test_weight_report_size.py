# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from sglang_omni.models.ming_tts.weight_loading import (
    MingTTSWeightManifest,
    MingTTSWeightReport,
)


def _report_with_size(size_bytes: int) -> MingTTSWeightReport:
    manifest = MingTTSWeightManifest(
        model_path="unused",
        source="index",
        total_tensors=1,
        total_size_bytes=size_bytes,
        prefix_counts={},
        keys_by_owner={},
        shards_by_owner={},
        weight_map={"tensor": "model.safetensors"},
    )
    return MingTTSWeightReport(manifest=manifest)


def test_summary_steps_up_when_rounding_reaches_1024() -> None:
    just_below_mib = _report_with_size(1024 * 1024 - 1).summary()
    assert "1.00MiB" in just_below_mib
    assert "1024.00KiB" not in just_below_mib

    exact_mib = _report_with_size(1024 * 1024).summary()
    assert "1.00MiB" in exact_mib


def test_summary_keeps_ordinary_sizes_in_the_current_unit() -> None:
    assert "512.00B" in _report_with_size(512).summary()
    assert "2.00KiB" in _report_with_size(2048).summary()
