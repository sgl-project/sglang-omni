# SPDX-License-Identifier: Apache-2.0
"""Model presets and thresholds for TTS CI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from tests.utils import apply_mos_slack, apply_slack, apply_wer_slack


@dataclass(frozen=True)
class TtsCiModelPreset:
    model_path: str
    ref_format: Literal["flat", "references"] = "flat"
    token_count: int | Literal["auto"] | None = None
    worker_extra_args: str = ""
    startup_timeout: int = 180
    gate_thresholds: bool = True
    num_gpus_per_worker: int = 1


@dataclass(frozen=True)
class TtsCiThresholdPreset:
    non_stream_speed: dict[int, dict[str, float]]
    stream_speed: dict[int, dict[str, float]]
    wer_corpus: float
    stream_wer_corpus: float
    similarity_mean_min: float
    utmos_mean_min: float
    # Note: (Jiaxin Deng) False while the values are seeds rather than
    # worst-of-N observations from the CI runner. A seed can be far off for a
    # topology it was not measured on, so gating on one fails builds for no
    # reason or waves regressions through; a contract test refuses that pair.
    calibrated: bool = True


@dataclass(frozen=True)
class TtsCiPreset:
    model: TtsCiModelPreset
    thresholds: TtsCiThresholdPreset


# Slack factors applied to P95 reference values to derive CI thresholds.
# Higher-is-better metrics: threshold = P95 * slack_higher.
# Lower-is-better metrics: threshold = P95 * slack_lower.
THRESHOLD_SLACK_HIGHER = 0.75
THRESHOLD_SLACK_LOWER = 1.25


# Higgs thresholds.
HIGGS_VC_WER_MAX_CORPUS = 0.0109
HIGGS_VC_WER_CORPUS_THRESHOLD = apply_wer_slack(HIGGS_VC_WER_MAX_CORPUS)
HIGGS_VC_STREAM_WER_MAX_CORPUS = 0.0106
HIGGS_VC_STREAM_WER_CORPUS_THRESHOLD = apply_wer_slack(HIGGS_VC_STREAM_WER_MAX_CORPUS)
HIGGS_VC_SIMILARITY_MEAN_MIN = 66.06310302734374
HIGGS_VC_UTMOS_MEAN_REFERENCE = 4.163
HIGGS_VC_UTMOS_MEAN_MIN = apply_mos_slack(HIGGS_VC_UTMOS_MEAN_REFERENCE)

_HIGGS_VC_NON_STREAM_P95 = {
    16: {
        "throughput_qps": 15.846,
        "output_tok_per_req_s": 131.1,
        "latency_mean_s": 1.005,
        "rtf_mean": 0.2451,
    }
}

_HIGGS_VC_STREAM_P95 = {
    16: {
        "throughput_qps": 17.467,
        "latency_mean_s": 0.854,
        "rtf_mean": 0.2049,
    }
}

HIGGS_VC_NON_STREAM_THRESHOLDS = apply_slack(
    _HIGGS_VC_NON_STREAM_P95, THRESHOLD_SLACK_HIGHER, THRESHOLD_SLACK_LOWER
)
HIGGS_VC_STREAM_THRESHOLDS = apply_slack(
    _HIGGS_VC_STREAM_P95, THRESHOLD_SLACK_HIGHER, THRESHOLD_SLACK_LOWER
)


# MOSS Local thresholds.
MOSS_VC_WER_MAX_CORPUS = 0.0222
MOSS_VC_WER_CORPUS_THRESHOLD = apply_wer_slack(MOSS_VC_WER_MAX_CORPUS)
MOSS_VC_STREAM_WER_MAX_CORPUS = 0.0229
MOSS_VC_STREAM_WER_CORPUS_THRESHOLD = apply_wer_slack(MOSS_VC_STREAM_WER_MAX_CORPUS)
MOSS_VC_SIMILARITY_MEAN_MIN = 62.690567626953126
MOSS_VC_UTMOS_MEAN_REFERENCE = 3.9545
MOSS_VC_UTMOS_MEAN_MIN = apply_mos_slack(MOSS_VC_UTMOS_MEAN_REFERENCE)

_MOSS_VC_NON_STREAM_P95 = {
    16: {
        "throughput_qps": 14.166,
        "output_tok_per_req_s": 69.2,
        "latency_mean_s": 1.123,
        "rtf_mean": 0.2615,
    }
}

_MOSS_VC_STREAM_P95 = {
    16: {
        "throughput_qps": 7.209,
        "latency_mean_s": 2.199,
        "rtf_mean": 0.5223,
    }
}

MOSS_VC_NON_STREAM_THRESHOLDS = apply_slack(
    _MOSS_VC_NON_STREAM_P95, THRESHOLD_SLACK_HIGHER, THRESHOLD_SLACK_LOWER
)
MOSS_VC_STREAM_THRESHOLDS = apply_slack(
    _MOSS_VC_STREAM_P95, THRESHOLD_SLACK_HIGHER, THRESHOLD_SLACK_LOWER
)

# Qwen3-TTS 1.7B. This is the variant the community deploys, and it is gated as
# a single instance: on one H100 the tuned single instance beats the same-card
# MPS-DP2 pool on peak throughput and holds a several-fold better first-audio
# latency, so colocation is no longer the recommended topology for it.
#
# Note: (Jiaxin Deng) calibrated on the CI host, lane 0,1 pinned cpuset
# (2-15,66-79), worst-of-5 clean rounds with destructive rejection
# (run .tune-runs/20260813T021112Z_tts_qwen3tts_r5). Raw pre-slack references
# only; the CI slack calculation is unchanged.
QWEN3_TTS_VC_WER_MAX_CORPUS = 0.011
QWEN3_TTS_VC_WER_CORPUS_THRESHOLD = apply_wer_slack(QWEN3_TTS_VC_WER_MAX_CORPUS)
QWEN3_TTS_VC_STREAM_WER_MAX_CORPUS = 0.0116
QWEN3_TTS_VC_STREAM_WER_CORPUS_THRESHOLD = apply_wer_slack(
    QWEN3_TTS_VC_STREAM_WER_MAX_CORPUS
)
QWEN3_TTS_VC_SIMILARITY_MEAN_MIN = 69.4295639038086
QWEN3_TTS_VC_UTMOS_MEAN_REFERENCE = 4.195
QWEN3_TTS_VC_UTMOS_MEAN_MIN = apply_mos_slack(QWEN3_TTS_VC_UTMOS_MEAN_REFERENCE)

_QWEN3_TTS_VC_NON_STREAM_P95 = {
    16: {
        "throughput_qps": 17.66,
        "output_tok_per_req_s": 71.1,
        "latency_mean_s": 0.901,
        "rtf_mean": 0.2236,
    }
}

_QWEN3_TTS_VC_STREAM_P95 = {
    16: {
        "throughput_qps": 15.324,
        "latency_mean_s": 1.039,
        "rtf_mean": 0.257,
    }
}

QWEN3_TTS_VC_NON_STREAM_THRESHOLDS = apply_slack(
    _QWEN3_TTS_VC_NON_STREAM_P95, THRESHOLD_SLACK_HIGHER, THRESHOLD_SLACK_LOWER
)
QWEN3_TTS_VC_STREAM_THRESHOLDS = apply_slack(
    _QWEN3_TTS_VC_STREAM_P95, THRESHOLD_SLACK_HIGHER, THRESHOLD_SLACK_LOWER
)


TTS_CI_PRESETS: dict[str, TtsCiPreset] = {
    "higgs": TtsCiPreset(
        model=TtsCiModelPreset(
            model_path="bosonai/higgs-tts-3-4b",
            # On H100, SGLang 0.5.16 cold startup can spend about a minute
            # capturing the largest decode CUDA graph after loading all TTS
            # stages. Two managed workers can therefore exceed 180 seconds.
            startup_timeout=300,
        ),
        thresholds=TtsCiThresholdPreset(
            non_stream_speed=HIGGS_VC_NON_STREAM_THRESHOLDS,
            stream_speed=HIGGS_VC_STREAM_THRESHOLDS,
            wer_corpus=HIGGS_VC_WER_CORPUS_THRESHOLD,
            stream_wer_corpus=HIGGS_VC_STREAM_WER_CORPUS_THRESHOLD,
            similarity_mean_min=HIGGS_VC_SIMILARITY_MEAN_MIN,
            utmos_mean_min=HIGGS_VC_UTMOS_MEAN_MIN,
        ),
    ),
    "qwen3-tts": TtsCiPreset(
        model=TtsCiModelPreset(
            model_path="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            ref_format="references",
            # Note: (Jiaxin Deng) the shipped defaults cap the AR engine at 16
            # running requests and colocate every stage in one process; both are
            # what kept this variant behind, so CI measures the tuned point.
            worker_extra_args=(
                "--max-running-requests 64 "
                "--cuda-graph-max-bs 64 "
                "--talker-torch-compile-max-bs 64 "
                "--stages.vocoder.process vocoder "
                "--stages.tts_engine.runtime.resources."
                "total-gpu-memory-fraction 0.85 "
                "--stages.vocoder.runtime.resources."
                "total-gpu-memory-fraction 0.10"
            ),
            startup_timeout=300,
            gate_thresholds=True,
        ),
        thresholds=TtsCiThresholdPreset(
            non_stream_speed=QWEN3_TTS_VC_NON_STREAM_THRESHOLDS,
            stream_speed=QWEN3_TTS_VC_STREAM_THRESHOLDS,
            wer_corpus=QWEN3_TTS_VC_WER_CORPUS_THRESHOLD,
            stream_wer_corpus=QWEN3_TTS_VC_STREAM_WER_CORPUS_THRESHOLD,
            similarity_mean_min=QWEN3_TTS_VC_SIMILARITY_MEAN_MIN,
            utmos_mean_min=QWEN3_TTS_VC_UTMOS_MEAN_MIN,
        ),
    ),
    "moss": TtsCiPreset(
        model=TtsCiModelPreset(
            model_path="OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
            ref_format="references",
            token_count="auto",
            gate_thresholds=True,
        ),
        thresholds=TtsCiThresholdPreset(
            non_stream_speed=MOSS_VC_NON_STREAM_THRESHOLDS,
            stream_speed=MOSS_VC_STREAM_THRESHOLDS,
            wer_corpus=MOSS_VC_WER_CORPUS_THRESHOLD,
            stream_wer_corpus=MOSS_VC_STREAM_WER_CORPUS_THRESHOLD,
            similarity_mean_min=MOSS_VC_SIMILARITY_MEAN_MIN,
            utmos_mean_min=MOSS_VC_UTMOS_MEAN_MIN,
        ),
    ),
}


def select_tts_ci_preset(model_name: str | None = None) -> tuple[str, TtsCiPreset]:
    selected = model_name or os.environ.get("TTS_CI_MODEL", "higgs")
    preset = TTS_CI_PRESETS.get(selected)
    if preset is None:
        allowed = ", ".join(sorted(TTS_CI_PRESETS))
        raise ValueError(
            f"Unsupported TTS_CI_MODEL={selected!r}; expected one of: {allowed}"
        )
    return selected, preset
