# SPDX-License-Identifier: Apache-2.0
"""Model-specific calibration references for the shared Omni CI stages."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from tests.utils import (
    MetricCheckCollector,
    apply_mos_slack,
    apply_slack,
    apply_wer_slack,
)


@dataclass(frozen=True, kw_only=True)
class OmniCiThresholdPreset:
    speed: dict[int, dict[str, float]]
    accuracy: float | None = None
    wer: float | None = None
    n_above_50: float | None = None
    similarity: float | None = None
    utmos: float | None = None
    calibrated: bool = True

    def require_calibrated(
        self, model: str, stage: str, checks: MetricCheckCollector | None = None
    ) -> None:
        if not self.calibrated and checks is not None:
            checks.assert_all()
        assert self.calibrated, (
            f"{model} {stage} thresholds are uncalibrated; metrics were collected "
            "but this stage cannot qualify until calibration is complete"
        )


@dataclass(frozen=True, kw_only=True)
class OmniCiModelPreset:
    name: Literal["qwen3-omni", "minicpmo"]
    model_path: str
    reference_audio_field: Literal["audios", "audio.ref_audio"]
    thresholds: dict[str, OmniCiThresholdPreset]


QWEN3_OMNI_TTS_P95 = {
    16: {
        "throughput_qps": 16.455,
        "output_tok_per_req_s": 16.1,
        "latency_mean_s": 0.911,
        "rtf_mean": 0.2749,
    }
}
QWEN3_OMNI_TTS_WER_BELOW_50_CORPUS_MAX = 0.0213
QWEN3_OMNI_TTS_N_ABOVE_50_MAX = 0
QWEN3_OMNI_TTS_SIMILARITY_MEAN_MIN = 60.0
QWEN3_OMNI_TTS_UTMOS_MEAN_REFERENCE = 4.4444

QWEN3_OMNI_MMMU_P95 = {
    16: {"throughput_qps": 1.888, "output_tok_per_req_s": 88.5, "latency_mean_s": 7.627}
}
QWEN3_OMNI_MMMU_MIN_ACCURACY = 0.62

QWEN3_OMNI_MMMU_TALKER_P95 = {
    16: {
        "throughput_qps": 2.175,
        "output_tok_per_req_s": 25.4,
        "latency_mean_s": 5.579,
        "rtf_mean": 0.1315,
    }
}
QWEN3_OMNI_MMMU_TALKER_MIN_ACCURACY = 0.75
QWEN3_OMNI_MMMU_TALKER_WER_BELOW_50_CORPUS_MAX = 0.1589
QWEN3_OMNI_MMMU_TALKER_N_ABOVE_50_MAX = 3.0

QWEN3_OMNI_MMSU_P95 = {
    16: {
        "throughput_qps": 115.795,
        "output_tok_per_req_s": 15.1,
        "latency_mean_s": 0.137,
    }
}
QWEN3_OMNI_MMSU_MIN_ACCURACY = 0.7055

QWEN3_OMNI_MMSU_TALKER_P95 = {
    16: {
        "throughput_qps": 2.607,
        "output_tok_per_req_s": 11.0,
        "latency_mean_s": 5.829,
        "rtf_mean": 0.2825,
    }
}
QWEN3_OMNI_MMSU_TALKER_MIN_ACCURACY = 0.625
QWEN3_OMNI_MMSU_TALKER_WER_BELOW_50_CORPUS_MAX = 0.0306
QWEN3_OMNI_MMSU_TALKER_N_ABOVE_50_MAX = 0.0

QWEN3_OMNI_VIDEOMME_P95 = {
    16: {
        "throughput_qps": 1.235,
        "output_tok_per_req_s": 10.0,
        "latency_mean_s": 11.546,
    }
}
QWEN3_OMNI_VIDEOMME_MIN_ACCURACY = 0.58

QWEN3_OMNI_VIDEOMME_TALKER_P95 = {
    16: {
        "throughput_qps": 1.026,
        "output_tok_per_req_s": 4.7,
        "latency_mean_s": 10.427,
        "rtf_mean": 0.9379,
    }
}
QWEN3_OMNI_VIDEOMME_TALKER_MIN_ACCURACY = 0.6
QWEN3_OMNI_VIDEOMME_TALKER_WER_BELOW_50_CORPUS_MAX = 0.0332
QWEN3_OMNI_VIDEOMME_TALKER_N_ABOVE_50_MAX = 1.0

QWEN3_OMNI_VIDEOAMME_P95 = {
    16: {"throughput_qps": 1.602, "output_tok_per_req_s": 5.7, "latency_mean_s": 8.84}
}
QWEN3_OMNI_VIDEOAMME_MIN_ACCURACY = 0.64

QWEN3_OMNI_VIDEOAMME_TALKER_P95 = {
    16: {
        "throughput_qps": 0.252,
        "output_tok_per_req_s": 1.2,
        "latency_mean_s": 37.525,
        "rtf_mean": 3.0823,
    }
}
QWEN3_OMNI_VIDEOAMME_TALKER_MIN_ACCURACY = 0.5
QWEN3_OMNI_VIDEOAMME_TALKER_WER_BELOW_50_CORPUS_MAX = 0.0157
QWEN3_OMNI_VIDEOAMME_TALKER_N_ABOVE_50_MAX = 0.0

# note (wenyao): Raw references measured on H100 with DP2; slack is applied once below.
MINICPMO_TTS_P95 = {
    16: {
        "throughput_qps": 19.275,
        "output_tok_per_req_s": 20.0,
        "latency_mean_s": 0.776,
        "rtf_mean": 0.1929,
    }
}
MINICPMO_TTS_WER_BELOW_50_CORPUS_MAX = 0.0179
MINICPMO_TTS_N_ABOVE_50_MAX = 2.0
MINICPMO_TTS_SIMILARITY_MEAN_MIN = 42.81593160629272
MINICPMO_TTS_UTMOS_MEAN_REFERENCE = 4.2791

MINICPMO_MMMU_P95 = {
    16: {
        "throughput_qps": 2.263,
        "output_tok_per_req_s": 123.0,
        "latency_mean_s": 6.015,
    }
}
MINICPMO_MMMU_MIN_ACCURACY = 0.62

MINICPMO_MMMU_TALKER_P95 = {
    16: {
        "throughput_qps": 2.335,
        "output_tok_per_req_s": 31.4,
        "latency_mean_s": 4.745,
        "rtf_mean": 0.1195,
    }
}
MINICPMO_MMMU_TALKER_MIN_ACCURACY = 0.7
MINICPMO_MMMU_TALKER_WER_BELOW_50_CORPUS_MAX = 0.2025
MINICPMO_MMMU_TALKER_N_ABOVE_50_MAX = 8.0

MINICPMO_MMSU_P95 = {
    16: {
        "throughput_qps": 44.74,
        "output_tok_per_req_s": 24.9,
        "latency_mean_s": 0.356,
    }
}
MINICPMO_MMSU_MIN_ACCURACY = 0.532

MINICPMO_MMSU_TALKER_P95 = {
    16: {
        "throughput_qps": 5.11,
        "output_tok_per_req_s": 16.2,
        "latency_mean_s": 2.853,
        "rtf_mean": 0.2084,
    }
}
MINICPMO_MMSU_TALKER_MIN_ACCURACY = 0.6
MINICPMO_MMSU_TALKER_WER_BELOW_50_CORPUS_MAX = 0.0242
MINICPMO_MMSU_TALKER_N_ABOVE_50_MAX = 0.0

MINICPMO_VIDEOMME_P95 = {
    16: {"throughput_qps": 0.936, "output_tok_per_req_s": 4.7, "latency_mean_s": 14.618}
}
MINICPMO_VIDEOMME_MIN_ACCURACY = 0.64

MINICPMO_VIDEOMME_TALKER_P95 = {
    16: {
        "throughput_qps": 0.889,
        "output_tok_per_req_s": 2.6,
        "latency_mean_s": 11.594,
        "rtf_mean": 1.3521,
    }
}
MINICPMO_VIDEOMME_TALKER_MIN_ACCURACY = 0.6
MINICPMO_VIDEOMME_TALKER_WER_BELOW_50_CORPUS_MAX = 0.0756
MINICPMO_VIDEOMME_TALKER_N_ABOVE_50_MAX = 1.0

MINICPMO_VIDEOAMME_P95 = {
    16: {"throughput_qps": 0.93, "output_tok_per_req_s": 2.3, "latency_mean_s": 14.79}
}
MINICPMO_VIDEOAMME_MIN_ACCURACY = 0.66

MINICPMO_VIDEOAMME_TALKER_P95 = {
    16: {
        "throughput_qps": 0.87,
        "output_tok_per_req_s": 4.3,
        "latency_mean_s": 7.274,
        "rtf_mean": 0.9681,
    }
}
MINICPMO_VIDEOAMME_TALKER_MIN_ACCURACY = 0.7
MINICPMO_VIDEOAMME_TALKER_WER_BELOW_50_CORPUS_MAX = 0.0081
MINICPMO_VIDEOAMME_TALKER_N_ABOVE_50_MAX = 0.0

QWEN3_OMNI_SEEDTTS_RTF_MEAN_MAX = 0.9536

OMNI_CI_PRESETS: dict[str, OmniCiModelPreset] = {
    "qwen3-omni": OmniCiModelPreset(
        name="qwen3-omni",
        model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        reference_audio_field="audios",
        thresholds={
            "tts": OmniCiThresholdPreset(
                speed=apply_slack(QWEN3_OMNI_TTS_P95),
                wer=apply_wer_slack(QWEN3_OMNI_TTS_WER_BELOW_50_CORPUS_MAX),
                n_above_50=QWEN3_OMNI_TTS_N_ABOVE_50_MAX,
                similarity=QWEN3_OMNI_TTS_SIMILARITY_MEAN_MIN,
                utmos=apply_mos_slack(QWEN3_OMNI_TTS_UTMOS_MEAN_REFERENCE),
            ),
            "mmmu": OmniCiThresholdPreset(
                speed=apply_slack(QWEN3_OMNI_MMMU_P95),
                accuracy=QWEN3_OMNI_MMMU_MIN_ACCURACY,
            ),
            "mmmu_talker": OmniCiThresholdPreset(
                speed=apply_slack(QWEN3_OMNI_MMMU_TALKER_P95),
                accuracy=QWEN3_OMNI_MMMU_TALKER_MIN_ACCURACY,
                wer=apply_wer_slack(QWEN3_OMNI_MMMU_TALKER_WER_BELOW_50_CORPUS_MAX),
                n_above_50=QWEN3_OMNI_MMMU_TALKER_N_ABOVE_50_MAX,
            ),
            "mmsu": OmniCiThresholdPreset(
                speed=apply_slack(QWEN3_OMNI_MMSU_P95),
                accuracy=QWEN3_OMNI_MMSU_MIN_ACCURACY,
            ),
            "mmsu_talker": OmniCiThresholdPreset(
                speed=apply_slack(QWEN3_OMNI_MMSU_TALKER_P95),
                accuracy=QWEN3_OMNI_MMSU_TALKER_MIN_ACCURACY,
                wer=apply_wer_slack(QWEN3_OMNI_MMSU_TALKER_WER_BELOW_50_CORPUS_MAX),
                n_above_50=QWEN3_OMNI_MMSU_TALKER_N_ABOVE_50_MAX,
            ),
            "videomme": OmniCiThresholdPreset(
                speed=apply_slack(QWEN3_OMNI_VIDEOMME_P95),
                accuracy=QWEN3_OMNI_VIDEOMME_MIN_ACCURACY,
            ),
            "videomme_talker": OmniCiThresholdPreset(
                speed=apply_slack(QWEN3_OMNI_VIDEOMME_TALKER_P95),
                accuracy=QWEN3_OMNI_VIDEOMME_TALKER_MIN_ACCURACY,
                wer=apply_wer_slack(QWEN3_OMNI_VIDEOMME_TALKER_WER_BELOW_50_CORPUS_MAX),
                n_above_50=QWEN3_OMNI_VIDEOMME_TALKER_N_ABOVE_50_MAX,
            ),
            "videoamme": OmniCiThresholdPreset(
                speed=apply_slack(QWEN3_OMNI_VIDEOAMME_P95),
                accuracy=QWEN3_OMNI_VIDEOAMME_MIN_ACCURACY,
            ),
            "videoamme_talker": OmniCiThresholdPreset(
                speed=apply_slack(QWEN3_OMNI_VIDEOAMME_TALKER_P95),
                accuracy=QWEN3_OMNI_VIDEOAMME_TALKER_MIN_ACCURACY,
                wer=apply_wer_slack(
                    QWEN3_OMNI_VIDEOAMME_TALKER_WER_BELOW_50_CORPUS_MAX
                ),
                n_above_50=QWEN3_OMNI_VIDEOAMME_TALKER_N_ABOVE_50_MAX,
            ),
        },
    ),
    "minicpmo": OmniCiModelPreset(
        name="minicpmo",
        model_path="openbmb/MiniCPM-o-4_5",
        reference_audio_field="audio.ref_audio",
        thresholds={
            "tts": OmniCiThresholdPreset(
                speed=apply_slack(MINICPMO_TTS_P95),
                wer=apply_wer_slack(MINICPMO_TTS_WER_BELOW_50_CORPUS_MAX),
                n_above_50=MINICPMO_TTS_N_ABOVE_50_MAX,
                similarity=MINICPMO_TTS_SIMILARITY_MEAN_MIN,
                utmos=apply_mos_slack(MINICPMO_TTS_UTMOS_MEAN_REFERENCE),
                calibrated=True,
            ),
            "mmmu": OmniCiThresholdPreset(
                speed=apply_slack(MINICPMO_MMMU_P95),
                accuracy=MINICPMO_MMMU_MIN_ACCURACY,
                calibrated=True,
            ),
            "mmmu_talker": OmniCiThresholdPreset(
                speed=apply_slack(MINICPMO_MMMU_TALKER_P95),
                accuracy=MINICPMO_MMMU_TALKER_MIN_ACCURACY,
                wer=apply_wer_slack(MINICPMO_MMMU_TALKER_WER_BELOW_50_CORPUS_MAX),
                n_above_50=MINICPMO_MMMU_TALKER_N_ABOVE_50_MAX,
                calibrated=True,
            ),
            "mmsu": OmniCiThresholdPreset(
                speed=apply_slack(MINICPMO_MMSU_P95),
                accuracy=MINICPMO_MMSU_MIN_ACCURACY,
                calibrated=True,
            ),
            "mmsu_talker": OmniCiThresholdPreset(
                speed=apply_slack(MINICPMO_MMSU_TALKER_P95),
                accuracy=MINICPMO_MMSU_TALKER_MIN_ACCURACY,
                wer=apply_wer_slack(MINICPMO_MMSU_TALKER_WER_BELOW_50_CORPUS_MAX),
                n_above_50=MINICPMO_MMSU_TALKER_N_ABOVE_50_MAX,
                calibrated=True,
            ),
            "videomme": OmniCiThresholdPreset(
                speed=apply_slack(MINICPMO_VIDEOMME_P95),
                accuracy=MINICPMO_VIDEOMME_MIN_ACCURACY,
                calibrated=True,
            ),
            "videomme_talker": OmniCiThresholdPreset(
                speed=apply_slack(MINICPMO_VIDEOMME_TALKER_P95),
                accuracy=MINICPMO_VIDEOMME_TALKER_MIN_ACCURACY,
                wer=apply_wer_slack(MINICPMO_VIDEOMME_TALKER_WER_BELOW_50_CORPUS_MAX),
                n_above_50=MINICPMO_VIDEOMME_TALKER_N_ABOVE_50_MAX,
                calibrated=True,
            ),
            "videoamme": OmniCiThresholdPreset(
                speed=apply_slack(MINICPMO_VIDEOAMME_P95),
                accuracy=MINICPMO_VIDEOAMME_MIN_ACCURACY,
                calibrated=True,
            ),
            "videoamme_talker": OmniCiThresholdPreset(
                speed=apply_slack(MINICPMO_VIDEOAMME_TALKER_P95),
                accuracy=MINICPMO_VIDEOAMME_TALKER_MIN_ACCURACY,
                wer=apply_wer_slack(MINICPMO_VIDEOAMME_TALKER_WER_BELOW_50_CORPUS_MAX),
                n_above_50=MINICPMO_VIDEOAMME_TALKER_N_ABOVE_50_MAX,
                calibrated=True,
            ),
        },
    ),
}
OMNI_CI_PRESETS["qwen3-omni"].thresholds["tts"].speed[16]["rtf_mean_max"] = min(
    OMNI_CI_PRESETS["qwen3-omni"].thresholds["tts"].speed[16]["rtf_mean_max"],
    QWEN3_OMNI_SEEDTTS_RTF_MEAN_MAX,
)


def select_omni_ci_preset(
    model_name: str | None = None,
) -> tuple[str, OmniCiModelPreset]:
    selected = model_name or os.environ.get("OMNI_CI_MODEL", "qwen3-omni")
    if selected not in OMNI_CI_PRESETS:
        allowed = ", ".join(sorted(OMNI_CI_PRESETS))
        raise ValueError(
            f"Unsupported OMNI_CI_MODEL={selected!r}; expected one of: {allowed}"
        )
    return selected, OMNI_CI_PRESETS[selected]
