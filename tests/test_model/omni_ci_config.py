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
        "throughput_qps": 10.228,
        "output_tok_per_req_s": 16.2,
        "latency_mean_s": 0.903,
        "rtf_mean": 0.2726,
    }
}
QWEN3_OMNI_TTS_WER_BELOW_50_CORPUS_MAX = 0.0249
QWEN3_OMNI_TTS_N_ABOVE_50_MAX = 0
QWEN3_OMNI_TTS_SIMILARITY_MEAN_MIN = 60.0
QWEN3_OMNI_TTS_UTMOS_MEAN_REFERENCE = 4.4312

QWEN3_OMNI_MMMU_P95 = {
    16: {"throughput_qps": 1.948, "output_tok_per_req_s": 95.9, "latency_mean_s": 7.394}
}
QWEN3_OMNI_MMMU_MIN_ACCURACY = 0.58

QWEN3_OMNI_MMMU_TALKER_P95 = {
    16: {
        "throughput_qps": 2.144,
        "output_tok_per_req_s": 27.6,
        "latency_mean_s": 4.819,
        "rtf_mean": 0.1217,
    }
}
QWEN3_OMNI_MMMU_TALKER_MIN_ACCURACY = 0.7
QWEN3_OMNI_MMMU_TALKER_WER_BELOW_50_CORPUS_MAX = 0.1539
QWEN3_OMNI_MMMU_TALKER_N_ABOVE_50_MAX = 3.0

QWEN3_OMNI_MMSU_P95 = {
    16: {
        "throughput_qps": 118.685,
        "output_tok_per_req_s": 15.4,
        "latency_mean_s": 0.134,
    },
    96: {
        "throughput_qps": 234.11,
        "output_tok_per_req_s": 5.1,
        "latency_mean_s": 0.405,
    },
}
QWEN3_OMNI_MMSU_MIN_ACCURACY = 0.709

QWEN3_OMNI_MMSU_TALKER_P95 = {
    16: {
        "throughput_qps": 3.123,
        "output_tok_per_req_s": 13.2,
        "latency_mean_s": 4.712,
        "rtf_mean": 0.2391,
    }
}
QWEN3_OMNI_MMSU_TALKER_MIN_ACCURACY = 0.625
QWEN3_OMNI_MMSU_TALKER_WER_BELOW_50_CORPUS_MAX = 0.0247
QWEN3_OMNI_MMSU_TALKER_N_ABOVE_50_MAX = 0.0

QWEN3_OMNI_VIDEOMME_P95 = {
    16: {
        "throughput_qps": 1.528,
        "output_tok_per_req_s": 13.2,
        "latency_mean_s": 8.991,
    }
}
QWEN3_OMNI_VIDEOMME_MIN_ACCURACY = 0.6

QWEN3_OMNI_VIDEOMME_TALKER_P95 = {
    16: {
        "throughput_qps": 1.349,
        "output_tok_per_req_s": 5.9,
        "latency_mean_s": 7.685,
        "rtf_mean": 0.7074,
    }
}
QWEN3_OMNI_VIDEOMME_TALKER_MIN_ACCURACY = 0.6
QWEN3_OMNI_VIDEOMME_TALKER_WER_BELOW_50_CORPUS_MAX = 0.017
QWEN3_OMNI_VIDEOMME_TALKER_N_ABOVE_50_MAX = 0.0

QWEN3_OMNI_VIDEOAMME_P95 = {
    16: {"throughput_qps": 1.956, "output_tok_per_req_s": 7.5, "latency_mean_s": 7.033}
}
QWEN3_OMNI_VIDEOAMME_MIN_ACCURACY = 0.68

QWEN3_OMNI_VIDEOAMME_TALKER_P95 = {
    16: {
        "throughput_qps": 0.256,
        "output_tok_per_req_s": 1.2,
        "latency_mean_s": 37.959,
        "rtf_mean": 2.9778,
    }
}
QWEN3_OMNI_VIDEOAMME_TALKER_MIN_ACCURACY = 0.5
QWEN3_OMNI_VIDEOAMME_TALKER_WER_BELOW_50_CORPUS_MAX = 0.0079
QWEN3_OMNI_VIDEOAMME_TALKER_N_ABOVE_50_MAX = 0.0

# note (wenyao): Raw references measured on H100 with DP2; slack is applied once below.
MINICPMO_TTS_P95 = {
    16: {
        "throughput_qps": 18.622,
        "output_tok_per_req_s": 19.9,
        "latency_mean_s": 0.784,
        "rtf_mean": 0.1988,
    }
}
MINICPMO_TTS_WER_BELOW_50_CORPUS_MAX = 0.0196
MINICPMO_TTS_N_ABOVE_50_MAX = 1.0
MINICPMO_TTS_SIMILARITY_MEAN_MIN = 43.91056877136231
MINICPMO_TTS_UTMOS_MEAN_REFERENCE = 4.2745

MINICPMO_MMMU_P95 = {
    16: {
        "throughput_qps": 2.205,
        "output_tok_per_req_s": 126.6,
        "latency_mean_s": 5.8,
    }
}
MINICPMO_MMMU_MIN_ACCURACY = 0.62

MINICPMO_MMMU_TALKER_P95 = {
    16: {
        "throughput_qps": 2.202,
        "output_tok_per_req_s": 28.5,
        "latency_mean_s": 4.83,
        "rtf_mean": 0.1188,
    }
}
MINICPMO_MMMU_TALKER_MIN_ACCURACY = 0.55
MINICPMO_MMMU_TALKER_WER_BELOW_50_CORPUS_MAX = 0.205
MINICPMO_MMMU_TALKER_N_ABOVE_50_MAX = 8.0

MINICPMO_MMSU_P95 = {
    16: {
        "throughput_qps": 44.19,
        "output_tok_per_req_s": 24.4,
        "latency_mean_s": 0.36,
    }
}
MINICPMO_MMSU_MIN_ACCURACY = 0.5315

MINICPMO_MMSU_TALKER_P95 = {
    16: {
        "throughput_qps": 5.246,
        "output_tok_per_req_s": 17.5,
        "latency_mean_s": 2.62,
        "rtf_mean": 0.1918,
    }
}
MINICPMO_MMSU_TALKER_MIN_ACCURACY = 0.55
MINICPMO_MMSU_TALKER_WER_BELOW_50_CORPUS_MAX = 0.0209
MINICPMO_MMSU_TALKER_N_ABOVE_50_MAX = 0.0

MINICPMO_VIDEOMME_P95 = {
    16: {"throughput_qps": 2.321, "output_tok_per_req_s": 10.5, "latency_mean_s": 6.203}
}
MINICPMO_VIDEOMME_MIN_ACCURACY = 0.6

MINICPMO_VIDEOMME_TALKER_P95 = {
    16: {
        "throughput_qps": 1.796,
        "output_tok_per_req_s": 4.7,
        "latency_mean_s": 6.136,
        "rtf_mean": 0.7682,
    }
}
MINICPMO_VIDEOMME_TALKER_MIN_ACCURACY = 0.55
MINICPMO_VIDEOMME_TALKER_WER_BELOW_50_CORPUS_MAX = 0.0832
MINICPMO_VIDEOMME_TALKER_N_ABOVE_50_MAX = 1.0

MINICPMO_VIDEOAMME_P95 = {
    16: {"throughput_qps": 2.429, "output_tok_per_req_s": 6.2, "latency_mean_s": 5.677}
}
MINICPMO_VIDEOAMME_MIN_ACCURACY = 0.64

MINICPMO_VIDEOAMME_TALKER_P95 = {
    16: {
        "throughput_qps": 1.543,
        "output_tok_per_req_s": 5.7,
        "latency_mean_s": 5.453,
        "rtf_mean": 0.6804,
    }
}
MINICPMO_VIDEOAMME_TALKER_MIN_ACCURACY = 0.6
MINICPMO_VIDEOAMME_TALKER_WER_BELOW_50_CORPUS_MAX = 0.0404
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
