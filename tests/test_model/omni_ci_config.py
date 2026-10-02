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


# TTS speed comes from #1021; its similarity floor stays disabled pending #483.
QWEN3_OMNI_TTS_P95 = {
    16: {
        "throughput_qps": 12.35,
        "output_tok_per_req_s": 12.2,
        "latency_mean_s": 1.198,
        "rtf_mean": 0.3442,
    }
}
QWEN3_OMNI_TTS_WER_BELOW_50_CORPUS_MAX = 0.0213
QWEN3_OMNI_TTS_N_ABOVE_50_MAX = 0
QWEN3_OMNI_TTS_SIMILARITY_MEAN_MIN = 60.0
QWEN3_OMNI_TTS_UTMOS_MEAN_REFERENCE = 4.4507

QWEN3_OMNI_MMMU_P95 = {
    16: {"throughput_qps": 1.934, "output_tok_per_req_s": 91.8, "latency_mean_s": 7.126}
}
QWEN3_OMNI_MMMU_MIN_ACCURACY = 0.6

QWEN3_OMNI_MMMU_TALKER_P95 = {
    16: {
        "throughput_qps": 1.042,
        "output_tok_per_req_s": 15.1,
        "latency_mean_s": 9.192,
        "rtf_mean": 0.2193,
    }
}
QWEN3_OMNI_MMMU_TALKER_MIN_ACCURACY = 0.7
QWEN3_OMNI_MMMU_TALKER_WER_BELOW_50_CORPUS_MAX = 0.1449
QWEN3_OMNI_MMMU_TALKER_N_ABOVE_50_MAX = 3.0

QWEN3_OMNI_MMSU_P95 = {
    16: {
        "throughput_qps": 85.514,
        "output_tok_per_req_s": 11.1,
        "latency_mean_s": 0.187,
    }
}
QWEN3_OMNI_MMSU_MIN_ACCURACY = 0.7055

QWEN3_OMNI_MMSU_TALKER_P95 = {
    16: {
        "throughput_qps": 1.985,
        "output_tok_per_req_s": 8.2,
        "latency_mean_s": 7.585,
        "rtf_mean": 0.3812,
    }
}
QWEN3_OMNI_MMSU_TALKER_MIN_ACCURACY = 0.625
QWEN3_OMNI_MMSU_TALKER_WER_BELOW_50_CORPUS_MAX = 0.0279
QWEN3_OMNI_MMSU_TALKER_N_ABOVE_50_MAX = 0.0

QWEN3_OMNI_VIDEOMME_P95 = {
    16: {
        "throughput_qps": 1.301,
        "output_tok_per_req_s": 10.1,
        "latency_mean_s": 10.727,
    }
}
QWEN3_OMNI_VIDEOMME_MIN_ACCURACY = 0.58

QWEN3_OMNI_VIDEOMME_TALKER_P95 = {
    16: {
        "throughput_qps": 1.139,
        "output_tok_per_req_s": 4.8,
        "latency_mean_s": 9.62,
        "rtf_mean": 0.8202,
    }
}
QWEN3_OMNI_VIDEOMME_TALKER_MIN_ACCURACY = 0.6
QWEN3_OMNI_VIDEOMME_TALKER_WER_BELOW_50_CORPUS_MAX = 0.0442
QWEN3_OMNI_VIDEOMME_TALKER_N_ABOVE_50_MAX = 0.0

QWEN3_OMNI_VIDEOAMME_P95 = {
    16: {"throughput_qps": 1.686, "output_tok_per_req_s": 6.0, "latency_mean_s": 8.173}
}
QWEN3_OMNI_VIDEOAMME_MIN_ACCURACY = 0.68

QWEN3_OMNI_VIDEOAMME_TALKER_P95 = {
    16: {
        "throughput_qps": 0.234,
        "output_tok_per_req_s": 1.2,
        "latency_mean_s": 40.443,
        "rtf_mean": 2.7832,
    }
}
QWEN3_OMNI_VIDEOAMME_TALKER_MIN_ACCURACY = 0.5
QWEN3_OMNI_VIDEOAMME_TALKER_WER_BELOW_50_CORPUS_MAX = 0.0113
QWEN3_OMNI_VIDEOAMME_TALKER_N_ABOVE_50_MAX = 0.0

# note (wenyao): Raw references measured on H100 with DP2; slack is applied once below.
MINICPMO_TTS_P95 = {
    16: {
        "throughput_qps": 7.82,
        "output_tok_per_req_s": 8.1,
        "latency_mean_s": 1.91,
        "rtf_mean": 0.4754,
    }
}
MINICPMO_TTS_WER_BELOW_50_CORPUS_MAX = 0.0108
MINICPMO_TTS_N_ABOVE_50_MAX = 1.0
MINICPMO_TTS_SIMILARITY_MEAN_MIN = 43.3565
MINICPMO_TTS_UTMOS_MEAN_REFERENCE = 4.2885

MINICPMO_MMMU_P95 = {
    16: {
        "throughput_qps": 2.236,
        "output_tok_per_req_s": 129.6,
        "latency_mean_s": 5.832,
    }
}
MINICPMO_MMMU_MIN_ACCURACY = 0.64

MINICPMO_MMMU_TALKER_P95 = {
    16: {
        "throughput_qps": 1.31,
        "output_tok_per_req_s": 17.7,
        "latency_mean_s": 7.99,
        "rtf_mean": 0.1952,
    }
}
MINICPMO_MMMU_TALKER_MIN_ACCURACY = 0.7
MINICPMO_MMMU_TALKER_WER_BELOW_50_CORPUS_MAX = 0.2543
MINICPMO_MMMU_TALKER_N_ABOVE_50_MAX = 8.0

MINICPMO_MMSU_P95 = {
    16: {
        "throughput_qps": 40.869,
        "output_tok_per_req_s": 22.5,
        "latency_mean_s": 0.39,
    }
}
MINICPMO_MMSU_MIN_ACCURACY = 0.533

MINICPMO_MMSU_TALKER_P95 = {
    16: {
        "throughput_qps": 3.008,
        "output_tok_per_req_s": 10.3,
        "latency_mean_s": 4.661,
        "rtf_mean": 0.3418,
    }
}
MINICPMO_MMSU_TALKER_MIN_ACCURACY = 0.6
MINICPMO_MMSU_TALKER_WER_BELOW_50_CORPUS_MAX = 0.018
MINICPMO_MMSU_TALKER_N_ABOVE_50_MAX = 0.0

MINICPMO_VIDEOMME_P95 = {
    16: {"throughput_qps": 0.668, "output_tok_per_req_s": 3.3, "latency_mean_s": 20.606}
}
MINICPMO_VIDEOMME_MIN_ACCURACY = 0.64

MINICPMO_VIDEOMME_TALKER_P95 = {
    16: {
        "throughput_qps": 0.68,
        "output_tok_per_req_s": 2.0,
        "latency_mean_s": 15.284,
        "rtf_mean": 1.8585,
    }
}
MINICPMO_VIDEOMME_TALKER_MIN_ACCURACY = 0.55
MINICPMO_VIDEOMME_TALKER_WER_BELOW_50_CORPUS_MAX = 0.0713
MINICPMO_VIDEOMME_TALKER_N_ABOVE_50_MAX = 0.0

MINICPMO_VIDEOAMME_P95 = {
    16: {"throughput_qps": 0.736, "output_tok_per_req_s": 1.9, "latency_mean_s": 18.631}
}
MINICPMO_VIDEOAMME_MIN_ACCURACY = 0.66

MINICPMO_VIDEOAMME_TALKER_P95 = {
    16: {
        "throughput_qps": 0.669,
        "output_tok_per_req_s": 3.2,
        "latency_mean_s": 9.73,
        "rtf_mean": 1.2842,
    }
}
MINICPMO_VIDEOAMME_TALKER_MIN_ACCURACY = 0.7
MINICPMO_VIDEOAMME_TALKER_WER_BELOW_50_CORPUS_MAX = 0.0041
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
