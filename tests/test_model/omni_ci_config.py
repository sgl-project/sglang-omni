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
        "throughput_qps": 15.206,
        "output_tok_per_req_s": 14.9,
        "latency_mean_s": 0.986,
        "rtf_mean": 0.2926,
    }
}
QWEN3_OMNI_TTS_WER_BELOW_50_CORPUS_MAX = 0.0196
QWEN3_OMNI_TTS_N_ABOVE_50_MAX = 0
QWEN3_OMNI_TTS_SIMILARITY_MEAN_MIN = 60.0
QWEN3_OMNI_TTS_UTMOS_MEAN_REFERENCE = 4.4508

QWEN3_OMNI_MMMU_P95 = {
    16: {"throughput_qps": 1.67, "output_tok_per_req_s": 91.1, "latency_mean_s": 6.865}
}
QWEN3_OMNI_MMMU_MIN_ACCURACY = 0.62

QWEN3_OMNI_MMMU_TALKER_P95 = {
    16: {
        "throughput_qps": 1.769,
        "output_tok_per_req_s": 22.3,
        "latency_mean_s": 6.288,
        "rtf_mean": 0.1502,
    }
}
QWEN3_OMNI_MMMU_TALKER_MIN_ACCURACY = 0.7
QWEN3_OMNI_MMMU_TALKER_WER_BELOW_50_CORPUS_MAX = 0.1738
QWEN3_OMNI_MMMU_TALKER_N_ABOVE_50_MAX = 3.0

QWEN3_OMNI_MMSU_P95 = {
    16: {
        "throughput_qps": 80.885,
        "output_tok_per_req_s": 10.5,
        "latency_mean_s": 0.197,
    }
}
QWEN3_OMNI_MMSU_MIN_ACCURACY = 0.7055

QWEN3_OMNI_MMSU_TALKER_P95 = {
    16: {
        "throughput_qps": 2.56,
        "output_tok_per_req_s": 10.5,
        "latency_mean_s": 5.876,
        "rtf_mean": 0.2967,
    }
}
QWEN3_OMNI_MMSU_TALKER_MIN_ACCURACY = 0.625
QWEN3_OMNI_MMSU_TALKER_WER_BELOW_50_CORPUS_MAX = 0.0306
QWEN3_OMNI_MMSU_TALKER_N_ABOVE_50_MAX = 0.0

QWEN3_OMNI_VIDEOMME_P95 = {
    16: {
        "throughput_qps": 1.199,
        "output_tok_per_req_s": 9.8,
        "latency_mean_s": 11.53,
    }
}
QWEN3_OMNI_VIDEOMME_MIN_ACCURACY = 0.56

QWEN3_OMNI_VIDEOMME_TALKER_P95 = {
    16: {
        "throughput_qps": 1.112,
        "output_tok_per_req_s": 4.8,
        "latency_mean_s": 9.557,
        "rtf_mean": 0.8486,
    }
}
QWEN3_OMNI_VIDEOMME_TALKER_MIN_ACCURACY = 0.6
QWEN3_OMNI_VIDEOMME_TALKER_WER_BELOW_50_CORPUS_MAX = 0.0332
QWEN3_OMNI_VIDEOMME_TALKER_N_ABOVE_50_MAX = 1.0

QWEN3_OMNI_VIDEOAMME_P95 = {
    16: {"throughput_qps": 1.684, "output_tok_per_req_s": 5.8, "latency_mean_s": 8.184}
}
QWEN3_OMNI_VIDEOAMME_MIN_ACCURACY = 0.68

QWEN3_OMNI_VIDEOAMME_TALKER_P95 = {
    16: {
        "throughput_qps": 0.241,
        "output_tok_per_req_s": 1.2,
        "latency_mean_s": 39.672,
        "rtf_mean": 3.0823,
    }
}
QWEN3_OMNI_VIDEOAMME_TALKER_MIN_ACCURACY = 0.5
QWEN3_OMNI_VIDEOAMME_TALKER_WER_BELOW_50_CORPUS_MAX = 0.0157
QWEN3_OMNI_VIDEOAMME_TALKER_N_ABOVE_50_MAX = 0.0

# note (wenyao): Raw references measured on H100 with DP2; slack is applied once below.
MINICPMO_TTS_P95 = {
    16: {
        "throughput_qps": 12.41,
        "output_tok_per_req_s": 13.2,
        "latency_mean_s": 1.181,
        "rtf_mean": 0.2919,
    }
}
MINICPMO_TTS_WER_BELOW_50_CORPUS_MAX = 0.0215
MINICPMO_TTS_N_ABOVE_50_MAX = 1.0
MINICPMO_TTS_SIMILARITY_MEAN_MIN = 42.86711128234863
MINICPMO_TTS_UTMOS_MEAN_REFERENCE = 4.2922

MINICPMO_MMMU_P95 = {
    16: {
        "throughput_qps": 2.269,
        "output_tok_per_req_s": 129.5,
        "latency_mean_s": 5.729,
    }
}
MINICPMO_MMMU_MIN_ACCURACY = 0.64

MINICPMO_MMMU_TALKER_P95 = {
    16: {
        "throughput_qps": 1.648,
        "output_tok_per_req_s": 21.6,
        "latency_mean_s": 6.846,
        "rtf_mean": 0.1684,
    }
}
MINICPMO_MMMU_TALKER_MIN_ACCURACY = 0.6
MINICPMO_MMMU_TALKER_WER_BELOW_50_CORPUS_MAX = 0.2594
MINICPMO_MMMU_TALKER_N_ABOVE_50_MAX = 9.0

MINICPMO_MMSU_P95 = {
    16: {
        "throughput_qps": 39.784,
        "output_tok_per_req_s": 21.9,
        "latency_mean_s": 0.4,
    }
}
MINICPMO_MMSU_MIN_ACCURACY = 0.533

MINICPMO_MMSU_TALKER_P95 = {
    16: {
        "throughput_qps": 3.758,
        "output_tok_per_req_s": 12.2,
        "latency_mean_s": 3.873,
        "rtf_mean": 0.2787,
    }
}
MINICPMO_MMSU_TALKER_MIN_ACCURACY = 0.6
MINICPMO_MMSU_TALKER_WER_BELOW_50_CORPUS_MAX = 0.0213
MINICPMO_MMSU_TALKER_N_ABOVE_50_MAX = 0.0

MINICPMO_VIDEOMME_P95 = {
    16: {"throughput_qps": 0.708, "output_tok_per_req_s": 3.5, "latency_mean_s": 19.313}
}
MINICPMO_VIDEOMME_MIN_ACCURACY = 0.64

MINICPMO_VIDEOMME_TALKER_P95 = {
    16: {
        "throughput_qps": 0.665,
        "output_tok_per_req_s": 1.9,
        "latency_mean_s": 15.589,
        "rtf_mean": 1.8954,
    }
}
MINICPMO_VIDEOMME_TALKER_MIN_ACCURACY = 0.55
MINICPMO_VIDEOMME_TALKER_WER_BELOW_50_CORPUS_MAX = 0.0631
MINICPMO_VIDEOMME_TALKER_N_ABOVE_50_MAX = 0.0

MINICPMO_VIDEOAMME_P95 = {
    16: {"throughput_qps": 0.709, "output_tok_per_req_s": 1.8, "latency_mean_s": 19.278}
}
MINICPMO_VIDEOAMME_MIN_ACCURACY = 0.66

MINICPMO_VIDEOAMME_TALKER_P95 = {
    16: {
        "throughput_qps": 0.643,
        "output_tok_per_req_s": 3.1,
        "latency_mean_s": 10.123,
        "rtf_mean": 1.3387,
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
