# SPDX-License-Identifier: Apache-2.0
"""Model presets and thresholds for the SeedTTS ASR CI (ASR CI stage 2).

Mirrors tests/test_model/tts_ci_config.py: each preset bundles the model
path and calibrated gate thresholds for one ASR model, and CI selects one
preset per run through the ASR_CI_MODEL environment variable (or the
--asr-ci-model pytest option). tests/test_model/test_asr_ci_seedtts.py is
model-agnostic and reads everything model-specific from the selected preset.

Threshold constants live here (not in the test file) so that the
tune-ci-thresholds skill can read and rewrite them; the skill locates this
file through the `threshold_file` key in
.claude/skills/tune-ci-thresholds/models/asr/config.yaml and claims each
preset's constants by name prefix (e.g. ^FUN_ASR_).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from benchmarks.tasks.asr import (
    FUN_ASR_MODEL_PATH,
    OMNI_WHISPER_MODEL_PATH,
    QWEN3_ASR_MODEL_PATH,
)
from tests.utils import apply_wer_slack


@dataclass(frozen=True)
class AsrCiThresholdPreset:
    """CI gate values with slack already applied."""

    en_corpus_wer_max: float
    en_sample_wer_max: float
    zh_corpus_wer_max: float
    zh_sample_wer_max: float
    throughput_min: float
    latency_mean_max_s: float
    latency_p95_max_s: float
    rtf_mean_max: float
    rtf_p95_max: float


@dataclass(frozen=True)
class AsrCiPreset:
    model_path: str
    display_name: str
    thresholds: AsrCiThresholdPreset
    # note (Jeffro): False means threshold assertions are skipped while a
    # newly onboarded model awaits calibration;
    gate_thresholds: bool = True


# Slack factors applied to worst-of-N reference values to derive CI gates.
# Higher-is-better metrics: threshold = reference * THRESHOLD_SLACK_HIGHER.
# Lower-is-better metrics: threshold = reference * THRESHOLD_SLACK_LOWER.
THRESHOLD_SLACK_HIGHER = 0.9
THRESHOLD_SLACK_LOWER = 1.1


# note (Jeffro): thresholds ported from #1220.
FUN_ASR_EN_CORPUS_WER_MAX = 0.0172
FUN_ASR_EN_SAMPLE_WER_MAX = 0.3077
FUN_ASR_ZH_CORPUS_WER_MAX = 0.0138
FUN_ASR_ZH_SAMPLE_WER_MAX = 0.8334
FUN_ASR_THROUGHPUT_MIN = 214.5627738182959
FUN_ASR_LATENCY_MEAN_MAX_S = 0.145
FUN_ASR_LATENCY_P95_MAX_S = 0.33612206925172283
FUN_ASR_RTF_MEAN_MAX = 0.03182396443701368
FUN_ASR_RTF_P95_MAX = 0.0781

FUN_ASR_EN_CORPUS_WER_THRESHOLD = apply_wer_slack(
    FUN_ASR_EN_CORPUS_WER_MAX, THRESHOLD_SLACK_LOWER
)
FUN_ASR_EN_SAMPLE_WER_THRESHOLD = apply_wer_slack(
    FUN_ASR_EN_SAMPLE_WER_MAX, THRESHOLD_SLACK_LOWER
)
FUN_ASR_ZH_CORPUS_WER_THRESHOLD = apply_wer_slack(
    FUN_ASR_ZH_CORPUS_WER_MAX, THRESHOLD_SLACK_LOWER
)
FUN_ASR_ZH_SAMPLE_WER_THRESHOLD = apply_wer_slack(
    FUN_ASR_ZH_SAMPLE_WER_MAX, THRESHOLD_SLACK_LOWER
)
FUN_ASR_THROUGHPUT_THRESHOLD = round(FUN_ASR_THROUGHPUT_MIN * THRESHOLD_SLACK_HIGHER, 3)
FUN_ASR_LATENCY_MEAN_THRESHOLD_S = round(
    FUN_ASR_LATENCY_MEAN_MAX_S * THRESHOLD_SLACK_LOWER, 3
)
FUN_ASR_LATENCY_P95_THRESHOLD_S = round(
    FUN_ASR_LATENCY_P95_MAX_S * THRESHOLD_SLACK_LOWER, 3
)
FUN_ASR_RTF_MEAN_THRESHOLD = round(FUN_ASR_RTF_MEAN_MAX * THRESHOLD_SLACK_LOWER, 4)
FUN_ASR_RTF_P95_THRESHOLD = round(FUN_ASR_RTF_P95_MAX * THRESHOLD_SLACK_LOWER, 4)

QWEN3_ASR_EN_CORPUS_WER_MAX = 0.0122
QWEN3_ASR_EN_SAMPLE_WER_MAX = 0.1819
QWEN3_ASR_ZH_CORPUS_WER_MAX = 0.0061
QWEN3_ASR_ZH_SAMPLE_WER_MAX = 0.2106
QWEN3_ASR_THROUGHPUT_MIN = 245.6725125706553
QWEN3_ASR_LATENCY_MEAN_MAX_S = 0.12809534221485272
QWEN3_ASR_LATENCY_P95_MAX_S = 0.1924563570413738
QWEN3_ASR_RTF_MEAN_MAX = 0.0279
QWEN3_ASR_RTF_P95_MAX = 0.0441

QWEN3_ASR_EN_CORPUS_WER_THRESHOLD = apply_wer_slack(
    QWEN3_ASR_EN_CORPUS_WER_MAX, THRESHOLD_SLACK_LOWER
)
QWEN3_ASR_EN_SAMPLE_WER_THRESHOLD = apply_wer_slack(
    QWEN3_ASR_EN_SAMPLE_WER_MAX, THRESHOLD_SLACK_LOWER
)
QWEN3_ASR_ZH_CORPUS_WER_THRESHOLD = apply_wer_slack(
    QWEN3_ASR_ZH_CORPUS_WER_MAX, THRESHOLD_SLACK_LOWER
)
QWEN3_ASR_ZH_SAMPLE_WER_THRESHOLD = apply_wer_slack(
    QWEN3_ASR_ZH_SAMPLE_WER_MAX, THRESHOLD_SLACK_LOWER
)
QWEN3_ASR_THROUGHPUT_THRESHOLD = round(
    QWEN3_ASR_THROUGHPUT_MIN * THRESHOLD_SLACK_HIGHER, 3
)
QWEN3_ASR_LATENCY_MEAN_THRESHOLD_S = round(
    QWEN3_ASR_LATENCY_MEAN_MAX_S * THRESHOLD_SLACK_LOWER, 3
)
QWEN3_ASR_LATENCY_P95_THRESHOLD_S = round(
    QWEN3_ASR_LATENCY_P95_MAX_S * THRESHOLD_SLACK_LOWER, 3
)
QWEN3_ASR_RTF_MEAN_THRESHOLD = round(QWEN3_ASR_RTF_MEAN_MAX * THRESHOLD_SLACK_LOWER, 4)
QWEN3_ASR_RTF_P95_THRESHOLD = round(QWEN3_ASR_RTF_P95_MAX * THRESHOLD_SLACK_LOWER, 4)


# note (Junnan Li): worst-of-5 raw references from tune-ci-thresholds on the H100
# CI lane 0,1 (main 201e5572 + this preset).
WHISPER_ASR_EN_CORPUS_WER_MAX = 0.0138
WHISPER_ASR_EN_SAMPLE_WER_MAX = 0.2858
WHISPER_ASR_ZH_CORPUS_WER_MAX = 0.0667
WHISPER_ASR_ZH_SAMPLE_WER_MAX = 0.75
WHISPER_ASR_THROUGHPUT_MIN = 115.38853978219092
WHISPER_ASR_LATENCY_MEAN_MAX_S = 0.275390744645633
WHISPER_ASR_LATENCY_P95_MAX_S = 0.3991700013633816
WHISPER_ASR_RTF_MEAN_MAX = 0.0598
WHISPER_ASR_RTF_P95_MAX = 0.0868

WHISPER_ASR_EN_CORPUS_WER_THRESHOLD = apply_wer_slack(
    WHISPER_ASR_EN_CORPUS_WER_MAX, THRESHOLD_SLACK_LOWER
)
WHISPER_ASR_EN_SAMPLE_WER_THRESHOLD = apply_wer_slack(
    WHISPER_ASR_EN_SAMPLE_WER_MAX, THRESHOLD_SLACK_LOWER
)
WHISPER_ASR_ZH_CORPUS_WER_THRESHOLD = apply_wer_slack(
    WHISPER_ASR_ZH_CORPUS_WER_MAX, THRESHOLD_SLACK_LOWER
)
WHISPER_ASR_ZH_SAMPLE_WER_THRESHOLD = apply_wer_slack(
    WHISPER_ASR_ZH_SAMPLE_WER_MAX, THRESHOLD_SLACK_LOWER
)
WHISPER_ASR_THROUGHPUT_THRESHOLD = round(
    WHISPER_ASR_THROUGHPUT_MIN * THRESHOLD_SLACK_HIGHER, 3
)
WHISPER_ASR_LATENCY_MEAN_THRESHOLD_S = round(
    WHISPER_ASR_LATENCY_MEAN_MAX_S * THRESHOLD_SLACK_LOWER, 3
)
WHISPER_ASR_LATENCY_P95_THRESHOLD_S = round(
    WHISPER_ASR_LATENCY_P95_MAX_S * THRESHOLD_SLACK_LOWER, 3
)
WHISPER_ASR_RTF_MEAN_THRESHOLD = round(
    WHISPER_ASR_RTF_MEAN_MAX * THRESHOLD_SLACK_LOWER, 4
)
WHISPER_ASR_RTF_P95_THRESHOLD = round(
    WHISPER_ASR_RTF_P95_MAX * THRESHOLD_SLACK_LOWER, 4
)


ASR_CI_PRESETS: dict[str, AsrCiPreset] = {
    "fun": AsrCiPreset(
        model_path=FUN_ASR_MODEL_PATH,
        display_name="Fun-ASR",
        thresholds=AsrCiThresholdPreset(
            en_corpus_wer_max=FUN_ASR_EN_CORPUS_WER_THRESHOLD,
            en_sample_wer_max=FUN_ASR_EN_SAMPLE_WER_THRESHOLD,
            zh_corpus_wer_max=FUN_ASR_ZH_CORPUS_WER_THRESHOLD,
            zh_sample_wer_max=FUN_ASR_ZH_SAMPLE_WER_THRESHOLD,
            throughput_min=FUN_ASR_THROUGHPUT_THRESHOLD,
            latency_mean_max_s=FUN_ASR_LATENCY_MEAN_THRESHOLD_S,
            latency_p95_max_s=FUN_ASR_LATENCY_P95_THRESHOLD_S,
            rtf_mean_max=FUN_ASR_RTF_MEAN_THRESHOLD,
            rtf_p95_max=FUN_ASR_RTF_P95_THRESHOLD,
        ),
    ),
    "qwen3": AsrCiPreset(
        model_path=QWEN3_ASR_MODEL_PATH,
        display_name="Qwen3-ASR",
        thresholds=AsrCiThresholdPreset(
            en_corpus_wer_max=QWEN3_ASR_EN_CORPUS_WER_THRESHOLD,
            en_sample_wer_max=QWEN3_ASR_EN_SAMPLE_WER_THRESHOLD,
            zh_corpus_wer_max=QWEN3_ASR_ZH_CORPUS_WER_THRESHOLD,
            zh_sample_wer_max=QWEN3_ASR_ZH_SAMPLE_WER_THRESHOLD,
            throughput_min=QWEN3_ASR_THROUGHPUT_THRESHOLD,
            latency_mean_max_s=QWEN3_ASR_LATENCY_MEAN_THRESHOLD_S,
            latency_p95_max_s=QWEN3_ASR_LATENCY_P95_THRESHOLD_S,
            rtf_mean_max=QWEN3_ASR_RTF_MEAN_THRESHOLD,
            rtf_p95_max=QWEN3_ASR_RTF_P95_THRESHOLD,
        ),
    ),
    "whisper": AsrCiPreset(
        model_path=OMNI_WHISPER_MODEL_PATH,
        display_name="Whisper-large-v3",
        thresholds=AsrCiThresholdPreset(
            en_corpus_wer_max=WHISPER_ASR_EN_CORPUS_WER_THRESHOLD,
            en_sample_wer_max=WHISPER_ASR_EN_SAMPLE_WER_THRESHOLD,
            zh_corpus_wer_max=WHISPER_ASR_ZH_CORPUS_WER_THRESHOLD,
            zh_sample_wer_max=WHISPER_ASR_ZH_SAMPLE_WER_THRESHOLD,
            throughput_min=WHISPER_ASR_THROUGHPUT_THRESHOLD,
            latency_mean_max_s=WHISPER_ASR_LATENCY_MEAN_THRESHOLD_S,
            latency_p95_max_s=WHISPER_ASR_LATENCY_P95_THRESHOLD_S,
            rtf_mean_max=WHISPER_ASR_RTF_MEAN_THRESHOLD,
            rtf_p95_max=WHISPER_ASR_RTF_P95_THRESHOLD,
        ),
    ),
}


def select_asr_ci_preset(model_name: str | None = None) -> tuple[str, AsrCiPreset]:
    selected = model_name or os.environ.get("ASR_CI_MODEL", "fun")
    preset = ASR_CI_PRESETS.get(selected)
    if preset is None:
        allowed = ", ".join(sorted(ASR_CI_PRESETS))
        raise ValueError(
            f"Unsupported ASR_CI_MODEL={selected!r}; expected one of: {allowed}"
        )
    return selected, preset
