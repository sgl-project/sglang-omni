# SPDX-License-Identifier: Apache-2.0
"""Strategy presets and references for the Full-Duplex-Bench CI stages.

Each preset bundles one interaction strategy's launch shape and its gate
references, and CI selects one preset per run through the FDB_CI_STRATEGY
environment variable. tests/test_model/test_fdb_ci.py is strategy-agnostic and
tests/utils/fdb_ci.py applies the slack. The reference constants live here, not
in the test file, so that the calibrate-h100-ci skill can locate this file
through its threshold_file key and claim each strategy's constants by prefix
(^MINICPMO_FDB_, ^QWEN3_OMNI_FDB_, ^CASCADE_FDB_).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from tests.utils.fdb_ci import FdbCiStrategy

FdbCiStrategyName = Literal["full-duplex", "turn-based", "cascaded"]


@dataclass(frozen=True, kw_only=True)
class FdbCiPreset:
    """calibrated=False runs and reports the metrics but fails the stage, so
    unmeasured references never gate a pull request."""

    strategy: FdbCiStrategy
    references: dict[str, float]
    calibrated: bool


# note (jeffro): one H100 run of each selection; calibrated stays False until
# calibration replaces them.
MINICPMO_FDB_RESPOND_ON_INTERRUPTION_REFERENCE = 0.6
MINICPMO_FDB_RESUME_ON_BACKCHANNEL_REFERENCE = 1.0
MINICPMO_FDB_RESUME_ON_TALKING_TO_OTHER_REFERENCE = 0.6
MINICPMO_FDB_RESUME_ON_BACKGROUND_SPEECH_REFERENCE = 0.8
MINICPMO_FDB_OVERLAP_HANDLED_REFERENCE = 0.8
MINICPMO_FDB_GROUNDED_REPLY_REFERENCE = 0.7059
MINICPMO_FDB_STOP_LATENCY_MEAN_S_REFERENCE = 1.4665
MINICPMO_FDB_RESPONSE_LATENCY_MEAN_S_REFERENCE = 2.7855
MINICPMO_FDB_SYNTHETIC_PAUSE_INTRUSION_RATE_REFERENCE = 0.2
MINICPMO_FDB_CANDOR_PAUSE_INTRUSION_RATE_REFERENCE = 0.4
MINICPMO_FDB_TURN_RESPONSE_RATE_2S_REFERENCE = 0.6
MINICPMO_FDB_OMISSION_RATE_REFERENCE = 0.0994
MINICPMO_FDB_UTMOS_MEAN_REFERENCE = 4.2725
MINICPMO_FDB_VOICE_SIMILARITY_MEAN_REFERENCE = 74.9218

# note (jeffro): the v1.0 rates are the H200 blog table's; the earlier
# single-H100 v1.0 run ran out of memory.
QWEN3_OMNI_FDB_RESPOND_ON_INTERRUPTION_REFERENCE = 1.0
QWEN3_OMNI_FDB_RESUME_ON_BACKCHANNEL_REFERENCE = 0.8
QWEN3_OMNI_FDB_RESUME_ON_TALKING_TO_OTHER_REFERENCE = 0.0
QWEN3_OMNI_FDB_RESUME_ON_BACKGROUND_SPEECH_REFERENCE = 0.0
QWEN3_OMNI_FDB_OVERLAP_HANDLED_REFERENCE = 0.4737
QWEN3_OMNI_FDB_GROUNDED_REPLY_REFERENCE = 0.6471
QWEN3_OMNI_FDB_STOP_LATENCY_MEAN_S_REFERENCE = 0.167
QWEN3_OMNI_FDB_RESPONSE_LATENCY_MEAN_S_REFERENCE = 1.055
QWEN3_OMNI_FDB_SYNTHETIC_PAUSE_INTRUSION_RATE_REFERENCE = 0.1
QWEN3_OMNI_FDB_CANDOR_PAUSE_INTRUSION_RATE_REFERENCE = 0.1
QWEN3_OMNI_FDB_TURN_RESPONSE_RATE_2S_REFERENCE = 1.0
QWEN3_OMNI_FDB_OMISSION_RATE_REFERENCE = 0.036
QWEN3_OMNI_FDB_UTMOS_MEAN_REFERENCE = 4.477
QWEN3_OMNI_FDB_VOICE_SIMILARITY_MEAN_REFERENCE = 83.604

CASCADE_FDB_RESPOND_ON_INTERRUPTION_REFERENCE = 0.4
CASCADE_FDB_RESUME_ON_BACKCHANNEL_REFERENCE = 0.6
CASCADE_FDB_RESUME_ON_TALKING_TO_OTHER_REFERENCE = 0.0
CASCADE_FDB_RESUME_ON_BACKGROUND_SPEECH_REFERENCE = 0.2
CASCADE_FDB_OVERLAP_HANDLED_REFERENCE = 0.4737
CASCADE_FDB_GROUNDED_REPLY_REFERENCE = 0.9333
CASCADE_FDB_STOP_LATENCY_MEAN_S_REFERENCE = 0.2604
CASCADE_FDB_RESPONSE_LATENCY_MEAN_S_REFERENCE = 1.9602
CASCADE_FDB_SYNTHETIC_PAUSE_INTRUSION_RATE_REFERENCE = 0.4
CASCADE_FDB_CANDOR_PAUSE_INTRUSION_RATE_REFERENCE = 0.0
CASCADE_FDB_TURN_RESPONSE_RATE_2S_REFERENCE = 1.0
CASCADE_FDB_OMISSION_RATE_REFERENCE = 0.0
CASCADE_FDB_UTMOS_MEAN_REFERENCE = 4.3991
CASCADE_FDB_VOICE_SIMILARITY_MEAN_REFERENCE = 66.8536

MINICPMO_FDB_PRESET = FdbCiPreset(
    strategy=FdbCiStrategy(
        engine_name="minicpmo",
        gpu_arguments=("--gpus", "0,1"),
        num_shards=16,
        should_run_semantic_judge=True,
    ),
    references={
        "v15_respond_on_interruption": MINICPMO_FDB_RESPOND_ON_INTERRUPTION_REFERENCE,
        "v15_resume_on_backchannel": MINICPMO_FDB_RESUME_ON_BACKCHANNEL_REFERENCE,
        "v15_resume_on_talking_to_other": MINICPMO_FDB_RESUME_ON_TALKING_TO_OTHER_REFERENCE,
        "v15_resume_on_background_speech": MINICPMO_FDB_RESUME_ON_BACKGROUND_SPEECH_REFERENCE,
        "v15_overlap_handled": MINICPMO_FDB_OVERLAP_HANDLED_REFERENCE,
        "v15_grounded_reply": MINICPMO_FDB_GROUNDED_REPLY_REFERENCE,
        "v15_stop_latency_mean_s": MINICPMO_FDB_STOP_LATENCY_MEAN_S_REFERENCE,
        "v15_response_latency_mean_s": MINICPMO_FDB_RESPONSE_LATENCY_MEAN_S_REFERENCE,
        "v10_synthetic_pause_intrusion_rate": MINICPMO_FDB_SYNTHETIC_PAUSE_INTRUSION_RATE_REFERENCE,
        "v10_candor_pause_intrusion_rate": MINICPMO_FDB_CANDOR_PAUSE_INTRUSION_RATE_REFERENCE,
        "v10_turn_response_rate_2s": MINICPMO_FDB_TURN_RESPONSE_RATE_2S_REFERENCE,
        "listening_omission_rate": MINICPMO_FDB_OMISSION_RATE_REFERENCE,
        "listening_utmos_mean": MINICPMO_FDB_UTMOS_MEAN_REFERENCE,
        "listening_voice_similarity_mean": MINICPMO_FDB_VOICE_SIMILARITY_MEAN_REFERENCE,
    },
    calibrated=False,
)
QWEN3_OMNI_FDB_PRESET = FdbCiPreset(
    strategy=FdbCiStrategy(
        engine_name="qwen3-omni",
        gpu_arguments=(),
        num_shards=4,
        should_run_semantic_judge=True,
    ),
    references={
        "v15_respond_on_interruption": QWEN3_OMNI_FDB_RESPOND_ON_INTERRUPTION_REFERENCE,
        "v15_resume_on_backchannel": QWEN3_OMNI_FDB_RESUME_ON_BACKCHANNEL_REFERENCE,
        "v15_resume_on_talking_to_other": QWEN3_OMNI_FDB_RESUME_ON_TALKING_TO_OTHER_REFERENCE,
        "v15_resume_on_background_speech": QWEN3_OMNI_FDB_RESUME_ON_BACKGROUND_SPEECH_REFERENCE,
        "v15_overlap_handled": QWEN3_OMNI_FDB_OVERLAP_HANDLED_REFERENCE,
        "v15_grounded_reply": QWEN3_OMNI_FDB_GROUNDED_REPLY_REFERENCE,
        "v15_stop_latency_mean_s": QWEN3_OMNI_FDB_STOP_LATENCY_MEAN_S_REFERENCE,
        "v15_response_latency_mean_s": QWEN3_OMNI_FDB_RESPONSE_LATENCY_MEAN_S_REFERENCE,
        "v10_synthetic_pause_intrusion_rate": QWEN3_OMNI_FDB_SYNTHETIC_PAUSE_INTRUSION_RATE_REFERENCE,
        "v10_candor_pause_intrusion_rate": QWEN3_OMNI_FDB_CANDOR_PAUSE_INTRUSION_RATE_REFERENCE,
        "v10_turn_response_rate_2s": QWEN3_OMNI_FDB_TURN_RESPONSE_RATE_2S_REFERENCE,
        "listening_omission_rate": QWEN3_OMNI_FDB_OMISSION_RATE_REFERENCE,
        "listening_utmos_mean": QWEN3_OMNI_FDB_UTMOS_MEAN_REFERENCE,
        "listening_voice_similarity_mean": QWEN3_OMNI_FDB_VOICE_SIMILARITY_MEAN_REFERENCE,
    },
    calibrated=False,
)
CASCADE_FDB_PRESET = FdbCiPreset(
    strategy=FdbCiStrategy(
        engine_name="cascade",
        gpu_arguments=(),
        num_shards=16,
        should_run_semantic_judge=True,
    ),
    references={
        "v15_respond_on_interruption": CASCADE_FDB_RESPOND_ON_INTERRUPTION_REFERENCE,
        "v15_resume_on_backchannel": CASCADE_FDB_RESUME_ON_BACKCHANNEL_REFERENCE,
        "v15_resume_on_talking_to_other": CASCADE_FDB_RESUME_ON_TALKING_TO_OTHER_REFERENCE,
        "v15_resume_on_background_speech": CASCADE_FDB_RESUME_ON_BACKGROUND_SPEECH_REFERENCE,
        "v15_overlap_handled": CASCADE_FDB_OVERLAP_HANDLED_REFERENCE,
        "v15_grounded_reply": CASCADE_FDB_GROUNDED_REPLY_REFERENCE,
        "v15_stop_latency_mean_s": CASCADE_FDB_STOP_LATENCY_MEAN_S_REFERENCE,
        "v15_response_latency_mean_s": CASCADE_FDB_RESPONSE_LATENCY_MEAN_S_REFERENCE,
        "v10_synthetic_pause_intrusion_rate": CASCADE_FDB_SYNTHETIC_PAUSE_INTRUSION_RATE_REFERENCE,
        "v10_candor_pause_intrusion_rate": CASCADE_FDB_CANDOR_PAUSE_INTRUSION_RATE_REFERENCE,
        "v10_turn_response_rate_2s": CASCADE_FDB_TURN_RESPONSE_RATE_2S_REFERENCE,
        "listening_omission_rate": CASCADE_FDB_OMISSION_RATE_REFERENCE,
        "listening_utmos_mean": CASCADE_FDB_UTMOS_MEAN_REFERENCE,
        "listening_voice_similarity_mean": CASCADE_FDB_VOICE_SIMILARITY_MEAN_REFERENCE,
    },
    calibrated=False,
)

FDB_CI_PRESETS: dict[FdbCiStrategyName, FdbCiPreset] = {
    "full-duplex": MINICPMO_FDB_PRESET,
    "turn-based": QWEN3_OMNI_FDB_PRESET,
    "cascaded": CASCADE_FDB_PRESET,
}


def select_fdb_ci_preset() -> tuple[FdbCiStrategyName, FdbCiPreset]:
    selected = os.environ.get("FDB_CI_STRATEGY", "full-duplex")
    if selected not in FDB_CI_PRESETS:
        allowed = ", ".join(FDB_CI_PRESETS)
        raise ValueError(
            f"Unsupported FDB_CI_STRATEGY={selected!r}; expected one of: {allowed}"
        )
    else:
        return selected, FDB_CI_PRESETS[selected]
