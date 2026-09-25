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
    # note (luojiaxuan): a checkpoint that serves named voices takes the text
    # alone, so the reference fields above go unused and the reference-based
    # similarity stage has nothing to score.
    voice: str | None = None
    voice_clone: bool = True


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
class TtsCiLatencyPoint:
    """One open-loop streaming operating point of the latency stage."""

    request_rate: float
    samples: int
    # note (luojiaxuan): None until the point is calibrated, so the stage
    # prints the value without judging it.
    ttfp_median_max_s: float | None = None
    ttfp_p95_max_s: float | None = None
    c50_min_pct: float | None = None


@dataclass(frozen=True)
class TtsCiLatencyPreset:
    """Streaming first-audio latency points measured against one worker."""

    points: tuple[TtsCiLatencyPoint, ...]
    calibrated: bool = False


@dataclass(frozen=True)
class TtsCiPreset:
    model: TtsCiModelPreset
    thresholds: TtsCiThresholdPreset
    # note (luojiaxuan): only the arms whose first-audio latency is being
    # worked on carry this; the stage skips the others.
    latency: TtsCiLatencyPreset | None = None


# Slack factors applied to P95 reference values to derive CI thresholds.
# Higher-is-better metrics: threshold = P95 * slack_higher.
# Lower-is-better metrics: threshold = P95 * slack_lower.
THRESHOLD_SLACK_HIGHER = 0.75
THRESHOLD_SLACK_LOWER = 1.25


# Higgs thresholds.
HIGGS_VC_WER_MAX_CORPUS = 0.0105
HIGGS_VC_WER_CORPUS_THRESHOLD = apply_wer_slack(HIGGS_VC_WER_MAX_CORPUS)
HIGGS_VC_STREAM_WER_MAX_CORPUS = 0.0104
HIGGS_VC_STREAM_WER_CORPUS_THRESHOLD = apply_wer_slack(HIGGS_VC_STREAM_WER_MAX_CORPUS)
HIGGS_VC_SIMILARITY_MEAN_MIN = 65.95185356140136
HIGGS_VC_UTMOS_MEAN_REFERENCE = 4.1628
HIGGS_VC_UTMOS_MEAN_MIN = apply_mos_slack(HIGGS_VC_UTMOS_MEAN_REFERENCE)

HIGGS_VC_NON_STREAM_P95 = {
    16: {
        "throughput_qps": 19.992,
        "output_tok_per_req_s": 160.7,
        "latency_mean_s": 0.796,
        "rtf_mean": 0.1894,
    }
}

HIGGS_VC_STREAM_P95 = {
    16: {
        "throughput_qps": 20.016,
        "latency_mean_s": 0.796,
        "rtf_mean": 0.1872,
    }
}

HIGGS_VC_NON_STREAM_THRESHOLDS = apply_slack(
    HIGGS_VC_NON_STREAM_P95, THRESHOLD_SLACK_HIGHER, THRESHOLD_SLACK_LOWER
)
HIGGS_VC_STREAM_THRESHOLDS = apply_slack(
    HIGGS_VC_STREAM_P95, THRESHOLD_SLACK_HIGHER, THRESHOLD_SLACK_LOWER
)


# MOSS Local thresholds.
MOSS_VC_WER_MAX_CORPUS = 0.0272
MOSS_VC_WER_CORPUS_THRESHOLD = apply_wer_slack(MOSS_VC_WER_MAX_CORPUS)
MOSS_VC_STREAM_WER_MAX_CORPUS = 0.0265
MOSS_VC_STREAM_WER_CORPUS_THRESHOLD = apply_wer_slack(MOSS_VC_STREAM_WER_MAX_CORPUS)
MOSS_VC_SIMILARITY_MEAN_MIN = 64.33411296844483
MOSS_VC_UTMOS_MEAN_REFERENCE = 3.9521
MOSS_VC_UTMOS_MEAN_MIN = apply_mos_slack(MOSS_VC_UTMOS_MEAN_REFERENCE)

MOSS_VC_NON_STREAM_P95 = {
    16: {
        "throughput_qps": 20.463,
        "output_tok_per_req_s": 90.3,
        "latency_mean_s": 0.777,
        "rtf_mean": 0.1795,
    }
}

MOSS_VC_STREAM_P95 = {
    16: {
        "throughput_qps": 16.778,
        "latency_mean_s": 0.949,
        "rtf_mean": 0.2176,
    }
}

MOSS_VC_NON_STREAM_THRESHOLDS = apply_slack(
    MOSS_VC_NON_STREAM_P95, THRESHOLD_SLACK_HIGHER, THRESHOLD_SLACK_LOWER
)
MOSS_VC_STREAM_THRESHOLDS = apply_slack(
    MOSS_VC_STREAM_P95, THRESHOLD_SLACK_HIGHER, THRESHOLD_SLACK_LOWER
)

# Qwen3-TTS 1.7B. This is the variant the community deploys, and it is gated as
# a single instance: on one H100 the tuned single instance beats the same-card
# MPS-DP2 pool on peak throughput and holds a several-fold better first-audio
# latency, so colocation is no longer the recommended topology for it.
#
# Note: (wenyao) recalibrated on the CI host, lane 2,3 pinned cpuset
# (16-31,80-95), worst-of-5 clean rounds with destructive rejection
# (run .tune-runs/20260830T024753Z_tts_combined). Raw pre-slack references
# only; the CI slack calculation is unchanged.
QWEN3_TTS_VC_WER_MAX_CORPUS = 0.0114
QWEN3_TTS_VC_WER_CORPUS_THRESHOLD = apply_wer_slack(QWEN3_TTS_VC_WER_MAX_CORPUS)
QWEN3_TTS_VC_STREAM_WER_MAX_CORPUS = 0.0109
QWEN3_TTS_VC_STREAM_WER_CORPUS_THRESHOLD = apply_wer_slack(
    QWEN3_TTS_VC_STREAM_WER_MAX_CORPUS
)
QWEN3_TTS_VC_SIMILARITY_MEAN_MIN = 69.13817592620849
QWEN3_TTS_VC_UTMOS_MEAN_REFERENCE = 4.193
QWEN3_TTS_VC_UTMOS_MEAN_MIN = apply_mos_slack(QWEN3_TTS_VC_UTMOS_MEAN_REFERENCE)

QWEN3_TTS_VC_NON_STREAM_P95 = {
    16: {
        "throughput_qps": 21.083,
        "output_tok_per_req_s": 88.4,
        "latency_mean_s": 0.755,
        "rtf_mean": 0.1859,
    }
}

QWEN3_TTS_VC_STREAM_P95 = {
    16: {
        "throughput_qps": 19.526,
        "latency_mean_s": 0.815,
        "rtf_mean": 0.1992,
    }
}

# note (luojiaxuan): docs/cookbook/qwen3_tts.md, 1.7B CustomVoice, Ryan/English,
# concurrency 16 on one H200; single runs, so these are references to print
# next to, not worst-of-N observations from the CI host.
QWEN3_TTS_CUSTOM_VOICE_NON_STREAM_REFERENCE = {
    16: {
        "throughput_qps": 14.788,
        "latency_mean_s": 1.075,
        "rtf_mean": 0.2335,
    }
}
QWEN3_TTS_CUSTOM_VOICE_STREAM_REFERENCE = {
    16: {
        "throughput_qps": 10.098,
        "latency_mean_s": 1.573,
        "rtf_mean": 0.3380,
    }
}
QWEN3_TTS_CUSTOM_VOICE_NON_STREAM_THRESHOLDS = apply_slack(
    QWEN3_TTS_CUSTOM_VOICE_NON_STREAM_REFERENCE,
    THRESHOLD_SLACK_HIGHER,
    THRESHOLD_SLACK_LOWER,
)
QWEN3_TTS_CUSTOM_VOICE_STREAM_THRESHOLDS = apply_slack(
    QWEN3_TTS_CUSTOM_VOICE_STREAM_REFERENCE,
    THRESHOLD_SLACK_HIGHER,
    THRESHOLD_SLACK_LOWER,
)
QWEN3_TTS_CUSTOM_VOICE_WER_CORPUS_THRESHOLD = apply_wer_slack(0.01608)
QWEN3_TTS_CUSTOM_VOICE_STREAM_WER_CORPUS_THRESHOLD = apply_wer_slack(0.02085)
QWEN3_TTS_CUSTOM_VOICE_UTMOS_MEAN_MIN = apply_mos_slack(4.1723)

QWEN3_TTS_VC_NON_STREAM_THRESHOLDS = apply_slack(
    QWEN3_TTS_VC_NON_STREAM_P95, THRESHOLD_SLACK_HIGHER, THRESHOLD_SLACK_LOWER
)
QWEN3_TTS_VC_STREAM_THRESHOLDS = apply_slack(
    QWEN3_TTS_VC_STREAM_P95, THRESHOLD_SLACK_HIGHER, THRESHOLD_SLACK_LOWER
)

# Fun-CosyVoice3 0.5B.
#
# note(ratish): placeholder references until the H100 calibration fills them,
# so the preset keeps gate_thresholds off until then.
COSYVOICE3_VC_WER_MAX_CORPUS = 1.0
COSYVOICE3_VC_WER_CORPUS_THRESHOLD = apply_wer_slack(COSYVOICE3_VC_WER_MAX_CORPUS)
COSYVOICE3_VC_STREAM_WER_MAX_CORPUS = 1.0
COSYVOICE3_VC_STREAM_WER_CORPUS_THRESHOLD = apply_wer_slack(
    COSYVOICE3_VC_STREAM_WER_MAX_CORPUS
)
COSYVOICE3_VC_SIMILARITY_MEAN_MIN = 1.0
COSYVOICE3_VC_UTMOS_MEAN_REFERENCE = 1.0
COSYVOICE3_VC_UTMOS_MEAN_MIN = apply_mos_slack(COSYVOICE3_VC_UTMOS_MEAN_REFERENCE)

COSYVOICE3_VC_NON_STREAM_P95 = {
    16: {
        "throughput_qps": 1.0,
        "output_tok_per_req_s": 1.0,
        "latency_mean_s": 1.0,
        "rtf_mean": 1.0,
    }
}

COSYVOICE3_VC_STREAM_P95 = {
    16: {
        "throughput_qps": 1.0,
        "latency_mean_s": 1.0,
        "rtf_mean": 1.0,
    }
}

COSYVOICE3_VC_NON_STREAM_THRESHOLDS = apply_slack(
    COSYVOICE3_VC_NON_STREAM_P95, THRESHOLD_SLACK_HIGHER, THRESHOLD_SLACK_LOWER
)
COSYVOICE3_VC_STREAM_THRESHOLDS = apply_slack(
    COSYVOICE3_VC_STREAM_P95, THRESHOLD_SLACK_HIGHER, THRESHOLD_SLACK_LOWER
)


# note (luojiaxuan): 1 rps is the idle first-chunk path, 20 rps the loaded one;
# the loaded point is the full EN corpus, so its p95 and its continuity rate
# have support and only there are they gated. Each arm has its own references
# because a cloned voice encodes the reference audio before the first chunk and
# a named voice does not. First playable is timed from each request's planned
# arrival. These are raw worst-of-five references; the gates below apply the
# slack once.
QWEN3_TTS_LATENCY_VC_R1_TTFP_MEDIAN_REF_S = 0.0596
QWEN3_TTS_LATENCY_VC_R20_TTFP_MEDIAN_REF_S = 0.1221
QWEN3_TTS_LATENCY_VC_R20_TTFP_P95_REF_S = 0.3082
QWEN3_TTS_LATENCY_VC_R20_C50_REF_PCT = 95.0
QWEN3_TTS_LATENCY_CUSTOM_VOICE_R1_TTFP_MEDIAN_REF_S = 0.0223
QWEN3_TTS_LATENCY_CUSTOM_VOICE_R20_TTFP_MEDIAN_REF_S = 0.0353
QWEN3_TTS_LATENCY_CUSTOM_VOICE_R20_TTFP_P95_REF_S = 0.0590
QWEN3_TTS_LATENCY_CUSTOM_VOICE_R20_C50_REF_PCT = 95.0


def latency_gate(reference_s: float) -> float:
    return round(reference_s * THRESHOLD_SLACK_LOWER, 4)


def continuity_gate(reference_pct: float) -> float:
    return round(reference_pct * THRESHOLD_SLACK_HIGHER, 2)


QWEN3_TTS_VC_LATENCY_GATES = TtsCiLatencyPreset(
    points=(
        TtsCiLatencyPoint(
            request_rate=1.0,
            samples=60,
            ttfp_median_max_s=latency_gate(QWEN3_TTS_LATENCY_VC_R1_TTFP_MEDIAN_REF_S),
        ),
        TtsCiLatencyPoint(
            request_rate=20.0,
            samples=1088,
            ttfp_median_max_s=latency_gate(QWEN3_TTS_LATENCY_VC_R20_TTFP_MEDIAN_REF_S),
            ttfp_p95_max_s=latency_gate(QWEN3_TTS_LATENCY_VC_R20_TTFP_P95_REF_S),
            c50_min_pct=continuity_gate(QWEN3_TTS_LATENCY_VC_R20_C50_REF_PCT),
        ),
    ),
    calibrated=True,
)
QWEN3_TTS_CUSTOM_VOICE_LATENCY_GATES = TtsCiLatencyPreset(
    points=(
        TtsCiLatencyPoint(
            request_rate=1.0,
            samples=60,
            ttfp_median_max_s=latency_gate(
                QWEN3_TTS_LATENCY_CUSTOM_VOICE_R1_TTFP_MEDIAN_REF_S
            ),
        ),
        TtsCiLatencyPoint(
            request_rate=20.0,
            samples=1088,
            ttfp_median_max_s=latency_gate(
                QWEN3_TTS_LATENCY_CUSTOM_VOICE_R20_TTFP_MEDIAN_REF_S
            ),
            ttfp_p95_max_s=latency_gate(
                QWEN3_TTS_LATENCY_CUSTOM_VOICE_R20_TTFP_P95_REF_S
            ),
            c50_min_pct=continuity_gate(QWEN3_TTS_LATENCY_CUSTOM_VOICE_R20_C50_REF_PCT),
        ),
    ),
    calibrated=True,
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
            # Note: (Jiaxin Deng) the shipped defaults colocate every stage in one
            # process, which is what kept this variant behind, so CI splits the
            # vocoder out and measures the tuned point.
            worker_extra_args=(
                "--vocoder.process vocoder "
                "--tts_engine.gpu_memory_fraction 0.85 "
                "--vocoder.gpu_memory_fraction 0.10"
            ),
            # note (luojiaxuan): a cold Inductor cache compiles the vocoder steady
            # shapes at startup, which takes two workers past five minutes.
            startup_timeout=900,
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
        latency=QWEN3_TTS_VC_LATENCY_GATES,
    ),
    "qwen3-tts-custom-voice": TtsCiPreset(
        model=TtsCiModelPreset(
            model_path="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            voice="Ryan",
            voice_clone=False,
            # note (luojiaxuan): same tuned point as the Base arm, so the two
            # differ only in the checkpoint and the request shape.
            worker_extra_args=(
                "--tts_engine.engine.max_running_requests 64 "
                "--tts_engine.engine.cuda_graph_max_bs 64 "
                "--tts_engine.engine.torch_compile_max_bs 64 "
                "--vocoder.process vocoder "
                "--tts_engine.gpu_memory_fraction 0.85 "
                "--vocoder.gpu_memory_fraction 0.10"
            ),
            # note (luojiaxuan): a cold Inductor cache compiles the vocoder steady
            # shapes at startup, which takes two workers past five minutes.
            startup_timeout=900,
            gate_thresholds=False,
        ),
        # note (luojiaxuan): printed next to the stage results; this arm gates
        # nothing until it is calibrated on the CI host, which a contract test
        # enforces. The similarity stage skips named voices, so that field is
        # never read here.
        thresholds=TtsCiThresholdPreset(
            non_stream_speed=QWEN3_TTS_CUSTOM_VOICE_NON_STREAM_THRESHOLDS,
            stream_speed=QWEN3_TTS_CUSTOM_VOICE_STREAM_THRESHOLDS,
            wer_corpus=QWEN3_TTS_CUSTOM_VOICE_WER_CORPUS_THRESHOLD,
            stream_wer_corpus=QWEN3_TTS_CUSTOM_VOICE_STREAM_WER_CORPUS_THRESHOLD,
            similarity_mean_min=0.0,
            utmos_mean_min=QWEN3_TTS_CUSTOM_VOICE_UTMOS_MEAN_MIN,
            calibrated=False,
        ),
        latency=QWEN3_TTS_CUSTOM_VOICE_LATENCY_GATES,
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
    "cosyvoice3": TtsCiPreset(
        model=TtsCiModelPreset(
            model_path="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
            gate_thresholds=False,
        ),
        thresholds=TtsCiThresholdPreset(
            non_stream_speed=COSYVOICE3_VC_NON_STREAM_THRESHOLDS,
            stream_speed=COSYVOICE3_VC_STREAM_THRESHOLDS,
            wer_corpus=COSYVOICE3_VC_WER_CORPUS_THRESHOLD,
            stream_wer_corpus=COSYVOICE3_VC_STREAM_WER_CORPUS_THRESHOLD,
            similarity_mean_min=COSYVOICE3_VC_SIMILARITY_MEAN_MIN,
            utmos_mean_min=COSYVOICE3_VC_UTMOS_MEAN_MIN,
            calibrated=False,
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
