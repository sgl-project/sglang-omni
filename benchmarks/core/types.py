from __future__ import annotations

from dataclasses import dataclass
from typing import Any

INPUT_PREVIEW_LENGTH = 60


@dataclass(frozen=True)
class NormalizedSample:
    sample_id: str
    input_preview: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkCaseSpec:
    case_id: str
    scenario_id: str
    description: str
    requires_dataset_path: bool
    supports_stream: bool = True
    default_max_output_tokens: int | None = None


@dataclass(frozen=True)
class BenchmarkRunConfig:
    base_url: str
    model: str
    model_profile: str
    case_id: str | None = None
    dataset_path: str | None = None
    output_dir: str = "results/benchmark"
    stream: bool = False
    max_samples: int | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    warmup: int = 1
    max_concurrency: int = 1
    request_rate: float = float("inf")
    save_audio: bool = False
    disable_tqdm: bool = False


@dataclass(frozen=True)
class PreparedRequest:
    request_id: str
    input_preview: str
    payload: dict[str, Any]


@dataclass
class PerRequestResult:
    request_id: str
    input_preview: str
    success: bool = False
    latency_s: float = 0.0
    first_audio_latency_s: float | None = None
    audio_duration_s: float | None = None
    rtf: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    engine_time_s: float | None = None
    tok_per_s: float | None = None
    error: str | None = None
    # Transient field for save-audio support; not serialized.
    audio_bytes: bytes | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.request_id,
            "input_preview": self.input_preview,
            "success": self.success,
            "latency_s": _round_or_none(self.latency_s, 4),
            "first_audio_latency_s": _round_or_none(self.first_audio_latency_s, 4),
            "audio_duration_s": _round_or_none(self.audio_duration_s, 4),
            "rtf": _round_or_none(self.rtf, 4),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "engine_time_s": _round_or_none(self.engine_time_s, 4),
            "tok_per_s": _round_or_none(self.tok_per_s, 1),
            "error": self.error,
        }


@dataclass
class RunSummary:
    model_profile: str
    case: str
    scenario: str
    request_family: str
    completed_requests: int
    failed_requests: int
    run_wall_time_s: float
    latency_mean_s: float | None = None
    latency_median_s: float | None = None
    latency_p95_s: float | None = None
    latency_p99_s: float | None = None
    throughput_qps: float | None = None
    audio_duration_mean_s: float | None = None
    rtf_mean: float | None = None
    rtf_median: float | None = None
    tok_per_s_mean: float | None = None
    tok_per_s_median: float | None = None
    tok_per_s_agg: float | None = None
    prompt_tokens_mean: float | None = None
    prompt_tokens_total: int | None = None
    completion_tokens_mean: float | None = None
    completion_tokens_total: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_profile": self.model_profile,
            "case": self.case,
            "scenario": self.scenario,
            "request_family": self.request_family,
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
            "run_wall_time_s": round(self.run_wall_time_s, 4),
            "latency_mean_s": _round_or_none(self.latency_mean_s, 4),
            "latency_median_s": _round_or_none(self.latency_median_s, 4),
            "latency_p95_s": _round_or_none(self.latency_p95_s, 4),
            "latency_p99_s": _round_or_none(self.latency_p99_s, 4),
            "throughput_qps": _round_or_none(self.throughput_qps, 4),
            "audio_duration_mean_s": _round_or_none(self.audio_duration_mean_s, 4),
            "rtf_mean": _round_or_none(self.rtf_mean, 4),
            "rtf_median": _round_or_none(self.rtf_median, 4),
            "tok_per_s_mean": _round_or_none(self.tok_per_s_mean, 1),
            "tok_per_s_median": _round_or_none(self.tok_per_s_median, 1),
            "tok_per_s_agg": _round_or_none(self.tok_per_s_agg, 1),
            "prompt_tokens_mean": _round_or_none(self.prompt_tokens_mean, 1),
            "prompt_tokens_total": self.prompt_tokens_total,
            "completion_tokens_mean": _round_or_none(self.completion_tokens_mean, 1),
            "completion_tokens_total": self.completion_tokens_total,
        }


@dataclass(frozen=True)
class ResolvedProfileSelection:
    model_profile_name: str
    case_spec: BenchmarkCaseSpec
    request_family: str
    served_model_name: str
    dataset_loader_name: str
    model_adapter_name: str
    model_profile: Any


@dataclass
class BenchmarkResults:
    summary: RunSummary
    config: dict[str, Any]
    per_request: list[PerRequestResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "config": dict(self.config),
            "per_request": [result.to_dict() for result in self.per_request],
        }


def _round_or_none(value: float | None, digits: int) -> float | None:
    if value is None:
        return None
    return round(value, digits)
