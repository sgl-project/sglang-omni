"""Adapt timed benchmark results to Restage's joint SLO evaluation."""

from benchmarks.benchmarker.data import RequestResult
from sglang_omni.restage.evaluation import Observation


def to_observation(result: RequestResult, *, quality_pass: bool | None) -> Observation:
    """Use measured monotonic timestamps, without inferring quality success."""
    if result.scheduled_s is None:
        raise ValueError("Benchmark result has no scheduled arrival timestamp")
    return Observation(
        request_id=result.request_id,
        scheduled_s=result.scheduled_s,
        completed_s=result.completed_s,
        success=result.is_success,
        quality_pass=quality_pass,
        first_output_s=result.first_audio_s,
        audio_duration_s=result.audio_duration_s,
        max_playback_underrun_s=result.max_playback_underrun_s,
    )
