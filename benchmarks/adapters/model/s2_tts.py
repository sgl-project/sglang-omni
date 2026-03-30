from __future__ import annotations

from benchmarks.core.types import BenchmarkCaseSpec, BenchmarkRunConfig, NormalizedSample, PreparedRequest

from .base import ModelAdapter


class S2TTSModelAdapter(ModelAdapter):
    name = "s2_tts_adapter"

    def build_request(
        self,
        *,
        sample: NormalizedSample,
        case_spec: BenchmarkCaseSpec,
        run_config: BenchmarkRunConfig,
        served_model_name: str,
    ) -> PreparedRequest:
        references = sample.payload.get("references") or []
        payload: dict[str, object] = {
            "model": served_model_name,
            "input": sample.payload["text"],
            "response_format": "wav",
            "stream": run_config.stream,
        }
        max_output_tokens = (
            run_config.max_output_tokens or case_spec.default_max_output_tokens
        )
        if max_output_tokens is not None:
            payload["max_new_tokens"] = max_output_tokens
        if run_config.temperature is not None:
            payload["temperature"] = run_config.temperature
        if run_config.top_p is not None:
            payload["top_p"] = run_config.top_p
        if run_config.top_k is not None:
            payload["top_k"] = run_config.top_k
        if run_config.repetition_penalty is not None:
            payload["repetition_penalty"] = run_config.repetition_penalty

        if case_spec.case_id == "voice-cloning" and references:
            payload["references"] = references

        return PreparedRequest(
            request_id=sample.sample_id,
            input_preview=sample.input_preview,
            payload=payload,
        )

