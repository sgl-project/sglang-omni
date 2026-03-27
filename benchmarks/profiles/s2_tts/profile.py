from __future__ import annotations

from benchmarks.adapters.model.s2_tts import S2TTSModelAdapter
from benchmarks.core.types import BenchmarkCaseSpec
from benchmarks.datasets.seed_tts_eval import SeedTTSEvalLoader
from benchmarks.profiles.base import BenchmarkProfile
from benchmarks.profiles.registry import register_profile


@register_profile
class S2TTSBenchmarkProfile(BenchmarkProfile):
    profile_name = "s2_tts"
    aliases = ("s2-pro", "fishaudio/s2-pro")
    request_family = "speech_http"
    default_case = "voice-cloning"

    @property
    def cases(self) -> dict[str, BenchmarkCaseSpec]:
        return {
            "voice-cloning": BenchmarkCaseSpec(
                case_id="voice-cloning",
                scenario_id="text_to_speech",
                description="TTS voice cloning with reference audio",
                requires_dataset_path=True,
                default_max_output_tokens=2048,
            ),
            "plain-tts": BenchmarkCaseSpec(
                case_id="plain-tts",
                scenario_id="text_to_speech",
                description="Plain TTS without reference audio",
                requires_dataset_path=True,
                default_max_output_tokens=2048,
            ),
        }

    @property
    def dataset_loader(self) -> SeedTTSEvalLoader:
        return SeedTTSEvalLoader()

    def build_model_adapter(self) -> S2TTSModelAdapter:
        return S2TTSModelAdapter()
