# SPDX-License-Identifier: Apache-2.0
"""Shared dataset and transcription defaults for speech benchmarks."""

SEEDTTS_DATASET_ID = "zhaochenyang20/seed-tts-eval-arrow"
SEEDTTS_DATASET_REVISION = "27f4c1adee83b5b29b7c4b375f6b976324bda308"
QWEN3_ASR_MODEL_PATH = "Qwen/Qwen3-ASR-1.7B"
# note (aaron): ASR transcription fan-out for WER, not TTS generation concurrency.
DEFAULT_ASR_TRANSCRIBE_CONCURRENCY = 32
