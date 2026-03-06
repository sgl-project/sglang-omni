# FishAudio OpenAudio-S2-Pro

Text-to-speech via the FishQwen3Omni (MoE slow head + 10-codebook fast head) architecture with DAC codec vocoding.

## Quick Start

```bash
# Basic TTS
python examples/run_fishaudio_s2pro_e2e.py \
    --text "Hello, how are you?" \
    --output output.wav

# Voice cloning
python examples/run_fishaudio_s2pro_e2e.py \
    --text "Hello, how are you?" \
    --reference-audio ref.wav --reference-text "Transcript of ref audio." \
    --output output.wav
```
