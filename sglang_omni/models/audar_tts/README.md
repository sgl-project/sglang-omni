# Audar-TTS-V1 Turbo

Install the optional GGUF and codec dependencies:

```bash
pip install -e '.[audar-tts]'
```

For a CUDA build of llama.cpp, install `llama-cpp-python` with the build flags
required by the target CUDA image before installing SGLang Omni.

Start the server with the explicit config because the Turbo Hugging Face repo
contains GGUF weights and no Transformers `config.json`:

```bash
sgl-omni serve --config examples/configs/audar_tts_turbo.yaml \
  --allowed-local-media-path /path/to/references
```

Send one 5-15 second reference clip and its transcript:

```bash
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "audarai/Audar-TTS-V1-Turbo",
    "input": "مرحبا، أهلا وسهلا بكم.",
    "ref_audio": "file:///path/to/references/voice.wav",
    "ref_text": "النص المطابق للمقطع المرجعي.",
    "response_format": "wav"
  }' \
  --output audar.wav
```

The Audar backend infers the output language from `input`; the optional API
`language` field is accepted as metadata but is not consumed by this model.

## Refactor validation

PR [#1090](https://github.com/sgl-project/sglang-omni/pull/1090), stacked on
[#1096](https://github.com/sgl-project/sglang-omni/pull/1096), reduced the
production-equivalent integration from 797 to 619 non-test, non-documentation
lines (22.3%). All 50 paired pre/post PCM WAV outputs were byte-identical. The
Arabic ASR result was 5.43% WER, 1.46% CER, 88.75 BLEU, and 95.57 chrF++; the
paired H100 runs showed performance parity.
