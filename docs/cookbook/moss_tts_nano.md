# MOSS-TTS-Nano

[MOSS-TTS-Nano-100M](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M) is a 100M-parameter multilingual text-to-speech model from MOSI.AI and the OpenMOSS team. It generates native 48 kHz stereo audio with MOSS-Audio-Tokenizer-Nano and supports reference-free synthesis and zero-shot voice cloning.

SGLang-Omni runs Nano through a CPU-first single-stage pipeline. This preserves the checkpoint's global/local autoregressive generation and codec implementation while exposing the standard `/v1/audio/speech` API. The initial integration returns complete audio responses; streaming is not yet supported.

## Launch

```bash
sgl-omni serve \
  --model-path OpenMOSS-Team/MOSS-TTS-Nano-100M \
  --config examples/configs/moss_tts_nano.yaml
```

The built-in config uses CPU and float32 weights. To run on CUDA, override the stage device and dtype:

```bash
sgl-omni serve \
  --model-path OpenMOSS-Team/MOSS-TTS-Nano-100M \
  --config examples/configs/moss_tts_nano.yaml \
  --tts.gpu 0 \
  --tts.factory.device cuda \
  --tts.factory.dtype bfloat16
```

## Synthesize speech

Reference-free synthesis:

```bash
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "OpenMOSS-Team/MOSS-TTS-Nano-100M",
    "input": "MOSS TTS Nano is running through SGLang Omni.",
    "response_format": "wav"
  }' \
  --output nano.wav
```

Voice cloning from a local reference file:

```bash
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "OpenMOSS-Team/MOSS-TTS-Nano-100M",
    "input": "This sentence uses the reference voice.",
    "ref_audio": "file:///path/to/reference.wav",
    "response_format": "wav"
  }' \
  --output cloned.wav
```

Start the server with `--allowed-local-media-path /path/to` when using local reference files. A reference transcript is optional: omitting it selects Nano's voice-clone mode, while providing `ref_text` selects continuation mode.
