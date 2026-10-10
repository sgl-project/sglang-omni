# Irodori-TTS v4.1 Small

Irodori is integrated as a model-specific SGLang Omni pipeline, following the model-package structure used by integrations such as AuK. Its DiT, duration predictor, rectified-flow sampler, tokenizer, and DACVAE adapter live in this package and run in the SGLang worker. A separate Irodori server or Python environment is not needed. The model core retains its MIT license and source attribution in LICENSE and UPSTREAM.md.

Irodori predicts continuous latents that require its matching DACVAE codec. Omni loads the DACVAE Python library in the same worker process as the Irodori stage. Install the model-specific optional dependencies into the existing SGLang Omni environment:

```bash
uv pip install -e '.[irodori-tts]'
```

The codec weights are separate from the Irodori model checkpoint and are loaded from the Hugging Face cache on first use. For offline inference, download the codec checkpoint separately and set stages.synthesis.factory.codec_repo to its local path.

Start SGLang Omni with the local checkpoint directory:

```bash
sgl-omni serve --model-path /path/to/Irodori \
  --allowed-local-media-path /path/to/reference-audio
```

Send Japanese text with an optional reference recording and style caption:

```bash
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/path/to/Irodori",
    "input": "こんにちは。音声合成のテストです。",
    "ref_audio": "file:///path/to/reference-audio/voice.wav",
    "instructions": "落ち着いた自然な声",
    "response_format": "wav"
  }' \
  --output irodori.wav
```

Irodori does not require a reference transcript. When no reference audio is provided, Omni supplies an empty speaker condition. The checkpoint tokenizer is loaded from /path/to/Irodori/tokenizer.

Concurrent speech requests are coalesced by the Omni synthesis scheduler. The default batch limit is 16 requests with an 8 ms collection window; variable output lengths are padded with an attention mask, and the scheduler limits the padded generation/reference frame budget per model batch.

The default model precision is BF16 and the default maximum generated duration is 30 seconds. Set stages.synthesis.factory.model_precision=fp32 when comparing output with the official FP32 inference path.
