# Voxtral Realtime ASR

This cookbook describes how to serve `mistralai/Voxtral-Mini-4B-Realtime-2602`
with sglang-omni.

> **Status:** prototype / experimental.  Offline/batched transcription works
> first; chunked realtime streaming is still being wired.

## Launch

```bash
export all_proxy=http://127.0.0.1:7890  # if HuggingFace download needs proxy
sgl-omni serve --config examples/configs/voxtral_asr.yaml --host 0.0.0.0 --port 8000
```

The launcher uses the stage factory
`sglang_omni.models.voxtral_asr.stages.create_sglang_voxtral_asr_executor`.

## Transcribe a single audio file

`/v1/audio/transcriptions` accepts multipart file uploads:

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@/path/to/audio.wav \
  -F model=voxtral
```

Example response:

```json
{"text": " Mary had a little lamb, ...", "usage": {"type": "duration", "seconds": 16}}
```

## Configuration knobs

The stage factory accepts the usual SGLang-Omni AR stage arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `device` | `cuda:0` | GPU device for the stage |
| `dtype` | `bfloat16` | Weight/activation dtype |
| `max_running_requests` | 32 | Max concurrent requests |
| `max_new_tokens` | 4096 | Max text tokens to generate |
| `enable_torch_compile` | `False` | Torch-compile the text backbone |
| `mm_attention_backend` | auto | Attention backend for the encoder path |

## Throughput tuning

1. **Use non-eager CUDA graphs.**  The text decoder is captured automatically;
   the audio encoder is still eager in the prototype.
2. **Set `max_running_requests`** to saturate the GPU without exceeding memory.
3. **Chunk long audio.**  The encoder has a sliding window of 750 pooled tokens
   (~15s); the prototype processes the whole attached audio in one pass, so
   very long inputs should be chunked by the client.
4. **Reduce delay if latency is critical.**  The model default delay is 480ms
   (6 delay tokens).  Lower delays increase quality risk but reduce per-step
   encoder work.

## Next steps

- Implement streaming token feedback so generated text tokens are appended to
  the next audio chunk's prompt.
- Capture CUDA graphs for the causal Whisper encoder.
- Validate numerical parity against the vLLM reference.
