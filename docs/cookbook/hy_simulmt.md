# Hy-SimulMT

[Hy-SimulMT-30B-A3B](https://huggingface.co/tencent/Hy-SimulMT-30B-A3B) translates Chinese speech to English text and English speech to Chinese text. This integration supports the ordinary, unquantized checkpoint. It uses the shared ASR scheduler, the Qwen3 audio encoder, a frame-stacking adapter, and SGLang's Hunyuan V3 decoder.

## Start the server

Install SGLang-Omni using the [installation guide](../get_started/installation.md), then run:

```bash
sgl-omni serve \
  --model-path tencent/Hy-SimulMT-30B-A3B \
  --mem-fraction-static 0.75 \
  --enable-realtime \
  --port 8000
```

The BF16 weights occupy approximately 57 GiB before KV cache and runtime allocations. A single H200 can host this configuration. Decode CUDA graphs are enabled and Torch compilation is disabled by default.

## Translate an audio file

For Chinese speech to English text, use the audio translation endpoint:

```bash
curl http://localhost:8000/v1/audio/translations \
  -F file=@chinese.wav \
  -F language=zh \
  -F response_format=json
```

The translation endpoint always targets English. Specify the source language because this checkpoint does not provide language-detection tokens. Individual audio inputs must be at most 30 seconds; longer inputs are rejected instead of silently truncated.

For English speech to Chinese text, use the shared transcription route with the source language:

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@english.wav \
  -F language=en \
  -F response_format=json
```

The shared `/v1/audio/transcriptions` route also works: for this translation-only model, `language=en` produces Chinese translation and `language=zh` produces English translation. It does not produce a source-language transcript. Omitted language defaults to Chinese-to-English.

### Stream the translated text

Set `stream=true` on either audio endpoint to receive `transcript.text.delta` events followed by the authoritative `transcript.text.done`. This streams the output of a completed audio upload.

## Live audio

With `--enable-realtime`, connect to `/v1/realtime?intent=transcription`. The shared ASR WebSocket accepts mono PCM16 at 16 kHz:

1. Send `session.update` with `session.language` set to the source language (`zh` or `en`).
2. Send base64 PCM chunks with `input_audio_buffer.append`.
3. Receive `transcription.segment` translation hypotheses, refreshed every 1.6 seconds by default.
4. Send `input_audio_buffer.commit` to finish a segment or `transcription.done` to finish input.

Each partial request retains earlier audio turns and their translations, including turns that emitted no text. Audio embeddings and prefix tokens use the existing multimodal and radix-cache paths. At a segment boundary, the final hypothesis translates the complete segment to cover speech deferred by earlier turns; it may revise partial translations. The shared runtime splits segments at 30 seconds, after which translation context starts fresh. This initial policy does not reproduce SimulInfer's VAD and recovery heuristics.

Outputs are text. Speech synthesis requires a separate TTS stage. Interleave checkpoints and quantized checkpoints are outside this implementation.

## Reference validation

Use identical 16 kHz waveforms, greedy decoding, and the checkpoint's audio-window boundaries when comparing with Transformers. In the SimulTrain reference at revision `918868feee81c290bb31ebe373d5c3e235a51959`, use `flash_attention_2` for the audio encoder: its SDPA path does not pass the block mask to attention, so clips spanning multiple windows can produce different translations. This integration preserves `n_window_infer=800`, consistent with the checkpoint and the SimulInfer serving encoder.

Validation used checkpoint revision `8461c998403d10a4c30bcb0e55bfb0c981b92774` and 13 requests derived from two speech clips: seven offline inputs covering short audio, an 8-second window boundary, and the 30-second limit, plus three successive streaming turns in each direction. Prompt tokens and BF16 mel features match exactly. Audio-embedding relative L2 error is at most 0.00414. Greedy output token sequences match on 11 of 13 requests; the 8-second and repeated 30-second inputs diverge at near-tied choices with an HF top-two logit gap of 0.125. At those positions, both implementations retain the same top-two token set. Across 15 aligned next-token distributions, maximum HF-to-Omni KL divergence is 0.0193. This establishes basic implementation agreement; it does not establish translation quality on a representative corpus.

Decode CUDA graphs produce identical token sequences to eager Omni execution on all 13 requests. Concurrent Chinese and English realtime sessions produce partial translations and final text matching offline decoding of the same PCM16 audio. The 30-second automatic segment boundary also produces the same final translation.

### Latency check

The following batch-size-one medians use one warmup and three measured calls on H200, BF16, greedy decoding, and disabled radix and multimodal embedding caches. The two full clips have identical generated token sequences in both runtimes. HF timing includes feature extraction, audio encoding, and generation; Omni timing additionally includes local HTTP transport and audio decoding. Model loading and warmup are excluded. HF uses PyTorch 2.9.1, Transformers 5.8.0, encoder FA2, and decoder SDPA; Omni uses PyTorch 2.14.1, SGLang 0.5.22, and FA3.

| Request | HF | Omni eager | Omni CUDA graph | HF / Omni graph |
| --- | ---: | ---: | ---: | ---: |
| Chinese speech, 4.20 seconds, 14 output tokens | 661.9 ms | 897.8 ms | 140.7 ms | 4.70x |
| English speech, 15.05 seconds, 38 output tokens | 1751.2 ms | 2416.1 ms | 269.7 ms | 6.49x |
| Streaming WAIT turn, one terminal token | 82.1 ms | 88.2 ms | 90.1 ms | 0.91x |

The WAIT request ends after prefill, so decode graphs do not accelerate it. These small-sample measurements do not characterize concurrent throughput. To repeat an uncached serving comparison, add `--asr.engine.disable_radix_cache true`; add `--asr.engine.disable_cuda_graph true` for the eager control.
