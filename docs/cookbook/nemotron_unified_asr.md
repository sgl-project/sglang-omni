# Nemotron Unified ASR

[Nemotron-Labs-Unified-ASR-0.6B](https://huggingface.co/nvidia/nemotron-labs-unified-asr-0.6b)
is a multilingual RNN-T speech-recognition model with a non-causal
FastConformer encoder; one checkpoint serves offline and streaming
recognition. It runs on the [Nemotron 3.5 ASR](nemotron3_5_asr.md) stage, so
serving, request parameters and the WebSocket session protocol are the same.

## Server Configuration

NVIDIA publishes the model as a NeMo `.nemo` archive. The stage reads the
archive at startup and builds the Transformers `Nemotron3_5AsrForRNNT` model
with a `parakeet_encoder` encoder, without NeMo. A Transformers-format copy of
the checkpoint also works.

```bash
sgl-omni serve \
  --model-path nvidia/nemotron-labs-unified-asr-0.6b \
  --enable-realtime \
  --port 8000
```

`language` accepts the checkpoint's prompt-dictionary keys, such as `en`,
`de-DE` or `auto`. The model emits no locale tags, so responses echo the
requested value. `num_lookahead_tokens` and the encoder CUDA graph options do
not apply.

## Streaming

Streaming follows NeMo's buffered RNN-T inference with 0.56 s chunks, 5.6 s
of left context and 0.32 s of right context. Each hop re-encodes its window
with chunked attention and decodes the chunk's frames with the session's RNN-T
state; the encoder keeps no state between hops. The first text needs 0.88 s
of audio, later hops run every 0.56 s, and EOS decodes the remaining frames.
Streams shorter than 20 ms return an empty final.

On LibriSpeech test-clean in `float32` with `en`, offline transcripts match
NeMo `transcribe()` on 200 of 200 utterances, and streaming matches NeMo's
streaming script on 198, the other two differing by one punctuation mark.
