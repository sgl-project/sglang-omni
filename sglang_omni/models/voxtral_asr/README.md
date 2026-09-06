# Voxtral Realtime ASR for sglang-omni

This package adds `mistralai/Voxtral-Mini-4B-Realtime-2602` (and future
realtime Voxtral checkpoints) to sglang-omni as a single-stage ASR pipeline.

## Status

- [x] Package scaffold (`config.py`, `stages.py`, `request_builders.py`,
      `sglang_model.py`, `model_config.py`)
- [x] Mistral text backbone reuse from `voxtral_tts`
- [x] Causal Whisper encoder (mel -> causal convs -> causal transformer with
      RoPE + sliding window)
- [x] Audio-language adapter and time embedding
- [x] Model registration in `sglang_model_runner.py` and `utils/hf.py`
- [x] Example YAML
- [ ] Streaming token feedback loop (decode tokens fed back as next-frame
      prompt tokens)
- [ ] Streaming encoder KV cache (currently each audio chunk is encoded in one
      full pass; the causal encoder supports incremental encoding but it is
      not wired yet)
- [ ] CUDA graph / torch.compile warmup for the audio encoder
- [ ] TP/PP support for the audio encoder
- [ ] Numerical parity validation against vLLM
- [ ] Benchmark harness

## Architecture

The model follows the vLLM `VoxtralRealtimeGeneration` design:

1. Raw audio waveform -> log-mel spectrogram (`compute_melspec`).
2. Causal 1-D convolutions (`conv1`, `conv2`) -> post-conv frame embeddings.
3. Causal Transformer encoder with RoPE and sliding-window self-attention.
4. Downsample by `downsample_factor` and project through `AudioLanguageAdapter`.
5. Add audio embeddings to text token embeddings, add `TimeEmbedding(delay)`.
6. Run Mistral text decoder with SGLang's `RadixAttention` / CUDA graph paths.

## Running

```bash
export all_proxy=http://127.0.0.1:7890  # if needed for HF downloads
python -m sglang_omni.launcher \
  --config examples/configs/voxtral_asr.yaml
```

The stage factory is `sglang_omni.models.voxtral_asr.stages.create_sglang_voxtral_asr_executor`.

## Known Limitations

- The first version targets **offline/batched ASR**; it attaches the full audio
  to the prompt and encodes it once during prefill.  True streaming (chunked
  audio with token feedback) requires extending `request_builders.py` and the
  scheduler to carry per-request stream state across decode steps.
- The encoder is not yet captured in CUDA graphs; the text decoder benefits
  from SGLang's normal CUDA-graph path.
- The causal encoder runs with plain `F.scaled_dot_product_attention` and a
  sliding-window causal mask.  This is correct for full-pass encoding but does
  not reuse KV state across chunks.
