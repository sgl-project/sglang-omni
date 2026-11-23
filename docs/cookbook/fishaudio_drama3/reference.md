---
orphan: true
---

# Fish Audio Drama 3 Reference

[Back to the Fish Audio Drama 3 quick start](../fishaudio_drama3.md).

The limits and the checkpoint layout that the [Fish Audio Drama 3](../fishaudio_drama3.md) page summarizes under Known limitations and Choose a model.

## Serving Limits

### Reference Audio Length

Each reference given as audio may be at most 60 seconds long. The limit applies to every reference in the request separately, and a longer one fails the request with HTTP 400 before it is encoded:

```text
Drama 3 request: references.1: reference audio lasts 75.3 s, longer than the 60 s limit; shorten it or raise max_reference_audio_seconds
```

`references.N` names the reference by its position in the request's `references` list, counted from 0.

Shorten that reference, or raise the limit when you start the server, for example to 90 seconds:

```bash
sgl-omni serve \
  --model-path <checkpoint> \
  --preprocessing.factory.max_reference_audio_seconds 90 \
  --preprocessing.total_reserve_bytes 9GiB
```

or in a config file:

```yaml
stages:
  preprocessing:
    factory:
      max_reference_audio_seconds: 90
    total_reserve_bytes: 9GiB
```

The GPU memory the preprocessing stage needs grows with the length of the reference it encodes, and the stage is capped at `total_reserve_bytes` (6 GiB by default, sized for the 60-second limit). When you raise the limit, raise the reserve with it, measured on your GPU; a reference that does not fit the reserve fails with an out-of-memory error in the preprocessing stage. The budgets of all stages on the GPU must still fit the card, which the server checks at startup.

### Prompt Length

A prompt holds the tokens of the prompt template, the input text and the reference transcripts, plus one row for every codec frame of its reference audio, about 21.5 rows per second of audio (the codec turns 44.1 kHz audio into frames of 2048 samples). A prompt may have at most 8192 rows; a longer one fails the request with HTTP 400:

```text
Drama 3 request: a prompt of 8193 rows exceeds the maximum allowed length of 8192 rows
```

Change the cap with `--tts_engine.factory.max_prompt_rows`. A prompt and its `max_new_tokens` must also fit the engine's context length.

## Int4 Weight Layout

In [`fishaudio/drama-3-int4`](https://huggingface.co/fishaudio/drama-3-int4), the routed MoE experts and the attention projections (`q/k/v/o_proj` of the full attention layers, `in_proj_qkv`, `in_proj_z` and `out_proj` of the linear attention layers) hold symmetric 4-bit weights with one scale per 128 input channels, and run on GPTQ-Marlin kernels. The shared expert, the router, the linear attention's `in_proj_a/b`, the output head and the codebook-MTP audio decoder stay in bf16. The engine's weights take 20.8 GB of GPU memory instead of 37.0 GB, which leaves more of the engine's share for the KV cache. GPTQ-Marlin needs compute capability 8.0 or newer.
