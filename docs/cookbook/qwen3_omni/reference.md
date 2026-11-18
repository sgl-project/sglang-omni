# Qwen3-Omni Reference

[Back to the Qwen3-Omni quick start](../qwen3_omni.md).

## Compatibility Matrix

Colocated topology requires `--config examples/configs/qwen3_omni_colocated_h20.yaml` (or `qwen3_omni_colocated_h200.yaml` on H200) to set per-stage GPU memory budgets.

All configurations listed below are supported.

| Mode | Topology | Thinker TP | Precision |
|---|---|---|---|
| Thinker-only | — | — | BF16 |
| Thinker-only | — | — | FP8 |
| Thinker-only | — | — | AutoRound INT4 |
| Thinker-Talker | Disaggregated | TP=1 | BF16 |
| Thinker-Talker | Disaggregated | TP=1 | FP8 |
| Thinker-Talker | Disaggregated | TP=1 | AutoRound INT4 thinker + BF16 talker/code2wav |
| Thinker-Talker | Disaggregated | TP=2 | BF16 |
| Thinker-Talker | Disaggregated | TP=2 | FP8 |
| Thinker-Talker | Disaggregated | TP=2 | AutoRound INT4 thinker + BF16 talker/code2wav |
| Thinker-Talker | Colocated | TP=1 | BF16 |
| Thinker-Talker | Colocated | TP=1 | FP8 |
| Thinker-Talker | Colocated | TP=1 | AutoRound INT4 thinker + BF16 talker/code2wav |

## Input and Output Modalities

Qwen3-Omni accepts text, image, audio, and video input in both text-only and speech deployments. Use the [shared request examples](../../basic_usage/omni.md#general-usage) for media inputs and text-and-speech replies, and [Request Parameters](../../basic_usage/omni.md#request-parameters) for frame sampling and pixel budgets. To return audio, launch the speech pipeline. Realtime audio negotiation has a separate contract described in [Realtime Sessions](platforms.md#realtime-sessions).

With `use_audio_in_video: true`, requests where all videos lack audio use video-only processing; mixing videos with and without audio is rejected. Audio decoding errors are reported.

## Sampling Parameters

The table lists Qwen3-Omni's thinker and talker defaults. Common field definitions, token-budget aliases, and per-stage overrides are documented in [Request Parameters](../../basic_usage/omni.md#request-parameters). When `modalities` includes `"audio"`, talker-specific parameters control speech generation.

| Parameter | Type | Default | Applies to |
|---|---|---|---|
| `temperature` | float | `0.0` (greedy) | Thinker |
| `top_p` | float | `1.0` | Thinker |
| `top_k` | int | `-1` | Thinker |
| `min_p` | float | `0.0` | Thinker |
| `repetition_penalty` | float | `1.0` | Thinker |
| `max_tokens` | int | `2048` | Thinker |
| `talker_temperature` | float | `0.9` | Talker (audio output only) |
| `talker_top_p` | float | `1.0` | Talker (audio output only) |
| `talker_top_k` | int | `50` | Talker (audio output only) |
| `talker_repetition_penalty` | float | `1.05` | Talker (audio output only) |
| `talker_max_new_tokens` | int | `4096` | Talker (audio output only) |

For HTTP and SSE errors, see [Known Limitations](../../basic_usage/omni.md#known-limitations).

## Known Limitations

- **Multiple videos must use the same sampled frame rate.** The processor uses one rate for video timestamps; different rates return an invalid-input error.
- **Colocated topology does not support `--thinker.tp_size 2`.** Startup raises a `ValueError`; use disaggregated topology for TP=2.
- **Context limits apply to prompt and output.** The preprocessor rejects requests when prompt tokens meet or exceed `max_seq_len`, or when prompt tokens plus `max_new_tokens` meet or exceed it. Without both `--preprocessing.factory.max_seq_len` and `--thinker.factory.max_seq_len`, the limit is the thinker's checkpoint context (65,536 tokens for Instruct); setting both lower caps it. The talker uses its checkpoint limit unless `--talker_ar.factory.max_seq_len` sets it. A long spoken prompt may leave too little room for `talker_max_new_tokens`, so audio can end before the text.
