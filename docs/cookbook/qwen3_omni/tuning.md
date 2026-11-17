# Qwen3-Omni Tuning

[Back to the Qwen3-Omni quick start](../qwen3_omni.md).

## Server Readiness

Before it reports ready, the server sends itself a chat request with a small image, one second of audio and a short instruction (text and audio output in speech mode, text output in text-only mode, at most 8 new tokens from each stage that generates tokens), once per replica of its most replicated process, one after another, so that every replica serves one. These requests pass through every stage to build the kernels for that request path. Inputs of other shapes, such as video or a large concurrent batch, can still build a kernel variant on first use. Health status, timeout, and the skip option are documented in [Server Readiness](../../basic_usage/omni.md#server-readiness).

## Memory and Concurrency

For automatic memory sizing, global `--mem-fraction-static`, and per-stage override syntax, see [Memory Budgets](../../basic_usage/omni.md#memory-budgets). Qwen3-Omni's AR stages are `thinker` and `talker_ar`; use the [colocated hardware configs](platforms.md#launch-a-speech-server) when they share one GPU.

The speech server launcher exposes the same per-stage controls:

```bash
python examples/run_omni.py qwen3-speech-server \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --gpu-thinker 0 \
  --gpu-talker 1 \
  --gpu-code-predictor 1 \
  --gpu-code2wav 0 \
  --port 8008 \
  --thinker-mem-fraction-static 0.88 \
  --talker-mem-fraction-static 0.88
```

The thinker admits up to 64 running requests by default. Use the thinker-specific flag to lower or raise that limit in either text-only or speech mode:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --thinker.engine.max_running_requests 16
```

To configure the thinker through a pipeline YAML file instead, set the same path under the stage entry:

```yaml
stages:
  thinker:
    engine:
      max_running_requests: 16
```

## Codec Coalescing and First-Audio Latency

The speech pipeline sets `codec_coalesce_frames=10`, `codec_coalesce_early_frames=10`, and `codec_coalesce_first_frames=0` under `stages.talker_ar.factory`. The first 10 codec frames are sent individually; later frames are coalesced into groups of 10. Omitting a YAML override keeps these pipeline defaults; set `codec_coalesce_early_frames=0` explicitly to disable the early prefix. This aligns with the default Code2Wav 10-frame threshold: the first three windows contain 10, 20, and 30 frames, and subsequent full windows contain 35 frames including left context. These shapes can use the captured windows when CUDA Graph is enabled. Each window decodes one 10-frame chunk past its context, so an early prefix of 12 keeps these window lengths and its two extra frames wait for the next window.

With `initial_codec_chunk_frames=2` and `stream_chunk_size=10`, explicitly set `codec_coalesce_early_frames=12` to make the first two windows eligible at generated frames 2 and 12. Uniform groups of 10 (`early_frames=0`, `first_frames=0`) instead publish the first group at step 11: the sender retains the newest row until the next step can exclude EOS, or the request finishes. For a request that continues past step 10, first-window input readiness therefore moves from step 2 to step 11, adding nine Talker decode intervals. If the interval is approximately `d` ms, the added input wait is approximately `9d` ms. With the default 10-frame first window, readiness instead moves from step 10 to step 11. The default 10-frame early prefix preserves readiness at step 10; the next two windows become ready at steps 21 and 31. The extra step after the prefix retains the newest row for EOS detection, compared with steps 20 and 30 without coalescing.

This is an input-readiness estimate, not a measured end-to-end TTFA delta or nine frames of audio playback time. Actual TTFA also depends on transport, queueing, and vocoder execution; the overall coalescing benchmark does not isolate the early-prefix setting.

## Code2Wav Execution

Exact-shape CUDA Graph replay is enabled by default for Qwen3-Omni Code2Wav. The default stage config supplies a 2% typed GPU memory budget; colocated example configs override it with their hardware-specific budget.

To disable replay, set it on the stage in the YAML config:

```yaml
stages:
  code2wav:
    factory:
      enable_cuda_graph: false
```

When replay is enabled, a custom Code2Wav stage must define `gpu_memory_fraction`; startup rejects a missing typed budget before loading the model.

The feature derives the exact `B=1` threshold windows from `stream_chunk_size` and `left_context_size`; the defaults capture `T{10,20,30,35}`. It also captures, best effort, the final window of every stream length: the 1 to `stream_chunk_size - 1` frames left after a stream's last threshold window plus the context it holds. With the threshold windows this covers every length from `T1` to `T35` at the defaults. A final window that capture skips runs eagerly, as does any other shape. Capture-time incompatibilities also fall back to eager execution.

The SnakeBeta activation of the Code2Wav decoder runs as one fused Triton kernel by default on CUDA devices, shared with the Qwen3-TTS vocoder. The kernel keeps the checkpoint parameters and every BF16 rounding step of the eager activation, so its output is bitwise identical, and it is compiled before graph capture, with or without CUDA Graph replay. Inputs outside its contract (a dtype other than BF16, a non-contiguous layout, or rows longer than 65535 x 1024 samples) run the eager arithmetic, and if the kernel cannot be compiled the eager modules stay installed. To keep the eager activation:

```yaml
stages:
  code2wav:
    factory:
      fused_snake_activation: false
```

Code2Wav takes every queued codec chunk before it decodes, then decodes the ready windows of one length in one replay, up to `max_replay_rows` (default 8) under `stages.code2wav.factory`, each row count a captured graph. A request alone still gets one window per replay; under load, the windows that several requests complete on the same talker step share one. A row of a shared replay can differ from the same window decoded alone in its last bits, since the convolution kernels depend on the batch.

Output overlap is also enabled by default on CUDA devices: each threshold window's waveform readback runs as an asynchronous device-to-host copy into a pinned staging buffer and is materialized while the GPU computes the next window, and the codec EOS check runs once per window instead of once per frame. The first window of every request stays synchronous, so time-to-first-audio is unchanged. For successfully completed requests, audio bytes and message boundaries are identical to the synchronous path. A request that fails or is aborted can drop an un-emitted pending window while its in-flight staging buffer is retired. To disable it:

```yaml
stages:
  code2wav:
    factory:
      enable_output_overlap: false
```

## Speech Stage Placement

At concurrency 8, the talker is the heaviest speech stage: it holds its GPU at about 86% median utilization. Give it a GPU of its own. The code2wav vocoder is light by comparison (9–13% median utilization) and shares the thinker's GPU by default. That default holds only when the thinker stays on its own default GPU — a `--thinker.gpu` override moves the thinker alone, not code2wav, so pass `--code2wav.gpu` explicitly too if you relocate the thinker.

This is the default topology for `sgl-omni serve` without GPU overrides: thinker alone, talker alone, code2wav on the thinker's GPU. When code2wav shares the thinker's GPU, the thinker's auto-sized KV pool shrinks to make room for it — about 4.3 GiB smaller, measured on H200, since the vocoder itself needs roughly 1.4–1.6 GiB. That adjustment only happens for the auto-sized budget: an explicitly pinned `--thinker.engine.mem_fraction_static` gets no automatic carve-out, so a tightly pinned fraction should leave headroom for code2wav.

See [speech stage placement measurements](benchmarks.md#speech-stage-placement) for the concurrency and single-stream trade-offs.

## Speculative Decoding

The thinker can verify tokens proposed by SpeechSpec's DFlash draft [`yuekai/qwen3_omni_30b_a3b_instruct_dflash_block8`](https://huggingface.co/yuekai/qwen3_omni_30b_a3b_instruct_dflash_block8) or EAGLE3 draft [`yuekai/qwen3_omni_30b_a3b_instruct_eagle3_audio`](https://huggingface.co/yuekai/qwen3_omni_30b_a3b_instruct_eagle3_audio) on a text-only server.

```bash
# DFlash: one 8-token block per step
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --text-only \
  --thinker.engine.speculative_algorithm DFLASH \
  --thinker.engine.speculative_draft_model_path yuekai/qwen3_omni_30b_a3b_instruct_dflash_block8 \
  --thinker.engine.speculative_num_draft_tokens 8 \
  --port 8000

# EAGLE3: a 3-step chain
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --text-only \
  --thinker.engine.speculative_algorithm EAGLE3 \
  --thinker.engine.speculative_draft_model_path yuekai/qwen3_omni_30b_a3b_instruct_eagle3_audio \
  --thinker.engine.speculative_num_steps 3 \
  --thinker.engine.speculative_eagle_topk 1 \
  --thinker.engine.speculative_num_draft_tokens 4 \
  --port 8000
```

- **Acceptance length.** Set `return_spec_tokens_details: true` to receive `sglext.spec_tokens_details`: `spec_accept_length` (output tokens per verify, including the prefill-sampled token, so it can exceed scheduler `accept len`), `spec_accept_rate`, draft counts, and a per-verify histogram. Streaming sends it in a chunk with empty `choices` before `[DONE]`. `spec_accept_rate` and `spec_num_proposed_drafts` are omitted when no drafts were proposed or draft width is adaptive. Only thinker drafts are reported; per-stage reporting can follow when Talker supports drafts.
- **Context.** These drafts have an 8,192-token context; the prompt, output budget, and speculative reservations must fit. Set `--preprocessing.factory.max_seq_len 8192` to reject oversized prompts early.
- **Sampling.** Use `repetition_penalty=1`, `frequency_penalty=0`, `presence_penalty=0`, and `min_p=0`; disable logprobs and `return_hidden_states`. For `/generate`, set `return_logprob: false`.
- **Scheduling.** Speculative decoding is synchronous by default. Add `--thinker.factory.enable_async_decode true` to overlap work. With `--thinker.factory.async_decode_min_batch_size` set above 1, a decode batch smaller than it, or a prefill after which fewer requests than it are decoding, has its result processed before the next batch is chosen; other batches overlap. Mixed prefill/decode batches (`enable_mixed_chunk=true`), sessions, and weight updates are unsupported with a draft.
- **Overlap trade-offs.** Overlap can improve decode throughput, but prefill-heavy workloads or short outputs may have higher TTFT and lower throughput.
- **Prefill boundary.** Set `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP=1` before launch to process the previous prefill batch's result before launching the next consecutive prefill batch. This can lower TTFT at a throughput cost; decode rounds still follow `async_decode_min_batch_size`.
- **Execution failure.** A failure during speculative overlap stops the pipeline and fails pending requests.
- **Speech output.** Drafts do not support speech generation; use `--text-only`.

Compare modes with matching workloads and prefix-cache conditions; measure TTFT, throughput, and accuracy for your workload.
