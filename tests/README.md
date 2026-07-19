## Folder Structure
```text
tests/
├── README.md
├── __init__.py
├── utils.py
├── data/
├── test_model/
│   ├── conftest.py
│   ├── test_rl_distributed_weight_update.py
│   ├── test_qwen3_omni_*_ci.py
│   ├── test_qwen3_omni_videoamme_talker_tp2_ci.py
│   ├── test_tts_ci.py
│   ├── test_asr_ci_multi_speaker.py
│   └── test_asr_ci_seedtts.py
└── unit_test/
    ├── benchmarks/
    │   └── test_dataset_regressions.py
    ├── test_tune_ci_thresholds.py
    ├── quantization/
    │   ├── test_autoround.py
    │   ├── test_fp8.py
    │   ├── test_integration.py
    │   └── test_weight_preprocess.py
    ├── fixtures/
    │   ├── fish_fakes.py
    │   ├── pipeline_fakes.py
    │   └── qwen_fakes.py
    ├── utils/
    │   └── test_audio.py
    ├── pipeline/
    │   ├── helpers.py
    │   ├── test_comm_engine_ack.py
    │   ├── test_comm_router.py
    │   ├── test_compile.py
    │   ├── test_coordinator.py
    │   ├── test_gpu_memory.py
    │   ├── test_ipc.py
    │   ├── test_placement.py
    │   ├── test_runtime_adapter.py
    │   ├── test_runtime_schema.py
    │   ├── test_scheduler.py
    │   ├── test_simple_scheduler_concurrent.py
    │   ├── test_stage.py
    │   ├── test_stage_process_env.py
    │   └── test_stage_streaming.py
    ├── relay/
    │   ├── test_cuda_ipc_relay.py
    │   └── test_shm_relay.py
    ├── models/
    │   └── test_model_capabilities.py
    ├── qwen3_omni/
    │   ├── test_cli.py
    │   ├── test_code2wav.py
    │   ├── test_colocation_config.py
    │   ├── test_config_manager.py
    │   ├── test_fp8_backend_config.py
    │   ├── test_example_launcher.py
    │   ├── test_logit_shaping.py
    │   ├── test_pipeline.py
    │   ├── test_sglang_ar_budget.py
    │   ├── test_streaming.py
    │   ├── test_talker.py
    │   └── test_text_template.py
    ├── ming_omni/
    │   ├── test_omni_serve.py
    │   ├── test_pipeline.py
    │   ├── test_streaming_decode.py
    │   ├── test_streaming_e2e_glue.py
    │   ├── test_streaming_speech_config.py
    │   ├── test_talker.py
    │   ├── test_talker_voice_validation.py
    │   ├── test_thinker.py
    │   ├── test_tokenizer.py
    │   ├── test_tp.py
    │   └── test_vision_patch_embed_linear.py
    ├── qwen3_asr/
    │   ├── test_pipeline.py
    │   └── test_request_builders.py
    ├── fun_asr/
    │   ├── test_pipeline.py
    │   └── test_request_builders.py
    ├── moss_transcribe_diarize/
    │   ├── test_encoder_cache.py
    │   ├── test_pipeline.py
    │   ├── test_request_builders.py
    │   ├── test_stream_output_builder.py
    │   └── test_transcription_adapter.py
    ├── qwen3_tts/
    │   └── test_pipeline.py
    ├── higgs_tts/
    │   ├── test_async_decode_runner.py
    │   ├── test_batched_step.py
    │   ├── test_cli_decode_mode.py
    │   ├── test_pipeline.py
    │   └── test_request_builders.py
    ├── moss_tts/
    │   └── test_pipeline.py
    ├── moss_tts_local/
    │   ├── test_pipeline.py
    │   ├── test_radix_hash.py
    │   ├── test_s0_gate.py
    │   ├── test_state_pool.py
    │   ├── test_streaming_vocoder.py
    │   └── test_vocoder_decoder.py
    ├── router/
    │   ├── test_app.py
    │   └── test_core.py
    ├── profiler/
    │   ├── test_event_recorder.py
    │   ├── test_stop_run_id.py
    │   └── test_views.py
    ├── serve/
    │   ├── test_generation_batch_policy.py
    │   ├── test_generation_server_args.py
    │   └── test_openai_api.py
    ├── scheduling/
    │   ├── test_engine_factory.py
    │   ├── test_pipeline_state.py
    │   ├── test_reference_encoder.py
    │   └── test_streaming_vocoder.py
    ├── fishaudio_s2_pro/
    │   ├── test_pipeline.py
    │   ├── test_streaming_vocoder.py
    │   ├── test_tts.py
    │   └── test_vocoder.py
    └── voxtral_tts/
        └── test_pipeline.py
```

## How To Add A Test


General rules:

- Protect user-visible contracts and component ownership, not incidental implementation structure.
- Keep imports thin and consistent. If a test monkeypatches a module object,
  call through that module alias instead of mixing direct symbol imports.
- Reuse existing helpers and fakes before adding another scheduler, relay, or
  lifecycle helper.
- Add a one-sentence docstring to non-obvious contract tests.
- Do not add root-level `tests/test_*.py` files.


## Markers

Markers are registered in `pyproject.toml` under `[tool.pytest.ini_options]`.
Tag each test with the marker that matches its lane and use it to filter runs.

- `benchmark`: GPU performance / parity tests in `test_model/`. May require a
  populated HF cache and tens of GB of GPU memory; per-test docstrings call
  out hardware needs.
- `tts_stage(name)`: in-file CI stage selector for TTS benchmarks.
  Combined with `--tts-stage` (see `test_model/conftest.py`).


## Root Files

- `README.md`: This file. It explains test ownership and where new tests belong.
- `__init__.py`: Keeps `tests` importable as a package.
- `utils.py`: Shared helpers used by model CI tests.

## `data/`

Small static fixtures shared by tests, such as images, audio, and short videos.
Keep these files small and deterministic. Large model artifacts, generated
outputs, and benchmark datasets should live outside the unit test tree.

## `test_model/`

End-to-end and model CI tests. These are allowed to depend on real servers,
model snapshots, benchmark artifacts, optional packages, and GPU/runtime
resources.

Expected command (GPU benchmark subset):

```bash
pytest tests/test_model -m benchmark -v -s
```

Relevant model CI ownership:

- Qwen3-Omni server fixtures in `conftest.py` span the five viable 2xH100
  serving topologies (one per stage type — see the H20->H100 migration PR):
  `qwen3_omni_fp8_colocated_server` (FP8 colocated DP2),
  `qwen3_omni_bf16_colocated_server` / `qwen3_omni_bf16_colocated_thinker_server`
  (BF16 colocated DP2, full / thinker-only), `qwen3_omni_bf16_disagg_server`
  (BF16 disaggregated), and `qwen3_omni_fp8_tp2_server` (FP8 thinker-TP=2);
  BF16 thinker-TP=2 is exercised by thinker_length via `_start_qwen3_omni_tp2`.
- `test_qwen3_omni_tts_ci.py`: gates the SeedTTS speed/WER path through the
  router at TTS generation concurrency 16 and verifies both colocated workers
  receive traffic. WER reuses saved audio after the Qwen3-Omni server is
  stopped, then transcribes through Qwen3-ASR at concurrency 32.
- `test_asr_ci_multi_speaker.py`: MOSS-Transcribe-Diarize multi-speaker
  ASR/diarization correctness + speed via the managed router at DP=2. It
  reuses the movies800 benchmark path, writes
  `moss_transcribe_diarize_results.json`, and enforces calibrated
  accuracy/speed thresholds generated from `tune-ci-thresholds`.
- `test_asr_ci_seedtts.py`: Qwen3-ASR correctness + speed via SGLang Omni
  router (`/v1/audio/transcriptions`). Uses the full 1088-sample English
  SeedTTS set; writes `qwen3_asr_results.json` for threshold calibration
  (`asr` in `tune-ci-thresholds`). Its stdout uses the same boxed
  summary style as the other benchmark stages: `ASR WER Benchmark Result`
  followed by `ASR Speed Benchmark Result`.
- `utils.py`: shared fixture/helpers for talker/TTS WER CI —
  stops the upstream model server, runs `delete_gpu_process.sh --kill-orphans`, then launches
  a Qwen3-ASR router. It also owns the WER ASR concurrency constant
  (`QWEN3_ASR_WER_CONCURRENCY`, currently 32). Used by Qwen3 talker WER tests
  and TTS WER tests instead of the in-process transformers Whisper pipeline.
- Talker / video WER CI (`test_qwen3_omni_*_talker_ci.py`, `test_tts_ci.py`):
  generate audio with the model router first, tear down that server, free both
  GPUs, then transcribe saved WAVs through the ASR router. Qwen3-Omni
  talker/TTS generation concurrency is 16, including the
  `videoamme_talker_tp2` stage; ASR/WER transcription concurrency is 32.
- CI env alignment on the H100 repro host: `source .github/scripts/ci_env.sh`
  then `source omni/bin/activate`.
  Omni CI (`omni-ci.yaml`) runs benchmark suites sequentially after one shared
  setup: PR Test (`test.yaml` unit tests) → ASR CI → TTS CI → Qwen3-Omni CI. A
  failure in an earlier suite does not skip later ones; only a failed setup
  blocks the chain.
  Full WER sweep: `.github/scripts/run_all_wer_ci_aligned.sh` (milestones on
  stdout; details in `/tmp/wer_ci_qwen3.log` and `/tmp/wer_ci_tts.log`).
- GPU handoff between stages: `.github/scripts/delete_gpu_process.sh --kill-orphans` (kills orphan
  spawn/router workers, waits for VRAM below threshold).
- `qwen3_omni_vision_sglang_env`: session-scoped SGLang dist + DP-attention
  init from `conftest.py`, shared by every Qwen3-Omni vision-encoder benchmark
  module — avoids re-initializing the process-global TP group when the combined
  `-m benchmark` command runs more than one module.
- `test_qwen3_omni_realtime.py`: starts `examples/run_qwen3_omni_server.py`
  with `--enable-realtime` and drives `/v1/realtime` through a real WebSocket
  client to cover text responses, server VAD transcription, and disconnect
  teardown.
- `test_rl_distributed_weight_update.py`: launches a Higgs TTS worker on one GPU
  and a rank-0 trainer subprocess on another GPU, initializes the distributed
  weight-update group, broadcasts base-model body weights, verifies the
  `tts_engine` checksum changes, destroys the update group, and checks the
  server still serves audio. It is skipped unless two GPUs and the required
  Higgs base checkpoint are already available in the Hugging Face cache.
- CLI flags `--s2pro-stage {nonstream,stream,consistency,all}` and
  `--concurrency {1,2,4,8,16,all}`: scope an S2-Pro CI sweep without editing
  source.

### Ming TP Parity

`tests/test_model/test_ming_tp_parity_ci.py` launches Ming-Omni twice, first
with TP=1 and then with TP=N, and compares deterministic text responses. It is
skipped by default because it requires a Ming checkpoint and enough GPUs.

Remote GPU example:

```bash
RUN_MING_TP_PARITY=1 \
MING_TP_PARITY_TP_SIZE=4 \
MING_TP_PARITY_CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
MING_OMNI_MODEL_PATH=inclusionAI/Ming-flash-omni-2.0 \
MING_OMNI_MODEL_NAME=ming-omni \
python3 -m pytest tests/test_model/test_ming_tp_parity_ci.py -q -s
```

- `test_tts_ci.py`: default TTS CI gate. It starts the TTS managed router
  with two one-GPU workers using the default model config, runs the
  full SeedTTS EN set (1088 samples) in non-streaming / streaming stages at
  concurrency 16, and frees the server GPUs before ASR/WER and
  speaker-similarity checks. Non-streaming and streaming WER pass the selected
  TTS generation concurrency into the result config while keeping Qwen3-ASR
  transcription concurrency at 32.
- `test_tts_consistency_artifacts.py`: CPU-only stage-3 check that compares
  TTS non-stream and streaming `speed_results.json` under
  `${OMNI_CI_HOME}/tts-stage-results/{nonstream,stream}/`.
- CLI flags `--tts-stage {tts-stage-1-nonstream,tts-stage-2-stream,tts-stage-3-consistency,all}`
  and `--concurrency {1,2,4,8,16,all}`: scope a TTS CI sweep without
  editing source.

## `unit_test/`

Fast contract tests that should run without model downloads or real server
startup. Keep these focused on the smallest component that owns the behavior.

Expected command:

```bash
pytest tests/unit_test -q
```
Choose the location by the behavior contract being protected, not by the file
that happened to contain an older version of the test.

- `unit_test/pipeline/`: Model-agnostic pipeline tests:
  - compile
  - placement planning
  - runtime wiring
  - runtime schema/adapter behavior
  - coordinator behavior
  - stage routing
  - centralized comm router selection, data-reference serialization, ack
    lifecycle, and sender backpressure release
  - local-object fan-out selector contracts, including negative coverage for
    shared mutable payload containers while preserving tensor leaf sharing
  - stage process environment
  - relay handling
  - stream relay/IPC selector contracts, including negative coverage for CPU
    tensor metadata and large inline metadata on same-GPU stream chunks
  - GPU memory accounting helpers
  - IPC lifecycle
  - scheduler batching
  - scheduler errors
  - scheduler concurrency
  - scheduler callable contracts, including sync wrappers and callable objects
    that return awaitables.
- `unit_test/relay/`: Low-level data-plane relay tests:
  - shared-memory relay byte movement, cleanup, and handle lifecycle on CPU
  - CUDA-IPC relay metadata/open/close behavior for GPU tensor handoff; CUDA
    tests require CUDA and multi-GPU coverage is hardware-gated
  - these tests prove transport mechanics, not full pipeline throughput,
    NVLink selection, or production backpressure behavior; keep those covered
    in `unit_test/pipeline/` integration tests and GPU benchmarks.
- `unit_test/benchmarks/`: Benchmark dataset/loading regression tests.
- `unit_test/test_tune_ci_thresholds.py`: Unit tests for
  `.claude/skills/tune-ci-thresholds/tune.py` calibration tooling — sample-scope
  discovery (`CONCURRENCY` must not be treated as a sample count), GPU cleanup
  scoping for concurrent calibration groups, metric dispersion/outlier reporting,
  Wilson accuracy intervals, and `merge-runs` validation for disjoint strict-ready
  partitions. Run with the rest of the fast suite:

  ```bash
  pytest tests/unit_test/test_tune_ci_thresholds.py -q
  ```

- `unit_test/utils/`: Shared utility tests:
  - audio loading helpers for data URIs, file URIs, HTTP URLs, timeout fallback,
    and mono/channel preservation.
- `unit_test/models/`: Model registry and cross-model contract tests:
  - static TTS `ModelCapabilities` declarations, registry lookup, aliases, and
    launcher startup logging.
- `unit_test/scheduling/`: Shared scheduling-service unit tests:
  - `ReferenceEncodeService` cache, same-key single-flight, timeout, failure,
    and revalidation semantics.
- `unit_test/qwen3_asr/`: Qwen3-ASR unit tests:
  - pipeline config and stage factory concurrency defaults
  - single-source audio token length formula used by both processor and
    request builder paths
  - token-level result adapter marker handling, avoiding decode/encode
    text round-trips for byte-level BPE output.
- `unit_test/fun_asr/`: Fun-ASR-Nano unit tests:
  - pipeline config and stage factory: single `asr` stage, `max_running_requests=32`,
    auto static KV budget, disabled multimodal embedding cache and torch.compile,
    and `FunAsrNanoForConditionalGeneration` registry wiring
  - request builder: inclusive audio offset recording, language-prompt prefix
    construction, and result adapter direct-transcript decoding.
- `unit_test/moss_transcribe_diarize/`: MOSS-Transcribe-Diarize unit tests:
  - pipeline config and stage factory default routing/memory contracts
  - request builder audio-source resolution, single-audio enforcement, audio
    token padding, and default transcribe+diarize prompt injection
  - verbose_json transcription adapter: architecture-based resolution, special
    token stripping, and speaker/timestamp segment parsing with fallback.
- `unit_test/qwen3_omni/` Qwen3-Omni unit tests:

  - public CLI/config behavior
  - example launcher config contract (TP/GPU/mem-fraction overrides)
  - SGLang argument builders
  - backend policy and quantization compatibility contracts
  - tokenizer and preprocessing fallback behavior
  - memory flag contracts
  - colocation config and SGLang AR budget contracts
  - `Qwen3OmniPipelineState` request builders, including projected payload container
    isolation for mutable streaming state
  - talker behavior, including partial-prefix startup gate, the real
    `_build_talker_request_data` propagation contract (input_ids,
    tts_pad_embed, sampling_seed, fallback chunks, thinker_done), and the
    `_rollback_decode_prep_after_skip` idempotency contract, projected prefill
    tensor storage/slicing, decode feedback/text FIFO consumption, and replay
    of generated-token input embeds after decode retract
  - Code2Wav streaming/cleanup behavior
  - logit-shaping helpers (e.g. repetition penalty) numerical equivalence with the original per-row scalar formulas.

- `unit_test/ming_omni/` Ming-Omni unit tests:

  - text + speech pipeline config and stage schema
  - omni serve CLI/config merge, default speech vs. text-only selection,
    launcher handoff, GPU placement, TP wiring, and unsupported flag capability
    boundaries
  - stage factory and scheduler contracts (preprocessing, encoders, thinker, talker, decode)
  - thinker bootstrap registration and Ming model runner wiring
  - multimodal embed injection (per-modality consumed state, pad-value fallback, short-embeds detection)
  - image/vision encoder TP context preservation
  - audio/image preprocessor placeholder construction and cache-key plumbing
  - talker executor request gating and result-builder modality merging
  - talker voice-preset validation (load-time manifest / wav existence, request-time prompt_wav_path priority), duration-cap heuristic, and `generate()` final-chunk flush across stop-token and step-ceiling exits
  - Bailing tokenizer loader fallback for vocab compatibility
  - TP topology validation (rank-specific stage specs, talker/thinker GPU collision detection, server_args alignment before infra init)
  - vision encoder `patch_embed` numerical equivalence: `nn.Conv3d` vs `F.linear` reshape at the substitution boundary, using synthetic weights without loading real Ming checkpoints.
  - streaming text decode (`MingStreamingDetokenizeScheduler` /
    `make_text_stream_output_builder`): per-token detokenization and delta
    emission with UTF-8 multibyte boundary safety, streaming vs. non-streaming
    final-result shape, stream-completion ordering races, per-request failure
    isolation, bounded orphan `_state` eviction with abort cleanup, and
    text-stream output gating on the `stream` flag, chunked prefill, and
    text-vs-audio-only output modalities
  - streaming speech glue and topology: thinker text/combined stream builders
    fanning token ids to decode and text to the segmenter (audio-only kept off
    decode), client merge of decode deltas with the talker stream, and
    `MingOmniStreamingSpeechPipelineConfig` wiring (segmenter between thinker and
    talker, terminal talker-stream stage, thinker/talker GPU-range collision
    rejection, streaming variant exposure).

- `unit_test/qwen3_tts/`: Qwen3-TTS unit tests:
  - pipeline config and registry contracts
  - OmniScheduler-backed AR stage factory wiring
  - request mapping for `ref_audio` / `ref_text` and `references`
  - model-owned default preservation for language and sampling parameters
  - Base, CustomVoice, and VoiceDesign request validation
  - voice-clone reference validation
  - pipeline payload state serialization.

- `unit_test/higgs_tts/`: Higgs TTS unit tests:
  - OmniScheduler-backed AR stage factory wiring
  - upstream Transformers codec binding and bundled-config state-dict structure
  - sampler-driven finish handling for eager and CUDA-graph paths
  - request builder sampling normalization and server-side token caps
  - model slot cleanup and engine timing in scheduler result adapters
  - async-decode one-step-lookahead parity with the synchronous collect path
  - async-decode default-on config + `--decode-mode async|sync` CLI override.

- `unit_test/moss_tts/`: MOSS-TTS unit tests:
  - pipeline config and registry contracts
  - OmniScheduler-backed AR/vocoder stage factory wiring
  - request mapping for `ref_audio`, `references`, and `token_count`
  - preprocessing handoff and abort cleanup behavior
  - delay-pattern runner, codec splitting, and seeded sampling contracts.

- `unit_test/moss_tts_local/`: MOSS-TTS Local unit tests:
  - pipeline config, request builders, and scheduler adapter contracts
  - decode-state pool acquisition, launch-state gathers, repetition-penalty history, cleanup, and resume/retraction lifecycle
  - chunked prefill feedback/journal suppression and postprocess alignment checks
  - synchronous frame-decode parity harness and S0 gate coverage
  - streaming vocoder session lifecycle, per-request chunk-threshold and
    coalescing contracts, decode-failure isolation, and non-streaming full-sequence
    decode through the codec path
  - MOSS-TTS Local vocoder decoder packing, local-causal FlashAttention window
    equivalence, CUDA bf16 packed-vs-SDPA parity, zero-length handling, and
    flash-unavailable fallback.

- `unit_test/router/`: SGLang-Omni Router unit tests:
  - router CLI/config behavior
  - worker metadata and health-state contracts
  - request routing, proxying, and streaming relay
  - worker selection policy behavior
  - managed launcher command construction and cleanup.

- `unit_test/serve/`: In-process serving API unit tests:
  - generation-stage SGLang server-args role mapping and CLI override capability boundaries
  - OpenAI-compatible request/response behavior
  - streaming response framing and failure semantics.

- `unit_test/fishaudio_s2_pro/`: FishAudio S2-Pro unit tests:
  - inference prompt segmentation, reference VQ edge cases, and state contracts
  - TTS scheduler behavior
  - model-runner state transitions
  - vocoder batching/trim behavior
  - streaming vocoder chunking, flush, and abort behavior.

- `unit_test/voxtral_tts/`: Voxtral-TTS unit tests:
  - pipeline config and registry contracts
  - current `StageConfig` schema wiring
  - SGLang-backed generation and vocoder GPU placement contracts
  - terminal stage behavior.

- `unit_test/profiler/`: Request-level profiler unit tests:
  - `RequestEvent` schema and JSONL emit/append behavior
  - concurrent emit safety under multiple threads
  - lifecycle (start / stop / run_id mismatch / stage substitution)
  - timeline reconstruction, stage breakdown, hop breakdown, malformed-line tolerance.

- `unit_test/quantization/`: Tests for the compatibility layer on top of
  SGLang's native quantization (`sglang_omni/quantization.py`):
  - `resolve_quant_config` discovery from root/nested sub-configs and
    `compression_config`, plus edge cases (missing/empty quantization_config)
  - FP8 detection (with/without weight_block_size), weight_scale_inv reciprocal
    conversion, and error handling (empty/zero/non-finite/non-float scale tensors)
  - AutoRound stage-prefix normalization for block_name_to_quantize (string
    input is rejoined as a string; list input is normalized in place and
    stays a list) and extra_config regex keys via `normalize_quant_config`
  - `get_weight_preprocessor` contract: identity by default (native block-FP8,
    AutoRound), FP8 reciprocal preprocessor only when `fp8_scale_inverted=True`,
    nested config traversal
  - model_worker integration: `_apply_omni_quantization_adapters` triggers
    stage-local normalization from hf_config and nested text_config only when
    needed


- `unit_test/fixtures/`: Shared fakes. Single-test
  helpers should stay local until a second test needs them.
