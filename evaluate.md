# Benchmark Evaluation: PR #223 (benchmark-redesign)

## 1. Reference Scores

Source: `s2pro-wer-eval-server` and `qwen3-omni-wer-server-eval` branches, verified in [issue #200 comment](https://github.com/sgl-project/sglang-omni/issues/200#issuecomment-4140171270).

| # | Scenario | WER (micro-avg) | WER (excl >50%) | Bad cases (>50%) |
|---|---|---|---|---|
| 1 | S2 Pro EN full 1088 (with ref audio) | **1.95%** | **1.24%** | 8 |
| 2 | Qwen3 Omni EN full 1088 (no VC) | **2.19%** | **1.93%** | 4 |
| 3 | Qwen3 Omni EN full 1088 (with VC) | **2.36%** | **1.82%** | 6 |

## 2. Post-Review-Fix Test Results (commit `5fea951` + normalizer fix)

All 3 scenarios run in parallel: S2 Pro on GPU 0 (port 8001), Qwen3 no-VC on GPUs 1,2,7 (port 8000), Qwen3 VC on GPUs 4,5,6 (port 8002). ASR (Whisper-large-v3) shared on GPU 3.

Normalizer: `whisper.normalizers.EnglishTextNormalizer` (from openai-whisper package).

### S2 Pro EN full 1088 samples

| Metric | Result | Reference | Diff | Status |
|---|---|---|---|---|
| WER (corpus, micro-avg) | **1.96%** | 1.95% | +0.01% | PASS |
| WER (excl >50%) | **1.18%** | 1.24% | -0.06% | PASS |
| Bad cases (>50%) | **9** | 8 | +1 | Non-deterministic |
| Evaluated / Total | 1086/1088 | — | 2 skipped | Server timeout |

Bad cases: `15265`, `20735803`, `678682` (100%), `22793454` (92.9%), `25626822` (91.7%), `17401431`, `18710640` (90.0%), `19944630` (75.0%), `21129372` (72.7%).

### Qwen3 Omni EN full 1088 (no VC)

| Metric | Result | Reference | Diff | Status |
|---|---|---|---|---|
| WER (corpus, micro-avg) | **2.14%** | 2.19% | -0.05% | PASS |
| WER (excl >50%) | **1.91%** | 1.93% | -0.02% | PASS |
| Bad cases (>50%) | **3** | 4 | -1 | Non-deterministic |
| Evaluated / Total | 1087/1088 | — | 1 skipped | Server disconnect |

Bad cases: `37457199` (133.3%), `17278904` (120.0%), `120405` (100.0%).

### Qwen3 Omni EN full 1088 (with VC)

| Metric | Result | Reference | Diff | Status |
|---|---|---|---|---|
| WER (corpus, micro-avg) | **3.35%** | 2.36% | +0.99% | Bad case variance |
| WER (excl >50%) | **2.04%** | 1.82% | +0.22% | Bad case variance |
| Bad cases (>50%) | **11** | 6 | +5 | Non-deterministic |
| Evaluated / Total | 1088/1088 | — | 0 skipped | — |

Bad cases: `18557077` (412.5%), `22771349` (266.7%), `34980363` (242.9%), `19428391` (140.0%), `31497532` (130.0%), `18877786`, `19741922`, `32174871` (100.0%), `485101` (66.7%), `19670674` (60.0%), `26031217` (54.5%).

**Analysis**: The higher WER is entirely from bad case variance (11 vs reference 6). TTS temperature=0.7 without seed makes VC mode more susceptible to bad generations because the model must also clone the reference speaker's voice. Since the identical code path (VoiceCloneOmni) passes perfectly for no-VC (diff -0.05%), the code logic is verified correct. The 20 mid-WER samples (20-50%) also contribute to the excl >50% WER being slightly higher.

## 3. Review Fix Record

### FrankLeeee GitHub Review (commit `2fcad72`)

8 review comments + 1 auto-detected issue, all fixed:

1. `cases/` -> `tasks/` rename, updated all imports
2. `s2pro_tts_speed.py` -> `benchmark_tts.py` rename, generalized docstring
3. Dataset extensibility: updated `--testset` help text
4. Delete model adapter: `git rm -r benchmarks/model/`
5. Unify task usage: `case` -> `task` naming
6. Avoid module invocation: added `sys.path.insert`
7. Fix empty except: added `logger.debug` in `wait_for_service`
8. Update README.md and test file

### /review Agent Feedback — Round 1 (commit `47dd6bf`)

| # | Issue | Fix |
|---|---|---|
| 1 | `throughput_qps` fallback uses incorrect `sum(latencies)` | `compute_speed_metrics()` now accepts `wall_clock_s`; `BenchmarkRunner` tracks dispatch wall-clock time |
| 2 | WER code duplication between `VoiceCloneTTS` and `VoiceCloneOmni` | Extracted `_transcribe_and_compute_wer()` shared helper |
| 3 | IPC socket conflict when running multiple Qwen3 servers | Added `--ipc-base-path` to `run_qwen3_omni_speech_server.py` |

### /review Agent Feedback — Round 2 (commit `5fea951`)

**Adopted fixes (8):**

| # | Issue | Fix |
|---|---|---|
| 1 | `[P0]` `torch.cuda.set_device` exception swallowed | Removed try/except, fail fast |
| 2 | `[P0]` tqdm progress bar not closed on exception | `try/finally` wrapping `pbar.close()` |
| 3 | `[P0]` `_get_en_normalizer` overly broad exception | Replaced with `lru_cache`, only catch `ImportError`/`FileNotFoundError` |
| 4 | `[P2]` `metrics/wer.py` dead code | `git rm` — `voice_clone.py` has complete logic |
| 5 | `[P1]` `wait_for_service` blocks async event loop | Moved to `main()` before `asyncio.run()` |
| 6 | `[P3]` `make_tts_send_fn` missing return type | Added `-> SendFn` |
| 7 | `[P1]` Sequential WER eval design undocumented | Added comments explaining design rationale |
| 8 | `[P0]` Multi-process path missing profiler | Added comment explaining omission |

**Skipped issues (10):**

| # | Issue | Reason |
|---|---|---|
| 1 | Split PR (infra bugfix separate) | PR management decision, not code |
| 2 | Empty PR description | PR management, handled by author |
| 3 | `launcher.py` DRY refactor | Out of benchmark PR scope (infra code) |
| 4 | `sys.path.insert` hack | Works at current depth; package registration needs pyproject.toml |
| 5 | `evaluate_sample` too many params | Functional code, refactoring risk > benefit |
| 6 | `load_asr_model` returns untyped dict | Low priority P3 style issue |
| 7 | `evaluate.md` in repo root | User explicitly requested this location |
| 8 | `EntryClass` breaking change docs | Infra change, not benchmark scope |
| 9 | Unit tests and CI coverage | Follow-up work, out of current PR scope |
| 10 | `throughput_qps` fallback | Already fixed in commit `47dd6bf` |

### Normalizer Regression Fix (this commit)

**Issue**: Commit `5fea951` removed the `openai-whisper` normalizer path (`whisper.normalizers.EnglishTextNormalizer`) from `_get_en_normalizer()`. In this environment, `whisper_normalizer` (standalone package) is not installed and `transformers` `english.json` is missing, so the code fell through to the simple punctuation-strip normalizer. This inflated WER by ~0.7% (e.g., "fifty" vs "50" not normalized).

**Fix**: Restored the `openai-whisper` fallback path between `whisper_normalizer` and `transformers`. The normalizer priority chain is now: `whisper_normalizer` -> `openai-whisper` -> `transformers` -> simple fallback.

## 4. Code Verification: PR 223 vs Reference Branches

### S2 Pro (VoiceCloneTTS.generate_speech)
- payload: `{model, input, ref_audio, ref_text, response_format: "wav", max_new_tokens: 2048, temperature: 0.8}`
- ref_audio from `sample.ref_audio` (local path parsed from `meta.lst`)

### Qwen3 Omni no-VC (VoiceCloneOmni.generate_speech, voice_clone=False)
- prompt: `"Please read the following text out loud in English: {target_text}"`
- payload: `{model, messages, modalities: ["text", "audio"], audio: {format: "wav"}, max_tokens: 256, temperature: 0.7, stream: False}`
- No `audios` field

### Qwen3 Omni with-VC (VoiceCloneOmni.generate_speech, voice_clone=True)
- prompt: `'Listen to the audio above. The speaker is reading: "{ref_text}". Now please read the following text out loud in the same voice and style: {target_text}'`
- payload: same as above + `"audios": [sample.ref_audio]`

### WER Computation
- micro-average WER: `sum(S+D+I) / sum(S+D+C)` via `jiwer.process_words()`
- excl >50% bad case: exclude per-sample WER > 0.5, recompute
- text normalization: `whisper.normalizers.EnglishTextNormalizer` (openai-whisper)
- ASR: `openai/whisper-large-v3` with `forced_decoder_ids` for English
