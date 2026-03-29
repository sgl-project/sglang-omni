# Benchmark Evaluation: PR #223 (benchmark-redesign)

## 1. Reference Scores

Source: `s2pro-wer-eval-server` and `qwen3-omni-wer-server-eval` branches, verified in [issue #200 comment](https://github.com/sgl-project/sglang-omni/issues/200#issuecomment-4140171270).

| # | Scenario | WER (micro-avg) | WER (excl >50%) | Bad cases (>50%) |
|---|---|---|---|---|
| 1 | S2 Pro EN first 50 (with ref audio) | **0.89%** | **0.89%** | 0 |
| 2 | S2 Pro EN full 1088 (with ref audio) | **1.95%** | **1.24%** | 8 |
| 3 | Qwen3 Omni EN full 1088 (no VC) | **2.19%** | **1.93%** | 4 |
| 4 | Qwen3 Omni EN full 1088 (with VC) | **2.36%** | **1.82%** | 6 |

## 2. Pre-Fix Test Results (before FrankLeeee review fixes)

Ran 4 scenarios on the original PR 223 code.

### Test 1: S2 Pro EN first 50 samples (with ref audio)

- **Server**: S2 Pro on GPU 0, port 8000
- **ASR**: Whisper-large-v3 on GPU 2

| Metric | Result | Reference | Match? |
|---|---|---|---|
| WER (corpus, micro-avg) | **0.89%** | 0.89% | EXACT |
| WER (excl >50%) | **0.89%** | 0.89% | EXACT |
| Bad cases (>50%) | **0** | 0 | EXACT |
| WER per-sample mean | 0.62% | 0.62% | EXACT |

### Test 2: S2 Pro EN full 1088 samples (with ref audio)

| Metric | Result | Reference | Match? |
|---|---|---|---|
| WER (corpus, micro-avg) | **2.03%** | 1.95% | ~0.08% diff (non-deterministic) |
| WER (excl >50%) | **1.24%** | 1.24% | EXACT |
| Bad cases (>50%) | **9** | 8 | +1 (non-deterministic) |

### Test 3: Qwen3 Omni EN full 1088 (no VC)

| Metric | Result | Reference | Match? |
|---|---|---|---|
| WER (corpus, micro-avg) | **2.34%** | 2.19% | Bad case count variance |
| WER (excl >50%) | **1.94%** | 1.93% | ~0.01% diff |
| Bad cases (>50%) | **6** | 4 | +2 (non-deterministic) |

**Alignment verification**: When excluding the **same 4 reference bad case samples** (not my 6), corpus WER = **2.19%** — **EXACT MATCH**. Code logic is identical; difference is purely from TTS randomness.

### Test 4: Qwen3 Omni EN full 1088 (with VC)

| Metric | Result | Reference | Match? |
|---|---|---|---|
| WER (corpus, micro-avg) | **2.70%** | 2.36% | Bad case count variance |
| WER (excl >50%) | **1.88%** | 1.82% | ~0.06% diff |
| Bad cases (>50%) | **5** | 6 | -1 (non-deterministic) |

### Non-Determinism Analysis

TTS generation at temperature=0.7 (Qwen3) / 0.8 (S2 Pro) without seed is stochastic. The same sample can produce WER=100% in one run and WER=7.7% in another.

Evidence:
- Reference bad case `common_voice_en_19845853` (WER=100% in reference) → WER=7.7% in my run
- My bad case `common_voice_en_20791751` (WER=118%) → likely not a bad case in reference run
- 3 out of 4 reference bad cases overlap with my bad cases (17324784, 19717736, 19284142)

**Conclusion**: PR 223 code logic is verified identical to reference branches. Score differences are purely from TTS non-determinism, not code bugs.

## 3. FrankLeeee GitHub Review Fixes (commit `2fcad72`)

8 review comments + 1 auto-detected issue, all fixed:

1. **`cases/` → `tasks/` rename** — `git mv benchmarks/cases benchmarks/tasks`, updated all imports
2. **`s2pro_tts_speed.py` → `benchmark_tts.py` rename** — generalized docstring
3. **Dataset extensibility** — updated `--testset` help text to accept any meta.lst format
4. **Delete model adapter** — `git rm -r benchmarks/model/` (6 files), updated README
5. **Unify task usage** — `case` → `task` variable naming across all eval scripts
6. **Avoid module invocation** — `python -m ...` → `python benchmarks/eval/...`, added `sys.path.insert`
7. **Fix empty except** — added `logger.debug` in `wait_for_service`
8. **Update README.md** — directory structure, script names, execution style
9. **Update test file** — `tests/test_model/test_s2pro_benchmark.py` uses direct script path

## 4. /review Agent Fixes (commit `47dd6bf`)

From 19 issues reported, only 2 were valid against current code (the rest referenced files/code that no longer existed):

1. **throughput_qps fix** (issue 2.7): `compute_speed_metrics()` now accepts `wall_clock_s` for correct throughput under concurrency. `BenchmarkRunner` tracks dispatch wall-clock time.
2. **Code duplication fix** (issue 2.10): Extracted `_transcribe_and_compute_wer()` shared helper from identical code in `VoiceCloneTTS.evaluate_sample` and `VoiceCloneOmni.evaluate_sample`.
3. **IPC socket isolation**: Added `--ipc-base-path` to `run_qwen3_omni_speech_server.py` for running multiple servers simultaneously.

## 5. Post-Fix Test Results (partial)

### Test 1 (post-fix): S2 Pro EN first 50

| Metric | Result | Reference | Match? |
|---|---|---|---|
| WER (corpus, micro-avg) | **0.89%** | 0.89% | EXACT |
| WER (excl >50%) | **0.89%** | 0.89% | EXACT |

**Tests 2, 3, 4 post-fix verification pending.**

## 6. Code Verification: PR 223 vs Reference Branches

### S2 Pro (VoiceCloneTTS.generate_speech)
- payload: `{model, input, ref_audio, ref_text, response_format: "wav", max_new_tokens: 2048, temperature: 0.8}` ✅
- ref_audio from `sample.ref_audio` (local path parsed from `meta.lst`) ✅

### Qwen3 Omni no-VC (VoiceCloneOmni.generate_speech, voice_clone=False)
- prompt: `"Please read the following text out loud in English: {target_text}"` ✅
- payload: `{model, messages, modalities: ["text", "audio"], audio: {format: "wav"}, max_tokens: 256, temperature: 0.7, stream: False}` ✅
- No `audios` field ✅

### Qwen3 Omni with-VC (VoiceCloneOmni.generate_speech, voice_clone=True)
- prompt: `'Listen to the audio above. The speaker is reading: "{ref_text}". Now please read the following text out loud in the same voice and style: {target_text}'` ✅
- payload: same as above + `"audios": [sample.ref_audio]` ✅

### WER Computation
- micro-average WER: `sum(S+D+I) / sum(S+D+C)` via `jiwer.process_words()` ✅
- excl >50% bad case: exclude per-sample WER > 0.5, recompute ✅
- text normalization: `whisper_normalizer.english.EnglishTextNormalizer` ✅
- ASR: `openai/whisper-large-v3` with `forced_decoder_ids` for English ✅
