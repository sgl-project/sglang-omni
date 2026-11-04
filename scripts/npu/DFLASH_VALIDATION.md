# Qwen3-ASR DFlash NPU validation

Instruction revision: ASR-DFLASH-NPU-03 (2026-11-04). Branch: `qwen3-asr-dflash-npu` in `wangyao-i/sglang-omni`. Originally based on PR #3462 at `c3bddd9e903c7304d17657e4aac12f471c5e0d29`; now synchronized with main `15be209c187552706a214d63cee0e3d5753610f1` after #3462 merged. Freeze and report the fetched branch's full commit SHA separately from this instruction revision. The current manifest requires SGLang 0.5.23. Server-reported 32k and pinned 48k runs used Omni `71bdf12e` and SGLang 0.5.22; those are historical bounded results, not native qualification of this changed mainline/runtime baseline. Reconcile and validate the current environment before reusing them. Revision 02 corrected the checkpoint attention check and bounded the initial context; those requirements remain in force.

## Scope and ownership

Validate the shared reduced-vocabulary DFlash path on Qwen3-ASR-1.7B, one exclusively assigned Ascend device, offline transcription, BF16, greedy, eager, concurrency one. Use an isolated image and checkout allocated by the user. Do not reuse another task's live checkout, modify site-packages, reset devices, kill unknown processes, or assume eight cards/sixteen dies can run independent experiments. Stop on an identity/configuration mismatch or the first complete failure; retain the logs rather than adding speculative workarounds. Graph, overlap, realtime, TP, quantization and high concurrency are outside this packet.

The question is whether the existing SGLang Ascend draft/verify path preserves the ASR hidden-state, checkpoint attention semantics, reduced-vocabulary mapping and accepted-token/KV contracts when admitted by Omni. CPU dispatch tests cannot answer this. Reuse the existing worker; if a dependency fails, identify its exact code path before proposing a separate upstream fix.

## Freeze the environment and weights

Record clean/dirty state, Omni commit, imported SGLang version and file paths plus source commit or package build identity, torch/torch_npu/CANN/sgl_kernel_npu/triton-ascend versions, image digest, device model and logical-device/die assignment. The repository's supported SGLang release and image pins are in `pyproject_npu.toml`; a matching version string alone does not prove matching source or operators. Confirm the imported worker has the Ascend draft preparation and verify path, `ascend` is an allowed draft attention backend, and the installed NPU cache-location operator exists. Do not install or replace the stack implicitly.

Reuse a complete local target snapshot of `Qwen/Qwen3-ASR-1.7B` at revision `7278e1e70fe206f11671096ffdd38061171dd6e5`. The separate draft is `wilsonzheng0327/Qwen3-ASR-1.7B-dflash` at the revision pinned in `examples/configs/qwen3_asr_1_7b_dflash_npu.yaml`. Its safetensors file is about 726 MB. Preserve its `config.json`, `model.safetensors` and provenance together; do not substitute the 0.6B draft or another publisher's export. No training or NPU-specific tensor conversion is part of this integration. The temporary flattened config links the original weights and is released after engine construction.

If absent, download the snapshots during an explicitly allocated preparation window (not during measurement). With an existing Hugging Face environment:

```python
from huggingface_hub import snapshot_download

target = snapshot_download(
    "Qwen/Qwen3-ASR-1.7B",
    revision="7278e1e70fe206f11671096ffdd38061171dd6e5",
)
draft = snapshot_download(
    "wilsonzheng0327/Qwen3-ASR-1.7B-dflash",
    revision="d62bd4051e68282e02f93de774036a35f22f1373",
)
print(target, draft)
```

Keep private paths on the server. For offline execution use these local snapshot paths as `--model-path` and `--asr.engine.speculative_draft_model_path`, with `HF_HUB_OFFLINE=1`. Record the revisions and hashes in returned evidence, not private paths.

## Local/native tests before serving

```bash
python -m pytest -q \
  tests/unit_test/speculative/test_dflash_draft.py \
  tests/unit_test/speculative/test_target_worker.py \
  tests/unit_test/qwen3_asr/test_pipeline.py \
  tests/unit_test/qwen3_asr/test_sglang_model.py \
  tests/unit_test/config/test_example_configs.py
```

Retain full collection and failures. Separate missing optional dependencies and reproduced baseline failures from candidate regressions. Local Windows checks were source-isolated only; do not treat them as this native gate passing.

## Paired eager correctness

Use fresh processes and logs sequentially on the same assigned device. Device visibility must be set using the isolated environment's supported Ascend launcher; record the actual assigned device and effective `device=npu`, rather than assuming CUDA visibility variables control NPU placement. For direct factory construction, set `PER_STREAM_QUEUE=1`, leave `TASK_QUEUE_ENABLE` unset or `1`, and leave `ASCEND_LAUNCH_BLOCKING` unset or `0` before importing torch_npu. The normal pipeline launcher supplies the queue default.

Prepare the baseline from the candidate example, removing only its speculative engine fields and disabling async decode to match the candidate's synchronous scheduling:

```python
from pathlib import Path
import yaml

config = yaml.safe_load(Path("examples/configs/qwen3_asr_1_7b_dflash_npu.yaml").read_text())
config["name"] = "qwen3-asr-npu-eager-baseline"
stage = config["stages"]["asr"]
stage["factory"]["enable_async_decode"] = False
stage["engine"] = {
    key: value for key, value in stage["engine"].items()
    if not key.startswith("speculative_")
}
Path("qwen3_asr_npu_eager_baseline.yaml").write_text(yaml.safe_dump(config))
```

Run each config with the same local target, port, inputs and resource budget; the draft command additionally names its local draft snapshot:

```bash
sgl-omni serve --config qwen3_asr_npu_eager_baseline.yaml \
  --model-path "$TARGET_SNAPSHOT" --port 8000

# After stopping only the owned baseline server and confirming device health:
sgl-omni serve --config examples/configs/qwen3_asr_1_7b_dflash_npu.yaml \
  --model-path "$TARGET_SNAPSHOT" \
  --asr.engine.speculative_draft_model_path "$DRAFT_SNAPSHOT" --port 8000
```

Confirm effective parsed configuration: one device, TP=1, BF16, target/draft attention `ascend`, block size 8, `temperature=0`, max running requests 1, encoder/prefill/decode graphs disabled, Torch Compile disabled, async decode/prefill and overlap disabled on both sides. The baseline has no draft worker; the candidate must load `ReducedVocabularyDFlashWorker` and its own trained head/mapping. The pinned 48k-vocabulary draft has `sliding_window_non_causal=false` and `sliding_window=32768`: retain causal attention after conversion (an omitted `is_causal` uses the causal default). Only a checkpoint declaring non-causal windows should be converted to `is_causal=false`. Bound effective `context_length` to 32,768 or less, including prefix and verification reserve, and inspect the installed Ascend path's actual sliding-window handling. This scope does not qualify sliding-window eviction or longer contexts. Record selected attention implementations and fallbacks. Startup timeout: 15 minutes; per-request timeout/no-progress limit: 120 seconds. Stop after the first timeout/device error or silent path fallback.

Use the SeedTTS source revision and deterministic 16 EN + 16 ZH selection described in PR #3462: dataset `zhaochenyang20/seed-tts-eval` at `8f5e1aa2a35d42f42e940074c1983358b9491f89`, sort each language's `prompt-wavs` by `(file size, path)` and select `round(i * (N - 1) / 15)` for `i=0..15`. Store the selected IDs, audio SHA-256 and reference text. Submit one request at a time to `/v1/audio/transcriptions` with `temperature=0` and `response_format=json`. Run baseline A1, candidate B, then baseline A2 in fresh servers to expose native greedy drift.

For each request retain transcript, final token IDs, finish reason, verify-round count, correct draft-token count and accepted commit lengths including bonus. The public transcript response alone is insufficient: use a separately identified observation-only logging patch if needed and retain its exact diff/hash. Record that instrumentation on every compared server; do not change sampling, attention, scheduling or acceptance. A positive target-path trace must show draft proposal, target verification, consumption of the target auxiliary hidden states and advancement by the returned commit lengths. Check EOS/max-token truncation does not emit padded or unaccepted tokens and rejected draft slots are not committed. Never equate speculative counters with accuracy.

Pass this gate only if every request completes without fallback/error, no output is empty for nonempty reference speech, the native handoff invariants hold, and candidate final token IDs and finish reasons match the baseline. If A1/A2 differ or candidate mismatches, retain all per-request evidence and investigate numerical/implementation ownership; do not waive differences from an aggregate WER alone. Report corpus WER/CER as additional context. Acceptance length is `(sum(correct draft tokens) + sum(verify rounds)) / sum(verify rounds)` and includes the bonus; preserve the raw counters and do not compare eager and graph numbers as one population.

After that gate passes, run 100 serial candidate requests cycling the same inputs. Require no timeout, ACL error, KV allocator exhaustion, stale hidden-state reuse, missing/duplicated tokens or growth in outstanding requests. Check temporary draft directories are absent after construction, the original snapshots remain, shutdown succeeds and owned processes release the device. Do not claim cancellation, output-budget boundaries or KV-pool exhaustion qualification from this serial run.

## Return and next decision

Return: instruction revision; code/image/dependency/weight/input identities; gates actually executed; effective config; positive path traces; per-request output/counter comparison and WER/CER; first failure and stack; health/cleanup; observation-patch identity; evidence IDs retained on server; and what remains unqualified. Redact host details and private paths. Classify failures as environment, baseline, Omni integration or SGLang/operator before any fix. No site-packages patching or automatic dependency upgrades.

A complete eager pass permits planning paired performance measurements. It does not establish a speedup, graph safety, realtime support, cancellation safety or high-concurrency stability. Do not run the CUDA-only speculative sweep driver on NPU or extrapolate its GPU results. Preserve the baseline and stop here for review before expanding hardware scope.
