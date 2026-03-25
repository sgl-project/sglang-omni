# Plan: S2 Pro TTS Benchmark CI

## 1. Design Principle

**`benchmarks/` = the only execution logic. `tests/` = orchestration + threshold assertion.**

The test file does NOT import any functions from the benchmark script. Instead, it calls `python -m benchmarks.performance.tts.benchmark_tts_speed ...` as a subprocess — exactly the same CLI command a developer would run manually. The test file's sole responsibilities are:

1. Managing server and dataset lifecycle (fixtures)
2. Invoking the benchmark CLI (subprocess)
3. Reading JSON results and asserting thresholds

No abstraction layers. No duplicate code.

---

## 2. Architecture

### File: `tests/test_model/test_s2pro_benchmark.py`

```
pytest fixtures (module scope)
├── dataset_dir        → download seed-tts-eval, yield path, cleanup on teardown
├── server_process     → start s2-pro server, wait healthy, yield, SIGTERM on teardown
│
test_voice_cloning_non_streaming(server_process, dataset_dir)
  └─ subprocess: python -m benchmarks.performance.tts.benchmark_tts_speed \
       --model fishaudio/s2-pro --port 18898 \
       --testset {dataset_dir}/en/meta.lst --max-samples 10 \
       --output-dir {tmpdir}/vc_nonstream
  └─ read {tmpdir}/vc_nonstream/speed_results.json → assert thresholds

test_voice_cloning_streaming(server_process, dataset_dir)
  └─ same + --stream

test_plain_tts_non_streaming(server_process, dataset_dir)
  └─ same + --no-ref-audio

test_plain_tts_streaming(server_process, dataset_dir)
  └─ same + --no-ref-audio --stream
```

### Server Lifecycle

Follow `test_video_integration.py` pattern exactly:

1. `subprocess.Popen` with `preexec_fn=os.setsid` (process group for clean kill)
2. Poll `/health` endpoint every 1s with timeout (600s)
3. On teardown: `SIGTERM` → wait 30s → `SIGKILL` fallback

Server command:
```bash
python -m sglang_omni.cli.cli serve \
    --model-path fishaudio/s2-pro \
    --config examples/configs/s2pro_tts.yaml \
    --port 18898
```

### Dataset Lifecycle

1. `gdown 1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP -O /tmp/seed-tts-eval.tar`
2. `tar xf /tmp/seed-tts-eval.tar -C /tmp/`
3. Yield `/tmp/seedtts_testset`
4. On teardown: `shutil.rmtree(/tmp/seedtts_testset)` + remove tar

### Test Execution

Each test function:
1. Builds the CLI command with appropriate flags
2. Runs `subprocess.run(cmd, check=True)` — if benchmark crashes, test fails immediately
3. Reads `speed_results.json` from the output dir
4. Extracts `summary` dict, asserts metrics against thresholds
5. Prints `[PERF]` markers for CI log readability

---

## 3. Performance Thresholds

Derived from issue #193 comment (2 runs each, observed ~5% variance):

| Mode | Metric | Observed Range | Threshold | Regression Value |
|---|---|---|---|---|
| **Voice cloning, non-streaming** | `tok_per_s_agg` | 85.9 – 86.9 | **≥ 55** | 27.8 (caught) |
| | `rtf_mean` | 2.18 – 2.28 | **≤ 4.0** | 2.55 (caught) |
| **Voice cloning, streaming** | `latency_mean_s` | 10.0 – 10.01 | **≤ 15.0** | — |
| | `throughput_qps` | 0.1 | **≥ 0.06** | — |
| **Plain TTS, non-streaming** | `tok_per_s_agg` | 87.0 – 87.5 | **≥ 55** | — |
| | `rtf_mean` | 0.27 | **≤ 0.6** | — |
| **Plain TTS, streaming** | `latency_mean_s` | 3.25 – 3.27 | **≤ 5.5** | — |
| | `throughput_qps` | 0.31 | **≥ 0.15** | — |

**Design rationale:**
- Thresholds are set at ~60-70% of observed values (generous lower bound)
- The known regression (tok/s: 88 → 28) is well below the 55 threshold
- The 5% run-to-run variance is safely within the ~35% margin
- `tok_per_s_agg` is the primary regression detector for non-streaming modes

---

## 4. CI Integration

The test file goes in `tests/test_model/`, so `pytest tests/ -v` in `.github/workflows/test.yaml` automatically picks it up. No workflow changes needed.

**Concern:** This test adds ~5-8 minutes (server startup + 4 modes × 10 samples). Combined with the existing video integration test, total CI time may approach the 30-minute timeout. If this becomes an issue, options include:
- Splitting into a separate CI job
- Adding a `@pytest.mark.benchmark` marker and filtering
- Reducing `--max-samples` to 5

For now, keep it simple — let it run within the existing `unit-test` job.

---

## 5. Implementation Checklist

- [ ] Create `tests/test_model/test_s2pro_benchmark.py`
  - [ ] Module-scoped fixture: dataset download + cleanup
  - [ ] Module-scoped fixture: server start + health check + teardown
  - [ ] Helper: `_run_benchmark(port, testset, output_dir, extra_args) -> dict` (subprocess + read JSON)
  - [ ] `test_voice_cloning_non_streaming` — assert tok/s ≥ 55, RTF ≤ 4.0
  - [ ] `test_voice_cloning_streaming` — assert latency ≤ 15.0, throughput ≥ 0.06
  - [ ] `test_plain_tts_non_streaming` — assert tok/s ≥ 55, RTF ≤ 0.6
  - [ ] `test_plain_tts_streaming` — assert latency ≤ 5.5, throughput ≥ 0.15
- [ ] Ensure `gdown` is available in CI (check if it's in `.[dev]` deps, add if not)
- [ ] Verify `examples/configs/s2pro_tts.yaml` exists and is correct
- [ ] Run locally to validate thresholds pass
