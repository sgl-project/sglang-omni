# Plan: S2 Pro TTS Benchmark CI

## Design Principle

**`benchmarks/` = the only execution logic. `tests/` = orchestration + threshold assertion.**

The test file does NOT import any functions from the benchmark script. It calls
`python -m benchmarks.performance.tts.benchmark_tts_speed ...` as a subprocess —
the same CLI command a developer would run manually.

---

## Architecture

### File: `tests/test_model/test_s2pro_benchmark.py`

```
pytest fixtures (module scope)
├── dataset_dir        → huggingface_hub.snapshot_download("zhaochenyang20/seed-tts-eval-mini")
│                        uses tmp_path_factory, cleanup automatic
├── server_process     → Popen + setsid + health poll + SIGTERM/SIGKILL teardown
│
@pytest.mark.benchmark
test_voice_cloning_non_streaming(server_process, dataset_dir)
  └─ subprocess: python -m benchmarks.performance.tts.benchmark_tts_speed \
       --model fishaudio/s2-pro --port 18898 \
       --testset {dataset_dir}/en/meta.lst --max-samples 10 \
       --output-dir {tmpdir}/vc_nonstream
  └─ read speed_results.json → assert thresholds

test_voice_cloning_streaming        (+ --stream)
test_plain_tts_non_streaming        (+ --no-ref-audio)
test_plain_tts_streaming            (+ --no-ref-audio --stream)
```

### Dataset

- `huggingface_hub.snapshot_download("zhaochenyang20/seed-tts-eval-mini", repo_type="dataset")`
- Mini dataset: 10 samples + `en/meta.lst` only
- Uses `tmp_path_factory` (module scope) — no `/tmp/` hardcoding
- No gdown dependency

### Server Lifecycle

Follow `test_video_integration.py` pattern:

1. `subprocess.Popen` with `preexec_fn=os.setsid`
2. Poll `/health` every 1s, timeout 600s
3. Teardown: `SIGTERM` → wait 30s → `SIGKILL`

```bash
python -m sglang_omni.cli.cli serve \
    --model-path fishaudio/s2-pro \
    --config examples/configs/s2pro_tts.yaml \
    --port 18898
```

4 tests share one module-scoped server (s2-pro supports both streaming and non-streaming).

### Thresholds (tightened, 15-25% margin from 4-run data)

| Mode | Metric | 4-Run Range | Threshold |
|---|---|---|---|
| VC non-streaming | tok_per_s_agg | 85.8 – 87.1 | **>= 80** |
| | rtf_mean | 2.18 – 2.37 | **<= 2.8** |
| VC streaming | latency_mean_s | 10.0 – 10.73 | **<= 12.5** |
| | throughput_qps | 0.093 – 0.10 | **>= 0.08** |
| Plain non-streaming | tok_per_s_agg | 87.0 – 87.5 | **>= 80** |
| | rtf_mean | 0.269 – 0.273 | **<= 0.35** |
| Plain streaming | latency_mean_s | 3.25 – 3.29 | **<= 4.0** |
| | throughput_qps | 0.304 – 0.308 | **>= 0.25** |

### CI Marker

- All tests decorated with `@pytest.mark.benchmark`
- Register marker in `pyproject.toml` under `[tool.pytest.ini_options]`

---

## Implementation Checklist

- [ ] Register `benchmark` marker in `pyproject.toml`
- [ ] Create `tests/test_model/test_s2pro_benchmark.py`
  - [ ] Module-scoped fixture: dataset download via huggingface_hub
  - [ ] Module-scoped fixture: server start + health check + teardown
  - [ ] Helper: `_run_benchmark(port, testset, output_dir, extra_args) -> dict`
  - [ ] `test_voice_cloning_non_streaming` — tok/s >= 80, RTF <= 2.8
  - [ ] `test_voice_cloning_streaming` — latency <= 12.5, throughput >= 0.08
  - [ ] `test_plain_tts_non_streaming` — tok/s >= 80, RTF <= 0.35
  - [ ] `test_plain_tts_streaming` — latency <= 4.0, throughput >= 0.25
