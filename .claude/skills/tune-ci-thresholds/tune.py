#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Observe CI metrics over N runs on the H20 repro host.
Emits worst-of-N markdown. Does NOT propose thresholds or edit tests.
Model-agnostic: pass --model <name>; config comes from
models/<name>/config.yaml. Metrics come from result JSONs that tests
already write under pytest's --basetemp (set fresh per run).
"""
from __future__ import annotations
import argparse, ast, datetime as dt, hashlib, json, math, os, re, shutil, signal
import subprocess, sys, time, tomllib
from pathlib import Path

__version__ = "0.4.3"

SKILL_DIR = Path(__file__).resolve().parent
MODELS_DIR = SKILL_DIR / "models"
DEFAULT_MODEL = "qwen3-omni-v1"
REPO_ROOT = Path("/sgl-workspace/sglang-omni")
if not REPO_ROOT.exists():
    REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ROOT = Path("/github/home/ci-threshold-runs")
RETRY_SIGS = ("OOM", "exit 137", "exit 139", "TimeoutExpired")
# Per-GPU memory must be strictly below this (MiB) before any pytest/server
# restart. 2048 MiB = 2 GiB. Do not launch on stale 17 GiB contexts.
_GPU_RETRY_MEM_MIB = 2048
_GPU_LAUNCH_RECHECK_S = 3
_GPU_WAIT_POLL_S = 5
_GPU_WAIT_TIMEOUT_S = 600
_PYTEST_POLL_S = 30
_MAX_RUN_ATTEMPTS = 4
_DEFAULT_CALIBRATION_PASSES = 10
_AGENT_POLL_INTERVAL_S = 120
_CI_HOME = Path("/github/home")
_CRASH_SIGS = (
    "Fatal Python error",
    "Segmentation fault",
    "CUDA error: an illegal memory access",
    "Child process died",
    "All workers failed",
    "Router failed to start",
    "Address already in use",
    "Connection refused",
    "Server process exited",
    "worker crashed",
)


def _flashinfer_cache_dirs(env: dict[str, str] | None = None) -> list[Path]:
    env = env or os.environ
    candidates = [
        Path(env.get("XDG_CACHE_HOME", "")) / "flashinfer"
        if env.get("XDG_CACHE_HOME")
        else None,
        Path(env.get("HOME", "")) / ".cache" / "flashinfer"
        if env.get("HOME")
        else None,
        _CI_HOME / ".cache" / "flashinfer",
    ]
    seen: set[Path] = set()
    paths: list[Path] = []
    for candidate in candidates:
        if candidate is None:
            continue
        path = candidate.expanduser()
        if path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def _cleanup_flashinfer_cache(env: dict[str, str] | None = None) -> None:
    # Runtime JIT dirs only — never remove the host wheel cache under
    # /github/home/.cache/flashinfer-jit-cache/ (see install_flashinfer_jit_cache.sh).
    for cache_dir in _flashinfer_cache_dirs(env):
        shutil.rmtree(cache_dir, ignore_errors=True)


def _venv_has_flashinfer_jit_cache(py: str) -> bool:
    r = subprocess.run(
        [
            py,
            "-c",
            "import importlib.util, sys;"
            "sys.exit(0 if importlib.util.find_spec('flashinfer_jit_cache') else 1)",
        ],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0

# Metric registry. Each entry encodes how a named metric should be
# displayed in the report and which stage group it belongs to. Scales
# assume the metric is read from the result JSON in its native unit
# (fractions 0-1 for accuracy/WER, raw for throughput/latency/RTF).
METRIC_SPECS = {
    "accuracy":             dict(worst="min", label="Acc (%)",               digits=2, scale=100, group="accuracy"),
    "corpus_wer":           dict(worst="max", label="Corpus WER (%)",        digits=2, scale=100, group="wer"),
    "per_sample_wer_max":   dict(worst="max", label="Max per-sample WER (%)", digits=2, scale=100, group="wer"),
    "wer_below_50_corpus":  dict(worst="max", label="Corpus WER ≤50% (%)", digits=2, scale=100, group="wer"),
    "n_above_50":           dict(worst="max", label="Samples >50% WER",      digits=0, scale=1,   group="wer"),
    "throughput_qps":       dict(worst="min", label="Throughput (req/s)",    digits=3, scale=1,   group="speed"),
    "output_tok_per_req_s": dict(
        worst="min", label="Output tok/req-s", digits=1, scale=1, group="speed"
    ),
    "latency_mean_s":       dict(worst="max", label="Latency mean (s)",      digits=3, scale=1,   group="speed"),
    "latency_p95_s":        dict(worst="max", label="Latency p95 (s)",       digits=3, scale=1,   group="speed"),
    "rtf_mean":             dict(worst="max", label="RTF mean",              digits=4, scale=1,   group="speed"),
    "rtf_p95":              dict(worst="max", label="RTF p95",               digits=4, scale=1,   group="speed"),
    "failed_requests":      dict(worst="max", label="Failed requests",       digits=0, scale=1,   group="reliability"),
    "similarity_mean":      dict(worst="min", label="Speaker sim mean",      digits=4, scale=1,   group="similarity"),
}
_NESTED = {
    "throughput_qps",
    "output_tok_per_req_s",
    "latency_mean_s",
    "latency_p95_s",
    "rtf_mean",
    "rtf_p95",
}

_DEFAULT_METRIC_PATHS = {
    "accuracy": "summary.accuracy",
    "corpus_wer": "wer.summary.wer_corpus",
    "per_sample_wer_max": "wer.summary.wer_per_sample_max",
    "wer_below_50_corpus": "wer.summary.wer_below_50_corpus",
    "n_above_50": "wer.summary.n_above_50_pct_wer",
    "throughput_qps": "speed.throughput_qps",
    "tok_per_s_agg": "speed.tok_per_s_agg",
    "latency_mean_s": "speed.latency_mean_s",
    "latency_p95_s": "speed.latency_p95_s",
    "rtf_mean": "speed.rtf_mean",
}
_DEFAULT_SAMPLE_COUNTS = {
    "total": "summary.total_samples",
    "ok": "speed.completed_requests",
}
_BENCHMARK_PATH_OVERRIDES = {
    "benchmark_omni_mmsu": {
        "paths": {
            "accuracy": "summary.overall_accuracy",
            "throughput_qps": "speed_metrics.throughput_qps",
            "tok_per_s_agg": "speed_metrics.tok_per_s_agg",
            "latency_mean_s": "speed_metrics.latency_mean_s",
            "latency_p95_s": "speed_metrics.latency_p95_s",
            "rtf_mean": "speed_metrics.rtf_mean",
        },
        "sample_counts": {"ok": "summary.successful_samples"},
    },
}

# test_tts_ci.py sample-scope presets — fixed by CI design, never apply targets.
_TTS_FIXED_PRESETS = frozenset({
    "SEEDTTS_EN_FULLSET_SAMPLES",
    "TTS_SIMILARITY_MAX_SAMPLES",
    "STREAMING_BENCHMARK_MAX_SAMPLES",
})


def match_metric(name, nested):
    if nested is not None:
        return nested if nested in _NESTED else None
    if name in _TTS_FIXED_PRESETS:
        return None
    if re.fullmatch(r".*_ACC(?:URACY)?_MIN", name) or re.fullmatch(r".*_MIN_ACCURACY", name):
        return "accuracy"
    # Partitioned WER (wer_below_50_corpus + n_above_50) must precede the
    # generic "WER_MAX_CORPUS" substring check to avoid false hits.
    if "WER_BELOW_50_CORPUS_MAX" in name: return "wer_below_50_corpus"
    if "N_ABOVE_50_MAX" in name: return "n_above_50"
    if name == "TTS_MAX_FAILED_REQUESTS" or "MAX_FAILED_REQUESTS" in name:
        return "failed_requests"
    if "SIMILARITY" in name and name.endswith("_MIN"):
        return "similarity_mean"
    if "WER_MAX_CORPUS" in name: return "corpus_wer"
    if "WER_MAX_PER_SAMPLE" in name: return "per_sample_wer_max"
    if "CORPUS_WER_MAX" in name: return "corpus_wer"
    if "SAMPLE_WER_MAX" in name: return "per_sample_wer_max"
    if "THROUGHPUT_MIN" in name: return "throughput_qps"
    if "LATENCY_MEAN_MAX" in name: return "latency_mean_s"
    if "LATENCY_P95_MAX" in name: return "latency_p95_s"
    if "RTF_MEAN_MAX" in name: return "rtf_mean"
    if "RTF_P95_MAX" in name: return "rtf_p95"
    return None


def now_iso(): return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def ts_dir():  return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
def sha256(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def available_models():
    if not MODELS_DIR.exists(): return []
    return sorted(d.name for d in MODELS_DIR.iterdir()
                  if d.is_dir() and (d / "config.yaml").exists())


def model_dir(name):  return MODELS_DIR / name
def stages_path(name): return model_dir(name) / "stages.yaml"


def _merge_omni_ci_home(cfg: dict) -> None:
    omni_home = cfg.get("omni_ci_home")
    if not omni_home:
        return
    auto = cfg.setdefault("auto_env", {})
    auto.setdefault("OMNI_CI_HOME", omni_home)
    auto.setdefault("XDG_CACHE_HOME", f"{omni_home}/.cache")
    auto.setdefault("TORCHINDUCTOR_CACHE_DIR", f"{omni_home}/.torchinductor")


def _apply_auto_env(cfg: dict) -> None:
    _merge_omni_ci_home(cfg)
    for key, value in cfg.setdefault("auto_env", {}).items():
        os.environ[key] = str(value)


def load_model_config(name):
    p = model_dir(name) / "config.yaml"
    if not p.exists():
        raise SystemExit(f"model config not found: {p}. "
                         f"Known models: {available_models()}")
    cfg = _load_yaml(p, top_is_dict=True)
    cfg.setdefault("extra_env", {})
    cfg.setdefault("auto_env", {})
    cfg.setdefault("metric_sources", {})
    cfg.setdefault("hf_model_ids_by_test", {})
    # Auto-apply env vars so the user doesn't need to export them
    # manually. Overrides any pre-existing value to match CI.
    _apply_auto_env(cfg)
    return cfg


def resolve_venv(flag, cfg):
    """Return (chosen_python, source, tried).

    `tried` is the ordered list of default candidates considered when
    `source == "default"` and is empty otherwise. precheck uses it to
    report every candidate path it looked at when none exist.
    """
    if flag: return flag, "flag", []
    if os.environ.get("TUNE_VENV_PYTHON"):
        return os.environ["TUNE_VENV_PYTHON"], "env", []
    default = cfg["default_venv_python"]
    # Accept either a single path (str) or an ordered list of candidate
    # paths. For a list, return the first one that exists; if none exist,
    # return the last entry so existing call sites still get a path.
    if isinstance(default, list):
        if not default:
            raise RuntimeError("default_venv_python list is empty in config.yaml")
        for p in default:
            if Path(p).exists():
                return p, "default", list(default)
        return default[-1], "default", list(default)
    return default, "default", [default]


def read_pins():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    pins = {}
    for dep in data["project"]["dependencies"]:
        m = re.match(r'^\s*"?([A-Za-z0-9_\-]+)"?\s*==\s*([^\s,"]+)', dep)
        if m: pins[m.group(1).lower()] = m.group(2)
    return pins


def venv_version(py, mod):
    r = subprocess.run([py, "-c", f"import {mod};print({mod}.__version__)"],
                       capture_output=True, text=True, timeout=60)
    if r.returncode: raise RuntimeError(f"{mod} version read failed: {r.stderr.strip()}")
    return r.stdout.strip()


def git_info():
    q = lambda c: subprocess.run(c, cwd=REPO_ROOT, capture_output=True,
                                  text=True, check=False).stdout.strip()
    return dict(sha=q(["git", "rev-parse", "HEAD"]),
                branch=q(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
                dirty=bool(q(["git", "status", "--porcelain"])))


def _unique_ordered(items):
    seen, out = set(), []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _all_model_ids(cfg):
    model_ids = [cfg["hf_model_id"]]
    for ids in (cfg.get("hf_model_ids_by_test") or {}).values():
        model_ids.extend(ids or [])
    return _unique_ordered(model_ids)


def _required_model_ids_for_tests(cfg, test_names):
    by_test = cfg.get("hf_model_ids_by_test") or {}
    model_ids = []
    for test_name in sorted(set(test_names)):
        model_ids.extend(by_test.get(test_name) or [cfg["hf_model_id"]])
    return _unique_ordered(model_ids)


def nvidia_smi_L():
    return subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                          text=True, check=False).stdout.strip()


def gpu_summary(smi):
    lines = [ln for ln in smi.splitlines() if ln.strip()]
    if not lines: return "unknown"
    m = re.search(r":\s*(.*?)\s*\(", lines[0])
    return f"{len(lines)}× {m.group(1) if m else 'unknown'}"


def _smi_lines(cols):
    r = subprocess.run(
        ["nvidia-smi", f"--query-gpu={cols}" if "=" not in cols else cols,
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False)
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]


def _gpu_memory_by_index():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False)
    out = {}
    for ln in r.stdout.splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
            mem_m = re.search(r"(\d+)", parts[1])
            if mem_m:
                out[idx] = int(mem_m.group(1))
        except ValueError:
            continue
    return out


def _gpu_memory_used_mib():
    return list(_gpu_memory_by_index().values())


def _kill_calibration_gpu_processes():
    """Match CI omni-post-stage cleanup between calibration runs."""
    script = REPO_ROOT / ".github/scripts/delete_gpu_process.sh"
    if script.exists():
        subprocess.run(["bash", str(script)], capture_output=True, check=False)
    for pattern in (
        "sgl-omni serve",
        "sglang_omni_router.serve",
        "stage_process",
        "pytest tests/test_model",
    ):
        subprocess.run(["pkill", "-9", "-f", pattern], check=False)


def _picked_gpus_mem_snapshot(picked: list[int]) -> dict[int, int]:
    mem = _gpu_memory_by_index()
    return {i: mem.get(i, -1) for i in picked}


def _picked_gpus_under_limit(picked: list[int]) -> bool:
    """True iff every picked GPU has no compute app and memory < 2 GiB."""
    if not picked:
        return False
    mem = _gpu_memory_by_index()
    busy = busy_gpu_indices()
    return all(
        i not in busy and mem.get(i, 99999) <= _GPU_RETRY_MEM_MIB
        for i in picked
    )


def _ready_gpu_indices(gpus_needed: int):
    """GPU indices with no compute app and memory <= _GPU_RETRY_MEM_MIB."""
    mem = _gpu_memory_by_index()
    busy = busy_gpu_indices()
    ready = [
        idx for idx in sorted(mem)
        if idx not in busy and mem[idx] <= _GPU_RETRY_MEM_MIB
    ]
    return ready, mem, busy


def _ensure_gpus_free(gpus_needed: int, timeout: int = _GPU_WAIT_TIMEOUT_S) -> bool:
    """Kill stale processes and wait until >= gpus_needed GPUs are each < 2 GiB."""
    _kill_calibration_gpu_processes()
    time.sleep(3)
    waited = 0
    last_log = -30
    while waited < timeout:
        ready, mem, busy = _ready_gpu_indices(gpus_needed)
        if len(ready) >= gpus_needed:
            picked_mem = {i: mem[i] for i in ready[:gpus_needed]}
            print(f"  GPU ready for launch {picked_mem} MiB (each <= "
                  f"{_GPU_RETRY_MEM_MIB}) after {waited}s")
            return True
        if waited - last_log >= 30:
            print(f"  waiting: need {gpus_needed} GPU(s) each <= "
                  f"{_GPU_RETRY_MEM_MIB} MiB ({waited}s/{timeout}s): "
                  f"mem={mem} busy={sorted(busy)} ready={len(ready)}")
            last_log = waited
        time.sleep(_GPU_WAIT_POLL_S)
        waited += _GPU_WAIT_POLL_S
        if waited % 60 == 0:
            _kill_calibration_gpu_processes()
    subprocess.run(["pkill", "-9", "-f", "sgl-omni"], check=False)
    time.sleep(5)
    ready, mem, busy = _ready_gpu_indices(gpus_needed)
    if len(ready) >= gpus_needed:
        print(f"  GPU ready after forced pkill: "
              f"{ {i: mem[i] for i in ready[:gpus_needed]} } MiB")
        return True
    print(f"  error: cannot launch — need {gpus_needed} GPU(s) each <= "
          f"{_GPU_RETRY_MEM_MIB} MiB; after {timeout}s mem={mem} "
          f"busy={sorted(busy)} ready={len(ready)}")
    return False


def _pick_gpus_for_launch(gpus_needed: int, label: str) -> tuple[list[int] | None, str]:
    """Select GPUs only after _ensure_gpus_free; abort if any picked GPU >= 2 GiB."""
    if not _ensure_gpus_free(gpus_needed):
        return None, "GPU memory not released after cleanup"
    picked, err = pick_free_gpus(gpus_needed)
    if picked is None:
        if not _ensure_gpus_free(gpus_needed):
            return None, err or "GPU memory not released after cleanup"
        picked, err = pick_free_gpus(gpus_needed)
    if picked is None:
        return None, err or "no GPU under 2 GiB memory limit"
    if not _picked_gpus_under_limit(picked):
        snap = _picked_gpus_mem_snapshot(picked)
        return None, f"picked GPUs not under {_GPU_RETRY_MEM_MIB} MiB: {snap}"
    return picked, ""


def _launch_gpu_gate(picked: list[int], gpus_needed: int, label: str) -> tuple[list[int] | None, str]:
    """Hard gate immediately before pytest Popen — recheck memory after brief pause."""
    time.sleep(_GPU_LAUNCH_RECHECK_S)
    if _picked_gpus_under_limit(picked):
        snap = _picked_gpus_mem_snapshot(picked)
        print(f"{label} launch gate: GPU mem={snap} MiB (each <= "
              f"{_GPU_RETRY_MEM_MIB}) — starting pytest")
        return picked, ""
    snap = _picked_gpus_mem_snapshot(picked)
    print(f"{label} launch gate BLOCKED: GPU mem={snap} MiB — "
          f"releasing before restart")
    return _pick_gpus_for_launch(gpus_needed, label)


def _stage_metrics_complete(stage, metrics):
    """True when every metric with a json_path has a non-None value."""
    stage_metrics = stage.get("metrics") or {}
    if not stage_metrics:
        return False
    for _mk, meta in stage_metrics.items():
        if meta.get("json_path") and metrics.get(_mk) is None:
            return False
    return True


def _observation_status(stage, metrics, pytest_status, pytest_reason):
    """Calibration treats a run as complete once metrics are extracted."""
    if not stage.get("metrics"):
        return pytest_status, pytest_reason
    if not _stage_metrics_complete(stage, metrics):
        return "failed", "incomplete_metrics_extraction"
    if pytest_status == "ok":
        return "ok", ""
    return "ok", f"threshold_assertion ({pytest_reason})"


def _should_halt_on_extraction(
    stage_keys,
    all_stages,
    metrics_by_stage,
    extraction_warnings,
    pytest_rc: int,
) -> bool:
    """True when pytest passed but threshold metrics could not be extracted."""
    if pytest_rc != 0:
        return False
    for warning in extraction_warnings:
        if "file missing" in warning or "read failed" in warning:
            return True
    for sk in stage_keys:
        stage = all_stages[sk]
        metrics = metrics_by_stage.get(sk) or {}
        if stage.get("metrics") and not _stage_metrics_complete(stage, metrics):
            return True
    return False


def _run_json_ok(path: Path, stage=None) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if stage is not None and stage.get("metrics"):
        return _stage_metrics_complete(stage, data.get("metrics") or {})
    return data.get("status") == "ok"


def _purge_incomplete_run(out: Path, stage_keys, all_stages, k: int):
    for sk in stage_keys:
        stage = all_stages[sk]
        p = out / sk / f"run{k}.json"
        if p.exists() and not _run_json_ok(p, stage):
            p.unlink(missing_ok=True)


def audit_completeness(run_dir: Path, all_stages=None, plan=None):
    plan = plan or json.loads((run_dir / "plan.json").read_text())
    if all_stages is None:
        sy = Path(plan.get("stages_yaml")
                  or stages_path(plan.get("model", DEFAULT_MODEL)))
        all_stages = _load_yaml(sy)
    repeats = plan["repeats"]
    total = len(plan["stages"]) * repeats
    ok = 0
    missing = []
    for sk in plan["stages"]:
        stage = all_stages[sk]
        for k in range(1, repeats + 1):
            p = run_dir / sk / f"run{k}.json"
            if _run_json_ok(p, stage):
                ok += 1
            else:
                reason = "missing file"
                if p.exists():
                    try:
                        d = json.loads(p.read_text())
                        reason = d.get("reason") or d.get("status") or "incomplete metrics"
                    except (json.JSONDecodeError, OSError):
                        reason = "corrupt json"
                missing.append({"stage_key": sk, "run": k, "reason": reason})
    return dict(
        complete=(ok == total),
        ok=ok,
        total=total,
        missing_count=len(missing),
        missing=missing,
        repeats=repeats,
        model=plan.get("model"),
    )


def busy_gpu_indices():
    """GPU indices with any running compute app."""
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid",
        "--format=csv,noheader"], capture_output=True, text=True, check=False)
    busy = {ln.strip() for ln in r.stdout.splitlines() if ln.strip()}
    if not busy: return set()
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,gpu_uuid",
        "--format=csv,noheader"], capture_output=True, text=True, check=False)
    out = set()
    for ln in r.stdout.splitlines():
        if "," in ln:
            idx, uuid = [x.strip() for x in ln.split(",", 1)]
            if uuid in busy:
                try: out.add(int(idx))
                except ValueError: pass
    return out


def all_gpu_indices():
    r = subprocess.run(["nvidia-smi", "--query-gpu=index",
        "--format=csv,noheader"], capture_output=True, text=True, check=False)
    out = []
    for ln in r.stdout.splitlines():
        try: out.append(int(ln.strip()))
        except ValueError: pass
    return out


def pick_free_gpus(n):
    """Pick n GPUs with no compute app and memory <= _GPU_RETRY_MEM_MIB (2 GiB)."""
    ready, mem, busy = _ready_gpu_indices(n)
    all_idx = all_gpu_indices()
    if len(ready) >= n:
        picked = ready[:n]
        if not _picked_gpus_under_limit(picked):
            snap = _picked_gpus_mem_snapshot(picked)
            return None, f"internal: picked GPUs exceed {_GPU_RETRY_MEM_MIB} MiB: {snap}"
        return picked, None
    return None, (
        f"need {n} GPU(s) each <= {_GPU_RETRY_MEM_MIB} MiB (2 GiB); "
        f"ready {len(ready)}/{len(all_idx)} mem={mem} busy={sorted(busy)}"
    )


def precheck(py, src, out, skip_ver, cfg, datasets_override=None,
             model_ids_override=None, tried=None, gpu_required_override=None):
    errs, warns = [], []
    print(f"model: {cfg['name']}")
    print(f"venv_python: {py} ({src})")
    if src == "default" and tried and len(tried) > 1:
        print(f"  (tried in order: {', '.join(tried)})")
    # Container-name / hostname checks removed deliberately — they were
    # fragile heuristics. Equivalence with CI is established by: same
    # image's venv + pinned sglang/torch + cached assets + clean GPUs.
    if not (REPO_ROOT / "pyproject.toml").exists():
        errs.append(f"repo not found at {REPO_ROOT}")
        return _summary(errs, warns)
    gi = git_info()
    print(f"git: {gi['branch']} @ {gi['sha'][:8]}{' (dirty)' if gi['dirty'] else ''}")
    if not Path(py).exists():
        if src == "default" and tried and len(tried) > 1:
            errs.append(
                "venv python not found at any default candidate: "
                f"{', '.join(tried)} — set $TUNE_VENV_PYTHON or pass "
                "--venv-python <path>")
        else:
            errs.append(f"venv python missing: {py} — override via "
                        "--venv-python or $TUNE_VENV_PYTHON")
        return _summary(errs, warns)
    pins, versions = read_pins(), {}
    for pkg in ("sglang", "torch"):
        try: v = venv_version(py, pkg)
        except RuntimeError as e: errs.append(str(e)); continue
        versions[pkg] = v
        exp = pins.get(pkg)
        ok = (pkg == "torch" and exp and v.startswith(exp)) or \
             (pkg == "sglang" and exp and v == exp)
        print(f"  {pkg}: {v} (pin {exp}) [{'ok' if ok else 'mismatch'}]")
        if not ok:
            (warns if skip_ver else errs).append(
                f"{pkg} version mismatch: {v} vs pin {exp}")
    # Proxy env vars are left alone. Tests use disable_proxy() / no_proxy_env()
    # helpers in tests/utils.py for loopback calls, matching real CI.
    # HF_ENDPOINT and other config-declared env vars are auto-set by
    # load_model_config; print them for visibility.
    for k, v in cfg["auto_env"].items():
        print(f"  auto env: {k}={v}")
    if not _venv_has_flashinfer_jit_cache(py):
        venv_root = Path(py).resolve().parent.parent
        venv_name = venv_root.name
        errs.append(
            "flashinfer-jit-cache missing from venv (CI MoE graph capture needs it).\n"
            f"  From repo root with OMNI_CI_HOME={cfg['auto_env'].get('OMNI_CI_HOME', '<set>')}:\n"
            f"    ln -sfn {venv_root} ./{venv_name}\n"
            f"    bash .github/scripts/install_flashinfer_jit_cache.sh {venv_name}\n"
            "  Host wheel cache: /github/home/.cache/flashinfer-jit-cache/ (shared across PRs)."
        )
    # WER normalizer check removed — the user manages venv contents
    # explicitly and the warning was producing noise without changing
    # behavior. `expected_wer_normalizer` in config.yaml is now ignored.
    def _cached(repo_id, kind):
        if kind == "dataset":
            r = subprocess.run(
                [
                    py,
                    "-c",
                    "from huggingface_hub import snapshot_download;"
                    f"snapshot_download(repo_id={repo_id!r}, repo_type='dataset', "
                    "local_files_only=True)",
                ],
                capture_output=True,
                text=True,
            )
            return r.returncode == 0

        r = subprocess.run(
            [
                py,
                "-c",
                "from pathlib import Path\n"
                "from huggingface_hub import snapshot_download\n"
                f"repo_id = {repo_id!r}\n"
                "try:\n"
                "    snapshot = Path(snapshot_download("
                "repo_id=repo_id, local_files_only=True))\n"
                "except Exception:\n"
                "    raise SystemExit(1)\n"
                "weight_files = [\n"
                "    path for path in snapshot.rglob('*')\n"
                "    if path.is_file()\n"
                "    and path.suffix in ('.safetensors', '.bin')\n"
                "    and not path.name.endswith('.incomplete')\n"
                "]\n"
                "raise SystemExit(0 if weight_files else 1)",
            ],
            capture_output=True,
            text=True,
        )
        return r.returncode == 0
    # When called from `run` with a resolved stage selection, only the
    # model checkpoints and datasets those tests actually use are required;
    # others become "optional" (printed if cached, absent is not an error).
    model_ids_required = (_all_model_ids(cfg) if model_ids_override is None
                          else model_ids_override)
    model_ids_optional = [m for m in _all_model_ids(cfg)
                          if m not in model_ids_required]
    datasets_required = (cfg["hf_datasets"] if datasets_override is None
                         else datasets_override)
    datasets_optional = [d for d in cfg["hf_datasets"]
                         if d not in datasets_required]
    label = "assets (all datasets)" if datasets_override is None \
        else f"assets (required for selected stages, {len(datasets_required)} dataset(s))"
    print(f"  {label}:")
    missing = []  # list of (repo_id, kind) — only for required
    for repo_id, kind in [(m, "model") for m in model_ids_required] + \
                         [(ds, "dataset") for ds in datasets_required]:
        ok = _cached(repo_id, kind)
        mark = "✓" if ok else "✗"
        print(f"    {mark} {kind}: {repo_id}")
        if not ok:
            missing.append((repo_id, kind))
    if model_ids_optional:
        print("  other models (not needed for this run):")
        for model_id in model_ids_optional:
            ok = _cached(model_id, "model")
            mark = "✓" if ok else "·"
            print(f"    {mark} model: {model_id}")
    if datasets_optional:
        print("  other datasets (not needed for this run):")
        for ds in datasets_optional:
            ok = _cached(ds, "dataset")
            mark = "✓" if ok else "·"
            print(f"    {mark} dataset: {ds}")
    if missing:
        lines = [f"{len(missing)} asset(s) not cached locally. "
                 "Run these to download via the HF mirror:"]
        for repo_id, kind in missing:
            flag = " --repo-type dataset" if kind == "dataset" else ""
            lines.append(
                "  HF_ENDPOINT=https://hf-mirror.com "
                f"huggingface-cli download {repo_id}{flag}"
            )
        errs.append("\n".join(lines))
    smi = nvidia_smi_L()
    if not smi:
        errs.append("nvidia-smi -L returned no GPUs")
    else:
        all_idx = all_gpu_indices()
        busy = busy_gpu_indices()
        free_count = len(all_idx) - len(busy)
        summary = gpu_summary(smi)
        if busy:
            print(f"  GPUs: {summary} — {free_count}/{len(all_idx)} free "
                  f"(busy: {sorted(busy)})")
        else:
            print(f"  GPUs: {summary} — {free_count}/{len(all_idx)} free")
        gpu_required = gpu_required_override
        if gpu_required is None:
            gpu_required = max((cfg.get("gpus_per_test") or {}).values(), default=1)
        if free_count < gpu_required:
            errs.append(f"need {gpu_required} free GPU(s); only {free_count} free "
                        f"(busy: {sorted(busy)}) — "
                        "free them yourself (e.g. stop your own jobs); "
                        "this skill no longer kills GPU processes")
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "precheck.json").write_text(json.dumps(dict(
            timestamp=now_iso(), model=cfg["name"],
            venv_python=py, venv_source=src, versions=versions,
            pins={"sglang": pins.get("sglang"), "torch": pins.get("torch")},
            git=gi, nvidia_smi_L=smi, gpu_summary=gpu_summary(smi)), indent=2))
    return _summary(errs, warns)


def _summary(errs, warns):
    for w in warns: print(f"warning: {w}")
    for e in errs:  print(f"error: {e}")
    if errs:
        print(f"\nprecheck FAILED ({len(errs)} error(s))"); return 1
    print("\nprecheck OK"); return 0


def _constants(tree):
    for n in tree.body:
        if not isinstance(n, ast.Assign): continue
        for t in n.targets:
            if not (isinstance(t, ast.Name)
                    and re.fullmatch(r"_?[A-Z][A-Z0-9_]*", t.id)):
                continue
            yield (t.id, None)
            if isinstance(n.value, ast.Dict):
                for vv in n.value.values:
                    if isinstance(vv, ast.Dict):
                        for kk in vv.keys:
                            if isinstance(kk, ast.Constant) and isinstance(kk.value, str):
                                yield (t.id, kk.value)


def _ctx_vars(tree):
    want = {"max_samples", "max_tokens", "max_new_tokens", "max_concurrency"}
    seen = []
    for n in ast.walk(tree):
        if isinstance(n, ast.keyword) and n.arg in want \
                and isinstance(n.value, ast.Name) \
                and re.fullmatch(r"_?[A-Z][A-Z0-9_]*", n.value.id) \
                and n.value.id not in seen \
                and n.value.id not in _TTS_FIXED_PRESETS:
            seen.append(n.value.id)
    return seen


def _stage_base(path, cfg):
    s = path.stem
    if "docs" in path.parts:
        return cfg.get("stage_base_docs_override") or s
    pre = cfg.get("stage_base_strip_prefix", "")
    suf = cfg.get("stage_base_strip_suffix", "")
    if pre and s.startswith(pre): s = s[len(pre):]
    if suf and s.endswith(suf):   s = s[:-len(suf)]
    return s


def _split_source(entry, default_file):
    """Resolve a config `paths` value into (json_file, json_path).

    Bare dotted path → uses default_file. `<file>::<dotted.path>` form →
    overrides the file. Returns (None, None) if entry is empty.
    """
    if not entry: return None, None
    if "::" in entry:
        f, _, p = entry.partition("::")
        return (f.strip() or None), (p.strip() or None)
    return default_file, entry


def _build_sample_counts(sc_raw, default_file):
    out = {}
    for ck in ("total", "ok"):
        jf, jp = _split_source(sc_raw.get(ck), default_file)
        out[ck] = dict(json_file=jf, json_path=jp)
    return out


def _emit_groups(constants, cfg_paths, default_file, counters):
    """Build {group: {metric_kind: metric_dict}} from a constant list."""
    groups = {}
    for name, nested in constants:
        mk = match_metric(name, nested)
        if mk is None:
            continue
        spec = METRIC_SPECS[mk]
        jf, jp = _split_source(cfg_paths.get(mk), default_file)
        status = "OK" if (jf and jp) else "NEEDS_CONFIG"
        if status == "OK": counters[0] += 1
        else: counters[1] += 1
        src = f"{name}[{nested!r}]" if nested else name
        groups.setdefault(spec["group"], {})[mk] = dict(source=src,
            json_file=jf, json_path=jp, worst=spec["worst"],
            display=dict(label=spec["label"], scale=spec["scale"],
                         digits=spec["digits"]), status=status)
    return groups


def _find_tmp_path_test(tree):
    """Return (func_name, output_subdir) from the test function using tmp_path."""
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(a.arg == "tmp_path" for a in node.args.args):
            continue
        for child in ast.walk(node):
            if (isinstance(child, ast.BinOp) and isinstance(child.op, ast.Div)
                    and isinstance(child.left, ast.Name) and child.left.id == "tmp_path"
                    and isinstance(child.right, ast.Constant)
                    and isinstance(child.right.value, str)):
                return node.name, child.right.value
    return None, None


def _find_benchmark_module(tree):
    """Return the benchmark module short name from imports, or None."""
    found = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module \
                and node.module.startswith("benchmarks.eval."):
            found.append(node.module.split(".")[-1])
    for mod in found:
        if mod in _BENCHMARK_PATH_OVERRIDES:
            return mod
    return found[0] if found else None


def _infer_metric_sources(tree):
    """Auto-infer metric_sources from AST for tmp_path-based tests."""
    func_name, output_subdir = _find_tmp_path_test(tree)
    if not func_name or not output_subdir:
        return None
    base = output_subdir[:-6] if output_subdir.endswith("_audio") else output_subdir
    json_file = f"{func_name[:30]}0/{output_subdir}/{base}_results.json"
    paths = dict(_DEFAULT_METRIC_PATHS)
    sample_counts = dict(_DEFAULT_SAMPLE_COUNTS)
    bench_mod = _find_benchmark_module(tree)
    overrides = _BENCHMARK_PATH_OVERRIDES.get(bench_mod) if bench_mod else None
    if overrides:
        paths.update(overrides.get("paths") or {})
        sample_counts.update(overrides.get("sample_counts") or {})
    return {"json_file": json_file, "paths": paths, "sample_counts": sample_counts}


def _merge_metric_sources(inferred, config):
    """Merge auto-inferred and config.yaml metric_sources (config wins)."""
    if not inferred:
        return config or {}
    if not config:
        return inferred
    merged = dict(inferred)
    for key, val in config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = {**merged[key], **val}
        else:
            merged[key] = val
    return merged


def _print_config_suggestion(test_name, inferred):
    """Print inferred metric_sources as a suggested config.yaml entry."""
    print(f"  [auto] {test_name}: inferred from AST (convention).")
    print(f"         Suggested config.yaml entry:")
    print(f"           {test_name}:")
    print(f"             json_file: \"{inferred['json_file']}\"")
    sc = inferred.get("sample_counts") or {}
    if sc:
        print(f"             sample_counts:")
        for ck in ("total", "ok"):
            if ck in sc:
                print(f"               {ck}: \"{sc[ck]}\"")
    paths = inferred.get("paths") or {}
    if paths:
        print(f"             paths:")
        for mk, jp in paths.items():
            print(f"               {mk}: \"{jp}\"")


def _validate_config(test_name, inferred, config):
    """Compare inferred metric_sources with config.yaml entry."""
    cfg_jf = config.get("json_file", "")
    inf_jf = inferred.get("json_file", "")
    inf_paths = inferred.get("paths") or {}
    cfg_paths = config.get("paths") or {}
    jf_match = cfg_jf == inf_jf
    paths_match = all(cfg_paths.get(k) == v for k, v in inf_paths.items()
                      if k in cfg_paths)
    if jf_match and paths_match:
        print(f"  [ok] {test_name}: config matches inference")
    else:
        diffs = []
        if not jf_match:
            diffs.append("json_file")
        if not paths_match:
            diffs.append("paths")
        print(f"  [override] {test_name}: config.yaml overrides "
              f"convention ({', '.join(diffs)} differ)")


def discover(out, only, cfg):
    files = []
    for g in cfg["test_globs"]:
        files.extend(sorted(REPO_ROOT.glob(g)))
    sources = cfg.get("metric_sources", {}) or {}
    stages = {}
    counters = [0, 0]  # [configured, needs_cfg]
    for tp in files:
        base = _stage_base(tp, cfg)
        tree = ast.parse(tp.read_text())
        ctx = _ctx_vars(tree)
        rel = tp.relative_to(REPO_ROOT).as_posix()
        sha = sha256(tp)
        extra = cfg["extra_env"].get(tp.name, {})
        if "docs" in tp.parts:
            stages[base] = dict(test=rel, title="Docs smoke", group="docs",
                extra_env=extra, context_vars=ctx, test_file_sha256=sha,
                last_discovered_at=now_iso(), metrics={})
            continue
        ms = sources.get(tp.name, {}) or {}
        inferred = _infer_metric_sources(tree)
        if inferred and not ms:
            _print_config_suggestion(tp.name, inferred)
        elif inferred and ms:
            _validate_config(tp.name, inferred, ms)
        elif not inferred and not ms:
            print(f"  [warn] {tp.name}: no auto-inference available, "
                  f"needs metric_sources in config.yaml")
        ms = _merge_metric_sources(inferred, ms)
        all_constants = list(_constants(tree))
        variants = ms.get("variants") or {}
        if variants:
            # Per-variant routing: each constant assigned to at most one
            # variant by `constant_filter` regex (matched against the bare
            # name, ignoring leading underscore).
            for vname, vcfg in variants.items():
                pat = re.compile(vcfg.get("constant_filter", ".*"))
                claimed = [(n, k) for (n, k) in all_constants
                           if pat.match(n.lstrip("_"))]
                v_default = vcfg.get("json_file")
                v_paths = vcfg.get("paths") or {}
                v_sc_default = _build_sample_counts(
                    vcfg.get("sample_counts") or {}, v_default)
                sc_by_group = vcfg.get("sample_counts_by_group") or {}
                v_groups = _emit_groups(claimed, v_paths, v_default, counters)
                for g, metrics in v_groups.items():
                    if g in sc_by_group:
                        group_sc = _build_sample_counts(sc_by_group[g], v_default)
                    else:
                        group_sc = v_sc_default
                    key = f"{base}_{vname}_{g}"
                    title = (f"{base.replace('_', ' ').upper()} "
                             f"{vname.upper()} {g.capitalize()}")
                    stages[key] = dict(test=rel, title=title, group=g,
                        variant=vname, extra_env=extra, context_vars=ctx,
                        test_file_sha256=sha, last_discovered_at=now_iso(),
                        metrics=metrics, sample_counts=group_sc)
        else:
            # Single-source flow (one result-JSON tree per test file).
            default_file = ms.get("json_file")
            cfg_paths = ms.get("paths", {}) or {}
            sample_counts = _build_sample_counts(
                ms.get("sample_counts") or {}, default_file)
            groups = _emit_groups(all_constants, cfg_paths, default_file, counters)
            for g, metrics in groups.items():
                key = f"{base}_{g}"
                title = f"{base.replace('_', ' ').upper()} {g.capitalize()}"
                stages[key] = dict(test=rel, title=title, group=g,
                    extra_env=extra, context_vars=ctx, test_file_sha256=sha,
                    last_discovered_at=now_iso(), metrics=metrics,
                    sample_counts=sample_counts)
    if only: stages = {k: v for k, v in stages.items() if k == only}
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_yaml(stages, out)
    print(f"[{cfg['name']}] {len(stages)} stages written to {out}, "
          f"{counters[0]} metric(s) OK, {counters[1]} need config")
    return 0


def _yq(s): return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _write_yaml(stages, path):
    L = ["# Generated by tune.py discover; re-run discover to refresh."]
    for key, e in stages.items():
        L += [f"{key}:", f"  test: {e['test']}",
              f"  title: {_yq(e['title'])}", f"  group: {e['group']}"]
        if e["extra_env"]:
            L.append("  extra_env:")
            L += [f"    {k}: {_yq(v)}" for k, v in e["extra_env"].items()]
        else:
            L.append("  extra_env: {}")
        if e["context_vars"]:
            L.append("  context_vars:")
            L += [f"    - {c}" for c in e["context_vars"]]
        else:
            L.append("  context_vars: []")
        if e.get("variant"):
            L.append(f"  variant: {_yq(e['variant'])}")
        L += [f"  test_file_sha256: {e['test_file_sha256']}",
              f"  last_discovered_at: {e['last_discovered_at']}"]
        sc = e.get("sample_counts") or {}
        if sc:
            L.append("  sample_counts:")
            for ck in ("total", "ok"):
                c = sc.get(ck) or {}
                L += [f"    {ck}:",
                      "      json_file: null" if not c.get("json_file")
                      else f"      json_file: {_yq(c['json_file'])}",
                      "      json_path: null" if not c.get("json_path")
                      else f"      json_path: {_yq(c['json_path'])}"]
        if not e["metrics"]:
            L.append("  metrics: {}"); continue
        L.append("  metrics:")
        for mk, m in e["metrics"].items():
            L += [f"    {mk}:", f"      source: {_yq(m['source'])}",
                  "      json_file: null" if not m.get("json_file")
                  else f"      json_file: {_yq(m['json_file'])}",
                  "      json_path: null" if not m.get("json_path")
                  else f"      json_path: {_yq(m['json_path'])}",
                  f"      worst: {m['worst']}", "      display:",
                  f"        label: {_yq(m['display']['label'])}",
                  f"        scale: {m['display']['scale']}",
                  f"        digits: {m['display']['digits']}",
                  f"      status: {m['status']}"]
    path.write_text("\n".join(L) + "\n")


def _load_yaml(path, top_is_dict=False):
    lines = [ln for ln in path.read_text().splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    idx = [0]
    def ind(s): return len(s) - len(s.lstrip(" "))
    def sc(v):
        v = v.strip()
        if v == "null": return None
        if v == "[]":   return []
        if v == "{}":   return {}
        if v.startswith('"') and v.endswith('"'):
            return v[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        try: return float(v) if "." in v else int(v)
        except ValueError: return v
    def blk(need):
        r = {}
        while idx[0] < len(lines):
            ln = lines[idx[0]]
            if ind(ln) < need: return r
            key, _, rest = ln.strip().partition(":")
            key, rest = key.strip(), rest.strip()
            idx[0] += 1
            if rest: r[key] = sc(rest); continue
            if idx[0] < len(lines) and lines[idx[0]].lstrip().startswith("- "):
                items = []
                while (idx[0] < len(lines) and ind(lines[idx[0]]) > need
                       and lines[idx[0]].lstrip().startswith("- ")):
                    items.append(sc(lines[idx[0]].lstrip()[2:]))
                    idx[0] += 1
                r[key] = items
            elif idx[0] < len(lines) and ind(lines[idx[0]]) > need:
                r[key] = blk(need + 2)
            else: r[key] = {}
        return r
    if top_is_dict:
        idx[0] = 0
        return blk(0)
    out = {}
    while idx[0] < len(lines):
        key = lines[idx[0]].rstrip(":").strip(); idx[0] += 1
        out[key] = blk(2)
    return out


def _stage_base_of(sk, stage):
    """Derive the test-file base of a stage (strips the _<group> suffix)."""
    g = stage.get("group", "")
    return sk[:-len(g) - 1] if g and sk.endswith(f"_{g}") else sk


def _stage_meta_base(sk, stage):
    """Strip the _<variant> suffix from a base, if any.

    Lets users type the test-file base (e.g. `tts`) to mean all variants.
    Returns None when no variant applies.
    """
    base = _stage_base_of(sk, stage)
    v = stage.get("variant")
    if v and base.endswith(f"_{v}"):
        return base[:-len(v) - 1]
    return None


def _build_aliases(all_stages):
    by_base, by_group = {}, {}
    for sk, v in all_stages.items():
        b = _stage_base_of(sk, v)
        by_base.setdefault(b, []).append(sk)
        meta = _stage_meta_base(sk, v)
        if meta and meta != b:
            by_base.setdefault(meta, []).append(sk)
        by_group.setdefault(v.get("group", ""), []).append(sk)
    return by_base, by_group


def _expand_stages(tokens, all_stages):
    """Resolve exact keys, test-file bases, and @group aliases to stage keys."""
    by_base, by_group = _build_aliases(all_stages)
    out, unknown = [], []
    for t in tokens:
        hits = None
        if t in all_stages: hits = [t]
        elif t.startswith("@") and t[1:] in by_group: hits = by_group[t[1:]]
        elif t in by_base: hits = by_base[t]
        if hits is None:
            unknown.append(t); continue
        for sk in hits:
            if sk not in out: out.append(sk)
    if unknown:
        print(f"error: unknown stage token(s): {unknown}")
        print(f"  stage keys: {sorted(all_stages)}")
        print(f"  test-file bases: {sorted(by_base)}")
        print(f"  groups: {sorted('@' + g for g in by_group if g)}")
        return None
    return out


def stages_list(cfg):
    sy = stages_path(cfg["name"])
    if not sy.exists():
        print(f"error: stages.yaml not found at {sy} — run: "
              f"tune.py --model {cfg['name']} discover")
        return 2
    all_stages = _load_yaml(sy)
    by_base, by_group = _build_aliases(all_stages)
    print("bases:")
    for base in sorted(by_base):
        keys = by_base[base]
        tp = all_stages[keys[0]]["test"]
        print(f"  {base}  ({tp})")
        for sk in keys:
            g = all_stages[sk].get("group", "")
            print(f"    {sk}  [{g}]")
    print("groups:")
    for g in sorted(by_group):
        if not g: continue
        print(f"  @{g} ({len(by_group[g])}): {', '.join(sorted(by_group[g]))}")
    return 0


def run_cmd(args):
    cfg = load_model_config(args.model)
    py, src, _ = resolve_venv(args.venv_python, cfg)
    out = Path(args.output_dir) if args.output_dir \
        else DEFAULT_RUN_ROOT / f"{cfg['name']}-{ts_dir()}"
    out.mkdir(parents=True, exist_ok=True)
    # Tee all stdout prints (banners, progress lines) to <out>/run.log so
    # the user can `tail -f` it mid-run. Bash-captured output is still
    # visible when the command ends.
    _log_fh = open(out / "run.log", "w", buffering=1)
    _orig_stdout = sys.stdout
    sys.stdout = _Tee(_orig_stdout, _log_fh)
    try:
        return _run_cmd_inner(args, cfg, py, src, out)
    finally:
        sys.stdout = _orig_stdout
        _log_fh.close()


def _run_cmd_inner(args, cfg, py, src, out):
    print(f"run log: {out / 'run.log'}")
    # Resolve stages BEFORE precheck so we know which datasets are
    # actually needed (don't force the user to cache datasets they're
    # not going to touch this run).
    sy = Path(args.stages_yaml) if args.stages_yaml else stages_path(cfg["name"])
    if not sy.exists():
        print(f"error: stages.yaml not found at {sy} — run: "
              f"tune.py discover --model {cfg['name']}")
        return 2
    all_stages = _load_yaml(sy)
    if args.stages == "ALL":
        sel = list(all_stages.keys())
    else:
        tokens = [s.strip() for s in args.stages.split(",") if s.strip()]
        sel = _expand_stages(tokens, all_stages)
        if sel is None: return 2
        if set(sel) != set(tokens):
            print(f"expanded --stages {args.stages!r} → "
                  f"{len(sel)} stage(s): {', '.join(sel)}")
    for s in sel:
        tf = REPO_ROOT / all_stages[s]["test"]
        if not tf.exists():
            print(f"error: test file missing for {s}: {tf}"); return 2
        if sha256(tf) != all_stages[s].get("test_file_sha256"):
            print(f"warning: {s} test sha mismatch — "
                  f"run `tune.py discover --model {cfg['name']}`")
    # Which datasets do the selected tests actually reference?
    # Each test uses DATASETS["key"]; resolve key → repo_id via the
    # canonical benchmarks/dataset/prepare.py:DATASETS dict so we don't
    # depend on naming coincidences between test keys and config repo ids.
    ds_map = _parse_datasets_dict()
    needed_repos = set()
    for s in sel:
        try:
            _, ds_keys = _read_test_context(REPO_ROOT / all_stages[s]["test"])
        except Exception:
            continue
        for k in ds_keys:
            if k in ds_map:
                needed_repos.add(ds_map[k])
    required_ds = sorted(needed_repos) if needed_repos else list(cfg["hf_datasets"])
    # Heads-up if AST-derived needs include a repo not in cfg
    extras = [d for d in required_ds if d not in cfg["hf_datasets"]]
    if extras:
        print(f"note: test(s) reference repo(s) not listed in "
              f"config.yaml hf_datasets: {extras}")
    selected_tests = [Path(all_stages[s]["test"]).name for s in sel]
    required_models = _required_model_ids_for_tests(cfg, selected_tests)
    gpus_per_test = cfg.get("gpus_per_test", {}) or {}
    selected_gpu_requirement = max(
        (gpus_per_test.get(Path(all_stages[s]["test"]).name, 2) for s in sel),
        default=1,
    )
    if not args.skip_precheck:
        rc = precheck(py, src, out, args.skip_version_check, cfg,
                      datasets_override=required_ds,
                      model_ids_override=required_models,
                      gpu_required_override=selected_gpu_requirement)
        if rc: return rc
    else:
        smi = nvidia_smi_L()
        (out / "precheck.json").write_text(json.dumps(dict(
            timestamp=now_iso(), model=cfg["name"],
            venv_python=py, venv_source=src,
            versions={}, pins={}, git=git_info(),
            nvidia_smi_L=smi, gpu_summary=gpu_summary(smi)), indent=2))
    gi = git_info()
    (out / "plan.json").write_text(json.dumps(dict(
        model=cfg["name"], stages=sel, repeats=args.repeats,
        timestamp=now_iso(), git_sha=gi["sha"], git_branch=gi["branch"],
        dirty=gi["dirty"], venv_python=py, venv_source=src,
        tune_version=__version__, stages_yaml=str(sy)), indent=2))
    if gi["dirty"]:
        d = subprocess.run(["git", "diff", "HEAD"], cwd=REPO_ROOT,
                           capture_output=True, text=True, check=False).stdout
        (out / "workspace.diff").write_text(d)
    # Group stages by test file so one pytest invocation covers all stages
    # tied to that file (e.g. mmmu_accuracy + mmmu_speed come from a single
    # `test_mmmu_accuracy_and_speed` function — run it once per repeat,
    # extract both metric groups from the same result JSON).
    by_test = {}
    for sk in sel:
        by_test.setdefault(all_stages[sk]["test"], []).append(sk)
    for sk in sel:
        (out / sk).mkdir(parents=True, exist_ok=True)
    max_passes = getattr(args, "max_passes", _DEFAULT_CALIBRATION_PASSES)
    max_gpus = max(
        (gpus_per_test.get(Path(all_stages[s]["test"]).name, 2) for s in sel),
        default=2,
    )
    for pass_num in range(1, max_passes + 1):
        ran_any = False
        for k in range(1, args.repeats + 1):
            for test_path, stage_keys in by_test.items():
                if all(_run_json_ok(out / sk / f"run{k}.json", all_stages[sk])
                       for sk in stage_keys):
                    if pass_num == 1 and args.resume:
                        print(f"[{Path(test_path).stem}] run {k}/{args.repeats} "
                              f"skipped (complete, {len(stage_keys)} stage(s))")
                    continue
                _purge_incomplete_run(out, stage_keys, all_stages, k)
                needed = gpus_per_test.get(Path(test_path).name, 2)
                extra_args = (cfg.get("pytest_extra_args", {}) or {}).get(
                    Path(test_path).name, []) or []
                if pass_num > 1:
                    print(f"=== calibration pass {pass_num}/{max_passes}: "
                          f"retry {Path(test_path).stem} run {k} ===")
                if _run_shared(test_path, stage_keys, all_stages, out, k, py,
                               args.repeats, needed, extra_args):
                    audit = audit_completeness(out, all_stages)
                    print(f"completeness at HALT: "
                          f"{audit['ok']}/{audit['total']} stage-runs complete")
                    return 1
                ran_any = True
        audit = audit_completeness(out, all_stages)
        print(f"completeness after pass {pass_num}: "
              f"{audit['ok']}/{audit['total']} stage-runs complete")
        if audit["complete"]:
            break
        if not ran_any:
            break
        if pass_num < max_passes:
            print(f"{audit['missing_count']} incomplete — GPU cleanup before next pass")
            if not _ensure_gpus_free(max_gpus):
                print("error: GPU memory not cleared; stopping calibration passes")
                break
    audit = audit_completeness(out, all_stages)
    if not audit["complete"]:
        print(f"error: calibration incomplete ({audit['ok']}/{audit['total']}). "
              f"Missing {audit['missing_count']} stage-run(s):")
        for m in audit["missing"][:20]:
            print(f"  {m['stage_key']}/run{m['run']}: {m['reason']}")
        if audit["missing_count"] > 20:
            print(f"  … and {audit['missing_count'] - 20} more")
        print("resume with: tune.py --model … run --output-dir … --resume")
        return 1
    return report(out)


class _Tee:
    """Duplicate writes across multiple file-like streams (line-flushed)."""
    def __init__(self, *streams): self._s = streams
    def write(self, s):
        for x in self._s:
            try: x.write(s); x.flush()
            except Exception: pass
    def flush(self):
        for x in self._s:
            try: x.flush()
            except Exception: pass


def _best(cmd, desc):
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode:
        print(f"warning: {desc} failed (rc={r.returncode}): "
              f"{r.stderr.strip()[:160]}")


def _read_test_context(test_path):
    """AST-scan a test file for runtime knobs and dataset keys.
    Returns (kwargs, datasets) where kwargs maps max_samples/max_tokens/
    max_new_tokens/max_concurrency to their literal values (resolving
    UPPER_CASE name references), and datasets is the list of DATASETS["..."]
    keys referenced in the file.
    """
    tree = ast.parse(Path(test_path).read_text())
    consts = {}
    for n in tree.body:
        if not isinstance(n, ast.Assign): continue
        for t in n.targets:
            if (isinstance(t, ast.Name)
                    and re.fullmatch(r"_?[A-Z][A-Z0-9_]*", t.id)
                    and isinstance(n.value, ast.Constant)):
                consts[t.id] = n.value.value
    wanted = ("max_samples", "max_tokens", "max_new_tokens", "max_concurrency")
    kwargs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg in wanted:
            if isinstance(node.value, ast.Constant):
                kwargs[node.arg] = node.value.value
            elif isinstance(node.value, ast.Name):
                kwargs[node.arg] = consts.get(node.value.id, f"<{node.value.id}>")
    datasets = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id == "DATASETS"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            datasets.add(node.slice.value)
    return kwargs, sorted(datasets)


def _parse_datasets_dict():
    """Pull the canonical `DATASETS` key→repo_id map from
    benchmarks/dataset/prepare.py so we can translate what a test
    asks for (e.g. DATASETS["seedtts-mini"]) into the HF repo id
    (e.g. "zhaochenyang20/seed-tts-eval-mini")."""
    p = REPO_ROOT / "benchmarks" / "dataset" / "prepare.py"
    if not p.exists(): return {}
    try:
        tree = ast.parse(p.read_text())
    except SyntaxError:
        return {}
    for n in tree.body:
        # Accept both bare `DATASETS = {...}` and annotated
        # `DATASETS: dict[str, str] = {...}` (which is ast.AnnAssign).
        if isinstance(n, ast.Assign):
            targets, value = n.targets, n.value
        elif isinstance(n, ast.AnnAssign) and n.value is not None:
            targets, value = [n.target], n.value
        else:
            continue
        for t in targets:
            if not (isinstance(t, ast.Name) and t.id == "DATASETS"): continue
            if not isinstance(value, ast.Dict): continue
            out = {}
            for kk, vv in zip(value.keys, value.values):
                if (isinstance(kk, ast.Constant) and isinstance(kk.value, str)
                        and isinstance(vv, ast.Constant) and isinstance(vv.value, str)):
                    out[kk.value] = vv.value
            return out
    return {}


def _print_run_banner(label, test_path, stage_keys, all_stages):
    """Print dataset / sample size / concurrency / max_tokens / tracked
    metrics before kicking off pytest. Purely informational."""
    try:
        kwargs, datasets = _read_test_context(REPO_ROOT / test_path)
    except Exception as e:
        print(f"{label} (couldn't read test context: {e})")
        return
    print(f"{label} config:")
    shown = False
    if datasets:
        print(f"  dataset(s): {', '.join(datasets)}")
        shown = True
    for key in ("max_samples", "max_tokens", "max_new_tokens", "max_concurrency"):
        if key in kwargs:
            print(f"  {key}: {kwargs[key]}")
            shown = True
    metric_lines = []
    for sk in stage_keys:
        metrics = all_stages[sk].get("metrics") or {}
        if not metrics: continue
        parts = [f"{mk}({m['worst']})" for mk, m in metrics.items()]
        metric_lines.append(f"    {sk}: {', '.join(parts)}")
    if metric_lines:
        print("  tracked metrics:")
        for ln in metric_lines: print(ln)
        shown = True
    if not shown:
        print("  (docs smoke — no benchmark params, pass/fail only)")


def _run_shared(test_path, stage_keys, all_stages, out, k, py, total, gpus_needed, extra_args=None):
    """Run pytest once on test_path; write per-stage run{k}.json from
    the result JSONs written under the fresh pytest basetemp.
    """
    test_base = Path(test_path).stem
    shared = out / "_pytest" / test_base
    shared.mkdir(parents=True, exist_ok=True)
    log = shared / f"run{k}.log"
    basetemp = shared / f"basetemp_run{k}"
    # extra_env is derived from test filename at discover — all stages
    # sharing a test file have identical extra_env; just use the first.
    env = os.environ.copy()
    # Never inherit a shell-level CUDA_VISIBLE_DEVICES — CI gets a fresh
    # container per stage; tune.py picks GPUs after cleanup instead.
    env.pop("CUDA_VISIBLE_DEVICES", None)
    # Match CI's `export PYTHONPATH=$PWD`: server subprocesses launched by
    # tests are invoked as `python examples/<launcher>.py`, which only puts
    # `examples/` on sys.path. Prepend the repo root so local package imports
    # resolve without clobbering user-set values.
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{existing_pp}" if existing_pp else str(REPO_ROOT)
    )
    # Match `source <venv>/bin/activate`: put the venv's bin dir on PATH so
    # console_scripts installed by the venv (e.g. `sgl-omni`, used by the
    # router's local launcher) resolve. We invoke pytest via the venv's
    # python directly without activating the venv, so PATH would otherwise
    # only contain the parent shell's entries.
    venv_bin = str(Path(py).parent)
    existing_path = env.get("PATH", "")
    env["PATH"] = (
        f"{venv_bin}{os.pathsep}{existing_path}" if existing_path else venv_bin
    )
    # Exclude loopback addresses from any inherited HTTP_PROXY/HTTPS_PROXY.
    # Servers spawned by tests (router, workers) make loopback health-check
    # calls with httpx/requests, which honor *_PROXY by default; without
    # NO_PROXY the proxy intercepts the loopback request and returns 502,
    # so the router never sees its workers as healthy.
    loopback_no_proxy = "localhost,127.0.0.1,::1"
    existing_no_proxy = env.get("NO_PROXY") or env.get("no_proxy") or ""
    merged = (
        f"{loopback_no_proxy},{existing_no_proxy}"
        if existing_no_proxy else loopback_no_proxy
    )
    env["NO_PROXY"] = merged
    env["no_proxy"] = merged
    env.update(all_stages[stage_keys[0]].get("extra_env") or {})
    label = f"[{test_base}] run {k}/{total} ({len(stage_keys)} stage(s), needs {gpus_needed} GPU)"
    _print_run_banner(label, test_path, stage_keys, all_stages)
    attempts, status, reason, dur, text, pytest_rc = 0, "ok", "", 0.0, "", 0
    while attempts < _MAX_RUN_ATTEMPTS:
        attempts += 1
        picked, pick_err = _pick_gpus_for_launch(gpus_needed, label)
        if picked is None:
            status, reason, dur = "failed", pick_err, 0.0
            print(f"{label} {pick_err}")
            break
        _cleanup_flashinfer_cache(env)
        shutil.rmtree(basetemp, ignore_errors=True)
        basetemp.mkdir(parents=True)
        picked, gate_err = _launch_gpu_gate(picked, gpus_needed, label)
        if picked is None:
            status, reason, dur = "failed", gate_err or "launch GPU gate failed", 0.0
            print(f"{label} {reason}")
            break
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, picked))
        print(f"{label} using GPU(s) {picked} "
              f"(CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']})")
        t0 = time.monotonic()
        # Never pass -x / --exitfirst: calibration must collect every stage's
        # metrics even when an earlier threshold assertion fails. Only hard
        # failures (OOM, crash, timeout) may leave metrics missing.
        pytest_cmd = [py, "-m", "pytest", test_path,
                      "-v", "-s", f"--basetemp={basetemp}"]
        if extra_args:
            pytest_cmd.extend(extra_args)
        with open(log, "w") as lf:
            pytest_proc = subprocess.Popen(
                pytest_cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=lf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            pytest_rc = _wait_pytest_with_watchdog(pytest_proc, log, label)
        _cleanup_after_pytest(test_path, pytest_proc.pid, basetemp)
        if not _ensure_gpus_free(gpus_needed):
            status, reason, dur = "failed", "GPU memory not released after run", 0.0
            break
        dur = time.monotonic() - t0
        text = log.read_text(errors="replace")
        if pytest_rc == 0:
            status, reason = "ok", ""
            break
        reason = _classify(text, pytest_rc)
        status = "failed"
        retryable = (
            any(s in reason for s in RETRY_SIGS)
            or reason.startswith("crashed")
            or "GPU memory" in reason
        )
        if attempts < _MAX_RUN_ATTEMPTS and retryable:
            print(f"{label} {reason} — must clear GPU to <2 GiB before retry "
                  f"({attempts}/{_MAX_RUN_ATTEMPTS})")
            if not _ensure_gpus_free(gpus_needed):
                status, reason = "failed", "GPU memory not released before retry"
                break
            continue
        break
    if status == "ok":
        print(f"{label} ok ({dur:.1f}s)")
    else:
        print(f"{label} failed: {reason} ({dur:.1f}s)")
    extraction_warnings = []
    metrics_by_stage = {}
    for sk in stage_keys:
        stage = all_stages[sk]
        sd = out / sk
        metrics = _extract(stage, basetemp, stage_key=sk, warnings=extraction_warnings)
        metrics_by_stage[sk] = metrics
        sample_counts = _extract_counts(stage, basetemp)
        obs_status, obs_reason = _observation_status(
            stage, metrics, status, reason)
        run_payload = dict(
            status=obs_status, reason=obs_reason, metrics=metrics,
            sample_counts=sample_counts,
            duration_s=round(dur, 2), attempts=attempts,
            pytest_log=str(log.resolve()),
            basetemp=str(basetemp.resolve()))
        if stage.get("metrics"):
            run_payload["pytest_rc"] = pytest_rc
        (sd / f"run{k}.json").write_text(json.dumps(run_payload, indent=2))
        (sd / f"run{k}.log").write_text(
            f"# Shared pytest log (one invocation covered all stages from "
            f"{test_path}):\n# {log.resolve()}\n# basetemp: {basetemp.resolve()}\n")
        brief_parts = []
        if sample_counts.get("total") is not None or sample_counts.get("ok") is not None:
            brief_parts.append(f"samples={sample_counts.get('ok')}/{sample_counts.get('total')}")
        brief_parts += [f"{k2}={_fmt(v, stage['metrics'][k2]['display'])}"
                        for k2, v in metrics.items() if v is not None]
        brief = " ".join(brief_parts)
        # Flag stages where every tracked metric ended up None — almost always
        # a config bug (wrong path / missing CONCURRENCY / variant filter).
        if stage.get("metrics") and all(v is None for v in metrics.values()):
            print(f"  → {sk}: ⚠ ALL metrics None — likely config bug "
                  f"(check models/<M>/config.yaml metric_sources for {Path(test_path).name})")
        if obs_status == "ok":
            note = (f" (pytest rc={pytest_rc})" if obs_reason else "")
            print(f"  → {sk}: {brief or '(no metrics extracted)'}{note}")
        else:
            print(f"  → {sk}: failed ({obs_reason or reason})"
                  + (f" — {brief}" if brief else ""))
    if extraction_warnings:
        print(f"  ⚠ metric extraction warnings ({len(extraction_warnings)}):")
        for w in extraction_warnings[:20]:  # cap to keep stdout readable
            print(w)
        if len(extraction_warnings) > 20:
            print(f"  ... and {len(extraction_warnings) - 20} more "
                  f"(see {basetemp} listing to debug)")
    if _should_halt_on_extraction(
            stage_keys, all_stages, metrics_by_stage, extraction_warnings, pytest_rc):
        print(f"error: metric extraction HALT after {label}")
        print("  pytest passed but one or more threshold metrics are missing.")
        print("  Fix models/<M>/config.yaml metric_sources (or test JSON output),")
        print(f"  purge incomplete run{k}.json for: {', '.join(stage_keys)},")
        print("  then: tune.py --model … run --output-dir … --resume")
        return True
    return False


def _cleanup_after_pytest(test_path, process_group_id, basetemp):
    """Clean up subprocesses owned by this pytest invocation."""
    cleanup_process_group_ids = _read_cleanup_manifest_process_groups(basetemp)
    if not cleanup_process_group_ids:
        return
    cleanup_process_group_ids.add(process_group_id)

    for sig, delay in ((signal.SIGTERM, 5), (signal.SIGKILL, 1)):
        for process_group in sorted(cleanup_process_group_ids):
            try:
                os.killpg(process_group, sig)
            except ProcessLookupError:
                continue
        time.sleep(delay)


def _read_cleanup_manifest_process_groups(basetemp):
    process_groups = set()
    for manifest in Path(basetemp).rglob("router_pgids.txt"):
        for line in manifest.read_text().splitlines():
            try:
                process_groups.add(int(line.strip()))
            except ValueError:
                continue
    return process_groups


def _log_crash_detected(text: str) -> str | None:
    lower = text.lower()
    for sig in _CRASH_SIGS:
        if sig.lower() in lower:
            return sig
    if "assertionerror" in lower and any(
            x in lower for x in ("router", "worker", "server", "health")):
        return "server startup failure"
    return None


def _wait_pytest_with_watchdog(pytest_proc, log_path: Path, label: str) -> int:
    """Poll pytest every _PYTEST_POLL_S; abort early on log crash signatures."""
    last_size = 0
    stall_s = 0
    while True:
        rc = pytest_proc.poll()
        text = ""
        if log_path.exists():
            text = log_path.read_text(errors="replace")
            size = len(text)
            stall_s = 0 if size > last_size else stall_s + _PYTEST_POLL_S
            last_size = size
        if rc is not None:
            return rc
        crash = _log_crash_detected(text[-8000:] if text else "")
        if crash:
            print(f"{label} crash detected in log ({crash}) — stopping pytest")
            try:
                os.killpg(pytest_proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pytest_proc.kill()
            pytest_proc.wait(timeout=30)
            return -1
        mem = _gpu_memory_by_index()
        print(f"{label} running… GPU mem={mem} log={last_size}B stall={stall_s}s")
        time.sleep(_PYTEST_POLL_S)


def _classify(text, rc):
    if rc == -1:
        crash = _log_crash_detected(text)
        return f"crashed ({crash})" if crash else "crashed (watchdog)"
    if "CUDA out of memory" in text or "OutOfMemoryError" in text:
        return "OOM"
    if rc == 137:
        return "exit 137 (killed)"
    if rc == 139:
        return "exit 139 (segfault)"
    if "TimeoutExpired" in text:
        return "TimeoutExpired"
    return f"exit {rc}"


def _extract(stage, basetemp, stage_key=None, warnings=None):
    """Pull each metric from its result JSON under the pytest basetemp.

    Appends a one-line message to `warnings` (if given) for each missing
    file or unreadable key, so the caller can surface them prominently
    instead of silently writing N/A into the report.
    """
    o = {}
    cache = {}  # json_file → parsed dict (so we only load each file once)
    sk = stage_key or stage.get("title", "?")
    for mk, m in stage["metrics"].items():
        jf, jp = m.get("json_file"), m.get("json_path")
        if not (jf and jp):
            o[mk] = None
            if warnings is not None:
                warnings.append(f"  {sk}.{mk}: no json_file/json_path in stages.yaml")
            continue
        p = Path(basetemp) / jf
        if not p.exists():
            o[mk] = None
            if warnings is not None:
                warnings.append(f"  {sk}.{mk}: file missing — {p}")
            continue
        try:
            data = cache.get(jf)
            if data is None:
                data = json.loads(p.read_text())
                cache[jf] = data
            for key in jp.split("."):
                data = data[key]
            o[mk] = float(data)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            o[mk] = None
            if warnings is not None:
                warnings.append(f"  {sk}.{mk}: read failed at {jf}::{jp} — {type(exc).__name__}")
    return o


def _extract_counts(stage, basetemp):
    """Pull sample counts (total/ok) from the stage's result JSON(s).

    Always called, regardless of pytest status — tests dump their JSONs
    before the assertion, so counts are usually present even when the
    test fails on a threshold check.
    """
    o = {}
    cache = {}
    for ck in ("total", "ok"):
        c = (stage.get("sample_counts") or {}).get(ck) or {}
        jf, jp = c.get("json_file"), c.get("json_path")
        if not (jf and jp): o[ck] = None; continue
        p = Path(basetemp) / jf
        if not p.exists(): o[ck] = None; continue
        try:
            data = cache.get(jf)
            if data is None:
                data = json.loads(p.read_text())
                cache[jf] = data
            keys = jp.split(".")
            if keys[-1] == "__len__":
                for key in keys[:-1]:
                    data = data[key]
                o[ck] = len(data) if isinstance(data, list) else None
            else:
                for key in keys:
                    data = data[key]
                o[ck] = int(data)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            o[ck] = None
    return o


def _fmt(v, d): return "N/A" if v is None else f"{v * d['scale']:.{d['digits']}f}"
def _fmt_count(v): return "N/A" if v is None else str(v)


def status_cmd(run_dir: Path):
    """JSON snapshot for agent polling (every ~120s during calibration)."""
    plan_path = run_dir / "plan.json"
    if not plan_path.exists():
        print(json.dumps({"error": f"no plan.json in {run_dir}"}, indent=2))
        return 1
    plan = json.loads(plan_path.read_text())
    sy = Path(plan.get("stages_yaml")
              or stages_path(plan.get("model", DEFAULT_MODEL)))
    all_stages = _load_yaml(sy)
    audit = audit_completeness(run_dir, all_stages, plan)
    mem = _gpu_memory_by_index()
    busy = sorted(busy_gpu_indices())
    ready, _, _ = _ready_gpu_indices(max(len(mem), 1))
    under_2gb = {
        str(i): m for i, m in mem.items() if m <= _GPU_RETRY_MEM_MIB and i not in busy
    }
    launch_allowed = len(ready) >= max(len(mem), 1) and all(
        m <= _GPU_RETRY_MEM_MIB for i, m in mem.items() if i not in busy
    ) if mem else False
    run_log = run_dir / "run.log"
    log_tail = []
    if run_log.exists():
        lines = run_log.read_text(errors="replace").splitlines()
        log_tail = lines[-15:]
    pytest_active = bool(subprocess.run(
        ["pgrep", "-f", f"pytest.*{run_dir.name}"],
        capture_output=True, check=False).stdout.strip())
    out = dict(
        run_dir=str(run_dir),
        model=plan.get("model"),
        repeats=plan["repeats"],
        complete=audit["complete"],
        ok=audit["ok"],
        total=audit["total"],
        missing_count=audit["missing_count"],
        missing=audit["missing"][:30],
        gpu_memory_mib=mem,
        gpu_busy=sorted(busy),
        gpu_ready=len(ready),
        gpus_under_2gb_mib=under_2gb,
        launch_allowed=launch_allowed,
        gpu_mem_limit_mib=_GPU_RETRY_MEM_MIB,
        pytest_active=pytest_active,
        run_log_tail=log_tail,
        agent_poll_interval_s=_AGENT_POLL_INTERVAL_S,
        timestamp=now_iso(),
    )
    print(json.dumps(out, indent=2))
    return 0 if audit["complete"] else 1


def report(run_dir):
    audit = audit_completeness(run_dir)
    if not audit["complete"]:
        print(f"error: refusing report — incomplete calibration "
              f"({audit['ok']}/{audit['total']})")
        for m in audit["missing"][:15]:
            print(f"  {m['stage_key']}/run{m['run']}: {m['reason']}")
        return 1
    plan = json.loads((run_dir / "plan.json").read_text())
    pre = json.loads((run_dir / "precheck.json").read_text()) \
        if (run_dir / "precheck.json").exists() else {}
    sy = Path(plan.get("stages_yaml") or stages_path(plan.get("model", DEFAULT_MODEL)))
    all_stages = _load_yaml(sy)
    N = plan["repeats"]
    L = ["# CI Threshold Observation Report", ""]
    for idx, sk in enumerate(plan["stages"], start=1):
        s = all_stages[sk]
        L += [f"## {idx}. {s['title']}", "", f"{{{{CONTEXT:{sk}}}}}", ""]
        results = [json.loads((run_dir / sk / f"run{k}.json").read_text())
                   if (run_dir / sk / f"run{k}.json").exists() else None
                   for k in range(1, N + 1)]
        if not s["metrics"]:
            L += ["| Run | Result |", "|-----|--------|"]
            cells = ["PASS" if r and r["status"] == "ok" else "FAIL"
                     for r in results]
            for i, c in enumerate(cells, start=1): L.append(f"| {i} | {c} |")
            worst = "FAIL" if any(c == "FAIL" for c in cells) else "PASS"
            L += [f"| **Worst-of-{N}** | **{worst}** |", ""]
            continue
        keys = list(s["metrics"].keys())
        disp = {k: s["metrics"][k]["display"] for k in keys}
        worst = {k: s["metrics"][k]["worst"] for k in keys}
        # Metrics that can never populate (no json_path configured) render
        # as N/A for every run and a warning line at the bottom.
        nulls = {k for k, m in s["metrics"].items() if not m.get("json_path")}
        has_counts = bool(s.get("sample_counts"))
        count_headers = ["Samples run", "Samples ok"] if has_counts else []
        L += ["| Run | " + " | ".join(count_headers + [disp[k]["label"] for k in keys]) + " |",
              "|-----|" + "|".join(["-" * 8] * (len(keys) + len(count_headers))) + "|"]
        vals = {k: [] for k in keys}
        cvals = {"total": [], "ok": []}
        for i, r in enumerate(results, start=1):
            cells = []
            if has_counts:
                if r is None:
                    cells += ["MISSING", "MISSING"]
                else:
                    sc = r.get("sample_counts") or {}
                    tot, okc = sc.get("total"), sc.get("ok")
                    cells += [_fmt_count(tot), _fmt_count(okc)]
                    if tot is not None: cvals["total"].append(tot)
                    if okc is not None: cvals["ok"].append(okc)
            for k in keys:
                if r is None: cells.append("MISSING")
                elif k in nulls: cells.append("N/A")
                else:
                    v = (r.get("metrics") or {}).get(k)
                    cells.append(_fmt(v, disp[k]))
                    if v is not None: vals[k].append(v)
            L.append(f"| {i} | " + " | ".join(cells) + " |")
        wc = []
        if has_counts:
            # Samples run/ok are diagnostic counts, not quality metrics.
            # Worst-of-N has no meaning for them — per-run values above
            # already surface any coverage drop. Leave these cells blank.
            wc.append("—")
            wc.append("—")
        for k in keys:
            if k in nulls or not vals[k]: wc.append("**N/A**"); continue
            v = min(vals[k]) if worst[k] == "min" else max(vals[k])
            wc.append(f"**{_fmt(v, disp[k])}**")
        L += [f"| **Worst-of-{N}** | " + " | ".join(wc) + " |", ""]
        # `Threshold: ...` lines were removed — what actually got written
        # into the test files (if anything) is recorded in the
        # "Applied changes" table appended after mode-smart/mode-full
        # apply (see SKILL.md step 9).
        for k in nulls:
            L.append(f"> ⚠ {disp[k]['label']}: no `json_path` in stages.yaml "
                     "(config.yaml `metric_sources` missing this metric)")
        if nulls: L.append("")
    dirty = " (dirty)" if plan.get("dirty") else ""
    diff = " — see `workspace.diff`" if plan.get("dirty") else ""
    v = pre.get("versions", {}) or {}
    L += ["## Provenance", "",
          f"- Model: {plan.get('model', '?')}",
          f"- Branch: {plan.get('git_branch', '?')} "
          f"@ {plan.get('git_sha', '?')[:8]}{dirty}{diff}",
          f"- Venv Python: {plan.get('venv_python', '?')} "
          f"({plan.get('venv_source', '?')})",
          f"- sglang {v.get('sglang', '?')} · torch {v.get('torch', '?')}",
          f"- GPU: {pre.get('gpu_summary', '?')}",
          f"- tune-ci-thresholds v{__version__}",
          f"- Ran {plan.get('timestamp', '?')} – {now_iso()}"]
    (run_dir / "report.md").write_text("\n".join(L) + "\n")
    print(f"report written to {run_dir / 'report.md'}")
    return 0


# ---- apply-plan ---------------------------------------------------------
# Emits a JSON describing, for every metric in the run's stages: the
# worst-of-N raw value (already rounded to display.digits), the current
# threshold literal in the test file, the source kind ("bare" vs
# "nested"), and a direction tag ("tightens" / "loosens" / "equal" /
# "unknown"). The skill consumes this to drive the apply UX (modes 1/2/3
# in SKILL.md). Read-only — no files are edited here.

_SOURCE_BARE_RE = re.compile(r"^([A-Z][A-Z0-9_]*)$")
_SOURCE_NESTED_RE = re.compile(
    r"^(_?[A-Z][A-Z0-9_]*)\[\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*\]$")


def _parse_source(src):
    m = _SOURCE_BARE_RE.match(src)
    if m: return ("bare", m.group(1), None)
    m = _SOURCE_NESTED_RE.match(src)
    if m: return ("nested", m.group(1), m.group(2))
    return ("unknown", src, None)


def _read_concurrency(text):
    m = re.search(r"^CONCURRENCY\s*=\s*(\d+)\s*$", text, re.M)
    return int(m.group(1)) if m else None


def _read_bare_value(text, symbol):
    m = re.search(rf"^{re.escape(symbol)}\s*=\s*([+-]?\d+(?:\.\d+)?)\s*$",
                  text, re.M)
    return float(m.group(1)) if m else None


def _read_nested_value(text, symbol, conc, subkey):
    m = re.search(rf"^{re.escape(symbol)}\s*=\s*\{{", text, re.M)
    if not m or conc is None: return None
    # Walk braces from the opening `{` to find the matching close.
    i, depth, end = m.end() - 1, 0, None
    while i < len(text):
        ch = text[i]
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1; break
        i += 1
    if end is None: return None
    block = text[m.start():end]
    sub = re.search(rf"\b{conc}\s*:\s*\{{(.*?)\}}", block, re.S)
    if not sub: return None
    val = re.search(
        rf"['\"]{re.escape(subkey)}['\"]\s*:\s*([+-]?\d+(?:\.\d+)?)",
        sub.group(1))
    return float(val.group(1)) if val else None


def _classify_direction(worst_op, current, new):
    if current is None or new is None: return "unknown"
    if abs(current - new) < 1e-12: return "equal"
    if worst_op == "min":
        return "tightens" if new > current else "loosens"
    if worst_op == "max":
        return "tightens" if new < current else "loosens"
    return "unknown"


def _ceil_wer_reference(worst_raw: float) -> float:
    """Ceil worst-of-N WER to 4 decimal places (max-bound must not tighten)."""
    return math.ceil(worst_raw * 10000) / 10000


def _apply_write_value(worst_op: str, worst_raw: float | None,
                       worst_rounded: float | None,
                       stage_group: str | None) -> float | None:
    """Return the literal to write into a test file.

    WER max-bound references are ceiled to 4 dp; accuracy uses worst_raw.
    For speed, use worst_rounded unless that would tighten beyond worst_raw.
    """
    if worst_raw is None:
        return None
    if stage_group == "wer":
        return _ceil_wer_reference(worst_raw)
    if stage_group == "reliability":
        return math.ceil(worst_raw)
    if stage_group in ("accuracy", "similarity"):
        return worst_raw
    if worst_rounded is None:
        return worst_raw
    if worst_op == "min" and worst_rounded > worst_raw:
        return worst_raw
    if worst_op == "max" and worst_rounded < worst_raw:
        return worst_raw
    return worst_rounded


def apply_plan(run_dir):
    plan = json.loads((run_dir / "plan.json").read_text())
    sy = Path(plan.get("stages_yaml")
              or stages_path(plan.get("model", DEFAULT_MODEL)))
    all_stages = _load_yaml(sy)
    N = plan["repeats"]
    out = {"model": plan["model"], "run_dir": str(run_dir),
           "repeats": N, "stages": []}
    for sk in plan["stages"]:
        s = all_stages[sk]
        if not s["metrics"]:  # docs stage
            continue
        test_path = REPO_ROOT / s["test"]
        text = test_path.read_text()
        conc = _read_concurrency(text)
        per_run = []
        for k in range(1, N + 1):
            p = run_dir / sk / f"run{k}.json"
            per_run.append(json.loads(p.read_text()) if p.exists() else None)
        sg = {"stage_key": sk, "test": str(test_path), "title": s["title"],
              "stage_group": s.get("group"), "concurrency": conc,
              "metrics": []}
        for mk, m in s["metrics"].items():
            kind, sym, sub = _parse_source(m["source"])
            display = m.get("display", {})
            digits = display.get("digits", 4)
            worst_op = m["worst"]
            vals = []
            for r in per_run:
                if r and r.get("metrics") is not None:
                    v = r["metrics"].get(mk)
                    if v is not None: vals.append(v)
            if vals:
                worst = (min if worst_op == "min" else max)(vals)
                worst_rounded = round(worst, digits)
            else:
                worst, worst_rounded = None, None
            write_value = _apply_write_value(
                worst_op, worst, worst_rounded, s.get("group"))
            if kind == "bare":
                cur = _read_bare_value(text, sym)
            elif kind == "nested":
                cur = _read_nested_value(text, sym, conc, sub)
            else:
                cur = None
            direction = _classify_direction(worst_op, cur, write_value)
            sg["metrics"].append({
                "metric_key": mk,
                "source": m["source"],
                "source_kind": kind,
                "symbol": sym,
                "subkey": sub,
                "stage_group": s.get("group"),
                "worst_op": worst_op,
                "per_run_raw": vals,
                "worst_raw": worst,
                "worst_rounded": worst_rounded,
                "write_value": write_value,
                "digits": digits,
                "scale": display.get("scale", 1),
                "label": display.get("label", mk),
                "current_raw": cur,
                "direction": direction,
            })
        out["stages"].append(sg)
    print(json.dumps(out, indent=2))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="tune.py")
    p.add_argument("--venv-python")
    p.add_argument("--skip-version-check", action="store_true")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"model config name under models/ (default: {DEFAULT_MODEL})")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("models-list")
    sub.add_parser("stages-list")
    sa = sub.add_parser("precheck"); sa.add_argument("--output-dir")
    sb = sub.add_parser("discover"); sb.add_argument("--output")
    sb.add_argument("--stage")
    sc = sub.add_parser("run")
    sc.add_argument("--stages", default="ALL")
    sc.add_argument("--repeats", type=int, default=5)
    sc.add_argument("--output-dir")
    sc.add_argument("--resume", action="store_true")
    sc.add_argument("--skip-precheck", action="store_true")
    sc.add_argument("--stages-yaml")
    sc.add_argument("--max-passes", type=int, default=_DEFAULT_CALIBRATION_PASSES,
                    help="max retry passes until all stage-runs have complete metrics")
    sd = sub.add_parser("report"); sd.add_argument("--run-dir", required=True)
    se = sub.add_parser("apply-plan"); se.add_argument("--run-dir", required=True)
    sf = sub.add_parser("status"); sf.add_argument("--run-dir", required=True)
    args = p.parse_args(argv)
    if args.cmd == "models-list":
        for m in available_models(): print(m)
        return 0
    if args.cmd == "report":
        return report(Path(args.run_dir))
    if args.cmd == "apply-plan":
        return apply_plan(Path(args.run_dir))
    if args.cmd == "status":
        return status_cmd(Path(args.run_dir))
    cfg = load_model_config(args.model)
    if args.cmd == "stages-list":
        return stages_list(cfg)
    py, src, tried = resolve_venv(args.venv_python, cfg)
    if args.cmd == "precheck":
        o = Path(args.output_dir) if args.output_dir else None
        if o: o.mkdir(parents=True, exist_ok=True)
        return precheck(py, src, o, args.skip_version_check, cfg, tried=tried)
    if args.cmd == "discover":
        out = Path(args.output) if args.output else stages_path(cfg["name"])
        return discover(out, args.stage, cfg)
    if args.cmd == "run":
        args.venv_python = py
        return run_cmd(args)
    p.print_help(); return 2


if __name__ == "__main__":
    sys.exit(main())
