#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Observe CI metrics over N runs on the H100 repro host.
Emits worst-of-N markdown. Does NOT propose thresholds or edit tests.
Model-agnostic: pass --model <name>; config comes from
models/<name>/config.yaml. Metrics come from result JSONs that tests
already write under pytest's --basetemp (set fresh per run).
"""
from __future__ import annotations
import argparse, ast, datetime as dt, hashlib, json, math, os, platform, re, shutil, signal
import statistics, subprocess, sys, time, tomllib
from pathlib import Path

__version__ = "0.7.0"

SKILL_DIR = Path(__file__).resolve().parent
MODELS_DIR = SKILL_DIR / "models"
HOSTS_DIR = SKILL_DIR / "hosts"
DEFAULT_MODEL = "omni"
_SPEAKER_SIM_MIN_BYTES = 100 * 1024 * 1024
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
_MAX_RUN_ATTEMPTS = 4  # infra-failure retries (OOM/crash/GPU-not-clear) to obtain one clean repeat; calibration-specific, unrelated to CI's per-test failure retry
_DEFAULT_CALIBRATION_PASSES = 10
_AGENT_POLL_INTERVAL_S = 120

# Destructive-observation rejection.
#
# A pytest round can complete with full sample scope and non-null metrics and
# still be worthless: host contention, a cold autotune cache, or a thrashing
# server produce numbers that describe the machine, not the model. Feeding one
# such round into strict worst-of-N sets the CI reference from the accident —
# observed inflation up to 4.53x on 2026-08-01.
#
# A value is destructive only when BOTH hold:
#   * robust z (MAD) above _DESTRUCTIVE_Z — it is far from the centre; and
#   * it is separated from its nearest neighbour by a real gap.
# The gap test is what separates a broken round from the tail of a small
# sample. At n=5, MAD alone flags ordinary tail points: on the 2026-08-01 data
# it fired in every round of the 27-metric serving unit, which would make the
# reject-and-replace loop non-terminating.
#
# Rejection is per ROUND, not per metric: a round whose speed collapsed cannot
# be trusted for accuracy either, so any single destructive metric discards the
# whole round for every stage in that pytest invocation.
_DESTRUCTIVE_Z = 3.5
_DESTRUCTIVE_GAP = 0.20
_DESTRUCTIVE_MIN_OBS = 5      # need this many values before judging outliers
_DESTRUCTIVE_FULL_RERUN_N = 3  # n>=this: the "others agree" premise is gone
# Two comparable populations this far apart mean the robust centre has moved
# into one of them and outlier identification has inverted. Deliberately much
# larger than _DESTRUCTIVE_GAP: mild bimodality is a noisy stage (surfaced by
# the speed-health check), not a detector failure.
_DEGENERATE_SPLIT = 0.50
_DESTRUCTIVE_MAX_RESTARTS = 1
_DESTRUCTIVE_MAX_ROUNDS = 15
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
    # Wipe only this job's FlashInfer JIT dir so kernels recompile cleanly.
    # Concurrent calibration groups must use distinct XDG_CACHE_HOME / HOME
    # partitions; never delete every candidate path (that races live workers).
    env = env or os.environ
    cache_dirs = _flashinfer_cache_dirs(env)
    if not cache_dirs:
        return
    shutil.rmtree(cache_dirs[0], ignore_errors=True)


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
    "cer_percent":          dict(worst="max", label="CER (%)",               digits=2, scale=1,   group="diarization"),
    "cer_no_spk_percent":   dict(worst="max", label="CER no-spk (%)",        digits=2, scale=1,   group="diarization"),
    "cp_cer_percent":       dict(worst="max", label="cpCER (%)",             digits=2, scale=1,   group="diarization"),
    "cer_no_spk_cp_valid_percent": dict(worst="max", label="CER no-spk cp-valid (%)", digits=2, scale=1, group="diarization"),
    "delta_cer_percent":    dict(worst="max", label="Delta CER (%)",         digits=2, scale=1,   group="diarization"),
    "cer_no_spk_below_50_corpus": dict(worst="max", label="CER no-spk ≤50% corpus (%)", digits=2, scale=1, group="diarization"),
    "n_above_50_pct_cer":   dict(worst="max", label="Samples >50% CER",      digits=0, scale=1,   group="diarization"),
    "speaker_timestamp_der_percent": dict(worst="max", label="Speaker-timestamp DER (%)", digits=2, scale=1, group="diarization"),
    "cer_valid_samples":    dict(worst="min", label="CER valid samples",     digits=0, scale=1,   group="diarization"),
    "cp_cer_valid_samples": dict(worst="min", label="cpCER valid samples",   digits=0, scale=1,   group="diarization"),
    "throughput_qps":       dict(worst="min", label="Throughput (req/s)",    digits=3, scale=1,   group="speed"),
    "output_tok_per_req_s": dict(
        worst="min", label="Output tok/req-s", digits=1, scale=1, group="speed"
    ),
    "completion_tokens_min": dict(
        worst="min", label="Completion tokens min", digits=0, scale=1, group="speed"
    ),
    "audio_duration_min_s": dict(
        worst="min", label="Audio duration min (s)", digits=3, scale=1, group="speed"
    ),
    "prompt_tokens_min": dict(
        worst="min", label="Prompt tokens min", digits=0, scale=1, group="speed"
    ),
    "latency_mean_s":       dict(worst="max", label="Latency mean (s)",      digits=3, scale=1,   group="speed"),
    "latency_p95_s":        dict(worst="max", label="Latency p95 (s)",       digits=3, scale=1,   group="speed"),
    "latency_max_s":        dict(worst="max", label="Latency max (s)",       digits=3, scale=1,   group="speed"),
    "rtf_mean":             dict(worst="max", label="RTF mean",              digits=4, scale=1,   group="speed"),
    "rtf_p95":              dict(worst="max", label="RTF p95",               digits=4, scale=1,   group="speed"),
    "ttfa_p95_s":           dict(worst="max", label="TTFA p95 (s)",          digits=4, scale=1,   group="speed"),
    "text_ttft_p95_s":      dict(worst="max", label="Text TTFT p95 (s)",     digits=4, scale=1,   group="speed"),
    "inter_chunk_p95_s":    dict(worst="max", label="Inter-chunk p95 (s)",   digits=4, scale=1,   group="speed"),
    "failed_requests":      dict(worst="max", label="Failed requests",       digits=0, scale=1,   group="reliability"),
    "similarity_mean":      dict(worst="min", label="Speaker sim mean",      digits=4, scale=1,   group="similarity"),
    "utmos_mean":           dict(worst="min", label="UTMOS mean",            digits=4, scale=1,   group="utmos"),
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

# Assertion literals that stay hand-pinned. Discover must not emit them as
# calibration metrics, and apply must never rewrite them from worst-of-N.
# MOSS streaming n_above_50 is too unstable for worst-of-N; keep the test
# constant fixed (currently 31) across calibration cycles.
_FIXED_THRESHOLD_SYMBOLS = frozenset({
    "MOSS_TD_STREAM_N_ABOVE_50_CER_MAX",
})


def match_metric(name, nested):
    if nested is not None:
        return nested if nested in _NESTED else None
    if name in _TTS_FIXED_PRESETS or name in _FIXED_THRESHOLD_SYMBOLS:
        return None
    if re.fullmatch(r".*_ACC(?:URACY)?_MIN", name) or re.fullmatch(r".*_MIN_ACCURACY", name):
        return "accuracy"
    # Partitioned WER (wer_below_50_corpus + n_above_50) must precede the
    # generic "WER_MAX_CORPUS" substring check to avoid false hits.
    if "WER_BELOW_50_CORPUS_MAX" in name: return "wer_below_50_corpus"
    if "N_ABOVE_50_MAX" in name: return "n_above_50"
    if name == "TTS_MAX_FAILED_REQUESTS" or "MAX_FAILED_REQUESTS" in name:
        return "failed_requests"
    if "CER_NO_SPK_CP_VALID_PERCENT" in name:
        return "cer_no_spk_cp_valid_percent"
    if "CER_NO_SPK_BELOW_50_PERCENT" in name:
        return "cer_no_spk_below_50_corpus"
    if "CER_NO_SPK_PERCENT" in name:
        return "cer_no_spk_percent"
    if "CP_CER_PERCENT" in name:
        return "cp_cer_percent"
    if "DELTA_CER_PERCENT" in name:
        return "delta_cer_percent"
    # Non-streaming uses *_N_ABOVE_50_CER_REF (+ slack-derived MAX).
    # Streaming MAX is in _FIXED_THRESHOLD_SYMBOLS and never matches here.
    # Match REF so discover/apply calibrate the reference; slack-derived MAX
    # is skipped separately via _slack_derived_threshold_names.
    if "N_ABOVE_50_CER_REF" in name or "N_ABOVE_50_CER_MAX" in name:
        return "n_above_50_pct_cer"
    # DER calibrates the reference constant (test derives the MAX via slack),
    # so match the *_REF symbol; the computed *_MAX literal is left unmatched.
    if "SPEAKER_TIMESTAMP_DER_PERCENT_REF" in name:
        return "speaker_timestamp_der_percent"
    if "CP_CER_VALID_SAMPLES_MIN" in name:
        return "cp_cer_valid_samples"
    if "CER_VALID_SAMPLES_MIN" in name:
        return "cer_valid_samples"
    if "CER_PERCENT" in name:
        return "cer_percent"
    if "SIMILARITY" in name and name.endswith("_MIN"):
        return "similarity_mean"
    if "UTMOS" in name and name.endswith("_REFERENCE"):
        return "utmos_mean"
    if "WER_MAX_CORPUS" in name: return "corpus_wer"
    if "WER_MAX_PER_SAMPLE" in name: return "per_sample_wer_max"
    if "CORPUS_WER_MAX" in name: return "corpus_wer"
    if "SAMPLE_WER_MAX" in name: return "per_sample_wer_max"
    if "THROUGHPUT_QPS" in name or "THROUGHPUT_MIN" in name:
        return "throughput_qps"
    if "OUTPUT_TOK_PER_REQ" in name:
        return "output_tok_per_req_s"
    if "PROMPT_TOKENS" in name and "MIN" in name:
        return "prompt_tokens_min"
    if "COMPLETION_TOKENS" in name and "MIN" in name:
        return "completion_tokens_min"
    if "AUDIO_DURATION" in name and "MIN" in name:
        return "audio_duration_min_s"
    if "LATENCY_MEAN" in name:
        return "latency_mean_s"
    if "LATENCY_P95" in name:
        return "latency_p95_s"
    if "LATENCY_MAX" in name:
        return "latency_max_s"
    if "RTF_MEAN" in name:
        return "rtf_mean"
    if "RTF_P95" in name:
        return "rtf_p95"
    if "TTFA" in name and "P95" in name:
        return "ttfa_p95_s"
    if "TEXT_TTFT" in name and "P95" in name:
        return "text_ttft_p95_s"
    if "INTER_CHUNK" in name and "P95" in name:
        return "inter_chunk_p95_s"
    return None


def now_iso(): return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def ts_dir():  return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
def sha256(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def available_models():
    if not MODELS_DIR.exists(): return []
    return sorted(d.name for d in MODELS_DIR.iterdir()
                  if d.is_dir() and (d / "config.yaml").exists())


def available_hosts():
    if not HOSTS_DIR.exists():
        return []
    return sorted(p.stem for p in HOSTS_DIR.glob("*.yaml"))


def load_host_profile(name: str | None = None) -> dict | None:
    """Load hosts/<name>.yaml, or autodetect by socket.gethostname()."""
    import socket

    if name is None:
        name = os.environ.get("TUNE_HOST")
    if name:
        path = HOSTS_DIR / f"{name}.yaml"
        if not path.exists():
            raise SystemExit(
                f"host profile not found: {path} "
                f"(known: {', '.join(available_hosts()) or 'none'})")
        return _load_yaml(path, top_is_dict=True)
    if not HOSTS_DIR.exists():
        return None
    hostname = socket.gethostname()
    for path in HOSTS_DIR.glob("*.yaml"):
        profile = _load_yaml(path, top_is_dict=True)
        if profile.get("hostname") == hostname:
            return profile
    return None


def resolve_repo_root(host: dict | None) -> Path:
    if os.environ.get("TUNE_REPO_ROOT"):
        return Path(os.environ["TUNE_REPO_ROOT"])
    if host and host.get("repo_root"):
        return Path(host["repo_root"])
    default = Path("/sgl-workspace/sglang-omni")
    if default.exists():
        return default
    return Path(__file__).resolve().parents[3]


def apply_host_profile(cfg: dict, host: dict) -> None:
    """Map host physical paths onto tune.py auto_env (no symlinks required)."""
    physical = host.get("physical") or {}
    auto = cfg.setdefault("auto_env", {})
    if physical.get("hf_hub"):
        auto["HF_HOME"] = str(physical["hf_hub"])
    if physical.get("speaker_sim"):
        auto["SEEDTTS_SIM_CACHE_DIR"] = str(physical["speaker_sim"])
    if physical.get("omni_ci_home"):
        cfg["omni_ci_home"] = str(physical["omni_ci_home"])
    cfg["_host"] = host


def _speaker_sim_assets_ok(cache_dir: Path) -> tuple[bool, str]:
    marker = cache_dir / ".complete"
    base = cache_dir / "wavlm_large.pt"
    finetune = cache_dir / "wavlm_large_finetune.pth"
    if not marker.is_file():
        return False, "missing .complete — run speaker_similarity_assets --warm-cache"
    for path in (base, finetune):
        if not path.is_file():
            return False, f"missing {path.name}"
        if path.stat().st_size < _SPEAKER_SIM_MIN_BYTES:
            return False, f"{path.name} too small (truncated?)"
    return True, "wavlm_large.pt + wavlm_large_finetune.pth"


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


def load_model_config(name, host: dict | None = None):
    p = model_dir(name) / "config.yaml"
    if not p.exists():
        raise SystemExit(f"model config not found: {p}. "
                         f"Known models: {available_models()}")
    cfg = _load_yaml(p, top_is_dict=True)
    cfg.setdefault("extra_env", {})
    cfg.setdefault("auto_env", {})
    cfg.setdefault("metric_sources", {})
    cfg.setdefault("hf_model_ids_by_test", {})
    if host:
        apply_host_profile(cfg, host)
    # Auto-apply env vars so the user doesn't need to export them
    # manually. Overrides any pre-existing value to match CI / host profile.
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
    # Prefer importlib.metadata so precheck does not import heavy packages
    # (importing sglang touches CUDA and can be SIGKILL'd by a sibling group's
    # scoped GPU cleanup during parallel calibration).
    r = subprocess.run(
        [
            py,
            "-c",
            (
                "import importlib.metadata as m\n"
                f"print(m.version({mod!r}))\n"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip()
    r = subprocess.run(
        [py, "-c", f"import {mod};print({mod}.__version__)"],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    if r.returncode:
        raise RuntimeError(f"{mod} version read failed: {r.stderr.strip()}")
    return r.stdout.strip()


def git_info():
    q = lambda c: subprocess.run(c, cwd=REPO_ROOT, capture_output=True,
                                  text=True, check=False).stdout.strip()
    return dict(sha=q(["git", "rev-parse", "HEAD"]),
                branch=q(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
                dirty=bool(q(["git", "status", "--porcelain"])))


def _command_output(cmd: list[str], timeout: int = 30) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=timeout
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def environment_fingerprint(py: str, cfg: dict, versions: dict) -> dict:
    """Capture enough runtime identity to compare calibration with CI."""
    freeze = _command_output([py, "-m", "pip", "freeze"], timeout=120)
    gpu_csv = _command_output([
        "nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ])
    topology = _command_output(["nvidia-smi", "topo", "-m"])
    env_keys = (
        "HOME", "OMNI_CI_HOME", "HF_HOME", "HF_HUB_DISABLE_XET",
        "XDG_CACHE_HOME", "HF_ENDPOINT", "TORCHINDUCTOR_CACHE_DIR",
        "FLASHINFER_DISABLE_VERSION_CHECK", "SEEDTTS_SIM_CACHE_DIR",
        "TUNE_GPU_INCLUDE", "TUNE_GPU_EXCLUDE", "LD_LIBRARY_PATH",
    )
    image_digest = (
        os.environ.get("OMNI_CI_IMAGE_DIGEST")
        or os.environ.get("CONTAINER_IMAGE_DIGEST")
    )
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    hub = hf_home if hf_home.name == "hub" else hf_home / "hub"

    def cached_revisions(repo_id: str, repo_type: str = "model") -> list[str]:
        prefix = "datasets--" if repo_type == "dataset" else "models--"
        snapshots = hub / f"{prefix}{repo_id.replace('/', '--')}" / "snapshots"
        return sorted(path.name for path in snapshots.glob("*") if path.is_dir())

    model_ids = _all_model_ids(cfg)
    dataset_ids = cfg.get("hf_datasets", [])
    return dict(
        captured_at=now_iso(), hostname=platform.node(), platform=platform.platform(),
        container_image=os.environ.get("OMNI_CI_IMAGE"),
        container_image_digest=image_digest,
        image_identity_status="verified" if image_digest else "unverified",
        python=_command_output([py, "--version"]), executable=str(Path(py).resolve()),
        package_versions=versions,
        dependency_freeze_sha256=(
            hashlib.sha256(freeze.encode()).hexdigest() if freeze else None
        ),
        dependency_freeze=freeze.splitlines(),
        gpu_inventory=gpu_csv.splitlines(), topology=topology,
        gpu_group=calibration_gpu_pool(cfg.get("_host")),
        environment={key: os.environ.get(key) for key in env_keys},
        model_ids=model_ids, dataset_ids=dataset_ids,
        model_revisions={repo: cached_revisions(repo) for repo in model_ids},
        dataset_revisions={repo: cached_revisions(repo, "dataset") for repo in dataset_ids},
        cache_state="recorded" if hub.exists() else "unknown",
        comparability=(
            "core-pins-match; image-digest-unverified"
            if not image_digest else "core-pins-and-image-identity-recorded"
        ),
    )


def _plan_calibration_sha(plan: dict) -> str:
    """Git commit this calibration session is bound to."""
    return plan.get("calibration_git_sha") or plan.get("git_sha") or ""


def _ast_without_numbers(source: str) -> str | None:
    """AST dump with every numeric literal blanked, or None if unparseable."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant)
                and isinstance(node.value, (int, float))
                and not isinstance(node.value, bool)):
            node.value = 0
    return ast.dump(tree)


def _git_show(sha: str, path: str) -> str | None:
    r = subprocess.run(["git", "show", f"{sha}:{path}"], cwd=REPO_ROOT,
                       capture_output=True, text=True, check=False)
    return r.stdout if r.returncode == 0 else None


def measurement_equivalent_commits(a: str, b: str) -> tuple[bool, list[str]]:
    """True when nothing between two commits can change a measured metric.

    Threshold constants and this skill's own files do not affect what the
    benchmarks measure — only whether an assertion passes. Refusing to reuse
    observations across such a commit throws away hours of valid GPU time for
    no integrity gain. Anything else (logic, non-Python files) is treated as
    a real change and blocks reuse.
    """
    if a == b:
        return True, []
    r = subprocess.run(["git", "diff", "--name-only", f"{a}..{b}"],
                       cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        return False, ["git diff failed (unrelated histories?)"]
    reasons = []
    for path in [p for p in r.stdout.split() if p]:
        if path.startswith(".claude/skills/"):
            continue                      # calibration tooling, not measured code
        if not path.endswith(".py"):
            reasons.append(f"{path}: non-Python change")
            continue
        old, new = _git_show(a, path), _git_show(b, path)
        if old is None or new is None:
            reasons.append(f"{path}: added or removed")
            continue
        da, db = _ast_without_numbers(old), _ast_without_numbers(new)
        if da is None or db is None or da != db:
            reasons.append(f"{path}: logic changed")
    return (not reasons), reasons


def audit_git_provenance(run_dir: Path, plan=None) -> dict:
    """Every run{k}.json must record the same git_sha as plan calibration_git_sha."""
    plan = plan or json.loads((run_dir / "plan.json").read_text())
    cal_sha = _plan_calibration_sha(plan)
    if not cal_sha:
        return dict(
            ok=False,
            calibration_git_sha="",
            missing_sha=[],
            mismatches=[],
            reason="plan.json has no calibration_git_sha",
        )
    # Rounds produced after a measurement-equivalent commit are still the same
    # experiment; only threshold constants or tooling moved underneath them.
    accepted = {cal_sha} | set(plan.get("equivalent_commits") or [])
    missing_sha = []
    mismatches = []
    repeats = plan["repeats"]
    for sk in plan["stages"]:
        # Replacement rounds extend past `repeats`; provenance covers them too.
        for k in (_round_indices(run_dir, [sk]) or list(range(1, repeats + 1))):
            p = run_dir / sk / f"run{k}.json"
            if not p.exists():
                continue
            try:
                data = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                mismatches.append(f"{sk}/run{k}: corrupt json")
                continue
            run_sha = data.get("git_sha")
            if not run_sha:
                missing_sha.append(f"{sk}/run{k}")
            elif run_sha not in accepted:
                mismatches.append(
                    f"{sk}/run{k}: artifact {run_sha[:8]} != calibration {cal_sha[:8]}"
                )
    ok = not missing_sha and not mismatches
    reason = ""
    if missing_sha:
        reason = (
            f"{len(missing_sha)} run(s) missing git_sha "
            "(pre-v0.6.0 or stale — purge and re-run on current commit)"
        )
    elif mismatches:
        reason = f"{len(mismatches)} run(s) recorded on a different commit"
    return dict(
        ok=ok,
        calibration_git_sha=cal_sha,
        missing_sha=missing_sha,
        mismatches=mismatches,
        reason=reason,
    )


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


def _required_model_ids_for_stages(cfg, all_stages, stage_keys):
    by_test = cfg.get("hf_model_ids_by_test") or {}
    model_ids = []
    for sk in stage_keys:
        stage = all_stages[sk]
        stage_model_ids = stage.get("hf_model_ids")
        if stage_model_ids:
            model_ids.extend(stage_model_ids)
        else:
            test_name = Path(stage["test"]).name
            model_ids.extend(by_test.get(test_name) or [cfg["hf_model_id"]])
    return _unique_ordered(model_ids)


def _required_dataset_ids_for_stages(cfg, all_stages, stage_keys):
    by_test = cfg.get("hf_datasets_by_test") or {}
    dataset_map = _parse_datasets_dict()
    dataset_ids = []
    test_names = sorted({Path(all_stages[sk]["test"]).name for sk in stage_keys})
    for test_name in test_names:
        if test_name in by_test:
            dataset_ids.extend(by_test[test_name] or [])
            continue
        test_path = next(
            REPO_ROOT / all_stages[sk]["test"]
            for sk in stage_keys
            if Path(all_stages[sk]["test"]).name == test_name
        )
        try:
            _, dataset_keys = _read_test_context(test_path)
        except Exception:
            dataset_ids.extend(cfg["hf_datasets"])
            continue
        dataset_ids.extend(
            dataset_map[key] for key in dataset_keys if key in dataset_map
        )
    return _unique_ordered(dataset_ids)


def _requires_speaker_sim_for_stages(cfg, all_stages, stage_keys):
    by_test = cfg.get("requires_speaker_sim_by_test") or {}
    default = bool(cfg.get("requires_speaker_sim", cfg["name"] in ("tts", "omni")))
    test_names = {Path(all_stages[sk]["test"]).name for sk in stage_keys}
    return any(bool(by_test.get(test_name, default)) for test_name in test_names)


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


def _kill_calibration_gpu_processes(gpu_indices: list[int] | tuple[int, ...]) -> None:
    """Kill processes only on GPUs owned by the current pytest invocation.

    Calibration groups may run concurrently under the same Unix user. A caller
    must therefore provide the exact physical GPU indices it owns; global
    process-pattern kills are never safe here.
    """
    targets = sorted(set(int(i) for i in gpu_indices))
    if not targets:
        return
    script = REPO_ROOT / ".github/scripts/delete_gpu_process.sh"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, targets))
    if script.exists():
        # Keep stdout/stderr visible so cross-group mis-kills are auditable.
        subprocess.run(
            ["bash", str(script), "--kill-orphans"],
            check=False,
            env=env,
        )


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


def _ready_gpu_indices(gpus_needed: int, host: dict | None = None):
    """GPU indices with no compute app and memory <= _GPU_RETRY_MEM_MIB."""
    mem = _gpu_memory_by_index()
    busy = busy_gpu_indices()
    excluded = excluded_gpu_indices(host)
    pool = set(calibration_gpu_pool(host))
    ready = [
        idx for idx in sorted(mem)
        if idx in pool
        and idx not in excluded
        and idx not in busy
        and mem[idx] <= _GPU_RETRY_MEM_MIB
    ]
    return ready, mem, busy


def _ensure_gpus_free(gpus_needed: int, timeout: int = _GPU_WAIT_TIMEOUT_S,
                      host: dict | None = None,
                      target_gpus: list[int] | None = None) -> bool:
    """Wait for free GPUs, cleaning only an explicitly owned GPU group."""
    if target_gpus:
        _kill_calibration_gpu_processes(target_gpus)
    time.sleep(3)
    waited = 0
    last_log = -30
    while waited < timeout:
        ready, mem, busy = _ready_gpu_indices(gpus_needed, host)
        if len(ready) >= gpus_needed:
            picked_mem = {i: mem[i] for i in ready[:gpus_needed]}
            print(f"  GPU ready for launch {picked_mem} MiB (each <= "
                  f"{_GPU_RETRY_MEM_MIB}) after {waited}s")
            return True
        if waited - last_log >= 30:
            excluded = sorted(excluded_gpu_indices(host))
            print(f"  waiting: need {gpus_needed} GPU(s) each <= "
                  f"{_GPU_RETRY_MEM_MIB} MiB ({waited}s/{timeout}s): "
                  f"mem={mem} busy={sorted(busy)} ready={len(ready)} "
                  f"pool={calibration_gpu_pool(host)} excluded={excluded}")
            last_log = waited
        time.sleep(_GPU_WAIT_POLL_S)
        waited += _GPU_WAIT_POLL_S
        if target_gpus and waited % 60 == 0:
            _kill_calibration_gpu_processes(target_gpus)
    time.sleep(5)
    ready, mem, busy = _ready_gpu_indices(gpus_needed, host)
    if len(ready) >= gpus_needed:
        print(f"  GPU ready after final scoped cleanup: "
              f"{ {i: mem[i] for i in ready[:gpus_needed]} } MiB")
        return True
    print(f"  error: cannot launch — need {gpus_needed} GPU(s) each <= "
          f"{_GPU_RETRY_MEM_MIB} MiB; after {timeout}s mem={mem} "
          f"busy={sorted(busy)} ready={len(ready)}")
    return False


def _pick_gpus_for_launch(gpus_needed: int, label: str,
                          host: dict | None = None) -> tuple[list[int] | None, str]:
    """Select GPUs only after _ensure_gpus_free; abort if any picked GPU >= 2 GiB."""
    pinned = included_gpu_indices(host)
    owned = sorted(pinned) if pinned is not None else None
    if not _ensure_gpus_free(gpus_needed, host=host, target_gpus=owned):
        return None, "GPU memory not released after cleanup"
    picked, err = pick_free_gpus(gpus_needed, host)
    if picked is None:
        if not _ensure_gpus_free(gpus_needed, host=host, target_gpus=owned):
            return None, err or "GPU memory not released after cleanup"
        picked, err = pick_free_gpus(gpus_needed, host)
    if picked is None:
        return None, err or "no GPU under 2 GiB memory limit"
    if not _picked_gpus_under_limit(picked):
        snap = _picked_gpus_mem_snapshot(picked)
        return None, f"picked GPUs not under {_GPU_RETRY_MEM_MIB} MiB: {snap}"
    return picked, ""


def _launch_gpu_gate(picked: list[int], gpus_needed: int, label: str,
                     host: dict | None = None) -> tuple[list[int] | None, str]:
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
    return _pick_gpus_for_launch(gpus_needed, label, host)


def _stage_metrics_complete(stage, metrics):
    """True when every metric with a json_path has a non-None value."""
    stage_metrics = stage.get("metrics") or {}
    if not stage_metrics:
        return False
    for _mk, meta in stage_metrics.items():
        if meta.get("json_path") and metrics.get(_mk) is None:
            return False
    return True


def _stage_counts_complete(stage, sample_counts):
    """True when configured sample counts are present and ok == total."""
    counts_cfg = stage.get("sample_counts") or {}
    if not counts_cfg:
        return True
    total_cfg = counts_cfg.get("total") or {}
    ok_cfg = counts_cfg.get("ok") or {}
    if not (
        total_cfg.get("json_file")
        and total_cfg.get("json_path")
        and ok_cfg.get("json_file")
        and ok_cfg.get("json_path")
    ):
        return True
    total = (sample_counts or {}).get("total")
    ok = (sample_counts or {}).get("ok")
    if total is None or ok is None or ok != total:
        return False
    expected = stage.get("expected_samples")
    if expected is not None and total != expected:
        return False
    return True


def _observation_status(
    stage,
    metrics,
    observation_valid,
    allowed_pytest_failure,
    pytest_status,
    pytest_reason,
):
    """Calibration treats a run as complete once metrics are extracted."""
    if not stage.get("metrics"):
        return pytest_status, pytest_reason
    if not _stage_metrics_complete(stage, metrics):
        return "failed", "incomplete_metrics_extraction"
    if stage.get("observation_validity"):
        if observation_valid is None:
            return "failed", "observation_validity_missing"
        if observation_valid is not True:
            return "failed", "observation_validity_failed"
    if pytest_status == "ok":
        if stage.get("allowed_pytest_failure") and allowed_pytest_failure is True:
            return "failed", "unexpected_recorded_pytest_failure"
        return "ok", ""
    if stage.get("allowed_pytest_failure"):
        if allowed_pytest_failure is None:
            return "failed", "allowed_pytest_failure_missing"
        if allowed_pytest_failure is not True:
            return "failed", f"unexpected_pytest_failure ({pytest_reason})"
        if pytest_reason != "exit 1":
            return "failed", pytest_reason
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
        if data.get("status") != "ok":
            return False
        if not (
            _stage_metrics_complete(stage, data.get("metrics") or {})
            and _stage_counts_complete(stage, data.get("sample_counts") or {})
        ):
            return False
        if stage.get("observation_validity"):
            return data.get("observation_valid") is True
        return True
    return data.get("status") == "ok"


def _purge_incomplete_run(out: Path, stage_keys, all_stages, k: int):
    for sk in stage_keys:
        stage = all_stages[sk]
        p = out / sk / f"run{k}.json"
        if p.exists() and not _run_json_ok(p, stage):
            p.unlink(missing_ok=True)


def _round_indices(run_dir: Path, stage_keys) -> list[int]:
    """Every round index that produced a result json for this unit."""
    found = set()
    for sk in stage_keys:
        for p in (run_dir / sk).glob("run*.json"):
            m = re.fullmatch(r"run(\d+)\.json", p.name)
            if m:
                found.add(int(m.group(1)))
    return sorted(found)


def _display_resolution(stage: dict, metric_key: str) -> float:
    """Smallest raw difference the report would render as distinct.

    Guards the gap test on near-zero metrics, where a relative gap explodes:
    a WER moving 0.0085 -> 0.0106 is a 24% relative swing but only 0.2
    percentage points, and must not count as destructive separation.
    """
    disp = ((stage.get("metrics") or {}).get(metric_key) or {}).get("display") or {}
    digits = disp.get("digits")
    scale = disp.get("scale") or 1
    if digits is None:
        return 0.0
    try:
        return (10.0 ** -int(digits)) / float(scale)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _destructive_value_indices(values, floor: float = 0.0) -> dict[int, dict]:
    """Positions in `values` that are destructive outliers, with evidence.

    Both conditions must hold — see _DESTRUCTIVE_Z / _DESTRUCTIVE_GAP.
    """
    # A non-finite metric is a broken measurement, not a slow one. Condemn it
    # outright and keep it out of the statistics, where it would poison the
    # median and make every comparison return False.
    flagged = {i: dict(value=v, z=None, rel_gap=None, abs_gap=None,
                       rest_median=None, reason="non-finite value")
               for i, v in enumerate(values) if not math.isfinite(v)}
    finite = [(i, v) for i, v in enumerate(values) if math.isfinite(v)]
    if len(finite) < _DESTRUCTIVE_MIN_OBS:
        return flagged
    values = [v for _, v in finite]
    centre = statistics.median(values)
    mad = statistics.median([abs(v - centre) for v in values])
    for pos, (i, v) in enumerate(finite):
        rest = [x for j, x in enumerate(values) if j != pos]
        rest_centre = statistics.median(rest)
        abs_gap = min(abs(v - x) for x in rest)
        rel_gap = abs_gap / abs(rest_centre) if rest_centre else 0.0
        if mad:
            z = abs(v - centre) / (1.4826 * mad)
        else:
            z = math.inf if v != centre else 0.0
        if z <= _DESTRUCTIVE_Z:
            continue
        if rel_gap < _DESTRUCTIVE_GAP or abs_gap <= floor:
            continue
        flagged[i] = dict(value=v, z=(None if z == math.inf else round(z, 2)),
                          rel_gap=round(rel_gap, 4), abs_gap=abs_gap,
                          rest_median=rest_centre)
    return flagged


def _degenerate_series(values, floor: float = 0.0) -> tuple[float, float] | None:
    """Detect a sample that splits into two populations of comparable size.

    Beyond roughly 40% contamination the median and MAD sit *inside* the bad
    cluster, so outlier identification silently inverts and reports the good
    rounds as the anomalies. When that happens nothing in the sample can be
    trusted to say which side is real, so the caller must discard the whole
    block rather than pick a side. Returns (gap, centre) when degenerate.
    """
    if len(values) < _DESTRUCTIVE_MIN_OBS:
        return None
    ordered = sorted(values)
    gap, at = max((ordered[i + 1] - ordered[i], i)
                  for i in range(len(ordered) - 1))
    if min(at + 1, len(ordered) - at - 1) < 2:
        return None          # a lone outlier is the normal, detectable case
    centre = statistics.median(ordered)
    if not centre or gap <= floor:
        return None
    return (gap, centre) if gap / abs(centre) >= _DEGENERATE_SPLIT else None


def _unit_metric_series(run_dir: Path, stage_keys, all_stages, rounds):
    """(stage_key, metric, [(round, value)…]) for every numeric metric."""
    for sk in stage_keys:
        stage = all_stages.get(sk) or {}
        if not stage.get("metrics"):
            continue
        series = {}
        for k in rounds:
            p = run_dir / sk / f"run{k}.json"
            if not _run_json_ok(p, stage):
                continue
            for mk, mv in (json.loads(p.read_text()).get("metrics") or {}).items():
                if isinstance(mv, (int, float)) and not isinstance(mv, bool):
                    series.setdefault(mk, []).append((k, float(mv)))
        for mk, pairs in series.items():
            yield sk, stage, mk, pairs


def _detect_destructive_rounds(run_dir: Path, stage_keys, all_stages,
                               candidate_rounds=None) -> dict[int, list[dict]]:
    """Destructive round indices for one pytest unit, with per-metric evidence.

    Only rounds not already condemned are judged, and only against each other.
    An already-rejected round left in the sample would sit next to the next bad
    round and cancel its gap, so two similar broken rounds would mask each
    other and both survive.
    """
    rounds = [k for k in (candidate_rounds if candidate_rounds is not None
                          else _round_indices(run_dir, stage_keys))
              if not _round_rejected(run_dir, stage_keys, k)]
    evidence: dict[int, list[dict]] = {}
    for sk, stage, mk, pairs in _unit_metric_series(
            run_dir, stage_keys, all_stages, rounds):
        ks = [k for k, _ in pairs]
        values = [v for _, v in pairs]
        floor = _display_resolution(stage, mk)
        for pos, ev in _destructive_value_indices(values, floor).items():
            evidence.setdefault(ks[pos], []).append(
                dict(stage_key=sk, metric=mk, **ev))
    return evidence


def _detect_degenerate_unit(run_dir: Path, stage_keys, all_stages,
                            candidate_rounds=None) -> list[dict]:
    """Metrics whose remaining rounds split into two comparable populations."""
    rounds = [k for k in (candidate_rounds if candidate_rounds is not None
                          else _round_indices(run_dir, stage_keys))
              if not _round_rejected(run_dir, stage_keys, k)]
    found = []
    for sk, stage, mk, pairs in _unit_metric_series(
            run_dir, stage_keys, all_stages, rounds):
        split = _degenerate_series([v for _, v in pairs],
                                   _display_resolution(stage, mk))
        if split:
            found.append(dict(stage_key=sk, metric=mk, gap=split[0],
                              centre=split[1],
                              values=[v for _, v in pairs]))
    return found


def _mark_destructive_rounds(run_dir: Path, stage_keys, evidence,
                             block_discarded: bool = False) -> None:
    """Stamp destructive rounds across every stage of the unit.

    The json is kept, never deleted: the excluded values are evidence and the
    report has to be able to show them. `block_discarded` marks rounds thrown
    away wholesale by a restart, so a later resume can tell them apart from
    individually condemned rounds.
    """
    for k, flags in evidence.items():
        for sk in stage_keys:
            p = run_dir / sk / f"run{k}.json"
            if not p.exists():
                continue
            try:
                data = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            data["destructive"] = True
            data["destructive_evidence"] = flags
            if block_discarded:
                data["block_discarded"] = True
            p.write_text(json.dumps(data, indent=2))


def _run_json_block_discarded(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("block_discarded") is True
    except (json.JSONDecodeError, OSError):
        return False


def _run_json_destructive(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("destructive") is True
    except (json.JSONDecodeError, OSError):
        return False


def _round_rejected(run_dir: Path, stage_keys, k: int) -> bool:
    """Is round k condemned for this unit?

    Asks every stage, not one representative: rejection is stamped on all of
    them, and if the stage we happened to probe is missing that round the
    round would look clean and be re-condemned on every cycle.
    """
    return any(_run_json_destructive(run_dir / sk / f"run{k}.json")
               for sk in stage_keys)


def _round_block_discarded(run_dir: Path, stage_keys, k: int) -> bool:
    return any(_run_json_block_discarded(run_dir / sk / f"run{k}.json")
               for sk in stage_keys)


def _clean_rounds(run_dir: Path, sk: str, stage: dict,
                  rounds=None) -> list[int]:
    """Round indices usable for worst-of-N: complete and not destructive."""
    candidates = rounds if rounds is not None else _round_indices(run_dir, [sk])
    out = []
    for k in candidates:
        p = run_dir / sk / f"run{k}.json"
        if _run_json_ok(p, stage) and not _run_json_destructive(p):
            out.append(k)
    return sorted(out)


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
        rounds = _round_indices(run_dir, [sk]) or list(range(1, repeats + 1))
        clean = _clean_rounds(run_dir, sk, stage, rounds)
        # Completeness is measured in *clean* observations, capped at the
        # target: extra rounds earned by rejection do not inflate the count.
        ok += min(len(clean), repeats)
        for k in rounds:
            p = run_dir / sk / f"run{k}.json"
            if k in clean:
                continue
            if _run_json_destructive(p):
                continue  # rejected on purpose; a replacement round covers it
            reason = "missing file"
            if p.exists():
                try:
                    d = json.loads(p.read_text())
                    sc = d.get("sample_counts") or {}
                    expected = stage.get("expected_samples")
                    if (
                        expected is not None
                        and sc.get("total") is not None
                        and sc.get("total") != expected
                    ):
                        reason = (
                            f"sample scope mismatch "
                            f"(total {sc.get('total')} != expected {expected})"
                        )
                    else:
                        reason = d.get("reason") or d.get("status") or "incomplete metrics"
                except (json.JSONDecodeError, OSError):
                    reason = "corrupt json"
            missing.append({"stage_key": sk, "run": k, "reason": reason})
        shortfall = repeats - len(clean)
        if shortfall > 0:
            missing.append({
                "stage_key": sk, "run": None,
                "reason": f"needs {shortfall} more clean observation(s) "
                          f"({len(clean)}/{repeats} after destructive rejection)",
            })
    return dict(
        complete=(ok == total),
        ok=ok,
        total=total,
        missing_count=len(missing),
        missing=missing,
        repeats=repeats,
        model=plan.get("model"),
    )


def strict_classify_cell(path: Path, stage: dict) -> str:
    """Classify one stage-run for strict worst-of-N (✓ / △ / ✗ / D / —)."""
    if not path.exists():
        return "—"
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return "✗"
    if data.get("destructive") is True:
        # Complete but rejected: excluded from aggregation, replaced by a
        # fresh round rather than re-run in place.
        return "D"
    sample_counts = data.get("sample_counts") or {}
    total = sample_counts.get("total")
    ok = sample_counts.get("ok")
    metrics = data.get("metrics") or {}
    has_all = bool(metrics) and all(v is not None for v in metrics.values())
    if data.get("status") != "ok":
        return "✗"
    if not has_all:
        return "✗"
    if total is None or ok is None:
        return "✗"
    if ok != total:
        return "△"
    expected = stage.get("expected_samples")
    if expected is not None and total != expected:
        return "✗"
    if stage.get("observation_validity") and data.get("observation_valid") is not True:
        return "✗"
    return "✓"


def strict_audit(run_dir: Path, all_stages=None, plan=None) -> dict:
    """Strict worst-of-N readiness — every stage must have N full-scope ✓ repeats."""
    plan = plan or json.loads((run_dir / "plan.json").read_text())
    if all_stages is None:
        sy = Path(plan.get("stages_yaml")
                  or stages_path(plan.get("model", DEFAULT_MODEL)))
        all_stages = _load_yaml(sy)
    repeats = plan["repeats"]
    stage_rows = []
    ready = 0
    for sk in plan["stages"]:
        stage = all_stages[sk]
        # Rounds are no longer 1..repeats: destructive rounds are replaced by
        # additional ones, so effective N varies per stage.
        rounds = _round_indices(run_dir, [sk]) or list(range(1, repeats + 1))
        cells = [
            strict_classify_cell(run_dir / sk / f"run{k}.json", stage)
            for k in rounds
        ]
        strict_ok = cells.count("✓")
        if strict_ok >= repeats:
            ready += 1
        stage_rows.append(dict(
            stage_key=sk,
            cells=cells,
            rounds=rounds,
            strict_ok=strict_ok,
            effective_n=strict_ok,
            destructive=[k for k, c in zip(rounds, cells) if c == "D"],
            expected_samples=stage.get("expected_samples"),
        ))
    return dict(
        strict_ready=ready,
        total_stages=len(plan["stages"]),
        repeats=repeats,
        stages=stage_rows,
        strict_complete=(ready == len(plan["stages"])),
        git_provenance=audit_git_provenance(run_dir, plan),
    )


def strict_audit_cmd(run_dir: Path) -> int:
    plan_path = run_dir / "plan.json"
    if not plan_path.exists():
        print(f"error: no plan.json in {run_dir}")
        return 1
    plan = json.loads(plan_path.read_text())
    sy = Path(plan.get("stages_yaml")
              or stages_path(plan.get("model", DEFAULT_MODEL)))
    all_stages = _load_yaml(sy)
    audit = strict_audit(run_dir, all_stages, plan)
    git = audit["git_provenance"]
    cal_sha = git["calibration_git_sha"]
    if cal_sha:
        print(f"calibration_git_sha: {cal_sha}")
    if not git["ok"]:
        print(f"GIT PROVENANCE: FAIL — {git['reason']}")
        for item in git["mismatches"][:10]:
            print(f"  {item}")
        for item in git["missing_sha"][:10]:
            print(f"  {item}")
        if len(git["mismatches"]) > 10 or len(git["missing_sha"]) > 10:
            extra = len(git["mismatches"]) + len(git["missing_sha"]) - 10
            print(f"  … and {extra} more")
    else:
        print("GIT PROVENANCE: ok (all artifacts match calibration commit)")
    for row in audit["stages"]:
        cells = row["cells"]
        exp = row["expected_samples"]
        exp_note = f", expected={exp}" if exp is not None else ""
        print(
            f"{row['stage_key']}: {''.join(cells)} "
            f"({row['strict_ok']}/{audit['repeats']} strict{exp_note})"
        )
    print(
        f"STRICT READY: {audit['strict_ready']}/{audit['total_stages']} stages "
        f"({audit['repeats']} repeats each)"
    )
    metrics_ok = audit["strict_complete"]
    git_ok = git["ok"]
    if metrics_ok and git_ok:
        print("STRICT + GIT: READY")
    elif metrics_ok and not git_ok:
        print("STRICT metrics ok but GIT PROVENANCE failed — purge stale runs or "
              "start a fresh --output-dir on the current commit")
    return 0 if metrics_ok and git_ok else 1


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


def _parse_gpu_index_list(raw: str | list | tuple | None) -> set[int]:
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple)):
        return {int(x) for x in raw}
    out: set[int] = set()
    for part in str(raw).split(","):
        part = part.strip()
        if part.isdigit():
            out.add(int(part))
    return out


def excluded_gpu_indices(host: dict | None = None) -> set[int]:
    """GPUs calibration must never pick or kill (e.g. CI-reserved 6,7 on 8× hosts)."""
    raw = os.environ.get("TUNE_GPU_EXCLUDE")
    if raw:
        return _parse_gpu_index_list(raw)
    if host:
        return _parse_gpu_index_list(host.get("gpu_exclude"))
    return set()


def included_gpu_indices(host: dict | None = None) -> set[int] | None:
    """When set via ``TUNE_GPU_INCLUDE``, pin pick/cleanup to exactly these GPUs.

    Used for concurrent calibration sessions on shared hosts (e.g. 0,1 vs 2,3).
    """
    raw = os.environ.get("TUNE_GPU_INCLUDE")
    if not raw:
        return None
    included = _parse_gpu_index_list(raw)
    excluded = excluded_gpu_indices(host)
    pinned = included - excluded
    if not pinned:
        raise RuntimeError(
            f"TUNE_GPU_INCLUDE={raw!r} empty after excluding {sorted(excluded)}"
        )
    return pinned


def calibration_gpu_pool(host: dict | None = None) -> list[int]:
    excluded = excluded_gpu_indices(host)
    pool = [i for i in all_gpu_indices() if i not in excluded]
    pinned = included_gpu_indices(host)
    if pinned is not None:
        pool = [i for i in pool if i in pinned]
    return pool


def pick_free_gpus(n, host: dict | None = None):
    """Pick n GPUs with no compute app and memory <= _GPU_RETRY_MEM_MIB (2 GiB)."""
    pinned = included_gpu_indices(host)
    if pinned is not None and len(pinned) == n:
        picked = sorted(pinned)
        if _picked_gpus_under_limit(picked):
            return picked, None
        snap = _picked_gpus_mem_snapshot(picked)
        return None, (
            f"pinned GPU(s) {picked} not each <= {_GPU_RETRY_MEM_MIB} MiB: {snap}"
        )
    ready, mem, busy = _ready_gpu_indices(n, host)
    all_idx = calibration_gpu_pool(host)
    if len(ready) >= n:
        picked = ready[:n]
        if not _picked_gpus_under_limit(picked):
            snap = _picked_gpus_mem_snapshot(picked)
            return None, f"internal: picked GPUs exceed {_GPU_RETRY_MEM_MIB} MiB: {snap}"
        return picked, None
    pin_msg = f" pinned={sorted(pinned)}" if pinned is not None else ""
    return None, (
        f"need {n} GPU(s) each <= {_GPU_RETRY_MEM_MIB} MiB (2 GiB); "
        f"ready {len(ready)}/{len(all_idx)} mem={mem} busy={sorted(busy)}{pin_msg}"
    )


def precheck(
    py,
    src,
    out,
    skip_ver,
    cfg,
    datasets_override=None,
    model_ids_override=None,
    tried=None,
    gpu_required_override=None,
    requires_speaker_sim_override=None,
):
    errs, warns = [], []
    print(f"model: {cfg['name']}")
    print(f"venv_python: {py} ({src})")
    if src == "default" and tried and len(tried) > 1:
        print(f"  (tried in order: {', '.join(tried)})")
    # Equivalence with CI is established by: same image's venv + pinned
    # sglang/torch + cached assets + clean GPUs.
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
                 "Run these to download from Hugging Face:"]
        for repo_id, kind in missing:
            flag = " --repo-type dataset" if kind == "dataset" else ""
            lines.append(
                "  HF_ENDPOINT=https://huggingface.co "
                f"huggingface-cli download {repo_id}{flag}"
            )
        errs.append("\n".join(lines))
    sim_dir = cfg["auto_env"].get("SEEDTTS_SIM_CACHE_DIR")
    needs_speaker_sim = (
        bool(requires_speaker_sim_override)
        if requires_speaker_sim_override is not None
        else bool(cfg.get("requires_speaker_sim", cfg["name"] in ("tts", "omni")))
    )
    if sim_dir and needs_speaker_sim:
        sim_ok, sim_detail = _speaker_sim_assets_ok(Path(sim_dir))
        mark = "✓" if sim_ok else "✗"
        print(f"    {mark} speaker_sim: {sim_dir} ({sim_detail})")
        if not sim_ok:
            bootstrap = ((cfg.get("_host") or {}).get("speaker_similarity_bootstrap")
                         or {})
            cmd = (bootstrap.get("warm_cache") or {}).get("command", "").strip()
            hint = (
                f"speaker_sim incomplete at {sim_dir}: {sim_detail}. "
                "Run benchmarks.metrics.speaker_similarity_assets --warm-cache "
                f"with SEEDTTS_SIM_CACHE_DIR={sim_dir}."
            )
            if cmd:
                hint += f" See hosts profile warm_cache.command."
            errs.append(hint)
    smi = nvidia_smi_L()
    if not smi:
        errs.append("nvidia-smi -L returned no GPUs")
    else:
        pool = calibration_gpu_pool(cfg.get("_host"))
        excluded = sorted(excluded_gpu_indices(cfg.get("_host")))
        busy = busy_gpu_indices()
        ready, _, _ = _ready_gpu_indices(1, cfg.get("_host"))
        free_count = len(ready)
        summary = gpu_summary(smi)
        pinned = sorted(included_gpu_indices(cfg.get("_host")) or [])
        pool_note = f" pool={pool}" if (excluded or pinned) else ""
        pin_note = f" pinned={pinned}" if pinned else ""
        if busy or excluded or pinned:
            print(f"  GPUs: {summary} — {free_count}/{len(pool)} calibration-ready"
                  f"{pool_note}{pin_note} (busy: {sorted(busy)}"
                  f"{f', excluded: {excluded}' if excluded else ''})")
        else:
            print(f"  GPUs: {summary} — {free_count}/{len(pool)} calibration-ready")
        gpu_required = gpu_required_override
        if gpu_required is None:
            gpu_required = max((cfg.get("gpus_per_test") or {}).values(), default=1)
        if free_count < gpu_required:
            errs.append(f"need {gpu_required} free GPU(s); only {free_count} free "
                        f"(busy: {sorted(busy)}) — "
                        "free them yourself (e.g. stop your own jobs); "
                        "precheck does not kill GPU processes")
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        fingerprint = environment_fingerprint(py, cfg, versions)
        (out / "environment-fingerprint.json").write_text(
            json.dumps(fingerprint, indent=2)
        )
        (out / "precheck.json").write_text(json.dumps(dict(
            timestamp=now_iso(), model=cfg["name"],
            venv_python=py, venv_source=src, versions=versions,
            pins={"sglang": pins.get("sglang"), "torch": pins.get("torch")},
            git=gi, nvidia_smi_L=smi, gpu_summary=gpu_summary(smi),
            environment_fingerprint="environment-fingerprint.json",
            comparability=fingerprint["comparability"]), indent=2))
    return _summary(errs, warns)


def _summary(errs, warns):
    for w in warns: print(f"warning: {w}")
    for e in errs:  print(f"error: {e}")
    if errs:
        print(f"\nprecheck FAILED ({len(errs)} error(s))"); return 1
    print("\nprecheck OK"); return 0


def _constants(tree):
    for n in tree.body:
        if isinstance(n, ast.Assign):
            targets, value = n.targets, n.value
        elif isinstance(n, ast.AnnAssign) and n.value is not None:
            targets, value = [n.target], n.value
        else:
            continue
        for t in targets:
            if not (isinstance(t, ast.Name)
                    and re.fullmatch(r"_?[A-Z][A-Z0-9_]*", t.id)):
                continue
            yield (t.id, None)
            if isinstance(value, ast.Dict):
                for vv in value.values:
                    if isinstance(vv, ast.Dict):
                        for kk in vv.keys:
                            if isinstance(kk, ast.Constant) and isinstance(kk.value, str):
                                yield (t.id, kk.value)


_SLACK_HELPER_CALLS = frozenset({"apply_wer_slack", "apply_mos_slack"})
_SLACK_ENV_NAMES = frozenset({"THRESHOLD_SLACK_LOWER", "THRESHOLD_SLACK_HIGHER"})
# Prefixed variants (e.g. AISHELL4_LONG_THRESHOLD_SLACK_LOWER) are matched by
# suffix so a stage group can widen its slack without touching this module.
_SLACK_ENV_SUFFIXES = ("THRESHOLD_SLACK_LOWER", "THRESHOLD_SLACK_HIGHER")


def _expr_uses_slack(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and (
            child.id in _SLACK_ENV_NAMES or child.id.endswith(_SLACK_ENV_SUFFIXES)
        ):
            return True
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
            if child.func.id in _SLACK_HELPER_CALLS:
                return True
    return False


def _slack_derived_threshold_names(tree: ast.AST) -> set[str]:
    """Names assigned from a reference via THRESHOLD_SLACK_* or apply_*_slack().

    Calibration must write the reference literal only; CI derives assertion
    thresholds (MAX/MIN/THRESHOLD) from slack — never apply worst-of-N to these.
    """
    derived: set[str] = set()
    for n in tree.body:
        if isinstance(n, ast.Assign):
            targets, value = n.targets, n.value
        elif isinstance(n, ast.AnnAssign) and n.value is not None:
            targets, value = [n.target], n.value
        else:
            continue
        if not _expr_uses_slack(value):
            continue
        for t in targets:
            if isinstance(t, ast.Name):
                derived.add(t.id)
    return derived


def _pick_calibration_constant(candidates: list[tuple[str, str | None]]) -> tuple[str, str | None]:
    """Prefer *_REF / *_REFERENCE over bare *_MAX / *_MIN calibration targets."""
    for suffix in ("_REF", "_REFERENCE"):
        for name, nested in candidates:
            if name.endswith(suffix):
                return name, nested
    return candidates[0]


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


def _int_constant(tree, name: str) -> int | None:
    """Return int literal for ``NAME = <int>``; None if missing, non-int, or ``None``."""
    for n in tree.body:
        if isinstance(n, ast.Assign):
            targets, value = n.targets, n.value
        elif isinstance(n, ast.AnnAssign) and n.value is not None:
            targets, value = [n.target], n.value
        else:
            continue
        for t in targets:
            if isinstance(t, ast.Name) and t.id == name:
                if isinstance(value, ast.Constant):
                    if isinstance(value.value, int):
                        return value.value
                    return None
    return None


def _expected_samples(
    tree,
    group: str,
    variant: str | None,
    context_vars: list[str] | None,
) -> int | None:
    """Resolve CI sample scope from test-file constants (written to stages.yaml)."""
    for const_name in context_vars or []:
        if const_name == "CONCURRENCY":
            continue
        value = _int_constant(tree, const_name)
        if isinstance(value, int):
            return value
    if group == "similarity":
        value = _int_constant(tree, "TTS_SIMILARITY_MAX_SAMPLES")
        if isinstance(value, int):
            return value
    if variant == "stream":
        stream_cap = _int_constant(tree, "STREAMING_BENCHMARK_MAX_SAMPLES")
        if isinstance(stream_cap, int):
            return stream_cap
        fullset = _int_constant(tree, "SEEDTTS_EN_FULLSET_SAMPLES")
        if isinstance(fullset, int):
            return fullset
    if group in ("wer", "speed", "utmos"):
        value = _int_constant(tree, "SEEDTTS_EN_FULLSET_SAMPLES")
        if isinstance(value, int):
            return value
    value = _int_constant(tree, "SEEDTTS_ASR_CORRECTNESS_SAMPLES")
    if isinstance(value, int):
        return value
    return None


def _configured_expected_samples(source: dict, group: str) -> int | None:
    configured = source.get("expected_samples")
    if isinstance(configured, int):
        return configured
    if isinstance(configured, dict) and isinstance(configured.get(group), int):
        return configured[group]
    return None


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


def _build_observation_validity(raw, default_file):
    jf, jp = _split_source(raw, default_file)
    if not (jf and jp):
        return None
    return dict(json_file=jf, json_path=jp)


def _build_allowed_pytest_failure(raw, default_file):
    jf, jp = _split_source(raw, default_file)
    if not (jf and jp):
        return None
    return dict(json_file=jf, json_path=jp)


def _calibration_presets(ms, base_extra):
    """Return stage env/GPU overlays for non-random calibration presets.

    CI may randomly pick one runtime preset, but calibration must observe
    every configured preset. A metric_source can therefore declare
    `calibration_presets`, and discover expands one stage set per preset.
    """
    presets = ms.get("calibration_presets") or {}
    if not presets:
        return [(None, base_extra, None, None)]
    out = []
    for name, raw in presets.items():
        raw = raw or {}
        env = dict(base_extra)
        env.update(raw.get("extra_env") or {})
        gpus = raw.get("gpus")
        model_ids = raw.get("hf_model_ids")
        out.append((name, env, gpus, model_ids))
    return out


def _stage_key(base, group, variant=None, calibration_preset=None):
    parts = [base]
    if calibration_preset:
        parts.append(str(calibration_preset))
    if variant:
        parts.append(str(variant))
    parts.append(group)
    return "_".join(parts)


def _stage_title(base, group, variant=None, calibration_preset=None):
    parts = [base.replace("_", " ").upper()]
    if calibration_preset:
        parts.append(str(calibration_preset).upper())
    if variant:
        parts.append(str(variant).upper())
    parts.append(group.capitalize())
    return " ".join(parts)


def _stage_entry(
    rel,
    title,
    group,
    extra_env,
    context_vars,
    test_file_sha256,
    metrics,
    sample_counts,
    observation_validity=None,
    allowed_pytest_failure=None,
    variant=None,
    calibration_preset=None,
    gpus=None,
    hf_model_ids=None,
    threshold_file=None,
    threshold_file_sha256=None,
    expected_samples=None,
):
    entry = dict(
        test=rel,
        title=title,
        group=group,
        extra_env=extra_env,
        context_vars=context_vars,
        test_file_sha256=test_file_sha256,
        last_discovered_at=now_iso(),
        metrics=metrics,
        sample_counts=sample_counts,
    )
    if observation_validity:
        entry["observation_validity"] = observation_validity
    if allowed_pytest_failure:
        entry["allowed_pytest_failure"] = allowed_pytest_failure
    if expected_samples is not None:
        entry["expected_samples"] = int(expected_samples)
    if variant:
        entry["variant"] = variant
    if calibration_preset:
        entry["calibration_preset"] = calibration_preset
    if gpus:
        entry["gpus"] = int(gpus)
    if hf_model_ids:
        entry["hf_model_ids"] = hf_model_ids
    if threshold_file:
        entry["threshold_file"] = threshold_file
        entry["threshold_file_sha256"] = threshold_file_sha256
    return entry


def _emit_groups(constants, cfg_paths, default_file, counters,
                 slack_derived: set[str] | None = None):
    """Build {group: {metric_kind: metric_dict}} from a constant list."""
    slack_derived = slack_derived or set()
    by_metric: dict[str, list[tuple[str, str | None]]] = {}
    for name, nested in constants:
        if name in slack_derived or name.endswith("_THRESHOLD"):
            continue
        mk = match_metric(name, nested)
        if mk is None:
            continue
        by_metric.setdefault(mk, []).append((name, nested))
    groups: dict[str, dict] = {}
    for mk, candidates in by_metric.items():
        name, nested = _pick_calibration_constant(candidates)
        spec = METRIC_SPECS[mk]
        jf, jp = _split_source(cfg_paths.get(mk), default_file)
        status = "OK" if (jf and jp) else "NEEDS_CONFIG"
        if status == "OK":
            counters[0] += 1
        else:
            counters[1] += 1
        src = f"{name}[{nested!r}]" if nested else name
        groups.setdefault(spec["group"], {})[mk] = dict(
            source=src,
            json_file=jf,
            json_path=jp,
            worst=spec["worst"],
            display=dict(
                label=spec["label"],
                scale=spec["scale"],
                digits=spec["digits"],
            ),
            status=status,
        )
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
        threshold_rel = ms.get("threshold_file")
        threshold_tp = REPO_ROOT / threshold_rel if threshold_rel else tp
        threshold_tree = ast.parse(threshold_tp.read_text())
        threshold_sha = sha256(threshold_tp)
        threshold_file_sha = threshold_sha if threshold_rel else None
        ignored_constants = set(ms.get("ignored_constants") or [])
        slack_derived = _slack_derived_threshold_names(threshold_tree)
        all_constants = [
            (n, k) for (n, k) in _constants(threshold_tree)
            if n not in ignored_constants
        ]
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
                observation_validity = _build_observation_validity(
                    vcfg.get(
                        "observation_validity",
                        ms.get("observation_validity"),
                    ),
                    v_default,
                )
                allowed_pytest_failure = _build_allowed_pytest_failure(
                    vcfg.get(
                        "allowed_pytest_failure",
                        ms.get("allowed_pytest_failure"),
                    ),
                    v_default,
                )
                sc_by_group = vcfg.get("sample_counts_by_group") or {}
                for (
                    preset_name,
                    preset_env,
                    preset_gpus,
                    preset_model_ids,
                ) in _calibration_presets(ms, extra):
                    preset_raw = (ms.get("calibration_presets") or {}).get(
                        preset_name, {}
                    ) or {}
                    preset_filter = preset_raw.get("constant_filter")
                    if preset_filter:
                        ppat = re.compile(preset_filter)
                        preset_claimed = [
                            (n, k)
                            for (n, k) in claimed
                            if ppat.match(n.lstrip("_"))
                        ]
                    else:
                        preset_claimed = claimed
                    v_groups = _emit_groups(
                        preset_claimed, v_paths, v_default, counters,
                        slack_derived=slack_derived,
                    )
                    effective_base = vcfg.get("stage_base_override") or base
                    effective_variant = (
                        None if vcfg.get("omit_variant_suffix") else vname
                    )
                    variant_ctx = vcfg.get("context_vars") or ctx
                    for g, metrics in v_groups.items():
                        if g in sc_by_group:
                            group_sc = _build_sample_counts(
                                sc_by_group[g], v_default
                            )
                        else:
                            group_sc = v_sc_default
                        key = _stage_key(
                            effective_base,
                            g,
                            variant=effective_variant,
                            calibration_preset=preset_name,
                        )
                        stages[key] = _stage_entry(
                            rel,
                            _stage_title(
                                effective_base,
                                g,
                                variant=effective_variant,
                                calibration_preset=preset_name,
                            ),
                            g,
                            preset_env,
                            variant_ctx,
                            sha,
                            metrics,
                            group_sc,
                            observation_validity=observation_validity,
                            allowed_pytest_failure=allowed_pytest_failure,
                            variant=effective_variant,
                            calibration_preset=preset_name,
                            gpus=preset_gpus,
                            hf_model_ids=preset_model_ids,
                            threshold_file=threshold_rel,
                            threshold_file_sha256=threshold_file_sha,
                            expected_samples=(
                                _configured_expected_samples(vcfg, g)
                                or _configured_expected_samples(ms, g)
                                or _expected_samples(
                                    tree, g, effective_variant, variant_ctx)
                            ),
                        )
        else:
            # Single-source flow (one result-JSON tree per test file).
            default_file = ms.get("json_file")
            cfg_paths = ms.get("paths", {}) or {}
            default_sample_counts = _build_sample_counts(
                ms.get("sample_counts") or {}, default_file)
            observation_validity = _build_observation_validity(
                ms.get("observation_validity"),
                default_file,
            )
            allowed_pytest_failure = _build_allowed_pytest_failure(
                ms.get("allowed_pytest_failure"),
                default_file,
            )
            sc_by_group = ms.get("sample_counts_by_group") or {}
            for (
                preset_name,
                preset_env,
                preset_gpus,
                preset_model_ids,
            ) in _calibration_presets(ms, extra):
                # Presets own disjoint constant namespaces here exactly as they
                # do under `variants`; without this every preset would claim
                # the first preset's symbols and calibration would write one
                # preset's worst-of-N over another's.
                preset_filter = (
                    (ms.get("calibration_presets") or {}).get(preset_name) or {}
                ).get("constant_filter")
                if preset_filter:
                    ppat = re.compile(preset_filter)
                    preset_constants = [
                        (n, k)
                        for (n, k) in all_constants
                        if ppat.match(n.lstrip("_"))
                    ]
                else:
                    preset_constants = all_constants
                groups = _emit_groups(
                    preset_constants, cfg_paths, default_file, counters,
                    slack_derived=slack_derived,
                )
                for g, metrics in groups.items():
                    if g in sc_by_group:
                        sample_counts = _build_sample_counts(
                            sc_by_group[g], default_file
                        )
                    else:
                        sample_counts = default_sample_counts
                    key = _stage_key(
                        base,
                        g,
                        calibration_preset=preset_name,
                    )
                    stages[key] = _stage_entry(
                        rel,
                        _stage_title(
                            base,
                            g,
                            calibration_preset=preset_name,
                        ),
                        g,
                        preset_env,
                        ctx,
                        sha,
                        metrics,
                        sample_counts,
                        observation_validity=observation_validity,
                        allowed_pytest_failure=allowed_pytest_failure,
                        calibration_preset=preset_name,
                        gpus=preset_gpus,
                        hf_model_ids=preset_model_ids,
                        threshold_file=threshold_rel,
                        threshold_file_sha256=threshold_file_sha,
                        expected_samples=(
                            _configured_expected_samples(ms, g)
                            or _expected_samples(tree, g, None, ctx)
                        ),
                    )
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
        if e.get("calibration_preset"):
            L.append(f"  calibration_preset: {_yq(e['calibration_preset'])}")
        if e.get("gpus"):
            L.append(f"  gpus: {int(e['gpus'])}")
        if e.get("hf_model_ids"):
            L.append("  hf_model_ids:")
            L += [f"    - {_yq(v)}" for v in e["hf_model_ids"]]
        L += [f"  test_file_sha256: {e['test_file_sha256']}",
              f"  last_discovered_at: {e['last_discovered_at']}"]
        if e.get("expected_samples") is not None:
            L.append(f"  expected_samples: {int(e['expected_samples'])}")
        if e.get("threshold_file"):
            L += [f"  threshold_file: {_yq(e['threshold_file'])}",
                  f"  threshold_file_sha256: {e['threshold_file_sha256']}"]
        observation_validity = e.get("observation_validity") or {}
        if observation_validity:
            L += [
                "  observation_validity:",
                f"    json_file: {_yq(observation_validity['json_file'])}",
                f"    json_path: {_yq(observation_validity['json_path'])}",
            ]
        allowed_pytest_failure = e.get("allowed_pytest_failure") or {}
        if allowed_pytest_failure:
            L += [
                "  allowed_pytest_failure:",
                f"    json_file: {_yq(allowed_pytest_failure['json_file'])}",
                f"    json_path: {_yq(allowed_pytest_failure['json_path'])}",
            ]
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
        if v == "true": return True
        if v == "false": return False
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


def _stage_alias_bases(sk, stage):
    """Return every base token that should expand to this stage.

    Example: `tts_moss_nonstream_speed` advertises:
    `tts_moss_nonstream`, `tts_moss`, and `tts`.
    """
    bases = []

    def add(value):
        if value and value not in bases:
            bases.append(value)

    base = _stage_base_of(sk, stage)
    add(base)
    v = stage.get("variant")
    if v and base.endswith(f"_{v}"):
        add(base[:-len(v) - 1])
    p = stage.get("calibration_preset")
    for candidate in list(bases):
        if p and candidate.endswith(f"_{p}"):
            add(candidate[:-len(p) - 1])
    return bases


def _build_aliases(all_stages):
    by_base, by_group = {}, {}
    for sk, v in all_stages.items():
        for b in _stage_alias_bases(sk, v):
            by_base.setdefault(b, []).append(sk)
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


def _stage_gpus(stage, gpus_per_test):
    if stage.get("gpus"):
        return int(stage["gpus"])
    return gpus_per_test.get(Path(stage["test"]).name, 2)


def _stage_env_key(stage):
    return tuple(sorted((stage.get("extra_env") or {}).items()))


def _run_namespace(test_path, stage_keys, all_stages):
    test_base = Path(test_path).stem
    presets = sorted({
        str(all_stages[sk].get("calibration_preset"))
        for sk in stage_keys
        if all_stages[sk].get("calibration_preset")
    })
    if presets:
        suffix = "__" + "_".join(presets)
    else:
        suffix = ""
    return f"{test_base}{suffix}"


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
            details = [g]
            if all_stages[sk].get("calibration_preset"):
                details.append(f"preset={all_stages[sk]['calibration_preset']}")
            if all_stages[sk].get("gpus"):
                details.append(f"gpus={all_stages[sk]['gpus']}")
            print(f"    {sk}  [{', '.join(details)}]")
    print("groups:")
    for g in sorted(by_group):
        if not g: continue
        print(f"  @{g} ({len(by_group[g])}): {', '.join(sorted(by_group[g]))}")
    return 0


def run_cmd(args):
    cfg = load_model_config(args.model, host=getattr(args, "host_profile", None))
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
            print(f"error: {s} test sha mismatch — "
                  f"run `tune.py discover --model {cfg['name']}`")
            return 2
        threshold_file = all_stages[s].get("threshold_file")
        if threshold_file:
            th = REPO_ROOT / threshold_file
            if not th.exists():
                print(f"error: threshold file missing for {s}: {th}")
                return 2
            if sha256(th) != all_stages[s].get("threshold_file_sha256"):
                print(f"error: {s} threshold sha mismatch — "
                      f"run `tune.py discover --model {cfg['name']}`")
                return 2
    required_ds = _required_dataset_ids_for_stages(cfg, all_stages, sel)
    # Heads-up if AST-derived needs include a repo not in cfg
    extras = [d for d in required_ds if d not in cfg["hf_datasets"]]
    if extras:
        print(f"note: test(s) reference repo(s) not listed in "
              f"config.yaml hf_datasets: {extras}")
    required_models = _required_model_ids_for_stages(cfg, all_stages, sel)
    requires_speaker_sim = _requires_speaker_sim_for_stages(
        cfg,
        all_stages,
        sel,
    )
    gpus_per_test = cfg.get("gpus_per_test", {}) or {}
    selected_gpu_requirement = max(
        (_stage_gpus(all_stages[s], gpus_per_test) for s in sel),
        default=1,
    )
    if not args.skip_precheck:
        rc = precheck(py, src, out, args.skip_version_check, cfg,
                      datasets_override=required_ds,
                      model_ids_override=required_models,
                      gpu_required_override=selected_gpu_requirement,
                      requires_speaker_sim_override=requires_speaker_sim)
        if rc: return rc
    else:
        smi = nvidia_smi_L()
        (out / "precheck.json").write_text(json.dumps(dict(
            timestamp=now_iso(), model=cfg["name"],
            venv_python=py, venv_source=src,
            versions={}, pins={}, git=git_info(),
            nvidia_smi_L=smi, gpu_summary=gpu_summary(smi)), indent=2))
    gi = git_info()
    plan_path = out / "plan.json"
    if not args.resume and plan_path.exists():
        print(
            "error: output-dir already contains plan.json.\n"
            "  A new calibration request requires a **fresh** --output-dir named "
            "with the current UTC timestamp (e.g. "
            ".tune-runs/$(date -u +%Y%m%dT%H%M%SZ)_<label>/).\n"
            "  Pass --resume only when the user explicitly asked to continue "
            "the **same** interrupted session on the **same** commit.\n"
            "  Never reuse an old run dir or report for a new calibration request."
        )
        return 2
    if args.resume and plan_path.exists():
        existing = json.loads(plan_path.read_text())
        plan_stages = existing.get("stages") or sel
        plan_repeats = existing.get("repeats", args.repeats)
        cal_sha = _plan_calibration_sha(existing)
        if cal_sha and gi["sha"] != cal_sha:
            equivalent, reasons = measurement_equivalent_commits(cal_sha, gi["sha"])
            if not equivalent:
                print(
                    "error: HEAD moved since this run dir was started, and the "
                    "change can affect what is measured.\n"
                    f"  calibration_git_sha: {cal_sha}\n"
                    f"  current HEAD:        {gi['sha']}\n"
                    + "".join(f"  - {r}\n" for r in reasons[:10]) +
                    "  Start a **new** --output-dir on the current commit."
                )
                return 2
            print(
                f"note: HEAD moved {cal_sha[:8]} -> {gi['sha'][:8]}, but every "
                f"change is a threshold constant or calibration-tooling file — "
                f"no measured code differs, so existing observations stay valid."
            )
            existing.setdefault("equivalent_commits", [])
            if gi["sha"] not in existing["equivalent_commits"]:
                existing["equivalent_commits"].append(gi["sha"])
        if set(sel) - set(plan_stages):
            print(
                "error: --resume --stages must be a subset of the existing "
                f"plan stages: {plan_stages}"
            )
            return 2
        if args.repeats != plan_repeats:
            print(
                f"warning: --resume ignores --repeats {args.repeats}; "
                f"using plan repeats={plan_repeats}"
            )
        if set(sel) != set(plan_stages):
            print(
                f"resume: executing subset {sel} within plan "
                f"({len(plan_stages)} stages, repeats={plan_repeats})"
            )
        (out / "plan.json").write_text(json.dumps(dict(
            model=existing.get("model", cfg["name"]),
            stages=plan_stages,
            repeats=plan_repeats,
            timestamp=existing.get("timestamp", now_iso()),
            calibration_started_at=existing.get(
                "calibration_started_at", existing.get("timestamp", now_iso())),
            calibration_git_sha=cal_sha or gi["sha"],
            git_sha=cal_sha or gi["sha"],
            git_branch=existing.get("git_branch", gi["branch"]),
            dirty=gi["dirty"],
            venv_python=existing.get("venv_python", py),
            venv_source=existing.get("venv_source", src),
            tune_version=existing.get("tune_version", __version__),
            stages_yaml=existing.get("stages_yaml", str(sy)),
            last_resume_at=now_iso(),
            last_resume_git_sha=gi["sha"],
            # Commits proven not to change any measured code. Rounds recorded
            # under them are part of the same experiment; provenance accepts
            # them. Rebuilt field-by-field here, so this must be carried over
            # explicitly or the proof is lost on the next resume.
            equivalent_commits=existing.get("equivalent_commits") or [],
        ), indent=2))
        args.repeats = plan_repeats
    else:
        started = now_iso()
        (out / "plan.json").write_text(json.dumps(dict(
            model=cfg["name"], stages=sel, repeats=args.repeats,
            timestamp=started,
            calibration_started_at=started,
            calibration_git_sha=gi["sha"],
            git_sha=gi["sha"], git_branch=gi["branch"],
            dirty=gi["dirty"], venv_python=py, venv_source=src,
            tune_version=__version__, stages_yaml=str(sy)), indent=2))
    if gi["dirty"]:
        d = subprocess.run(["git", "diff", "HEAD"], cwd=REPO_ROOT,
                           capture_output=True, text=True, check=False).stdout
        (out / "workspace.diff").write_text(d)
    # Group stages by test file and env so one pytest invocation covers all
    # compatible stages tied to that file. Multi-preset TTS stages deliberately
    # do not share a pytest invocation: `TTS_CI_MODEL=higgs` and
    # `TTS_CI_MODEL=moss` must each run their own full observation.
    by_test = {}
    for sk in sel:
        stage = all_stages[sk]
        key = (stage["test"], _stage_env_key(stage))
        by_test.setdefault(key, []).append(sk)
    for sk in sel:
        (out / sk).mkdir(parents=True, exist_ok=True)
    max_passes = getattr(args, "max_passes", _DEFAULT_CALIBRATION_PASSES)
    max_gpus = max(
        (_stage_gpus(all_stages[s], gpus_per_test) for s in sel),
        default=2,
    )
    host = cfg.get("_host")
    # Rounds are allocated per unit and grow when a round is rejected, so a
    # stage can end up with more than `repeats` observations.
    # Seed from what is already on disk, not just the baseline: a resumed run
    # must see the replacement rounds a previous process created, otherwise it
    # re-plans from 1..repeats and never finishes replenishing.
    unit_rounds = {
        u: sorted(set(range(1, args.repeats + 1))
                  | set(_round_indices(out, stage_keys)))
        for u, stage_keys in by_test.items()
    }
    restarts = {u: 0 for u in by_test}

    def _unit_label(u):
        return Path(u[0]).stem

    def _unit_clean_count(u) -> int:
        return min((len(_clean_rounds(out, sk, all_stages[sk], unit_rounds[u]))
                    for sk in by_test[u]), default=0)

    def _fill_pending() -> int:
        """Run every not-yet-complete (unit, round). 0 = ok, 1 = halt."""
        for pass_num in range(1, max_passes + 1):
            ran_any = pending = False
            for u, stage_keys in by_test.items():
                test_path = u[0]
                for k in unit_rounds[u]:
                    if all(_run_json_ok(out / sk / f"run{k}.json", all_stages[sk])
                           for sk in stage_keys):
                        if pass_num == 1 and args.resume:
                            print(f"[{_unit_label(u)}] run {k} skipped "
                                  f"(complete, {len(stage_keys)} stage(s))")
                        continue
                    pending = True
                    _purge_incomplete_run(out, stage_keys, all_stages, k)
                    needed = max(
                        (_stage_gpus(all_stages[s], gpus_per_test) for s in stage_keys),
                        default=2,
                    )
                    extra_args = (cfg.get("pytest_extra_args", {}) or {}).get(
                        Path(test_path).name, []) or []
                    if pass_num > 1:
                        print(f"=== calibration pass {pass_num}/{max_passes}: "
                              f"retry {_unit_label(u)} run {k} ===")
                    if _run_shared(test_path, stage_keys, all_stages, out, k, py,
                                   max(unit_rounds[u]), needed, extra_args,
                                   host=host):
                        return 1
                    ran_any = True
            audit = audit_completeness(out, all_stages)
            print(f"completeness after pass {pass_num}: "
                  f"{audit['ok']}/{audit['total']} stage-runs complete")
            if not pending or not ran_any:
                break
            if pass_num < max_passes:
                print(f"{audit['missing_count']} incomplete — GPU cleanup before next pass")
                pinned = included_gpu_indices(host)
                if not _ensure_gpus_free(
                    max_gpus,
                    host=host,
                    target_gpus=sorted(pinned) if pinned is not None else None,
                ):
                    print("error: GPU memory not cleared; stopping calibration passes")
                    break
        return 0

    reject = not getattr(args, "no_destructive_rejection", False)
    if reject and args.repeats < _DESTRUCTIVE_MIN_OBS:
        # MAD on fewer than this many points cannot separate a broken round
        # from ordinary spread, so the detector declines to judge. Say so:
        # silently running without the protection is the dangerous outcome.
        print(f"warning: --repeats {args.repeats} < {_DESTRUCTIVE_MIN_OBS}; "
              f"destructive-round rejection is INACTIVE (too few observations "
              f"to identify an outlier). Worst-of-N will include broken rounds.")
        reject = False
    while True:
        if _fill_pending():
            audit = audit_completeness(out, all_stages)
            print(f"completeness at HALT: "
                  f"{audit['ok']}/{audit['total']} stage-runs complete")
            return 1
        if not reject:
            break
        grew = False
        for u, stage_keys in by_test.items():
            label = _unit_label(u)
            already = {k for k in unit_rounds[u]
                       if _round_rejected(out, stage_keys, k)}
            # Rounds thrown away by an earlier restart must not count toward
            # this block's n, or a resumed run re-triggers the restart and
            # discards the good rounds of the current block with them.
            already_block = {k for k in already
                             if not _round_block_discarded(out, stage_keys, k)}
            degenerate = _detect_degenerate_unit(out, stage_keys, all_stages,
                                                 unit_rounds[u])
            ev = _detect_destructive_rounds(out, stage_keys, all_stages,
                                            unit_rounds[u])
            fresh = sorted(set(ev) - already)
            if degenerate:
                # Majority contamination: the robust centre has moved into the
                # bad cluster, so which side is "correct" is unknowable from
                # this sample. Force the restart path.
                d0 = degenerate[0]
                print(f"[{label}] DEGENERATE sample: {d0['stage_key']}."
                      f"{d0['metric']} splits into two populations "
                      f"({' '.join(f'{v:.6g}' for v in sorted(d0['values']))}) "
                      f"— cannot identify which rounds are broken")
                n, fresh = _DESTRUCTIVE_FULL_RERUN_N, sorted(
                    set(unit_rounds[u]) - already)
            else:
                n = len(already_block) + len(fresh)
                # No *new* rejection is not the same as nothing to do: a run
                # resumed after a teardown carries marks whose replacement
                # rounds were never launched, and must still top up.
                if not fresh and not already:
                    continue
            for k in fresh:
                if not ev.get(k):
                    continue          # condemned by the degenerate guard, no per-metric evidence
                worst = max(ev[k], key=lambda f: f.get("rel_gap") or 0)
                if worst.get("rel_gap") is None:
                    why = worst.get("reason", "invalid value")
                else:
                    why = (f"vs rest median {_g(worst['rest_median'])} "
                           f"(gap {worst['rel_gap']:.0%})")
                print(f"[{label}] round {k} REJECTED as destructive "
                      f"({len(ev[k])} metric(s)); e.g. {worst['stage_key']}."
                      f"{worst['metric']}={_g(worst['value'])} {why}")
            if n >= _DESTRUCTIVE_FULL_RERUN_N:
                if restarts[u] >= _DESTRUCTIVE_MAX_RESTARTS:
                    # One restart already failed. Rejecting this many rounds
                    # would leave the stage short and block the report, which
                    # is worse than keeping them: worst-of-N over everything is
                    # the conservative bound. Keep the data, flag it loudly.
                    print(f"[{label}] STILL unstable after a restart "
                          f"(n={n}{' , degenerate sample' if degenerate else ''})"
                          f" — this is a property of the test or the host, not "
                          f"a single bad round. Keeping all observations; "
                          f"worst-of-N stays conservative. REVIEW THIS STAGE "
                          f"before applying its thresholds.")
                    continue
                print(f"[{label}] n={n} >= {_DESTRUCTIVE_FULL_RERUN_N}: the "
                      f"'others agree' premise fails — discarding all "
                      f"{len(unit_rounds[u])} round(s) and restarting")
                _mark_destructive_rounds(out, stage_keys, {
                    k: ev.get(k, [{"stage_key": stage_keys[0], "metric": "*",
                                   "reason": f"unit restart (n={n})"}])
                    for k in unit_rounds[u]}, block_discarded=True)
                restarts[u] += 1
                start = max(unit_rounds[u]) + 1
                unit_rounds[u] = list(range(start, start + args.repeats))
                grew = True
                continue
            _mark_destructive_rounds(out, stage_keys, {k: ev[k] for k in fresh})
            if _unit_clean_count(u) >= args.repeats:
                continue
            # Declarative, not incremental: the block should hold
            # repeats + 2n rounds for n rejections. Recomputing the target from
            # the marks on disk makes this idempotent, so a resumed run tops up
            # correctly even when it observes no *new* rejections.
            n_total = len(already_block | set(fresh))
            # Size against the CURRENT block: rounds thrown away by a restart
            # are not part of it, so counting them would under-provision a
            # resumed restart and leave the stage permanently short.
            block_rounds = [k for k in unit_rounds[u]
                            if not _round_block_discarded(out, stage_keys, k)]
            start = max(unit_rounds[u]) + 1
            extra = min(args.repeats + 2 * n_total - len(block_rounds),
                        max(0, _DESTRUCTIVE_MAX_ROUNDS - (start - 1)))
            if extra <= 0:
                print(f"[{label}] round cap {_DESTRUCTIVE_MAX_ROUNDS} reached; "
                      f"leaving it incomplete")
                continue
            print(f"[{label}] replenishing {extra} round(s) "
                  f"({start}..{start + extra - 1}) for {n_total} rejected")
            unit_rounds[u] += list(range(start, start + extra))
            grew = True
        if not grew:
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


def _run_shared(test_path, stage_keys, all_stages, out, k, py, total, gpus_needed,
                extra_args=None, host=None):
    """Run pytest once on test_path; write per-stage run{k}.json from
    the result JSONs written under the fresh pytest basetemp.
    """
    test_base = Path(test_path).stem
    run_namespace = _run_namespace(test_path, stage_keys, all_stages)
    shared = out / "_pytest" / run_namespace
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
    stage_env = _stage_env_key(all_stages[stage_keys[0]])
    for sk in stage_keys[1:]:
        if _stage_env_key(all_stages[sk]) != stage_env:
            raise RuntimeError(
                f"internal grouping error: {test_path} stages have different env"
            )
    env.update(dict(stage_env))
    label = (
        f"[{run_namespace}] run {k}/{total} "
        f"({len(stage_keys)} stage(s), needs {gpus_needed} GPU)"
    )
    _print_run_banner(label, test_path, stage_keys, all_stages)
    attempts, status, reason, dur, text, pytest_rc = 0, "ok", "", 0.0, "", 0
    attempt_history = []
    picked = []
    while attempts < _MAX_RUN_ATTEMPTS:
        attempts += 1
        picked, pick_err = _pick_gpus_for_launch(gpus_needed, label, host)
        if picked is None:
            status, reason, dur = "failed", pick_err, 0.0
            attempt_history.append(dict(
                attempt=attempts, status=status, reason=reason,
                duration_s=0.0, pytest_rc=None, gpu_indices=[]))
            print(f"{label} {pick_err}")
            break
        _cleanup_flashinfer_cache(env)
        shutil.rmtree(basetemp, ignore_errors=True)
        basetemp.mkdir(parents=True)
        picked, gate_err = _launch_gpu_gate(picked, gpus_needed, label, host)
        if picked is None:
            status, reason, dur = "failed", gate_err or "launch GPU gate failed", 0.0
            attempt_history.append(dict(
                attempt=attempts, status=status, reason=reason,
                duration_s=0.0, pytest_rc=None, gpu_indices=[]))
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
        if not _ensure_gpus_free(
            gpus_needed, host=host, target_gpus=list(picked)
        ):
            status, reason, dur = "failed", "GPU memory not released after run", 0.0
            attempt_history.append(dict(
                attempt=attempts, status=status, reason=reason,
                duration_s=0.0, pytest_rc=pytest_rc, gpu_indices=list(picked)))
            break
        dur = time.monotonic() - t0
        text = log.read_text(errors="replace")
        if pytest_rc == 0:
            status, reason = "ok", ""
            attempt_history.append(dict(
                attempt=attempts, status=status, reason=reason,
                duration_s=round(dur, 2), pytest_rc=pytest_rc,
                gpu_indices=list(picked)))
            break
        reason = _classify(text, pytest_rc)
        status = "failed"
        attempt_history.append(dict(
            attempt=attempts, status=status, reason=reason,
            duration_s=round(dur, 2), pytest_rc=pytest_rc,
            gpu_indices=list(picked)))
        retryable = (
            any(s in reason for s in RETRY_SIGS)
            or reason.startswith("crashed")
            or "GPU memory" in reason
        )
        if attempts < _MAX_RUN_ATTEMPTS and retryable:
            print(f"{label} {reason} — must clear GPU to <2 GiB before retry "
                  f"({attempts}/{_MAX_RUN_ATTEMPTS})")
            if not _ensure_gpus_free(
                gpus_needed, host=host, target_gpus=list(picked)
            ):
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
        observation_valid = _extract_observation_validity(
            stage,
            basetemp,
            stage_key=sk,
            warnings=extraction_warnings,
        )
        allowed_pytest_failure = _extract_allowed_pytest_failure(
            stage,
            basetemp,
            stage_key=sk,
            warnings=extraction_warnings,
        )
        obs_status, obs_reason = _observation_status(
            stage,
            metrics,
            observation_valid,
            allowed_pytest_failure,
            status,
            reason,
        )
        rec_gi = git_info()
        run_payload = dict(
            status=obs_status, reason=obs_reason, metrics=metrics,
            sample_counts=sample_counts,
            duration_s=round(dur, 2), attempts=attempts,
            attempt_history=attempt_history, gpu_indices=list(picked or []),
            seed=dict(
                calibration=os.environ.get("CALIBRATION_SEED"),
                python_hash=os.environ.get("PYTHONHASHSEED"),
                policy=("explicit" if os.environ.get("CALIBRATION_SEED")
                        else "test-default/unset"),
            ),
            git_sha=rec_gi["sha"],
            recorded_at=now_iso(),
            pytest_log=str(log.resolve()),
            basetemp=str(basetemp.resolve()))
        if stage.get("metrics"):
            run_payload["pytest_rc"] = pytest_rc
        if stage.get("observation_validity"):
            run_payload["observation_valid"] = observation_valid
        if stage.get("allowed_pytest_failure"):
            run_payload["allowed_pytest_failure"] = allowed_pytest_failure
        (sd / f"run{k}.json").write_text(json.dumps(run_payload, indent=2))
        (sd / f"run{k}.log").write_text(
            f"# Shared pytest log (one invocation covered all stages from "
            f"{test_path}):\n# {log.resolve()}\n# basetemp: {basetemp.resolve()}\n")
        brief_parts = []
        if sample_counts.get("total") is not None or sample_counts.get("ok") is not None:
            brief_parts.append(f"samples={sample_counts.get('ok')}/{sample_counts.get('total')}")
        expected = stage.get("expected_samples")
        if (
            expected is not None
            and sample_counts.get("total") is not None
            and sample_counts.get("total") != expected
        ):
            print(
                f"  ⚠ {sk}: sample scope mismatch "
                f"got {sample_counts.get('total')} expected {expected} "
                f"(strict ✗ — purge and re-run on current branch)"
            )
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


def _extract_boolean_source(
    stage,
    source_key,
    basetemp,
    stage_key=None,
    warnings=None,
):
    source = stage.get(source_key) or {}
    if not source:
        return None
    jf, jp = source.get("json_file"), source.get("json_path")
    sk = stage_key or stage.get("title", "?")
    label = source_key.replace("_", " ")
    if not (jf and jp):
        if warnings is not None:
            warnings.append(f"  {sk}: {label} has no json_file/json_path")
        return None
    path = Path(basetemp) / jf
    if not path.exists():
        if warnings is not None:
            warnings.append(f"  {sk}: {label} file missing — {path}")
        return None
    try:
        data = json.loads(path.read_text())
        for key in jp.split("."):
            data = data[key]
        if not isinstance(data, bool):
            raise TypeError(f"{label} must be boolean")
        return data
    except (KeyError, TypeError, json.JSONDecodeError, OSError) as exc:
        if warnings is not None:
            warnings.append(
                f"  {sk}: {label} read failed at "
                f"{jf}::{jp} — {type(exc).__name__}"
            )
        return None


def _extract_observation_validity(stage, basetemp, stage_key=None, warnings=None):
    return _extract_boolean_source(
        stage,
        "observation_validity",
        basetemp,
        stage_key=stage_key,
        warnings=warnings,
    )


def _extract_allowed_pytest_failure(stage, basetemp, stage_key=None, warnings=None):
    return _extract_boolean_source(
        stage,
        "allowed_pytest_failure",
        basetemp,
        stage_key=stage_key,
        warnings=warnings,
    )


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
    strict = strict_audit(run_dir, all_stages, plan)
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
        strict_ready=f"{strict['strict_ready']}/{strict['total_stages']}",
        strict_complete=strict["strict_complete"],
        timestamp=now_iso(),
    )
    print(json.dumps(out, indent=2))
    return 0 if audit["complete"] and strict["strict_complete"] else 1


def validate_run_ready(run_dir: Path) -> tuple[dict | None, list[str]]:
    """Shared hard gate for every command that consumes final observations."""
    errors = []
    plan_path = run_dir / "plan.json"
    if not plan_path.exists():
        return None, [f"no plan.json in {run_dir}"]
    plan = json.loads(plan_path.read_text())
    sy = Path(plan.get("stages_yaml") or stages_path(plan.get("model", DEFAULT_MODEL)))
    if not sy.exists():
        return None, [f"stages schema missing: {sy}"]
    all_stages = _load_yaml(sy)
    git = audit_git_provenance(run_dir, plan)
    if not git["ok"]:
        errors.append(f"git provenance failed: {git['reason']}")
    strict = strict_audit(run_dir, all_stages, plan)
    if not strict["strict_complete"]:
        errors.append(
            f"strict audit incomplete: {strict['strict_ready']}/"
            f"{strict['total_stages']} stages"
        )
    complete = audit_completeness(run_dir, all_stages, plan)
    if not complete["complete"]:
        errors.append(
            f"calibration incomplete: {complete['ok']}/{complete['total']} stage-runs"
        )
    return dict(
        plan=plan, stages_yaml=sy, all_stages=all_stages,
        git=git, strict=strict, completeness=complete,
    ), errors


def merge_runs(run_dirs: list[Path], output_dir: Path) -> int:
    """Merge strict-ready, disjoint stage partitions into one reportable run."""
    if len(run_dirs) < 2:
        print("error: merge-runs requires at least two --run-dir inputs")
        return 2
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"error: merge output directory is not empty: {output_dir}")
        return 2
    validated = []
    for run_dir in run_dirs:
        ready, errors = validate_run_ready(run_dir)
        if errors:
            print(f"error: input run is not ready: {run_dir}: {'; '.join(errors)}")
            return 1
        validated.append((run_dir, ready))
    first_plan = validated[0][1]["plan"]
    schema_hash = sha256(validated[0][1]["stages_yaml"])
    seen = set()
    fingerprints = []
    for run_dir, ready in validated:
        plan = ready["plan"]
        for key in ("model", "repeats", "calibration_git_sha"):
            if plan.get(key) != first_plan.get(key):
                print(f"error: incompatible {key} in {run_dir}")
                return 2
        if sha256(ready["stages_yaml"]) != schema_hash:
            print(f"error: stage schema differs in {run_dir}")
            return 2
        overlap = seen.intersection(plan["stages"])
        if overlap:
            print(f"error: overlapping stage ownership in {run_dir}: {sorted(overlap)}")
            return 2
        seen.update(plan["stages"])
        fp_path = run_dir / "environment-fingerprint.json"
        fingerprints.append(json.loads(fp_path.read_text()) if fp_path.exists() else {})
    identity_keys = ("dependency_freeze_sha256", "container_image_digest")
    for key in identity_keys:
        known = {fp.get(key) for fp in fingerprints if fp.get(key)}
        if len(known) > 1:
            print(f"error: incompatible environment fingerprint {key}: {sorted(known)}")
            return 2
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_stages = []
    for run_dir, ready in validated:
        for stage in ready["plan"]["stages"]:
            shutil.copytree(run_dir / stage, output_dir / stage)
            merged_stages.append(stage)
    merged_plan = dict(first_plan)
    merged_plan.update(
        stages=merged_stages,
        merged_at=now_iso(),
        source_runs=[str(path.resolve()) for path, _ready in validated],
    )
    (output_dir / "plan.json").write_text(json.dumps(merged_plan, indent=2))
    first_run = validated[0][0]
    for name in ("precheck.json", "environment-fingerprint.json"):
        if (first_run / name).exists():
            shutil.copy2(first_run / name, output_dir / name)
    (output_dir / "environment-fingerprints.json").write_text(
        json.dumps(fingerprints, indent=2)
    )
    print(f"merged {len(merged_stages)} stages into {output_dir}")
    return report(output_dir)


def _metric_statistics(values: list[float], worst_op: str) -> dict:
    ordered = sorted(values)
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    q1, q3 = (ordered[1], ordered[-2]) if len(ordered) >= 4 else (ordered[0], ordered[-1])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return dict(
        worst=(min(values) if worst_op == "min" else max(values)),
        median=statistics.median(values), min=min(values), max=max(values),
        range=max(values) - min(values), mean=mean, std=std,
        cv=(std / abs(mean) if mean else None),
        outlier_runs=[i + 1 for i, value in enumerate(values)
                      if value < low or value > high],
    )


_CORRECTNESS_HINTS = ("accuracy", "wer", "cer", "similarity", "utmos",
                      "n_above", "der", "corpus")


def _g(v) -> str:
    """Compact number for evidence tables (no float dust)."""
    return "N/A" if v is None else f"{v:.6g}"


def _is_correctness_metric(stage: dict, metric_key: str) -> bool:
    key = metric_key.lower()
    group = (stage.get("group") or "").lower()
    return (any(h in key for h in _CORRECTNESS_HINTS)
            or group in ("diarization", "wer", "similarity", "utmos", "accuracy"))


def _destructive_report_sections(run_dir: Path, plan: dict, all_stages: dict) -> list:
    """Rejected rounds, plus the ones that look like genuine defects.

    Host contention explains a slow round; it does not explain a wrong one. A
    rejected round whose *correctness* metrics moved is a lead worth chasing,
    so it is surfaced separately instead of vanishing into the exclusion.
    """
    rows, suspects = [], []
    for sk in plan["stages"]:
        stage = all_stages.get(sk) or {}
        for k in _round_indices(run_dir, [sk]):
            p = run_dir / sk / f"run{k}.json"
            if not _run_json_destructive(p):
                continue
            data = json.loads(p.read_text())
            for ev in data.get("destructive_evidence") or []:
                if ev.get("stage_key") != sk:
                    continue
                gap = ev.get("rel_gap")
                val, med = _g(ev.get("value")), _g(ev.get("rest_median"))
                z = ev.get("z")
                rows.append(
                    f"| {sk} | run{k} | `{ev.get('metric')}` | {val} | {med} | "
                    f"{'—' if gap is None else f'{gap:.0%}'} | "
                    f"{'∞' if z is None else _g(z)} |")
                if _is_correctness_metric(stage, str(ev.get("metric"))):
                    suspects.append(
                        f"| {sk} | run{k} | `{ev.get('metric')}` | {val} | {med} |")
    if not rows:
        return []
    out = ["## Rejected destructive rounds", "",
           "These rounds completed with full sample scope but were excluded "
           "from worst-of-N and replaced by additional rounds. Values are "
           "retained here as evidence.", "",
           "| Stage | Round | Metric | Rejected value | Rest median | Gap | z |",
           "|---|---|---|---:|---:|---:|---:|"]
    out += rows
    out.append("")
    if suspects:
        out += ["### Suspected real defects — investigate, do not dismiss", "",
                "Contention explains a slow round, not a wrong one. These "
                "**correctness** metrics moved in a rejected round, which may "
                "indicate a genuine intermittent defect rather than a noisy "
                "measurement.", "",
                "| Stage | Round | Metric | Rejected value | Rest median |",
                "|---|---|---|---:|---:|"]
        out += suspects
        out.append("")
    return out


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def report(run_dir):
    ready, errors = validate_run_ready(run_dir)
    if errors:
        print("error: refusing report — " + "; ".join(errors))
        return 1
    plan = ready["plan"]
    all_stages = ready["all_stages"]
    pre = json.loads((run_dir / "precheck.json").read_text()) \
        if (run_dir / "precheck.json").exists() else {}
    sy = Path(plan.get("stages_yaml") or stages_path(plan.get("model", DEFAULT_MODEL)))
    all_stages = _load_yaml(sy)
    N = plan["repeats"]
    cal_sha = _plan_calibration_sha(plan)
    started = plan.get("calibration_started_at") or plan.get("timestamp", "?")
    L = [
        "# CI Threshold Observation Report",
        "",
        f"**Calibration commit:** `{cal_sha}`",
        f"**Branch:** {plan.get('git_branch', '?')}",
        f"**Run directory:** `{run_dir}`",
        f"**Calibration started:** {started}",
        f"**Report generated:** {now_iso()}",
        "",
    ]
    for idx, sk in enumerate(plan["stages"], start=1):
        s = all_stages[sk]
        L += [f"## {idx}. {s['title']}", "", f"{{{{CONTEXT:{sk}}}}}", ""]
        rounds = _round_indices(run_dir, [sk]) or list(range(1, N + 1))
        results = [json.loads((run_dir / sk / f"run{k}.json").read_text())
                   if (run_dir / sk / f"run{k}.json").exists() else None
                   for k in rounds]
        rejected = {k for k, r in zip(rounds, results)
                    if r is not None and r.get("destructive") is True}
        eff_n = sum(1 for k, r in zip(rounds, results)
                    if r is not None and k not in rejected)
        if not s["metrics"]:
            L += ["| Run | Result |", "|-----|--------|"]
            cells = ["PASS" if r and r["status"] == "ok" else "FAIL"
                     for r in results]
            for i, c in zip(rounds, cells): L.append(f"| {i} | {c} |")
            worst = "FAIL" if any(c == "FAIL" for c in cells) else "PASS"
            L += [f"| **Worst-of-{eff_n}** | **{worst}** |", ""]
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
        for i, r in zip(rounds, results):
            drop = i in rejected
            cells = []
            if has_counts:
                if r is None:
                    cells += ["MISSING", "MISSING"]
                else:
                    sc = r.get("sample_counts") or {}
                    tot, okc = sc.get("total"), sc.get("ok")
                    cells += [_fmt_count(tot), _fmt_count(okc)]
                    if not drop:
                        if tot is not None: cvals["total"].append(tot)
                        if okc is not None: cvals["ok"].append(okc)
            for k in keys:
                if r is None: cells.append("MISSING")
                elif k in nulls: cells.append("N/A")
                else:
                    v = (r.get("metrics") or {}).get(k)
                    cells.append(_fmt(v, disp[k]))
                    # Destructive rounds stay visible but never aggregate.
                    if v is not None and not drop: vals[k].append(v)
            label = f"{i} (rejected)" if drop else str(i)
            L.append(f"| {label} | " + " | ".join(cells) + " |")
        wc = []
        if has_counts:
            # Samples run/ok are diagnostic counts, not quality metrics.
            # Worst-of-N has no meaning for them — per-run values above
            # already surface any coverage drop. Leave these cells blank.
            wc.append("—")
            wc.append("—")
        stage_stats = {}
        for k in keys:
            if k in nulls or not vals[k]: wc.append("**N/A**"); continue
            stage_stats[k] = _metric_statistics(vals[k], worst[k])
            v = stage_stats[k]["worst"]
            wc.append(f"**{_fmt(v, disp[k])}**")
        L += [f"| **Worst-of-{eff_n}** | " + " | ".join(wc) + " |", ""]
        if rejected:
            L += [
                f"*Destructive rounds rejected: "
                f"{', '.join('run' + str(k) for k in sorted(rejected))}. "
                f"Worst-of-N uses the {eff_n} surviving observation(s); the "
                f"rejected values are shown above but never aggregated.*",
                "",
            ]
        if stage_stats:
            L += ["### Metric calibration", "",
                  "| Metric | Worst | Median | Min | Max | Range | Std | CV | Outliers |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
            for k in keys:
                if k not in stage_stats:
                    continue
                st = stage_stats[k]
                d = disp[k]
                scaled = lambda v: v * d["scale"]
                cv = "N/A" if st["cv"] is None else f"{st['cv']:.3f}"
                outliers = ", ".join(f"run{i}" for i in st["outlier_runs"]) or "none"
                L.append(
                    f"| {d['label']} | {scaled(st['worst']):.{d['digits']}f} | "
                    f"{scaled(st['median']):.{d['digits']}f} | "
                    f"{scaled(st['min']):.{d['digits']}f} | "
                    f"{scaled(st['max']):.{d['digits']}f} | "
                    f"{scaled(st['range']):.{d['digits']}f} | "
                    f"{scaled(st['std']):.{d['digits']}f} | {cv} | {outliers} |"
                )
            if "accuracy" in stage_stats and cvals["total"]:
                total = sum(cvals["total"])
                successes = round(sum(vals["accuracy"][i] * cvals["total"][i]
                                      for i in range(min(len(vals["accuracy"]), len(cvals["total"])))))
                lo, hi = _wilson_interval(successes, total)
                L += ["", f"Accuracy aggregate: {successes}/{total}; "
                      f"95% Wilson CI {lo * 100:.2f}%–{hi * 100:.2f}%."]
            L.append("")
        # What actually got written into the test files (if anything) is
        # recorded in the "Applied changes" table appended after
        # mode-smart/mode-full apply (see SKILL.md step 9).
        for k in nulls:
            L.append(f"> ⚠ {disp[k]['label']}: no `json_path` in stages.yaml "
                     "(config.yaml `metric_sources` missing this metric)")
        if nulls: L.append("")
    L += _destructive_report_sections(run_dir, plan, all_stages)
    L += ["## Operational reliability", "",
          "| Stage | Runs | Attempts | Retry successes | Failed attempts | Startup | OOM | Timeout | Partial |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for sk in plan["stages"]:
        runs = [json.loads((run_dir / sk / f"run{k}.json").read_text())
                for k in (_round_indices(run_dir, [sk]) or list(range(1, N + 1)))]
        histories = [r.get("attempt_history") or [] for r in runs]
        attempts = sum(len(h) or int(r.get("attempts", 1))
                       for h, r in zip(histories, runs))
        failed_attempts = [a for h in histories for a in h if a.get("status") != "ok"]
        failed = len(failed_attempts)
        retry_successes = sum(
            1 for r in runs
            if int(r.get("attempts", 1)) > 1 and r.get("status") == "ok"
        )
        reasons = [str(a.get("reason", "")).lower() for a in failed_attempts]
        startup = sum(1 for reason in reasons if any(
            token in reason for token in ("start", "connection refused", "address already")
        ))
        oom = sum(1 for reason in reasons if "oom" in reason or "out of memory" in reason)
        timeout = sum(1 for reason in reasons if "timeout" in reason)
        partial = sum(1 for r in runs if (r.get("sample_counts") or {}).get("ok") !=
                      (r.get("sample_counts") or {}).get("total"))
        L.append(
            f"| {sk} | {N} | {attempts} | {retry_successes} | {failed} | "
            f"{startup} | {oom} | {timeout} | {partial} |"
        )
    L.append("")
    dirty = " (dirty)" if plan.get("dirty") else ""
    diff = " — see `workspace.diff`" if plan.get("dirty") else ""
    v = pre.get("versions", {}) or {}
    fingerprint = {}
    fingerprint_path = run_dir / "environment-fingerprint.json"
    if fingerprint_path.exists():
        fingerprint = json.loads(fingerprint_path.read_text())
    L += ["## Provenance", "",
          f"- Model: {plan.get('model', '?')}",
          f"- Calibration commit: `{cal_sha}`",
          f"- Branch: {plan.get('git_branch', '?')}{dirty}{diff}",
          f"- Calibration started: {started}",
          f"- Venv Python: {plan.get('venv_python', '?')} "
          f"({plan.get('venv_source', '?')})",
          f"- sglang {v.get('sglang', '?')} · torch {v.get('torch', '?')}",
          f"- GPU: {pre.get('gpu_summary', '?')}",
          f"- GPU group: {fingerprint.get('gpu_group', '?')}",
          f"- Environment comparability: {fingerprint.get('comparability', 'unverified')}",
          f"- Container image digest: {fingerprint.get('container_image_digest') or 'unverified'}",
          f"- Dependency freeze SHA256: {fingerprint.get('dependency_freeze_sha256', '?')}",
          f"- tune-ci-thresholds v{__version__}",
          f"- Report generated: {now_iso()}"]
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
    m = re.search(
        rf"^{re.escape(symbol)}(?:\s*:\s*[^=]+)?\s*=\s*"
        r"(None|[+-]?\d+(?:\.\d+)?)\s*$",
        text,
        re.M,
    )
    if not m or m.group(1) == "None":
        return None
    return float(m.group(1))


def _read_nested_value(text, symbol, conc, subkey):
    m = re.search(rf"^{re.escape(symbol)}\s*=\s*\{{", text, re.M)
    if not m: return None
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
    if conc is None:
        keys = sorted(
            set(
                int(k)
                for k in re.findall(r"^\s*(\d+)\s*:\s*\{", block, re.M)
            )
        )
        if len(keys) != 1:
            return None
        conc = keys[0]
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
    if stage_group in ("accuracy", "diarization", "similarity", "utmos"):
        return worst_raw
    if worst_rounded is None:
        return worst_raw
    if worst_op == "min" and worst_rounded > worst_raw:
        return worst_raw
    if worst_op == "max" and worst_rounded < worst_raw:
        return worst_raw
    return worst_rounded


def apply_plan(run_dir):
    ready, errors = validate_run_ready(run_dir)
    if errors:
        print(json.dumps(dict(
            error="run_not_ready",
            reasons=errors,
        ), indent=2))
        return 1
    plan = ready["plan"]
    all_stages = ready["all_stages"]
    N = plan["repeats"]
    out = {"model": plan["model"], "run_dir": str(run_dir),
           "repeats": N, "stages": []}
    for sk in plan["stages"]:
        s = all_stages[sk]
        if not s["metrics"]:  # docs stage
            continue
        test_path = REPO_ROOT / s["test"]
        test_text = test_path.read_text()
        threshold_path = REPO_ROOT / s.get("threshold_file", s["test"])
        threshold_text = threshold_path.read_text()
        conc = _read_concurrency(test_text) or _read_concurrency(threshold_text)
        rounds = _round_indices(run_dir, [sk]) or list(range(1, N + 1))
        per_run, rejected_rounds = [], []
        for k in rounds:
            p = run_dir / sk / f"run{k}.json"
            data = json.loads(p.read_text()) if p.exists() else None
            if data is not None and data.get("destructive") is True:
                rejected_rounds.append(k)
                continue          # never feeds worst-of-N
            per_run.append(data)
        sg = {
            "stage_key": sk,
            "rounds": rounds,
            "rejected_rounds": rejected_rounds,
            "effective_n": sum(1 for r in per_run if r is not None),
            "test": str(test_path),
            "threshold_file": str(threshold_path),
            "title": s["title"],
            "stage_group": s.get("group"),
            "variant": s.get("variant"),
            "calibration_preset": s.get("calibration_preset"),
            "gpus": s.get("gpus"),
            "concurrency": conc,
            "metrics": [],
        }
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
            if kind == "bare":
                cur = _read_bare_value(threshold_text, sym)
            elif kind == "nested":
                cur = _read_nested_value(threshold_text, sym, conc, sub)
            else:
                cur = None
            # Defense in depth: fixed symbols stay out of discover, but if an
            # old stages.yaml still lists one, never propose rewriting it.
            if sym in _FIXED_THRESHOLD_SYMBOLS:
                write_value = None
                direction = "fixed"
            else:
                write_value = _apply_write_value(
                    worst_op, worst, worst_rounded, s.get("group"))
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


def _bootstrap_from_host(host: dict | None) -> None:
    global REPO_ROOT
    if not host:
        return
    REPO_ROOT = resolve_repo_root(host)
    venv = host.get("venv_python")
    if venv and not os.environ.get("TUNE_VENV_PYTHON"):
        os.environ.setdefault("TUNE_VENV_PYTHON", str(venv))
    if host.get("gpu_exclude") and not os.environ.get("TUNE_GPU_EXCLUDE"):
        os.environ.setdefault(
            "TUNE_GPU_EXCLUDE",
            ",".join(str(i) for i in host["gpu_exclude"]),
        )


def main(argv=None):
    global REPO_ROOT
    p = argparse.ArgumentParser(prog="tune.py")
    p.add_argument("--venv-python")
    p.add_argument("--skip-version-check", action="store_true")
    p.add_argument(
        "--host",
        help="host profile under hosts/ (default: $TUNE_HOST or autodetect by hostname)",
    )
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"model config name under models/ (default: {DEFAULT_MODEL})")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("models-list")
    sub.add_parser("hosts-list")
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
    sc.add_argument("--no-destructive-rejection", action="store_true",
                    help="keep destructive rounds in worst-of-N instead of "
                         "rejecting and replenishing them (escape hatch)")
    sd = sub.add_parser("report"); sd.add_argument("--run-dir", required=True)
    se = sub.add_parser("apply-plan"); se.add_argument("--run-dir", required=True)
    sf = sub.add_parser("status"); sf.add_argument("--run-dir", required=True)
    sg = sub.add_parser("strict-audit"); sg.add_argument("--run-dir", required=True)
    sm = sub.add_parser("merge-runs")
    sm.add_argument("--run-dir", action="append", required=True, dest="run_dirs")
    sm.add_argument("--output-dir", required=True)
    args = p.parse_args(argv)
    if args.cmd == "models-list":
        for m in available_models(): print(m)
        return 0
    if args.cmd == "hosts-list":
        for h in available_hosts():
            print(h)
        return 0
    host = load_host_profile(args.host)
    _bootstrap_from_host(host)
    args.host_profile = host
    if host:
        print(f"host: {host.get('name', args.host or 'autodetect')} "
              f"(repo={REPO_ROOT})")
    if args.cmd == "report":
        return report(Path(args.run_dir))
    if args.cmd == "apply-plan":
        return apply_plan(Path(args.run_dir))
    if args.cmd == "status":
        return status_cmd(Path(args.run_dir))
    if args.cmd == "strict-audit":
        return strict_audit_cmd(Path(args.run_dir))
    if args.cmd == "merge-runs":
        return merge_runs([Path(path) for path in args.run_dirs], Path(args.output_dir))
    cfg = load_model_config(args.model, host=host)
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
