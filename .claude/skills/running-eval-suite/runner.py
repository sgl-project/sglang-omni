#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""running-eval-suite — backend for the slash command.

Subcommands:
  precheck   verify env (venv, sglang/torch, HF cache, GPU, hardware tag).
  run        launch server + client per row, capture result.json, then
             edit benchmarks/eval/benchmark_*.py in place — replace the
             row that matches the host's hardware tag, or append a new
             row at the end of the section's table when no row matches.
             Local-Pipeline tables are skipped.

Skill flow: precheck → run → git add benchmarks/eval/ + git commit.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any

__version__ = "0.2.0"

SKILL_DIR = Path(__file__).resolve().parent
MODELS_DIR = SKILL_DIR / "models"
DEFAULT_MODEL = "qwen3-omni"
DEFAULT_RUN_ROOT_REL = ".eval-runs"
REPO_ROOT_CANDIDATES = [
    Path("/sgl-workspace/sglang-omni"),
    Path("/data/sglang-omni"),
    Path(__file__).resolve().parents[3],
]

_LOCAL_MARKERS = (
    "Local v1 Pipeline Result",
    "Local Speech Pipeline",
    "local v1 sweep",
)
_OFFICIAL_MARKERS = (
    "Full-Set Reference Results",
    "Reference Results",
    "H200 Reference Results",
)
_PARENT_SCAN_LINES = 80  # how far up to scan for a Local/Official marker


# ---------- yaml ----------

try:
    import yaml  # type: ignore

    def load_yaml(path: Path) -> Any:
        with open(path) as fh:
            return yaml.safe_load(fh)

    def dump_yaml(data: Any) -> str:
        return yaml.safe_dump(data, sort_keys=False)
except ImportError:  # pragma: no cover

    def load_yaml(path: Path) -> Any:
        raise SystemExit(
            "PyYAML missing — `pip install pyyaml` in your venv first"
        )

    def dump_yaml(data: Any) -> str:
        raise SystemExit(
            "PyYAML missing — `pip install pyyaml` in your venv first"
        )


# ---------- helpers ----------

def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ts_dir() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def repo_root() -> Path:
    for cand in REPO_ROOT_CANDIDATES:
        if cand.exists() and (cand / "benchmarks" / "eval").is_dir():
            return cand
    raise SystemExit(
        "repo root not found — run from a sglang-omni clone "
        "(need benchmarks/eval/)"
    )


def git_source_prefix(root: Path) -> str:
    """Return source-tag prefix: PR #N | main <sha7> | local <sha7> | dirty <sha7>."""
    def _git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    sha7 = _git("rev-parse", "HEAD")[:7]
    dirty_unstaged = subprocess.run(
        ["git", "-C", str(root), "diff", "--quiet"], capture_output=True
    ).returncode != 0
    dirty_staged = subprocess.run(
        ["git", "-C", str(root), "diff", "--cached", "--quiet"], capture_output=True
    ).returncode != 0
    if dirty_unstaged or dirty_staged:
        return f"dirty {sha7}"
    try:
        proc = subprocess.run(
            ["gh", "pr", "view", "--json", "number", "-q", ".number"],
            cwd=str(root), capture_output=True, text=True, timeout=5,
        )
        pn = proc.stdout.strip()
        if proc.returncode == 0 and pn.isdigit():
            return f"PR #{pn}"
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    if branch == "main":
        return f"main {sha7}"
    return f"local {sha7}"


def detect_hardware() -> str:
    """Canonical hw short tag for the local host: H200 / H100 / H800 / A100 / ...

    Reads `nvidia-smi --query-gpu=name`, takes the first GPU's name,
    extracts the model token. Returns "unknown" if nvidia-smi missing.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unknown"
    names = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
    if not names:
        return "unknown"
    name = names[0].upper()
    m = re.search(r"\b(H200|H100|H800|A100|A800|L40|L4|RTX[0-9]+)\b", name)
    if m:
        return m.group(1)
    cleaned = name.replace("NVIDIA ", "").split()
    return cleaned[0] if cleaned else "unknown"


def model_dir(name: str) -> Path:
    return MODELS_DIR / name


def available_models() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    return sorted(
        d.name for d in MODELS_DIR.iterdir()
        if d.is_dir() and (d / "config.yaml").exists()
    )


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


def load_model_config(name: str) -> dict:
    p = model_dir(name) / "config.yaml"
    if not p.exists():
        raise SystemExit(
            f"model config not found: {p}\nAvailable: {available_models()}"
        )
    cfg = load_yaml(p) or {}
    cfg.setdefault("name", name)
    cfg.setdefault("auto_env", {})
    cfg.setdefault("rows", [])
    _apply_auto_env(cfg)
    return cfg


def resolve_venv_python(flag: str | None, cfg: dict) -> tuple[str, str, list[str]]:
    if flag:
        return flag, "flag", []
    env = os.environ.get("EVAL_VENV_PYTHON")
    if env:
        return env, "env", []
    default = cfg.get("default_venv_python")
    if not default:
        raise SystemExit("default_venv_python missing in config.yaml")
    cands = default if isinstance(default, list) else [default]
    for cand in cands:
        if Path(cand).exists():
            return cand, "default", cands
    return cands[-1], "default", cands


def free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def port_is_available(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def free_port_range(count: int, excluded: set[int] | None = None) -> int:
    excluded_ports = excluded or set()
    for _ in range(100):
        base_port = free_port()
        candidates = [base_port + offset for offset in range(count)]
        if any(port in excluded_ports for port in candidates):
            continue
        if all(port_is_available(port) for port in candidates):
            return base_port
    raise RuntimeError(f"failed to find {count} consecutive available ports")


def parse_gpu_status() -> tuple[list[dict], str | None]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,memory.total,utilization.gpu,uuid",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
    except FileNotFoundError:
        return [], "nvidia-smi not found (no GPUs?)"
    except subprocess.SubprocessError as exc:
        return [], f"nvidia-smi failed: {exc}"
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            rows.append({
                "index": int(parts[0]),
                "mem_used_mb": int(parts[1]),
                "mem_total_mb": int(parts[2]),
                "util_pct": int(parts[3]),
                "uuid": parts[4],
            })
        except ValueError:
            continue
    return rows, None


def free_gpu_indices(gpus: list[dict], free_thresh_mb: int = 1000) -> list[int]:
    # A GPU counts as free if either: (a) memory below threshold, or
    # (b) no compute process is bound to it. Case (b) handles orphaned
    # CUDA contexts left by previous killed processes.
    busy_gpu_uuids = _gpus_with_active_processes()
    out: list[int] = []
    for g in gpus:
        if g["mem_used_mb"] < free_thresh_mb:
            out.append(g["index"])
            continue
        uuid = g.get("uuid")
        if uuid and uuid not in busy_gpu_uuids:
            out.append(g["index"])
    return out


def _wait_for_memory_release(gpu_indices: list[int], *,
                             low_watermark_mb: int, max_wait_s: int) -> None:
    """Poll until each assigned GPU's used memory drops below watermark.

    Talker mp_runner can leave ~60-100 GB of CUDA context allocated for
    seconds after process exit. Without this wait the next round boots
    a server while the prior context still holds memory and OOMs.
    """
    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        gpus, err = parse_gpu_status()
        if err:
            time.sleep(2)
            continue
        by_idx = {g["index"]: g for g in gpus}
        all_clear = True
        for idx in gpu_indices:
            g = by_idx.get(idx)
            if g is None:
                continue
            if g["mem_used_mb"] >= low_watermark_mb:
                all_clear = False
                break
        if all_clear:
            return
        time.sleep(3)


def _gpus_with_active_processes() -> set[str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-compute-apps=gpu_uuid",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return set()
    return {ln.strip() for ln in out.strip().splitlines() if ln.strip()}


def gpu_busy_processes() -> list[dict]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    items = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            try:
                items.append({"pid": int(parts[0]), "used_mb": int(parts[1])})
            except ValueError:
                continue
    return items


def _benchmark_selectors(row: dict) -> set[str]:
    stem = Path(row.get("file", "")).stem
    if stem.startswith("benchmark_omni_"):
        short_name = stem[len("benchmark_omni_"):]
        if short_name == "seedtts":
            return {"omni_seedtts", "qwen_seedtts"}
        return {short_name}
    if stem.startswith("benchmark_tts_"):
        short_name = stem[len("benchmark_tts_"):]
        if short_name == "seedtts":
            return {"tts_seedtts", "s2pro_seedtts"}
        return {f"tts_{short_name}"}
    return {stem}


def _select_rows(rows: list[dict], spec: str | None,
                 exclude_ids: str | None = None) -> list[dict]:
    if not spec or spec == "all":
        kept = list(rows)
    else:
        wanted = {s.strip() for s in spec.split(",") if s.strip()}
        known = set().union(*(_benchmark_selectors(row) for row in rows))
        unknown = wanted - known
        if unknown:
            raise SystemExit(
                f"unknown benchmark selector(s): {sorted(unknown)}. "
                f"Known: {sorted(known)}"
            )
        kept = [r for r in rows if _benchmark_selectors(r) & wanted]
    if exclude_ids:
        tokens = [s.strip() for s in exclude_ids.split(",") if s.strip()]
        if tokens:
            kept = [r for r in kept
                    if not any(t in str(r.get("id", "")) for t in tokens)]
    return kept


def _row_needs_asr_gpu(row: dict) -> bool:
    return "{asr_gpu}" in str(row.get("client", ""))


def _row_total_gpus(row: dict, cfg: dict) -> int:
    return _row_server_gpus(row, cfg) + (1 if _row_needs_asr_gpu(row) else 0)


# ---------- precheck ----------

def cmd_precheck(args: argparse.Namespace) -> int:
    cfg = load_model_config(args.model)
    py, src, cands = resolve_venv_python(args.venv_python, cfg)
    out_dir = Path(args.output_dir) if args.output_dir else None

    print(f"model: {cfg.get('name', args.model)}")
    print(f"venv_python: {py} ({src})")
    if src == "default" and len(cands) > 1:
        print(f"  (tried in order: {', '.join(cands)})")

    errs: list[str] = []
    warns: list[str] = []
    versions: dict[str, str] = {}
    gpus: list[dict] = []

    if not Path(py).exists():
        errs.append(f"venv python missing: {py}")
        if src == "default" and len(cands) > 1:
            errs.append(f"  (tried: {', '.join(cands)})")
        return _emit_precheck_summary(
            errs, warns, out_dir, cfg, py, src,
            versions=versions, gpus=gpus, hardware="unknown",
        )

    for pkg in ("sglang", "torch"):
        proc = subprocess.run(
            [py, "-c", f"import {pkg}; print({pkg}.__version__)"],
            capture_output=True, text=True, timeout=20,
        )
        if proc.returncode == 0:
            versions[pkg] = proc.stdout.strip()
            print(f"  ✓ {pkg}: {versions[pkg]}")
        else:
            tail = (proc.stderr or "").strip().splitlines()
            tail_msg = tail[-1] if tail else "(no stderr)"
            errs.append(f"{pkg} import failed: {tail_msg[:200]}")
            print(f"  ✗ {pkg}: import failed ({tail_msg[:80]})")

    # Aggregate HF assets needed (from both config-level and per-row).
    hf_models: set[str] = set()
    if cfg.get("hf_model_id"):
        hf_models.add(cfg["hf_model_id"])
    for row in cfg.get("rows", []):
        if row.get("hf_model_id"):
            hf_models.add(row["hf_model_id"])
    hf_datasets = set(cfg.get("hf_datasets") or [])

    for mid in sorted(hf_models):
        ok = _hf_cached(py, mid, "model")
        print(f"  {'✓' if ok else '✗'} HF model: {mid}")
        if not ok:
            errs.append(
                f"HF model not cached: {mid} — "
                f"`HF_ENDPOINT=https://hf-mirror.com huggingface-cli download {mid}`"
            )
    for ds in sorted(hf_datasets):
        ok = _hf_cached(py, ds, "dataset")
        print(f"  {'✓' if ok else '✗'} HF dataset: {ds}")
        if not ok:
            errs.append(
                f"HF dataset not cached: {ds} — "
                "`HF_ENDPOINT=https://hf-mirror.com "
                f"huggingface-cli download --repo-type dataset {ds}`"
            )

    # Auxiliary scoring resources (ASR models, normalizer pkgs, repo-root symlinks).
    # These are orthogonal to the main model+dataset and easy to forget —
    # without them the run usually succeeds at generation but produces zero
    # evaluated samples (silent quality fail) or skips whole rows.
    for aux in cfg.get("auxiliary_models", []) or []:
        kind = aux.get("kind", "hf")
        aid = aux.get("id", "?")
        if kind == "hf":
            ok = _hf_cached(py, aid, "model")
        elif kind == "modelscope":
            ok = _modelscope_cached(aid)
        else:
            ok = False
        used = aux.get("used_for", "")
        suffix = f" — {used}" if used else ""
        print(f"  {'✓' if ok else '✗'} aux model ({kind}): {aid}{suffix}")
        if not ok:
            hint = aux.get("install_cmd") or f"<install {aid}>"
            errs.append(f"aux model not cached: {aid} — `{hint}`")

    for pkg in cfg.get("python_packages", []) or []:
        meta = pkg if isinstance(pkg, dict) else {"name": pkg}
        name = meta.get("name", "?")
        ok = _pkg_importable(py, name)
        used = meta.get("used_for", "")
        suffix = f" — {used}" if used else ""
        print(f"  {'✓' if ok else '✗'} python pkg: {name}{suffix}")
        if not ok:
            hint = meta.get("install_cmd") or f"uv pip install {name}"
            errs.append(f"python pkg missing: {name} — `{hint}`")

    _root = repo_root()
    for link in cfg.get("path_links", []) or []:
        target = link.get("target", "")
        tp = _root / target
        ok = tp.exists()
        used = link.get("used_for", "")
        suffix = f" — {used}" if used else ""
        print(f"  {'✓' if ok else '✗'} path link: {target}{suffix}")
        if not ok:
            hint = link.get("install_cmd") or f"create {tp}"
            errs.append(f"path link missing: {tp} — `{hint}`")

    hw = detect_hardware()
    print(f"  hardware: {hw}")

    gpus, gpu_err = parse_gpu_status()
    if gpu_err:
        warns.append(gpu_err)
        print(f"  ⚠ GPU: {gpu_err}")
    else:
        free = free_gpu_indices(gpus)
        gpu_pool = getattr(args, "gpu_pool", None)
        if gpu_pool:
            pool_set = {int(s.strip()) for s in gpu_pool.split(",") if s.strip()}
            free = [gpu for gpu in free if gpu in pool_set]
        busy_pids = gpu_busy_processes()
        print(f"  GPUs: total {len(gpus)}, free {len(free)} ({free}), "
              f"busy {len(gpus) - len(free)}")
        rows_for_gpu_check = _select_rows(
            cfg.get("rows", []),
            getattr(args, "benchmarks", "all"),
            getattr(args, "exclude_ids", None),
        )
        required_gpus = max(
            (_row_total_gpus(row, cfg) for row in rows_for_gpu_check),
            default=1,
        )
        if len(free) < required_gpus:
            err = (
                f"need {required_gpus} free GPU(s), found {len(free)} — "
                f"busy: {gpus}; free them yourself "
                "(this skill never kills user processes)"
            )
            if busy_pids:
                err += f"; busy PIDs: {busy_pids}"
            errs.append(err)

    return _emit_precheck_summary(
        errs, warns, out_dir, cfg, py, src,
        versions=versions, gpus=gpus, hardware=hw,
    )


def _hf_cached(py: str, repo_id: str, kind: str) -> bool:
    extra = ", repo_type='dataset'" if kind == "dataset" else ""
    proc = subprocess.run(
        [py, "-c",
         "from huggingface_hub import try_to_load_from_cache;"
         f"p = try_to_load_from_cache('{repo_id}', 'config.json'{extra});"
         "print(bool(p))"],
        capture_output=True, text=True, timeout=10,
    )
    if proc.returncode == 0 and proc.stdout.strip() == "True":
        return True
    cache = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    prefix = "models--" if kind == "model" else "datasets--"
    slug = prefix + repo_id.replace("/", "--")
    return (cache / "hub" / slug).exists()


def _modelscope_cached(repo_id: str) -> bool:
    """Check modelscope's default cache: ~/.cache/modelscope/hub/models/<id>."""
    base = Path(os.environ.get(
        "MODELSCOPE_CACHE",
        os.path.expanduser("~/.cache/modelscope/hub/models"),
    ))
    return (base / repo_id).exists()


def _pkg_importable(py: str, name: str) -> bool:
    """Try `import <name>` under the configured venv python."""
    proc = subprocess.run(
        [py, "-c", f"import {name}"],
        capture_output=True, text=True, timeout=20,
    )
    return proc.returncode == 0


_FAIL_EXCERPT_KEYWORDS = re.compile(
    r"Error|FAILED|Failed|OutOfMemory|ModuleNotFound|FileNotFound"
    r"|ValueError|RuntimeError|TypeError|Killed",
    re.IGNORECASE,
)


def _extract_fail_excerpt(*log_paths) -> str | None:
    """Pull a useful 1-paragraph hint out of failure logs.

    Looks (in order: first arg first) for the last Python Traceback (up to
    ~2 KB), else the last line in the tail matching common error keywords.
    Returns None if nothing relevant found.
    """
    for p in [Path(x) for x in log_paths]:
        if not p.exists():
            continue
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        # Last full Traceback block.
        tb_starts = [
            m.start() for m in
            re.finditer(r"^Traceback \(most recent call last\):",
                        text, re.MULTILINE)
        ]
        if tb_starts:
            tb = text[tb_starts[-1]:]
            stop = re.search(r"\n\d{4}-\d{2}-\d{2}|\n\[(INFO|WARN)", tb)
            if stop:
                tb = tb[:stop.start()]
            return tb[:2000].rstrip()
        # Otherwise scan the tail for an error-shaped line.
        for ln in reversed(text.splitlines()[-50:]):
            ln = ln.strip()
            if ln and _FAIL_EXCERPT_KEYWORDS.search(ln):
                return ln[:500]
    return None


def _emit_precheck_summary(errs, warns, out_dir, cfg, py, src, **extra) -> int:
    print("")
    if errs:
        print(f"FAIL ({len(errs)} error(s)):")
        for err in errs:
            print(f"  ✗ {err}")
    if warns:
        print(f"WARN ({len(warns)}):")
        for warn in warns:
            print(f"  ⚠ {warn}")
    if not errs:
        print("OK")
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "precheck.json").write_text(json.dumps({
            "ts": now_iso(),
            "model": cfg.get("name"),
            "venv_python": py,
            "venv_source": src,
            "errors": errs,
            "warnings": warns,
            **extra,
        }, indent=2))
    return 0 if not errs else 1


# ---------- run ----------

class ServerHandle:
    def __init__(
        self,
        popen: subprocess.Popen,
        port: int,
        log_path: Path,
        cleanup_manifest: Path | None = None,
    ):
        self.popen = popen
        self.port = port
        self.log_path = log_path
        self.cleanup_manifest = cleanup_manifest

    def stop(self) -> None:
        try:
            try:
                os.killpg(os.getpgid(self.popen.pid), signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                self.popen.wait(timeout=15)
                return
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(os.getpgid(self.popen.pid), signal.SIGKILL)
            except ProcessLookupError:
                return
            try:
                self.popen.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        finally:
            if self.cleanup_manifest is not None:
                _cleanup_process_groups_from_manifest(self.cleanup_manifest)


def launch_server(cmd_template: str, *, port: int, model: str,
                  gpus_csv: str, log_path: Path, repo_root: Path = Path("."),
                  server_extra: str = "",
                  env_extra: dict | None = None) -> ServerHandle:
    cmd = " ".join(part for part in (cmd_template, server_extra) if part)
    cmd = cmd.format(port=port, model=model, gpus=gpus_csv,
                     repo_root=str(repo_root))
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    if gpus_csv:
        env["CUDA_VISIBLE_DEVICES"] = gpus_csv
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w")
    try:
        popen = subprocess.Popen(
            cmd, shell=True, stdout=log_fh, stderr=subprocess.STDOUT,
            env=env, cwd=str(repo_root), start_new_session=True,
        )
    finally:
        log_fh.close()
    return ServerHandle(popen, port, log_path)


def launch_row_server(
    row: dict,
    cfg: dict,
    py: str,
    *,
    port: int,
    model: str,
    gpus_csv: str,
    log_path: Path,
    round_dir: Path,
    repo_root: Path,
    server_extra: str = "",
) -> ServerHandle:
    profile = _resolve_server_profile(row, cfg, model)
    if profile is None:
        return launch_server(
            row["server"],
            port=port,
            model=model,
            gpus_csv=gpus_csv,
            log_path=log_path,
            repo_root=repo_root,
            server_extra=server_extra,
        )
    return launch_managed_router_server(
        profile,
        py,
        port=port,
        model=model,
        gpus_csv=gpus_csv,
        log_path=log_path,
        round_dir=round_dir,
        repo_root=repo_root,
        server_extra=server_extra,
    )


def launch_managed_router_server(
    profile: dict,
    py: str,
    *,
    port: int,
    model: str,
    gpus_csv: str,
    log_path: Path,
    round_dir: Path,
    repo_root: Path,
    server_extra: str = "",
) -> ServerHandle:
    num_workers = int(profile.get("num_workers", 2))
    worker_base_port = free_port_range(num_workers, excluded={port})
    cleanup_manifest = round_dir / "router_pgids.txt"
    launcher_config = round_dir / "launcher.yaml"
    worker_extra_args = str(profile.get("worker_extra_args", "")).format(
        model=model,
        repo_root=str(repo_root),
    )
    if server_extra:
        worker_extra_args = f"{worker_extra_args} {server_extra}".strip()
    launcher_config.write_text(
        dump_yaml(
            {
                "launcher": {
                    "backend": "local",
                    "model_path": str(profile.get("model_path", model)).format(
                        model=model,
                        repo_root=str(repo_root),
                    ),
                    "model_name": str(profile["model_name"]).format(
                        model=model,
                        repo_root=str(repo_root),
                    ),
                    "num_workers": num_workers,
                    "num_gpus_per_worker": int(profile.get("num_gpus_per_worker", 1)),
                    "worker_host": profile.get("worker_host", "127.0.0.1"),
                    "worker_base_port": worker_base_port,
                    "worker_extra_args": worker_extra_args,
                    "wait_timeout": int(profile.get("wait_timeout", 900)),
                }
            }
        ),
        encoding="utf-8",
    )
    env = dict(os.environ)
    if gpus_csv:
        env["CUDA_VISIBLE_DEVICES"] = gpus_csv
    env["SGLANG_OMNI_ROUTER_CLEANUP_MANIFEST"] = str(cleanup_manifest)
    cmd = [
        py,
        "-m",
        "sglang_omni_router.serve",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--launcher-config",
        str(launcher_config),
        "--policy",
        profile.get("router_policy", "least_request"),
        "--health-success-threshold",
        "1",
        "--health-failure-threshold",
        "2",
        "--health-check-interval-secs",
        "2",
        "--log-level",
        "info",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w")
    try:
        popen = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(repo_root),
            start_new_session=True,
        )
    finally:
        log_fh.close()
    try:
        _record_process_group(cleanup_manifest, os.getpgid(popen.pid))
    except Exception:
        ServerHandle(popen, port, log_path, cleanup_manifest=cleanup_manifest).stop()
        raise
    return ServerHandle(popen, port, log_path, cleanup_manifest=cleanup_manifest)


def _resolve_server_profile(row: dict, cfg: dict, model_id: str) -> dict | None:
    profiles = cfg.get("server_profiles") or {}
    profile_name = row.get("server_profile")
    if profile_name is None:
        profile_name = (cfg.get("server_profile_by_hf_model_id") or {}).get(model_id)
    if profile_name is None:
        profile_name = cfg.get("default_server_profile")
    if profile_name is None:
        return None
    try:
        profile = dict(profiles[profile_name])
    except KeyError as exc:
        raise SystemExit(f"server profile not found: {profile_name}") from exc
    profile["name"] = profile_name
    return profile


def _row_server_gpus(row: dict, cfg: dict) -> int:
    model_id = row.get("hf_model_id") or cfg.get("hf_model_id", "")
    profile = _resolve_server_profile(row, cfg, model_id)
    if profile is not None:
        return int(profile.get("server_gpus", row.get("server_gpus", 1)))
    return int(row.get("server_gpus", 1))


def _row_server_boot_timeout(row: dict, cfg: dict) -> int:
    model_id = row.get("hf_model_id") or cfg.get("hf_model_id", "")
    profile = _resolve_server_profile(row, cfg, model_id)
    if profile is not None:
        return int(
            profile.get("server_boot_timeout_s", row.get("server_boot_timeout_s", 300))
        )
    return int(row.get("server_boot_timeout_s", 300))


def _record_process_group(manifest: Path, process_group_id: int) -> None:
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(f"{process_group_id}\n")


def _cleanup_process_groups_from_manifest(manifest: Path) -> None:
    if not manifest.exists():
        return
    process_group_ids: set[int] = set()
    for line in manifest.read_text().splitlines():
        try:
            process_group_ids.add(int(line.strip()))
        except ValueError:
            continue
    for sig, delay in ((signal.SIGTERM, 5), (signal.SIGKILL, 1)):
        remaining: set[int] = set()
        for process_group_id in process_group_ids:
            try:
                os.killpg(process_group_id, sig)
                remaining.add(process_group_id)
            except ProcessLookupError:
                continue
        if not remaining:
            return
        time.sleep(delay)
        process_group_ids = {
            process_group_id
            for process_group_id in remaining
            if _process_group_exists(process_group_id)
        }


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
        return True
    except ProcessLookupError:
        return False


def wait_health(url: str, timeout_s: int) -> tuple[bool, str]:
    deadline = time.monotonic() + timeout_s
    last_err = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if 200 <= resp.status < 300:
                    return True, "ok"
        except urllib.error.URLError as exc:
            last_err = str(exc)
        except Exception as exc:  # noqa: BLE001
            last_err = repr(exc)
        time.sleep(2)
    return False, f"timeout after {timeout_s}s; last_err={last_err}"


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_model_config(args.model)
    py, _, _ = resolve_venv_python(args.venv_python, cfg)
    if not Path(py).exists():
        raise SystemExit(f"venv python missing: {py} — run precheck first")
    venv_bin = str(Path(py).parent)
    os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"
    root = repo_root()
    out = (Path(args.output_dir) if args.output_dir
           else root / DEFAULT_RUN_ROOT_REL / ts_dir())
    out.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {out}")

    rows = cfg.get("rows", [])
    selected = _select_rows(rows, args.benchmarks, args.exclude_ids)
    if args.retry_failed_from:
        prior_path = Path(args.retry_failed_from)
        if prior_path.is_dir():
            prior_path = prior_path / "run-state.json"
        if not prior_path.exists():
            raise SystemExit(f"--retry-failed-from: state file not found: {prior_path}")
        prior = json.loads(prior_path.read_text())
        prior_by_id = {r["id"]: r for r in prior.get("rows", [])}

        def _is_done(rid: str) -> bool:
            ps = prior_by_id.get(rid)
            if not ps or ps.get("status") != "ok":
                return False
            edit = ps.get("edit") or {}
            return (edit.get("status") == "ok"
                    and edit.get("action") in ("replaced", "appended"))

        before = len(selected)
        selected = [r for r in selected if not _is_done(r["id"])]
        skipped = before - len(selected)
        print(f"retry-failed-from {prior_path}: "
              f"skipping {skipped} row(s) already done")

    if not selected:
        known = set().union(*(_benchmark_selectors(row) for row in rows))
        raise SystemExit(
            "no rows selected — pass --benchmarks NAME,NAME or --benchmarks all. "
            f"Known: {sorted(known)}"
        )

    rounds = max(1, args.rounds)
    smoke = args.smoke
    hw = detect_hardware()
    src_prefix = git_source_prefix(root)
    print(f"hardware: {hw}; source_prefix: {src_prefix}")
    print(f"selected: {len(selected)} row(s); rounds={rounds}; smoke={smoke}")

    state = {
        "ts": now_iso(),
        "model": cfg["name"],
        "rounds": rounds,
        "smoke": smoke,
        "venv_python": py,
        "hardware": hw,
        "source_prefix": src_prefix,
        "rows": [],
    }
    gpu_pool: list[int] | None = None
    if args.gpu_pool:
        gpu_pool = [int(s.strip()) for s in args.gpu_pool.split(",") if s.strip()]
        print(f"gpu_pool: {gpu_pool}")
    for row in selected:
        row_state = _run_one_row(row, py, root, out, rounds, smoke, cfg, hw,
                                 gpu_pool=gpu_pool)
        if row_state["status"] == "ok" and smoke is None:
            row_state["edit"] = _try_edit_inplace(
                row, row_state, root, hw, src_prefix,
            )
        elif smoke is not None:
            row_state["edit"] = {
                "status": "skipped",
                "reason": f"smoke run --max-samples {smoke}; no edits applied",
            }
        else:
            row_state["edit"] = {
                "status": "skipped",
                "reason": "row run failed; no edit attempted",
            }
        state["rows"].append(row_state)
        (out / "run-state.json").write_text(json.dumps(state, indent=2))

    runs_failed = [r for r in state["rows"] if r["status"] != "ok"]
    edits_failed = [r for r in state["rows"]
                    if r.get("edit", {}).get("status") not in ("ok", "skipped")]
    actions: dict[str, int] = {}
    for r in state["rows"]:
        a = r.get("edit", {}).get("action")
        if a:
            actions[a] = actions.get(a, 0) + 1

    print("")
    print("=== summary ===")
    print(f"  ran ok:   {len(state['rows']) - len(runs_failed)}/{len(state['rows'])}")
    if actions:
        for a, n in sorted(actions.items()):
            print(f"  edits {a}: {n}")
    if runs_failed:
        print(f"\n  failed runs:")
        for r in runs_failed:
            print(f"    ✗ {r['id']}: {r['reason']}")
            ex = r.get("fail_excerpt")
            if ex:
                for ln in ex.splitlines()[-3:]:
                    print(f"      | {ln}")
    if edits_failed:
        print(f"\n  failed edits (run was ok but write didn't go through):")
        for r in edits_failed:
            print(f"    ⚠ {r['id']}: {r['edit']['reason']}")
    print(f"\nrun-state: {out / 'run-state.json'}")
    if runs_failed or edits_failed:
        return 2
    return 0


def _run_one_row(row: dict, py: str, root: Path, out_root: Path,
                 rounds: int, smoke: int | None, cfg: dict, hw: str,
                 gpu_pool: list[int] | None = None) -> dict:
    row_id = row["id"]
    row_dir = out_root / row_id
    row_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[row] {row_id}")

    gpu_count = _row_server_gpus(row, cfg)
    total_gpu_count = _row_total_gpus(row, cfg)
    pool_set = set(gpu_pool) if gpu_pool is not None else None
    last_err = ""
    free: list[int] = []
    # Retry GPU availability check — when a previous row crashes fast,
    # CUDA memory release lags behind process exit by a few seconds.
    for attempt in range(10):
        gpus, last_err = parse_gpu_status()
        if last_err:
            time.sleep(3)
            continue
        free = free_gpu_indices(gpus)
        if pool_set is not None:
            free = [g for g in free if g in pool_set]
        if len(free) >= total_gpu_count:
            break
        time.sleep(3)
    else:
        pool_note = f" (pool={gpu_pool})" if gpu_pool is not None else ""
        reason = last_err or (
            f"not enough free GPUs (need {total_gpu_count}, have {len(free)})"
            f"{pool_note}"
        )
        return {"id": row_id, "status": "fail", "reason": reason, "rounds": []}
    chosen = free[:gpu_count]
    gpus_csv = ",".join(str(g) for g in chosen)
    # Pick an ASR GPU from the leftover free pool (highest index first, to
    # leave low indices for subsequent server allocations). Rows with
    # `{asr_gpu}` require a real free GPU because the server remains live while
    # the client transcribes.
    remaining_free = [g for g in free if g not in set(chosen)]
    asr_device = f"cuda:{remaining_free[-1]}" if _row_needs_asr_gpu(row) else "cpu"
    print(f"  gpus: {gpus_csv}  asr: {asr_device}")

    rounds_state: list[dict] = []
    for k in range(1, rounds + 1):
        round_dir = row_dir / f"round_{k}"
        round_dir.mkdir(parents=True, exist_ok=True)
        rstate = _run_one_round(row, py, root, round_dir, gpus_csv, smoke, cfg,
                                hw,
                                asr_device=asr_device)
        rounds_state.append(rstate)
        if rstate["status"] != "ok":
            break
        # Talker pipelines (server_gpus=2) have shown round-to-round
        # GPU memory leak: stage processes are torn down but driver
        # contexts can hold ~60 GB until refcounts hit zero. Wait
        # actively for the assigned GPUs to drop below half-utilized
        # before the next round, up to ~60 s.
        if k < rounds and gpu_count >= 2:
            _wait_for_memory_release(chosen, low_watermark_mb=70_000,
                                     max_wait_s=60)
    overall = "ok" if rounds_state and all(
        r["status"] == "ok" for r in rounds_state
    ) else "fail"
    last_reason = next(
        (r["reason"] for r in reversed(rounds_state) if r["status"] != "ok"),
        "",
    )
    last_excerpt = next(
        (r.get("fail_excerpt") for r in reversed(rounds_state)
         if r["status"] != "ok"),
        None,
    )
    return {
        "id": row_id,
        "status": overall,
        "reason": last_reason,
        "fail_excerpt": last_excerpt,
        "gpus": gpus_csv,
        "rounds": rounds_state,
    }


def _run_one_round(row: dict, py: str, root: Path, round_dir: Path,
                   gpus_csv: str, smoke: int | None, cfg: dict, hw: str, *,
                   asr_device: str = "cpu") -> dict:
    port = free_port()
    server_log = round_dir / "server.log"
    client_log = round_dir / "client.log"
    out_dir = round_dir / "out"
    out_dir.mkdir(exist_ok=True)

    server_extra = (row.get("server_extra_by_hardware") or {}).get(hw, "")
    model_id = row.get("hf_model_id") or cfg.get("hf_model_id", "")

    print(f"  [{round_dir.name}] launching server (port={port}) ...", flush=True)
    handle = launch_row_server(
        row,
        cfg,
        py,
        port=port,
        model=model_id,
        gpus_csv=gpus_csv,
        log_path=server_log,
        round_dir=round_dir,
        repo_root=root,
        server_extra=server_extra,
    )
    try:
        health = row.get("server_health", "http://localhost:{port}/health").format(port=port)
        timeout = _row_server_boot_timeout(row, cfg)
        ok, msg = wait_health(health, timeout)
        if not ok:
            return {
                "status": "fail",
                "reason": f"server boot: {msg}",
                "round_dir": str(round_dir),
                "server_log": str(server_log),
                "fail_excerpt": _extract_fail_excerpt(server_log),
            }
        print(f"    server ready (<{timeout}s)")

        client_tmpl = row["client"]
        if smoke is not None:
            client_tmpl = _override_max_samples(client_tmpl, smoke)
        client_cmd = client_tmpl.format(
            port=port, output_dir=str(out_dir), model=model_id,
            asr_gpu=asr_device,
        )
        print(f"    client: {_truncate(client_cmd, 110)}")
        with open(client_log, "w") as flog:
            r = subprocess.run(
                client_cmd, shell=True, stdout=flog, stderr=subprocess.STDOUT,
                cwd=str(root),
                env={**os.environ, "OUTPUT_DIR": str(out_dir)},
            )
        if r.returncode != 0:
            return {
                "status": "fail",
                "reason": f"client exit {r.returncode}",
                "round_dir": str(round_dir),
                "client_log": str(client_log),
                "server_log": str(server_log),
                "fail_excerpt": _extract_fail_excerpt(client_log, server_log),
            }

        rj = out_dir / row["result_json"]
        if not rj.exists():
            return {
                "status": "fail",
                "reason": f"result.json missing: {rj.name}",
                "round_dir": str(round_dir),
                "client_log": str(client_log),
                "server_log": str(server_log),
                "fail_excerpt": _extract_fail_excerpt(client_log, server_log),
            }

        # Sanity check the result.json BEFORE declaring the row ok. Some
        # benchmark clients catch per-sample exceptions and write a valid
        # JSON with `evaluated=0` (e.g. an entire transcribe phase skipped
        # because a python dep was missing). Without this check the runner
        # would happily write "0.00%" cells over real reference values.
        try:
            rj_data = json.loads(rj.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            return {
                "status": "fail",
                "reason": f"result.json unreadable: {exc}",
                "round_dir": str(round_dir),
                "client_log": str(client_log),
                "server_log": str(server_log),
                "fail_excerpt": _extract_fail_excerpt(client_log, server_log),
            }
        sanity_issues = _sanity_check_result(rj_data)
        if sanity_issues:
            return {
                "status": "fail",
                "reason": "sanity: " + "; ".join(sanity_issues),
                "round_dir": str(round_dir),
                "client_log": str(client_log),
                "server_log": str(server_log),
                "fail_excerpt": _extract_fail_excerpt(client_log, server_log),
            }
        return {
            "status": "ok",
            "result_json": str(rj),
            "round_dir": str(round_dir),
            "client_log": str(client_log),
            "server_log": str(server_log),
        }
    finally:
        print(f"    stopping server", flush=True)
        handle.stop()


def _sanity_check_result(rj: dict) -> list[str]:
    """Catch upstream-skip bugs that yield a writable but garbage result.json.

    Heuristics avoid false positives on legitimately-zero metrics (e.g.
    WER ~ 0% is fine for high-quality TTS) — instead we verify that
    ENOUGH samples were actually processed, using whichever processed/
    total field pair the schema provides. Returns a list of human-readable
    issues; empty list = OK.
    """
    if not isinstance(rj, dict):
        return ["result.json top-level is not a dict"]
    s = rj.get("summary")
    if not isinstance(s, dict):
        return ["result.json has no `summary` dict"]

    issues: list[str] = []
    total = s.get("total_samples")
    if isinstance(total, int) and total > 0:
        for processed_key in ("evaluated", "parseable_samples"):
            n = s.get(processed_key)
            if isinstance(n, int) and n < total * 0.5:
                issues.append(
                    f"{processed_key}={n}/{total} (>50% skipped — "
                    f"likely upstream dep failure)"
                )
    completed = s.get("completed_requests")
    failed = s.get("failed_requests")
    if (isinstance(completed, int) and isinstance(failed, int)
            and completed + failed > 0
            and failed > completed * 0.5):
        issues.append(
            f"failed_requests={failed} > completed={completed} "
            "(>50% server-side failures)"
        )
    return issues


def _override_max_samples(client: str, n: int) -> str:
    if "--max-samples" in client:
        return re.sub(r"--max-samples\s+\d+", f"--max-samples {n}", client)
    return client + f" --max-samples {n}"


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


# ---------- in-place edit (the core of the new design) ----------

@contextmanager
def _file_edit_lock(file_path: Path):
    """Cross-process advisory lock so parallel runners editing the same
    benchmark_*.py serialize cleanly. Lock file lives next to target."""
    lock_path = file_path.with_suffix(file_path.suffix + ".edit.lock")
    fp = open(lock_path, "w")
    try:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
        fp.close()


def _try_edit_inplace(row: dict, row_state: dict, root: Path,
                     hw: str, src_prefix: str) -> dict:
    """Apply this row's run output to benchmark_*.py.

    Returns: {status: ok|fail|skipped, action?: replaced|appended,
              reason?: str, signature?: str}.
    """
    file_rel = row["file"]
    file_path = root / file_rel
    if not file_path.exists():
        return {"status": "fail", "reason": f"file missing: {file_rel}"}

    with _file_edit_lock(file_path):
        return _try_edit_inplace_locked(row, row_state, root, hw, src_prefix,
                                        file_path, file_rel)


def _try_edit_inplace_locked(row: dict, row_state: dict, root: Path,
                              hw: str, src_prefix: str,
                              file_path: Path, file_rel: str) -> dict:
    locate = row.get("locate", {}) or {}
    section_substring = locate.get("section_substring", "")
    config_substring = locate.get("config_substring", "")
    workload = locate.get("source_workload", "")
    if not section_substring or not config_substring:
        return {
            "status": "fail",
            "reason": "row.locate.{section_substring, config_substring} required",
        }

    full_hw_tag = f"[{hw}, {workload}]" if workload else f"[{hw}]"

    text = file_path.read_text()
    candidates = list(_find_table_rows_official(text))

    section_rows = [
        c for c in candidates
        if section_substring in (c.get("section_line") or "")
        and config_substring in c["row_text"]
    ]
    if not section_rows:
        return {
            "status": "fail",
            "reason": (
                f"no official table row matches "
                f"section={section_substring!r} + "
                f"config_substring={config_substring!r}"
            ),
        }

    matched = [c for c in section_rows if full_hw_tag in c["row_text"]]
    if len(matched) > 1:
        return {
            "status": "fail",
            "reason": (
                f"multiple rows match {full_hw_tag} "
                f"({len(matched)}); narrow your locate.* substrings"
            ),
        }

    if not row_state.get("rounds"):
        return {"status": "fail", "reason": "no completed rounds"}
    rj_path = Path(row_state["rounds"][0]["result_json"])
    try:
        rj = json.loads(rj_path.read_text())
    except json.JSONDecodeError as exc:
        return {"status": "fail", "reason": f"result.json invalid: {exc}"}
    try:
        new_cells_by_col = _build_new_cells(row, rj)
    except RuntimeError as exc:
        return {"status": "fail", "reason": str(exc)}

    new_source = (f"{src_prefix} [{hw}, {workload}]" if workload
                  else f"{src_prefix} [{hw}]")

    if matched:
        target = matched[0]
        new_row_text = _build_replacement_row(
            target, new_cells_by_col, new_source,
        )
        new_text = text.replace(target["row_text"], new_row_text, 1)
        if new_text == text:
            return {
                "status": "fail",
                "reason": "row text not replaced (string mismatch — probably duplicate)",
            }
        file_path.write_text(new_text)
        return {
            "status": "ok",
            "action": "replaced",
            "signature": _row_sig(target),
        }

    # No matching hw row — append at end of this section's table.
    template = section_rows[-1]
    new_row_text = _build_appended_row(
        template, new_cells_by_col, new_source,
    )
    lines = text.splitlines(keepends=True)
    table_end = _find_table_end(lines, template["line_no"] - 1)
    new_lines = lines[:table_end] + [new_row_text + "\n"] + lines[table_end:]
    file_path.write_text("".join(new_lines))
    return {
        "status": "ok",
        "action": "appended",
        "signature": f"new {full_hw_tag} row in {file_rel}:{table_end + 1}",
    }


def _build_new_cells(row: dict, result_json: Any) -> dict[str, str]:
    cell_specs = row.get("cells", {}) or {}
    new_by_col: dict[str, str] = {}
    for col_name, spec in cell_specs.items():
        try:
            if "paths" in spec and spec["paths"]:
                values = {alias: _json_path(result_json, p)
                          for alias, p in spec["paths"].items()}
                fmt = spec.get("format",
                               " | ".join(f"{a}={{{a}}}" for a in values))
                new_by_col[col_name] = fmt.format(**values)
            elif spec.get("path"):
                value = _json_path(result_json, spec["path"])
                fmt = spec.get("format", "{v}")
                new_by_col[col_name] = fmt.format(v=value)
            else:
                raise RuntimeError(
                    f"cell {col_name}: missing 'path' or 'paths'"
                )
        except KeyError as exc:
            raise RuntimeError(
                f"cell {col_name}: JSON path missing: {exc}"
            ) from exc
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"cell {col_name}: format error: {exc}"
            ) from exc
    return new_by_col


def _build_replacement_row(target: dict, new_cells_by_col: dict[str, str],
                           new_source: str) -> str:
    headers = target["headers"]
    cells = target["cells_text"]
    out_cells: list[str] = []
    for i, col in enumerate(headers):
        if col == "Source":
            out_cells.append(new_source)
        elif col in new_cells_by_col:
            out_cells.append(new_cells_by_col[col])
        else:
            out_cells.append(cells[i])
    return _rebuild_row(target["row_text"], out_cells)


def _build_appended_row(template: dict, new_cells_by_col: dict[str, str],
                        new_source: str) -> str:
    """Build an append-row using the template's existing column widths.

    Model + Config columns copy the template's values verbatim (Config
    is shared across hw rows in our schema). Other columns come from
    new_cells_by_col; Source is the supplied string."""
    headers = template["headers"]
    template_cells = template["cells_text"]
    out_cells: list[str] = []
    for i, col in enumerate(headers):
        if col == "Source":
            out_cells.append(new_source)
        elif col in new_cells_by_col:
            out_cells.append(new_cells_by_col[col])
        else:
            out_cells.append(template_cells[i])
    return _rebuild_row(template["row_text"], out_cells)


def _row_sig(target: dict) -> str:
    cells = target["cells_text"]
    sec = (target.get("section_line") or "")[:30]
    config_part = cells[1].strip() if len(cells) >= 2 else ""
    source_part = cells[-1].strip() if cells else ""
    return f"{sec} / {config_part} / {source_part}"


# ---------- markdown table parsing ----------

_SEPARATOR_CELL_RE = re.compile(r"^\s*:?-+:?\s*$")


def _is_pipe_row(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.endswith("|") and "|" in s[1:-1]


def _is_separator(line: str) -> bool:
    if not _is_pipe_row(line):
        return False
    cells = _split_md_row(line)
    return all(_SEPARATOR_CELL_RE.match(c) for c in cells)


def _split_md_row(line: str) -> list[str]:
    s = line.rstrip()
    if not (s.lstrip().startswith("|") and s.endswith("|")):
        return []
    indent = len(s) - len(s.lstrip())
    inner = s[indent + 1: -1]
    return inner.split("|")


def _is_local_table(lines: list[str], data_idx_0based: int) -> bool:
    """Walk up from a data row; True if its parent header is a Local section."""
    upper_bound = max(0, data_idx_0based - _PARENT_SCAN_LINES)
    for k in range(data_idx_0based - 1, upper_bound - 1, -1):
        s = lines[k].strip()
        if not s:
            continue
        if any(m in s for m in _LOCAL_MARKERS):
            return True
        if any(m in s for m in _OFFICIAL_MARKERS):
            return False
    return False  # default: official


def _find_table_rows_official(text: str):
    """Yield each data row in every markdown table EXCEPT Local sections.

    Each item: {line_no (1-based), row_text, headers, cells_text, section_line}.
    """
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if not _is_pipe_row(lines[i]):
            i += 1
            continue
        if i + 1 >= len(lines) or not _is_separator(lines[i + 1]):
            i += 1
            continue
        if _is_local_table(lines, i):
            j = i + 2
            while j < len(lines) and _is_pipe_row(lines[j]) and not _is_separator(lines[j]):
                j += 1
            i = max(j, i + 1)
            continue
        headers = [c.strip() for c in _split_md_row(lines[i])]
        section_line: str | None = None
        for k in range(i - 1, -1, -1):
            if lines[k].strip() == "":
                continue
            if _is_pipe_row(lines[k]):
                break
            section_line = lines[k].strip()
            break
        j = i + 2
        while j < len(lines) and _is_pipe_row(lines[j]) and not _is_separator(lines[j]):
            yield {
                "line_no": j + 1,
                "row_text": lines[j],
                "headers": headers,
                "cells_text": [c.strip() for c in _split_md_row(lines[j])],
                "section_line": section_line,
            }
            j += 1
        i = max(j, i + 1)


def _find_table_end(lines: list[str], start_idx_0based: int) -> int:
    """Given a 0-based index that's a data row, return the 0-based index of
    the line immediately AFTER the last consecutive pipe-row of that table."""
    i = start_idx_0based
    while i < len(lines) and _is_pipe_row(lines[i]):
        i += 1
    return i


def _json_path(obj: Any, path: str) -> Any:
    if not path:
        raise KeyError("(empty path)")
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                raise KeyError(part)
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError) as exc:
                raise KeyError(part) from exc
        else:
            raise KeyError(part)
    return cur


def _rebuild_row(orig_row: str, new_cells: list[str]) -> str:
    """Rebuild a markdown row preserving the original column widths.

    `orig_row` is the raw line including any leading indent; `new_cells`
    is the list of stripped new cell values, one per column. If counts
    mismatch we fall back to a minimally-spaced row.
    """
    raw_cells = _split_md_row(orig_row)
    indent = orig_row[: len(orig_row) - len(orig_row.lstrip())]
    if len(raw_cells) != len(new_cells):
        return indent + "| " + " | ".join(c.strip() for c in new_cells) + " |"
    out_parts: list[str] = []
    for raw, new in zip(raw_cells, new_cells):
        target = len(raw)
        new_str = new.strip()
        if len(new_str) + 2 <= target:
            cell = " " + new_str + " " * (target - 2 - len(new_str)) + " "
        else:
            cell = " " + new_str + " "
        out_parts.append(cell)
    return indent + "|" + "|".join(out_parts) + "|"


# ---------- main ----------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="runner.py",
        description="running-eval-suite skill backend",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"model name under models/ (default: {DEFAULT_MODEL})")
    ap.add_argument("--venv-python", default=None,
                    help="override venv python; or set $EVAL_VENV_PYTHON")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("precheck",
                       help="verify env (venv, sglang/torch, HF cache, GPU, hardware)")
    p.add_argument("--output-dir", default=None,
                   help="dir to write precheck.json (default: stdout only)")
    p.add_argument("--benchmarks", default="all",
                   help="same benchmark selector filter as run; used for GPU "
                        "requirement checks")
    p.add_argument("--exclude-ids", default=None,
                   help="same row-id substring exclusion filter as run")
    p.add_argument("--gpu-pool", default=None,
                   help="same GPU pool restriction as run")

    p = sub.add_parser("run",
                       help="launch server + client per row, edit benchmark_*.py in place")
    p.add_argument("--benchmarks", default="all",
                   help="comma-separated benchmark short names "
                        "(e.g. mmsu,mmmu,omni_seedtts,tts_seedtts), "
                        "or 'all' for every row in the model's config.yaml")
    p.add_argument("--exclude-ids", default=None,
                   help="comma-separated row-id substrings to skip "
                        "(e.g. 's2pro-' to drop S2-Pro rows when --benchmarks "
                        "would otherwise pick them up via shared short names)")
    p.add_argument("--retry-failed-from", default=None,
                   help="path to a prior run's state.json (or run-dir containing "
                        "one); skip rows that succeeded in that run "
                        "(status=ok AND edit.action in {replaced, appended}). "
                        "Use after a previous run to re-execute just the failed "
                        "rows without recomputing --exclude-ids by hand.")
    p.add_argument("--gpu-pool", default=None,
                   help="comma-separated GPU indices to restrict eligible "
                        "GPUs (e.g. '0,1' for a 2-GPU pool). Use to run "
                        "multiple runners in parallel on disjoint pools; "
                        "default: all visible GPUs")
    p.add_argument("--rounds", type=int, default=1,
                   help="rounds per row (default 1)")
    p.add_argument("--smoke", type=int, default=None,
                   help="set --max-samples N (smoke; no edits applied)")
    p.add_argument("--output-dir", default=None,
                   help=f"run dir; default: <repo>/{DEFAULT_RUN_ROOT_REL}/<ts>")

    args = ap.parse_args(argv)
    if args.cmd == "precheck":
        return cmd_precheck(args)
    if args.cmd == "run":
        return cmd_run(args)
    raise SystemExit(f"unknown subcommand: {args.cmd}")


if __name__ == "__main__":
    sys.exit(main() or 0)
