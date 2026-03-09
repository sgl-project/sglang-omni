# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for Qwen3-Omni parity harnesses."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


DEFAULT_MODEL_PATH = Path(
    "/root/.cache/huggingface/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/"
    "snapshots/26291f793822fb6be9555850f06dfe95f2d7e695"
)
DEFAULT_PROMPT = (
    "Please speak this exact sentence once, naturally and clearly: "
    "Hello there, this is a longer speech validation sample generated "
    "after the Talker parity fix."
)
DEFAULT_SPEAKER = "Ethan"
DEFAULT_SEED = 123
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_OUT_DIR = Path("/tmp/qwen3_omni_parity")
FLOAT_PASS_THRESHOLD = 0.9999


def add_repo_root_to_syspath() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


def default_model_path() -> Path:
    return Path(os.environ.get("QWEN3_OMNI_MODEL_PATH", DEFAULT_MODEL_PATH))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def prompt_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]


def file_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def save_json(data: Any, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return path


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def metric(a: torch.Tensor | Iterable[float], b: torch.Tensor | Iterable[float]) -> dict[str, float]:
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(list(a))
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(list(b))
    a = a.float().reshape(-1).cpu()
    b = b.float().reshape(-1).cpu()
    if a.numel() == 0 and b.numel() == 0:
        return {
            "cosine": 1.0,
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "rmse": 0.0,
        }
    if a.numel() == 0 or b.numel() == 0:
        raise ValueError(
            f"metric() requires equal non-empty shapes or both empty, got {tuple(a.shape)} vs {tuple(b.shape)}"
        )
    diff = (a - b).abs()
    return {
        "cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0).item()),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rmse": float(torch.sqrt(torch.mean((a - b) ** 2)).item()),
    }


def common_prefix_len(a: list[Any], b: list[Any]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def normalize_codec_rows(codec_codes: list[list[int]]) -> list[list[int]]:
    if not codec_codes:
        return []
    num_codebooks = len(codec_codes)
    num_steps = len(codec_codes[0])
    return [[codec_codes[cb][step] for cb in range(num_codebooks)] for step in range(num_steps)]


def find_base_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return max(s.getsockname()[1], 20000)


def extract_audio_waveform(chunk: dict[str, Any]) -> np.ndarray | None:
    raw = chunk.get("audio_waveform")
    if raw is None:
        return None
    if isinstance(raw, memoryview):
        raw = raw.tobytes()
    dtype = np.dtype(chunk.get("audio_waveform_dtype", "float32"))
    arr = np.frombuffer(raw, dtype=dtype)
    shape = chunk.get("audio_waveform_shape")
    if shape:
        arr = arr.reshape(shape)
    return arr.astype(np.float32, copy=True).reshape(-1)


def parse_env_overrides(values: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected KEY=VALUE env override, got {value!r}")
        key, raw = value.split("=", 1)
        env[key.strip()] = raw
    return env


def apply_env_overrides(overrides: dict[str, str]) -> None:
    for key, value in overrides.items():
        os.environ[key] = value


def copy_file_if_exists(src: str | Path, dst_dir: str | Path) -> str | None:
    src_path = Path(src)
    if not src_path.exists():
        return None
    dst_dir = ensure_dir(Path(dst_dir))
    dst = dst_dir / src_path.name
    shutil.copy2(src_path, dst)
    return str(dst)


def collect_runtime_artifacts(request_id: str, out_dir: str | Path, temp_dir: str | Path = "/tmp") -> dict[str, Any]:
    temp_dir = Path(temp_dir)
    out_dir = ensure_dir(Path(out_dir))
    artifacts: dict[str, Any] = {}

    single_files = {
        "talker_prefill": temp_dir / f"talker_prefill_{request_id}.pt",
        "talker_prefill_logits": temp_dir / f"talker_prefill_logits_{request_id}.pt",
        "code_predictor_debug": temp_dir / f"code_predictor_debug_{request_id}.pt",
        "code2wav_codes": temp_dir / f"code2wav_codes_{request_id}.pt",
    }
    for key, path in single_files.items():
        copied = copy_file_if_exists(path, out_dir)
        if copied is not None:
            artifacts[key] = copied

    multi_patterns = {
        "talker_feedback_input_steps": f"talker_feedback_input_{request_id}_step*.pt",
        "talker_layer_inputs_steps": f"talker_layer_inputs_{request_id}_step*.pt",
        "talker_decode_qk_steps": f"talker_decode_layer*_qk_{request_id}_step*.pt",
        "talker_decode_attn_steps": f"talker_decode_layer*_attn_{request_id}_step*.pt",
        "talker_decode_mlp_steps": f"talker_decode_layer*_mlp_{request_id}_step*.pt",
        "talker_trailing_events": f"talker_trailing_event_{request_id}_idx*.pt",
    }
    for key, pattern in multi_patterns.items():
        copied_paths: list[str] = []
        for path in sorted(temp_dir.glob(pattern)):
            copied = copy_file_if_exists(path, out_dir)
            if copied is not None:
                copied_paths.append(copied)
        if copied_paths:
            artifacts[key] = copied_paths

    return artifacts


def resolve_runtime_cp_path(runtime_capture: dict[str, Any], explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    artifacts = runtime_capture.get("artifacts", {})
    path = artifacts.get("code_predictor_debug")
    if not path:
        raise FileNotFoundError("Runtime capture does not reference code_predictor_debug")
    return Path(path)


def resolve_runtime_prefill_path(runtime_capture: dict[str, Any], explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    artifacts = runtime_capture.get("artifacts", {})
    path = artifacts.get("talker_prefill")
    if not path:
        raise FileNotFoundError("Runtime capture does not reference talker_prefill")
    return Path(path)


def resolve_hf_prefill_path(hf_capture: dict[str, Any], explicit: str | None = None) -> Path | None:
    if explicit:
        return Path(explicit)
    path = hf_capture.get("talker_prefill_path")
    return Path(path) if path else None


def load_hf_talker_model(
    model_path: str | Path,
    *,
    device: str = "cuda:1",
    dtype: torch.dtype = torch.bfloat16,
):
    from transformers import AutoConfig
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling

    from sglang_omni.models.weight_loader import load_module

    cfg = AutoConfig.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    model = hf_modeling.Qwen3OmniMoeTalkerForConditionalGeneration._from_config(
        cfg.talker_config
    )
    model = load_module(
        model,
        str(model_path),
        prefix="talker.",
        dtype=dtype,
        device=device,
        strict=False,
        local_files_only=True,
    )
    model.eval()
    return model


def load_hf_full_model(
    model_path: str | Path,
    *,
    device: str = "cuda:1",
    dtype: torch.dtype = torch.bfloat16,
):
    from transformers import AutoProcessor, Qwen3OmniMoeForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    return processor, model
