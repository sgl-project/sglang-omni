# SPDX-License-Identifier: Apache-2.0
"""Load pinned external scoring code without implicit model downloads."""

from __future__ import annotations

import ast
import json
import sys
import types
import typing
from collections import Counter
from pathlib import Path
from typing import Protocol

from pydantic import JsonValue

from benchmarks.duplex.reference_core import (
    JUDGE_MODEL,
    REFERENCE_FILES,
    REFERENCE_REVISION,
    load_module,
    sha256_file,
)


class AudioTensor(Protocol):
    def squeeze(self, dim: int) -> AudioTensor: ...


class SileroModel(Protocol):
    def reset_states(self) -> None: ...

    def __call__(self, waveform: AudioTensor, sample_rate: int) -> AudioTensor: ...


class SileroLoader(Protocol):
    def __call__(self) -> SileroModel: ...


class WaveformLoader(Protocol):
    def __call__(self, path: Path) -> AudioTensor: ...


class ReferenceBehavior(Protocol):
    instruction: str
    model: str
    initial_seed: int

    def template(
        self,
        input_clean_text: str,
        input_noisy_text: str,
        output_clean_text: str,
        output_noisy_text: str,
    ) -> str: ...

    def json_dict_to_compact_text(self, transcript: JsonValue) -> str: ...

    def extract_json(self, text: str, key: str = "behaviour") -> JsonValue: ...

    def parse_eval(self, prediction: JsonValue) -> dict[str, JsonValue]: ...

    def stats_by_axis(
        self, records: list[dict[str, JsonValue]]
    ) -> tuple[
        dict[str, Counter[str]], dict[str, int], dict[str, dict[str, float]]
    ]: ...


def verify_reference(source: Path) -> dict[str, Path]:
    """Refuse any checkout whose used files differ from the pinned revision."""
    paths = {}
    for key, (relative_path, expected) in REFERENCE_FILES.items():
        path = source / relative_path
        actual = sha256_file(path)
        if actual != expected:
            raise SystemExit(
                f"{relative_path} sha256 {actual} != pinned {expected} ({REFERENCE_REVISION})"
            )
        else:
            pass
        paths[key] = path
    return paths


def load_official_timing(
    path: Path, silero_loader: SileroLoader | None = None
) -> tuple[types.ModuleType, dict[str, JsonValue]]:
    """Import get_timing.py with its unpinned torch.hub.load bound to packaged Silero.

    Returns (module, bridge_record). Formulas and constants are the file's own.
    """
    import torch

    if silero_loader is None:

        def silero_loader() -> SileroModel:
            from silero_vad import load_silero_vad

            return load_silero_vad(onnx=False)

    else:
        pass

    calls = []

    def hub_load(
        repo_or_dir: str, model: str, *args: JsonValue, **kwargs: JsonValue
    ) -> tuple[SileroModel, None]:
        call = {
            "repo_or_dir": repo_or_dir,
            "model": model,
            "args": list(args),
            "kwargs": kwargs,
        }
        if call != {
            "repo_or_dir": "snakers4/silero-vad",
            "model": "silero_vad",
            "args": [],
            "kwargs": {"trust_repo": True, "onnx": False},
        }:
            raise RuntimeError(f"unexpected torch.hub.load call {call}")
        else:
            pass
        calls.append(call)
        return silero_loader(), None

    original = torch.hub.load
    torch.hub.load = hub_load
    try:
        module = load_module(path, "fdb_v15_get_timing_3e799c4")
    finally:
        torch.hub.load = original
    silero = sys.modules.get("silero_vad")
    record = {
        "vad_branch": (
            "VoiceActivityDetector"
            if hasattr(module, "_VAD")
            else "get_speech_timestamps"
        ),
        "torch_hub_load_calls": calls,
        "silero_bridge": "torch.hub.load('snakers4/silero-vad', ...) -> silero_vad.load_silero_vad(onnx=False)",
        "silero_module_file": getattr(silero, "__file__", None),
        "silero_jit_sha256": silero_jit_hash(silero),
        "constants": {
            constant_name: getattr(module, constant_name)
            for constant_name in (
                "SR",
                "USER_MERGE_GAP",
                "MODEL_MERGE_GAP",
                "OUT_FILENAME",
            )
        },
    }
    return module, record


def silero_jit_hash(silero: types.ModuleType | None) -> str | None:
    if silero is None or not getattr(silero, "__file__", None):
        return None
    else:
        pass
    model_path = Path(silero.__file__).parent / "data" / "silero_vad.jit"
    return sha256_file(model_path) if model_path.exists() else None


def soundfile_load_wav(sr_target: int) -> WaveformLoader:
    """Bridge for torchaudio.load without torchcodec: same float32 [C,T] then official resample/squeeze."""
    import soundfile
    import torch
    import torchaudio

    def load_wav(path: Path) -> AudioTensor:
        waveform, sample_rate = soundfile.read(
            str(path), dtype="float32", always_2d=True
        )
        audio_tensor = torch.from_numpy(waveform.T.copy())
        if sample_rate != sr_target:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, sample_rate, sr_target
            )
        else:
            pass
        return audio_tensor.squeeze(0)

    return load_wav


def load_official_behavior(path: Path, instruction_path: Path) -> ReferenceBehavior:
    # Note (wenyao): Importing the reference module would initialize an unused OpenAI client.
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    names = (
        "json_dict_to_compact_text",
        "extract_json",
        "parse_eval",
        "stats_by_axis",
    )
    namespace = {
        "json": json,
        "Counter": Counter,
        "Dict": typing.Dict,
        "Any": typing.Any,
        "Union": typing.Union,
        "List": typing.List,
    }
    exec(
        compile(
            ast.Module(body=[functions[node] for node in names], type_ignores=[]),
            str(path),
            "exec",
        ),
        namespace,
    )

    final_input = [
        node
        for node in ast.walk(functions["eval_behavior_all"])
        if isinstance(node, ast.Assign)
        and [getattr(target, "id", None) for target in node.targets] == ["final_input"]
    ]
    if len(final_input) != 1 or not isinstance(final_input[0].value, ast.JoinedStr):
        raise RuntimeError("eval_behavior_all final_input f-string not found")
    else:
        pass
    fields = (
        "input_clean_text",
        "input_noisy_text",
        "output_clean_text",
        "output_noisy_text",
    )
    template_expression = ast.Expression(
        ast.Lambda(
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=field_name) for field_name in fields],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=final_input[0].value,
        )
    )
    ast.fix_missing_locations(template_expression)
    template = eval(compile(template_expression, str(path), "eval"), {})

    with open(instruction_path, "r", encoding="utf-8") as file_handle:
        instruction = file_handle.read()
    return types.SimpleNamespace(
        template=template,
        instruction=instruction,
        model=JUDGE_MODEL,
        initial_seed=1,
        **{node: namespace[node] for node in names},
    )
