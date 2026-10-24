# SPDX-License-Identifier: Apache-2.0
"""Compare the native CosyVoice3 probe with an explicitly supplied pinned legacy model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models.cache import KVCache

REFERENCE_REVISION = "28c3e35a33bef2f45b3315431f4a512d6ed463cf"
REFERENCE_SHA256 = "ecaa3733c64d58245433effe9ba5355fed3e0bf2650e31a8e6fa81b84f1ebf91"


@dataclass(kw_only=True)
class Comparison:
    name: str
    shape: list[int]
    max_absolute_error: float
    max_relative_error: float
    reference_argmax: list[int]
    actual_argmax: list[int]
    passed: bool


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(
    name: str,
    reference: np.ndarray,
    actual: np.ndarray,
    absolute_tolerance: float,
    relative_tolerance: float,
    *,
    logits: bool,
) -> Comparison:
    if reference.shape != actual.shape:
        raise ValueError(f"{name}: shapes differ: {reference.shape} != {actual.shape}")
    else:
        pass
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError(f"{name}: non-finite values")
    else:
        pass
    difference = np.abs(reference.astype(np.float64) - actual.astype(np.float64))
    relative = difference / np.maximum(np.abs(reference), 1e-6)
    reference_argmax = (
        np.argmax(reference, axis=-1).reshape(-1).tolist() if logits else []
    )
    actual_argmax = np.argmax(actual, axis=-1).reshape(-1).tolist() if logits else []
    return Comparison(
        name=name,
        shape=list(reference.shape),
        max_absolute_error=float(difference.max()),
        max_relative_error=float(relative.max()),
        reference_argmax=reference_argmax,
        actual_argmax=actual_argmax,
        passed=bool(
            np.allclose(
                reference, actual, atol=absolute_tolerance, rtol=relative_tolerance
            )
            and reference_argmax == actual_argmax
        ),
    )


def run_probe(
    executable: Path,
    model_directory: Path,
    output_directory: Path,
    name: str,
    text_ids: list[int],
    speech_ids: list[int],
    decode_ids: list[int],
    timeout_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    logits_path = output_directory / f"{name}.logits.npy"
    embeddings_path = output_directory / f"{name}.embeddings.npy"
    command = [
        str(executable),
        "--model-path", str(model_directory),
        "--text-ids", json.dumps(text_ids),
        "--prompt-speech-ids", json.dumps(speech_ids),
        "--decode-ids", json.dumps(decode_ids),
        "--logits-output", str(logits_path),
        "--embeddings-output", str(embeddings_path),
    ]  # fmt: skip
    # Note (lgyStoic): Remove stale outputs so a successful no-op probe cannot pass.
    logits_path.unlink(missing_ok=True)
    embeddings_path.unlink(missing_ok=True)
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout_seconds
    )
    (output_directory / f"{name}.process.json").write_text(
        json.dumps(
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
            indent=2,
        )
        + "\n"
    )
    completed.check_returncode()
    summary = json.loads(completed.stdout)
    prompt_token_count = 2 + len(text_ids) + len(speech_ids)
    if (
        summary["cache_token_count"] != prompt_token_count + len(decode_ids)
        or summary["prompt_token_count"] != prompt_token_count
        or summary["decode_token_count"] != len(decode_ids)
        or summary["logits_shape"] != [1 + len(decode_ids), 6761]
    ):
        raise ValueError(f"{name}: incorrect probe counts: {summary}")
    else:
        return np.load(logits_path, allow_pickle=False), np.load(
            embeddings_path, allow_pickle=False
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-file",
        type=Path,
        required=True,
        help=f"Unmodified legacy model.py from {REFERENCE_REVISION}",
    )
    parser.add_argument(
        "--raw-model",
        type=Path,
        required=True,
        help="Official bundle containing llm.pt and CosyVoice-BlankEN/config.json",
    )
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Export checkpoint and Python reference; does not claim native parity",
    )
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--rtol", type=float, default=0.002)
    parser.add_argument("--timeout-seconds", type=float, default=180)
    arguments = parser.parse_args()
    if arguments.atol < 0 or arguments.rtol < 0 or arguments.timeout_seconds <= 0:
        parser.error("tolerances must be nonnegative and timeout must be positive")
    else:
        pass
    reference_file = arguments.reference_file.expanduser().resolve()
    raw_model = arguments.raw_model.expanduser().resolve()
    executable = arguments.probe.expanduser().resolve()
    output_directory = arguments.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / "parity-report.json"
    report_path.write_text(json.dumps({"status": "incomplete"}) + "\n")
    if sha256(reference_file) != REFERENCE_SHA256:
        raise ValueError(
            "Reference SHA256 does not match the pinned legacy implementation"
        )
    else:
        pass
    specification = importlib.util.spec_from_file_location(
        "cosyvoice3_parity_reference", reference_file
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot import {reference_file}")
    else:
        reference_module = importlib.util.module_from_spec(specification)
        sys.modules[specification.name] = reference_module
        specification.loader.exec_module(reference_module)

    model_arguments, backbone, speech_embedding, llm_decoder = (
        reference_module.load_raw_backbone(raw_model, dtype=mx.float16)
    )
    model = reference_module.CosyVoice3MlxModel(backbone, speech_embedding, llm_decoder)
    mx.eval(model.parameters())
    model_directory = output_directory / "converted"
    model_directory.mkdir(exist_ok=True)
    weights = {
        f"qwen2.{name}": tensor for name, tensor in tree_flatten(backbone.parameters())
    }
    weights["speech_embedding.weight"] = speech_embedding
    weights["llm_decoder.weight"] = llm_decoder
    mx.save_safetensors(str(model_directory / "model.safetensors"), weights)
    (model_directory / "config.json").write_text(
        json.dumps({"llm": asdict(model_arguments)}, indent=2) + "\n"
    )
    report = {
        "status": "prepared",
        "reference_revision": REFERENCE_REVISION,
        "reference_file": str(reference_file),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("mlx", "mlx-lm", "torch", "numpy")
        },
        "dtype": "float16",
        "tolerances": {
            "atol": arguments.atol,
            "rtol": arguments.rtol,
            "relative_error_denominator_floor": 1e-6,
        },
        "sha256": {
            str(path): sha256(path)
            for path in (
                reference_file,
                raw_model / "llm.pt",
                raw_model / "CosyVoice-BlankEN/config.json",
                model_directory / "config.json",
                model_directory / "model.safetensors",
            )
        },
        "comparisons": [],
        "invalid_inputs": [],
    }
    comparisons = []
    run_native = not arguments.prepare_only and executable.is_file()
    cases = [
        ("text", [151643, 9707, 11, 1879, 0], []),
        ("text_speech", [151643, 9707, 11, 1879, 0], [0, 17, 1234, 6560]),
    ]
    decode_ids = [42, 1024]
    report["cases"] = [
        {
            "name": name,
            "text_ids": text_ids,
            "prompt_speech_ids": speech_ids,
            "decode_ids": decode_ids,
        }
        for name, text_ids, speech_ids in cases
    ]
    for name, text_ids, speech_ids in cases:
        embeddings = model.build_prompt_embeddings(text_ids, speech_ids)
        cache = [KVCache() for _ in backbone.layers]
        cached_logits = [model.forward_embeddings(embeddings, cache=cache)]
        mx.eval(cached_logits[-1])
        full_logits = [model.forward_embeddings(embeddings)]
        prefix_embeddings = embeddings
        for token_id in decode_ids:
            token_ids = mx.array([[token_id]], dtype=mx.int32)
            cached_logits.append(model(token_ids, cache=cache))
            mx.eval(cached_logits[-1])
            prefix_embeddings = mx.concatenate(
                [prefix_embeddings, model.speech_embedding(token_ids)], axis=1
            )
            full_logits.append(model.forward_embeddings(prefix_embeddings))
            mx.eval(full_logits[-1])
        reference_logits = np.asarray(
            mx.concatenate(cached_logits, axis=1).astype(mx.float32)
        )[0]
        full_prefix_logits = np.asarray(
            mx.concatenate(full_logits, axis=1).astype(mx.float32)
        )[0]
        reference_embeddings = np.asarray(embeddings.astype(mx.float32))
        np.save(output_directory / f"{name}.reference.logits.npy", reference_logits)
        np.save(
            output_directory / f"{name}.reference.full-prefix.npy", full_prefix_logits
        )
        np.save(
            output_directory / f"{name}.reference.embeddings.npy", reference_embeddings
        )
        comparisons.append(
            compare(
                f"{name}.python_cached_vs_full",
                full_prefix_logits,
                reference_logits,
                arguments.atol,
                arguments.rtol,
                logits=True,
            )
        )
        if not run_native:
            continue
        else:
            pass
        native_logits, native_embeddings = run_probe(
            executable,
            model_directory,
            output_directory,
            name,
            text_ids,
            speech_ids,
            decode_ids,
            arguments.timeout_seconds,
        )
        comparisons.append(
            compare(
                f"{name}.embeddings",
                reference_embeddings,
                native_embeddings,
                0.0,
                0.0,
                logits=False,
            )
        )
        for index in range(len(decode_ids) + 1):
            comparisons.append(
                compare(
                    f"{name}.logits.{index}",
                    reference_logits[index : index + 1],
                    native_logits[index : index + 1],
                    arguments.atol,
                    arguments.rtol,
                    logits=True,
                )
            )
        for count in range(1, len(decode_ids) + 1):
            prefix_logits, _ = run_probe(
                executable,
                model_directory,
                output_directory,
                f"{name}.full-prefix.{count}",
                text_ids,
                speech_ids + decode_ids[:count],
                [],
                arguments.timeout_seconds,
            )
            comparisons.append(
                compare(
                    f"{name}.native_cached_vs_full.{count}",
                    prefix_logits,
                    native_logits[count : count + 1],
                    arguments.atol,
                    arguments.rtol,
                    logits=True,
                )
            )
    report["comparisons"] = [asdict(comparison) for comparison in comparisons]
    if not run_native:
        report["status"] = (
            "prepared" if arguments.prepare_only else "blocked_missing_probe"
        )
        exit_code = (
            0
            if arguments.prepare_only and all(result.passed for result in comparisons)
            else 2
        )
    else:
        report["sha256"][str(executable)] = sha256(executable)
        invalid_cases = [
            ("--text-ids", "[-1]", "integers in"),
            ("--text-ids", "[151936]", "integers in"),
            ("--text-ids", "[true]", "integers in"),
            ("--text-ids", "{}", "JSON array"),
            ("--prompt-speech-ids", "[6561]", "prompt speech ids"),
            ("--decode-ids", "[6761]", "decode ids"),
        ]
        for index, (option, value, expected_error) in enumerate(invalid_cases):
            invalid_output = output_directory / f"invalid-{index}.npy"
            invalid_output.unlink(missing_ok=True)
            completed = subprocess.run(
                [
                    str(executable),
                    "--model-path",
                    str(model_directory),
                    "--logits-output",
                    str(invalid_output),
                    option,
                    value,
                ],
                capture_output=True,
                text=True,
                timeout=arguments.timeout_seconds,
            )
            report["invalid_inputs"].append(
                {
                    "option": option,
                    "value": value,
                    "returncode": completed.returncode,
                    "stderr": completed.stderr,
                    "passed": completed.returncode > 0
                    and expected_error in completed.stderr
                    and not invalid_output.exists(),
                }
            )
        passed = all(result.passed for result in comparisons) and all(
            result["passed"] for result in report["invalid_inputs"]
        )
        report["status"] = "passed" if passed else "failed"
        exit_code = 0 if passed else 1
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
else:
    pass
