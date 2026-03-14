# SPDX-License-Identifier: Apache-2.0
"""Top-down parity matrix for one runtime capture vs one HF capture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.qwen3_omni_parity.common import (
    FLOAT_PASS_THRESHOLD,
    common_prefix_len,
    load_json,
    metric,
    normalize_codec_rows,
    resolve_hf_prefill_path,
    resolve_runtime_cp_path,
    resolve_runtime_prefill_path,
    save_json,
)


def _float_contract_status(metrics: dict[str, dict[str, float]]) -> str:
    if not metrics:
        return "unknown"
    if all(item["cosine"] >= FLOAT_PASS_THRESHOLD for item in metrics.values()):
        return "pass"
    return "fail"


def _tensor_metric_or_mismatch(
    runtime_tensor: torch.Tensor | None,
    hf_tensor: torch.Tensor | None,
) -> tuple[dict[str, float] | None, dict[str, Any] | None]:
    if runtime_tensor is None and hf_tensor is None:
        return None, {"status": "both_missing"}
    if runtime_tensor is None or hf_tensor is None:
        return None, {
            "status": "missing",
            "runtime_shape": (
                list(runtime_tensor.shape)
                if isinstance(runtime_tensor, torch.Tensor)
                else None
            ),
            "hf_shape": (
                list(hf_tensor.shape) if isinstance(hf_tensor, torch.Tensor) else None
            ),
        }
    if tuple(runtime_tensor.shape) != tuple(hf_tensor.shape):
        return None, {
            "status": "shape_mismatch",
            "runtime_shape": list(runtime_tensor.shape),
            "hf_shape": list(hf_tensor.shape),
        }
    return metric(runtime_tensor, hf_tensor), None


def build_topdown_matrix(
    runtime_capture: dict[str, Any],
    hf_capture: dict[str, Any],
    *,
    runtime_cp_dump: dict[str, Any] | None = None,
    runtime_prefill_dump: dict[str, Any] | None = None,
    hf_prefill_dump: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "canonical_runtime_rid": runtime_capture.get("request_id"),
        "contracts": {},
        "first_failing_contract": None,
    }

    runtime_ids = runtime_capture.get("thinker_generated_ids")
    hf_ids = hf_capture.get("generated_ids", [])
    thinker_text_checks = {
        "text_match": runtime_capture.get("runtime_delta_text")
        == hf_capture.get("generated_text"),
        "final_text_match": runtime_capture.get("runtime_final_text_from_event")
        == hf_capture.get("generated_text"),
        "decode_complete_match": runtime_capture.get("runtime_decode_complete_text")
        == hf_capture.get("generated_text"),
    }
    if runtime_ids is None:
        thinker_status = "unknown" if all(thinker_text_checks.values()) else "fail"
        token_ids_match = None
        runtime_num_tokens = None
    else:
        token_ids_match = runtime_ids == hf_ids
        runtime_num_tokens = len(runtime_ids)
        thinker_status = (
            "pass" if token_ids_match and all(thinker_text_checks.values()) else "fail"
        )
    result["contracts"]["thinker_text"] = {
        "status": thinker_status,
        "details": {
            "runtime_num_tokens": runtime_num_tokens,
            "hf_num_tokens": len(hf_ids),
            "token_ids_match": token_ids_match,
            **thinker_text_checks,
        },
    }

    if runtime_prefill_dump is not None and hf_prefill_dump is not None:
        prefill_metrics = {
            "input_embeds": metric(
                runtime_prefill_dump["input_embeds"], hf_prefill_dump["input_embeds"]
            ),
            "tts_pad_embed": metric(
                runtime_prefill_dump["tts_pad_embed"], hf_prefill_dump["tts_pad_embed"]
            ),
        }
        runtime_trailing = runtime_prefill_dump.get("trailing_text_hidden")
        hf_trailing = hf_prefill_dump.get("trailing_text_hidden")
        trailing_metric, trailing_mismatch = _tensor_metric_or_mismatch(
            runtime_trailing, hf_trailing
        )
        if trailing_metric is not None:
            prefill_metrics["trailing_text_hidden"] = trailing_metric
        ids_match = torch.equal(
            runtime_prefill_dump["input_ids"], hf_prefill_dump["input_ids"]
        )
        prefill_status = (
            "pass"
            if ids_match
            and trailing_mismatch is None
            and _float_contract_status(prefill_metrics) == "pass"
            else "fail"
        )
        result["contracts"]["talker_prefill"] = {
            "status": prefill_status,
            "details": {
                "input_ids_match": bool(ids_match),
                **prefill_metrics,
                "trailing_text_hidden": trailing_mismatch
                or prefill_metrics.get("trailing_text_hidden"),
            },
        }
    else:
        result["contracts"]["talker_prefill"] = {
            "status": "unknown",
            "details": {"reason": "missing runtime or hf talker prefill dump"},
        }

    if runtime_cp_dump is not None:
        runtime_layer0 = runtime_cp_dump["layer0_codes"].tolist()
        runtime_rows = runtime_cp_dump["output_codes"].tolist()
        hf_layer0 = hf_capture.get("layer0_codes", [])
        hf_rows = normalize_codec_rows(hf_capture.get("codec_codes", []))

        layer0_prefix = common_prefix_len(runtime_layer0, hf_layer0)
        row_prefix = common_prefix_len(runtime_rows, hf_rows)

        result["contracts"]["code_predictor_full_code_rows"] = {
            "status": (
                "pass" if row_prefix == min(len(runtime_rows), len(hf_rows)) else "fail"
            ),
            "details": {
                "runtime_steps": len(runtime_rows),
                "hf_steps": len(hf_rows),
                "common_prefix_len": row_prefix,
                "runtime_step0": runtime_rows[0] if runtime_rows else None,
                "hf_step0": hf_rows[0] if hf_rows else None,
                "runtime_step1": runtime_rows[1] if len(runtime_rows) > 1 else None,
                "hf_step1": hf_rows[1] if len(hf_rows) > 1 else None,
            },
        }
        result["contracts"]["talker_layer0_sequence"] = {
            "status": (
                "pass"
                if layer0_prefix == min(len(runtime_layer0), len(hf_layer0))
                else "fail"
            ),
            "details": {
                "runtime_len": len(runtime_layer0),
                "hf_len": len(hf_layer0),
                "common_prefix_len": layer0_prefix,
                "first_runtime_tokens": runtime_layer0[:16],
                "first_hf_tokens": hf_layer0[:16],
                "first_runtime_mismatch_token": (
                    runtime_layer0[layer0_prefix]
                    if layer0_prefix < len(runtime_layer0)
                    else None
                ),
                "first_hf_mismatch_token": (
                    hf_layer0[layer0_prefix] if layer0_prefix < len(hf_layer0) else None
                ),
            },
        }
    else:
        result["contracts"]["code_predictor_full_code_rows"] = {
            "status": "unknown",
            "details": {"reason": "missing runtime code predictor dump"},
        }
        result["contracts"]["talker_layer0_sequence"] = {
            "status": "unknown",
            "details": {"reason": "missing runtime code predictor dump"},
        }

    same_prompt = runtime_capture.get("prompt") == hf_capture.get("prompt")
    text_match = runtime_capture.get("runtime_decode_complete_text") == hf_capture.get(
        "generated_text"
    )
    audio_match = (
        runtime_capture.get("audio_sha256")
        and hf_capture.get("audio_sha256")
        and runtime_capture.get("audio_sha256") == hf_capture.get("audio_sha256")
    )
    result["contracts"]["e2e_audio"] = {
        "status": "pass" if same_prompt and text_match and audio_match else "fail",
        "details": {
            "same_prompt": same_prompt,
            "text_match": text_match,
            "runtime_duration_sec": runtime_capture.get("duration_sec"),
            "hf_duration_sec": hf_capture.get("wav_duration_sec"),
            "duration_ratio": (
                float(runtime_capture.get("duration_sec"))
                / float(hf_capture.get("wav_duration_sec"))
                if runtime_capture.get("duration_sec")
                and hf_capture.get("wav_duration_sec")
                else None
            ),
            "audio_sha_match": bool(audio_match),
        },
    }

    for name in [
        "thinker_text",
        "talker_prefill",
        "code_predictor_full_code_rows",
        "talker_layer0_sequence",
        "e2e_audio",
    ]:
        contract = result["contracts"].get(name)
        if contract and contract["status"] == "fail":
            result["first_failing_contract"] = name
            break

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-capture", type=Path, required=True)
    parser.add_argument("--hf-capture", type=Path, required=True)
    parser.add_argument("--runtime-cp-dump", type=Path, default=None)
    parser.add_argument("--runtime-prefill", type=Path, default=None)
    parser.add_argument("--hf-prefill", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    runtime_capture = load_json(args.runtime_capture)
    hf_capture = load_json(args.hf_capture)
    runtime_cp_path = resolve_runtime_cp_path(
        runtime_capture, str(args.runtime_cp_dump) if args.runtime_cp_dump else None
    )
    try:
        runtime_prefill_path = resolve_runtime_prefill_path(
            runtime_capture,
            str(args.runtime_prefill) if args.runtime_prefill else None,
        )
    except FileNotFoundError:
        runtime_prefill_path = None
    hf_prefill_path = resolve_hf_prefill_path(
        hf_capture, str(args.hf_prefill) if args.hf_prefill else None
    )

    result = build_topdown_matrix(
        runtime_capture,
        hf_capture,
        runtime_cp_dump=torch.load(runtime_cp_path, map_location="cpu"),
        runtime_prefill_dump=(
            torch.load(runtime_prefill_path, map_location="cpu")
            if runtime_prefill_path and runtime_prefill_path.exists()
            else None
        ),
        hf_prefill_dump=(
            torch.load(hf_prefill_path, map_location="cpu")
            if hf_prefill_path and hf_prefill_path.exists()
            else None
        ),
    )
    if args.out:
        save_json(result, args.out)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
