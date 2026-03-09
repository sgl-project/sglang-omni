# SPDX-License-Identifier: Apache-2.0
"""Compare runtime Talker layer-boundary dumps against exact-runtime-input HF replay."""

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
    DEFAULT_MODEL_PATH,
    FLOAT_PASS_THRESHOLD,
    load_hf_talker_model,
    load_json,
    metric,
    resolve_runtime_cp_path,
    resolve_runtime_prefill_path,
    save_json,
)


def load_feedback_step_dumps(
    runtime_capture: dict[str, Any] | None,
    request_id: str,
    artifacts_dir: Path,
) -> dict[int, dict[str, Any]]:
    step_dumps: dict[int, dict[str, Any]] = {}
    if runtime_capture is not None:
        for path_str in runtime_capture.get("artifacts", {}).get("talker_feedback_input_steps", []):
            path = Path(path_str)
            step = int(path.stem.rsplit("step", 1)[-1])
            step_dumps[step] = torch.load(path, map_location="cpu")
        if step_dumps:
            return step_dumps
    for path in sorted(artifacts_dir.glob(f"talker_feedback_input_{request_id}_step*.pt")):
        step = int(path.stem.rsplit("step", 1)[-1])
        step_dumps[step] = torch.load(path, map_location="cpu")
    return step_dumps


def load_layer_input_dump(path: Path) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu")
    for key in [
        "layer_inputs",
        "decoder_hidden_states",
        "decoder_residuals",
        "decoder_effective_inputs",
        "decoder_output_hidden_states",
        "decoder_output_residuals",
        "decoder_output_effective_inputs",
        "mlp_inputs",
        "mlp_outputs",
    ]:
        value = data.get(key)
        if isinstance(value, dict):
            normalized = {}
            for raw_key, tensor in value.items():
                try:
                    normalized[int(raw_key)] = tensor
                except Exception:
                    continue
            data[key] = normalized
    return data


def capture_hf_layer_boundaries(
    model: Any,
    *,
    runtime_prefill_dump: dict[str, Any],
    runtime_cp_dump: dict[str, Any],
    feedback_step_dumps: dict[int, dict[str, Any]],
    target_step: int,
    capture_layers: list[int],
    device: str,
) -> dict[str, dict[int, torch.Tensor]]:
    captures: dict[str, dict[int, torch.Tensor]] = {
        "decoder_effective_inputs": {},
        "layer_inputs": {},
        "mlp_inputs": {},
        "mlp_outputs": {},
        "decoder_output_effective_inputs": {},
    }

    handles = []

    def _pick_hidden(args, kwargs) -> torch.Tensor | None:
        hidden_states = None
        if kwargs:
            hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        return hidden_states if isinstance(hidden_states, torch.Tensor) else None

    for layer_idx in capture_layers:
        if layer_idx < 0 or layer_idx >= len(model.model.layers):
            continue
        layer = model.model.layers[layer_idx]

        def _layer_pre(module, args, kwargs, *, layer_idx=layer_idx):
            hidden_states = _pick_hidden(args, kwargs)
            if hidden_states is not None:
                captures["decoder_effective_inputs"][layer_idx] = (
                    hidden_states[0, -1].detach().cpu()
                )

        def _attn_pre(module, args, kwargs, *, layer_idx=layer_idx):
            hidden_states = _pick_hidden(args, kwargs)
            if hidden_states is not None:
                captures["layer_inputs"][layer_idx] = (
                    hidden_states[0, -1].detach().cpu()
                )

        def _mlp_pre(module, args, kwargs, *, layer_idx=layer_idx):
            hidden_states = _pick_hidden(args, kwargs)
            if hidden_states is not None:
                captures["mlp_inputs"][layer_idx] = hidden_states[0, -1].detach().cpu()

        def _mlp_post(module, args, kwargs, output, *, layer_idx=layer_idx):
            hidden_states = output[0] if isinstance(output, tuple) else output
            if isinstance(hidden_states, torch.Tensor):
                captures["mlp_outputs"][layer_idx] = hidden_states[0, -1].detach().cpu()

        def _layer_post(module, args, kwargs, output, *, layer_idx=layer_idx):
            hidden_states = output[0] if isinstance(output, tuple) else output
            if isinstance(hidden_states, torch.Tensor):
                captures["decoder_output_effective_inputs"][layer_idx] = (
                    hidden_states[0, -1].detach().cpu()
                )

        handles.append(layer.register_forward_pre_hook(_layer_pre, with_kwargs=True))
        handles.append(layer.self_attn.register_forward_pre_hook(_attn_pre, with_kwargs=True))
        handles.append(layer.mlp.register_forward_pre_hook(_mlp_pre, with_kwargs=True))
        handles.append(layer.mlp.register_forward_hook(_mlp_post, with_kwargs=True))
        handles.append(layer.register_forward_hook(_layer_post, with_kwargs=True))

    input_embeds = runtime_prefill_dump["input_embeds"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    input_ids = runtime_prefill_dump["input_ids"].unsqueeze(0).to(device=device)
    trailing = runtime_prefill_dump.get("trailing_text_hidden")
    if isinstance(trailing, torch.Tensor):
        trailing = trailing.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    tts_pad_embed = runtime_prefill_dump["tts_pad_embed"].to(device=device, dtype=torch.bfloat16)
    attention_mask = torch.ones((1, input_embeds.shape[1]), device=device, dtype=torch.long)
    runtime_tokens = runtime_cp_dump["layer0_codes"].tolist()

    with torch.no_grad():
        outputs = model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            talker_input_ids=input_ids,
            trailing_text_hidden=trailing,
            tts_pad_embed=tts_pad_embed,
        )
        past_key_values = outputs.past_key_values
        generation_step = getattr(outputs, "generation_step", 0)

        for step in range(1, target_step + 1):
            step_dump = feedback_step_dumps[step]
            token_ids = torch.tensor([[runtime_tokens[step - 1]]], device=device, dtype=torch.long)
            feedback_input = step_dump["combined_feedback_input_embeds"].to(
                device=device,
                dtype=torch.bfloat16,
            ).view(1, 1, -1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=device, dtype=torch.long)],
                dim=1,
            )
            cache_position = torch.tensor(
                [input_embeds.shape[1] + step - 1],
                device=device,
                dtype=torch.long,
            )
            outputs = model(
                input_ids=token_ids,
                inputs_embeds=feedback_input,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                generation_step=generation_step,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            generation_step = getattr(outputs, "generation_step", generation_step + 1)
            if step == target_step:
                break

    for handle in handles:
        handle.remove()

    return captures


def analyze_layer_boundary(
    *,
    runtime_capture: dict[str, Any] | None,
    request_id: str,
    runtime_prefill_path: Path,
    runtime_cp_path: Path,
    layer_dump_path: Path,
    artifacts_dir: Path,
    model_path: str | Path,
    device: str,
) -> dict[str, Any]:
    layer_dump = load_layer_input_dump(layer_dump_path)
    capture_layers = sorted(layer_dump.get("layer_inputs", {}).keys())
    target_step = int(layer_dump.get("generation_steps", 0))
    feedback_step_dumps = load_feedback_step_dumps(runtime_capture, request_id, artifacts_dir)
    if target_step not in feedback_step_dumps:
        raise FileNotFoundError(
            f"Missing feedback dump for step {target_step} under request {request_id}"
        )

    model = load_hf_talker_model(model_path, device=device, dtype=torch.bfloat16)
    hf_captures = capture_hf_layer_boundaries(
        model,
        runtime_prefill_dump=torch.load(runtime_prefill_path, map_location="cpu"),
        runtime_cp_dump=torch.load(runtime_cp_path, map_location="cpu"),
        feedback_step_dumps=feedback_step_dumps,
        target_step=target_step,
        capture_layers=capture_layers,
        device=device,
    )

    sections = [
        ("decoder_effective_inputs", "decoder_effective_inputs"),
        ("layer_inputs", "layer_inputs"),
        ("mlp_inputs", "mlp_inputs"),
        ("mlp_outputs", "mlp_outputs"),
        ("decoder_output_effective_inputs", "decoder_output_effective_inputs"),
    ]
    comparisons: dict[str, list[dict[str, Any]]] = {}
    first_failure = None
    for runtime_key, hf_key in sections:
        rows = []
        runtime_section = layer_dump.get(runtime_key, {})
        hf_section = hf_captures.get(hf_key, {})
        for layer_idx in capture_layers:
            if layer_idx not in runtime_section or layer_idx not in hf_section:
                continue
            row = {"layer": layer_idx, **metric(runtime_section[layer_idx], hf_section[layer_idx])}
            rows.append(row)
            if first_failure is None and row["cosine"] < FLOAT_PASS_THRESHOLD:
                first_failure = {"boundary": runtime_key, "layer": layer_idx, **row}
        comparisons[runtime_key] = rows

    return {
        "request_id": request_id,
        "generation_steps": target_step,
        "capture_layers": capture_layers,
        "input_token": layer_dump.get("input_token"),
        "comparisons": comparisons,
        "first_failure": first_failure,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-capture", type=Path, default=None)
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--runtime-prefill", type=Path, default=None)
    parser.add_argument("--runtime-cp-dump", type=Path, default=None)
    parser.add_argument("--layer-dump", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    runtime_capture = None
    if args.runtime_capture:
        runtime_capture = load_json(args.runtime_capture)
    if runtime_capture is not None:
        request_id = runtime_capture["request_id"]
        runtime_cp_path = resolve_runtime_cp_path(
            runtime_capture,
            str(args.runtime_cp_dump) if args.runtime_cp_dump else None,
        )
        runtime_prefill_path = resolve_runtime_prefill_path(
            runtime_capture,
            str(args.runtime_prefill) if args.runtime_prefill else None,
        )
    else:
        if args.request_id is None or args.runtime_prefill is None or args.runtime_cp_dump is None:
            raise ValueError(
                "Either --runtime-capture or all of --request-id/--runtime-prefill/--runtime-cp-dump are required"
            )
        request_id = args.request_id
        runtime_cp_path = args.runtime_cp_dump
        runtime_prefill_path = args.runtime_prefill

    result = analyze_layer_boundary(
        runtime_capture=runtime_capture,
        request_id=request_id,
        runtime_prefill_path=runtime_prefill_path,
        runtime_cp_path=runtime_cp_path,
        layer_dump_path=args.layer_dump,
        artifacts_dir=args.artifacts_dir,
        model_path=args.model_path,
        device=args.device,
    )
    if args.out:
        save_json(result, args.out)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
