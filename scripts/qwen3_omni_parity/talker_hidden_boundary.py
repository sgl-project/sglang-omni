# SPDX-License-Identifier: Apache-2.0
"""Compare runtime Talker hidden states against exact-runtime-input HF replay."""

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
    DEFAULT_SEED,
    FLOAT_PASS_THRESHOLD,
    load_hf_talker_model,
    metric,
    resolve_runtime_cp_path,
    resolve_runtime_prefill_path,
    save_json,
)


def load_feedback_step_dumps(
    runtime_capture: dict[str, Any] | None, request_id: str, artifacts_dir: Path
) -> dict[int, dict[str, Any]]:
    step_dumps: dict[int, dict[str, Any]] = {}
    if runtime_capture is not None:
        for path_str in runtime_capture.get("artifacts", {}).get(
            "talker_feedback_input_steps", []
        ):
            path = Path(path_str)
            step = int(path.stem.rsplit("step", 1)[-1])
            step_dumps[step] = torch.load(path, map_location="cpu")
        if step_dumps:
            return step_dumps
    for path in sorted(
        artifacts_dir.glob(f"talker_feedback_input_{request_id}_step*.pt")
    ):
        step = int(path.stem.rsplit("step", 1)[-1])
        step_dumps[step] = torch.load(path, map_location="cpu")
    return step_dumps


def analyze_talker_hidden_boundary(
    runtime_prefill_dump: dict[str, Any],
    runtime_cp_dump: dict[str, Any],
    feedback_step_dumps: dict[int, dict[str, Any]],
    *,
    model_path: str | Path,
    device: str,
    steps: int,
) -> dict[str, Any]:
    model = load_hf_talker_model(model_path, device=device, dtype=torch.bfloat16)
    input_embeds = (
        runtime_prefill_dump["input_embeds"]
        .unsqueeze(0)
        .to(device=device, dtype=torch.bfloat16)
    )
    input_ids = runtime_prefill_dump["input_ids"].unsqueeze(0).to(device=device)
    trailing = runtime_prefill_dump.get("trailing_text_hidden")
    if isinstance(trailing, torch.Tensor):
        trailing = trailing.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    tts_pad_embed = runtime_prefill_dump["tts_pad_embed"].to(
        device=device, dtype=torch.bfloat16
    )
    attention_mask = torch.ones(
        (1, input_embeds.shape[1]), device=device, dtype=torch.long
    )
    runtime_hidden = runtime_cp_dump["talker_hidden"]
    runtime_tokens = runtime_cp_dump["layer0_codes"].tolist()

    step_metrics = []
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
        prefill_hidden = outputs.hidden_states[0][-1][:, -1].detach().cpu()[0]
        step_metrics.append(
            {
                "step": 0,
                "token": runtime_tokens[0] if runtime_tokens else None,
                **metric(runtime_hidden[0], prefill_hidden),
            }
        )

        past_key_values = outputs.past_key_values
        generation_step = getattr(outputs, "generation_step", 0)
        for step in range(1, min(steps + 1, runtime_hidden.shape[0])):
            step_dump = feedback_step_dumps.get(step)
            if not step_dump or not isinstance(
                step_dump.get("combined_feedback_input_embeds"), torch.Tensor
            ):
                break
            token_ids = torch.tensor(
                [[runtime_tokens[step - 1]]], device=device, dtype=torch.long
            )
            feedback_input = (
                step_dump["combined_feedback_input_embeds"]
                .to(device=device, dtype=torch.bfloat16)
                .view(1, 1, -1)
            )
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=device, dtype=torch.long)],
                dim=1,
            )
            cache_position = torch.tensor(
                [input_embeds.shape[1] + step - 1], device=device, dtype=torch.long
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
            current_hidden = outputs.hidden_states[0][-1][:, -1].detach().cpu()[0]
            step_metrics.append(
                {
                    "step": step,
                    "token": runtime_tokens[step],
                    **metric(runtime_hidden[step], current_hidden),
                }
            )
            past_key_values = outputs.past_key_values
            generation_step = getattr(outputs, "generation_step", generation_step + 1)

    first_failing_step = next(
        (
            item["step"]
            for item in step_metrics
            if item["cosine"] < FLOAT_PASS_THRESHOLD
        ),
        None,
    )
    return {
        "steps_checked": len(step_metrics),
        "step_metrics": step_metrics,
        "first_failing_step": first_failing_step,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-capture", type=Path, default=None)
    parser.add_argument("--runtime-prefill", type=Path, default=None)
    parser.add_argument("--runtime-cp-dump", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    del args.seed

    runtime_capture = None
    if args.runtime_capture:
        runtime_capture = json.loads(args.runtime_capture.read_text())
    if runtime_capture is not None:
        request_id = runtime_capture["request_id"]
        runtime_cp_path = resolve_runtime_cp_path(
            runtime_capture, str(args.runtime_cp_dump) if args.runtime_cp_dump else None
        )
        runtime_prefill_path = resolve_runtime_prefill_path(
            runtime_capture, str(args.runtime_prefill) if args.runtime_prefill else None
        )
    else:
        if (
            args.request_id is None
            or args.runtime_prefill is None
            or args.runtime_cp_dump is None
        ):
            raise ValueError(
                "Either --runtime-capture or all of --request-id/--runtime-prefill/--runtime-cp-dump are required"
            )
        request_id = args.request_id
        runtime_cp_path = args.runtime_cp_dump
        runtime_prefill_path = args.runtime_prefill

    feedback_step_dumps = load_feedback_step_dumps(
        runtime_capture, request_id, args.artifacts_dir
    )
    result = analyze_talker_hidden_boundary(
        torch.load(runtime_prefill_path, map_location="cpu"),
        torch.load(runtime_cp_path, map_location="cpu"),
        feedback_step_dumps,
        model_path=args.model_path,
        device=args.device,
        steps=args.steps,
    )
    if args.out:
        save_json(result, args.out)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
