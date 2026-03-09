# SPDX-License-Identifier: Apache-2.0
"""Recursive predictor-boundary split for runtime vs HF parity."""

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
    load_hf_talker_model,
    load_json,
    metric,
    save_json,
)


def replay_predictor_rows(
    model: Any,
    hidden_rows: list[torch.Tensor],
    layer0_codes: list[int],
    *,
    device: str,
    dtype: torch.dtype,
    seed: int,
    mode: str,
) -> list[list[int]]:
    rows: list[list[int]] = []
    if mode == "sequential":
        torch.manual_seed(seed)
    for idx, (hidden_row, layer0_code) in enumerate(zip(hidden_rows, layer0_codes, strict=False)):
        if mode == "fresh":
            torch.manual_seed(seed)
        hidden = hidden_row.to(device=device, dtype=dtype).view(1, 1, -1)
        code = torch.tensor([[layer0_code]], device=device, dtype=torch.long)
        layer0_embed = model.get_input_embeddings()(code)
        out = model.code_predictor.generate(
            inputs_embeds=torch.cat((hidden, layer0_embed), dim=1),
            max_new_tokens=model.config.num_code_groups - 1,
            do_sample=True,
            top_k=50,
            top_p=0.8,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        rows.append(torch.cat((code, out.sequences.to(code.device)), dim=-1).view(-1).cpu().tolist())
    return rows


def rebuild_feedback_from_row(model: Any, hidden_row: torch.Tensor, full_row: list[int], *, device: str, dtype: torch.dtype) -> torch.Tensor:
    del hidden_row  # Hidden is unused in the current HF-compatible reconstruction.
    codes = torch.tensor(full_row, device=device, dtype=torch.long).view(1, -1)
    layer0_embed = model.get_input_embeddings()(codes[:, :1])
    residual_tokens = codes[:, 1:]
    residual_hiddens = []
    for group_idx in range(residual_tokens.shape[1] - 1):
        emb = model.code_predictor.get_input_embeddings()[group_idx](
            residual_tokens[:, group_idx : group_idx + 1]
        ).to(layer0_embed.device)
        residual_hiddens.append(emb)
    last_hidden = model.code_predictor.get_input_embeddings()[-1](
        residual_tokens[:, -1:]
    ).to(layer0_embed.device)
    codec_hiddens = torch.cat([layer0_embed] + residual_hiddens + [last_hidden], dim=1)
    return codec_hiddens.sum(1)[0].detach().cpu().float()


def analyze_predictor_boundary(
    runtime_cp_dump: dict[str, Any],
    hf_predictor_capture: dict[str, Any],
    *,
    model_path: str | Path,
    device: str,
    seed: int,
    steps: int,
) -> dict[str, Any]:
    model = load_hf_talker_model(model_path, device=device, dtype=torch.bfloat16)
    runtime_hidden_rows = [runtime_cp_dump["talker_hidden"][idx] for idx in range(min(steps, runtime_cp_dump["talker_hidden"].shape[0]))]
    hf_hidden_rows = [
        torch.tensor(hf_predictor_capture["predictor_calls"][idx]["talker_hidden"])
        for idx in range(min(steps, len(hf_predictor_capture["predictor_calls"])))
    ]
    layer0_codes = runtime_cp_dump["layer0_codes"][:steps].tolist()

    runtime_replay_fresh = replay_predictor_rows(
        model,
        runtime_hidden_rows,
        layer0_codes,
        device=device,
        dtype=torch.bfloat16,
        seed=seed,
        mode="fresh",
    )
    runtime_replay_sequential = replay_predictor_rows(
        model,
        runtime_hidden_rows,
        layer0_codes,
        device=device,
        dtype=torch.bfloat16,
        seed=seed,
        mode="sequential",
    )
    hf_replay_sequential = replay_predictor_rows(
        model,
        hf_hidden_rows,
        layer0_codes,
        device=device,
        dtype=torch.bfloat16,
        seed=seed,
        mode="sequential",
    )

    runtime_rows = runtime_cp_dump["output_codes"][:steps].tolist()
    hf_rows = [
        [layer0_codes[idx]] + hf_predictor_capture["predictor_calls"][idx]["sequences"][0]
        for idx in range(min(steps, len(hf_predictor_capture["predictor_calls"])))
    ]

    hidden_metrics = [
        {
            "step": idx,
            **metric(runtime_hidden_rows[idx], hf_hidden_rows[idx]),
        }
        for idx in range(min(len(runtime_hidden_rows), len(hf_hidden_rows)))
    ]

    feedback_metrics = []
    for idx in range(min(steps, len(runtime_rows), len(hf_rows))):
        runtime_feedback = runtime_cp_dump["feedbacks"][idx].float()
        rebuilt_runtime = rebuild_feedback_from_row(
            model,
            runtime_hidden_rows[idx],
            runtime_rows[idx],
            device=device,
            dtype=torch.bfloat16,
        )
        rebuilt_hf = rebuild_feedback_from_row(
            model,
            runtime_hidden_rows[idx],
            hf_rows[idx],
            device=device,
            dtype=torch.bfloat16,
        )
        feedback_metrics.append(
            {
                "step": idx,
                "runtime_feedback_vs_rebuilt_runtime_row": metric(runtime_feedback, rebuilt_runtime),
                "runtime_feedback_vs_rebuilt_hf_row_on_runtime_hidden": metric(runtime_feedback, rebuilt_hf),
            }
        )

    result = {
        "steps_checked": steps,
        "hidden_input_metrics": hidden_metrics,
        "runtime_rows": runtime_rows,
        "hf_rows": hf_rows,
        "runtime_replay_fresh": runtime_replay_fresh,
        "runtime_replay_sequential": runtime_replay_sequential,
        "hf_replay_sequential": hf_replay_sequential,
        "feedback_metrics": feedback_metrics,
        "first_failing_subcontract": None,
    }

    if runtime_replay_sequential != runtime_rows:
        result["first_failing_subcontract"] = "predictor_sampling_impl"
    elif hf_replay_sequential != hf_rows:
        result["first_failing_subcontract"] = "hf_predictor_capture_consistency"
    elif runtime_rows != hf_rows:
        result["first_failing_subcontract"] = "talker_hidden_into_predictor"

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-cp-dump", type=Path, required=True)
    parser.add_argument("--hf-predictor-capture", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result = analyze_predictor_boundary(
        torch.load(args.runtime_cp_dump, map_location="cpu"),
        load_json(args.hf_predictor_capture),
        model_path=args.model_path,
        device=args.device,
        seed=args.seed,
        steps=args.steps,
    )
    if args.out:
        save_json(result, args.out)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
