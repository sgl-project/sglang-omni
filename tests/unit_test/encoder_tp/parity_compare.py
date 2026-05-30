# SPDX-License-Identifier: Apache-2.0
"""Compare encoder-output pickles from the parity harness.

The legacy printout still reports strict elementwise allclose numbers, but
``--tp`` is the production TP-parity gate: it checks direction-preserving
per-token similarity and coarse outlier counts instead of pretending fp16 TP
collectives can satisfy bit-style tolerances.
"""
from __future__ import annotations

import argparse
import dataclasses
import pickle
import sys

import torch

DEFAULT_TENSOR_LABELS = ("image_embeds", "video_embeds", "audio_embeds")
DEFAULT_DEEPSTACK_LABELS = (
    "deepstack",
    "deepstack_visual_embeds_image",
    "deepstack_visual_embeds_video",
)


@dataclasses.dataclass(frozen=True)
class TensorStats:
    label: str
    shape: tuple[int, ...]
    abs_diff_max: float
    abs_diff_mean: float
    token_diff_gt_01: int
    token_diff_gt_1: int
    cos_min: float
    cos_mean: float


def _stats_for_tensors(label: str, le: torch.Tensor, re: torch.Tensor) -> TensorStats:
    abs_diff = (le - re).abs()
    token_max = abs_diff.flatten(1).max(dim=-1).values
    cos = torch.nn.functional.cosine_similarity(le.flatten(1), re.flatten(1), dim=-1)
    return TensorStats(
        label=label,
        shape=tuple(le.shape),
        abs_diff_max=abs_diff.max().item(),
        abs_diff_mean=abs_diff.mean().item(),
        token_diff_gt_01=int((token_max > 0.1).sum().item()),
        token_diff_gt_1=int((token_max > 1.0).sum().item()),
        cos_min=cos.min().item(),
        cos_mean=cos.mean().item(),
    )


def _to_float_tensor(value: object) -> torch.Tensor | None:
    if isinstance(value, list) or not torch.is_tensor(value):
        return None
    return value.to(torch.float32)


def _summarize(left: dict, right: dict, label: str) -> TensorStats | None:
    if label not in left or label not in right:
        return None
    le = _to_float_tensor(left[label])
    re = _to_float_tensor(right[label])
    if le is None or re is None:
        return None

    print(f"\n{label}: shape={tuple(le.shape)}")
    print(
        "  left stats:  "
        f"abs_max={le.abs().max():.3f} mean={le.mean():.4f} std={le.std():.4f}"
    )
    print(
        "  right stats: "
        f"abs_max={re.abs().max():.3f} mean={re.mean():.4f} std={re.std():.4f}"
    )

    stats = _stats_for_tensors(label, le, re)
    abs_diff = (le - re).abs()
    print(
        "  abs diff: "
        f"max={abs_diff.max().item():.4f} mean={abs_diff.mean().item():.4f}"
    )
    flat_diff = abs_diff.flatten(1)
    token_max = flat_diff.max(dim=-1).values
    quantiles = torch.quantile(
        token_max,
        torch.tensor([0.5, 0.9, 0.99, 0.999], dtype=torch.float32),
    )
    print(
        "  per-token max abs diff: "
        f"p50={quantiles[0].item():.4f} p90={quantiles[1].item():.4f} "
        f"p99={quantiles[2].item():.4f} p999={quantiles[3].item():.4f} "
        f">0.1={(token_max > 0.1).sum().item()} "
        f">1.0={(token_max > 1.0).sum().item()}"
    )

    cos = torch.nn.functional.cosine_similarity(le.flatten(1), re.flatten(1), dim=-1)
    print(
        "  per-token cosine_sim: "
        f"min={cos.min().item():.6f} mean={cos.mean().item():.6f}"
    )
    for lo, hi in [
        (0.0, 0.9),
        (0.9, 0.99),
        (0.99, 0.999),
        (0.999, 0.9999),
        (0.9999, 1.0001),
    ]:
        count = ((cos >= lo) & (cos < hi)).sum().item()
        print(f"    cos in [{lo:.4f}, {hi:.4f}): {count}")

    for atol, rtol in [(1e-3, 1e-3), (1e-2, 1e-2), (5e-2, 5e-2)]:
        ok = torch.allclose(le, re, atol=atol, rtol=rtol)
        marker = "  <-- RFC" if (atol, rtol) == (1e-3, 1e-3) else ""
        print(f"  allclose(atol={atol}, rtol={rtol}): {ok}{marker}")

    return stats


def _summarize_deepstack(
    left: dict,
    right: dict,
    *,
    key: str,
    label_prefix: str,
) -> list[TensorStats]:
    stats: list[TensorStats] = []
    if not left.get(key) or not right.get(key):
        return stats
    for i, (left_ds, right_ds) in enumerate(zip(left[key], right[key])):
        ds_stats = _summarize(
            {f"{label_prefix}[{i}]": left_ds},
            {f"{label_prefix}[{i}]": right_ds},
            f"{label_prefix}[{i}]",
        )
        if ds_stats is not None:
            stats.append(ds_stats)
    return stats


def _summarize_layers(left: dict, right: dict, *, top_layers: int) -> None:
    left_layers = left.get("layers")
    right_layers = right.get("layers")
    if not isinstance(left_layers, dict) or not isinstance(right_layers, dict):
        print("\nlayer summary: no layer captures in one or both inputs")
        return

    stats = []
    shape_mismatches = []
    for key in sorted(set(left_layers) & set(right_layers)):
        le = _to_float_tensor(left_layers[key])
        re = _to_float_tensor(right_layers[key])
        if le is None or re is None:
            continue
        if le.shape != re.shape:
            shape_mismatches.append((key, tuple(le.shape), tuple(re.shape)))
            continue
        stats.append(_stats_for_tensors(key, le.flatten(1), re.flatten(1)))

    stats.sort(key=lambda item: item.abs_diff_max, reverse=True)
    print(f"\nlayer summary: {len(stats)} common captured tensors")
    print(
        "  label                                     max_abs   mean_abs  >0.1  >1.0  cos_mean  cos_min"
    )
    for item in stats[:top_layers]:
        print(
            f"  {item.label:<40} "
            f"{item.abs_diff_max:7.4f} "
            f"{item.abs_diff_mean:9.6f} "
            f"{item.token_diff_gt_01:5d} "
            f"{item.token_diff_gt_1:5d} "
            f"{item.cos_mean:8.6f} "
            f"{item.cos_min:8.6f}"
        )
    if shape_mismatches:
        print(
            "\nlayer summary: skipped shape-mismatched shard-local tensors "
            f"({len(shape_mismatches)})"
        )
        for label, left_shape, right_shape in shape_mismatches[:top_layers]:
            print(f"  {label:<40} left={left_shape} right={right_shape}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("left_pkl")
    p.add_argument("right_pkl")
    p.add_argument(
        "--tp",
        action="store_true",
        help="Apply the tp=1 vs tp>1 parity gate.",
    )
    p.add_argument("--min-mean-cos", type=float, default=0.999)
    p.add_argument("--max-tokens-gt-1", type=int, default=0)
    p.add_argument(
        "--layers",
        action="store_true",
        help="Print a compact max-diff ranking for captured layer tensors.",
    )
    p.add_argument("--top-layers", type=int, default=30)
    p.add_argument(
        "--labels",
        default=None,
        help=(
            "Comma-separated top-level tensor labels to compare. Defaults to "
            "image_embeds,video_embeds,audio_embeds when present."
        ),
    )
    args = p.parse_args()

    left = pickle.load(open(args.left_pkl, "rb"))
    right = pickle.load(open(args.right_pkl, "rb"))

    stats: list[TensorStats] = []
    labels = (
        tuple(label.strip() for label in args.labels.split(",") if label.strip())
        if args.labels
        else DEFAULT_TENSOR_LABELS
    )
    for label in labels:
        tensor_stats = _summarize(left, right, label)
        if tensor_stats is not None:
            stats.append(tensor_stats)
    for key in DEFAULT_DEEPSTACK_LABELS:
        stats.extend(
            _summarize_deepstack(
                left,
                right,
                key=key,
                label_prefix="ds" if key == "deepstack" else key,
            )
        )

    if args.layers:
        _summarize_layers(left, right, top_layers=args.top_layers)

    if not args.tp:
        return

    failures = []
    for item in stats:
        if item.cos_mean < args.min_mean_cos:
            failures.append(
                f"{item.label}: mean cosine {item.cos_mean:.6f} "
                f"< {args.min_mean_cos:.6f}"
            )
        if item.token_diff_gt_1 > args.max_tokens_gt_1:
            failures.append(
                f"{item.label}: tokens with max abs diff > 1.0 "
                f"{item.token_diff_gt_1} > {args.max_tokens_gt_1}"
            )

    if failures:
        print("\nTP parity: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print("\nTP parity: PASS")


if __name__ == "__main__":
    main()
