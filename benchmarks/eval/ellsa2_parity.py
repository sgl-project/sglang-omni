# SPDX-License-Identifier: Apache-2.0
"""ELLSA 2 motion node vs the official implementation on the ELLSA 2 test set.

Runs the node's model path (filterbank, history, generator, streaming RVQ
decode, one 0.8 s block at a time) on each clip's target and partner audio
with the clip's own style reference, runs the official `infer.py` on the same
clips (or reads its outputs), and compares the 18 codes of every step and the
decoded SMPL-H pose and FLAME motion code. Release decoding is greedy, so the
same inputs must give the same codes.

    python -m benchmarks.eval.ellsa2_parity \\
        --checkpoint /path/to/ELLSA2/checkpoints --data /path/to/ELLSA2/data \\
        --reference-code /path/to/ELLSA2/code --reference-python /path/to/env/bin/python \\
        --reference-dir outputs/reference --output-dir outputs/parity \\
        --encoder-attn-implementation flash_attention_2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
import torch

from sglang_omni.models.ellsa2.formats import POSE_FEATURES
from sglang_omni.models.ellsa2.generator import BLOCK_SAMPLES
from sglang_omni.models.ellsa2.loading import load_motion_models
from sglang_omni.models.ellsa2.motion import MotionDecoder, style_codes
from sglang_omni.models.ellsa2.session import MotionHooks, MotionSession
from sglang_omni.proto.streams import SESSION_INPUT_SAMPLE_RATE_HZ


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        frames = source.readframes(source.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768


def run_reference(args: argparse.Namespace, ids: list[str]) -> None:
    """The official infer.py on the clips whose outputs are missing."""
    missing = [gid for gid in ids if not (args.reference_dir / f"{gid}.npz").is_file()]
    for gid in missing:
        subprocess.run(
            [
                args.reference_python,
                "infer.py",
                "--model-path",
                str(args.checkpoint / "standalone_generator"),
                "--test-id",
                gid,
                "--output-dir",
                str(args.reference_dir),
                "--save-motion-tokens",
                "true",
            ],
            cwd=args.reference_code,
            check=True,
        )


# The official style encoding (encode_motion_style_tokens), run in the reference environment; argv: checkpoint, output dir, clip ids.
REFERENCE_STYLE_PROGRAM = """
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from infer import RELEASE_ARGS
from scripts import infer_ellsa2_gesture_expression_from_audio as inference
from scripts.prepare_testset import read_manifest
from release_paths import DATA_ROOT
checkpoint, out, ids = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
rows = {row["group_id"]: row for row in read_manifest()}
first = rows[ids[0]]
args = inference.finalize_args(inference.parse_args([*RELEASE_ARGS, "--model-path", str(checkpoint / "standalone_generator"),
    "--device", "cuda:0", "--motion-style-pose-path", str(DATA_ROOT / first["motion_style_pose_path"])]))
device = inference.choose_device(args.device)
decoders = inference.load_motion_decoders(args, device)
stats = inference.load_gesture_stats(args.stats_dir, args.stats_prefix, args.stats_suffix)
flame = inference.load_flame_decoder(args.vqvae_flame_path, device, inference.resolve_rvq_context_frames(args, "expression"),
    args.rvq_encoder_dtype, args.rvq_quantizer_distance_dtype)
mean, std = inference.load_flame_stats(args.flame_stats_path)
for gid in ids:
    row = rows[gid]
    codes = inference.encode_motion_style_tokens(str(DATA_ROOT / row["motion_style_pose_path"]),
        str(DATA_ROOT / row["motion_style_flame_path"]), 0, 612, decoders, stats, flame, mean, std, args)
    np.save(out / (gid + "_style.npy"), codes[: len(codes) // 6 * 6].numpy())
"""


def run_reference_style(args: argparse.Namespace, ids: list[str]) -> None:
    """The official style codes of the clips that lack them."""
    missing = [
        gid for gid in ids if not (args.reference_dir / f"{gid}_style.npy").is_file()
    ]
    if missing:
        subprocess.run(
            [
                args.reference_python,
                "-c",
                REFERENCE_STYLE_PROGRAM,
                str(args.checkpoint),
                str(args.reference_dir),
                *missing,
            ],
            cwd=args.reference_code,
            check=True,
        )
    else:
        pass


def reference_motion(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """The official output's codes [steps, 18] and rows [frames, 264] in smplh_flame_motion order."""
    with np.load(path) as data:
        pose = np.concatenate(
            [
                data["global_orient"],
                data["body_pose"],
                data["left_hand_pose"],
                data["right_hand_pose"],
            ],
            axis=1,
        )
        return data["rvq_combined_code_idx"], np.concatenate(
            [pose, data["motioncode"]], axis=1
        ).astype(np.float32)


def first_divergence(ours: np.ndarray, theirs: np.ndarray) -> dict | None:
    steps = min(len(ours), len(theirs))
    differ = np.argwhere(ours[:steps] != theirs[:steps])
    if len(differ) == 0:
        return None
    else:
        step, codebook = (int(value) for value in differ[0])
    return {
        "block": step // 6,
        "step": step % 6,
        "codebook": codebook,
        "ours": int(ours[step, codebook]),
        "reference": int(theirs[step, codebook]),
        "differing_steps": int(len(np.unique(differ[:, 0]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--reference-code", type=Path)
    parser.add_argument("--reference-python")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-id", action="append")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--encoder-attn-implementation",
        choices=("sdpa", "flash_attention_2"),
        default="flash_attention_2",
    )
    parser.add_argument(
        "--cudnn-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="let the backbone's attention use cuDNN, as the reference does; bit-for-bit parity needs it",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-outputs", action="store_true")
    parser.add_argument(
        "--style-codes",
        choices=("ours", "reference"),
        default="ours",
        help="ours: the node's own style encoding (the full node path); reference: the official style codes, which isolates the generator",
    )
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in (args.data / "testset/manifest.jsonl").read_text().splitlines()
        if line
    ]
    if args.test_id:
        rows = [row for row in rows if row["group_id"] in args.test_id]
    else:
        pass
    rows = rows[: args.limit or None]
    ids = [row["group_id"] for row in rows]
    if args.reference_python:
        run_reference(args, ids)
        run_reference_style(args, ids)
    else:
        pass
    args.output_dir.mkdir(parents=True, exist_ok=True)
    first = rows[0]
    models = load_motion_models(
        model_path=str(args.checkpoint),
        data_path=str(args.data),
        style=Path(first["motion_style_pose_path"]).stem,
        encoder_attn_implementation=args.encoder_attn_implementation,
        cudnn_attention=args.cudnn_attention,
        device=torch.device(args.device),
    )
    hooks = MotionHooks(models, SESSION_INPUT_SAMPLE_RATE_HZ)
    results = []
    for row in rows:
        gid = row["group_id"]
        ours_style = style_codes(
            args.data / row["motion_style_pose_path"],
            args.data / row["motion_style_flame_path"],
            models.tokenizers,
            models.normalization,
        )
        reference_style_path = args.reference_dir / f"{gid}_style.npy"
        reference_style = (
            torch.from_numpy(np.load(reference_style_path)).to(models.device)
            if reference_style_path.is_file()
            else None
        )
        models.style = (
            reference_style if args.style_codes == "reference" else ours_style
        )
        target = read_wav(args.data / row["audio20_target"])
        partner = read_wav(args.data / row["audio20_partner"])
        session = MotionSession(
            decoder=MotionDecoder(models.tokenizers, models.normalization)
        )
        codes, motion, block_ms = [], [], []
        for start in range(
            0,
            min(len(target), len(partner)) // BLOCK_SAMPLES * BLOCK_SAMPLES,
            BLOCK_SAMPLES,
        ):
            torch.cuda.synchronize(models.device)
            began = time.perf_counter()
            motion.append(
                hooks.generate(
                    session,
                    target[start : start + BLOCK_SAMPLES],
                    partner[start : start + BLOCK_SAMPLES],
                )
            )
            torch.cuda.synchronize(models.device)
            block_ms.append((time.perf_counter() - began) * 1000)
            codes.append(session.history[-1].codes.cpu().numpy())
        codes = np.concatenate(codes)
        motion = np.concatenate(motion)
        reference_codes, reference_rows = reference_motion(
            args.reference_dir / f"{gid}.npz"
        )
        divergence = first_divergence(codes, reference_codes)
        difference = np.abs(motion - reference_rows)
        results.append(
            {
                "group_id": gid,
                "blocks": len(codes) // 6,
                "codes_identical": divergence is None
                and codes.shape == reference_codes.shape,
                "first_divergence": divergence,
                "pose_max_abs_diff": float(difference[:, :POSE_FEATURES].max()),
                "flame_max_abs_diff": float(difference[:, POSE_FEATURES:].max()),
                "block_ms_p50": float(np.percentile(block_ms, 50)),
            }
        )
        print(json.dumps(results[-1]), flush=True)
        if args.save_outputs:
            np.savez(args.output_dir / f"{gid}.npz", codes=codes, rows=motion)
        else:
            pass
    summary = {
        "encoder_attn_implementation": args.encoder_attn_implementation,
        "cudnn_attention": args.cudnn_attention,
        "style_codes": args.style_codes,
        "clips": len(results),
        "clips_codes_identical": sum(r["codes_identical"] for r in results),
        "pose_max_abs_diff": max(r["pose_max_abs_diff"] for r in results),
        "flame_max_abs_diff": max(r["flame_max_abs_diff"] for r in results),
        "pose_max_abs_diff_identical_clips": max(
            (r["pose_max_abs_diff"] for r in results if r["codes_identical"]),
            default=None,
        ),
        "flame_max_abs_diff_identical_clips": max(
            (r["flame_max_abs_diff"] for r in results if r["codes_identical"]),
            default=None,
        ),
        "first_divergences": {
            r["group_id"]: r["first_divergence"]
            for r in results
            if r["first_divergence"]
        },
        "block_ms_p50": float(np.median([r["block_ms_p50"] for r in results])),
    }
    (args.output_dir / "parity.json").write_text(
        json.dumps({"summary": summary, "clips": results}, indent=1)
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
