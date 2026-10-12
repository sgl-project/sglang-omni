# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert an EasyMagpie ``.nemo`` talker + codec into an SGLang-Omni model directory.

Usage::

    python -m easymagpie_converter \\
        --nemo-checkpoint talker.nemo --codec-checkpoint codec.nemo \\
        --phoneme-tokenizer ipa_bpe.json --context-texts-csv context_texts.csv \\
        --output /path/to/new_model_dir
"""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from pathlib import Path

from easymagpie_converter.codec import CODEC_SUBDIR, convert_codec
from easymagpie_converter.contexts import (
    BUNDLE_FILE,
    REGISTRY_FILE,
    SPEAKER_SUBDIR,
    plan_contexts,
    write_contexts,
)
from easymagpie_converter.talker import DTYPES, codec_layout, convert_talker, load_model, write_json

LOG = logging.getLogger("easymagpie_converter")

REQUIRED_OUTPUTS = (
    "config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    f"{CODEC_SUBDIR}/config.json",
    f"{CODEC_SUBDIR}/model.safetensors",
    f"{SPEAKER_SUBDIR}/{BUNDLE_FILE}",
    f"{SPEAKER_SUBDIR}/{REGISTRY_FILE}",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nemo-checkpoint", required=True, type=Path, help="EasyMagpie talker .nemo")
    parser.add_argument("--codec-checkpoint", required=True, type=Path, help="25 fps spectral codec .nemo")
    parser.add_argument(
        "--phoneme-tokenizer",
        type=Path,
        default=None,
        help="IPA BPE tokenizer JSON; overrides the path stored in the checkpoint",
    )
    parser.add_argument(
        "--context-texts-csv",
        required=True,
        type=Path,
        help="CSV with speaker_name, languages, context_text columns",
    )
    parser.add_argument("--output", required=True, type=Path, help="New model directory (must not exist)")
    parser.add_argument("--default-voice", default=None, help="Default speaker; first CSV voice if omitted")
    parser.add_argument(
        "--text-tokenizer",
        default=None,
        help="HF tokenizer name or local directory; defaults to the checkpoint's text tokenizer",
    )
    parser.add_argument("--dtype", default="float16", choices=sorted(DTYPES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precompute-batch-size", type=int, default=64)
    parser.add_argument(
        "--disable-cas-for-context-text",
        action="store_true",
        help="Only for legacy checkpoints trained without CAS embeddings on context text",
    )
    return parser.parse_args()


def existing_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def file_record(path: Path) -> dict[str, object]:
    return {"name": path.name, "bytes": path.stat().st_size}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse_args()
    nemo_checkpoint = existing_file(args.nemo_checkpoint, "talker checkpoint")
    codec_checkpoint = existing_file(args.codec_checkpoint, "codec checkpoint")
    phoneme_tokenizer = (
        None if args.phoneme_tokenizer is None else existing_file(args.phoneme_tokenizer, "phoneme tokenizer")
    )
    contexts_csv = existing_file(args.context_texts_csv, "context texts CSV")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    plan = plan_contexts(contexts_csv, args.default_voice)
    LOG.info(
        "Planned %d contexts for %d voices; default voice %r",
        len(plan.contexts),
        len(plan.voice_contexts),
        plan.default_speaker_id,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        model = load_model(
            nemo_checkpoint,
            codec_checkpoint,
            phoneme_tokenizer,
            disable_cas_for_context_text=args.disable_cas_for_context_text,
            device=args.device,
        )
        num_codebooks, levels, stacking = codec_layout(model)
        config = convert_talker(
            model,
            staging,
            dtype=args.dtype,
            precompute_batch_size=args.precompute_batch_size,
            text_tokenizer=args.text_tokenizer,
        )
        convert_codec(
            codec_checkpoint,
            staging,
            num_codebooks=num_codebooks,
            num_levels_per_group=levels,
            frame_stacking_factor=stacking,
        )
        write_contexts(model, staging, plan, config)
        missing = [name for name in REQUIRED_OUTPUTS if not (staging / name).is_file()]
        if missing:
            raise RuntimeError(f"conversion did not produce {missing}")
        write_json(
            staging / "conversion.json",
            {
                "talker": file_record(nemo_checkpoint),
                "codec": file_record(codec_checkpoint),
                "phoneme_tokenizer": None if phoneme_tokenizer is None else file_record(phoneme_tokenizer),
                "context_texts_csv": file_record(contexts_csv),
                "dtype": args.dtype,
            },
        )
        staging.chmod(0o755)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    LOG.info("Done: %s", output)


if __name__ == "__main__":
    main()
