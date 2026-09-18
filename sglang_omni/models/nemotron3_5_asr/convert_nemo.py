# SPDX-License-Identifier: Apache-2.0
"""Convert the original NeMo checkpoint to official Transformers tensor names.

Only small processor/tokenizer metadata is obtained from Hugging Face. All
learned parameters come from the supplied .nemo archive (e.g. from ModelScope).
No NeMo installation or remote model code is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

from .hf_compat import Nemotron3_5AsrConfig, Nemotron3_5AsrForRNNT

METADATA_REPO = "nvidia/nemotron-3.5-asr-streaming-0.6b"
METADATA_REVISION = "1c8deaecc64b91f034d73e08dd8b64625eb3395d"
METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def rename_weight(name: str) -> str | None:
    if name.startswith("preprocessor."):
        return None
    replacements = {
        "encoder.pre_encode.out.": "encoder.subsampling.linear.",
        "encoder.pre_encode.conv.0.": "encoder.subsampling.conv_in.",
        "encoder.pre_encode.conv.2.": "encoder.subsampling.layers.0.depthwise_conv.",
        "encoder.pre_encode.conv.3.": "encoder.subsampling.layers.0.pointwise_conv.",
        "encoder.pre_encode.conv.5.": "encoder.subsampling.layers.1.depthwise_conv.",
        "encoder.pre_encode.conv.6.": "encoder.subsampling.layers.1.pointwise_conv.",
        ".self_attn.linear_q.": ".self_attn.q_proj.",
        ".self_attn.linear_k.": ".self_attn.k_proj.",
        ".self_attn.linear_v.": ".self_attn.v_proj.",
        ".self_attn.linear_out.": ".self_attn.o_proj.",
        ".self_attn.linear_pos.": ".self_attn.relative_k_proj.",
        ".self_attn.pos_bias_u": ".self_attn.bias_u",
        ".self_attn.pos_bias_v": ".self_attn.bias_v",
        ".conv.batch_norm.": ".conv.norm.",
        "decoder.prediction.embed.": "decoder.embedding.",
        "decoder.prediction.dec_rnn.lstm.": "decoder.lstm.",
        "joint.enc.": "encoder_projector.",
        "joint.pred.": "decoder.decoder_projector.",
        "joint.joint_net.2.": "joint.head.",
        "prompt_kernel.0.": "prompt_projector.linear_1.",
        "prompt_kernel.2.": "prompt_projector.linear_2.",
    }
    for source, target in replacements.items():
        name = name.replace(source, target)
    return name


def convert(nemo_path: Path, output: Path, *, metadata_path: Path | None = None):
    output.mkdir(parents=True, exist_ok=True)
    target = output / "model.safetensors"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    hasher = hashlib.sha256()
    with nemo_path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if metadata_path is None:
        metadata_path = Path(
            snapshot_download(
                METADATA_REPO,
                revision=METADATA_REVISION,
                allow_patterns=list(METADATA_FILES),
            )
        )
    config = Nemotron3_5AsrConfig.from_pretrained(metadata_path, local_files_only=True)
    with tempfile.TemporaryDirectory(prefix="nemotron-convert-") as temp:
        checkpoint = Path(temp) / "model_weights.ckpt"
        # Copy a single regular member; never extract archive paths or symlinks.
        with tarfile.open(nemo_path, "r|*") as archive:
            for member in archive:
                if (
                    member.name.removeprefix("./") == "model_weights.ckpt"
                    and member.isfile()
                ):
                    with (
                        archive.extractfile(member) as source,
                        checkpoint.open("wb") as dest,
                    ):
                        shutil.copyfileobj(source, dest)
                    break
            else:
                raise ValueError(
                    "NeMo archive has no regular model_weights.ckpt member"
                )
        original = torch.load(
            checkpoint, map_location="cpu", weights_only=True, mmap=True
        )
        weights = {}
        for name, value in original.items():
            mapped = rename_weight(name)
            if mapped is None:
                continue
            if mapped in weights:
                raise ValueError(f"Duplicate converted parameter: {mapped}")
            weights[mapped] = value.contiguous()
        with torch.device("meta"):
            reference = Nemotron3_5AsrForRNNT(config)
        expected = reference.state_dict()
        if weights.keys() != expected.keys():
            raise ValueError(
                f"Missing: {expected.keys() - weights.keys()}; unexpected: {weights.keys() - expected.keys()}"
            )
        for name, tensor in weights.items():
            if tensor.shape != expected[name].shape:
                raise ValueError(
                    f"{name}: expected {expected[name].shape}, got {tensor.shape}"
                )
        save_file(weights, str(target), metadata={"format": "pt"})
    for name in METADATA_FILES:
        if (metadata_path / name).resolve() != (output / name).resolve():
            shutil.copyfile(metadata_path / name, output / name)
    (output / "conversion.json").write_text(
        json.dumps(
            {
                "source": nemo_path.name,
                "source_sha256": digest,
                "metadata_repo": METADATA_REPO,
                "metadata_revision": METADATA_REVISION,
                "parameters": len(weights),
                "format": "official Transformers float32",
            },
            indent=2,
        )
        + "\n"
    )
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metadata-path", type=Path, help="Use already downloaded official metadata"
    )
    args = parser.parse_args()
    print(convert(args.nemo_path, args.output, metadata_path=args.metadata_path))


if __name__ == "__main__":
    main()
