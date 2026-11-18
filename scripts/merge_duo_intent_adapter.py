"""Prepare a native Intent checkpoint using the reference duplex PEFT loader."""

import argparse
import json
from pathlib import Path

import torch
from torchtitan.models.duo_realization.modules.intent import load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intent", required=True)
    parser.add_argument("--intent-adapter", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--decode", choices=["duo_5_17", "sana_1_4"], default="duo_5_17"
    )
    args = parser.parse_args()
    args.origin_intent = False
    args.intent_load_device = "cpu"
    args.intent_attention = "sdpa"
    args.intent_max_pixels = 16384
    args.output.mkdir(parents=True, exist_ok=False)
    processor, model, identity = load(args, dtype=torch.bfloat16)
    model = model.merge_and_unload(safe_merge=True)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="5GB")
    processor.save_pretrained(args.output)
    identity["base_source"] = str(Path(args.intent).resolve())
    identity["adapter_source"] = str(Path(args.intent_adapter).resolve())
    (args.output / "duplex_adapter_merge.json").write_text(
        json.dumps(identity, indent=2)
    )


if __name__ == "__main__":
    main()
