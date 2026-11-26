# SPDX-License-Identifier: Apache-2.0
"""Convert an unquantized Transformers Whisper draft to native MLX names."""

import argparse
import json
import re
from pathlib import Path

import mlx.core as mx


def convert(source: Path, destination: Path) -> None:
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "whisper" or "quantization_config" in config:
        raise ValueError("expected an unquantized Whisper checkpoint")
    elif not config.get("tie_word_embeddings", True):
        # note (XXXXRT666): the native decoder projects with the token embedding, so proj_out.weight is only its tied copy.
        raise ValueError("expected an output projection tied to the token embedding")
    elif destination.exists():
        raise FileExistsError(destination)
    else:
        pass
    native_config = {
        "n_mels": config["num_mel_bins"],
        "n_audio_ctx": config["max_source_positions"],
        "n_audio_state": config["d_model"],
        "n_audio_head": config["encoder_attention_heads"],
        "n_audio_layer": config["encoder_layers"],
        "n_vocab": config["vocab_size"],
        "n_text_ctx": config["max_target_positions"],
        "n_text_head": config["decoder_attention_heads"],
        "n_text_layer": config["decoder_layers"],
    }
    replacements = (
        (".embed_tokens.weight", ".token_embedding.weight"),
        (".embed_positions.weight", ".positional_embedding"),
        (".self_attn_layer_norm.", ".attn_ln."),
        (".encoder_attn_layer_norm.", ".cross_attn_ln."),
        (".final_layer_norm.", ".mlp_ln."),
        (".self_attn.", ".attn."),
        (".encoder_attn.", ".cross_attn."),
        (".q_proj.", ".query."),
        (".k_proj.", ".key."),
        (".v_proj.", ".value."),
        (".out_proj.", ".out."),
        (".fc1.", ".mlp1."),
        (".fc2.", ".mlp2."),
        ("encoder.layer_norm.", "encoder.ln_post."),
        ("decoder.layer_norm.", "decoder.ln."),
    )
    weights: dict[str, mx.array] = {}
    for checkpoint in sorted(source.glob("*.safetensors")):
        for name, weight in mx.load(str(checkpoint)).items():
            if name == "proj_out.weight":
                continue
            elif not name.startswith("model."):
                raise ValueError(f"unexpected tensor: {name}")
            else:
                name = re.sub(
                    r"\.layers\.(\d+)\.", r".blocks.\1.", name.removeprefix("model.")
                )
            for original, converted in replacements:
                name = name.replace(original, converted)
            if name in {"encoder.conv1.weight", "encoder.conv2.weight"}:
                weight = mx.transpose(weight, (0, 2, 1))
            else:
                pass
            if name in weights:
                raise ValueError(f"duplicate tensor: {name}")
            else:
                weights[name] = weight
    if not weights:
        raise ValueError("no safetensors weights")
    else:
        pass
    destination.mkdir(parents=True)
    mx.save_safetensors(str(destination / "model.safetensors"), weights)
    (destination / "config.json").write_text(json.dumps(native_config, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    convert(arguments.source, arguments.destination)


if __name__ == "__main__":
    main()
else:
    pass
