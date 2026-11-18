"""Prepare verified Duo release metadata and Codec paths without changing tensors."""

import argparse
import copy
import json
import shutil
from pathlib import Path

import yaml
from tools.ckpt_to_oss.duo_protocol import checksum, verify_file
from torchtitan.models.duo_realization.modules.codec_spec import validate_codec


def prepare_checkpoint(
    release: Path,
    base_release: Path,
    output: Path,
    codec_config_directory: Path,
    rope: str,
    adaln_timestep: str,
) -> None:
    release = release.resolve()
    base_release = base_release.resolve()
    source = release / "realization"
    manifest = json.loads((source / "complete.json").read_text())
    if manifest["format"] != "duo-release-dcp-v1" or manifest["stage"] != "dmd":
        raise ValueError("Expected a Duo DMD DCP release; ModelOpt is a separate input")
    else:
        pass
    if manifest["inference"] != {"flow_shift": 5.0, "sampling_steps": 4}:
        raise ValueError("This deployment profile requires four-step, flow-shift-5 DMD")
    else:
        pass
    video_config_path = codec_config_directory / "video_vae.yaml"
    audio_config_path = codec_config_directory / "audio_vae.yaml"
    video_config = yaml.safe_load(video_config_path.read_text())
    audio_config = yaml.safe_load(audio_config_path.read_text())
    video_settings = {
        "latents_mean": video_config["latents_mean"],
        "latents_std": video_config["latents_std"],
        "latent_channels": video_config["z_channels"],
        **{
            f"vae_{name}": video_config["forward_kwargs"][name]
            for name in (
                "clip_length",
                "token_drop",
                "encoder_tiling",
                "decoder_tiling",
                "tile_size",
                "tile_overlap_min",
            )
        },
    }
    # Parallel tile placement follows the SGLang execution configuration.
    audio_settings = {
        "latents_mean": audio_config["latents_mean"],
        "latents_std": audio_config["latents_std"],
        "sample_rate": audio_config["sample_rate"],
        "latent_channels": audio_config["vae_latent_channels"],
    }
    for name, record in manifest["files"].items():
        verify_file(source / name, record)
    intent_identity = manifest["identity"]["intent"]
    adapter = base_release / "intent_adapter"
    if (
        checksum(adapter / "protocol_manifest.json")
        != intent_identity["protocol_manifest_sha256"]
    ):
        raise ValueError("Intent protocol differs from Realization identity")
    else:
        pass
    for name, digest in intent_identity["adapter_weights"].items():
        if checksum(adapter / name) != digest:
            raise ValueError(f"Intent adapter checksum differs: {name}")
        else:
            pass
    entry = copy.deepcopy(manifest["models"]["generator"])
    backbone = (base_release / "realization" / entry["base"]["path"]).resolve()
    for name, field in (
        ("config.json", "config_sha256"),
        ("model.safetensors.index.json", "index_sha256"),
    ):
        if checksum(backbone / name) != entry["base"][field]:
            raise ValueError(f"Realization backbone identity differs: {name}")
        else:
            pass
    if entry["base"]["format"] != "original_duo":
        raise ValueError("Expected the original Duo safetensors backbone")
    else:
        pass
    base_model = entry["decfg"]["base_model"]
    entry["base"].update(
        format="original_h3", base_model=base_model, path=str(backbone)
    )
    # DeCFG is a training-only teacher branch, not part of learned DMD state.
    entry["decfg"] = None
    codec_identity = dict(
        manifest["identity"]["codec"],
        mode="duo_5_17",
        normalization="duo_channel_mean_std",
    )
    validate_codec({"codec": codec_identity}, "duo_5_17")
    codec_paths = (
        (
            video_settings,
            base_release / "codec/video_vae/model.safetensors",
            Path("FL2VA/video_vae/source/model.safetensors"),
            Path("FL2VA/video_vae/config.json"),
        ),
        (
            audio_settings,
            base_release / "codec/audio_vae/diffusion_pytorch_model.safetensors",
            Path("audio_vae/diffusion_pytorch_model.safetensors"),
            Path("audio_vae/config.json"),
        ),
    )
    for _, weights, _, _ in codec_paths:
        if checksum(weights) != codec_identity["weights"][weights.name]:
            raise ValueError(f"Codec weights do not match normalization: {weights}")
        else:
            pass
    output.mkdir(parents=True, exist_ok=False)
    for component_settings, weights, destination, settings in codec_paths:
        target = output / "codec" / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(weights.resolve())
        (output / "codec" / settings).write_text(
            json.dumps(component_settings, indent=2) + "\n"
        )
    (output / "codec/model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "MiniMaxH3ModularPipeline",
                "_diffusers_version": json.loads(
                    (backbone / "config.json").read_text()
                )["_diffusers_version"],
                "vae": ["diffusers", "AutoencoderKLMiniMaxH3"],
                "audio_vae": ["diffusers", "AutoencoderKLMiniMaxH3Audio"],
            },
            indent=2,
        )
        + "\n"
    )
    (output / "codec/vae").symlink_to("FL2VA/video_vae", target_is_directory=True)
    checkpoint = output / "realization"
    checkpoint.mkdir()
    # Copy the small adapter DCP; its tensors and original inventory stay intact.
    shutil.copytree(source / "dcp", checkpoint / "dcp")
    prepared = {
        "format": "h3-dcp-v1",
        "checkpoint_id": manifest["checkpoint_id"] + "-sglang",
        "stage": manifest["stage"],
        "identity": {
            "h3": dict(
                manifest["identity"]["realization"],
                rope=rope,
                adaln_timestep=adaln_timestep,
                base_model=base_model,
                sampling_steps=manifest["inference"]["sampling_steps"],
                flow_shift=manifest["inference"]["flow_shift"],
            ),
            "qwen": intent_identity,
            "codec": codec_identity,
        },
        "models": {"generator": entry},
        "files": manifest["files"],
        "conversion": {
            "source": str(release),
            "base_release": str(base_release),
            "source_manifest_sha256": checksum(source / "complete.json"),
            "codec_config_sha256": {
                "video_vae.yaml": checksum(video_config_path),
                "audio_vae.yaml": checksum(audio_config_path),
            },
            "method": "unchanged-adapter-dcp-and-original-backbone",
        },
    }
    for name, record in prepared["files"].items():
        verify_file(checkpoint / name, record)
    for name in ("manifest.json", "complete.json"):
        (checkpoint / name).write_text(json.dumps(prepared, indent=2) + "\n")
    print(f"CODEC_MODEL={output.resolve() / 'codec'}")
    print(f"REALIZATION_CHECKPOINT={checkpoint.resolve()}")
    print(f"INTENT_ADAPTER={adapter}")
    print(f"ASR_MODEL={base_release / 'speech/recognizer'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--base-release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codec-config-dir", type=Path, required=True)
    parser.add_argument("--rope", choices=("old", "new"), required=True)
    parser.add_argument("--adaln-timestep", choices=("old", "new"), required=True)
    options = parser.parse_args()
    prepare_checkpoint(
        options.release,
        options.base_release,
        options.output,
        options.codec_config_dir,
        options.rope,
        options.adaln_timestep,
    )


if __name__ == "__main__":
    main()
