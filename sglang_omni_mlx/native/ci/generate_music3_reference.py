# SPDX-License-Identifier: Apache-2.0
"""Reproduces the frozen Python MLX Music3 reference and compares native WAVs."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as audio_functional
from provision import download, sha256
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--runtime-bin", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--prompt-only", action="store_true")
    arguments = parser.parse_args()
    directory = arguments.output_directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    ci_directory = Path(__file__).resolve().parent
    pin = json.loads((ci_directory / "music3_reference.json").read_text())
    checkpoint_pin = json.loads((ci_directory / "model_pins.json").read_text())[
        "mlx-community/MiniMax-Music3-mxfp8"
    ]
    for name, expected in checkpoint_pin["files"].items():
        if arguments.prompt_only and not name.startswith("tokenizer/"):
            continue
        else:
            pass
        if sha256(arguments.model_path / name) != expected:
            raise SystemExit(f"checkpoint hash mismatch: {name}")
        else:
            pass
    source_directory = directory / "reference-source"
    source_directory.mkdir(exist_ok=True)
    for path, digest in pin["files"].items():
        download(
            f"https://raw.githubusercontent.com/{pin['repository']}/{pin['revision']}/{path}",
            source_directory / Path(path).name,
            digest,
        )
    (source_directory / "__init__.py").write_text("")
    specification = importlib.util.spec_from_file_location(
        "music3_reference",
        source_directory / "__init__.py",
        submodule_search_locations=[str(source_directory)],
    )
    package = importlib.util.module_from_spec(specification)
    sys.modules["music3_reference"] = package
    specification.loader.exec_module(package)
    prompt = importlib.import_module("music3_reference.prompt")
    golden = json.loads((ci_directory / "model_golden/minimax-music3.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        arguments.model_path / "tokenizer", trust_remote_code=False
    )
    prompt.validate_tokenizer_ids(tokenizer)
    prompt_cases: list[tuple[str, str, str]] = [
        ("nested_bullet", "- * acoustic guitar", "[Verse]\nHello"),
        ("inline_bullet_spaces", "Style:  •   rock", "[Verse]\nHello"),
        ("wide_space", "Acoustic guitar\u3000", "[Verse]\nHello"),
        ("nbsp_heading", "#\u00a0Title", "[Verse]\nHello"),
        ("carriage_return", "# Folk\r- Piano", "[Verse]\r\nHello\r\n"),
        ("unicode_lines", "# Folk\u2028- Piano\u2029Soft vocals", "Hello"),
        (
            "splitlines_controls",
            "Folk\vPiano\fVocals\x1cDrums\x1dBass\x1eFlute\x85Warm",
            "Hello",
        ),
        ("control_whitespace", "\x1f#\x1fFolk\x1f", "Hello"),
        ("unicode_special", "<|\u3000style\u00a0gentle folk\u3000|>", "Hello"),
        ("markdown", "\t# **Folk** *piano* *soft*\n---\n• Guitar", "Hello"),
        ("leading_rule", "\n---\n", "Hello"),
        ("unicode_tags", "Folk", "[Куплет]\nHello\n[ÉTÉ]\nWorld"),
        ("greek_sigma", "Folk", "[ΟΣ]\nHello"),
        ("expanding_lowercase", "Folk", "[İNTRO]\nHello"),
        ("lyric_breaks", "Folk", "[Verse] discarded\nWords [CHŒUR] ^ next\n\n"),
        ("multiline_tag", "Folk", "[VERSE\nCHORUS]\nHello"),
        ("unclosed_tags", "<|style unfinished", "[ab[ab[ab"),
    ]
    prompt_cases.extend(
        (case["id"], case["instructions"], case["input"]) for case in golden["cases"]
    )
    prompt_requests = directory / "prompt-requests.jsonl"
    prompt_requests.write_text(
        "".join(
            json.dumps({"id": identifier, "instructions": caption, "input": lyrics})
            + "\n"
            for identifier, caption, lyrics in prompt_cases
        )
    )
    completed = subprocess.run([
        str(arguments.runtime_bin / "music3_synthesize"),
        "--model-path", str(arguments.model_path.resolve()),
        "--requests", str(prompt_requests), "--prompt-only",
    ], capture_output=True, text=True, check=True)  # fmt: skip
    native_prompts = [json.loads(line) for line in completed.stdout.splitlines()]
    if len(native_prompts) != len(prompt_cases):
        raise SystemExit("native prompt output count differs from the reference cases")
    else:
        pass
    for (identifier, caption, lyrics), native in zip(prompt_cases, native_prompts):
        reference_prompt = prompt.build_prompt(caption, lyrics)
        if (
            native["id"] != identifier
            or native["prompt"] != reference_prompt
            or native["token_ids"] != tokenizer(reference_prompt)["input_ids"]
        ):
            raise SystemExit(
                f"{identifier}: reference/native prompt or token IDs differ"
            )
        else:
            pass
    (directory / "prompt-parity.json").write_text(
        json.dumps({"matched_cases": [case[0] for case in prompt_cases]}, indent=2)
        + "\n"
    )
    print(f"Prompt text and token IDs match on {len(prompt_cases)} reference cases.")
    if arguments.prompt_only:
        return
    else:
        pass
    loader = importlib.import_module("music3_reference.loader")
    autoregressive = importlib.import_module("music3_reference.ar")
    euler = importlib.import_module("music3_reference.euler")
    request_file = directory / "requests.jsonl"
    request_file.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in golden["cases"])
    )
    native_directory = directory / "native"
    subprocess.run([
        str(arguments.runtime_bin / "music3_synthesize"),
        "--model-path", str(arguments.model_path.resolve()),
        "--requests", str(request_file),
        "--output-directory", str(native_directory),
    ], check=True)  # fmt: skip
    ar_model = loader.load_mlx_ar_model(str(arguments.model_path))
    acoustic_model = loader.load_mlx_acoustic_model(str(arguments.model_path))
    mx.set_cache_limit(1024**3)
    report = []
    for case in golden["cases"]:
        conditional = mx.array(
            tokenizer(prompt.build_prompt(case["instructions"], case["input"]))[
                "input_ids"
            ]
        )[None]
        unconditional = mx.concatenate(
            [
                conditional[:, :1],
                mx.full((1, conditional.shape[1] - 3), 151654, dtype=mx.int32),
                conditional[:, -2:],
            ],
            axis=1,
        )
        frames = autoregressive.generate_frame_hiddens(
            ar_model.language_model,
            ar_model.rvq_depth_decoder,
            ar_model.config,
            mx.concatenate([conditional, unconditional], axis=0),
            case["max_new_tokens"],
            seed=case["seed"],
        ).astype(mx.float16)
        previous_latent = previous_condition = None
        chunks = []
        for chunk_index, start in enumerate(range(0, frames.shape[1], 100)):
            end = min(start + 200, frames.shape[1])
            hidden = frames[:, start:end].astype(
                acoustic_model.condition_encoder.proj.weight.dtype
            )
            condition = acoustic_model.condition_encoder(hidden)
            digest = hashlib.blake2b(digest_size=8, person=b"minimax-ttm")
            digest.update(case["seed"].to_bytes(8, "little"))
            for part in ("dit", str(chunk_index)):
                encoded = part.encode()
                digest.update(len(encoded).to_bytes(4, "little"))
                digest.update(encoded)
            seed = int.from_bytes(digest.digest(), "little") & (2**63 - 1)
            noise = mx.random.normal(
                (1, acoustic_model.config.dit_in_channels, condition.shape[1]),
                key=mx.random.key(seed),
            ).astype(condition.dtype)
            latent, condition = euler.denoise_chunk(
                acoustic_model.transformer,
                noise,
                condition,
                num_inference_steps=case["dit_steps"],
                guidance_scale=1.7,
                previous_latent=previous_latent,
                previous_condition=previous_condition,
            )
            waveform = acoustic_model.vocoder(latent)[0].astype(mx.float32)
            mx.eval(waveform)
            waveform = np.clip(np.asarray(waveform), -1, 1)
            right = None if end == frames.shape[1] else -258 * 512
            chunks.append(waveform[:, 0 if start == 0 else 86 * 512 : right])
            saved_start = max(0, latent.shape[-1] - 344)
            saved_end = max(saved_start, latent.shape[-1] - 172)
            previous_latent = latent[..., saved_start:saved_end]
            previous_condition = condition[:, saved_start:saved_end]
            if end == frames.shape[1]:
                break
            else:
                pass
        waveform = (
            audio_functional.resample(
                torch.from_numpy(np.concatenate(chunks, axis=1).copy()), 44100, 32000
            )
            .numpy()
            .T
        )
        path = directory / f"{case['id']}-reference.wav"
        pcm = np.clip(np.rint(waveform * 32768), -32768, 32767).astype(np.int16)
        sf.write(path, pcm, 32000, subtype="PCM_16")
        native, sample_rate = sf.read(
            native_directory / f"{case['id']}.wav", always_2d=True
        )
        reference = pcm.astype(np.float64) / 32768
        if native.shape != reference.shape or sample_rate != 32000:
            raise SystemExit(f"{case['id']}: reference/native waveform shape differs")
        else:
            pass
        difference = native - reference
        peak = float(np.max(np.abs(difference)))
        rms = float(np.sqrt(np.mean(difference**2)))
        report.append(
            {
                "id": case["id"],
                "sample_count": len(native),
                "peak_error": peak,
                "rms_error": rms,
            }
        )
        if peak > 1e-3 or rms > 1e-4:
            raise SystemExit(
                f"{case['id']}: reference/native waveform differs: peak={peak}, rms={rms}"
            )
        else:
            pass
    (directory / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.copyfile(
        ci_directory / "music3_reference.json", directory / "reference-pin.json"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
