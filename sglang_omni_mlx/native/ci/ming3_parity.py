"""Compare Ming frontend and cached decoder tensors with Python MLX Audio."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_audio.stt import load
from mlx_audio.stt.models.ming_v3_asr.ming_v3_asr import ASR_INSTRUCTION, Model


def compare_audio(
    model: Model, model_path: Path, runtime_bin: Path, audio_path: Path
) -> None:
    features = model.prepare_audio(str(audio_path))
    expected_audio = model.encode_audio(features)
    mx.eval(expected_audio)
    with tempfile.TemporaryDirectory(prefix="ming-parity-") as directory:
        root = Path(directory)
        subprocess.run(
            [
                str(runtime_bin / "ming3_asr_probe"),
                "encode",
                str(audio_path),
                str(root / "audio.safetensors"),
                str(model_path),
            ],
            check=True,
        )
        actual_audio = mx.load(str(root / "audio.safetensors"))
        feature_difference = mx.abs(
            features.astype(mx.float32)
            - actual_audio["features"].astype(mx.bfloat16).astype(mx.float32)
        )
        audio_difference = mx.abs(
            expected_audio.astype(mx.float32) - actual_audio["audio"].astype(mx.float32)
        )
        print(
            json.dumps(
                {
                    "audio": str(audio_path),
                    "feature_max_abs": mx.max(feature_difference).item(),
                    "audio_max_abs": mx.max(audio_difference).item(),
                },
                indent=2,
            ),
            flush=True,
        )
        np.testing.assert_allclose(
            np.array(features.astype(mx.float32)),
            np.array(actual_audio["features"].astype(mx.bfloat16).astype(mx.float32)),
            atol=0.0,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.array(expected_audio.astype(mx.float32)),
            np.array(actual_audio["audio"].astype(mx.float32)),
            atol=0.0,
            rtol=0.0,
        )
        prompt_token_ids = model.tokenizer.encode(
            ASR_INSTRUCTION + model.audio_tokens[0], add_special_tokens=False
        )
        prompt_token_ids += [model.config.audio_token_id] * expected_audio.shape[1]
        prompt_token_ids += model.tokenizer.encode(
            model.audio_tokens[1] + "English\t", add_special_tokens=False
        )
        _, embeddings = model.inject_audio(prompt_token_ids, expected_audio)
        mx.save_safetensors(str(root / "input.safetensors"), {"embeddings": embeddings})
        for chunk_size in dict.fromkeys((embeddings.shape[1], 31, 1)):
            cache = model.make_cache()
            hidden_chunks: list[mx.array] = []
            for start in range(0, embeddings.shape[1], chunk_size):
                hidden_states = model.model(
                    inputs_embeds=embeddings[:, start : start + chunk_size], cache=cache
                )
                mx.eval(hidden_states, [state.state for state in cache])
                hidden_chunks.append(hidden_states)
            expected = {
                "hidden": mx.concatenate(hidden_chunks, axis=1),
                "logits": model.lm_head(hidden_chunks[-1][:, -1:]).astype(mx.float32),
            }
            for layer, state in enumerate(cache):
                if model.config.text_config.uses_mla(layer):
                    expected[f"{layer}.keys"], expected[f"{layer}.values"] = state.state
                else:
                    expected[f"{layer}.recurrent"] = state[3]
                    expected.update(
                        {f"{layer}.conv.{index}": state[index] for index in range(3)}
                    )
            mx.eval(list(expected.values()))
            subprocess.run(
                [
                    str(runtime_bin / "ming3_asr_probe"),
                    "forward",
                    str(root / "input.safetensors"),
                    str(root / "forward.safetensors"),
                    str(model_path),
                    str(chunk_size),
                ],
                check=True,
            )
            actual = mx.load(str(root / "forward.safetensors"))
            differences = {
                name: mx.max(
                    mx.abs(value.astype(mx.float32) - actual[name].astype(mx.float32))
                ).item()
                for name, value in expected.items()
            }
            print(
                json.dumps(
                    {
                        "audio": str(audio_path),
                        "chunk_size": chunk_size,
                        "max_abs": differences,
                    },
                    indent=2,
                ),
                flush=True,
            )
            for name, value in expected.items():
                np.testing.assert_allclose(
                    np.array(value.astype(mx.float32)),
                    np.array(actual[name].astype(mx.float32)),
                    atol=0.0,
                    rtol=0.0,
                    err_msg=f"chunk={chunk_size}, tensor={name}",
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-bin", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--audio", type=Path, action="append", required=True)
    arguments = parser.parse_args()
    model: Model = load(str(arguments.model_path), strict=True)
    for audio_path in arguments.audio:
        compare_audio(model, arguments.model_path, arguments.runtime_bin, audio_path)
    with tempfile.TemporaryDirectory(prefix="ming-minimum-audio-") as directory:
        minimum_audio_path = Path(directory) / "201-samples.wav"
        with wave.open(str(minimum_audio_path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(np.linspace(-12000, 16000, 201).astype("<i2").tobytes())
        compare_audio(
            model, arguments.model_path, arguments.runtime_bin, minimum_audio_path
        )


if __name__ == "__main__":
    main()
