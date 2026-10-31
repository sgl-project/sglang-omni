# SPDX-License-Identifier: Apache-2.0
"""Prepares the pinned checkpoints of the native models beside Qwen3-ASR.

    python provision_models.py <data_root> [--model REPO ...]

Every file is checked against the SHA-256 in model_pins.json. A checkpoint
lands in <data_root>/models/<org>_<name>; a Whisper checkpoint also gets the
tokenizer files of its openai/whisper-* repo in that directory, where Voxt
puts them, and Fun-CosyVoice3 gets its S3 speech tokenizer's checkpoint in
speech_tokenizer/. The corpus comes from provision.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NotRequired

from provision import CI_DIRECTORY, FileSource, provision_files, provision_model


class TokenizerSource(FileSource):
    repo: str
    # Where under the model directory its files go; the model directory itself
    # when absent.
    subdirectory: NotRequired[str]


class ModelPin(FileSource):
    tokenizer: NotRequired[TokenizerSource]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument(
        "--model", action="append", help="repo to provision (default: all pinned)"
    )
    parser.add_argument("--exclude-model", action="append", default=[])
    arguments = parser.parse_args()
    pins: dict[str, ModelPin] = json.loads(
        (CI_DIRECTORY / "model_pins.json").read_text()
    )
    for repo in arguments.model or list(pins):
        if repo in arguments.exclude_model:
            continue
        else:
            pass
        pin = pins[repo]
        model_directory = provision_model(arguments.data_root, repo, pin)
        if "tokenizer" in pin:
            tokenizer = pin["tokenizer"]
            provision_files(
                tokenizer["repo"],
                tokenizer,
                model_directory / tokenizer.get("subdirectory", ""),
            )
        else:
            pass
        print(f"model {repo}: {model_directory}", flush=True)


if __name__ == "__main__":
    main()
