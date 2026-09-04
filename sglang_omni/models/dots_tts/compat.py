# SPDX-License-Identifier: Apache-2.0
"""Import dots.tts on the pinned torch stack."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
import threading
from types import ModuleType

_IMPORT_LOCK = threading.Lock()


def import_dots_tts() -> ModuleType:
    """Import the dots_tts package past its torch/torchaudio version check.

    dots.tts refuses to import unless the torch and torchaudio distributions
    share a minor version (dots_tts/__init__.py, _check_torch_install, which
    reads both through importlib.metadata.version). sglang 0.5.18 pins torch
    2.13.0 with torchaudio 2.11.0, and omni pins the same pair: torchaudio
    2.11.0 is its last release, and its release note states it is compatible
    with torch 2.11 and with future torch versions. The check is stricter than
    that supported pair, and the torchaudio surface dots.tts uses
    (functional.resample, compliance.kaldi fbank, transforms.Resample) is
    torch ops. For the single package import, the torchaudio distribution is
    reported at torch's version and the reader is restored before returning.
    The replacement is process-global for that moment, so every omni import
    of dots_tts goes through this function and nothing else may read
    distribution versions concurrently with the first one. Remove this once
    torchaudio ships a release matching torch's minor or dots.tts drops the
    equality check.
    """
    with _IMPORT_LOCK:
        if "dots_tts" in sys.modules:
            return importlib.import_module("dots_tts")
        reader = importlib.metadata.version

        def bridged(distribution_name: str) -> str:
            if distribution_name == "torchaudio":
                return reader("torch")
            return reader(distribution_name)

        importlib.metadata.version = bridged
        try:
            return importlib.import_module("dots_tts")
        finally:
            importlib.metadata.version = reader
