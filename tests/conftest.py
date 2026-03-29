# SPDX-License-Identifier: Apache-2.0
"""Root-level pytest configuration.

Stubs out optional heavy dependencies that are not installed in unit-test
environments so that imports like `sglang_omni.models.minicpm_v.io` succeed
without requiring a full production dependency set.

Modules stubbed here:
- av             : PyAV video codec (requires ffmpeg build)
- librosa        : Audio analysis library
- qwen_vl_utils  : Qwen VL utilities (video preprocessing)
- mooncake       : Mooncake transfer engine (proprietary)
- nixl           : NixL relay (optional)
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types


def _stub_module(name: str, **attrs) -> types.ModuleType:
    """Create and register a minimal stub module with a valid spec in sys.modules."""
    mod = types.ModuleType(name)
    # Create a minimal spec so importlib.util.find_spec() doesn't raise ValueError
    spec = importlib.machinery.ModuleSpec(name, loader=None)
    mod.__spec__ = spec
    mod.__loader__ = None
    mod.__package__ = name.rsplit(".", 1)[0] if "." in name else name
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _maybe_stub(name: str, **attrs) -> None:
    """Stub a module only if it is not already importable."""
    if name in sys.modules:
        return
    try:
        __import__(name)
    except ImportError:
        _stub_module(name, **attrs)


# ---------------------------------------------------------------------------
# av — PyAV video / audio I/O
# torchvision.io.video tries: av.logging, av.video.frame.VideoFrame,
# av.FFmpegError (new in av 14), av.AVError (legacy)
# ---------------------------------------------------------------------------

class _AVLogging:
    ERROR = 16
    WARNING = 24
    INFO = 32
    VERBOSE = 40
    DEBUG = 48

    @staticmethod
    def set_level(level):
        pass


class _VideoFrame:
    """Minimal stub for av.video.frame.VideoFrame."""
    pict_type = None


class FFmpegError(Exception):
    """Stub for av.FFmpegError (added in av 14)."""


class AVError(Exception):
    """Stub for av.AVError (legacy name)."""


_av_video_frame = _stub_module("av.video.frame", VideoFrame=_VideoFrame)
_av_video = _stub_module("av.video", frame=_av_video_frame)
_av_audio = _stub_module("av.audio")
_av_container = _stub_module("av.container")
_maybe_stub(
    "av",
    logging=_AVLogging,
    video=_av_video,
    audio=_av_audio,
    container=_av_container,
    FFmpegError=FFmpegError,
    AVError=AVError,
    open=lambda *a, **kw: None,
)

# ---------------------------------------------------------------------------
# librosa — audio analysis (used in sglang_omni/preprocessing/video.py)
# ---------------------------------------------------------------------------

def _librosa_load(path, sr=None, **kwargs):
    import numpy as np
    return np.zeros(1000, dtype=np.float32), sr or 16000


_maybe_stub(
    "librosa",
    load=_librosa_load,
)

# ---------------------------------------------------------------------------
# qwen_vl_utils — Qwen VL preprocessing helpers
# (used at module level in sglang_omni/preprocessing/video.py)
# ---------------------------------------------------------------------------

class _QwenVisionProcess:
    """Minimal stub for qwen_vl_utils.vision_process."""
    VIDEO_MIN_PIXELS = 128 * 28 * 28
    VIDEO_TOTAL_PIXELS = 512 * 28 * 28
    VIDEO_MAX_PIXELS = 768 * 28 * 28
    IMAGE_FACTOR = 28
    FRAME_FACTOR = 2
    VIDEO_READER_BACKENDS: dict = {}

    @staticmethod
    def get_video_reader_backend():
        return "torchvision"

    @staticmethod
    def smart_resize(height, width, factor=28, min_pixels=None, max_pixels=None):
        return height, width


_qwen_vision_process = _stub_module(
    "qwen_vl_utils.vision_process",
    **{k: v for k, v in vars(_QwenVisionProcess).items() if not k.startswith("__")},
)
_maybe_stub(
    "qwen_vl_utils",
    vision_process=_qwen_vision_process,
)

# ---------------------------------------------------------------------------
# mooncake / nixl — relay back-ends (imported as sub-packages)
# ---------------------------------------------------------------------------

class _TransferEngine:
    pass


class _NixlAgent:
    pass


class _NixlAgentConfig:
    def __init__(self, **kwargs):
        pass


_mooncake = _stub_module("mooncake")
_mooncake_engine = _stub_module(
    "mooncake.engine",
    TransferEngine=_TransferEngine,
    TransferNotify=type("TransferNotify", (), {}),
    TransferOpcode=type("TransferOpcode", (), {}),
)
_mooncake.__path__ = []  # type: ignore[attr-defined]
_mooncake_engine.__path__ = []  # type: ignore[attr-defined]

_nixl = _stub_module("nixl")
_nixl_api = _stub_module(
    "nixl._api",
    nixl_agent=_NixlAgent,
    nixl_agent_config=_NixlAgentConfig,
)
_nixl.__path__ = []  # type: ignore[attr-defined]
