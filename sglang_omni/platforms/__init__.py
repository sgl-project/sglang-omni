"""The platform this process runs on, resolved on first use.

``current_platform`` is looked up lazily, so importing this package costs no
torch or SGLang import. A process that never needs the platform, such as the
HTTP server process of a pipeline whose stages run elsewhere, never resolves
it, and its stage processes then resolve it themselves under the environment
they inherit.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import platform as host_platform
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.platforms.interface import SRTPlatform

    from sglang_omni.platforms.interface import OmniPlatform
else:
    pass

_resolution_lock = threading.Lock()
_resolved_platform: OmniPlatform | None = None

# Classes this package exports, each imported from its module on first use.
EXPORTED_CLASSES = {
    "AppleOmniPlatform": "sglang_omni.platforms.apple",
    "CPUOmniPlatform": "sglang_omni.platforms.cpu",
    "CUDAOmniPlatform": "sglang_omni.platforms.cuda",
    "MUSAOmniPlatform": "sglang_omni.platforms.musa",
    "NPUOmniPlatform": "sglang_omni.platforms.npu",
    "OmniPlatform": "sglang_omni.platforms.interface",
    "ROCMOmniPlatform": "sglang_omni.platforms.rocm",
    "XPUOmniPlatform": "sglang_omni.platforms.xpu",
}


def is_musa_available() -> bool:
    import torch

    try:
        musa = torch.musa
    except AttributeError:
        return False
    return bool(musa.is_available())


def is_apple_silicon_mps_available() -> bool:
    import torch

    return (
        host_platform.system() == "Darwin"
        and host_platform.machine() == "arm64"
        and bool(torch.backends.mps.is_available())
    )


def load_platform_class(qualname: str) -> type[OmniPlatform]:
    from sglang.srt.platforms.interface import SRTPlatform

    from sglang_omni.platforms.interface import OmniPlatform

    cls = pkgutil.resolve_name(qualname)
    if not isinstance(cls, type):
        raise TypeError(f"Expected a platform class, got {type(cls)}: {qualname}")
    else:
        pass
    if issubclass(cls, OmniPlatform):
        return cls
    else:
        pass
    if not issubclass(cls, SRTPlatform):
        raise TypeError(f"Expected an SRTPlatform subclass: {qualname}")
    else:
        pass
    return type(
        f"Omni{cls.__name__}",
        (cls, OmniPlatform),
        {"_omni_platform_qualname": qualname},
    )


def as_omni_platform(platform: SRTPlatform) -> OmniPlatform:
    from sglang.srt.platforms.interface import SRTPlatform

    from sglang_omni.platforms.apple import AppleOmniPlatform
    from sglang_omni.platforms.cpu import CPUOmniPlatform
    from sglang_omni.platforms.cuda import CUDAOmniPlatform
    from sglang_omni.platforms.musa import MUSAOmniPlatform
    from sglang_omni.platforms.npu import NPUOmniPlatform
    from sglang_omni.platforms.rocm import ROCMOmniPlatform
    from sglang_omni.platforms.xpu import XPUOmniPlatform

    if platform.is_cuda():
        return CUDAOmniPlatform()
    else:
        pass
    if platform.is_rocm():
        return ROCMOmniPlatform()
    else:
        pass
    if platform.is_cpu():
        return CPUOmniPlatform()
    else:
        pass
    if platform.is_xpu():
        return XPUOmniPlatform()
    else:
        pass
    if platform.is_npu():
        return NPUOmniPlatform()
    else:
        pass
    # Note (yexiaodong): Explicit CPU and registered platform selections must
    # win. SGLang otherwise leaves Apple Metal on its generic platform.
    if type(platform) is SRTPlatform and is_apple_silicon_mps_available():
        return AppleOmniPlatform()
    else:
        pass
    if type(platform) is SRTPlatform and is_musa_available():
        return MUSAOmniPlatform()
    else:
        pass
    qualname = f"{type(platform).__module__}.{type(platform).__qualname__}"
    return load_platform_class(qualname)()


def resolve_platform() -> OmniPlatform:
    from sglang.srt import platforms as srt_platforms

    return as_omni_platform(srt_platforms.current_platform)


def get_platform_spec(platform: OmniPlatform) -> str:
    if (
        platform._omni_platform_qualname is not None
    ):  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        return (
            platform._omni_platform_qualname
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
    else:
        pass
    return f"{type(platform).__module__}.{type(platform).__qualname__}"


def get_current_platform() -> OmniPlatform:
    """The process's platform, from the pinned spec or detected, once."""
    global _resolved_platform
    with _resolution_lock:
        if _resolved_platform is None:
            spec = os.environ.get("SGLANG_OMNI_PLATFORM_SPEC")
            _resolved_platform = (
                load_platform_class(spec)() if spec else resolve_platform()
            )
        else:
            pass
        return _resolved_platform


def resolved_platform() -> OmniPlatform | None:
    """The platform if this process has resolved it, without resolving it."""
    return _resolved_platform


def __getattr__(name: str):
    if name == "current_platform":
        return get_current_platform()
    else:
        pass
    if name in EXPORTED_CLASSES:
        return getattr(importlib.import_module(EXPORTED_CLASSES[name]), name)
    else:
        pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
