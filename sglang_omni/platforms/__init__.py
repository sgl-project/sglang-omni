import os
import pkgutil
import platform as host_platform

import torch
from sglang.srt import platforms as srt_platforms
from sglang.srt.platforms.interface import SRTPlatform

from sglang_omni.platforms.apple import AppleOmniPlatform
from sglang_omni.platforms.cpu import CPUOmniPlatform
from sglang_omni.platforms.cuda import CUDAOmniPlatform
from sglang_omni.platforms.interface import OmniPlatform
from sglang_omni.platforms.musa import MUSAOmniPlatform
from sglang_omni.platforms.npu import NPUOmniPlatform
from sglang_omni.platforms.rocm import ROCMOmniPlatform
from sglang_omni.platforms.xpu import XPUOmniPlatform


def is_musa_available() -> bool:
    try:
        musa = torch.musa
    except AttributeError:
        return False
    return bool(musa.is_available())


def is_apple_silicon_mps_available() -> bool:
    return (
        host_platform.system() == "Darwin"
        and host_platform.machine() == "arm64"
        and bool(torch.backends.mps.is_available())
    )


def load_platform_class(qualname: str) -> type[OmniPlatform]:
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


platform_spec = os.environ.get("SGLANG_OMNI_PLATFORM_SPEC")
current_platform = (
    load_platform_class(platform_spec)() if platform_spec else resolve_platform()
)
