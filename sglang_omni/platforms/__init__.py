import os
import pkgutil

import torch
from sglang.srt import platforms as srt_platforms
from sglang.srt.platforms.interface import SRTPlatform

from sglang_omni.platforms.cpu import CPUOmniPlatform
from sglang_omni.platforms.cuda import CUDAOmniPlatform, ROCMOmniPlatform
from sglang_omni.platforms.interface import OmniPlatform
from sglang_omni.platforms.npu import NPUOmniPlatform


def _is_npu_available() -> bool:
    try:
        npu = torch.npu
    except AttributeError:
        return False
    return bool(npu.is_available())


def _load_platform_class(qualname: str) -> type[OmniPlatform]:
    cls = pkgutil.resolve_name(qualname)
    if not isinstance(cls, type):
        raise TypeError(f"Expected a platform class, got {type(cls)}: {qualname}")
    if issubclass(cls, OmniPlatform):
        return cls
    if not issubclass(cls, SRTPlatform):
        raise TypeError(f"Expected an SRTPlatform subclass: {qualname}")
    return type(
        f"Omni{cls.__name__}",
        (cls, OmniPlatform),
        {"_omni_platform_qualname": qualname},
    )


def _as_omni_platform(platform: SRTPlatform) -> OmniPlatform:
    if platform.is_cuda():
        return CUDAOmniPlatform()
    if platform.is_rocm():
        return ROCMOmniPlatform()
    if platform.is_cpu():
        return CPUOmniPlatform()
    if type(platform) is SRTPlatform and _is_npu_available():
        return NPUOmniPlatform()
    qualname = f"{type(platform).__module__}.{type(platform).__qualname__}"
    return _load_platform_class(qualname)()


def _resolve_platform() -> OmniPlatform:
    return _as_omni_platform(srt_platforms.current_platform)


def get_platform_spec(platform: OmniPlatform) -> str:
    if platform._omni_platform_qualname is not None:
        return platform._omni_platform_qualname
    return f"{type(platform).__module__}.{type(platform).__qualname__}"


platform_spec = os.environ.get("SGLANG_OMNI_PLATFORM_SPEC")
current_platform = (
    _load_platform_class(platform_spec)() if platform_spec else _resolve_platform()
)
