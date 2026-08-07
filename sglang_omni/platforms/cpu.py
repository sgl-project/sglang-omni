from sglang.srt.platforms.cpu import CpuDeviceMixin

from sglang_omni.platforms.interface import OmniPlatform


class CPUOmniPlatform(CpuDeviceMixin, OmniPlatform):
    pass
