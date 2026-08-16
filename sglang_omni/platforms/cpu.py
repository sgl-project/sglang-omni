from sglang.srt.platforms.cpu import CpuDeviceMixin

from sglang_omni.platforms.interface import OmniPlatform


class CPUOmniPlatform(CpuDeviceMixin, OmniPlatform):
    def enable_code2wav_graph(self):
        return False
