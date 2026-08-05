from sglang_omni.platforms import PlatformEnum, ResolvedPlatformSpec

CUDA_PLATFORM_SPEC = ResolvedPlatformSpec(PlatformEnum.CUDA, "cuda", "nccl")
CPU_PLATFORM_SPEC = ResolvedPlatformSpec(PlatformEnum.CPU, "cpu", "gloo")
