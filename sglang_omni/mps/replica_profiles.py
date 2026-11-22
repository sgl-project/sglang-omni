# SPDX-License-Identifier: Apache-2.0
"""Replica counts measured on one GPU model, served by default where that GPU is detected."""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from sglang_omni.config.patch import ConfigPatch
from sglang_omni.config.path import ConfigPath
from sglang_omni.config.schema import PipelineConfig, StageConfig, stage_process_name
from sglang_omni.mps.manager import MpsControlClient, MpsManager
from sglang_omni.mps.runtime import (
    UNSUPPORTED_PROCESS_ENV,
    MpsDeviceInfo,
    default_state_root,
)
from sglang_omni.mps.state import MpsGpuPaths
from sglang_omni.platforms import current_platform

ProfileValue = int | float | str | list[int]

BYTES_PER_MIB = 1 << 20
BYTES_PER_GIB = 1 << 30
# note (Yuanfan Chen): drivers report a slightly different total for one card model.
GPU_MEMORY_TOLERANCE = 0.02
# A source that writes one of these sizes or places the replicas itself.
DEPLOYMENT_STAGE_FIELDS = frozenset(
    {"gpu_memory_fraction", "process", "total_reserve_bytes", "tp_size"}
)
DEPLOYMENT_ENGINE_FIELDS = frozenset(
    {"kv_cache_bytes", "max_total_tokens", "mem_fraction_static"}
)


@dataclass(frozen=True, kw_only=True)
class MeasuredGpu:
    """A GPU model by its NVML name, with the total memory NVML reported on the measured card."""

    name: str
    total_memory_mib: int


@dataclass(frozen=True, kw_only=True)
class ReplicaProfile:
    """How many replicas of one model a GPU model serves best, and what each replica is given."""

    config_cls: str
    model_path: str
    gpu: MeasuredGpu
    num_replicas: int
    replicated_processes: tuple[str, ...]
    settings: Mapping[str, int | float | str]


@dataclass(frozen=True, kw_only=True)
class ReplicaProfileDecision:
    """The writes a launch takes from a measured profile, and the log line that explains them."""

    profile: ReplicaProfile | None
    writes: dict[str, ProfileValue]
    message: str | None


B200 = MeasuredGpu(name="NVIDIA B200", total_memory_mib=183359)
RTX_4090_D = MeasuredGpu(name="NVIDIA GeForce RTX 4090 D", total_memory_mib=24564)

# note (Yuanfan Chen): measured with sglang-omni 8848e500. Each count served the most audio within a p99 latency of 0.5 s and keeps at least 88% of the best count's audio at 1 s and 3 s; docs/basic_usage/mps_dp.md has the numbers.
REPLICA_PROFILES: tuple[ReplicaProfile, ...] = (
    ReplicaProfile(
        config_cls="Qwen3TTSPipelineConfig",
        model_path="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        gpu=B200,
        num_replicas=9,
        replicated_processes=("pipeline", "vocoder"),
        settings={
            "stages.tts_engine.gpu_memory_fraction": 0.0667,
            "stages.tts_engine.engine.mem_fraction_static": 0.911,
            "stages.tts_engine.engine.max_running_requests": 16,
            "stages.tts_engine.engine.max_queued_requests": 16,
            "stages.tts_engine.engine.cuda_graph_max_bs": 32,
            "stages.tts_engine.engine.max_total_tokens": 32768,
            "stages.vocoder.gpu_memory_fraction": 0.0178,
            "stages.vocoder.process": "vocoder",
        },
    ),
    ReplicaProfile(
        config_cls="Qwen3TTSPipelineConfig",
        model_path="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        gpu=B200,
        num_replicas=7,
        replicated_processes=("pipeline", "vocoder"),
        settings={
            "stages.tts_engine.gpu_memory_fraction": 0.0667,
            "stages.tts_engine.engine.mem_fraction_static": 0.911,
            "stages.tts_engine.engine.max_running_requests": 16,
            "stages.tts_engine.engine.max_queued_requests": 16,
            "stages.tts_engine.engine.cuda_graph_max_bs": 32,
            "stages.tts_engine.engine.max_total_tokens": 32768,
            "stages.vocoder.gpu_memory_fraction": 0.0178,
            "stages.vocoder.process": "vocoder",
        },
    ),
    ReplicaProfile(
        config_cls="MossTTSLocalPipelineConfig",
        model_path="OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
        gpu=B200,
        num_replicas=5,
        replicated_processes=("pipeline", "vocoder"),
        settings={
            "stages.preprocessing.gpu_memory_fraction": 0.0222,
            "stages.tts_engine.gpu_memory_fraction": 0.1068,
            "stages.tts_engine.engine.mem_fraction_static": 0.0979,
            "stages.tts_engine.engine.max_running_requests": 16,
            "stages.tts_engine.engine.max_queued_requests": 16,
            "stages.tts_engine.engine.cuda_graph_max_bs": 32,
            "stages.tts_engine.engine.max_total_tokens": 16384,
            # note (Yuanfan Chen): the default trtllm_mha prefill has no full graph for this engine on a B200.
            "stages.tts_engine.engine.prefill_attention_backend": "flashinfer",
            "stages.tts_engine.engine.cuda_graph_backend_prefill": "full",
            "stages.vocoder.gpu_memory_fraction": 0.0623,
        },
    ),
    ReplicaProfile(
        config_cls="MossTTSPipelineConfig",
        model_path="OpenMOSS-Team/MOSS-TTS-v1.5",
        gpu=B200,
        num_replicas=2,
        replicated_processes=("pipeline", "vocoder"),
        settings={
            "stages.preprocessing.gpu_memory_fraction": 0.0222,
            "stages.tts_engine.gpu_memory_fraction": 0.1201,
            "stages.tts_engine.engine.mem_fraction_static": 0.1112,
            "stages.tts_engine.engine.max_running_requests": 16,
            "stages.tts_engine.engine.max_queued_requests": 16,
            "stages.tts_engine.engine.cuda_graph_max_bs": 32,
            "stages.tts_engine.engine.max_total_tokens": 16384,
            "stages.vocoder.gpu_memory_fraction": 0.0534,
        },
    ),
    ReplicaProfile(
        config_cls="Ming3TTSPipelineConfig",
        model_path="inclusionAI/Ming-3.0-tts-0.5B",
        gpu=B200,
        num_replicas=2,
        replicated_processes=("preprocessing", "tts_engine", "vocoder"),
        settings={
            "stages.preprocessing.gpu_memory_fraction": 0.0133,
            "stages.tts_engine.gpu_memory_fraction": 0.0311,
            "stages.tts_engine.engine.mem_fraction_static": 0.911,
            "stages.tts_engine.engine.max_running_requests": 16,
            "stages.tts_engine.engine.max_queued_requests": 16,
            "stages.tts_engine.engine.max_total_tokens": 32768,
            "stages.vocoder.gpu_memory_fraction": 0.0178,
        },
    ),
    ReplicaProfile(
        config_cls="Drama3PipelineConfig",
        model_path="fishaudio/drama-3",
        gpu=B200,
        num_replicas=2,
        replicated_processes=("preprocessing", "pipeline"),
        settings={
            "stages.tts_engine.gpu_memory_fraction": 0.2891,
            "stages.tts_engine.engine.max_running_requests": 16,
            "stages.tts_engine.engine.max_queued_requests": 16,
            "stages.tts_engine.engine.cuda_graph_max_bs": 32,
            "stages.vocoder.gpu_memory_fraction": 0.0667,
        },
    ),
    ReplicaProfile(
        config_cls="WhisperASRPipelineConfig",
        model_path="openai/whisper-large-v3",
        gpu=B200,
        num_replicas=6,
        replicated_processes=("asr",),
        settings={
            "stages.asr.gpu_memory_fraction": 0.0801,
            "stages.asr.engine.mem_fraction_static": 0.9733,
            "stages.asr.engine.max_running_requests": 16,
            "stages.asr.engine.max_queued_requests": 16,
            "stages.asr.engine.cuda_graph_max_bs": 32,
            "stages.asr.engine.max_total_tokens": 16384,
        },
    ),
    ReplicaProfile(
        config_cls="Qwen3ASRPipelineConfig",
        model_path="Qwen/Qwen3-ASR-1.7B",
        gpu=B200,
        num_replicas=5,
        replicated_processes=("asr",),
        settings={
            "stages.asr.gpu_memory_fraction": 0.0375,
            "stages.asr.engine.mem_fraction_static": 0.9733,
            "stages.asr.engine.max_running_requests": 16,
            "stages.asr.engine.max_queued_requests": 16,
            "stages.asr.engine.cuda_graph_max_bs": 32,
            "stages.asr.engine.max_total_tokens": 16384,
        },
    ),
    ReplicaProfile(
        config_cls="MossTranscribeDiarizePipelineConfig",
        model_path="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        gpu=B200,
        num_replicas=7,
        replicated_processes=("asr",),
        settings={
            "stages.asr.gpu_memory_fraction": 0.0375,
            "stages.asr.engine.mem_fraction_static": 0.9733,
            "stages.asr.engine.max_running_requests": 16,
            "stages.asr.engine.max_queued_requests": 16,
            "stages.asr.engine.cuda_graph_max_bs": 32,
            "stages.asr.engine.max_total_tokens": 16384,
        },
    ),
    ReplicaProfile(
        config_cls="Nemotron3_5ASRPipelineConfig",
        model_path="nvidia/nemotron-3.5-asr-streaming-0.6b",
        gpu=B200,
        num_replicas=8,
        replicated_processes=("asr",),
        settings={
            "stages.asr.gpu_memory_fraction": 0.0623,
            "stages.asr.factory.max_batch_size": 16,
            "stages.asr.factory.session_max_concurrency": 16,
            # note (Yuanfan Chen): under the runtime's default thread plan, which sets only OMP_NUM_THREADS, 12 replicas served 31 requests/s against 239 with these caps.
            "stages.asr.env.OMP_NUM_THREADS": "4",
            "stages.asr.env.MKL_NUM_THREADS": "4",
            "stages.asr.env.OPENBLAS_NUM_THREADS": "4",
        },
    ),
    # note (Jiaxin Deng): measured with sglang-omni a385eac6; on this GPU the engine's own 64 running requests and no queue cap beat the 16 and 16 above.
    ReplicaProfile(
        config_cls="MossTranscribeDiarizePipelineConfig",
        model_path="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        gpu=RTX_4090_D,
        num_replicas=2,
        replicated_processes=("asr",),
        settings={
            "stages.asr.gpu_memory_fraction": 0.2799,
            "stages.asr.engine.mem_fraction_static": 0.8008,
            "stages.asr.engine.max_running_requests": 64,
            "stages.asr.engine.cuda_graph_max_bs": 64,
            "stages.asr.engine.max_total_tokens": 32768,
        },
    ),
)


def placed_gpu_ids(stage: StageConfig) -> list[int]:
    """Every GPU a stage occupies: its scalar or list placement, or the runtime GPUs it owns."""
    if stage.runtime_gpu_ids is not None:
        return list(stage.runtime_gpu_ids)
    elif isinstance(stage.gpu, int):
        return [stage.gpu]
    elif stage.gpu is None:
        return []
    else:
        return list(stage.gpu)


def not_applied(config: PipelineConfig, reason: str) -> ReplicaProfileDecision:
    return ReplicaProfileDecision(
        profile=None,
        writes={},
        message=f"Measured replica profile for {config.model_path} not applied: {reason}",
    )


def decide_replica_profile(
    config: PipelineConfig,
    *,
    user_patches: Iterable[ConfigPatch],
    device_info: MpsDeviceInfo,
    mps_client: MpsControlClient,
) -> ReplicaProfileDecision:
    """Return the writes of this model's profile for the detected GPU, or why the launch keeps its own config.

    It applies when no source sizes or places the replicas, the pipeline sits on one idle GPU of the measured model, and MPS can start there: no state left by an earlier run blocks a new launch from joining the GPU's daemon.
    """
    model_path_parts = Path(config.model_path).parts
    candidates = [
        profile
        for profile in REPLICA_PROFILES
        if profile.config_cls == type(config).__name__
        and (
            profile.model_path == config.model_path.partition("@")[0]
            or "models--" + profile.model_path.replace("/", "--") in model_path_parts
        )
    ]
    if not candidates:
        return ReplicaProfileDecision(profile=None, writes={}, message=None)
    else:
        pass

    user_patches = list(user_patches)
    profile_paths = {path for profile in candidates for path in profile.settings}
    for patch in user_patches:
        segments = patch.key.split(".")
        if (
            segments[0] == "processes"
            or (segments[0] == "mps" and patch.value == "off")
            or (
                segments[0] == "stages"
                and len(segments) == 3
                and segments[2] in DEPLOYMENT_STAGE_FIELDS
            )
            or (
                segments[0] == "stages"
                and len(segments) == 4
                and segments[2] == "engine"
                and segments[3] in DEPLOYMENT_ENGINE_FIELDS
            )
            or any(
                patch.key == path
                or path.startswith(f"{patch.key}.")
                or patch.key.startswith(f"{path}.")
                for path in profile_paths
            )
        ):
            return not_applied(
                config, f"{patch.key} is set by {patch.source.describe()}"
            )
        else:
            pass

    gpu_ids = sorted(
        {gpu_id for stage in config.stages for gpu_id in placed_gpu_ids(stage)}
    )
    weight_share_env = os.environ.get("SGLANG_OMNI_WEIGHT_SHARE", "").strip()
    stage_env_overrides = sorted(
        {
            name
            for stage in config.stages
            for name in UNSUPPORTED_PROCESS_ENV
            if name in config.resolved_stage_env_defaults(stage.name)
        }
    )
    if any(stage.tp_size > 1 for stage in config.stages):
        return not_applied(config, "a stage runs tensor parallel")
    elif len(gpu_ids) != 1:
        return not_applied(config, f"its GPU stages run on GPUs {gpu_ids}")
    elif not current_platform.is_cuda():
        return not_applied(config, "the platform is not NVIDIA CUDA")
    elif shutil.which("nvidia-cuda-mps-control") is None:
        return not_applied(config, "nvidia-cuda-mps-control is not on PATH")
    elif "CUDA_MPS_PIPE_DIRECTORY" in os.environ:
        return not_applied(config, "the environment sets CUDA_MPS_PIPE_DIRECTORY")
    elif weight_share_env:
        return not_applied(config, "the environment sets SGLANG_OMNI_WEIGHT_SHARE")
    elif stage_env_overrides:
        return not_applied(config, f"a stage environment sets {stage_env_overrides}")
    else:
        pass

    gpu_id = gpu_ids[0]
    device = device_info.inspect([gpu_id])[gpu_id]
    if device.unsupported_reason is not None:
        return not_applied(
            config, f"GPU {gpu_id} cannot run MPS: {device.unsupported_reason}"
        )
    else:
        pass
    assert (
        device.gpu_uuid is not None
        and device.name is not None
        and device.total_memory_bytes is not None
        and device.free_memory_bytes is not None
    ), f"GPU {gpu_id} passed the MPS checks without a UUID, an NVML name and memory"
    total_memory_mib = device.total_memory_bytes // BYTES_PER_MIB
    matching = [
        profile
        for profile in candidates
        if profile.gpu.name == device.name
        and abs(total_memory_mib - profile.gpu.total_memory_mib)
        <= GPU_MEMORY_TOLERANCE * profile.gpu.total_memory_mib
    ]
    if not matching:
        measured_gpus = sorted(
            {f"{p.gpu.name} ({p.gpu.total_memory_mib} MiB)" for p in candidates}
        )
        return not_applied(
            config,
            f"GPU {gpu_id} is {device.name} ({total_memory_mib} MiB); "
            f"replica counts were measured on {', '.join(measured_gpus)}",
        )
    else:
        pass
    (profile,) = matching

    profiled_data = config.model_dump()
    for path_text, value in profile.settings.items():
        ConfigPath.parse(path_text, type(config)).write(profiled_data, value)
    profiled = type(config)(**profiled_data)
    process_names = {stage_process_name(stage) for stage in profiled.stages}
    assert set(profile.replicated_processes) <= process_names, (
        f"{profile.model_path} profile replicates {profile.replicated_processes}, "
        f"but the pipeline's processes are {sorted(process_names)}"
    )
    gpu_process_names = {
        stage_process_name(stage) for stage in profiled.stages if stage.gpu is not None
    }
    cpu_only_processes = [
        name for name in profile.replicated_processes if name not in gpu_process_names
    ]
    if cpu_only_processes:
        return not_applied(
            config,
            f"the profile replicates processes without a GPU stage: {cpu_only_processes}",
        )
    else:
        pass
    # note (Yuanfan Chen): a serve killed with SIGKILL leaves owner leases no join accepts; probe the state the same way the launch would join it.
    state_paths = MpsGpuPaths(
        state_root=default_state_root([device.gpu_uuid]), gpu_uuid=device.gpu_uuid
    )
    if (
        state_paths.state_dir.exists()
        and not MpsManager(
            paths=state_paths, gpu_uuid=device.gpu_uuid, client=mps_client
        )
        .inspect_existing_state()
        .joinable
    ):
        return not_applied(
            config,
            f"GPU {gpu_id} holds MPS state from an earlier run that a new launch "
            f"cannot join, in {state_paths.state_dir}",
        )
    else:
        pass
    reserved_bytes = sum(
        (
            (stage.gpu_memory_fraction or 0.0) * device.total_memory_bytes
            + (stage.total_reserve_bytes or 0)
        )
        * (
            profile.num_replicas
            if stage_process_name(stage) in profile.replicated_processes
            else 1
        )
        for stage in profiled.stages
        if stage.gpu is not None
    )
    if device.free_memory_bytes < reserved_bytes:
        return not_applied(
            config,
            f"GPU {gpu_id} has {device.free_memory_bytes / BYTES_PER_GIB:.1f} GiB "
            f"free and {profile.num_replicas} replicas reserve "
            f"{reserved_bytes / BYTES_PER_GIB:.1f} GiB",
        )
    else:
        pass

    writes: dict[str, ProfileValue] = dict(profile.settings)
    for process_name in profile.replicated_processes:
        writes[f"processes.{process_name}.num_replicas"] = profile.num_replicas
        writes[f"processes.{process_name}.replica_devices"] = [
            gpu_id
        ] * profile.num_replicas
    if not any(patch.key == "mps" for patch in user_patches):
        writes["mps"] = "on"
    else:
        pass
    return ReplicaProfileDecision(
        profile=profile,
        writes=writes,
        message=(
            f"Serving {profile.num_replicas} replicas of {config.model_path} on GPU "
            f"{gpu_id} ({device.name}) under MPS, the count measured for this GPU; "
            "--mps off serves one replica"
        ),
    )
