from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

from sglang.srt.arg_groups.model_override_base import (
    attention_backends_of,
    resolved_view,
)
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.platforms.cpu import CpuDeviceMixin
from sglang.srt.server_args import ServerArgs

from sglang_omni.platforms import cpu_numa
from sglang_omni.platforms.interface import OmniPlatform

if TYPE_CHECKING:
    from sglang_omni.pipeline.stage_workers import (
        StageLaunchConfig,
        StageWorkerProcessSpec,
    )


class CPUOmniPlatform(CpuDeviceMixin, OmniPlatform):
    def enable_code2wav_graph(self) -> bool:
        return False

    def apply_model_worker_backend_policy(
        self,
        server_args: ServerArgs,
        model_config: ModelConfig,
        model_arch_override: str | None,
    ) -> str | None:
        effective_quantization = super().apply_model_worker_backend_policy(
            server_args, model_config, model_arch_override
        )
        if model_arch_override == "WhisperForConditionalGeneration" and any(
            backend not in ("intel_amx", "torch_native")
            for backend in attention_backends_of(resolved_view(server_args))
        ):
            raise ValueError(
                "Whisper ASR on CPU supports intel_amx or torch_native for both "
                "prefill and decode attention. Remove unsupported attention "
                "backend overrides."
            )
        else:
            pass
        return effective_quantization

    def prepare_worker_process(self) -> None:
        cpu_numa.apply_bound_affinity()
        cpu_numa.install_binding_hook()

    def plan_process_placement(
        self, process_specs: list[StageWorkerProcessSpec]
    ) -> None:
        cpu_numa.reset_assignment()
        spec_envs = []
        for spec in process_specs:
            env = {}
            for stage_spec in spec.stage_specs:
                env.update(stage_spec.env_defaults)
            env = {**env, **os.environ}
            spec_envs.append((spec, env))
            cpu_numa.claim_explicit_nodes(spec.process_name, env)
        # Tensor-parallel groups need several nodes at once, so they pick before single-process stages take the free ones, whatever the start order.
        for spec, env in spec_envs:
            if any(stage_spec.tp_size > 1 for stage_spec in spec.stage_specs):
                self.get_process_placement_env(
                    spec.process_name, spec.stage_specs, env=env
                )

    def get_process_thread_budget(self) -> int | None:
        return cpu_numa.node_core_count()

    def get_process_cpu_mask(self, env: Mapping[str, str]) -> set[int] | None:
        return cpu_numa.bound_cpus(env)

    def get_process_placement_env(
        self,
        process_name: str,
        stage_specs: list[StageLaunchConfig],
        env: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        if not stage_specs:
            return {}
        return cpu_numa.threads_bind_env(
            process_name=process_name,
            stage_names=[spec.stage_name for spec in stage_specs],
            tp_size=max(spec.tp_size for spec in stage_specs),
            tp_rank=stage_specs[0].tp_rank,
            env=env,
        )
