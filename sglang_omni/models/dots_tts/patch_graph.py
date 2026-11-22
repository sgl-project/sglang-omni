# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from typing import Literal, NamedTuple, Protocol

import torch

from sglang_omni.models.dots_tts.compat import import_dots_tts
from sglang_omni.platforms.device_graph import (
    DeviceGraphBackend,
    DeviceGraphPool,
    ReplayableGraph,
)

import_dots_tts()

from dots_tts.modules.backbone.dit_inference import (
    CachedDiTRunner,
    DiTInferenceContext,
    DiTKvCache,
    DiTSolver,
    DiTSolverState,
    ODESchedule,
)

logger = logging.getLogger(__name__)


class BucketResolver(Protocol):
    def __call__(self, requested: int, /) -> int: ...


class PatchSolveGraph(NamedTuple):
    graph: ReplayableGraph
    static_inputs: dict[str, torch.Tensor]
    latent: torch.Tensor
    new_keys: torch.Tensor
    new_values: torch.Tensor


class PersistentCacheDiTRunner(CachedDiTRunner):

    def __init__(
        self,
        *,
        context: DiTInferenceContext,
        capacity_patches: int,
        capacity_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
        attn_backend: Literal["sdpa", "flex"],
        compile_step: bool,
        branch_multiplier: int,
    ) -> None:
        super().__init__(
            context=context,
            capacity_patches=capacity_patches,
            capacity_tokens=capacity_tokens,
            dtype=dtype,
            device=device,
            attn_backend=attn_backend,
            compile_step=compile_step,
            branch_multiplier=branch_multiplier,
        )
        self.persistent_caches: dict[tuple[int, int], DiTKvCache] = {}
        self.patch_graphs: dict[tuple[int, int], PatchSolveGraph] = {}

    def allocate_cache(self, *, batch_size: int, nfe: int) -> DiTKvCache:
        shape_key = (int(batch_size), int(nfe))
        cache = self.persistent_caches.get(shape_key)
        if cache is None:
            cache = super().allocate_cache(batch_size=batch_size, nfe=nfe)
            self.persistent_caches[shape_key] = cache
        else:
            cache.cache_k.zero_()
            cache.cache_v.zero_()
        return DiTKvCache(
            capacity_patches=cache.capacity_patches,
            capacity_tokens=cache.capacity_tokens,
            nfe=cache.nfe,
            cache_k=cache.cache_k,
            cache_v=cache.cache_v,
        )


class GraphedDiTSolver(DiTSolver):

    def __init__(
        self,
        context: DiTInferenceContext,
        *,
        optimize: bool,
        bucket_resolver: BucketResolver,
        meanflow: bool,
        graph_backend: DeviceGraphBackend,
        warmup_iterations: int = 2,
    ) -> None:
        super().__init__(
            context,
            optimize=optimize,
            bucket_resolver=bucket_resolver,
            meanflow=meanflow,
        )
        self.graph_backend = graph_backend
        self.graph_pool: DeviceGraphPool | None = None
        self.warmup_iterations = int(warmup_iterations)
        self.declined: set[tuple[int, int, int]] = set()
        self.replays = 0

    def _get_cached_runner(  # noqa: leading-underscore  # DiTSolver hook
        self, *, capacity_patches: int, device: torch.device, dtype: torch.dtype
    ) -> PersistentCacheDiTRunner:
        attn_backend = self._backend_for(
            device
        )  # noqa: leading-underscore  # DiTSolver method
        compile_step = bool(self.optimize and device.type == "cuda")
        runner_key = (
            int(capacity_patches),
            str(device),
            dtype,
            attn_backend,
            compile_step,
        )
        runners = (
            self._cached_runners  # noqa: leading-underscore  # DiTSolver attribute
        )
        runner = runners.get(runner_key)
        if runner is None:
            runner = PersistentCacheDiTRunner(
                context=self.context,
                capacity_patches=int(capacity_patches),
                capacity_tokens=int(capacity_patches) * self.context.unit_len,
                dtype=dtype,
                device=device,
                attn_backend=attn_backend,
                compile_step=compile_step,
                branch_multiplier=self.branch_multiplier,
            )
            runners[runner_key] = runner
        else:
            pass
        return runner

    def _decode_with_kv_cache(  # noqa: leading-underscore  # DiTSolver hook
        self,
        z: torch.Tensor,
        *,
        state: DiTSolverState,
        sequence: torch.Tensor,
        cfg_sequence: torch.Tensor | None,
        fm_seq_len: int,
        prefix_len: int,
        current_hidden: torch.Tensor,
        cfg_current_hidden: torch.Tensor | None,
        all_mods_by_ode: torch.Tensor,
        schedule: ODESchedule,
        nfe: int,
        batch_size: int,
        guidance: torch.Tensor | None,
        flow_dt: torch.Tensor | None,
    ) -> torch.Tensor:
        persistent_tokens = prefix_len - int(self.context.unit_len)
        runner, kv_cache = (
            self._ensure_cache(  # noqa: leading-underscore  # DiTSolver method
                state, sequence=sequence[:, :fm_seq_len], nfe=nfe
            )
        )
        if kv_cache.valid_tokens != persistent_tokens:
            runner.prefill(
                kv_cache=kv_cache,
                prefix_sequence=sequence[:, :persistent_tokens],
                cfg_prefix_sequence=(
                    None if self.meanflow else cfg_sequence[:, :persistent_tokens]
                ),
                all_mods_by_ode=all_mods_by_ode,
            )
        else:
            pass
        block_mask, sdpa_mask = runner.masks_for(
            valid_persistent_tokens=persistent_tokens
        )
        rotary_cos, rotary_sin = runner.rotary_for(start_pos=persistent_tokens)
        patch_inputs = {
            "z": z,
            "prev_unit": sequence[:, persistent_tokens:prefix_len],
            "current_hidden": current_hidden,
            "all_mods": all_mods_by_ode,
            "sdpa_mask": sdpa_mask,
            "rotary_cos": rotary_cos,
            "rotary_sin": rotary_sin,
            "times": schedule.times,
        }
        if self.meanflow:
            patch_inputs["durations"] = schedule.durations
        else:
            patch_inputs.update(
                cfg_prev_unit=cfg_sequence[:, persistent_tokens:prefix_len],
                cfg_current_hidden=cfg_current_hidden,
                guidance=guidance,
                flow_dt=flow_dt,
            )
        captured = (
            None
            if block_mask is not None
            else self.patch_graph(
                runner, kv_cache, patch_inputs, ode_steps=nfe, batch_size=batch_size
            )
        )
        if captured is None:
            return super()._decode_with_kv_cache(  # noqa: leading-underscore  # DiTSolver hook
                z,
                state=state,
                sequence=sequence,
                cfg_sequence=cfg_sequence,
                fm_seq_len=fm_seq_len,
                prefix_len=prefix_len,
                current_hidden=current_hidden,
                cfg_current_hidden=cfg_current_hidden,
                all_mods_by_ode=all_mods_by_ode,
                schedule=schedule,
                nfe=nfe,
                batch_size=batch_size,
                guidance=guidance,
                flow_dt=flow_dt,
            )
        else:
            pass
        for name, value in patch_inputs.items():
            captured.static_inputs[name].copy_(value)
        captured.graph.replay()
        self.replays += 1
        kv_cache.cache_k[:, :, :, :, persistent_tokens:prefix_len].copy_(
            captured.new_keys
        )
        kv_cache.cache_v[:, :, :, :, persistent_tokens:prefix_len].copy_(
            captured.new_values
        )
        kv_cache.valid_tokens = prefix_len
        return captured.latent.clone()

    def patch_graph(
        self,
        runner: PersistentCacheDiTRunner,
        kv_cache: DiTKvCache,
        patch_inputs: dict[str, torch.Tensor],
        *,
        ode_steps: int,
        batch_size: int,
    ) -> PatchSolveGraph | None:
        shape_key = (int(batch_size), int(ode_steps))
        persistent = runner.persistent_caches.get(shape_key)
        if persistent is None or persistent.cache_k is not kv_cache.cache_k:
            return None
        else:
            pass
        captured = runner.patch_graphs.get(shape_key)
        if captured is not None:
            return captured
        else:
            pass
        bucket_shape = (int(runner.capacity_patches), *shape_key)
        if bucket_shape in self.declined:
            return None
        else:
            pass
        try:
            captured = self.capture(
                runner,
                kv_cache,
                patch_inputs,
                ode_steps=ode_steps,
                batch_size=batch_size,
            )
        except Exception as exc:
            self.declined.add(bucket_shape)
            logger.warning(
                f"dots.tts DiT patch graph capture failed for "
                f"capacity_patches={runner.capacity_patches} batch={batch_size} "
                f"ode_steps={ode_steps}: {exc}; this bucket solves eagerly"
            )
            return None
        runner.patch_graphs[shape_key] = captured
        logger.info(
            f"dots.tts DiT patch graph captured: capacity_patches="
            f"{runner.capacity_patches} batch={batch_size} ode_steps={ode_steps}"
        )
        return captured

    def capture(
        self,
        runner: PersistentCacheDiTRunner,
        kv_cache: DiTKvCache,
        patch_inputs: dict[str, torch.Tensor],
        *,
        ode_steps: int,
        batch_size: int,
    ) -> PatchSolveGraph:
        device = kv_cache.cache_k.device
        device_module = torch.get_device_module(device)
        static_inputs = {name: value.clone() for name, value in patch_inputs.items()}
        schedule = ODESchedule(
            mode="meanflow" if self.meanflow else "flow_matching",
            times=static_inputs["times"],
            durations=static_inputs.get("durations"),
        )
        new_kv_shape = list(kv_cache.cache_k.shape)
        new_kv_shape[-2] = int(self.context.unit_len)
        new_keys = kv_cache.cache_k.new_empty(new_kv_shape)
        new_values = torch.empty_like(new_keys)

        def solve() -> torch.Tensor:
            z = static_inputs["z"]
            for ode_index in range(ode_steps):
                step_inputs = {
                    "prev_unit": static_inputs["prev_unit"],
                    "current_hidden": static_inputs["current_hidden"],
                    "all_mods": static_inputs["all_mods"][ode_index],
                    "cache_k": kv_cache.cache_k[ode_index],
                    "cache_v": kv_cache.cache_v[ode_index],
                    "block_mask": None,
                    "sdpa_mask": static_inputs["sdpa_mask"],
                    "rotary_cos": static_inputs["rotary_cos"],
                    "rotary_sin": static_inputs["rotary_sin"],
                    **schedule.step_kwargs(ode_index),
                }
                if self.meanflow:
                    flow_dt = None
                else:
                    step_inputs.update(
                        cfg_prev_unit=static_inputs["cfg_prev_unit"],
                        cfg_current_hidden=static_inputs["cfg_current_hidden"],
                        guidance_scale=static_inputs["guidance"],
                    )
                    flow_dt = static_inputs["flow_dt"]
                velocity, step_keys, step_values = runner.step(z, **step_inputs)
                z = schedule.advance(
                    z,
                    velocity,
                    ode_idx=ode_index,
                    batch_size=batch_size,
                    flow_dt=flow_dt,
                )
                new_keys[ode_index].copy_(step_keys)
                new_values[ode_index].copy_(step_values)
            return z

        warmup_stream = device_module.Stream(device=device)
        warmup_stream.wait_stream(device_module.current_stream(device))
        with device_module.stream(warmup_stream):
            for _ in range(self.warmup_iterations):
                solve()
        device_module.current_stream(device).wait_stream(warmup_stream)
        device_module.synchronize(device)
        if self.graph_pool is None:
            self.graph_pool = self.graph_backend.graph_pool_handle()
        else:
            pass
        with self.graph_backend.capture(
            pool=self.graph_pool, thread_local_errors=True
        ) as graph:
            latent = solve()
        device_module.synchronize(device)
        return PatchSolveGraph(graph, static_inputs, latent, new_keys, new_values)


__all__ = [
    "BucketResolver",
    "GraphedDiTSolver",
    "PatchSolveGraph",
    "PersistentCacheDiTRunner",
]
