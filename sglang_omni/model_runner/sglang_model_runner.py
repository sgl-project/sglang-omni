from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import PortArgs, ServerArgs

from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.utils.gpu_memory import (
    calculate_stage_budget_available_bytes,
    calculate_stage_load_delta_bytes,
    format_bytes_gib,
    get_gpu_device_info,
    get_process_gpu_memory_bytes,
)

logger = logging.getLogger(__name__)


def filter_weights_by_prefix(
    weights: Iterator[tuple[str, Any]],
    prefix: str | None,
) -> Iterator[tuple[str, Any]]:
    """Filter weight iterator by prefix, stripping matched prefix from names."""
    if not prefix:
        yield from weights
        return
    for name, tensor in weights:
        if name.startswith(prefix):
            yield name[len(prefix) :], tensor


@dataclass(slots=True, kw_only=True)
class _OmniKVCacheConfigurator(KVCacheConfigurator):
    """KV-cache configurator that honors an Omni colocated-stage memory budget.

    ``super()`` is deliberately spelled out below: ``@dataclass(slots=True)``
    rebuilds the class object, which leaves the zero-arg ``super()`` closure
    cell pointing at the pre-rebuild class.
    """

    total_gpu_memory_fraction: float | None = None

    def _profile_available_bytes(self, pre_model_load_memory: float) -> int:
        """Profile KV-cache headroom for colocated SGLang AR stages.

        Upstream SGLang profiles from global free-memory deltas. That is valid
        for a single AR engine, but colocated Omni stages can load multiple
        SGLang engines in separate processes on the same GPU. In that case
        another process can change global free memory while this process is
        loading weights, making the global delta too small or negative.

        When a stage total-memory budget is provided, compute cache headroom as
        total GPU memory times that budget minus this stage's measured memory.
        NVML process accounting is preferred. If NVML cannot identify the
        current process, use the stage-local load delta measured inside
        SGLang's serialized initialization window. Without a stage budget, keep
        upstream SGLang profiling semantics for ordinary non-colocated AR
        serving.
        """
        if self.total_gpu_memory_fraction is None:
            return KVCacheConfigurator._profile_available_bytes(
                self, pre_model_load_memory
            )

        process_memory = get_process_gpu_memory_bytes(self.gpu_id)
        device_info = get_gpu_device_info(self.gpu_id)
        total_memory = device_info.total_memory_bytes

        if total_memory is None:
            raise RuntimeError(
                "Colocated SGLang AR stage requires total GPU memory for "
                f"gpu_id={self.gpu_id}. Check CUDA_VISIBLE_DEVICES and CUDA "
                "device visibility."
            )

        if process_memory is None or process_memory <= 0:
            return self._profile_available_bytes_from_stage_load_delta(
                pre_model_load_memory,
                total_memory,
            )

        return self._profile_available_bytes_from_process_memory(
            total_memory,
            process_memory,
        )

    def _profile_available_bytes_from_stage_load_delta(
        self,
        pre_model_load_memory: float,
        total_memory: int,
    ) -> int:
        """Profile colocated KV headroom from this stage's load-time delta."""
        from sglang.srt.distributed.parallel_state import get_world_group
        from sglang.srt.utils.common import get_available_gpu_memory

        world_group = get_world_group()
        post_model_load_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=world_group.world_size > 1,
            cpu_group=world_group.cpu_group,
        )
        stage_load_bytes = calculate_stage_load_delta_bytes(
            pre_model_load_memory_gib=pre_model_load_memory,
            post_model_load_memory_gib=post_model_load_memory,
        )
        available_bytes = calculate_stage_budget_available_bytes(
            total_memory_bytes=total_memory,
            accounted_memory_bytes=stage_load_bytes,
            memory_fraction=self.total_gpu_memory_fraction,
            accounted_memory_label="stage_load_used",
        )
        logger.info(
            f"SGLang AR memory profile: gpu_mem_accounting=stage_load_fallback "
            f"gpu_id={self.gpu_id} "
            f"total_gpu_memory_fraction={self.total_gpu_memory_fraction:.3f} "
            f"mem_fraction_static={self.server_args.mem_fraction_static:.3f} "
            f"total={format_bytes_gib(total_memory)} "
            f"stage_load_used={format_bytes_gib(stage_load_bytes)} "
            f"available_for_kv={format_bytes_gib(available_bytes)}"
        )
        return available_bytes

    def _profile_available_bytes_from_process_memory(
        self,
        total_memory: int,
        process_memory: int,
    ) -> int:
        available_bytes = calculate_stage_budget_available_bytes(
            total_memory_bytes=total_memory,
            accounted_memory_bytes=process_memory,
            memory_fraction=self.total_gpu_memory_fraction,
            accounted_memory_label="process_used",
        )
        logger.info(
            f"SGLang AR memory profile: gpu_mem_accounting=nvml_process "
            f"gpu_id={self.gpu_id} "
            f"total_gpu_memory_fraction={self.total_gpu_memory_fraction:.3f} "
            f"mem_fraction_static={self.server_args.mem_fraction_static:.3f} "
            f"total={format_bytes_gib(total_memory)} "
            f"process_used={format_bytes_gib(process_memory)} "
            f"available_for_kv={format_bytes_gib(available_bytes)}"
        )
        return available_bytes


class SGLModelRunner(ModelRunner):
    """Thin wrapper to bootstrap SGLang ModelRunner from backend args."""

    def __init__(
        self,
        model_config: ModelConfig,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        moe_ep_size: int,
        pp_rank: int,
        pp_size: int,
        nccl_port: int,
        model_arch_override: str | None = None,
        weight_prefix: str | None = None,
        total_gpu_memory_fraction: float | None = None,
    ) -> None:
        self._weight_prefix = weight_prefix
        self._total_gpu_memory_fraction = total_gpu_memory_fraction
        self._model_arch_override = model_arch_override
        self._weight_share_config = None
        self._weight_share_record = None
        self._weight_ipc_leader_monitor = None
        self._register_omni_model()

        port_args = PortArgs.init_new(server_args)
        tp_size = server_args.tp_size
        self.nccl_port = port_args.nccl_port

        # model_config is already fully configured by ModelWorker._init_model_config()
        # (architecture override, text_config swap, etc. are all done there)

        attn_tp_rank, attn_tp_size, attn_dp_rank, attn_dp_size = (
            compute_dp_attention_world_info(
                server_args.enable_dp_attention,
                tp_rank,
                tp_size,
                server_args.dp_size,
                server_args.attn_cp_size,
            )
        )
        ps = ParallelState(
            tp_rank=tp_rank,
            tp_size=tp_size,
            pp_rank=pp_rank,
            pp_size=pp_size,
            dp_rank=None,
            dp_size=server_args.dp_size,
            attn_tp_rank=attn_tp_rank,
            attn_tp_size=attn_tp_size,
            attn_cp_rank=0,
            attn_cp_size=server_args.attn_cp_size,
            attn_dcp_rank=tp_rank % server_args.dcp_size,
            attn_dcp_size=server_args.dcp_size,
            attn_dp_rank=attn_dp_rank,
            attn_dp_size=attn_dp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=moe_ep_size,
            moe_dp_rank=None,
            moe_dp_size=server_args.moe_dp_size,
            gpu_id=gpu_id,
        )

        super().__init__(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            ps=ps,
            nccl_port=nccl_port,
            server_args=server_args,
        )

    def _extend_forward_kwargs(self, forward_batch, pp_proxy_tensors):
        """Expose Omni's private prefill sidecar after graph admission.

        Upstream owns ``mm_inputs`` and the official ``input_embeds`` batch
        field. Keeping both untouched during admission preserves their
        contracts; model kwargs are the supported late-bound transport used by
        both eager execution and breakable prefill graph capture/replay.
        """
        kwargs = super()._extend_forward_kwargs(forward_batch, pp_proxy_tensors)
        prefill_inputs = get_omni_prefill_inputs(forward_batch)
        if prefill_inputs is None:
            return kwargs

        if "input_embeds" in kwargs:
            raise RuntimeError(
                "Omni prefill sidecar conflicts with an upstream input_embeds "
                "forward kwarg"
            )

        if prefill_inputs.input_embeds.dtype != self.dtype:
            raise RuntimeError(
                "Omni prefill sidecar must be in model dtype "
                f"{self.dtype}, got {prefill_inputs.input_embeds.dtype}; the "
                "prefill graph slot copy would silently cast it"
            )
        kwargs["input_embeds"] = prefill_inputs.input_embeds
        kwargs["omni_prefill_rids"] = forward_batch.rids
        if prefill_inputs.input_embeds_are_projected is not None:
            kwargs["input_embeds_are_projected"] = (
                prefill_inputs.input_embeds_are_projected
            )
        return kwargs

    def _resolve_draft_load_format(self) -> str | None:
        """A weight-share follower builds its module tree with dummy weights.

        This is the runner's own load format, which upstream resolves in
        ModelRunner.__init__ and feeds to the loader, so the published
        load_format record is never touched.
        """
        from sglang_omni.utils import ipc_weights

        ws = ipc_weights.get_weight_share_config()
        if ws is not None and ws.role == "follower":
            return "dummy"
        return super()._resolve_draft_load_format()

    def load_model(self):
        """Load weights, honoring the same-GPU weight-share role, if any.

        Leader: normal checkpoint load, then publish CUDA-IPC handles for all
        parameters/buffers. Follower: wait for the leader's handle file, build
        the module tree with dummy weights (no checkpoint I/O), then alias
        every parameter/buffer onto the leader's storage. Tensors the
        architecture's share policy marks replica-private keep the follower's
        own storage. Both paths finish inside load_model, strictly before
        KV-pool profiling, warmup forwards, and CUDA graph capture.
        """
        from sglang_omni.utils import ipc_weights

        ws = ipc_weights.get_weight_share_config()
        self._weight_share_config = ws
        self._weight_share_record = None
        if ws is None:
            return super().load_model()

        # Note (Jiaxin Deng): TP/PP ranks are separate processes inheriting the
        # env var, and their shards share names/shapes/dtypes across ranks, so
        # the handle file would collide and followers would silently attach
        # another rank's shard. Refuse until the handle path is rank-qualified.
        if self.server_args.tp_size != 1 or self.server_args.pp_size != 1:
            raise ipc_weights.WeightShareError(
                "SGLANG_OMNI_WEIGHT_SHARE requires tp_size == pp_size == 1, got "
                f"tp={self.server_args.tp_size} pp={self.server_args.pp_size}"
            )

        architectures = (
            [self._model_arch_override]
            if self._model_arch_override is not None
            else self.model_config.hf_config.architectures
        )
        policy = ipc_weights.validate_weight_share_architecture(architectures)

        # Note (Jiaxin Deng): a follower frees its dummy weights before KV
        # profiling, so it must pin an explicit cap or it over-budgets KV.
        if ws.role == "follower" and self.server_args.max_total_tokens is None:
            raise ipc_weights.WeightShareError(
                "SGLANG_OMNI_WEIGHT_SHARE follower requires an explicit "
                "--max-total-tokens: post-alias memory profiling cannot derive "
                "a stable KV budget"
            )

        if ws.role == "leader":
            super().load_model()
            self._weight_share_record = ipc_weights.leader_export(
                self.model,
                ws.dir_path,
                model_path=str(self.server_args.model_path),
                model_revision=self.server_args.revision,
                run_id=ws.run_id,
                private_names=policy.private_tensor_names,
            )
            return

        # Note (Jiaxin Deng): wait for the leader BEFORE allocating dummy
        # weights so we never hold a full transient dummy copy while blocked.
        # The exact handle file name needs the constructed model's class, so
        # this pre-wait polls for any export in the directory; follower_attach
        # below still waits on (and validates) the engine's own file.
        import torch

        ipc_weights.wait_for_any_export(ws.dir_path, timeout_s=ws.attach_timeout_s)
        super().load_model()
        self._weight_share_record, self._weight_ipc_leader_monitor = (
            ipc_weights.follower_attach(
                self.model,
                ws.dir_path,
                timeout_s=ws.attach_timeout_s,
                model_path=str(self.server_args.model_path),
                model_revision=self.server_args.revision,
                run_id=ws.run_id,
                private_names=policy.private_tensor_names,
            )
        )
        # Note (Jiaxin Deng): a model that folds several tensors into one buffer
        # must re-derive that view here, or replicas diverge on the kernel path.
        attached_hook = getattr(self.model, "on_weight_share_attached", None)
        if callable(attached_hook):
            attached_hook()
        # Note (Jiaxin Deng): return the dropped dummy-weight blocks to the
        # driver so KV-pool profiling and later replicas see the freed memory.
        torch.cuda.empty_cache()

    def init_cuda_graphs(self, capture_decode_cuda_graph: bool = True):
        """Re-verify shared weights and finish post-capture KV sizing.

        Followers: catches any load-path step that re-created a parameter
        after attach (would silently serve dummy weights). Leader: catches a
        post-export .data rebind (would silently orphan the followers).

        SGLang optionally reserves the KV pool as virtual memory and
        backs its serving span only after CUDA graph capture. Omni has several
        deferred graph-capture call sites, so finalize here at the common
        capture boundary instead of relying on every stage to mirror the
        scheduler's post-capture hook.
        """
        record = self._weight_share_record
        if record is not None:
            from sglang_omni.utils import ipc_weights

            ipc_weights.verify_attachment(self.model, record)
        # Engine builders turn enable_torch_compile off on the exec bag after the
        # capture flags were seeded at publish; re-seed so capture honors that.
        from sglang.srt.runtime_context import get_exec, get_flags

        get_flags().capture.enable_torch_compile = get_exec().graph.enable_torch_compile
        result = super().init_cuda_graphs(capture_decode_cuda_graph)
        if self.token_to_kv_pool.post_capture_active:
            self.post_capture_resize_kv_pool()
        return result

    def _weight_update_blocked_reason(self) -> str | None:
        ws = self._weight_share_config
        if ws is None:
            return None
        return (
            f"weight updates are disabled while same-GPU weight sharing is "
            f"active (role={ws.role}): replicas alias the leader's storage, "
            "so an in-place update would corrupt every replica; restart the "
            "whole replica group with new weights instead"
        )

    # Kept on the runner so ModelWorker has one call target and the weight-share
    # guard applies to every update path.
    def update_weights_from_disk(self, *args, **kwargs):
        reason = self._weight_update_blocked_reason()
        if reason is not None:
            return False, reason
        return self.weight_updater.update_weights_from_disk(*args, **kwargs)

    def update_weights_from_tensor(self, *args, **kwargs):
        reason = self._weight_update_blocked_reason()
        if reason is not None:
            return False, reason
        return self.weight_updater.update_weights_from_tensor(*args, **kwargs)

    def update_weights_from_distributed(self, *args, **kwargs):
        reason = self._weight_update_blocked_reason()
        if reason is not None:
            return False, reason
        return self.weight_updater.update_weights_from_distributed(*args, **kwargs)

    # Process-group lifecycle does not mutate weights, so it stays unguarded.
    def init_weights_update_group(self, *args, **kwargs):
        return self.weight_updater.init_weights_update_group(*args, **kwargs)

    def destroy_weights_update_group(self, *args, **kwargs):
        return self.weight_updater.destroy_weights_update_group(*args, **kwargs)

    def _register_omni_model(self):
        # Register sglang_omni model classes directly in SGLang's model registry.
        import importlib

        from sglang.srt.models.registry import ModelRegistry

        sglang_omni_models = {
            "S2ProSGLangTextModel": "sglang_omni.models.fishaudio_s2_pro.sglang_model:S2ProSGLangTextModel",
            "Qwen3OmniTalker": "sglang_omni.models.qwen3_omni.components.talker:Qwen3OmniTalker",
            "Qwen3OmniThinkerForCausalLM": "sglang_omni.models.qwen3_omni.components.sglang_thinker:Qwen3OmniThinkerForCausalLM",
            "HiggsMultimodalQwen3ForConditionalGeneration": "sglang_omni.models.higgs_tts.model:HiggsTTSModel",
            "Qwen3TTSTalker": "sglang_omni.models.qwen3_tts.sglang_model:Qwen3TTSTalker",
            "MingTTSSGLangModel": "sglang_omni.models.ming_tts.sglang_model:MingTTSSGLangModel",
            "MossTTSDelaySGLangModel": "sglang_omni.models.moss_tts.sglang_model:MossTTSDelaySGLangModel",
            "MossTTSLocalSGLangModel": "sglang_omni.models.moss_tts_local.sglang_model:MossTTSLocalSGLangModel",
            "MossTranscribeDiarizeForConditionalGeneration": "sglang_omni.models.moss_transcribe_diarize.sglang_model:MossTranscribeDiarizeForConditionalGeneration",
            "VoxtralSGLangTTSModel": "sglang_omni.models.voxtral_tts.sglang_model:VoxtralSGLangTTSModel",
            "Zonos2SGLangModel": "sglang_omni.models.zonos2.sglang_model:Zonos2SGLangModel",
            "LLaDA2MoeModelLM": "sglang_omni.models.llada2_uni.components.thinker:LLaDA2MoeModelLM",
            "WhisperForConditionalGeneration": "sglang_omni.models.whisper_asr.sglang_model:WhisperForConditionalGeneration",
            "Qwen3ASRForConditionalGeneration": "sglang_omni.models.qwen3_asr.sglang_model:Qwen3ASRForConditionalGeneration",
            "FunAsrNanoForConditionalGeneration": "sglang_omni.models.fun_asr.sglang_model:FunAsrNanoForConditionalGeneration",
            "ArkasrForConditionalGeneration": "sglang_omni.models.arkasr.sglang_model:ArkasrForConditionalGeneration",
            "DotsTTSForConditionalGeneration": "sglang_omni.models.dots_tts.sglang_model:DotsTTSSGLangModel",
            "FunCosyVoice3SGLangModel": "sglang_omni.models.fun_cosyvoice3.sglang_model:FunCosyVoice3SGLangModel",
        }
        for arch, path in sglang_omni_models.items():
            module_path, _, attr = path.partition(":")
            try:
                ModelRegistry.models[arch] = getattr(
                    importlib.import_module(module_path), attr
                )
            except Exception as exc:
                logger.warning(f"sglang-omni: skipping model {arch} ({exc})")

        try:
            from sglang_omni.models.ming_omni.registration import (
                register_ming_hf_config,
                register_ming_model_registry,
            )

            register_ming_hf_config()
            register_ming_model_registry()
        except Exception as exc:
            logger.warning(f"sglang-omni: skipping Ming-Omni registration ({exc})")

    def init_kv_cache_configurator(self):
        """Swap in the Omni configurator so the colocated budget stays hooked.

        Upstream keeps _profile_available_bytes on the composed
        KVCacheConfigurator, not on the ModelRunner MRO. Rebuild upstream's instance as
        the Omni subclass, copying every declared field so upstream can add
        fields without silently dropping them here.
        """
        super().init_kv_cache_configurator()
        base = self.kv_cache_configurator
        self.kv_cache_configurator = _OmniKVCacheConfigurator(
            **{
                field.name: getattr(base, field.name)
                for field in dataclasses.fields(base)
                if field.init
            },
            total_gpu_memory_fraction=self._total_gpu_memory_fraction,
        )
