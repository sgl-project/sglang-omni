# SPDX-License-Identifier: Apache-2.0
"""Minimal SGLang-native encoder runner with partial encoder loading.

Owns SGLang's distributed init, encoder-only ``ServerArgs`` /
``ModelConfig`` / ``LoadConfig``, and a small ``EncoderModuleContainer``
that holds *only* the upstream encoder submodules an adapter declared
through its :class:`EncoderModuleSpec` list. The runner NEVER calls
``get_model()`` on the full upstream ``ForConditionalGeneration``
class — for Qwen3-Omni that would instantiate the thinker LLM (~57 GB
fp16) and the talker on every encoder GPU, which both wastes memory
and OOMs on H100-class hardware.

What the runner inherits from upstream:
    - SGLang distributed init / TP groups
    - SGLang loader pipeline (``DefaultModelLoader._get_all_weights``,
      ``load_weights_and_postprocess``)
    - SGLang TP-aware layers used inside the upstream encoder modules
      (``ColumnParallelLinear`` / ``RowParallelLinear``, fused
      attention, etc.)

What the runner brings:
    - The :class:`EncoderModuleContainer` with adapter-supplied
      submodules and a prefix-filtering ``load_weights``.

Naming note: this module deliberately uses ``entry_rank`` /
``non_entry_rank`` for the rank-0-vs-rest asymmetry rather than
``leader`` / ``follower``. The asymmetry here is just "who owns
external IO" — there's no leader election, no failover, no consensus
machinery. The Stage-level ``single/leader/follower`` role split
elsewhere in v1 is unrelated and not touched.

See ``docs/developer_reference/encoder_tp_path_b_design.md`` →
"Upstream Reuse Boundary" / "EncoderModuleSpec".
"""
from __future__ import annotations

import logging
from types import MethodType
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
    set_mscclpp_all_reduce,
    set_torch_symm_mem_all_reduce,
)
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.model_loader import get_model_loader
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.server_args import set_global_server_args_for_scheduler

from sglang_omni.scheduling.sglang_backend import build_sglang_encoder_server_args

if TYPE_CHECKING:
    from sglang_omni.models.qwen3_omni.encoder_adapters import (
        BatchPlan,
        EncoderModuleSpec,
    )

logger = logging.getLogger(__name__)


# Runner-managed kwargs that the helper consumes as direct keyword args.
# Reject these in `server_args_overrides` BEFORE the **splat — otherwise
# Python raises TypeError "got multiple values for keyword argument" before
# the helper's protected-key reject can fire.
_RUNNER_MANAGED_KEYS = frozenset({
    "model_path", "tp_size", "base_gpu_id", "dist_init_addr",
    "dtype", "load_format",
})

_TP_PARITY_MODES = frozenset({"default", "fp32_row_parallel", "fp32_linear"})


def _resolve_quant_config(model_config: ModelConfig, load_config: LoadConfig) -> Any:
    """Best-effort accessor for the model's quantization config.

    Different SGLang versions surface this in different places; encoder
    submodule constructors that accept a ``quant_config`` will work fine
    with ``None`` (unquantized), so we return ``None`` if we can't find
    one.
    """
    del load_config  # not currently consumed; reserved for future loaders
    qc = getattr(model_config, "quantization_config", None)
    if qc is not None:
        return qc
    return getattr(model_config, "quant_config", None)


def _configure_tp_collectives(server_args: Any) -> None:
    """Mirror SGLang ModelRunner's TP collective switch setup.

    ``initialize_model_parallel`` reads module-level switches in
    ``parallel_state`` when it builds the TP group. The encoder runner
    bypasses upstream ``ModelRunner``, so it must apply the same switches
    itself before TP initialization; otherwise ServerArgs overrides such
    as ``disable_custom_all_reduce`` are silently ignored.
    """
    set_custom_all_reduce(
        not bool(getattr(server_args, "disable_custom_all_reduce", False))
    )
    set_mscclpp_all_reduce(bool(getattr(server_args, "enable_mscclpp", False)))
    set_torch_symm_mem_all_reduce(
        bool(getattr(server_args, "enable_torch_symm_mem", False))
    )


def _resolve_tp_parity_mode(mode: str | None) -> str:
    resolved = mode or "default"
    if resolved not in _TP_PARITY_MODES:
        raise ValueError(
            f"Unsupported tp_parity_mode={mode!r}; "
            f"expected one of {sorted(_TP_PARITY_MODES)}"
        )
    return resolved


def _row_parallel_forward_fp32(
    self: Any,
    input_: torch.Tensor,
    skip_all_reduce: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """RowParallelLinear forward with fp32 local matmul + fp32 reduction.

    This is a parity/debug path for encoder TP experiments. It preserves the
    normal RowParallelLinear sharding semantics but avoids rounding each local
    partial result to fp16 before the TP all-reduce. The output is cast back to
    the input dtype so downstream kernels see the same dtype as the default
    path.
    """
    from sglang.srt.distributed import (
        get_tp_group,
        split_tensor_along_last_dim,
        tensor_model_parallel_all_reduce,
    )
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        use_symmetric_memory,
    )
    from sglang.srt.layers.dp_attention import is_allocation_symmetric

    if self.input_is_parallel:
        input_parallel = input_
    else:
        splitted_input = split_tensor_along_last_dim(
            input_, num_partitions=self.tp_size
        )
        input_parallel = splitted_input[self.tp_rank].contiguous()

    bias = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias

    weight = getattr(self, "weight", None)
    if weight is not None and getattr(self, "quant_config", None) is None:
        output_parallel = F.linear(
            input_parallel.to(torch.float32),
            weight.to(torch.float32),
            bias.to(torch.float32) if bias is not None else None,
        )
    else:
        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            output_parallel = self.quant_method.apply(
                self,
                input_parallel,
                bias=bias,
            )
        if output_parallel.is_floating_point():
            output_parallel = output_parallel.to(torch.float32)

    if self.reduce_results and self.tp_size > 1 and not skip_all_reduce:
        output = tensor_model_parallel_all_reduce(output_parallel)
    else:
        output = output_parallel

    if output.is_floating_point() and input_.dtype in (torch.float16, torch.bfloat16):
        output = output.to(input_.dtype)

    output_bias = self.bias if self.skip_bias_add else None
    return output, output_bias


def _column_parallel_forward_fp32(
    self: Any,
    input_: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """ColumnParallelLinear forward with fp32 local matmul.

    This covers ColumnParallelLinear subclasses too, including
    QKVParallelLinear and MergedColumnParallelLinear. It preserves the
    normal shard-local output contract; it only changes the local GEMM
    precision for parity/debug characterization.
    """
    from sglang.srt.distributed import tensor_model_parallel_all_gather

    bias = self.bias if not self.skip_bias_add else None
    weight = getattr(self, "weight", None)
    if weight is not None and getattr(self, "quant_config", None) is None:
        output_parallel = F.linear(
            input_.to(torch.float32),
            weight.to(torch.float32),
            bias.to(torch.float32) if bias is not None else None,
        )
    else:
        output_parallel = self.quant_method.apply(self, input_, bias)
        if output_parallel.is_floating_point():
            output_parallel = output_parallel.to(torch.float32)

    if self.gather_output:
        output = tensor_model_parallel_all_gather(output_parallel)
    else:
        output = output_parallel

    if output.is_floating_point() and input_.dtype in (torch.float16, torch.bfloat16):
        output = output.to(input_.dtype)

    output_bias = self.bias if self.skip_bias_add else None
    return output, output_bias


def _enable_fp32_linear(model: nn.Module, *, include_column_parallel: bool) -> dict[str, int]:
    """Patch TP linear modules in ``model`` for TP parity runs."""
    from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear

    counts = {"row_parallel": 0, "column_parallel": 0}
    for module in model.modules():
        if isinstance(module, RowParallelLinear):
            module.forward = MethodType(_row_parallel_forward_fp32, module)
            counts["row_parallel"] += 1
        elif include_column_parallel and isinstance(module, ColumnParallelLinear):
            module.forward = MethodType(_column_parallel_forward_fp32, module)
            counts["column_parallel"] += 1
    return counts


class EncoderModuleContainer(nn.Module):
    """Module holder for partial encoder loading.

    Builds only the submodules declared by the supplied
    :class:`EncoderModuleSpec` list. Provides a ``load_weights`` method
    that:

    - Filters incoming ``(name, tensor)`` pairs by each spec's
      ``checkpoint_prefixes`` — anything that doesn't match any spec
      is dropped without allocating a destination tensor.
    - Applies each spec's ``checkpoint_rewrites`` to map upstream
      checkpoint key names onto this container's parameter namespace.
    - Dispatches to ``param.weight_loader`` (set by SGLang's TP-aware
      layers) when present, otherwise falls back to
      ``default_weight_loader`` so unfused params still copy correctly.

    The container is intentionally lean: no language model, no logits
    head, no talker, no generation path, no scheduler state. Its sole
    job is to give SGLang's loader a parameter namespace that contains
    the selected encoder submodules and nothing else.
    """

    def __init__(
        self,
        hf_config: Any,
        *,
        encoder_specs: Sequence["EncoderModuleSpec"],
        quant_config: Any,
    ) -> None:
        super().__init__()
        if not encoder_specs:
            raise ValueError(
                "EncoderModuleContainer requires at least one EncoderModuleSpec; "
                "the encoder adapter must declare its upstream submodules."
            )
        names = [spec.name for spec in encoder_specs]
        if len(set(names)) != len(names):
            raise ValueError(
                f"EncoderModuleSpec names must be unique within a stage: {names}"
            )

        self._specs: dict[str, "EncoderModuleSpec"] = {}
        for spec in encoder_specs:
            module = spec.build_module(hf_config, quant_config)
            self.add_module(spec.name, module)
            self._specs[spec.name] = spec

    # -- Loader entry point -------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        """SGLang loader hook — filter, rewrite, and dispatch.

        Two-stage dispatch:

        1. Group the incoming stream by spec using
           ``checkpoint_prefixes``. Anything that doesn't match any spec
           is dropped without allocating.
        2. Per submodule:
           - If the submodule has its own ``load_weights`` (e.g.
             ``Qwen3OmniMoeVisionEncoder``), strip the leading
             ``"<spec.name>."`` from each rewritten key and forward the
             slice. The submodule handles its own fused-shard dispatch.
           - Otherwise, apply ``stacked_params_mapping`` from the spec
             before doing a direct param lookup. This is how
             ``audio_tower.layers.X.self_attn.q_proj.weight`` reaches
             ``audio_tower.layers.X.self_attn.qkv_proj.weight`` with
             ``shard_id="q"``. Without it the audio attention layers
             are loaded with random init weights and produce NaN in
             forward.
        """
        # 1. Bucket by spec.
        buckets: dict[str, list[tuple[str, torch.Tensor]]] = {
            n: [] for n in self._specs
        }
        skipped = 0
        for name, loaded_weight in weights:
            spec = self._spec_for(name)
            if spec is None:
                skipped += 1
                continue
            rewritten = name
            for old, new in spec.checkpoint_rewrites:
                rewritten = rewritten.replace(old, new)
            buckets[spec.name].append((rewritten, loaded_weight))

        # 2. Per-spec dispatch.
        loaded_total = 0
        for spec_name, bucket in buckets.items():
            if not bucket:
                continue
            spec = self._specs[spec_name]
            submodule = getattr(self, spec_name)
            if hasattr(submodule, "load_weights"):
                # Delegate. The submodule expects names relative to itself
                # (without the leading "{spec.name}.").
                relative = []
                inner_prefix = spec.name + "."
                for n, w in bucket:
                    if n.startswith(inner_prefix):
                        relative.append((n[len(inner_prefix):], w))
                    else:
                        # If rewrites didn't produce the expected prefix,
                        # forward as-is and let the submodule's own
                        # error reporting fire.
                        relative.append((n, w))
                submodule.load_weights(relative)
                loaded_total += len(relative)
            else:
                loaded_total += self._load_with_stacked_mapping(spec, bucket)

        logger.info(
            "EncoderModuleContainer.load_weights: loaded=%d skipped=%d",
            loaded_total, skipped,
        )

    def _load_with_stacked_mapping(
        self,
        spec: "EncoderModuleSpec",
        weights: list[tuple[str, torch.Tensor]],
    ) -> int:
        """Apply spec.stacked_params_mapping then fall back to direct lookup."""
        params = dict(self.named_parameters())
        loaded = 0
        for name, loaded_weight in weights:
            dispatched = False
            for target, source, shard_id in spec.stacked_params_mapping:
                if source not in name:
                    continue
                mapped = name.replace(source, target)
                if mapped not in params:
                    continue
                param = params[mapped]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                dispatched = True
                loaded += 1
                break
            if dispatched:
                continue
            if name in params:
                param = params[name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)
                loaded += 1
        return loaded

    def _spec_for(self, name: str) -> "EncoderModuleSpec | None":
        for spec in self._specs.values():
            if name.startswith(spec.checkpoint_prefixes):
                return spec
        return None


class SGLangEncoderRunner:
    """SGLang-native encoder model wrapper with partial loading.

    The runner process is invoked by ``MultiProcessPipelineRunner``,
    which has remapped ``CUDA_VISIBLE_DEVICES`` to a single physical GPU
    before importing torch (see ``encoder_tp_path_b_design.md`` "GPU
    placement"). Therefore ``cuda:0`` is the only visible device in this
    process, regardless of ``tp_size`` and the configured physical
    ``gpu``. The runner pins ``cuda_device=0`` and forwards
    ``dist_local_rank=tp_rank`` so SGLang's local-master / shard-index
    callsites observe a unique ``[0, world_size)`` rank.

    Args:
        model_path: HF model id or local checkpoint path.
        gpu_id: Physical GPU id (informational; the launcher remap means
            the runner actually sees this as ``cuda:0``).
        tp_rank: TP rank in ``[0, tp_size)``.
        tp_size: TP world size.
        nccl_port: Loopback NCCL bootstrap port allocated by the parent
            pipeline runner. Required even at ``tp_size=1`` so the launch
            path is uniform across TP lanes.
        encoder_specs: Adapter-supplied list of encoder submodules to
            load. Required — the runner does NOT load the full
            ``ForConditionalGeneration`` class.
        dtype: Optional dtype override forwarded to ``ServerArgs``.
        load_format: Optional load-format override forwarded to ``ServerArgs``.
        server_args_overrides: Extra ``ServerArgs`` kwargs (loader / processor
            knobs only — protected keys raise ``ValueError``).
        tp_parity_mode: Optional debug mode for tp=1 vs tp>1 numerical
            comparisons. ``"fp32_row_parallel"`` runs RowParallelLinear local
            matmuls and TP all-reduces in fp32, then casts outputs back to the
            model dtype. ``"fp32_linear"`` additionally runs shard-local
            ColumnParallelLinear/QKV/Gate-Up matmuls in fp32.
    """

    def __init__(
        self,
        *,
        model_path: str,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        nccl_port: int | None,
        encoder_specs: Sequence["EncoderModuleSpec"],
        dtype: str | None = None,
        load_format: str | None = None,
        server_args_overrides: dict[str, Any] | None = None,
        tp_parity_mode: str | None = None,
    ) -> None:
        if tp_size < 1:
            raise ValueError(f"tp_size must be >= 1 (got {tp_size})")
        if not (0 <= tp_rank < tp_size):
            raise ValueError(
                f"tp_rank={tp_rank} must satisfy 0 <= tp_rank < tp_size={tp_size}"
            )
        if nccl_port is None:
            raise ValueError("SGLangEncoderRunner requires nccl_port")
        if not encoder_specs:
            raise ValueError(
                "SGLangEncoderRunner requires encoder_specs; the adapter must "
                "declare its upstream submodules. The runner no longer falls "
                "back to get_model() on the full ForConditionalGeneration "
                "class — that would load the LLM/talker on every encoder GPU."
            )

        overrides = dict(server_args_overrides or {})
        clobbered = sorted(_RUNNER_MANAGED_KEYS & overrides.keys())
        if clobbered:
            raise ValueError(
                f"server_args_overrides cannot set runner-managed keys "
                f"{clobbered}. These are derived from StageConfig and "
                f"factory parameters; pass them through StageConfig "
                f"(model_path, tp_size, gpu, dtype, load_format) instead."
            )

        self.tp_rank = int(tp_rank)
        self.tp_size = int(tp_size)
        self.is_entry_rank = self.tp_rank == 0
        self.tp_parity_mode = _resolve_tp_parity_mode(tp_parity_mode)

        # Forwarded for telemetry / debugging only — the runner hard-pins
        # cuda_device=0 because the launcher remap leaves only one device
        # visible.
        self.physical_gpu_id = int(gpu_id)

        # ServerArgs.dist_init_addr expects "host:port", torch.distributed
        # expects the full "tcp://host:port" URL. Keep them separate.
        port = int(nccl_port)
        dist_addr = f"127.0.0.1:{port}"
        dist_init_method = f"tcp://{dist_addr}"

        cuda_device = 0
        dist_local_rank = self.tp_rank
        self.device = torch.device(f"cuda:{cuda_device}")

        server_args = build_sglang_encoder_server_args(
            model_path=model_path,
            tp_size=self.tp_size,
            base_gpu_id=cuda_device,
            dist_init_addr=dist_addr,
            dtype=dtype,
            load_format=load_format,
            **overrides,
        )
        set_global_server_args_for_scheduler(server_args)

        self.server_args = server_args
        self.model_config = ModelConfig.from_server_args(server_args)
        self.load_config = LoadConfig(
            load_format=server_args.load_format,
            download_dir=server_args.download_dir,
            model_loader_extra_config=server_args.model_loader_extra_config,
            remote_instance_weight_loader_seed_instance_ip=(
                server_args.remote_instance_weight_loader_seed_instance_ip
            ),
            remote_instance_weight_loader_seed_instance_service_port=(
                server_args.remote_instance_weight_loader_seed_instance_service_port
            ),
            remote_instance_weight_loader_send_weights_group_ports=(
                server_args.remote_instance_weight_loader_send_weights_group_ports
            ),
        )

        torch.cuda.set_device(cuda_device)

        # Always run init_distributed_environment + initialize_model_parallel,
        # including tp_size==1: SGLang's parallel layers
        # (ColumnParallelLinear / RowParallelLinear) call get_tp_group()
        # during their own __init__ and will assert if the group has not
        # been initialized.
        init_distributed_environment(
            world_size=self.tp_size,
            rank=self.tp_rank,
            distributed_init_method=dist_init_method,
            local_rank=dist_local_rank,
            backend="nccl",
        )
        _configure_tp_collectives(server_args)
        initialize_model_parallel(tensor_model_parallel_size=self.tp_size)
        initialize_dp_attention(server_args, self.model_config)
        self.tp_group = get_tp_group()

        self._device_config = DeviceConfig(device="cuda", gpu_id=cuda_device)

        # ---- Partial-load path ----------------------------------------
        # Build only the encoder submodules the adapter declared, then
        # iterate the upstream loader's weight stream and let the
        # container drop everything that doesn't match a declared
        # prefix.
        self.model = self._build_and_load_encoder(encoder_specs)
        if self.tp_parity_mode in {"fp32_row_parallel", "fp32_linear"}:
            counts = _enable_fp32_linear(
                self.model,
                include_column_parallel=self.tp_parity_mode == "fp32_linear",
            )
            logger.info(
                "SGLangEncoderRunner tp_parity_mode=%s patched "
                "%d RowParallelLinear and %d ColumnParallelLinear modules",
                self.tp_parity_mode,
                counts["row_parallel"],
                counts["column_parallel"],
            )
        self.model.eval()

        logger.info(
            "SGLangEncoderRunner ready (tp_rank=%d/%d, physical_gpu=%d, "
            "model=%s, submodules=%s, dist_addr=%s)",
            self.tp_rank, self.tp_size, self.physical_gpu_id,
            model_path, [s.name for s in encoder_specs], dist_addr,
        )

    # ------------------------------------------------------------------
    # Partial-load helper
    # ------------------------------------------------------------------

    def _build_and_load_encoder(
        self,
        encoder_specs: Sequence["EncoderModuleSpec"],
    ) -> EncoderModuleContainer:
        """Construct the container + run upstream loader's weight stream.

        We build the container under ``set_default_torch_dtype`` and on
        the target device so SGLang's TP-aware layers register their
        ``param.weight_loader`` hooks correctly during ``__init__``.
        Then we hand the container off to ``DefaultModelLoader``'s
        weight-postprocess machinery, which iterates the safetensors
        shards and calls ``container.load_weights(...)`` once.
        """
        from sglang.srt.model_loader.utils import set_default_torch_dtype

        loader = get_model_loader(self.load_config)
        # We deliberately do not use ``loader.load_model`` because that
        # path always calls ``_initialize_model`` on the full upstream
        # entry class. The upstream API does not expose a
        # partial-encoder-loader yet (see RFC Open Question 1).
        target_device = torch.device(self._device_config.device)

        quant_config = _resolve_quant_config(self.model_config, self.load_config)

        with set_default_torch_dtype(self.model_config.dtype):
            with target_device:
                container = EncoderModuleContainer(
                    self.model_config.hf_config,
                    encoder_specs=encoder_specs,
                    quant_config=quant_config,
                )

            if isinstance(loader, DefaultModelLoader):
                weights = loader._get_all_weights(self.model_config, container)
            else:
                # Other loaders (BitsAndBytes, ShardedState, ...) don't
                # expose the same iterator. Non-default load formats
                # raise here so users see a clear error rather than a
                # broken partial weight set.
                raise NotImplementedError(
                    f"Encoder partial-load currently requires "
                    f"DefaultModelLoader; got {type(loader).__name__}. "
                    f"Use load_format='auto' (the default)."
                )

            DefaultModelLoader.load_weights_and_postprocess(
                container, weights, target_device,
            )
        return container

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_batch(self, plan: "BatchPlan") -> dict[str, torch.Tensor | None]:
        """Run the encoder forward for one ``BatchPlan``.

        Determinism guarantee: this method is deterministic given identical
        ``plan`` inputs across ranks, which the EncoderScheduler ensures
        through the two-channel broadcast.
        """
        return plan.adapter.run_feature(self.model, plan)
