# SPDX-License-Identifier: Apache-2.0
"""Audio encoder backed by sglang main repo's native implementation.

Swaps HF's ``Qwen3OmniMoeAudioEncoder`` for
``sglang.srt.models.qwen3_omni_moe.Qwen3OmniMoeAudioEncoder`` which uses
fused QKV (:class:`QKVParallelLinear`), :class:`ColumnParallelLinear` +
:class:`RowParallelLinear` for the FFN, and the :class:`VisionAttention`
kernel dispatch (fa3 / triton / aiter / ascend).

Intended usage
--------------

Two-step initialization::

    init_sglang_env_for_encoder(model_path)    # populates dist + mp + server_args
    enc = Qwen3OmniAudioEncoderNative(model_path, device=..., dtype=...)

``init_sglang_env_for_encoder`` may also be called from outside this module
(e.g. by a benchmark harness that has its own reasons to touch dist state).
It is idempotent and safe to call multiple times.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn

from sglang_omni.models.qwen3_omni.components.common import load_thinker_config
from sglang_omni.models.weight_loader import load_weights_by_prefix, resolve_dtype

logger = logging.getLogger(__name__)

AUDIO_TOWER_PREFIX = ("thinker.audio_tower.", "audio_tower.")


def init_sglang_env_for_encoder(model_path: str) -> None:
    """Idempotently initialize sglang's global distributed / server-args state.

    The sglang main repo's encoder layers (``VisionAttention``,
    ``ColumnParallelLinear``) read from process-level singletons
    (``_WORLD``, ``_ATTN_TP_*``, global server args). Those are normally
    initialized by sglang's scheduler; when running this encoder standalone
    (tests, benchmarks, single-process serving), the caller must bootstrap
    them first.

    Reads world topology from torchrun env vars (``WORLD_SIZE`` / ``RANK`` /
    ``LOCAL_RANK``) when present; defaults to single-process TP=1 otherwise.
    If ``MASTER_PORT`` is unset, picks a free ephemeral port (matches the
    pattern in ``sglang_omni.engines.ar.sglang_backend.model_worker._resolve_nccl_port``)
    so concurrent standalone users don't clobber each other.
    """
    import socket

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in os.environ:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            os.environ["MASTER_PORT"] = str(sock.getsockname()[1])
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    import sglang.srt.distributed.parallel_state as _ps
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.distributed.parallel_state import model_parallel_is_initialized
    from sglang.srt.layers import dp_attention as _dp
    from sglang.srt.server_args import (
        ServerArgs,
        get_global_server_args,
        set_global_server_args_for_scheduler,
    )

    # Server args: ``VisionAttention`` reads ``mm_attention_backend`` from here.
    try:
        get_global_server_args()
    except ValueError:
        set_global_server_args_for_scheduler(ServerArgs(model_path=model_path))

    # World group.
    if _ps._WORLD is None:
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            backend="nccl",
        )

    # Model parallel group.
    if not model_parallel_is_initialized():
        initialize_model_parallel(tensor_model_parallel_size=world_size)

    # DP attention globals. ``initialize_dp_attention`` requires a full
    # ModelConfig; for standalone encoder use we populate the scalars directly.
    if _dp._ATTN_TP_SIZE is None:
        _dp._ATTN_TP_SIZE = world_size
        _dp._ATTN_TP_RANK = rank
        _dp._ATTN_DP_SIZE = 1
        _dp._ATTN_DP_RANK = 0
        _dp._LOCAL_ATTN_DP_SIZE = 1
        _dp._LOCAL_ATTN_DP_RANK = 0
        try:
            from sglang.srt.distributed.parallel_state import get_tp_group

            _dp._ATTN_TP_GROUP = get_tp_group()
        except (AssertionError, RuntimeError):
            # TP group not yet available (e.g. TP=1, some versions don't
            # create it). VisionAttention at TP=1 doesn't need the group.
            _dp._ATTN_TP_GROUP = None


def _load_sglang_weights(
    audio_tower: nn.Module,
    hf_state_dict: dict[str, torch.Tensor],
) -> None:
    """Load HF audio_tower weights into sglang native module.

    Uses sglang's per-param ``weight_loader(param, tensor, shard_id=...)`` for
    TP-sharded Linears so this works at TP>1 too.

    Mapping
    -------
    ``self_attn.{q,k,v}_proj.{w,b}`` → ``self_attn.qkv_proj.{w,b}`` (shard_id)
    ``self_attn.out_proj.{w,b}``      → ``self_attn.proj.{w,b}``   (RowParallel)
    ``fc1.{w,b}``                     → unchanged                  (ColumnParallel)
    ``fc2.{w,b}``                     → unchanged                  (RowParallel)
    rest                              → ``default_weight_loader``

    Raises
    ------
    RuntimeError
        If any HF weight cannot be placed, or if any native trainable param
        (``.weight`` / ``.bias``) remains at its random-init value after the
        load (silent random-weight inference would be a data-corruption hazard).
    """
    from sglang_omni.models.weight_loader import default_weight_loader

    params = dict(audio_tower.named_parameters())
    unplaced: list[str] = []
    touched: set[str] = set()

    for name, tensor in hf_state_dict.items():
        shard_id: str | None = None
        target = name
        if ".self_attn." in name and (
            ".q_proj." in name or ".k_proj." in name or ".v_proj." in name
        ):
            parts = name.split(".")
            layer_idx = parts[1]
            proj = parts[3]  # q_proj/k_proj/v_proj
            pname = parts[-1]  # weight/bias
            shard_id = proj[0]  # q / k / v
            target = f"layers.{layer_idx}.self_attn.qkv_proj.{pname}"
        elif ".self_attn.out_proj." in name:
            target = name.replace(".self_attn.out_proj.", ".self_attn.proj.")

        if target not in params:
            unplaced.append(name)
            continue

        param = params[target]
        loader = getattr(param, "weight_loader", None)
        if loader is not None and shard_id is not None:
            loader(param, tensor, shard_id)
        elif loader is not None:
            loader(param, tensor)
        else:
            default_weight_loader(param, tensor)
        touched.add(target)

    if unplaced:
        raise RuntimeError(
            f"Audio encoder weight load: {len(unplaced)} HF keys could not be "
            f"placed into sglang native module. First few: {unplaced[:5]}. "
            f"Checkpoint/schema drift — refusing to run with partial weights."
        )

    # Strict validation: every trainable param must have been loaded.
    missing = [
        n
        for n in params
        if n not in touched and (n.endswith(".weight") or n.endswith(".bias"))
    ]
    if missing:
        raise RuntimeError(
            f"Audio encoder weight load: {len(missing)} native params still "
            f"at random-init after load. First few: {missing[:5]}. Refusing "
            f"to run (silent corruption risk)."
        )


def _build_audio_tower_native(
    model_path: str,
    *,
    thinker_cfg: object,
    torch_dtype: torch.dtype | None,
    device: str,
) -> nn.Module:
    init_sglang_env_for_encoder(model_path)

    from sglang.srt.models.qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoder as SGLangAudioEncoder,
    )

    audio_cfg = thinker_cfg.audio_config
    # sglang's VisionAttention replaces HF's attention dispatch entirely;
    # HF's attn_implementation check is dead code here. HF's PreTrainedModel
    # __init__ reads both of these fields at different points (the public
    # one is resolved to the "_internal" one during __init__), so we set
    # both to "eager" to skip the _supports_sdpa check that the sglang
    # subclass doesn't declare.
    audio_cfg._attn_implementation = "eager"
    audio_cfg._attn_implementation_internal = "eager"

    audio_tower = SGLangAudioEncoder(audio_cfg)

    hf_sd = load_weights_by_prefix(model_path, prefix=AUDIO_TOWER_PREFIX)
    if not hf_sd:
        raise RuntimeError(
            f"No audio_tower weights found for prefixes={AUDIO_TOWER_PREFIX}"
        )
    _load_sglang_weights(audio_tower, hf_sd)

    audio_tower.eval()
    if torch_dtype is not None:
        audio_tower = audio_tower.to(dtype=torch_dtype)
    audio_tower = audio_tower.to(device=device)
    return audio_tower


class Qwen3OmniAudioEncoderNative(nn.Module):
    """Audio tower wrapper backed by sglang native ``Qwen3OmniMoeAudioEncoder``.

    Assumes :func:`init_sglang_env_for_encoder` has been called (or will be
    called implicitly by ``__init__``).
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str | torch.dtype | None = None,
    ) -> None:
        super().__init__()
        torch_dtype = resolve_dtype(dtype)
        thinker_cfg = load_thinker_config(model_path)
        self._device = torch.device(device)
        self.audio_tower = _build_audio_tower_native(
            model_path,
            thinker_cfg=thinker_cfg,
            torch_dtype=torch_dtype,
            device=device,
        )
        self._param_dtype = next(self.audio_tower.parameters()).dtype

    def forward(
        self,
        *,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor | None = None,
        audio_feature_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        from sglang.srt.models.qwen3_omni_moe import _get_feat_extract_output_lengths

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = (
                input_features.permute(0, 2, 1)[feature_attention_mask.bool()]
                .permute(1, 0)
                .contiguous()
            )
        if audio_feature_lengths is None:
            raise ValueError(
                "audio_feature_lengths or feature_attention_mask is required"
            )

        audio_feature_lengths = audio_feature_lengths.to(
            self._device, dtype=torch.long
        )
        outputs = self.audio_tower(
            input_features.to(device=self._device, dtype=self._param_dtype),
            feature_lens=audio_feature_lengths,
        )
        audio_embeds = outputs.last_hidden_state
        audio_output_lengths = _get_feat_extract_output_lengths(
            audio_feature_lengths
        )
        return {
            "audio_embeds": audio_embeds,
            "audio_feature_lengths": audio_feature_lengths,
            "audio_output_lengths": audio_output_lengths,
        }
