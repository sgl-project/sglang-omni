# SPDX-License-Identifier: Apache-2.0
"""Follower worker loop for TP ranks > 0."""
from __future__ import annotations

import logging
import multiprocessing as mp
from typing import Any

logger = logging.getLogger(__name__)


def register_omni_models() -> None:
    """Register omni models in SGLang's registry."""
    from sglang_omni.models.sglang_registry import register_omni_models_in_sglang

    register_omni_models_in_sglang()


def relocate_batch_tensors(batch, device) -> None:
    """Move all tensors in *batch* to *device*."""
    import torch

    seen: set[int] = set()

    def _move(obj):
        obj_id = id(obj)
        if obj_id in seen:
            return obj
        seen.add(obj_id)

        if isinstance(obj, torch.Tensor):
            return obj.to(device, non_blocking=True) if obj.device != device else obj
        if isinstance(obj, dict):
            for key, val in obj.items():
                obj[key] = _move(val)
            return obj
        if isinstance(obj, list):
            for i, val in enumerate(obj):
                obj[i] = _move(val)
            return obj
        if isinstance(obj, tuple):
            return tuple(_move(val) for val in obj)
        if hasattr(obj, "__dict__"):
            for attr, val in vars(obj).items():
                moved = _move(val)
                if moved is not val:
                    setattr(obj, attr, moved)
        return obj

    _move(batch)


def sync_page_table(batch, req_to_token_pool) -> None:
    """Write rank 0's page-table snapshot into the follower pool."""
    rows = getattr(batch, "tp_page_table_rows", None)
    if not rows:
        return
    pool_tensor = req_to_token_pool.req_to_token
    for i, row in enumerate(rows):
        idx = int(batch.req_pool_indices[i])
        seq_len = len(row)
        pool_tensor[idx, :seq_len] = row.to(pool_tensor.device)


def patch_batch_for_follower(batch, device, vocab_size: int = 0) -> None:
    """Restore sanitized batch fields and relocate tensors."""
    import torch
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

    relocate_batch_tensors(batch, device)

    if batch.reqs is None:
        batch.reqs = []
    if batch.sampling_info is None:
        bs = len(batch.seq_lens)
        batch.sampling_info = SamplingBatchInfo(
            temperatures=torch.ones(bs, device=device),
            top_ps=torch.ones(bs, device=device),
            top_ks=torch.zeros(bs, dtype=torch.int32, device=device),
            min_ps=torch.zeros(bs, device=device),
            is_all_greedy=True,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            vocab_size=vocab_size,
        )


def follower_worker_loop(
    tp_rank: int,
    gpu_id: int,
    server_args: Any,
    nccl_port: int,
    model_arch_override: str | None = None,
    weight_prefix: str | None = None,
) -> None:
    """Entry point for a follower TP worker process."""
    import torch

    torch.cuda.set_device(gpu_id)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [TP{tp_rank}] %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger(f"tp_follower.{tp_rank}")
    log.info("Starting follower on GPU %d", gpu_id)

    register_omni_models()

    from sglang_omni.engines.ar.sglang_backend.model_worker import (
        ModelWorker,
        ModelWorkerConfig,
    )

    worker = ModelWorker(
        config=ModelWorkerConfig(
            model_arch_override=model_arch_override,
            weight_prefix=weight_prefix,
            nccl_port=nccl_port,
        ),
        server_args=server_args,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
    )

    log.info("ModelWorker initialized, NCCL group joined")

    tp_cpu_group = worker.model_runner.tp_group.cpu_group
    device_group = worker.model_runner.tp_group.device_group

    import torch.distributed as dist
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.utils import broadcast_pyobj

    from sglang_omni.engines.omni.runtime.thinker_forward import thinker_forward_omni

    device = torch.device("cuda", gpu_id)
    model_vocab_size = worker.model_runner.model_config.vocab_size

    model = worker.model_runner.model
    outer_model = model.thinker if hasattr(model, "thinker") else model
    inner_model = getattr(outer_model, "model", outer_model)
    embed_tokens = getattr(inner_model, "embed_tokens", None)
    if embed_tokens is None:
        get_input_embeddings = getattr(inner_model, "get_input_embeddings", None)
        if callable(get_input_embeddings):
            embed_tokens = get_input_embeddings()
    if embed_tokens is None:
        embed_tokens = getattr(inner_model, "codec_embedding", None)

    step = 0
    while True:
        result = broadcast_pyobj([None], tp_rank, tp_cpu_group, src=0)
        batch = result[0] if result else None

        if batch is None:
            log.info("Received stop signal after %d steps", step)
            break

        patch_batch_for_follower(batch, device, vocab_size=model_vocab_size)
        sync_page_table(batch, worker.model_runner.req_to_token_pool)
        forward_batch = ForwardBatch.init_new(batch, worker.model_runner)

        has_mm_payload = getattr(batch, "tp_has_mm_payload", False)

        if has_mm_payload:
            # Symmetric base embed — mirrors rank 0's call exactly so the
            # VocabParallelEmbedding all_reduce pairs.
            embed_input_ids = forward_batch.input_ids.clamp(
                0, embed_tokens.num_embeddings - 1
            )
            _ = embed_tokens(embed_input_ids)

            meta_result = broadcast_pyobj([None], tp_rank, tp_cpu_group, src=0)
            payload_meta = meta_result[0] if meta_result else None
            if payload_meta is None:
                raise RuntimeError("Missing multimodal TP payload metadata from rank 0")

            input_embeds_shape = payload_meta["input_embeds_shape"]
            input_embeds = torch.empty(
                input_embeds_shape,
                dtype=payload_meta["input_embeds_dtype"],
                device=device,
            )
            dist.broadcast(input_embeds, src=0, group=device_group)

            ds_embeds = None
            ds_shapes = payload_meta["deepstack_shapes"]
            if ds_shapes is not None:
                ds_dtype = payload_meta["deepstack_dtype"]
                ds_embeds = [
                    torch.empty(shape, dtype=ds_dtype, device=device)
                    for shape in ds_shapes
                ]
                for t in ds_embeds:
                    dist.broadcast(t, src=0, group=device_group)

            vis_masks = None
            vis_mask_shape = payload_meta["visual_pos_mask_shape"]
            if vis_mask_shape is not None:
                vis_masks = torch.empty(vis_mask_shape, dtype=torch.bool, device=device)
                dist.broadcast(vis_masks, src=0, group=device_group)

            thinker_forward_omni(
                outer_model=outer_model,
                attn_backend=worker.model_runner.attn_backend,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
                deepstack_visual_embeds=ds_embeds,
                visual_pos_masks=vis_masks,
            )
        else:
            worker.model_runner.forward(forward_batch=forward_batch)
        step += 1

    log.info("Follower exiting")


def spawn_followers(
    server_args: Any,
    nccl_port: int,
    base_gpu_id: int,
    tp_size: int,
    model_arch_override: str | None = None,
    weight_prefix: str | None = None,
) -> list[mp.Process]:
    """Spawn TP follower processes."""
    processes = []
    ctx = mp.get_context("spawn")

    for rank in range(1, tp_size):
        gpu_id_step = getattr(server_args, "gpu_id_step", 1)
        gpu_id = base_gpu_id + rank * gpu_id_step
        proc = ctx.Process(
            target=follower_worker_loop,
            args=(
                rank,
                gpu_id,
                server_args,
                nccl_port,
                model_arch_override,
                weight_prefix,
            ),
            daemon=True,
        )
        proc.start()
        processes.append(proc)
        logger.info(
            "Spawned follower rank %d on GPU %d (pid=%d)", rank, gpu_id, proc.pid
        )

    return processes
