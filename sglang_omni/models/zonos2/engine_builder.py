# SPDX-License-Identifier: Apache-2.0
"""ZONOS2 SGLang AR engine builder."""

from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import tempfile
from typing import Any

from sglang_omni.models.zonos2.hf_config import (
    Zonos2Config,
    load_zonos2_pretrained_config,
)
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder
from sglang_omni.utils.checkpoint import resolve_checkpoint

logger = logging.getLogger(__name__)


def _build_config_shim(model_path: str, cfg: Zonos2Config) -> str:
    shim = tempfile.mkdtemp(prefix="zonos2_sglang_")
    atexit.register(shutil.rmtree, shim, ignore_errors=True)
    with open(os.path.join(model_path, "params.json")) as f:
        params = json.load(f)
    params.update(
        architectures=["Zonos2SGLangModel"],
        model_type="zonos2",
        hidden_size=cfg.dim,
        num_hidden_layers=cfg.n_layers,
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_kv_heads,
        head_dim=cfg.head_dim,
        intermediate_size=cfg.intermediate_size,
        vocab_size=cfg.audio_vocab,
        max_position_embeddings=cfg.max_seqlen,
        rms_norm_eps=cfg.norm_eps,
        torch_dtype="bfloat16",
        tie_word_embeddings=False,
    )
    with open(os.path.join(shim, "config.json"), "w") as f:
        json.dump(params, f)
    src = os.path.join(model_path, "model.pth")
    dst = os.path.join(shim, "pytorch_model.bin")
    # Prefer a symlink; fall back to a hardlink, then a copy, where symlinks are
    # unsupported (some network / Windows filesystems).
    try:
        os.symlink(src, dst)
    except (OSError, NotImplementedError):
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
    return shim


def _register_zonos2_autoconfig() -> None:
    from transformers import AutoConfig

    try:
        AutoConfig.register("zonos2", Zonos2Config)
    except (ValueError, KeyError):
        pass  # already registered


def _install_tuned_moe_configs() -> None:
    # note (Yue Yin): the fused-MoE Triton kernel (46% of decode GPU time,
    # profiled) ships no config for this deployment shape (E=16,N=3072 on H100),
    # so it falls back to get_default_config -> "Performance might be sub-optimal".
    # Install the bundled tuned configs into sglang's config dir so the kernel
    # picks them up. Quality-neutral (kernel tiling only); never clobbers an
    # existing config; device/triton-version-keyed filenames auto-ignore on a
    # mismatch (falls back to default). Best-effort: never block startup.
    try:
        from sglang.srt.layers.moe.moe_runner.triton_utils import (
            fused_moe_triton_config as _fc,
        )

        dst_root = os.path.join(
            os.path.dirname(os.path.realpath(_fc.__file__)), "configs"
        )
        src_root = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "moe_configs"
        )
        for vdir in os.listdir(src_root):
            sdir = os.path.join(src_root, vdir)
            if not os.path.isdir(sdir):
                continue
            ddir = os.path.join(dst_root, vdir)
            os.makedirs(ddir, exist_ok=True)
            for fn in os.listdir(sdir):
                dst = os.path.join(ddir, fn)
                if not os.path.exists(dst):
                    shutil.copy2(os.path.join(sdir, fn), dst)
    except Exception:
        logger.warning("Failed to install tuned ZONOS2 MoE configs", exc_info=True)


def _cuda_graph_buckets(max_bs: int) -> list[int]:
    """Power-of-two decode buckets up to max_bs (+ max_bs itself)."""
    bs = [b for b in (1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256) if b <= max_bs]
    if not bs or bs[-1] != max_bs:
        bs.append(max_bs)
    return bs


class Zonos2EngineBuilder(TtsEngineBuilder):
    model_name = "ZONOS2"
    model_arch_override = "Zonos2SGLangModel"

    def __init__(
        self,
        *,
        fp8: bool = False,
        frame_graph: bool = False,
        compile_sampler: bool = False,
        async_decode: bool = False,
        stream_emit_chunk_frames: int = 1,
        stream_emit_first_chunk_frames: int = 0,
        max_running_requests: int = 16,
        cuda_graph_max_bs: int = 16,
        mem_fraction_static: float = 0.5,
    ) -> None:
        self.fp8 = fp8
        self.frame_graph = frame_graph
        self.compile_sampler = compile_sampler
        self.async_decode = async_decode
        self.stream_emit_chunk_frames = stream_emit_chunk_frames
        self.stream_emit_first_chunk_frames = stream_emit_first_chunk_frames
        self.max_running_requests = max_running_requests
        self.cuda_graph_max_bs = cuda_graph_max_bs
        self.mem_fraction_static = mem_fraction_static
        self._cuda_graph_bs: list[int] = []

    def resolve_checkpoint(self, model_path: str) -> str:
        local = resolve_checkpoint(model_path)
        cfg = load_zonos2_pretrained_config(local)
        self.context_length = cfg.max_seqlen
        return _build_config_shim(local, cfg)

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        del checkpoint_dir
        _register_zonos2_autoconfig()
        _install_tuned_moe_configs()

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "max_running_requests": self.max_running_requests,
            "cuda_graph_max_bs": self.cuda_graph_max_bs,
            "disable_cuda_graph": False,
            # async-decode lookahead overlaps the resolve D2H with the next
            # forward; the overlap scheduler must be enabled for it.
            "disable_overlap_schedule": not self.async_decode,
            "enable_torch_compile": True,
            "mem_fraction_static": self.mem_fraction_static,
            "sampling_backend": "pytorch",
            "trust_remote_code": True,
            "dtype": dtype,
        }
        if self.fp8:
            # Dynamic FP8 on the MoE experts (bf16 -> fp8 at load, halving the
            # expert weights); bf16 nn.Linear projections are unaffected.
            defaults["quantization"] = "fp8"
        return defaults

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        self._cuda_graph_bs = _cuda_graph_buckets(int(overrides["cuda_graph_max_bs"]))
        overrides["cuda_graph_bs"] = self._cuda_graph_bs

    def customize_server_args(self, server_args: Any) -> None:
        # note (Chenchen Hong): per-frame feedback/EOS state has no rollback, so a
        # non-final chunked-prefill chunk would queue a spurious frame; disable
        # chunking (mirrors the Qwen3-Omni talker).
        server_args.chunked_prefill_size = 0

    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        del checkpoint_dir, device, gpu_id, server_args
        self.model = model_worker.model_runner.model

    def post_cuda_graph_setup(self, model: Any, server_args: Any) -> None:
        del server_args
        # Opt-in tail CUDA graph: capture the per-frame head+sample+embed+hash
        # tail (otherwise eager in the runner), one graph per decode bucket with
        # the default sampling params; the runner falls back to eager otherwise.
        if self.frame_graph:
            from sglang_omni.models.zonos2.components.text_frontend import (
                TTSSamplingParams,
            )

            model.capture_tail_graphs(self._cuda_graph_bs, TTSSamplingParams())

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        from sglang_omni.models.zonos2.model_runner import Zonos2ModelRunner

        return Zonos2ModelRunner(
            model_worker,
            output_proc,
            compile_sampler=self.compile_sampler,
            frame_graph=self.frame_graph,
            async_decode=self.async_decode,
            stream_emit_chunk_frames=self.stream_emit_chunk_frames,
            stream_emit_first_chunk_frames=self.stream_emit_first_chunk_frames,
        )

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        from sglang_omni.models.zonos2.request_builders import (
            make_zonos2_scheduler_adapters,
        )

        return make_zonos2_scheduler_adapters(model=model)

    def make_abort_callback(self) -> Any | None:
        assert self.model is not None
        return self.model.reset_request

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        return {"enable_async_decode": self.async_decode}

    def post_scheduler_setup(self, scheduler: Any, model_runner: Any) -> None:
        model_runner.set_stream_outbox(scheduler.outbox)
