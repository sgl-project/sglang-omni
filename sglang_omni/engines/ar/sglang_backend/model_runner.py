from __future__ import annotations

import os

from sglang_omni.vendor.sglang.core import (
    ModelConfig,
    ModelRunner,
    PortArgs,
    ServerArgs,
)


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
    ) -> None:
        self.model_config = model_config
        self.server_args = server_args
        self.gpu_id = gpu_id

        self._register_omni_model()

        model_config = server_args.get_model_config()
        port_args = PortArgs.init_new(server_args)

        tp_size = server_args.tp_size

        self.nccl_port = port_args.nccl_port

        super().__init__(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=tp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=moe_ep_size,
            pp_rank=pp_rank,
            pp_size=pp_size,
            nccl_port=nccl_port,
            server_args=server_args,
        )

    def _register_omni_model(self):
        # Ensure sglang_omni.models` custom modeling classes
        # are injected into the model registry before runner initialization.
        os.environ.setdefault("SGLANG_EXTERNAL_MODEL_PACKAGE", "sglang_omni.models")
