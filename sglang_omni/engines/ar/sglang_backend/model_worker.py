from dataclasses import dataclass

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import broadcast_pyobj, set_random_seed

from .model_runner import SGLModelRunner


@dataclass
class ModelWorkerConfig:
    pass


class ModelWorker:
    def __init__(
        self,
        config: ModelWorkerConfig,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int = 0,
    ):
        self.server_args = server_args

        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self._init_model_config()
        self._init_model_runner()

        self.device = self.model_runner.device
        self.random_seed = broadcast_pyobj(
            [server_args.random_seed],
            self.tp_rank,
            self.model_runner.tp_group.cpu_group,
        )[0]
        set_random_seed(self.random_seed)

    def _init_model_config(self):
        # TODO(ocss884) add support for mtp
        self.model_config = ModelConfig.from_server_args(
            server_args=self.server_args,
            model_path=self.server_args.model_path,
            model_revision=self.server_args.revision,
            is_draft_model=False,
        )

    def get_memory_pool(self):
        return (
            self.model_runner.req_to_token_pool,
            self.model_runner.token_to_kv_pool_allocator,
        )

    def _init_model_runner(self):
        self.model_runner = SGLModelRunner(
            model_config=self.model_config,
            server_args=self.server_args,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            moe_ep_rank=0,
            moe_ep_size=1,
            pp_rank=0,
            pp_size=1,
            nccl_port=30000,
        )

    def forward_batch_generation(
        self,
        forward_batch,
    ):
        out = self.model_runner.forward(forward_batch=forward_batch)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        batch_result = GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=can_run_cuda_graph,
            expert_distribution_metrics=out.expert_distribution_metrics,
        )
        return batch_result
