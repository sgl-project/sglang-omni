# SPDX-License-Identifier: Apache-2.0
"""Build the native duplex thinker on the shared generation engine."""

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from transformers import PreTrainedTokenizerBase

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.minicpm_o.components.sglang_thinker import (
    MiniCPMOThinkerForCausalLM,
)
from sglang_omni.models.minicpm_o.hf_config import register_minicpm_o_hf_config
from sglang_omni.models.minicpm_o.native_thinker_model_runner import (
    MiniCPMOThinkerModelRunner,
)
from sglang_omni.models.minicpm_o.session_adapters import ThinkerAdapter
from sglang_omni.scheduling.engine_factory import SGLangGenerationEngineBuilder
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor


class MiniCPMOThinkerEngineBuilder(SGLangGenerationEngineBuilder):
    model_name: str = "MiniCPM-o thinker"
    model_arch_override: str = "MiniCPMO"
    context_length: int = 8192
    tokenizer: PreTrainedTokenizerBase
    adapter: ThinkerAdapter

    def generation_defaults(self, *, dtype: str) -> dict[str, str | int | float | bool]:
        return dict(
            max_running_requests=4,
            dtype=dtype,
            enable_streaming_session=True,
            disable_overlap_schedule=True,
            disable_cuda_graph=True,
            chunked_prefill_size=-1,
            enable_return_hidden_states=True,
            sampling_backend="pytorch",
            mem_fraction_static=0.45,
            trust_remote_code=False,
        )

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        register_minicpm_o_hf_config()
        ModelRegistry.models["MiniCPMO"] = MiniCPMOThinkerForCausalLM
        self.tokenizer = get_tokenizer(checkpoint_dir, trust_remote_code=True)

    def make_model_runner(
        self, model_worker: ModelWorker, output_proc: SGLangOutputProcessor
    ) -> MiniCPMOThinkerModelRunner:
        return MiniCPMOThinkerModelRunner(model_worker, output_proc)

    def make_adapters(self, model: MiniCPMOThinkerForCausalLM) -> tuple[None, None]:
        self.adapter = ThinkerAdapter(self.tokenizer, model.config.vocab_size)
        return None, None

    def extra_scheduler_kwargs(self) -> dict[str, ThinkerAdapter | int]:
        return dict(session_adapter=self.adapter, request_build_max_workers=1)

    def build_runtime(
        self,
        *,
        model_worker: ModelWorker,
        model: MiniCPMOThinkerForCausalLM,
        output_proc: SGLangOutputProcessor,
        tree_cache: BasePrefixCache,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        server_args: ServerArgs,
        model_config: ModelConfig,
    ) -> tuple[OmniScheduler, MiniCPMOThinkerModelRunner]:
        output_proc = SGLangOutputProcessor(capture_hidden=True)
        return super().build_runtime(
            model_worker=model_worker,
            model=model,
            output_proc=output_proc,
            tree_cache=tree_cache,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            server_args=server_args,
            model_config=model_config,
        )
