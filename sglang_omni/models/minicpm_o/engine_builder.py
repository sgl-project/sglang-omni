from sglang.srt.models.registry import ModelRegistry
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

from sglang_omni.models.minicpm_o.components.sglang_thinker import (
    MiniCPMOThinkerForCausalLM,
)
from sglang_omni.models.minicpm_o.hf_config import register_minicpm_o_hf_config
from sglang_omni.models.minicpm_o.native_thinker_model_runner import (
    MiniCPMOThinkerModelRunner,
)
from sglang_omni.models.minicpm_o.session_adapters import ThinkerAdapter

# SPDX-License-Identifier: Apache-2.0
from sglang_omni.scheduling.engine_factory import SGLangGenerationEngineBuilder
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor


class MiniCPMOThinkerEngineBuilder(SGLangGenerationEngineBuilder):
    model_name = "MiniCPM-o thinker"
    model_arch_override = "MiniCPMO"
    context_length = 8192

    def generation_defaults(self, *, dtype):
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

    def pre_infra_setup(self, checkpoint_dir):

        register_minicpm_o_hf_config()
        ModelRegistry.models["MiniCPMO"] = MiniCPMOThinkerForCausalLM
        self.tokenizer = get_tokenizer(checkpoint_dir, trust_remote_code=True)

    def make_model_runner(self, model_worker, output_proc):

        return MiniCPMOThinkerModelRunner(model_worker, output_proc)

    def make_adapters(self, model):

        self.adapter = ThinkerAdapter(self.tokenizer, model.config.vocab_size)
        return None, None

    def extra_scheduler_kwargs(self):
        return dict(session_adapter=self.adapter, request_build_max_workers=1)

    def build_runtime(
        self,
        *,
        model_worker,
        model,
        output_proc,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        server_args,
        model_config,
    ):

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
