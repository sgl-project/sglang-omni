# SPDX-License-Identifier: Apache-2.0
"""Factory functions for creating OmniEngine instances."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from .engine import OmniEngine
from .model_runner import ModelRunner
from .runtime.ar import (
    ARBatchPlanner,
    ARInputPreparer,
    AROutputProcessor,
    ARResourceManager,
)
from .runtime.cache import SimpleCacheManager
from .runtime.common import (
    EosIterationController,
    SimpleResourceManager,
    SinglePassIterationController,
)
from .runtime.encoder import (
    EncoderBatchPlanner,
    EncoderInputPreparer,
    EncoderOutputProcessor,
)
from .scheduler import Scheduler
from .types import ModelRunnerOutput, RequestOutput, SchedulerOutput, SchedulerRequest

if TYPE_CHECKING:
    from sglang_omni.models.qwen3_omni.pipeline.engine_io import TalkerRequestData
    from sglang_omni.models.qwen3_omni.talker import Qwen3OmniTalker

logger = logging.getLogger(__name__)


def create_encoder_engine(
    model: torch.nn.Module,
    tokenizer: Any = None,
    max_batch_size: int = 32,
    pooling: str = "last",
    device: str = "cuda",
    use_cache: bool = False,
    cache_size: int | None = None,
) -> OmniEngine:
    """Create an encoder engine.

    Args:
        model: The encoder model (e.g., BERT, RoBERTa)
        tokenizer: Optional tokenizer (used to get pad_token_id)
        max_batch_size: Maximum batch size for scheduling
        pooling: Pooling strategy - "last", "mean", or "cls"
        device: Device to run on
        use_cache: Enable encoder output cache
        cache_size: Max cache entries (None for unbounded)

    Returns:
        OmniEngine configured for encoder models

    Example:
        from transformers import BertModel, BertTokenizer

        model = BertModel.from_pretrained("bert-base-uncased")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        engine = create_encoder_engine(model, tokenizer)
        await engine.start()

        # Create request data
        input_ids = tokenizer.encode("Hello world", return_tensors="pt")
        data = EncoderRequestData(input_ids=input_ids[0])

        await engine.add_request("req-1", data)
        result = await engine.get_result("req-1")  # Returns embeddings tensor
    """
    pad_token_id = tokenizer.pad_token_id if tokenizer is not None else 0

    scheduler = Scheduler(
        batch_planner=EncoderBatchPlanner(max_batch_size=max_batch_size),
        resource_manager=SimpleResourceManager(max_count=max_batch_size),
        iteration_controller=SinglePassIterationController(),
    )

    # Create model runner (stateless)
    model_runner = ModelRunner(
        model=model,
        input_preparer=EncoderInputPreparer(pad_token_id=pad_token_id),
        output_processor=EncoderOutputProcessor(pooling=pooling),
        device=device,
    )

    # Create cache manager (if needed)
    cache_manager = None
    if use_cache:
        cache_manager = SimpleCacheManager(max_size=cache_size)

    return OmniEngine(
        scheduler=scheduler, model_runner=model_runner, cache_manager=cache_manager
    )


def create_ar_engine(
    model: torch.nn.Module,
    tokenizer: Any = None,
    max_seq_len: int = 2048,
    device: str = "cuda",
) -> OmniEngine:
    """Create an AR engine (single request, HF KV cache).

    Args:
        model: The causal LM model (e.g., LLaMA, GPT-2)
        tokenizer: Tokenizer (used to get eos_token_id)
        max_seq_len: Maximum sequence length
        device: Device to run on

    Returns:
        OmniEngine configured for AR models

    Example:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        engine = create_ar_engine(model, tokenizer)
        await engine.start()

        # Create request data
        input_ids = tokenizer.encode("Once upon a time", return_tensors="pt")
        data = ARRequestData(
            input_ids=input_ids[0],
            max_new_tokens=256,
            temperature=0.7,
        )

        await engine.add_request("req-1", data)
        result = await engine.get_result("req-1")  # ARRequestData with output_ids

        generated_text = tokenizer.decode(result.output_ids)
    """
    eos_token_id = tokenizer.eos_token_id if tokenizer is not None else 2

    def _stream_adapter(request, output):
        token = output.data
        if isinstance(token, tuple):
            token = token[0]
        if token is None:
            return None
        return int(token)

    scheduler = Scheduler(
        batch_planner=ARBatchPlanner(),
        resource_manager=ARResourceManager(max_count=1),
        iteration_controller=EosIterationController(
            eos_token_id=eos_token_id,
            max_length=max_seq_len,
        ),
        stream_adapter=_stream_adapter,
    )

    # Create model runner
    model_runner = ModelRunner(
        model=model,
        input_preparer=ARInputPreparer(),
        output_processor=AROutputProcessor(),
        device=device,
    )

    return OmniEngine(scheduler=scheduler, model_runner=model_runner)


def create_sglang_ar_engine(
    server_args: Any,
    gpu_id: int = 0,
    capture_hidden_layers: list[int] | None = None,
) -> OmniEngine:
    """Create an AR engine backed by SGLang's ModelWorker and KV cache.

    Uses SGLang's PrefillManager, DecodeManager, and paged KV cache for
    continuous batching with chunked prefill support.

    Args:
        server_args: SGLang ServerArgs configuration
        gpu_id: GPU device ID

    Returns:
        OmniEngine configured with SGLang backend
    """
    from sglang_omni.engines.ar.sglang_backend.model_worker import (
        ModelWorker,
        ModelWorkerConfig,
    )
    from sglang_omni.engines.ar.sglang_backend.scheduler.cache import create_tree_cache
    from sglang_omni.engines.ar.sglang_backend.scheduler.decode import DecodeManager
    from sglang_omni.engines.ar.sglang_backend.scheduler.prefill import PrefillManager

    from .runtime.sglang_ar import (
        SGLangBatchPlanner,
        SGLangIterationController,
        SGLangModelRunner,
        SGLangOutputProcessor,
        SGLangResourceManager,
    )

    # Initialize model worker
    model_worker = ModelWorker(
        config=ModelWorkerConfig(),
        server_args=server_args,
        gpu_id=gpu_id,
    )

    # Get memory pools
    req_to_token_pool, token_to_kv_pool_allocator = model_worker.get_memory_pool()

    # Create tree cache
    tree_cache = create_tree_cache(
        server_args,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        server_args.page_size,
    )

    # Create prefill and decode managers
    prefill_mgr = PrefillManager(
        page_size=server_args.page_size,
        chunked_prefill_size=server_args.chunked_prefill_size,
        max_prefill_tokens=server_args.max_prefill_tokens,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        model_config=model_worker.model_config,
        enable_overlap=False,
    )
    decode_mgr = DecodeManager(
        server_args=server_args,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        on_retract=lambda req: prefill_mgr.add_one_request(req),
    )

    # Assemble SGLang-specific components
    batch_planner = SGLangBatchPlanner(prefill_mgr, decode_mgr, server_args)
    resource_mgr = SGLangResourceManager(
        token_to_kv_pool_allocator, req_to_token_pool, tree_cache
    )
    iteration_ctrl = SGLangIterationController(tree_cache)
    output_proc = SGLangOutputProcessor()

    def _stream_adapter(request, output):
        if request.data.req.is_chunked > 0:
            return None
        token = output.data
        return int(token) if token is not None else None

    scheduler = Scheduler(
        batch_planner=batch_planner,
        resource_manager=resource_mgr,
        iteration_controller=iteration_ctrl,
        stream_adapter=_stream_adapter,
    )

    decoder_layers = None
    embed_tokens = None
    if capture_hidden_layers:
        inner = model_worker.model_runner.model.thinker.model
        decoder_layers = inner.layers
        embed_tokens = inner.embed_tokens

    sglang_model_runner = SGLangModelRunner(
        model_worker,
        output_proc,
        batch_planner=batch_planner,
        capture_hidden_layers=capture_hidden_layers,
        decoder_layers=decoder_layers,
        embed_tokens=embed_tokens,
    )

    return OmniEngine(scheduler=scheduler, model_runner=sglang_model_runner)


def create_talker_codec_engine(
    talker_model: "Qwen3OmniTalker",
    gpu_id: int = 0,
) -> OmniEngine:
    """Create a talker engine for single-pass codec generation.

    Reuses generic scheduler components (SimpleResourceManager,
    SinglePassIterationController) instead of talker-specific ones.

    Args:
        talker_model: Pre-loaded Qwen3OmniTalker model instance.
        gpu_id: GPU device ID.

    Returns:
        OmniEngine configured for talker codec generation.
    """
    device = torch.device(f"cuda:{gpu_id}")

    scheduler = Scheduler(
        batch_planner=_SingleRequestBatchPlanner(),
        resource_manager=SimpleResourceManager(max_count=1),
        iteration_controller=SinglePassIterationController(),
    )

    codec_runner = TalkerCodecRunner(talker_model=talker_model, device=device)

    return OmniEngine(scheduler=scheduler, model_runner=codec_runner)


class TalkerCodecRunner:
    """Model runner that generates multi-layer RVQ codec codes.

    Pipeline:
      1. Project thinker outputs to talker hidden dim
      2. Forward through talker backbone (single pass)
      3. Layer-0 codes via argmax on logits
      4. Layers 1-15 via code predictor

    Note (chenyang): TalkerCodecRunner does not use SGLang ModelWorker,
    since talker runs a single full-sequence forward, no KV cache needed,
    variable input sequence lengths hurts CUDA graph, and so on.
    """

    def __init__(
        self,
        talker_model: "Qwen3OmniTalker",
        device: torch.device,
    ) -> None:
        self.talker_model = talker_model
        self.device = device

    @torch.inference_mode()
    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        if not scheduler_output.requests:
            return ModelRunnerOutput(outputs={}, req_ids=[], req_id_to_index={})

        request = scheduler_output.requests[0]
        codec_output = self._generate_codec(request.data)

        outputs = {
            request.request_id: RequestOutput(
                request_id=request.request_id,
                data=codec_output,
                finished=True,
                finish_reason="stop",
            ),
        }
        return ModelRunnerOutput(
            outputs=outputs,
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
        )

    def _generate_codec(
        self, request_data: "TalkerRequestData"
    ) -> dict[str, torch.Tensor]:
        """Run full codec generation pipeline for one request."""
        input_embeds = self._prepare_embeds(request_data)
        hidden_states = self._run_backbone(input_embeds)

        # Layer-0 codes: codec_head logits → argmax
        logits = self.talker_model.compute_logits(hidden_states.squeeze(0))
        layer0_codes = logits.argmax(dim=-1).unsqueeze(0)

        # Layers 1-15 via code predictor
        codec_codes, summed_embeds = self.talker_model.code_predictor_forward(
            layer0_codes=layer0_codes,
            talker_hidden=hidden_states,
        )

        return {
            "codec_codes": codec_codes.squeeze(0),
            "summed_codec_embeds": summed_embeds.squeeze(0),
        }

    def _prepare_embeds(self, request_data: "TalkerRequestData") -> torch.Tensor:
        """Project thinker outputs to talker hidden dim. Returns [1, seq, hidden]."""
        is_multimodal_mask = None
        if request_data.is_multimodal_mask is not None:
            is_multimodal_mask = request_data.is_multimodal_mask.to(self.device)

        input_embeds = self.talker_model.prepare_input_embeds(
            thinker_embeds=request_data.thinker_embed.to(self.device),
            thinker_hidden_states=request_data.thinker_hidden.to(self.device),
            is_multimodal_mask=is_multimodal_mask,
        )
        if input_embeds.dim() == 2:
            input_embeds = input_embeds.unsqueeze(0)
        return input_embeds

    def _run_backbone(self, input_embeds: torch.Tensor) -> torch.Tensor:
        """Run talker backbone forward."""
        seq_len = input_embeds.shape[1]
        flat_embeds = input_embeds.squeeze(0)
        positions = torch.arange(seq_len, device=self.device)
        forward_batch = _MockForwardBatch(seq_len, self.device)

        hidden_states = self.talker_model.forward(
            input_ids=torch.zeros(seq_len, dtype=torch.long, device=self.device),
            positions=positions,
            forward_batch=forward_batch,
            inputs_embeds=flat_embeds,
        )
        return hidden_states.unsqueeze(0)


class _SingleRequestBatchPlanner:
    """Selects one request per step."""

    def select_requests(
        self,
        waiting: list[SchedulerRequest],
        running: list[SchedulerRequest],
        resource_manager: Any,
    ) -> list[SchedulerRequest]:
        if running:
            return [running[0]]
        if not waiting:
            return []
        request = waiting[0]
        if not resource_manager.can_allocate(request):
            return []
        resource_manager.allocate(request)
        return [request]

    def build_batch(self, requests: list[SchedulerRequest]) -> None:
        return None


class _SDPAWrapper(torch.nn.Module):
    """Drop-in replacement for RadixAttention using scaled dot-product attention."""

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: Any,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        seq_len = q.shape[0]
        q = q.view(seq_len, self.num_heads, self.head_dim).unsqueeze(0).transpose(1, 2)
        k = (
            k.view(seq_len, self.num_kv_heads, self.head_dim)
            .unsqueeze(0)
            .transpose(1, 2)
        )
        v = (
            v.view(seq_len, self.num_kv_heads, self.head_dim)
            .unsqueeze(0)
            .transpose(1, 2)
        )

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return out.transpose(1, 2).squeeze(0).reshape(seq_len, -1)


def patch_talker_attention(talker_model: "Qwen3OmniTalker") -> None:
    """Replace RadixAttention in the talker backbone with SDPA wrappers."""
    text_model = talker_model.model
    for layer in text_model.layers:
        attn_mod = layer.self_attn
        sdpa = _SDPAWrapper(
            num_heads=attn_mod.num_heads,
            num_kv_heads=attn_mod.num_kv_heads,
            head_dim=attn_mod.head_dim,
        )
        attn_mod.attn = sdpa

        rotary = attn_mod.rotary_emb
        if rotary.cos_sin_cache.dtype != torch.float32:
            rotary.cos_sin_cache = rotary.cos_sin_cache.to(torch.float32)

        logger.info(
            "Patched talker layer %d attention: RadixAttention → SDPA",
            layer.layer_id,
        )


class _ExtendForwardMode:
    """Stub forward mode that reports extend (prefill) mode."""

    def is_extend(self) -> bool:
        return True

    def is_decode(self) -> bool:
        return False

    def is_idle(self) -> bool:
        return False


class _MockForwardBatch:
    """Minimal ForwardBatch substitute for the SDPA-patched talker backbone.

    Note (chenyang): The talker backbone does not use SGLang's inference
    pipeline fully, including KV cache, radix tree, etc.  The attention layers
    have been patched to plain SDPA (Scaled Dot-Product Attention ).  However
    the decoder layer forward() signature still expects a ForwardBatch
    object, so we provide this mock with the minimal fields required by
    forward_prepare and LayerCommunicator.
    """

    def __init__(self, seq_len: int, device: torch.device) -> None:
        self.seq_lens = torch.tensor([seq_len], device=device)
        self.req_pool_indices = torch.zeros(1, dtype=torch.long, device=device)
        self.seq_lens_sum = seq_len
        self.out_cache_loc = torch.arange(seq_len, device=device)
        self.forward_mode = _ExtendForwardMode()
        self.total_num_tokens = seq_len
        self.input_ids = torch.zeros(seq_len, dtype=torch.long, device=device)
        self.batch_size = 1
        self.is_extend_in_batch = True
        self.global_num_tokens_cpu = None
        self.num_token_non_padded = None
        self.can_run_dp_cuda_graph = False
        self.token_to_kv_pool = None
        self.nsa_cp_metadata = None
        self.extend_prefix_lens = None
        self.extend_seq_lens = None
        self.extend_logprob_start_lens = None
        self.top_logprobs_nums = None
        self.return_logprob = None
        self.positions = None
        self.mrope_positions = None
        self.spec_info = None
        self.capture_hidden_mode = None
