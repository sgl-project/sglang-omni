# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding for Whisper with a separate draft decoder that shares the target's encoder.

A request's encoder slots lead its req_to_token row, so every draft and verify forward here runs through the encoder-aware attention paths instead of SGLang's speculative workers, which address slots by decoder length alone.
"""

from __future__ import annotations

import copy
import glob
import json
import os
from collections.abc import Iterable
from tempfile import TemporaryDirectory
from typing import Protocol

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from sglang.srt.layers.moe.utils import draft_model_build_scope
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.mem_cache.kv_loc_plan import IdSpaceKind, KVLocPlan
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.runtime_context import get_device, get_flags, get_spec
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker, EagleDraftWorkerBase
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.spec_info import SpecInput, SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import draft_pp_context
from torch import nn
from transformers import WhisperConfig

from sglang_omni.models.whisper_asr.sglang_model import (
    LogitsProcessor,
    WhisperDecoderModel,
    WhisperForConditionalGeneration,
)
from sglang_omni.speculative.target_worker import SpeculativeTargetWorker

DRAFT_ARCHITECTURE = "WhisperDraftForCausalLM"
SHARED_ENCODER_TENSOR_NAMES = (
    "model.encoder.conv1.weight",
    "model.encoder.layer_norm.weight",
)


def materialize_draft_checkpoint(
    draft_model_path: str, revision: str | None
) -> TemporaryDirectory[str]:
    """Write a loadable copy of a Whisper draft that names the decoder-only draft class.

    WhisperForConditionalGeneration stays as a second architecture, so SGLang still serves the draft as an encoder-decoder model, and the config drops the encoder layers, which the draft does not load. The caller releases the copy once the engine has loaded.
    """
    source_dir = (
        draft_model_path
        if os.path.isdir(draft_model_path)
        else snapshot_download(draft_model_path, revision=revision)
    )
    with open(os.path.join(source_dir, "config.json")) as config_file:
        draft_config = json.load(config_file)
    if draft_config.get("model_type") != "whisper":
        raise ValueError(f"{draft_model_path} is not a Whisper checkpoint")
    else:
        pass
    draft_config["architectures"] = [
        DRAFT_ARCHITECTURE,
        "WhisperForConditionalGeneration",
    ]
    draft_config["encoder_layers"] = 0
    draft_checkpoint = TemporaryDirectory(prefix="whisper-draft-")
    with open(os.path.join(draft_checkpoint.name, "config.json"), "w") as config_file:
        json.dump(draft_config, config_file, indent=2)
    for file_name in os.listdir(source_dir):
        if file_name.endswith(".safetensors") or file_name.startswith("tokenizer"):
            os.symlink(
                os.path.realpath(os.path.join(source_dir, file_name)),
                os.path.join(draft_checkpoint.name, file_name),
            )
        else:
            pass
    return draft_checkpoint


class AudioEncoder(Protocol):
    def __call__(self, audio_features: torch.Tensor) -> torch.Tensor: ...


class WhisperDraftForCausalLM(WhisperForConditionalGeneration):
    """A Whisper decoder whose encoder states come from the target's encoder."""

    def __init__(
        self,
        config: WhisperConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        del prefix
        nn.Module.__init__(self)
        self.config = config
        self.model = WhisperDecoderModel(config, quant_config=quant_config)
        self.proj_out = self.model.decoder.embed_tokens
        self.lm_head = self.proj_out
        self.logits_processor = LogitsProcessor(config)
        self.start_layer = 0
        self.end_layer = int(config.decoder_layers) * 2
        self.encoder_graph_runner = None
        # note (wilsonzheng0327): the worker points this at the target's encoder.
        self.target_encoder: AudioEncoder | None = None

    def run_encoder(self, audio_features: torch.Tensor) -> torch.Tensor:
        if self.target_encoder is None:
            raise RuntimeError("the Whisper draft has no encoder of its own")
        else:
            return self.target_encoder(audio_features)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        super().load_weights(
            (name, weight)
            for name, weight in weights
            if not name.startswith("model.encoder.")
        )


class WhisperVerifyInput(DFlashVerifyInput):
    """A causal chain of tokens per request, after the request's committed decoder tokens.

    Cross-attention reads the encoder slots alone, so its lengths are not widened by the chain.
    """

    def generate_attn_arg_prefill(
        self,
        *,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        translator: KVIndexTranslator,
        plan: KVLocPlan,
        kv_start_idx: torch.Tensor | None = None,
        kv_indices_buf: torch.Tensor | None = None,
        kind: IdSpaceKind = IdSpaceKind.FULL,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        assert kv_start_idx is not None, "a Whisper verify row starts at a KV column"
        device = req_pool_indices.device
        batch_size = req_pool_indices.numel()
        query_indptr = torch.arange(
            0,
            (batch_size + 1) * self.draft_token_num,
            step=self.draft_token_num,
            dtype=torch.int32,
            device=device,
        )
        # note (wilsonzheng0327): self-attention rows start past the encoder slots and
        # cross-attention rows at 0, CUDA graph padding rows included.
        kv_lengths = torch.where(
            kv_start_idx > 0,
            paged_kernel_lens + self.draft_token_num,
            paged_kernel_lens,
        )
        kv_indptr = torch.zeros((batch_size + 1,), dtype=torch.int32, device=device)
        kv_indptr[1:] = torch.cumsum(kv_lengths, dim=0)
        kv_indices = (
            kv_indices_buf
            if kv_indices_buf is not None
            else torch.empty(
                paged_kernel_lens_sum + batch_size * self.draft_token_num,
                dtype=torch.int32,
                device=device,
            )
        )
        translator.pack_read_stream(
            plan,
            req_pool_indices=req_pool_indices,
            seq_lens=kv_lengths,
            indptr=kv_indptr,
            out=kv_indices,
            kv_start_idx=kv_start_idx,
            kind=kind,
        )
        return kv_indices, kv_indptr, query_indptr, None


def check_shared_encoder(
    target_model: WhisperForConditionalGeneration, draft_checkpoint_dir: str
) -> None:
    """Refuse a draft whose encoder weights differ from the target's, since it reads the target's encoder states."""
    target_parameters = dict(target_model.named_parameters())
    unchecked_names = set(SHARED_ENCODER_TENSOR_NAMES)
    for path in sorted(glob.glob(os.path.join(draft_checkpoint_dir, "*.safetensors"))):
        with safe_open(path, framework="pt") as checkpoint:
            for name in unchecked_names & set(checkpoint.keys()):
                draft_weight = checkpoint.get_tensor(name).float()
                target_weight = target_parameters[name].detach().float().cpu()
                # note (wilsonzheng0327): the two checkpoints may hold the same
                # weights in different half precisions.
                if draft_weight.shape != target_weight.shape or not torch.allclose(
                    draft_weight, target_weight, rtol=1e-2, atol=1e-5
                ):
                    raise ValueError(
                        f"the Whisper draft's {name} differs from the target's; "
                        "the draft must share the target's encoder"
                    )
                else:
                    unchecked_names.discard(name)
    if unchecked_names:
        raise ValueError(
            f"the Whisper draft checkpoint lacks {sorted(unchecked_names)}, "
            "so its encoder cannot be checked against the target's"
        )
    else:
        pass


def install_verify_graph_input(
    target_runner: ModelRunner, draft_token_num: int
) -> None:
    """Capture the target's verify graphs with the verify input its replays carry.

    SGLang captures a STANDALONE target with EAGLE's tree-mask verify input, and this worker verifies an unmasked causal chain.
    """
    upstream_get_spec_info = DecodeCudaGraphRunner.get_spec_info

    def get_spec_info(runner: DecodeCudaGraphRunner, num_tokens: int) -> SpecInput:
        if runner.model_runner is target_runner:
            return WhisperVerifyInput(
                draft_token=None,
                positions=None,
                draft_token_num=draft_token_num,
                custom_mask=None,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )
        else:
            return upstream_get_spec_info(runner, num_tokens)

    DecodeCudaGraphRunner.get_spec_info = get_spec_info


class WhisperDraftWorker(EagleDraftWorkerBase):
    """Holds the draft model runner for the hooks SGLang calls on a draft worker."""

    def __init__(self, draft_worker: TpModelWorker) -> None:
        super().__init__()
        self.draft_worker = draft_worker
        self.draft_runner = draft_worker.model_runner

    def draft(self) -> None:
        raise NotImplementedError("WhisperSpeculativeWorker drafts in its own step")

    def draft_extend(self) -> None:
        raise NotImplementedError("WhisperSpeculativeWorker drafts in its own step")

    def init_attention_backends(self) -> None:
        self.draft_worker.init_attention_backends()

    def init_cuda_graphs(self) -> None:
        # note (wilsonzheng0327): a compiled draft decode copies the layer's whole K/V
        # buffer on every KV store.
        flags = get_flags().capture
        enable_torch_compile = flags.enable_torch_compile
        flags.enable_torch_compile = False
        self.draft_worker.init_cuda_graphs(capture_decode_cuda_graph=True)
        flags.enable_torch_compile = enable_torch_compile


class WhisperSpeculativeWorker(BaseSpecWorker):
    """Greedy chain speculation: the draft proposes speculative_num_steps tokens and the target checks them in one forward.

    The draft's first step reads two tokens, the last committed one and the newest, so a fully accepted chain's last token gets its draft KV before it is attended.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        target_worker: SpeculativeTargetWorker,
    ) -> None:
        super().__init__()
        self.device = get_device().device
        self._target_worker = target_worker  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        self.speculative_num_steps = int(get_spec().speculative_num_steps)
        self.speculative_num_draft_tokens = int(get_spec().speculative_num_draft_tokens)
        if (
            int(get_spec().speculative_eagle_topk or 1) != 1
            or self.speculative_num_draft_tokens != self.speculative_num_steps + 1
        ):
            raise ValueError(
                "Whisper speculative decoding drafts a chain: set "
                "speculative_eagle_topk=1 and speculative_num_draft_tokens="
                "speculative_num_steps + 1"
            )
        else:
            pass
        with draft_pp_context(), draft_model_build_scope():
            draft_tp_worker = TpModelWorker(
                server_args=server_args,
                gpu_id=gpu_id,
                nccl_port=nccl_port,
                is_draft_worker=True,
                context_length=target_worker.model_runner.model_config.context_len,
            )
        self._draft_worker = WhisperDraftWorker(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            draft_tp_worker
        )
        self.draft_runner: ModelRunner = draft_tp_worker.model_runner
        # note (wilsonzheng0327): the draft decodes plainly, so it captures plain
        # decode graphs rather than EAGLE draft graphs.
        self.draft_runner.spec_algorithm = SpeculativeAlgorithm.NONE
        install_verify_graph_input(
            target_worker.model_runner, self.speculative_num_draft_tokens
        )
        target_model = target_worker.model_runner.model
        draft_model = self.draft_runner.model
        if draft_model.config.vocab_size != target_model.config.vocab_size:
            raise ValueError("the Whisper draft's vocabulary differs from the target's")
        else:
            pass
        check_shared_encoder(target_model, get_spec().speculative_draft_model_path)
        draft_model.target_encoder = target_model.run_encoder
        self.previous_token_ids: torch.Tensor | None = None

    def alloc_memory_pool(
        self,
        *,
        memory_pool_config: MemoryPoolConfig,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    ) -> None:
        self.draft_worker.draft_worker.alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.previous_token_ids = torch.zeros(
            req_to_token_pool.req_to_token.shape[0],
            dtype=torch.int64,
            device=self.device,
        )

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        if batch.forward_mode.is_extend():
            return self.forward_prefill(batch)
        else:
            return self.forward_decode(batch)

    def forward_prefill(self, batch: ScheduleBatch) -> GenerationBatchResult:
        draft_batch = ForwardBatch.init_new(
            batch,
            self.draft_runner,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            return_hidden_states_before_norm=False,
        )
        self.draft_runner.forward(draft_batch)
        result = self.target_worker.forward_batch_generation(
            batch, capture_hidden_mode=CaptureHiddenMode.NULL
        )
        # note (wilsonzheng0327): the sampled token has no KV yet, so the committed
        # length is the prompt's.
        result.new_seq_lens = batch.seq_lens
        last_prompt_rows = torch.cumsum(draft_batch.extend_seq_lens, dim=0) - 1
        self.previous_token_ids[batch.req_pool_indices] = draft_batch.input_ids[
            last_prompt_rows
        ].long()
        result.next_draft_input = self.draft_input(result.next_token_ids)
        return result

    def forward_decode(self, batch: ScheduleBatch) -> GenerationBatchResult:
        assert (
            batch.sampling_info.is_all_greedy
        ), "the request builder refuses sampled requests under speculation"
        batch_size = len(batch.reqs)
        draft_steps = self.speculative_num_steps
        verify_width = draft_steps + 1
        newest_token_ids = batch.spec_info.bonus_tokens.long()
        previous_token_ids = self.previous_token_ids[batch.req_pool_indices]
        # note (wilsonzheng0327): columns of the previous token, the newest token
        # and the draft_steps drafts.
        newest_columns = (batch.encoder_lens + batch.seq_lens).long()
        column_offsets = torch.arange(-1, verify_width, device=self.device)
        slots = (
            self.req_to_token_pool.req_to_token[batch.req_pool_indices]
            .gather(1, newest_columns[:, None] + column_offsets[None, :])
            .long()
        )

        logits = self.chain_forward(
            batch,
            self.draft_runner,
            token_ids=torch.stack([previous_token_ids, newest_token_ids], dim=1),
            offset=-1,
            slots=slots[:, :2],
        )
        drafts = [self.greedy(logits.view(batch_size, 2, -1)[:, 1], batch)]
        for step in range(1, draft_steps):
            logits = self.decode_forward(
                batch,
                self.draft_runner,
                token_ids=drafts[-1],
                offset=step + 1,
                slots=slots[:, step + 1],
            )
            drafts.append(self.greedy(logits, batch))
        draft_token_ids = torch.stack(drafts, dim=1)

        verify_token_ids = torch.cat(
            [newest_token_ids[:, None], draft_token_ids], dim=1
        )
        target_output = self.target_worker.forward_batch_generation(
            None,
            forward_batch=self.chain_batch(
                batch,
                self.target_worker.model_runner,
                token_ids=verify_token_ids,
                offset=0,
                slots=slots[:, 1:],
            ),
            is_verify=True,
        )
        logits_output = target_output.logits_output
        target_token_ids = self.greedy(
            logits_output.next_token_logits, batch, rows_per_request=verify_width
        ).view(batch_size, verify_width)
        num_correct_drafts = (
            (target_token_ids[:, :draft_steps] == draft_token_ids)
            .int()
            .cumprod(dim=1)
            .sum(dim=1)
        )
        accept_lens = (num_correct_drafts + 1).to(torch.int32)
        self.previous_token_ids[batch.req_pool_indices] = verify_token_ids.gather(
            1, num_correct_drafts[:, None].long()
        ).squeeze(1)
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=target_token_ids.reshape(-1).to(torch.int32),
            can_run_cuda_graph=target_output.can_run_cuda_graph,
            speculative_num_draft_tokens=verify_width,
            next_draft_input=self.draft_input(
                target_token_ids.gather(1, num_correct_drafts[:, None].long()).squeeze(
                    1
                )
            ),
            accept_lens=accept_lens,
            new_seq_lens=batch.seq_lens + accept_lens,
        )

    def draft_input(self, bonus_token_ids: torch.Tensor) -> EagleDraftInput:
        batch_size = bonus_token_ids.shape[0]
        return EagleDraftInput(
            bonus_tokens=bonus_token_ids.to(torch.int32),
            topk_p=torch.ones((batch_size, 1), dtype=torch.float32, device=self.device),
            topk_index=bonus_token_ids.view(batch_size, 1).long(),
            hidden_states=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

    def greedy(
        self, logits: torch.Tensor, batch: ScheduleBatch, rows_per_request: int = 1
    ) -> torch.Tensor:
        logit_bias = batch.sampling_info.logit_bias
        if logit_bias is not None:
            logits = logits + logit_bias.repeat_interleave(rows_per_request, dim=0)
        else:
            pass
        return logits.argmax(dim=-1)

    def step_batch(
        self,
        batch: ScheduleBatch,
        *,
        forward_mode: ForwardMode,
        token_ids: torch.Tensor,
        offset: int,
        slots: torch.Tensor,
        spec_info: WhisperVerifyInput | None = None,
    ) -> ScheduleBatch:
        """The batch at seq_lens + offset, with its CPU lengths moved on the host."""
        step = copy.copy(batch)
        step.forward_mode = forward_mode
        step.input_ids = token_ids.reshape(-1)
        step.out_cache_loc = slots.reshape(-1)
        step.seq_lens = batch.seq_lens + offset
        step.seq_lens_cpu = batch.seq_lens_cpu + offset
        step.seq_lens_sum = int(step.seq_lens_cpu.sum())
        step.spec_info = spec_info
        step.encoder_cached = [True] * len(batch.reqs)
        step.return_logprob = False
        return step

    def chain_batch(
        self,
        batch: ScheduleBatch,
        runner: ModelRunner,
        *,
        token_ids: torch.Tensor,
        offset: int,
        slots: torch.Tensor,
    ) -> ForwardBatch:
        tokens_per_request = token_ids.shape[1]
        spec_info = WhisperVerifyInput(
            draft_token=token_ids.reshape(-1),
            positions=(
                (batch.seq_lens + offset)[:, None]
                + torch.arange(tokens_per_request, device=self.device)[None, :]
            ).reshape(-1),
            draft_token_num=tokens_per_request,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        step = self.step_batch(
            batch,
            forward_mode=ForwardMode.TARGET_VERIFY,
            token_ids=token_ids,
            offset=offset,
            slots=slots,
            spec_info=spec_info,
        )
        return ForwardBatch.init_new(
            step,
            runner,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            return_hidden_states_before_norm=False,
        )

    def chain_forward(
        self,
        batch: ScheduleBatch,
        runner: ModelRunner,
        *,
        token_ids: torch.Tensor,
        offset: int,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        forward_batch = self.chain_batch(
            batch, runner, token_ids=token_ids, offset=offset, slots=slots
        )
        return runner.forward(forward_batch).logits_output.next_token_logits

    def decode_forward(
        self,
        batch: ScheduleBatch,
        runner: ModelRunner,
        *,
        token_ids: torch.Tensor,
        offset: int,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        step = self.step_batch(
            batch,
            forward_mode=ForwardMode.DECODE,
            token_ids=token_ids,
            offset=offset,
            slots=slots,
        )
        forward_batch = ForwardBatch.init_new(
            step,
            runner,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            return_hidden_states_before_norm=False,
        )
        return runner.forward(forward_batch).logits_output.next_token_logits
