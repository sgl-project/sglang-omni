# SPDX-License-Identifier: Apache-2.0
"""DSpark speculative decoding for the MiniCPM-o talker, with verification that keeps its sampling exact.

The draft is a DSpark checkpoint over the talker's codec ids that reads the talker's
residual stream after a few of its layers. The talker samples with its own shaping
(codec suppress mask, a repetition penalty over the last 16 outputs, and the stop id
masked until the minimum length) before temperature, top-k and top-p, so the worker
draws every proposal from the draft's logits shaped the same way and rejection-samples
it against the talker's distribution for each verify row (talker_dspark_sampling.py).
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterable

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
    is_unified_kv_triton,
)
from sglang.kernels.ops.speculative.dspark.dspark_accept import (
    FinalizeAcceptLens,
    finalize_accept_lens,
)
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    BuildOutTokens,
    build_out_tokens,
)
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.models.dspark import Qwen3DSparkModel
from sglang.srt.runtime_context import get_exec, get_parallel, get_spec
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockProposer,
    DraftBlockResult,
    DraftProposal,
    select_draft_hidden_without_anchor,
)
from sglang.srt.speculative.dspark_components.dspark_planner import VerifyWindow
from sglang.srt.speculative.dspark_components.dspark_verify import (
    AcceptOuts,
    TargetVerifyExecutor,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyMode,
    read_ragged_verify_mode,
)
from sglang.srt.utils import DynamicGradMode
from torch import nn
from transformers import Qwen3Config

from sglang_omni.model_runner.speculative_target_worker import SpeculativeTargetWorker
from sglang_omni.models.minicpm_o.components.sglang_talker import (
    MiniCPMOTalkerForCausalLM,
)
from sglang_omni.models.minicpm_o.talker_dspark_sampling import (
    TalkerSamplingState,
    exact_accept,
    sample_draft_block,
)
from sglang_omni.models.minicpm_o.talker_model_runner import (
    REP_PENALTY_WINDOW,
    TalkerSlotState,
)
from sglang_omni.models.minicpm_o.talker_slot_kernels import (
    commit_slot_rows,
    gather_slot_rows,
    has_slot_kernels,
)
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import DeviceGraphBackend, ReplayableGraph

DRAFT_ARCHITECTURE = "MiniCPMOTalkerDSparkDraftModel"
# Fields SGLang reads from the top level of a DSpark config; SpecForge exports nest them.
LIFTED_DSPARK_FIELDS = (
    "markov_rank",
    "markov_head_type",
    "mask_token_id",
    "target_layer_ids",
)


def talker_vocabulary_weights(model_path: str) -> dict[str, torch.Tensor]:
    """The talker's codec embedding and weight-normed codec head from its checkpoint."""
    with open(os.path.join(model_path, "model.safetensors.index.json")) as index_file:
        weight_map = json.load(index_file)["weight_map"]

    def tensor(name: str) -> torch.Tensor:
        with safe_open(os.path.join(model_path, weight_map[name]), "pt") as shard:
            return shard.get_tensor(name)

    gain = tensor("tts.head_code.0.parametrizations.weight.original0").float()
    direction = tensor("tts.head_code.0.parametrizations.weight.original1").float()
    return {
        "embed_tokens.weight": tensor("tts.emb_code.0.weight").contiguous(),
        "lm_head.weight": (gain * direction / direction.norm(dim=1, keepdim=True))
        .to(torch.bfloat16)
        .contiguous(),
    }


def materialize_draft_checkpoint(
    draft_model_path: str, model_path: str, revision: str | None
) -> tuple[str, int]:
    """Write a loadable copy of a SpecForge DSpark draft for the talker.

    The copy names the talker draft class, lifts the DSpark fields SGLang reads to the
    top level, turns off the confidence head (every verify checks a whole block), links
    the draft weights and adds the talker's codec embedding and head, which the draft
    owns. Returns the copy's path and the draft's block size.
    """
    source_dir = (
        draft_model_path
        if os.path.isdir(draft_model_path)
        else snapshot_download(draft_model_path, revision=revision)
    )
    with open(os.path.join(source_dir, "config.json")) as config_file:
        config = json.load(config_file)
    nested = config.get("dflash_config") or {}
    for field_name in LIFTED_DSPARK_FIELDS:
        if field_name in nested and field_name not in config:
            config[field_name] = nested[field_name]
        else:
            pass
    if int(config.get("markov_rank", 0)) <= 0:
        raise ValueError(f"{draft_model_path} is not a DSpark draft (no Markov head)")
    elif config.get("markov_head_type", "vanilla") != "vanilla":
        raise ValueError("the MiniCPM-o talker draft supports the vanilla Markov head")
    else:
        pass
    vocabulary = talker_vocabulary_weights(model_path)
    vocab_size = int(vocabulary["lm_head.weight"].shape[0])
    if int(config["vocab_size"]) != vocab_size:
        raise ValueError(
            f"{draft_model_path} has {config['vocab_size']} ids, the talker {vocab_size}"
        )
    else:
        pass
    config.update(
        architectures=[DRAFT_ARCHITECTURE],
        enable_confidence_head=False,
        tie_word_embeddings=False,
    )
    nested["enable_confidence_head"] = False
    config["dflash_config"] = nested
    draft_dir = tempfile.mkdtemp(prefix="minicpm-o-talker-dspark-")
    with open(os.path.join(draft_dir, "config.json"), "w") as config_file:
        json.dump(config, config_file, indent=2)
    for file_name in os.listdir(source_dir):
        if file_name.startswith("model") and file_name.endswith(".safetensors"):
            os.symlink(
                os.path.realpath(os.path.join(source_dir, file_name)),
                os.path.join(draft_dir, file_name),
            )
        else:
            pass
    save_file(vocabulary, os.path.join(draft_dir, "talker_vocabulary.safetensors"))
    return draft_dir, int(config["block_size"])


BlockDraw = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, TalkerSamplingState],
    tuple[torch.Tensor, torch.Tensor],
]
BlockAccept = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, TalkerSamplingState],
    tuple[torch.Tensor, torch.Tensor],
]


def draw_block(
    base_logits: torch.Tensor,
    previous_tokens: torch.Tensor,
    markov_embedding: torch.Tensor,
    markov_projection: torch.Tensor,
    state: TalkerSamplingState,
) -> tuple[torch.Tensor, torch.Tensor]:
    """sample_draft_block with its Exp(1) noise drawn here."""
    return sample_draft_block(
        base_logits,
        previous_tokens,
        markov_embedding,
        markov_projection,
        state,
        torch.empty_like(base_logits, dtype=torch.float32).exponential_(),
    )


def accept_block(
    target_logits: torch.Tensor,
    shaped_draft_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    state: TalkerSamplingState,
) -> tuple[torch.Tensor, torch.Tensor]:
    """exact_accept with its uniform and Exp(1) noise drawn here."""
    return exact_accept(
        target_logits,
        shaped_draft_logits,
        draft_tokens,
        state,
        torch.rand(draft_tokens.shape, device=draft_tokens.device),
        torch.empty(
            (draft_tokens.shape[0], target_logits.shape[1]), device=draft_tokens.device
        ).exponential_(),
    )


class StaticSamplingRows:
    """One step's sampling state at fixed addresses, read by the folded draft draw and
    the accept graphs.

    load copies a batch's rows in front; rows past it keep earlier values, which stay
    valid for the padded rows of a graph.
    """

    def __init__(
        self, *, capacity: int, vocab: int, eos_id: int, device: torch.device
    ) -> None:
        self.capacity = capacity
        self.state = TalkerSamplingState(
            recent_tokens=torch.full(
                (capacity, REP_PENALTY_WINDOW), vocab, dtype=torch.long, device=device
            ),
            generated=torch.zeros(capacity, dtype=torch.long, device=device),
            min_new_tokens=torch.zeros(capacity, dtype=torch.long, device=device),
            penalties=torch.ones(capacity, device=device),
            suppress=torch.zeros((capacity, vocab), dtype=torch.bool, device=device),
            temperatures=torch.ones(capacity, device=device),
            top_ks=torch.full((capacity,), vocab, dtype=torch.long, device=device),
            top_ps=torch.ones(capacity, device=device),
            greedy=torch.zeros(capacity, dtype=torch.bool, device=device),
            eos_id=eos_id,
        )
        self.loaded_count = 0

    def rows(self, size: int) -> TalkerSamplingState:
        state = self.state
        return TalkerSamplingState(
            recent_tokens=state.recent_tokens[:size],
            generated=state.generated[:size],
            min_new_tokens=state.min_new_tokens[:size],
            penalties=state.penalties[:size],
            suppress=state.suppress[:size],
            temperatures=state.temperatures[:size],
            top_ks=state.top_ks[:size],
            top_ps=state.top_ps[:size],
            greedy=state.greedy[:size],
            eos_id=state.eos_id,
        )

    def load(self, state: TalkerSamplingState) -> None:
        count = int(state.generated.shape[0])
        if count > self.capacity:
            self.loaded_count = 0
            return
        else:
            pass
        for name in (
            "recent_tokens",
            "generated",
            "min_new_tokens",
            "penalties",
            "suppress",
            "temperatures",
            "top_ks",
            "top_ps",
            "greedy",
        ):
            getattr(self.state, name)[:count].copy_(getattr(state, name))
        self.loaded_count = count

    def gather(
        self, slots: TalkerSlotState, rows: torch.Tensor
    ) -> TalkerSamplingState | None:
        """Load the rows' state straight from the runner's slots in one launch; None past capacity."""
        count = int(rows.shape[0])
        if count > self.capacity:
            self.loaded_count = 0
            return None
        else:
            gather_slot_rows(slots, rows, self.state)
            self.loaded_count = count
            return self.rows(count)


class FoldedTalkerDraw:
    """The talker-shaped draft draw as the tail hook of the draft model's graph.

    It stands in for SGLang's folded draft sampler, whose draw skips the talker's
    shaping. It draws each block from the step's state in rows loaded before the
    replay and keeps the shaped logits in float32, because the exact verify divides by
    them. The noise stays eager so each replay draws from the graph's random state;
    TalkerDraftProposer reads the block from out and corrected_out.
    """

    def __init__(
        self,
        *,
        model: MiniCPMOTalkerDSparkDraftModel,
        gamma: int,
        max_bs: int,
        device: torch.device,
        rows: StaticSamplingRows,
        draw_kernel: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        self.model = model
        self.gamma = gamma
        self.sample_from_anchor = bool(model.sample_from_anchor)
        self.query_token_num = gamma if self.sample_from_anchor else gamma + 1
        vocab = int(model.lm_head.weight.shape[0])
        self.out = torch.empty((max_bs * gamma,), dtype=torch.long, device=device)
        self.corrected_out = torch.empty(
            (max_bs * gamma, vocab), dtype=torch.float32, device=device
        )
        self.sampling_rows = rows
        self.draw_kernel = draw_kernel

    def stage_sampling_params(
        self, *, bs: int, sampling_info: SamplingBatchInfo
    ) -> None:
        """The worker loads the sampling rows before the forward."""
        return None

    def __call__(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> None:
        bs = hidden_states.shape[0] // self.query_token_num
        if self.sample_from_anchor:
            model_hidden = hidden_states
        else:
            model_hidden, _ = select_draft_hidden_without_anchor(
                hidden_states, bs=bs, gamma=self.gamma
            )
        base_logits, _ = self.model.compute_base_logits(model_hidden)
        base_logits = base_logits.view(bs, self.gamma, -1)
        markov_head = self.model.markov_head
        tokens, shaped = self.draw_kernel(
            base_logits,
            input_ids.view(bs, self.query_token_num)[:, 0].long(),
            markov_head.markov_w1.weight,
            markov_head.markov_w2.weight,
            self.sampling_rows.rows(bs),
            torch.empty_like(base_logits, dtype=torch.float32).exponential_(),
        )
        self.out[: bs * self.gamma].copy_(tokens.reshape(-1))
        self.corrected_out[: bs * self.gamma].copy_(shaped.reshape(bs * self.gamma, -1))


class ExactAcceptGraphs:
    """The exact verify and its commit lengths replayed as a device graph per padded batch size.

    Eager, the verify is a compiled-function dispatch plus dozens of small kernels per
    step, several times the Talker's GPU work for a block, and the commit lengths and
    output tokens are two more kernel launches; a replay launches them at once. The
    graphs read the draft tokens and shaped logits where the folded draw leaves them;
    batches above the largest size, or whose rows were not loaded, take the fallback.
    """

    def __init__(
        self,
        *,
        batch_sizes: list[int],
        gamma: int,
        rows: StaticSamplingRows,
        draft_tokens: torch.Tensor,
        shaped_draft_logits: torch.Tensor,
        target_logits_dtype: torch.dtype,
        fallback: BlockAccept,
        backend: DeviceGraphBackend,
        device: torch.device,
    ) -> None:
        self.batch_sizes = sorted(
            size for size in set(batch_sizes) if size <= rows.capacity
        )
        self.gamma = gamma
        self.rows_per_request = gamma + 1
        self.sampling_rows = rows
        self.fallback = fallback
        # note (0xtoward): compiled, the shaping, sort and scatters fuse into a few kernels;
        # the noise stays eager so each replay draws from the graph's random state.
        self.kernel = torch.compile(
            exact_accept, dynamic=True, options={"emulate_precision_casts": True}
        )
        largest = self.batch_sizes[-1]
        vocab = int(shaped_draft_logits.shape[-1])
        self.draft_tokens = draft_tokens
        self.shaped_draft_logits = shaped_draft_logits
        self.target_logits = torch.zeros(
            (largest * self.rows_per_request, vocab),
            dtype=target_logits_dtype,
            device=device,
        )
        self.accepted_count = torch.zeros(largest, dtype=torch.long, device=device)
        self.next_tokens = torch.zeros(largest, dtype=torch.long, device=device)
        self.prefix_lens = torch.zeros(largest, dtype=torch.long, device=device)
        self.no_trim = torch.zeros(largest, dtype=torch.int32, device=device)
        self.commit_lens = torch.zeros(largest, dtype=torch.int32, device=device)
        self.new_seq_lens = torch.zeros(largest, dtype=torch.long, device=device)
        self.out_tokens = torch.zeros(
            (largest, self.rows_per_request), dtype=torch.long, device=device
        )
        self.graphs: dict[int, ReplayableGraph] = {}
        device_module = torch.get_device_module(device)
        pool = backend.graph_pool_handle()
        stream = device_module.Stream(device=device)
        # note (0xtoward): largest first on one stream, so smaller graphs reuse its pool.
        for size in reversed(self.batch_sizes):
            stream.wait_stream(device_module.current_stream(device))
            with device_module.stream(stream):
                for _ in range(2):
                    self.run(size)
            # note (0xtoward): code2wav runs on other threads of this process, so a
            # capture failure there must not abort this one.
            with backend.capture(
                pool=pool, stream=stream, thread_local_errors=True
            ) as graph:
                self.run(size)
            self.graphs[size] = graph
            device_module.current_stream(device).wait_stream(stream)

    def run(self, size: int) -> None:
        draft_tokens = self.draft_tokens[: size * self.gamma].view(size, self.gamma)
        target_logits = self.target_logits[: size * self.rows_per_request]
        accepted_count, next_tokens = self.kernel(
            target_logits,
            self.shaped_draft_logits[: size * self.gamma].view(size, self.gamma, -1),
            draft_tokens,
            self.sampling_rows.rows(size),
            torch.rand(draft_tokens.shape, device=draft_tokens.device),
            torch.empty(
                (size, target_logits.shape[1]), device=draft_tokens.device
            ).exponential_(),
        )
        self.accepted_count[:size].copy_(accepted_count)
        self.next_tokens[:size].copy_(next_tokens)
        finalized = finalize_accept_lens(
            correct_len=accepted_count,
            cap_trim_lens=self.no_trim[:size],
            prefix_lens=self.prefix_lens[:size],
        )
        self.commit_lens[:size].copy_(finalized.commit_lens)
        self.new_seq_lens[:size].copy_(finalized.new_seq_lens)
        self.out_tokens[:size].copy_(
            build_out_tokens(
                draft_tokens=draft_tokens,
                correct_len=accepted_count,
                bonus=next_tokens,
                verify_num_draft_tokens=self.rows_per_request,
                gamma=self.gamma,
            )
        )

    def padded_size(self, count: int) -> int | None:
        if count != self.sampling_rows.loaded_count:
            return None
        else:
            pass
        return next((size for size in self.batch_sizes if size >= count), None)

    def replay(
        self,
        size: int,
        target_logits: torch.Tensor,
        shaped_draft_logits: torch.Tensor,
        draft_tokens: torch.Tensor,
    ) -> None:
        count = draft_tokens.shape[0]
        rows = count * self.gamma
        # An eager draft draw leaves its block elsewhere; the folded draw, in place.
        if shaped_draft_logits.data_ptr() != self.shaped_draft_logits.data_ptr():
            self.shaped_draft_logits[:rows].copy_(shaped_draft_logits.reshape(rows, -1))
            self.draft_tokens[:rows].copy_(draft_tokens.reshape(-1))
        else:
            pass
        self.target_logits[: count * self.rows_per_request].copy_(target_logits)
        self.graphs[size].replay()

    def accept_and_finalize(
        self,
        target_logits: torch.Tensor,
        shaped_draft_logits: torch.Tensor,
        draft_tokens: torch.Tensor,
        prefix_lens: torch.Tensor,
    ) -> AcceptOuts | None:
        """One replay for the verify, the commit lengths and the output tokens; None when no graph fits."""
        count = draft_tokens.shape[0]
        size = self.padded_size(count)
        if size is None:
            return None
        else:
            pass
        self.prefix_lens[:count].copy_(prefix_lens)
        self.replay(size, target_logits, shaped_draft_logits, draft_tokens)
        return AcceptOuts(
            correct_len=self.accepted_count[:count].clone(),
            bonus=self.next_tokens[:count].clone(),
            cap_trim_lens=self.no_trim[:count],
            commit_lens=self.commit_lens[:count].clone(),
            new_seq_lens=self.new_seq_lens[:count].clone(),
            out_tokens=self.out_tokens[:count].clone(),
        )

    def accept(
        self,
        target_logits: torch.Tensor,
        shaped_draft_logits: torch.Tensor,
        draft_tokens: torch.Tensor,
        state: TalkerSamplingState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = draft_tokens.shape[0]
        size = self.padded_size(count)
        if size is None:
            return self.fallback(
                target_logits.to(self.target_logits.dtype),
                shaped_draft_logits,
                draft_tokens,
                state,
            )
        else:
            pass
        self.replay(size, target_logits, shaped_draft_logits, draft_tokens)
        return self.accepted_count[:count].clone(), self.next_tokens[:count].clone()


class DraftKvInjectionGraphs:
    """The draft's KV write for the committed verify rows, replayed as a device graph per padded batch size.

    Eager it is the target-hidden projection, a projection to every draft layer's keys
    and values and a fused norm, rope and write kernel, launched from Python each step.
    Padded rows commit nothing, so the write masks them out.
    """

    def __init__(
        self,
        *,
        batch_sizes: list[int],
        rows_per_request: int,
        hidden_width: int,
        hidden_dtype: torch.dtype,
        inject: Callable[..., None],
        backend: DeviceGraphBackend,
        device: torch.device,
    ) -> None:
        self.batch_sizes = sorted(set(batch_sizes))
        self.rows_per_request = rows_per_request
        self.inject = inject
        largest = self.batch_sizes[-1]
        self.hidden = torch.zeros(
            (largest * rows_per_request, hidden_width),
            dtype=hidden_dtype,
            device=device,
        )
        self.cache_loc_2d = torch.zeros(
            (largest, rows_per_request), dtype=torch.long, device=device
        )
        self.positions = torch.zeros(
            largest * rows_per_request, dtype=torch.long, device=device
        )
        # Zero commit lengths make the warm-ups and padded rows write nothing.
        self.commit_lens = torch.zeros(largest, dtype=torch.int32, device=device)
        self.graphs: dict[int, ReplayableGraph] = {}
        device_module = torch.get_device_module(device)
        pool = backend.graph_pool_handle()
        stream = device_module.Stream(device=device)
        # note (0xtoward): largest first on one stream, so smaller graphs reuse its pool.
        for size in reversed(self.batch_sizes):
            stream.wait_stream(device_module.current_stream(device))
            with device_module.stream(stream):
                for _ in range(2):
                    self.run(size)
            with backend.capture(
                pool=pool, stream=stream, thread_local_errors=True
            ) as graph:
                self.run(size)
            self.graphs[size] = graph
            device_module.current_stream(device).wait_stream(stream)

    def run(self, size: int) -> None:
        rows = size * self.rows_per_request
        self.inject(
            target_hidden=self.hidden[:rows],
            cache_loc=self.cache_loc_2d[:size].reshape(-1),
            positions=self.positions[:rows],
            cache_loc_2d=self.cache_loc_2d[:size],
            commit_lens=self.commit_lens[:size],
        )

    def write(
        self,
        *,
        hidden: torch.Tensor,
        cache_loc_2d: torch.Tensor,
        positions: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> bool:
        """Write the committed rows' draft KV; False when no graph fits the batch."""
        count = cache_loc_2d.shape[0]
        size = next((size for size in self.batch_sizes if size >= count), None)
        if size is None:
            return False
        else:
            pass
        rows = count * self.rows_per_request
        self.hidden[:rows].copy_(hidden.reshape(rows, -1))
        self.cache_loc_2d[:count].copy_(cache_loc_2d)
        self.positions[:rows].copy_(positions)
        self.commit_lens[:count].copy_(commit_lens)
        self.commit_lens[count:size].zero_()
        self.graphs[size].replay()
        return True


class TalkerDraftProposer(DraftBlockProposer):
    """Draft proposer that draws every block from the talker's shaping of it.

    A replay of the draft graph leaves the folded draw's block in its buffers; any
    other draft forward draws the block here, because SGLang's own draw samples the
    Markov-corrected logits unshaped. Rejection sampling divides by the proposal
    distribution, so the block's corrected logits are exactly the shaped logits the
    draft drew from.
    """

    sampling_state: TalkerSamplingState | None = None
    block_sampler: BlockDraw = staticmethod(draw_block)

    def propose(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        verify_window: VerifyWindow,
        bs: int,
        device: str,
        target_model: MiniCPMOTalkerForCausalLM,
        sampling_info: SamplingBatchInfo,
    ) -> DraftProposal:
        state = self.sampling_state
        assert state is not None, "a draft step needs its sampling state"
        folded_draw = (
            self._draft_sampler
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        draft_forward = self._run_forward(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            batch=batch,
            draft_input=draft_input,
            verify_window=verify_window,
            bs=bs,
            device=device,
            embed_module=(
                target_model.get_input_embeddings()
                if self.sample_from_anchor
                else self.draft_model.embed_tokens
            ),
            draft_sampler=folded_draw,
            sampling_info=sampling_info,
        )
        draft_block_ids = draft_forward.draft_block_ids
        is_folded = folded_draw is not None and draft_forward.can_run_graph
        if is_folded:
            draft_tokens = folded_draw.out[: bs * self.gamma].view(bs, self.gamma)
            shaped_logits = folded_draw.corrected_out[: bs * self.gamma].view(
                bs, self.gamma, -1
            )
        else:
            base_logits, _ = self.draft_model.compute_base_logits(
                draft_forward.raw_hidden
            )
            markov_head = self.draft_model.markov_head
            draft_tokens, shaped_logits = self.block_sampler(
                base_logits.view(bs, self.gamma, -1),
                draft_block_ids[:, 0],
                markov_head.markov_w1.weight,
                markov_head.markov_w2.weight,
                state,
            )
        return DraftProposal(
            draft_block_ids=(
                draft_block_ids
                if self.sample_from_anchor
                else draft_block_ids[:, : self.gamma].contiguous()
            ),
            draft_block=DraftBlockResult(
                draft_tokens=draft_tokens,
                corrected_logits=shaped_logits,
                greedy_mask=state.greedy,
                temperatures=state.temperatures,
                greedy_draft=False,
            ),
            folded=is_folded,
        )


class MiniCPMOTalkerDSparkDraftModel(Qwen3DSparkModel):
    """DSpark draft over the talker's codec ids, with the talker's embedding and head."""

    uses_own_vocab_modules: bool = True

    def __init__(
        self,
        config: Qwen3Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        vocab_size = int(config.vocab_size)
        hidden_size = int(config.hidden_size)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def compute_base_logits(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return (hidden @ self.lm_head.weight.T).float(), None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        vocabulary: dict[str, torch.Tensor] = {}
        backbone_weights: list[tuple[str, torch.Tensor]] = []
        for name, loaded_weight in weights:
            normalized_name = name.removeprefix("model.")
            if normalized_name in ("embed_tokens.weight", "lm_head.weight"):
                vocabulary[normalized_name] = loaded_weight
            else:
                backbone_weights.append((name, loaded_weight))
        super().load_weights(backbone_weights)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(vocabulary["embed_tokens.weight"])
            self.lm_head.weight.copy_(vocabulary["lm_head.weight"])


def build_sampling_state(
    batch: ScheduleBatch, sampling_info: SamplingBatchInfo, eos_id: int
) -> TalkerSamplingState:
    """Each running request's committed history and the talker's sampling parameters."""
    device = sampling_info.temperatures.device
    vocab = int(sampling_info.vocab_size)
    recent: list[list[int]] = []
    request_values: list[list[float]] = []
    suppress = torch.zeros((len(batch.reqs), vocab), dtype=torch.bool)
    for row, request in enumerate(batch.reqs):
        tail = [int(token_id) for token_id in request.output_ids[-REP_PENALTY_WINDOW:]]
        recent.append([vocab] * (REP_PENALTY_WINDOW - len(tail)) + tail)
        inputs = request.omni_data.talker_model_inputs
        request_values.append(
            [
                float(inputs.get("rep_penalty", 1.0)),
                len(request.output_ids),
                int(inputs.get("min_new_tokens", 0)),
            ]
        )
        suppress_tokens = request.omni_data.suppress_tokens or getattr(
            request,
            "_codec_suppress_tokens",  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            None,
        )
        for token_id in suppress_tokens or ():
            if 0 <= int(token_id) < vocab:
                suppress[row, int(token_id)] = True
            else:
                pass
    # One host-to-device copy carries the three per-request values.
    values = torch.tensor(request_values, dtype=torch.float64).to(device)
    top_ks = sampling_info.top_ks.view(-1)
    top_ps = sampling_info.top_ps.view(-1).float()
    return TalkerSamplingState(
        recent_tokens=torch.tensor(recent, dtype=torch.long).to(device),
        generated=values[:, 1].long(),
        min_new_tokens=values[:, 2].long(),
        penalties=values[:, 0].float(),
        suppress=suppress.to(device),
        temperatures=sampling_info.temperatures.view(-1).float(),
        top_ks=top_ks,
        top_ps=top_ps,
        greedy=top_ks == 1,
        eos_id=eos_id,
    )


def state_from_slots(slots: TalkerSlotState, rows: torch.Tensor) -> TalkerSamplingState:
    """The step's sampling state gathered from the talker runner's per-slot state.

    The runner keeps each request's last outputs in a ring indexed by its output count;
    reading the ring from that count on puts the window in order, oldest first, with
    unwritten entries holding the vocabulary-size padding.
    """
    generated = slots.generated[rows]
    order = (
        generated.unsqueeze(1)
        + torch.arange(REP_PENALTY_WINDOW, device=rows.device).unsqueeze(0)
    ) % REP_PENALTY_WINDOW
    top_ks = slots.top_ks[rows].long()
    return TalkerSamplingState(
        recent_tokens=slots.windows[rows].gather(1, order),
        generated=generated,
        min_new_tokens=slots.min_new_tokens[rows],
        penalties=slots.penalties[rows],
        suppress=slots.suppress[rows],
        temperatures=slots.temperatures[rows].view(-1),
        top_ks=top_ks,
        top_ps=slots.top_ps[rows],
        greedy=top_ks == 1,
        eos_id=slots.eos_id,
    )


def commit_to_slots(
    slots: TalkerSlotState,
    rows: torch.Tensor,
    committed: torch.Tensor,
    counts: torch.Tensor,
) -> None:
    """Append each row's first counts committed tokens, as TalkerSlotState.append would one by one."""
    generated = slots.generated[rows]
    offsets = torch.arange(committed.shape[1], device=rows.device).unsqueeze(0)
    positions = (generated.unsqueeze(1) + offsets) % REP_PENALTY_WINDOW
    windows = slots.windows[rows]
    kept = windows.gather(1, positions)
    windows.scatter_(
        1, positions, torch.where(offsets < counts.unsqueeze(1), committed.long(), kept)
    )
    slots.windows[rows] = windows
    slots.generated[rows] = generated + counts.long()


class ExactVerifyExecutor(TargetVerifyExecutor):
    """Accept drafts against the talker's exact per-position distribution."""

    sampling_state: TalkerSamplingState | None = None
    accept_function: BlockAccept = staticmethod(accept_block)
    accept_graphs: ExactAcceptGraphs | None = None
    kv_graphs: DraftKvInjectionGraphs | None = None
    target_logits_dtype: torch.dtype = torch.float32

    def accept_and_finalize(
        self,
        *,
        folded_accept: bool,
        bs: int,
        verify_ids_2d: torch.Tensor,
        target_logits: torch.Tensor | None,
        draft_block: DraftBlockResult,
        sampling_info: SamplingBatchInfo,
        draft_input: DFlashDraftInputV2,
        layout: None,
        prefix_lens: torch.Tensor,
        draft_tokens: torch.Tensor,
        simulate_bonus_sampling_info: None,
    ) -> AcceptOuts:
        state = self.sampling_state
        assert not folded_accept and layout is None and target_logits is not None
        assert simulate_bonus_sampling_info is None
        assert state is not None, "a verify step needs its sampling state"
        draft_tokens = draft_tokens.long()
        if self.accept_graphs is not None:
            replayed = self.accept_graphs.accept_and_finalize(
                target_logits, draft_block.corrected_logits, draft_tokens, prefix_lens
            )
            if replayed is not None:
                return replayed
            else:
                pass
        else:
            pass
        accepted_count, next_tokens = self.accept_function(
            target_logits.to(self.target_logits_dtype),
            draft_block.corrected_logits,
            draft_tokens,
            state,
        )
        finalized = FinalizeAcceptLens.execute(
            correct_len=accepted_count,
            cap_trim_lens=torch.zeros_like(accepted_count),
            prefix_lens=prefix_lens,
        )
        out_tokens = BuildOutTokens.execute(
            draft_tokens=draft_tokens,
            correct_len=accepted_count,
            bonus=next_tokens,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            gamma=self.gamma,
        )
        return AcceptOuts(
            correct_len=accepted_count,
            bonus=next_tokens,
            cap_trim_lens=finalized.cap_trim_lens,
            commit_lens=finalized.commit_lens,
            new_seq_lens=finalized.new_seq_lens,
            out_tokens=out_tokens,
        )

    def commit_hidden(
        self,
        *,
        batch: ScheduleBatch,
        layout: None,
        hidden_strided: torch.Tensor | None,
        verify_window: VerifyWindow,
        logits_output: LogitsProcessorOutput,
        commit_lens: torch.Tensor,
        bs: int,
        run_compact: bool,
    ) -> None:
        """Write the committed rows' draft KV from a graph replay when one fits."""
        hidden = logits_output.hidden_states
        if (
            self.kv_graphs is not None
            and not run_compact
            and hidden is not None
            and self.kv_graphs.write(
                hidden=hidden,
                cache_loc_2d=self.kv_injector.ids_for(verify_window.kv_loc_plan).view(
                    verify_window.verify_cache_loc_2d.shape
                ),
                positions=verify_window.positions_2d.reshape(-1),
                commit_lens=commit_lens,
            )
        ):
            return
        else:
            pass
        super().commit_hidden(
            batch=batch,
            layout=layout,
            hidden_strided=hidden_strided,
            verify_window=verify_window,
            logits_output=logits_output,
            commit_lens=commit_lens,
            bs=bs,
            run_compact=run_compact,
        )


class MiniCPMOTalkerDSparkWorker(DSparkWorkerV2):
    """DSpark worker that mirrors the talker's sampling in drafting and verification."""

    sampling_rows: StaticSamplingRows | None = None
    folded_draw: FoldedTalkerDraw | None = None
    accept_graphs: ExactAcceptGraphs | None = None

    def __init__(
        self,
        *,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        target_worker: SpeculativeTargetWorker,
    ) -> None:
        # note (0xtoward): the draft draw and the acceptance noise are local, so
        # tensor-parallel ranks would disagree on the committed tokens.
        if get_parallel().tp_size != 1:
            raise ValueError("MiniCPM-o talker speculative decoding runs on one GPU")
        elif (
            read_ragged_verify_mode() != RaggedVerifyMode.STATIC
            or envs.SGLANG_SIMULATE_ACC_LEN.get() > 0
        ):
            raise ValueError(
                "MiniCPM-o talker speculative decoding verifies whole blocks; unset "
                "SGLANG_RAGGED_VERIFY_MODE and SGLANG_SIMULATE_ACC_LEN"
            )
        elif (
            envs.SGLANG_DSPARK_USE_GREEDY_DRAFT.get()
            or get_spec().speculative_use_block_verification
        ):
            raise ValueError(
                "MiniCPM-o talker speculative decoding drafts and verifies with the "
                "talker's own sampling; unset SGLANG_DSPARK_USE_GREEDY_DRAFT and "
                "--speculative-use-block-verification"
            )
        else:
            pass
        super().__init__(
            server_args=server_args,
            gpu_id=gpu_id,
            nccl_port=nccl_port,
            target_worker=target_worker,
        )
        # The stage builder always materializes the draft under this class.
        assert isinstance(self.draft_model, MiniCPMOTalkerDSparkDraftModel)
        self.eos_id = int(self.target_worker.model_runner.model.codec_eos_id)
        self.exact_verify_executor = ExactVerifyExecutor(
            target_worker=self.target_worker,
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
            kv_injector=self._kv_injector,  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            tp_sync=self._tp_sync,  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        )
        self.exact_verify_executor.target_logits_dtype = (
            self.target_worker.model_runner.model.head_code.weight.dtype
        )
        self._verify_executor = (
            self.exact_verify_executor
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        self.talker_proposer = TalkerDraftProposer(
            draft_model=self.draft_model,
            draft_model_runner=self.draft_model_runner,
            gamma=self.gamma,
            mask_token_id=self._mask_token_id,  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            draft_block_spec_info=self._draft_block_spec_info,  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            tp_sync=self._tp_sync,  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            random_seed=self.target_worker.random_seed,
            greedy_draft=False,
        )
        self._proposer = (
            self.talker_proposer
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        # note (0xtoward): batches the graphs do not cover draw and verify compiled; eager,
        # each is about a hundred small kernels per step.
        self.compiled_draw = torch.compile(draw_block, dynamic=True)
        self.compiled_accept = torch.compile(
            accept_block, dynamic=True, options={"emulate_precision_casts": True}
        )
        self.talker_proposer.block_sampler = self.compiled_draw
        self.exact_verify_executor.accept_function = self.compiled_accept
        self.warm_up_compiled_sampling()

    def init_cuda_graphs(self) -> None:
        """Capture the accept graphs too, once the request slots they are sized by exist."""
        super().init_cuda_graphs()
        device = torch.device(self.device)
        backend = current_platform.get_device_graph_backend(device)
        if backend is None or self.folded_draw is None:
            return
        else:
            pass
        slots = self.target_worker.model_runner.req_to_token_pool.req_to_token.shape[0]
        self.accept_graphs = ExactAcceptGraphs(
            batch_sizes=[size for size in (1, 2, 4, 8, 16, 32) if size <= slots] or [1],
            gamma=self.gamma,
            rows=self.sampling_rows,
            draft_tokens=self.folded_draw.out,
            shaped_draft_logits=self.folded_draw.corrected_out,
            target_logits_dtype=self.exact_verify_executor.target_logits_dtype,
            fallback=self.compiled_accept,
            backend=backend,
            device=device,
        )
        self.exact_verify_executor.accept_function = self.accept_graphs.accept
        self.exact_verify_executor.accept_graphs = self.accept_graphs
        # note (0xtoward): the unified SWA KV ring addresses rows by request slot, which
        # the KV graphs leave out.
        if not is_unified_kv_triton():
            self.exact_verify_executor.kv_graphs = DraftKvInjectionGraphs(
                batch_sizes=self.accept_graphs.batch_sizes,
                rows_per_request=self.gamma + 1,
                hidden_width=int(self.draft_model.fc.in_features),
                hidden_dtype=self.draft_model.fc.weight.dtype,
                inject=self._kv_injector.inject_target_hidden,  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
                backend=backend,
                device=device,
            )
        else:
            pass

    # note (0xtoward): compiled graphs guard on grad mode, so the warm-up runs under the
    # scheduler loop's mode.
    @DynamicGradMode()
    def warm_up_compiled_sampling(self) -> None:
        """Compile the draft block and the exact verify before the first request.

        Dynamic shapes still specialize a single running request, so both one and two
        requests are traced here instead of on the serving thread.
        """
        markov_head = self.draft_model.markov_head
        vocab = int(self.draft_model.lm_head.weight.shape[0])
        for batch_size in (1, 2):
            zeros = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            ones = torch.ones(batch_size, dtype=torch.float32, device=self.device)
            state = TalkerSamplingState(
                recent_tokens=torch.full(
                    (batch_size, REP_PENALTY_WINDOW),
                    vocab,
                    dtype=torch.long,
                    device=self.device,
                ),
                generated=zeros,
                min_new_tokens=zeros,
                penalties=ones,
                suppress=torch.zeros(
                    (batch_size, vocab), dtype=torch.bool, device=self.device
                ),
                temperatures=ones,
                top_ks=torch.full(
                    (batch_size,), vocab, dtype=torch.int32, device=self.device
                ),
                top_ps=ones,
                greedy=torch.zeros(batch_size, dtype=torch.bool, device=self.device),
                eos_id=self.eos_id,
            )
            base_logits = torch.zeros(
                (batch_size, self.gamma, vocab), device=self.device
            )
            draft_tokens, shaped_logits = self.talker_proposer.block_sampler(
                base_logits,
                zeros,
                markov_head.markov_w1.weight,
                markov_head.markov_w2.weight,
                state,
            )
            self.exact_verify_executor.accept_function(
                torch.zeros(
                    (batch_size * (self.gamma + 1), vocab),
                    device=self.device,
                    dtype=self.exact_verify_executor.target_logits_dtype,
                ),
                shaped_logits,
                draft_tokens,
                state,
            )

    def _maybe_build_draft_sampler(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        self,
    ) -> FoldedTalkerDraw:
        """Fold the talker-shaped draw into the draft model's graph.

        SGLang's own folded sampler would skip the talker's shaping; its rows hold
        about 0.2 MB per request.
        """
        device = torch.device(self.device)
        max_bs = max(get_exec().graph.cuda_graph_config.decode.bs)
        vocab = int(self.draft_model.lm_head.weight.shape[0])
        self.sampling_rows = StaticSamplingRows(
            capacity=max_bs, vocab=vocab, eos_id=self.eos_id, device=device
        )
        draw_kernel = torch.compile(sample_draft_block, dynamic=True)
        self.folded_draw = FoldedTalkerDraw(
            model=self.draft_model,
            gamma=self.gamma,
            max_bs=max_bs,
            device=device,
            rows=self.sampling_rows,
            draw_kernel=draw_kernel,
        )
        self.warm_up_folded_draw(draw_kernel)
        return self.folded_draw

    # note (0xtoward): compiled graphs guard on grad mode, so the warm-up runs under the
    # scheduler loop's mode.
    @DynamicGradMode()
    def warm_up_folded_draw(
        self, draw_kernel: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        """Compile the folded draw for one and several requests before the draft graphs capture it."""
        markov_head = self.draft_model.markov_head
        for batch_size in (1, 2):
            base_logits = torch.zeros(
                (batch_size, self.gamma, self.sampling_rows.state.suppress.shape[1]),
                device=self.device,
            )
            draw_kernel(
                base_logits,
                torch.zeros(batch_size, dtype=torch.long, device=self.device),
                markov_head.markov_w1.weight,
                markov_head.markov_w2.weight,
                self.sampling_rows.rows(batch_size),
                torch.empty_like(base_logits).exponential_(),
            )

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish: None = None,
        grammar_barrier: None = None,
        *,
        pp_proxy_tensors: PPProxyTensors | None = None,
    ) -> GenerationBatchResult:
        is_decode = not (
            batch.forward_mode.is_extend()
            or batch.is_extend_in_batch
            or batch.forward_mode.is_idle()
        )
        slots = None
        if is_decode:
            sampling_info = batch.sampling_info
            # The talker's request builder sets none of these, and the exact verify has
            # no term for them.
            assert not (
                sampling_info.need_min_p_sampling
                or sampling_info.logit_bias is not None
                or sampling_info.has_custom_logit_processor
                or batch.return_logprob
            )
            slots = self.runner_slot_state()
            rows = batch.req_pool_indices
            state = None
            if (
                slots is not None
                and self.sampling_rows is not None
                and has_slot_kernels(rows.device)
            ):
                state = self.sampling_rows.gather(slots, rows)
            else:
                pass
            if state is not None:
                pass
            else:
                state = (
                    state_from_slots(slots, rows)
                    if slots is not None
                    else build_sampling_state(batch, sampling_info, self.eos_id)
                )
                if self.sampling_rows is not None:
                    self.sampling_rows.load(state)
                else:
                    pass
            self.talker_proposer.sampling_state = state
            self.exact_verify_executor.sampling_state = state
        else:
            pass
        result = super().forward_batch_generation(
            batch,
            on_publish,
            grammar_barrier,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if is_decode and slots is not None:
            commit = (
                commit_slot_rows
                if has_slot_kernels(batch.req_pool_indices.device)
                else commit_to_slots
            )
            commit(
                slots,
                batch.req_pool_indices,
                result.next_token_ids.view(-1, self.gamma + 1),
                result.accept_lens,
            )
        else:
            pass
        return result

    def runner_slot_state(self) -> TalkerSlotState | None:
        """The talker runner's per-slot sampling state, which its prefill sample resets."""
        runner = self.target_worker.omni_runner
        return getattr(runner, "slot_state", None)
