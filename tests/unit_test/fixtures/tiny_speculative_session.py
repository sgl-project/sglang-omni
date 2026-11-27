# SPDX-License-Identifier: Apache-2.0
"""A tiny text session adapter, tiny Qwen3 target and draft checkpoints, and the engine run that drives them."""

import json
from collections.abc import Iterable
from pathlib import Path

import torch
from safetensors.torch import save_file
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from transformers import LlamaConfig, Qwen3Config, Qwen3ForCausalLM

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.profiler.event_recorder_types import (
    reset_active_stage,
    set_active_stage,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import (
    SESSION_METADATA_KEY,
    SessionIdentity,
    SessionOperation,
    TimedChunk,
)
from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.ar_session import (
    ARSessionAdapter,
    ARSessionPreparation,
)
from sglang_omni.scheduling.sglang_backend.history_cut import DraftSessionKV, HistoryCut
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.sglang_backend.server_args_builder import (
    build_sglang_server_args,
)
from sglang_omni.scheduling.types import RequestOutput

SPECULATIVE_SERVER_ARGS: dict[str, dict[str, int]] = {
    "DFLASH": {"speculative_num_draft_tokens": 4},
    "EAGLE3": {
        "speculative_num_steps": 3,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 4,
    },
}


class TinyTextAdapter(ARSessionAdapter):
    def __init__(self, target_kv: DraftSessionKV) -> None:
        self.target_kv = target_kv
        self.history_capacities: list[int] = []
        self.next_cut: HistoryCut | None = None
        self.next_preparation = ARSessionPreparation()

    def build(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> SGLangARRequestData:
        sampling_params = SamplingParams(
            temperature=0,
            top_k=1,
            max_new_tokens=payload.request.params["max_new_tokens"],
            ignore_eos=True,
        )
        sampling_params.normalize(None)
        request = Req(
            payload.request_id,
            "",
            payload.data["token_ids"],
            sampling_params,
            vocab_size=32,
        )
        request.tokenizer = None
        return SGLangARRequestData(req=request, stage_payload=payload)

    def prepare_unit(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> ARSessionPreparation:
        preparation = self.next_preparation
        self.next_preparation = ARSessionPreparation()
        return preparation

    def plan_history_cut(
        self,
        session_identity: SessionIdentity,
        request_data: SGLangARRequestData,
        history_length: int,
        history_capacity: int,
    ) -> HistoryCut | None:
        self.history_capacities.append(history_capacity)
        cut = self.next_cut
        self.next_cut = None
        return cut

    def shift_key_positions(
        self, keys: torch.Tensor, distances: torch.Tensor
    ) -> torch.Tensor:
        return self.target_kv.shift_key_positions(keys, distances)

    def result(
        self, session_identity: SessionIdentity, request_data: SGLangARRequestData
    ) -> StagePayload:
        payload = request_data.stage_payload
        return StagePayload(
            payload.request_id,
            payload.request,
            {"output_ids": list(request_data.output_ids)},
        )

    def stream(
        self,
        session_identity: SessionIdentity,
        request_data: SGLangARRequestData,
        output: RequestOutput,
    ) -> Iterable[TimedChunk]:
        return (
            TimedChunk(
                "text", 0, 0, 0, {"text": str(output.data)}, format="text_delta"
            ),
        )


def create_tiny_checkpoints(directory: Path) -> tuple[Path, dict[str, Path]]:
    """Build deterministic random target, DFlash and EAGLE3 weights on CPU, with no model downloads."""
    torch.manual_seed(23)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    target = Qwen3ForCausalLM(config)
    target.save_pretrained(directory / "target")
    config.num_hidden_layers = 1
    config.layer_types = ["full_attention"]
    draft = Qwen3ForCausalLM(config)
    weights = dict(draft.state_dict())
    weights["fc.weight"] = torch.eye(128)
    weights["hidden_norm.weight"] = torch.ones(128)
    dflash_dir = directory / "dflash"
    dflash_dir.mkdir(parents=True)
    save_file(weights, dflash_dir / "model.safetensors")
    draft_config = config.to_dict()
    draft_config.update(
        architectures=["DFlashDraftModel"],
        dflash_config={
            "target_layer_ids": [0],
            "mask_token_id": 31,
            "block_size": 4,
            "sample_from_anchor": False,
        },
    )
    (dflash_dir / "config.json").write_text(json.dumps(draft_config))
    hidden_size = config.hidden_size
    attention_width = config.num_attention_heads * config.head_dim
    key_value_width = config.num_key_value_heads * config.head_dim
    # fc reads the target's two captured states and the attention reads the token embedding beside the projected state, so both take two hidden rows.
    paired_hidden_size = 2 * hidden_size
    eagle3_weights = {
        name: torch.randn(shape) * config.initializer_range
        for name, shape in (
            ("fc.weight", (hidden_size, paired_hidden_size)),
            ("midlayer.self_attn.q_proj.weight", (attention_width, paired_hidden_size)),
            ("midlayer.self_attn.k_proj.weight", (key_value_width, paired_hidden_size)),
            ("midlayer.self_attn.v_proj.weight", (key_value_width, paired_hidden_size)),
            ("midlayer.self_attn.o_proj.weight", (hidden_size, attention_width)),
            ("midlayer.mlp.gate_proj.weight", (config.intermediate_size, hidden_size)),
            ("midlayer.mlp.up_proj.weight", (config.intermediate_size, hidden_size)),
            ("midlayer.mlp.down_proj.weight", (hidden_size, config.intermediate_size)),
        )
    }
    for name in (
        "midlayer.input_layernorm",
        "midlayer.post_attention_layernorm",
        "midlayer.hidden_norm",
        "norm",
    ):
        eagle3_weights[f"{name}.weight"] = torch.ones(hidden_size)
    eagle3_dir = directory / "eagle3"
    eagle3_dir.mkdir(parents=True)
    save_file(eagle3_weights, eagle3_dir / "model.safetensors")
    eagle3_config = LlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_position_embeddings=config.max_position_embeddings,
        rope_theta=10000.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
    ).to_dict()
    # The target captures after each named layer and -1 names the embeddings, so a two-layer target exposes two states; a null draft vocabulary shares the target's head.
    eagle3_config.update(
        architectures=["LlamaForCausalLMEagle3"],
        draft_vocab_size=None,
        eagle_config={"eagle_aux_hidden_state_layer_ids": [-1, 0]},
    )
    (eagle3_dir / "config.json").write_text(json.dumps(eagle3_config))
    return directory / "target", {"DFLASH": dflash_dir, "EAGLE3": eagle3_dir}


def session_payload(
    session_id: str, request_id: str, token_ids: list[int], max_new_tokens: int
) -> StagePayload:
    operation = SessionOperation(
        "append",
        SessionIdentity(session_id),
        ("tiny",),
        TimedChunk("text", 0, 0, 0, {}, format="text_delta"),
    )
    return StagePayload(
        request_id,
        OmniRequest(
            None,
            params={"max_new_tokens": max_new_tokens},
            metadata={SESSION_METADATA_KEY: operation.to_dict()},
        ),
        {"token_ids": token_ids},
    )


def run_tiny_sessions(
    target_path: str,
    speculative_algorithm: str | None,
    draft_path: str | None,
    captured: bool,
) -> dict[str, list[int]]:
    """Drive two sessions through appends, a history cut, a rewind, a cancel and a reset; return each completed turn's output ids by request id."""
    server_args = build_sglang_server_args(
        target_path,
        context_length=128,
        max_running_requests=4,
        max_total_tokens=256,
        max_prefill_tokens=128,
        chunked_prefill_size=-1,
        mem_fraction_static=0.1,
        dtype="bfloat16",
        attention_backend="fa3",
        sampling_backend="pytorch",
        disable_radix_cache=True,
        enable_streaming_session=True,
        enable_mixed_chunk=False,
        enable_unified_memory=False,
        page_size=1,
        skip_tokenizer_init=True,
        disable_cuda_graph=not captured,
        cuda_graph_bs=[1, 2],
        cuda_graph_max_bs=2,
        enable_torch_compile=False,
        speculative_algorithm=speculative_algorithm,
        speculative_draft_model_path=draft_path,
        **SPECULATIVE_SERVER_ARGS.get(speculative_algorithm, {}),
    )
    (
        tp_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        model_config,
    ) = create_sglang_infrastructure(server_args, 0, supports_speculative_decoding=True)
    adapter = TinyTextAdapter(
        DraftSessionKV(
            token_to_kv_pool_allocator.get_kvcache(),
            tp_worker.model_runner.model.model.layers[
                0
            ].self_attn.rotary_emb.cos_sin_cache,
        )
    )
    scheduler = OmniScheduler(
        tp_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        server_args,
        model_config,
        model_runner=ModelRunner(tp_worker, SGLangOutputProcessor()),
        session_adapter=adapter,
        enable_async_decode=False,
    )
    initial_free_slot_count = token_to_kv_pool_allocator.available_size()
    initial_free_request_row_count = req_to_token_pool.available_size()
    output_ids_by_request_id: dict[str, list[int]] = {}
    streamed_ids_by_request_id: dict[str, list[int]] = {}
    graph_replay_count = 0

    def run_step() -> None:
        nonlocal graph_replay_count
        scheduler.process_input_requests([])
        batch = scheduler.get_next_batch_to_run()
        scheduler.cur_batch = batch
        if batch is not None:
            batch_result = scheduler.run_batch(batch)
            graph_replay_count += int(batch_result.can_run_cuda_graph)
            scheduler.process_batch_result(batch, batch_result)
        else:
            pass
        scheduler.last_batch = batch

    def drain_outbox() -> None:
        while not scheduler.outbox.empty():
            message = scheduler.outbox.get_nowait()
            if message.type == "error":
                raise AssertionError(message.data)
            elif message.type == "result":
                output_ids_by_request_id[message.request_id] = message.data.data[
                    "output_ids"
                ]
            elif message.type == "stream":
                streamed_ids_by_request_id.setdefault(message.request_id, []).append(
                    int(message.data["payload"]["text"])
                )
            else:
                pass

    def run_until_complete(payloads: list[StagePayload]) -> None:
        scheduler.process_input_requests(payloads)
        for _ in range(100):
            run_step()
            drain_outbox()
            if all(
                payload.request_id in output_ids_by_request_id for payload in payloads
            ):
                return
            else:
                pass
        raise AssertionError("session append did not complete")

    active_stage_token = set_active_stage("tiny")
    with torch.inference_mode():
        for session_id in ("a", "b"):
            scheduler.session_bridge.apply_operation(
                StagePayload("open", OmniRequest(None), {}),
                SessionOperation("open", SessionIdentity(session_id), ("tiny",)),
            )
        run_until_complete(
            [
                session_payload("a", "a0", [1, 2, 3, 4], 7),
                session_payload("b", "b0", [5, 6, 7], 7),
            ]
        )
        run_until_complete(
            [
                session_payload("a", "a1", [8, 9], 5),
                session_payload("b", "b1", [10, 11], 5),
            ]
        )
        adapter.next_cut = HistoryCut(((1, 3),))
        run_until_complete([session_payload("a", "cut", [12, 13], 5)])
        adapter.next_preparation = ARSessionPreparation(kept_token_count=4)
        run_until_complete([session_payload("a", "rewind", [14], 5)])
        scheduler.process_input_requests([session_payload("b", "cancel", [15, 16], 30)])
        run_step()
        scheduler.session_bridge.cancel("cancel")
        drain_outbox()
        run_until_complete([session_payload("b", "after_cancel", [17], 5)])
        adapter.next_preparation = ARSessionPreparation(reset_history=True)
        run_until_complete([session_payload("a", "reset", [18, 19], 5)])
        for session in list(scheduler.session_bridge.sessions.values()):
            scheduler.session_bridge.close_session(session)
        for request_id, output_ids in output_ids_by_request_id.items():
            assert streamed_ids_by_request_id[request_id] == output_ids
        assert not tree_cache.session_records()
        assert token_to_kv_pool_allocator.available_size() == initial_free_slot_count
        assert req_to_token_pool.available_size() == initial_free_request_row_count
        if captured:
            assert graph_replay_count > 0
        else:
            pass
    scheduler.stop()
    reset_active_stage(active_stage_token)
    return output_ids_by_request_id
