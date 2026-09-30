# SPDX-License-Identifier: Apache-2.0
"""MOSS-TD integration with upstream PD state, factories, and DP placement."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.config.schema import ProcessConfig
from sglang_omni.config.topology import compile_logical_processes
from sglang_omni.models.moss_transcribe_diarize import pd, stages
from sglang_omni.models.moss_transcribe_diarize.config import (
    MossTranscribeDiarizePDPipelineConfig,
)
from sglang_omni.models.moss_transcribe_diarize.request_builders import (
    MossTranscribeDiarizeRequestData,
    make_moss_transcribe_diarize_scheduler_adapters,
    make_moss_transcribe_diarize_stream_output_builder,
    postprocess_moss_transcribe_diarize_text,
)
from sglang_omni.pipeline.replicas import (
    RoundRobinBindingPolicy,
    assign_replica_bindings,
    expand_replica_stages,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.pd_utils import (
    DecodeContinuation,
    continuation_from_req,
    req_from_continuation,
)
from tests.unit_test.moss_transcribe_diarize.test_pipeline import stub_factory_env
from tests.unit_test.moss_transcribe_diarize.test_request_builders import FakeProcessor
from tests.unit_test.moss_transcribe_diarize.test_stream_output_builder import (
    ByteTokenizer,
    make_req_output,
)
from tests.unit_test.pipeline.test_pd_utils import make_allocation, prefill_req


def test_pd_example_loads() -> None:
    path = Path(__file__).resolve().parents[3] / "examples/configs/moss_td_pd.yaml"
    config = ConfigManager.from_file(str(path)).config
    assert isinstance(config, MossTranscribeDiarizePDPipelineConfig)
    assert config.resolved_entry_stage == "asr_prefill"
    assert config.terminal_stages == ["asr_decode"]
    assert int(config.resolved_env_defaults()["OMP_NUM_THREADS"]) >= 1


@pytest.mark.parametrize("p_count,d_count", [(1, 1), (1, 2), (2, 2)])
def test_pd_layout_resolves_concrete_factory_identity(p_count, d_count) -> None:
    processes = {
        name: ProcessConfig(num_replicas=count, replica_devices=list(devices))
        for name, count, devices in (
            ("asr_prefill", p_count, range(p_count)),
            ("asr_decode", d_count, range(p_count, p_count + d_count)),
        )
        if count > 1
    }
    config = MossTranscribeDiarizePDPipelineConfig(model_path="m", processes=processes)
    plan, logical = compile_logical_processes(config)
    expanded, topology = expand_replica_stages(logical, plan)
    assert len(expanded) == p_count + d_count
    for stage in expanded:
        args = resolve_stage_factory_args(stage, config, gpu_id=stage.gpu)
        assert args["stage_name"] == stage.name
        assert args["gpu_id"] == stage.gpu
        assert args["pd_role"] == ("prefill" if "prefill" in stage.name else "decode")
        assert args["server_args_overrides"]["disable_radix_cache"] is True
        assert args["server_args_overrides"]["page_size"] == 1
    policy = RoundRobinBindingPolicy()
    bindings = [assign_replica_bindings(plan, policy, str(i)) for i in range(4)]
    for name, count in (("asr_prefill", p_count), ("asr_decode", d_count)):
        if count > 1:
            assert [binding[name] for binding in bindings] == [
                i % count for i in range(4)
            ]
            assert topology.to_dict()[name] == [f"{name}@r{i}" for i in range(count)]


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_pd_factory_uses_shared_scheduler_and_only_prefill_encoder(monkeypatch, role):
    calls = stub_factory_env(monkeypatch, want_cuda_graph=True)
    request_builder = Mock(return_value="prefill request")
    monkeypatch.setattr(
        pd.MossTranscribeDiarizeEngineBuilder,
        "make_adapters",
        lambda self, model: (request_builder, Mock()),
    )
    captured = {}

    def scheduler(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    scheduler_name = (
        "OmniPrefillScheduler" if role == "prefill" else "OmniDecodeScheduler"
    )
    monkeypatch.setattr(pd, scheduler_name, scheduler)
    config = MossTranscribeDiarizePDPipelineConfig(model_path="m")
    stage = config.stage_named(f"asr_{role}").model_copy(
        update={"name": f"asr_{role}@r1"}
    )
    args = resolve_stage_factory_args(stage, config, gpu_id=stage.gpu)
    stages.create_sglang_moss_transcribe_diarize_executor(**args)

    assert captured["stage_name"] == f"asr_{role}@r1"
    assert ("stream_output_builder" in captured) == (role == "decode")
    assert captured["enable_async_decode"] is False
    assert bool(calls["encoder_services"]) == (role == "prefill")
    assert bool(calls["init_encoder_graphs"]) == (role == "prefill")
    if role == "prefill":
        assert captured["partner_stage"] == "asr_decode"
        assert captured["state_builder"] is pd.build_decode_state
    else:
        assert captured["state_restorer"] is pd.restore_decode_state
        assert captured["resume_schema"] == pd.MOSS_TD_PD_RESUME_SCHEMA
        with pytest.raises(ValueError, match="KV continuation"):
            captured["request_builder"](StagePayload("r", OmniRequest(None), None))

    payload = StagePayload("r", OmniRequest(None, params={"stream": True}), None)
    if role == "prefill":
        assert captured["request_builder"](payload) == "prefill request"
        request_builder.assert_called_once_with(payload)
    else:
        with pytest.raises(ValueError, match="KV continuation"):
            captured["request_builder"](payload)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "first_token,max_new_tokens,decode_tokens,expected_text",
    [
        (42, 4, [43, 44, 2], "ABC"),
        (2, 4, [], ""),
        (42, 1, [], "A"),
        (42, 2, [43], "AB"),
        (45, 1, [], "\ufffd"),
        (45, 2, [46], "\ufffd"),
        (45, 3, [46, 47], "你"),
    ],
)
def test_moss_continuation_preserves_first_token_stops_and_result(
    monkeypatch: pytest.MonkeyPatch,
    first_token: int,
    max_new_tokens: int,
    decode_tokens: list[int],
    expected_text: str,
    stream: bool,
) -> None:
    source = prefill_req(max_new_tokens=max_new_tokens)
    source.output_ids[0] = first_token
    payload = StagePayload(
        source.rid,
        OmniRequest(
            inputs={"audio": torch.ones(160)},
            params={"stream": stream, "language": "en"},
            metadata={"audio": torch.ones(160)},
        ),
        data={"embedding": torch.ones(4, 8)},
    )
    source.omni_data = MossTranscribeDiarizeRequestData(
        req=source,
        input_ids=torch.tensor(list(source.origin_input_ids)),
        output_ids=source.output_ids,
        stage_payload=payload,
        prompt_token_ids=list(source.origin_input_ids),
        audio_duration_s=12.5,
        language="en",
        engine_start_s=100.0,
    )
    continuation = DecodeContinuation.decode(
        continuation_from_req(source, "transfer-1", pd.build_decode_state).encode()
    )
    assert continuation.stage_payload["data"] is None
    assert continuation.stage_payload["request"]["inputs"] is None
    assert continuation.stage_payload["request"]["metadata"] == {}
    pool = ReqToTokenPool(
        size=1, max_context_len=32, device="cpu", enable_memory_saver=False
    )
    resumed = req_from_continuation(
        continuation,
        make_allocation(),
        req_to_token_pool=pool,
        state_restorer=pd.restore_decode_state,
    )
    assert list(resumed.output_ids) == [first_token]
    assert resumed.sampling_params.max_new_tokens == max_new_tokens
    assert resumed.sampling_params.temperature == source.sampling_params.temperature
    assert resumed.sampling_params.stop_token_ids == {2}
    assert resumed.multimodal_inputs is None
    assert resumed.kv.kv_committed_len == 3
    assert pool.req_to_token[resumed.kv.req_pool_idx, :3].tolist() == [7, 8, 9]
    resumed.update_finish_state()
    assert resumed.finished() == (first_token == 2 or max_new_tokens == 1)

    processor = FakeProcessor()
    _, adapter = make_moss_transcribe_diarize_scheduler_adapters(
        processor, processor.tokenizer, 4, 32
    )
    monkeypatch.setattr("time.perf_counter", lambda: 101.0)
    assert adapter(resumed.omni_data).data == adapter(source.omni_data).data
    assert resumed.omni_data.enforce_request_limits is True

    tokenizer = ByteTokenizer(
        {
            2: b"<|im_end|>",
            42: b"A",
            43: b"B",
            44: b"C",
            45: b"\xe4",
            46: b"\xbd",
            47: b"\xa0",
        },
        special_token_ids={2},
    )
    stream_builder = make_moss_transcribe_diarize_stream_output_builder(
        tokenizer, eos_token_id=2, min_emit_interval_s=3600.0
    )
    messages = []
    for token_id in decode_tokens:
        assert not resumed.finished()
        messages.extend(
            stream_builder(resumed.rid, resumed.omni_data, make_req_output(token_id))
        )
        resumed.output_ids.append(token_id)
        resumed.update_finish_state()
    assert resumed.finished()
    messages.extend(stream_builder.flush(resumed.rid, resumed.omni_data))
    assert stream_builder.flush(resumed.rid, resumed.omni_data) == []
    streamed_text = "".join(message.data["text"] for message in messages)
    if stream:
        assert streamed_text == expected_text
        assert postprocess_moss_transcribe_diarize_text(streamed_text) == (
            postprocess_moss_transcribe_diarize_text(
                tokenizer.decode(resumed.output_ids)
            )
        )
        assert all(message.request_id == resumed.rid for message in messages)
    else:
        assert messages == []
    pool.free(resumed)
    assert pool.available_size() == 1


@pytest.mark.parametrize("resume", [None, {}, {"schema": "wrong"}])
def test_moss_rejects_incompatible_resume(resume):
    with pytest.raises(ValueError, match="invalid MOSS-TD PD resume state"):
        pd.restore_decode_state(SimpleNamespace(), SimpleNamespace(), resume)
