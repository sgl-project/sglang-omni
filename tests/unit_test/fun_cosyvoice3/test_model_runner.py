# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from array import array
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from flashinfer.sampling import get_sampling_module
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.penaltylib.repetition_penalty import apply_scaling_penalties
from sglang.srt.utils import DynamicGradMode
from torch._dynamo.utils import counters

import sglang_omni.models.fun_cosyvoice3.model_runner as model_runner_module
from sglang_omni.models.fun_cosyvoice3.model_runner import FunCosyVoice3ModelRunner
from sglang_omni.models.fun_cosyvoice3.request_builders import (
    CosyVoice3SGLangRequestData,
)
from sglang_omni.models.fun_cosyvoice3.sglang_model import (
    EOS_ID,
    TOTAL_VOCAB_SIZE,
    VOCAB_SIZE,
    FunCosyVoice3SGLangModel,
)
from sglang_omni.sampling.seed import SAMPLING_SEED_MASK
from tests.unit_test.fixtures.accelerator import require_cuda
from tests.unit_test.fun_cosyvoice3.prefill_graph_engine import PrefillRecord

REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINE_TIMEOUT_SECONDS = 900


def test_cosyvoice3_runner_collects_speech_tokens_and_skips_eos() -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    runner.outbox = None
    requests = [
        SimpleNamespace(data=CosyVoice3SGLangRequestData()),
        SimpleNamespace(data=CosyVoice3SGLangRequestData()),
    ]
    result = SimpleNamespace(next_token_ids=torch.tensor([[EOS_ID], [13]]))

    runner.collect_tokens(result, None, None, requests)

    assert requests[0].data.output_codes == []
    assert [code.item() for code in requests[1].data.output_codes] == [13]
    assert requests[1].data.output_codes[0].dtype == torch.long
    assert runner.resolve_host_token_ids(result).tolist() == [EOS_ID, 13]


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cosyvoice3_runner_reads_cuda_token_ids_through_pinned_staging() -> None:
    """A pageable read of the step's ids stalls other threads' kernel launches."""
    runner = object.__new__(FunCosyVoice3ModelRunner)
    runner.outbox = None
    runner.token_id_host_bufs = None
    runner.token_id_host_slot = 0
    runner.token_id_events = []
    runner.token_id_event_slot = 0
    requests = [
        SimpleNamespace(data=CosyVoice3SGLangRequestData()),
        SimpleNamespace(data=CosyVoice3SGLangRequestData()),
    ]
    result = SimpleNamespace(
        next_token_ids=torch.tensor([EOS_ID, 13], dtype=torch.int32, device="cuda")
    )

    runner.collect_tokens(result, None, None, requests)

    host_token_ids = runner.resolve_host_token_ids(result)
    assert host_token_ids.is_pinned()
    assert host_token_ids.tolist() == [EOS_ID, 13]


def test_cosyvoice3_runner_skips_all_control_tokens() -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    runner.outbox = None
    requests = [SimpleNamespace(data=CosyVoice3SGLangRequestData())]

    runner.collect_tokens(
        SimpleNamespace(next_token_ids=torch.tensor([VOCAB_SIZE + 3])),
        None,
        None,
        requests,
    )

    assert requests[0].data.output_codes == []


def test_cosyvoice3_runner_samples_before_prefill_and_decode_collection() -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    sampled = torch.tensor([13])

    def forward_without_tokens(*args) -> SimpleNamespace:
        return SimpleNamespace(next_token_ids=None, logits_output=object())

    runner.custom_prefill_forward = forward_without_tokens
    runner.tp_worker = SimpleNamespace(forward_batch_generation=forward_without_tokens)
    runner.sample_next_token_ids = lambda *args: sampled
    schedule_batch = SimpleNamespace(is_prefill_only=False)

    prefill = runner.prepare_and_forward(
        SimpleNamespace(), schedule_batch, [], is_prefill=True
    )
    decode = runner.prepare_and_forward(
        SimpleNamespace(), schedule_batch, [], is_prefill=False
    )

    assert runner.sample_before_post_prefill(None, None, []) is True
    assert runner.sample_before_post_decode(None, None, []) is True
    assert prefill.next_token_ids is sampled
    assert decode.next_token_ids is sampled


def test_cosyvoice3_torch_mps_seed_avoids_float64_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import nullcontext

    sampling_info = SimpleNamespace(sampling_seed=None, is_all_greedy=True)
    sampled_with = []

    class Runner(FunCosyVoice3ModelRunner):
        def apply_repetition_penalty(self, logits_output, requests):
            del logits_output, requests

        def apply_codec_suppress_tokens(self, logits_output, requests):
            del logits_output, requests

        def install_sampling_seeds(self, forward_batch, requests):
            del requests
            forward_batch.sampling_info.sampling_seed = torch.tensor([7])

    runner = object.__new__(Runner)
    runner.cosyvoice3_recent_tokens = {}
    runner.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            sample=lambda logits_output, forward_batch: sampled_with.append(
                forward_batch.sampling_info.sampling_seed
            )
            or torch.tensor([13])
        )
    )
    manual_seeds = []
    compiler_stances = []
    forked_devices = []
    monkeypatch.setattr(torch, "manual_seed", manual_seeds.append)
    monkeypatch.setattr(
        model_runner_module.current_platform,
        "is_float64_supported",
        lambda: False,
    )
    monkeypatch.setattr(
        torch.compiler,
        "set_stance",
        lambda stance: compiler_stances.append(stance) or nullcontext(),
    )
    monkeypatch.setattr(
        torch.random,
        "fork_rng",
        lambda *, devices, device_type: forked_devices.append((devices, device_type))
        or nullcontext(),
    )
    request = SimpleNamespace(
        request_id="req",
        data=SimpleNamespace(
            return_logprob=False,
            req=SimpleNamespace(
                origin_input_ids=[0, 0, 0],
                output_ids=[1, 2],
                sampling_params=SimpleNamespace(sampling_seed=7),
            ),
        ),
    )
    logits_output = SimpleNamespace(
        next_token_logits=SimpleNamespace(
            device=SimpleNamespace(type="mps", index=0),
        )
    )
    forward_batch = SimpleNamespace(sampling_info=sampling_info)

    token_ids = runner.sample_next_token_ids(
        logits_output,
        forward_batch,
        None,
        [request],
    )

    assert token_ids.tolist() == [13]
    assert sampled_with == [None]
    assert sampling_info.sampling_seed.tolist() == [7]
    assert manual_seeds == [(7 + 4 * 0x9E3779B1) & SAMPLING_SEED_MASK]
    assert compiler_stances == ["force_eager"]
    assert forked_devices == [([0], "mps")]


def test_cosyvoice3_torch_mps_ras_redraws_recent_speech_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    runner.cosyvoice3_recent_tokens = {"req": [4, 7, 9]}
    sampling_info = SimpleNamespace(is_all_greedy=False)
    logits_output = SimpleNamespace(
        next_token_logits=torch.tensor(
            [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
        ),
        next_token_logprobs=torch.tensor([-0.1]),
    )
    sampled = []

    def fake_multinomial(probs, num_samples):
        sampled.append((probs, num_samples))
        return torch.tensor([[6]], dtype=torch.long)

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)
    request = SimpleNamespace(request_id="req")

    result = runner.apply_ras_fallback(
        logits_output,
        torch.tensor([9], dtype=torch.int32),
        sampling_info,
        [request],
    )

    assert result.tolist() == [6]
    assert len(sampled) == 1
    assert sampled[0][1] == 1
    assert sampled[0][0][0, 9].item() == 0.0
    assert runner.cosyvoice3_recent_tokens["req"][-1] == 6
    assert torch.equal(
        logits_output.next_token_logprobs,
        torch.log(torch.tensor([0.7])),
    )


def test_cosyvoice3_torch_mps_ras_keeps_non_repeated_token() -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    runner.cosyvoice3_recent_tokens = {"req": [4, 7, 9]}
    sampling_info = SimpleNamespace(is_all_greedy=False)
    logits_output = SimpleNamespace(
        next_token_logits=torch.ones((1, VOCAB_SIZE), dtype=torch.float32),
        next_token_logprobs=None,
    )
    request = SimpleNamespace(request_id="req")

    result = runner.apply_ras_fallback(
        logits_output,
        torch.tensor([6], dtype=torch.int32),
        sampling_info,
        [request],
    )

    assert result.tolist() == [6]
    assert runner.cosyvoice3_recent_tokens["req"][-1] == 6


def test_cosyvoice3_torch_mps_clears_ras_history_on_finish() -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    runner.cosyvoice3_recent_tokens = {"req": [1], "keep": [2]}

    runner.on_request_finished("req", None)

    assert runner.cosyvoice3_recent_tokens == {"keep": [2]}


@pytest.mark.parametrize("use_ascend_backend", [False, True])
def test_cosyvoice3_ras_masks_and_redraws_repeated_speech_tokens(
    use_ascend_backend: bool,
) -> None:
    """A repeated speech token must not crowd out EOS until the length cap."""
    probs = torch.zeros((6, EOS_ID + 1))
    probs[0, 9], probs[0, EOS_ID] = 0.9, 0.1
    probs[[1, 2, 3, 5], 2] = 0.4
    probs[1, 5] = probs[2, 9] = probs[3, EOS_ID] = probs[5, 9] = 0.6
    probs[4, 9] = 1.0
    histories = [[9] + [4] * 9, [1, 2, 3], [9], [EOS_ID], [9], [9] + [4] * 10]
    runner = object.__new__(FunCosyVoice3ModelRunner)
    runner.outbox = None
    runner.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            sampler=SimpleNamespace(use_ascend_backend=use_ascend_backend)
        )
    )
    requests = [
        SimpleNamespace(
            data=SimpleNamespace(
                output_codes=[],
                req=SimpleNamespace(
                    output_ids=array("q", output_ids),
                    sampling_params=SimpleNamespace(top_k=1 if row == 2 else 20),
                ),
            )
        )
        for row, output_ids in enumerate(histories)
    ]
    result = SimpleNamespace(
        next_token_ids=torch.tensor([9, 5, 9, EOS_ID, 9, 9], dtype=torch.int32),
        logits_output=SimpleNamespace(
            next_token_logits=torch.log(probs) + 1 if use_ascend_backend else probs
        ),
    )
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        sampling_info=SimpleNamespace(sampling_seed=None),
        positions=torch.arange(6),
    )

    runner.collect_tokens(result, forward_batch, None, requests)

    assert result.next_token_ids.tolist() == [EOS_ID, 5, 9, EOS_ID, 9, 9]
    assert [
        [int(code) for code in request.data.output_codes] for request in requests
    ] == [[], [5], [9], [], [9], [9]]


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cosyvoice3_seeded_ras_redraw_of_a_row_subset_matches_the_full_draw() -> None:
    rows = 256
    probs = torch.zeros((rows, 8), device="cuda")
    probs[:, 3], probs[:, 5], probs[:, 6] = 0.9, 0.06, 0.04
    runner = object.__new__(FunCosyVoice3ModelRunner)
    runner.token_id_host_bufs = None
    runner.token_id_host_slot = 0
    runner.token_id_events = []
    runner.token_id_event_slot = 0
    runner.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(sampler=SimpleNamespace(use_ascend_backend=False))
    )
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        sampling_info=SimpleNamespace(
            sampling_seed=torch.arange(1, rows + 1, device="cuda")
        ),
        positions=torch.arange(rows, device="cuda"),
    )
    subset = range(1, rows, 3)

    def redraw(repeating_rows: range) -> list[int]:
        requests = [
            SimpleNamespace(
                data=SimpleNamespace(
                    req=SimpleNamespace(
                        output_ids=array("q", [3 if row in repeating_rows else 4]),
                        sampling_params=SimpleNamespace(top_k=20),
                    )
                )
            )
            for row in range(rows)
        ]
        result = SimpleNamespace(
            next_token_ids=torch.full((rows,), 3, dtype=torch.int32, device="cuda"),
            logits_output=SimpleNamespace(next_token_logits=probs),
        )
        return runner.redraw_repeated_tokens(
            result, forward_batch, requests, [3] * rows
        )

    full = redraw(range(rows))
    partial = redraw(subset)

    assert 3 not in full
    assert [partial[row] for row in subset] == [full[row] for row in subset]


def test_cosyvoice3_load_weights_maps_custom_and_backbone_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the loader path, not only the standalone key mapper."""
    model = object.__new__(FunCosyVoice3SGLangModel)
    torch.nn.Module.__init__(model)
    speech_embedding = torch.nn.Parameter(torch.zeros(2, 3))
    decoder = torch.nn.Parameter(torch.zeros(2, 3))
    model.cached_params_dict = {
        "speech_embedding.weight": speech_embedding,
        "llm_decoder.weight": decoder,
    }
    forwarded = []
    monkeypatch.setattr(
        "sglang.srt.models.qwen2.Qwen2ForCausalLM.load_weights",
        lambda _self, weights: forwarded.extend(weights),
    )

    speech_value = torch.ones(2, 3)
    decoder_value = torch.full((2, 3), 2.0)
    model.load_weights(
        [
            ("speech_embedding.weight", speech_value),
            ("llm_decoder.weight", decoder_value),
            ("llm.model.lm_head.weight", torch.ones(2, 3)),
            ("llm.model.model.layers.0.weight", torch.full((3, 3), 3.0)),
        ]
    )

    assert torch.equal(speech_embedding, speech_value)
    assert torch.equal(decoder, decoder_value)
    assert len(forwarded) == 1
    assert forwarded[0][0] == "model.layers.0.weight"
    assert torch.equal(forwarded[0][1], torch.full((3, 3), 3.0))


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cosyvoice3_speech_head_returns_float32_logits() -> None:
    """bf16 logits make the sampler's temperature and softmax run in bf16."""
    device = "cuda"
    model = object.__new__(FunCosyVoice3SGLangModel)
    torch.nn.Module.__init__(model)
    generator = torch.Generator().manual_seed(0)
    hidden_states = (torch.randn(4, 64, generator=generator) * 4).to(
        device=device, dtype=torch.bfloat16
    )
    model.speech_embedding = torch.nn.Embedding(8, 64).to(device, torch.bfloat16)
    model.llm_decoder = torch.nn.Linear(64, 8, bias=False).to(device, torch.bfloat16)
    model.model = (
        lambda input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors: (
            hidden_states
        )
    )
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True, is_extend=lambda: False)
    )
    token_ids = torch.zeros(4, dtype=torch.long, device=device)

    output = model(token_ids, token_ids, forward_batch)

    product = hidden_states.double() @ model.llm_decoder.weight.double().T
    assert output.next_token_logits.dtype == torch.float32
    assert (output.next_token_logits.double() - product).abs().max() < 1e-3


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cosyvoice3_precompile_leaves_the_first_penalized_samples_nothing_to_compile() -> (
    None
):
    model = object.__new__(FunCosyVoice3SGLangModel)
    torch.nn.Module.__init__(model)
    model.speech_embedding = torch.nn.Embedding(TOTAL_VOCAB_SIZE, 64).to(
        "cuda", torch.bfloat16
    )
    model.llm_decoder = torch.nn.Linear(64, TOTAL_VOCAB_SIZE, bias=False).to(
        "cuda", torch.bfloat16
    )
    model.model = (
        lambda input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors: (
            input_embeds
        )
    )
    torch.compiler.reset()

    with get_context().override_server_args(sampling_backend="flashinfer"):
        model.precompile_kernels_after_loading()
    compiled_graphs = counters["stats"]["unique_graphs"]

    with DynamicGradMode():
        for rows in (1, 3):
            forward_batch = SimpleNamespace(
                forward_mode=SimpleNamespace(
                    is_decode=lambda: False, is_extend=lambda: True
                ),
                extend_seq_lens=torch.full((rows,), 2, device="cuda"),
            )
            token_ids = torch.zeros(2 * rows, dtype=torch.long, device="cuda")
            hidden_states = torch.randn(
                2 * rows, 64, device="cuda", dtype=torch.bfloat16
            )
            logits = model(
                token_ids, token_ids, forward_batch, input_embeds=hidden_states
            ).next_token_logits
            apply_scaling_penalties(
                logits, torch.full(logits.shape, 1.21, device="cuda")
            )
    assert counters["stats"]["unique_graphs"] == compiled_graphs


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("sampling_backend", ["pytorch", "flashinfer"])
def test_cosyvoice3_load_hook_loads_flashinfer_sampling_only_for_its_backend(
    sampling_backend: str,
) -> None:
    model = object.__new__(FunCosyVoice3SGLangModel)
    torch.nn.Module.__init__(model)
    model.llm_decoder = torch.nn.Linear(64, TOTAL_VOCAB_SIZE, bias=False).to(
        "cuda", torch.bfloat16
    )
    get_sampling_module.cache_clear()

    with get_context().override_server_args(sampling_backend=sampling_backend):
        model.precompile_kernels_after_loading()

    assert get_sampling_module.cache_info().currsize == int(
        sampling_backend == "flashinfer"
    )


def test_cosyvoice3_runner_builds_prefill_embedding_slice_after_prefix() -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    runner.model = torch.nn.Linear(3, 3, bias=False)
    requests = [
        SimpleNamespace(
            data=SimpleNamespace(
                req=SimpleNamespace(extend_len=2, prefix_len=1),
                prompt_input_embeds=torch.arange(12, dtype=torch.float32).reshape(3, 4),
            )
        )
    ]
    forward_batch = SimpleNamespace(input_ids=torch.zeros(2, dtype=torch.long))

    result = runner.build_prefill_input_embeds(forward_batch, requests)

    assert torch.equal(
        result, torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]], dtype=torch.float32)
    )


@pytest.mark.accelerator
def test_cuda_prefill_graphs_replay_the_composed_rows_as_the_eager_forward(
    tmp_path: Path,
) -> None:
    require_cuda()
    output_path = tmp_path / "prefills.pt"
    engine = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.unit_test.fun_cosyvoice3.prefill_graph_engine",
            "--checkpoint-directory",
            str(tmp_path / "checkpoint"),
            "--output",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=ENGINE_TIMEOUT_SECONDS,
    )
    assert engine.returncode == 0, engine.stdout[-4000:] + engine.stderr[-4000:]
    records = [
        PrefillRecord(**record) for record in torch.load(output_path, weights_only=True)
    ]

    assert all(record.is_graph_replay for record in records)
    for record in records:
        torch.testing.assert_close(
            record.graph_hidden_states, record.eager_hidden_states
        )
    assert any(len(record.extend_token_counts) > 1 for record in records)
    assert any(
        record.extend_token_counts == [1] and record.prefix_token_counts[0] > 0
        for record in records
    )
    assert any(
        record.extend_token_counts[0] > 1 and record.prefix_token_counts[0] > 0
        for record in records
    )
