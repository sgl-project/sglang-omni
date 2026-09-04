# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3.model_runner import (
    FunCosyVoice3MlxSchedulerModelRunner,
    FunCosyVoice3ModelRunner,
)
from sglang_omni.models.fun_cosyvoice3.sglang_model import (
    EOS_ID,
    VOCAB_SIZE,
    FunCosyVoice3SGLangModel,
)
from sglang_omni.sampling.seed import SAMPLING_SEED_MASK


def test_cosyvoice3_runner_collects_speech_tokens_and_skips_eos() -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    requests = [
        SimpleNamespace(data=SimpleNamespace(output_codes=[])),
        SimpleNamespace(data=SimpleNamespace(output_codes=[])),
    ]
    result = SimpleNamespace(next_token_ids=torch.tensor([[EOS_ID], [13]]))

    runner._collect_tokens(result, None, None, requests)

    assert requests[0].data.output_codes == []
    assert [code.item() for code in requests[1].data.output_codes] == [13]
    assert requests[1].data.output_codes[0].dtype == torch.long


def test_cosyvoice3_runner_skips_all_control_tokens() -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    requests = [SimpleNamespace(data=SimpleNamespace(output_codes=[]))]

    runner._collect_tokens(
        SimpleNamespace(next_token_ids=torch.tensor([VOCAB_SIZE + 3])),
        None,
        None,
        requests,
    )

    assert requests[0].data.output_codes == []


def test_cosyvoice3_runner_samples_before_prefill_and_decode_collection() -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)

    assert runner.sample_before_post_prefill(None, None, []) is True
    assert runner.sample_before_post_decode(None, None, []) is True


def test_cosyvoice3_mlx_runner_collects_exact_scheduler_rows() -> None:
    runner = object.__new__(FunCosyVoice3MlxSchedulerModelRunner)
    runner._resolve_skip_rids = set()
    requests = [
        SimpleNamespace(request_id="first", data=SimpleNamespace(output_codes=[])),
        SimpleNamespace(request_id="second", data=SimpleNamespace(output_codes=[])),
    ]
    scheduler_output = SimpleNamespace(requests=requests)

    runner.post_process_outputs(
        SimpleNamespace(next_token_ids=torch.tensor([13, VOCAB_SIZE + 3])),
        scheduler_output,
        {},
    )

    assert [code.item() for code in requests[0].data.output_codes] == [13]
    assert requests[1].data.output_codes == []

    runner._resolve_skip_rids = {"first"}
    runner.post_process_outputs(
        SimpleNamespace(next_token_ids=torch.tensor([14, 15])),
        scheduler_output,
        {},
    )
    assert [code.item() for code in requests[0].data.output_codes] == [13]
    assert [code.item() for code in requests[1].data.output_codes] == [15]

    with pytest.raises(RuntimeError, match="row count"):
        runner.post_process_outputs(
            SimpleNamespace(next_token_ids=torch.tensor([13])),
            scheduler_output,
            {},
        )


def test_cosyvoice3_mlx_lookahead_accepts_owned_history_constraints() -> None:
    runner = object.__new__(FunCosyVoice3MlxSchedulerModelRunner)
    runner._last_mlx_pending = None
    req = SimpleNamespace(
        rid="req",
        sampling_params=SimpleNamespace(
            frequency_penalty=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.1,
            min_new_tokens=4,
            sampling_seed=7,
        ),
        custom_logit_processor=None,
    )

    assert runner.lookahead_eligible(SimpleNamespace(reqs=[req], has_grammar=False))
    assert not runner.lookahead_eligible(SimpleNamespace(reqs=[req], has_grammar=True))

    runner._last_mlx_pending = SimpleNamespace(
        launch=SimpleNamespace(mode="decode"),
        reqs=[SimpleNamespace(rid="another")],
    )
    assert not runner.lookahead_eligible(SimpleNamespace(reqs=[req], has_grammar=False))


def test_cosyvoice3_torch_mps_seed_avoids_float64_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import nullcontext

    sampling_info = SimpleNamespace(sampling_seed=None)
    sampled_with = []

    class _Runner(FunCosyVoice3ModelRunner):
        def _apply_repetition_penalty(self, logits_output, requests):
            del logits_output, requests

        def _apply_codec_suppress_tokens(self, logits_output, requests):
            del logits_output, requests

        def _install_sampling_seeds(self, forward_batch, requests):
            del requests
            forward_batch.sampling_info.sampling_seed = torch.tensor([7])

    runner = object.__new__(_Runner)
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
        data=SimpleNamespace(
            return_logprob=False,
            req=SimpleNamespace(
                origin_input_ids=[0, 0, 0],
                output_ids=[1, 2],
                sampling_params=SimpleNamespace(sampling_seed=7),
            ),
        )
    )
    logits_output = SimpleNamespace(
        next_token_logits=SimpleNamespace(
            device=SimpleNamespace(type="mps", index=0),
        )
    )
    forward_batch = SimpleNamespace(sampling_info=sampling_info)

    token_ids = runner._sample_next_token_ids(
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


def test_cosyvoice3_load_weights_maps_custom_and_backbone_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the loader path, not only the standalone key mapper."""
    model = object.__new__(FunCosyVoice3SGLangModel)
    torch.nn.Module.__init__(model)
    speech_embedding = torch.nn.Parameter(torch.zeros(2, 3))
    decoder = torch.nn.Parameter(torch.zeros(2, 3))
    model._cached_params_dict = {
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


def test_cosyvoice3_runner_builds_prefill_embedding_slice_after_prefix() -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    runner.model = torch.nn.Linear(3, 3, bias=False)
    requests = [
        SimpleNamespace(
            data=SimpleNamespace(
                req=SimpleNamespace(
                    extend_range=SimpleNamespace(length=2), prefix_indices=[99]
                ),
                prompt_input_embeds=torch.arange(12, dtype=torch.float32).reshape(3, 4),
            )
        )
    ]
    forward_batch = SimpleNamespace(input_ids=torch.zeros(2, dtype=torch.long))

    result = runner._build_prefill_input_embeds(forward_batch, requests)

    assert torch.equal(
        result, torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]], dtype=torch.float32)
    )
