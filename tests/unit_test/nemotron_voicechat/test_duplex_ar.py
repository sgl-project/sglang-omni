# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.nemotron_voicechat import duplex_ar
from sglang_omni.models.nemotron_voicechat.fusion import AddFusion
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity


def test_prefill_uses_exact_uncached_suffix_and_replays_fused_history(monkeypatch):
    captured = []
    monkeypatch.setattr(
        duplex_ar,
        "attach_omni_prefill_inputs",
        lambda batch, inputs: captured.append(inputs.input_embeds),
    )
    history = duplex_ar.FrameHistory(
        rows=[torch.arange(12).reshape(3, 4), torch.full((1, 4), 99)], positions=4
    )
    requests = [
        SimpleNamespace(
            data=SimpleNamespace(talker_model_inputs={"duplex_history": history})
        )
    ]
    for prefix in (0, 2, 3):
        duplex_ar.attach_rows(
            SimpleNamespace(
                extend_prefix_lens_cpu=[prefix], extend_seq_lens_cpu=[4 - prefix]
            ),
            requests,
        )
        assert torch.equal(captured[-1], torch.cat(history.rows)[prefix:])
    with pytest.raises(RuntimeError, match="not aligned"):
        duplex_ar.attach_rows(
            SimpleNamespace(extend_prefix_lens_cpu=[2], extend_seq_lens_cpu=[1]),
            requests,
        )


def test_thinker_continuation_fuses_prior_output_and_function_without_new_token(
    monkeypatch,
):
    emb = torch.nn.Embedding.from_pretrained(torch.arange(40).float().reshape(10, 4))
    model = SimpleNamespace(
        llm=SimpleNamespace(
            get_input_embeddings=lambda: emb, config=SimpleNamespace(vocab_size=10)
        ),
        fusion=AddFusion(
            {
                "duplex_user_channel_weight": 2,
                "duplex_text_channel_weight": 3,
                "duplex_function_channel_weight": 5,
            }
        ),
    )
    tokenizer = SimpleNamespace(
        all_special_ids=[0],
        convert_tokens_to_ids=lambda _: 0,
        decode=lambda ids: "x" * len(ids),
    )
    adapter = duplex_ar.ThinkerAdapter(
        SimpleNamespace(model=model),
        prompt_ids=[1, 2],
        pad_id=0,
        tokenizer=tokenizer,
        context_length=8,
    )
    ref = SessionIdentity("s")
    adapter.open(ref, OmniRequest(None))

    def request(payload, **kwargs):
        return SimpleNamespace(stage_payload=payload, talker_model_inputs={}, **kwargs)

    monkeypatch.setattr(duplex_ar, "ar_request", request)
    first = adapter.build(
        ref, None, StagePayload("1", OmniRequest(None), {"acoustic": torch.ones(1, 4)})
    )
    assert first.input_ids == [1, 2, 0]
    first.output_ids = [3]
    first.extra_model_outputs = {"function_ids": [4]}
    adapter.result(ref, first)
    second = adapter.build(
        ref,
        None,
        StagePayload("2", OmniRequest(None), {"acoustic": torch.full((1, 4), 7)}),
    )
    assert second.input_ids == [] and second.max_new_tokens == 1
    history = adapter.states[ref]
    assert history.positions == 4
    assert torch.equal(
        history.rows[-1],
        2 * torch.full((1, 4), 7) + 3 * emb.weight[3] + 5 * emb.weight[4],
    )
    adapter.close(ref)
    assert not adapter.states
