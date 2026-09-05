# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sglang")

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch

from sglang_omni.model_runner import speculative_target_worker as target_module


def test_native_target_prefill_samples_and_verify_preserves_logits(monkeypatch):
    seen = []
    logits = SimpleNamespace(hidden_states=torch.ones(2, 3))
    next_ids = torch.tensor([7])
    forward_batch = SimpleNamespace(
        is_prefill_only=False,
        apply_deprecated_skip_attn_backend_init=lambda value: seen.append(
            ("skip", value)
        ),
    )
    runner = SimpleNamespace(
        ps=object(),
        forward=lambda batch, **kwargs: SimpleNamespace(
            logits_output=logits,
            can_run_graph=True,
            expert_distribution_metrics="metrics",
            routed_experts_output="experts",
            indexer_topk_output="indexer",
        ),
        sample=lambda output, batch: seen.append(("sample", output, batch)) or next_ids,
    )
    worker = SimpleNamespace(
        model_runner=runner,
        dllm_algorithm=None,
        server_args=SimpleNamespace(
            tokenizer_path="local-tokenizer",
            model_path="local-model",
            tokenizer_mode="auto",
            trust_remote_code=True,
            revision=None,
            tokenizer_backend="huggingface",
        ),
    )
    monkeypatch.setattr(
        target_module, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)
    )
    monkeypatch.setattr(
        target_module, "get_tokenizer", lambda *args, **kwargs: "tokenizer"
    )

    def build_forward(batch, model_runner, **kwargs):
        assert model_runner is runner
        assert kwargs["capture_hidden_mode"] == CaptureHiddenMode.FULL
        assert kwargs["return_hidden_states_before_norm"] is False
        seen.append(("build", batch))
        return forward_batch

    monkeypatch.setattr(ForwardBatch, "init_new", build_forward)
    adapter = target_module.SpeculativeTargetWorker(worker)
    batch = SimpleNamespace(hicache_consumer_index=0)
    prefill = adapter.forward_batch_generation(
        batch, capture_hidden_mode=CaptureHiddenMode.FULL
    )
    assert prefill.next_token_ids is next_ids
    assert prefill.logits_output is logits
    assert prefill.routed_experts_output == "experts"
    assert prefill.indexer_topk_output == "indexer"
    assert adapter.model_runner is worker.model_runner

    verified = adapter.forward_batch_generation(
        batch=None,
        forward_batch=forward_batch,
        is_verify=True,
        skip_attn_backend_init=True,
    )
    assert verified.logits_output is logits
    assert verified.next_token_ids is None
    assert verified.can_run_cuda_graph
    assert [event[0] for event in seen] == ["build", "skip", "sample", "skip"]
    assert seen[-1] == ("skip", True)
