# SPDX-License-Identifier: Apache-2.0
"""Reduced-vocabulary DFlash drafts loaded from speculators checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import ReplayableGraph
from sglang_omni.speculative.dflash_draft import (
    DRAFT_ARCHITECTURE,
    DraftBlockSampler,
    ReducedVocabularyDFlashDraftModel,
    ReducedVocabularyDFlashWorker,
    materialize_draft_checkpoint,
)

SINGLE_FILE_WEIGHTS = ("model.safetensors",)
SHARDED_WEIGHTS = (
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
)


def write_speculators_checkpoint(
    directory: Path,
    *,
    weight_files: tuple[str, ...] = SINGLE_FILE_WEIGHTS,
    **config_fields: object,
) -> None:
    config = {
        "speculators_model_type": "dflash",
        "sample_from_anchor": False,
        "sliding_window_non_causal": False,
        "dtype": "bfloat16",
        "draft_vocab_size": 48000,
        "aux_hidden_state_layer_ids": [2, 14, 25],
        "mask_token_id": 151705,
        "block_size": 8,
        "transformer_layer_config": {
            "hidden_size": 1024,
            "vocab_size": 151936,
            "sliding_window": 131072,
            "tie_word_embeddings": True,
        },
        **config_fields,
    }
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "README.md").touch()
    for weight_file in weight_files:
        (directory / weight_file).touch()


@pytest.mark.parametrize(
    "weight_files",
    [
        pytest.param(SINGLE_FILE_WEIGHTS, id="single-file"),
        pytest.param(SHARDED_WEIGHTS, id="sharded"),
    ],
)
@pytest.mark.parametrize("non_causal", [False, True])
def test_flat_config_captures_before_the_layers_the_draft_was_trained_on(
    tmp_path: Path, weight_files: tuple[str, ...], non_causal: bool
) -> None:
    """Passing the aux ids straight through feeds the draft one layer too late, and a
    sharded draft loses its weights unless every shard is linked."""
    write_speculators_checkpoint(
        tmp_path, weight_files=weight_files, sliding_window_non_causal=non_causal
    )

    draft_checkpoint = materialize_draft_checkpoint(
        str(tmp_path), None, allow_non_causal=non_causal
    )
    draft_dir = Path(draft_checkpoint.name)

    config = json.loads((draft_dir / "config.json").read_text())
    assert config["architectures"] == [DRAFT_ARCHITECTURE]
    assert config["dflash_config"] == {
        "target_layer_ids": [1, 13, 24],
        "mask_token_id": 151705,
        "block_size": 8,
        "sample_from_anchor": False,
    }
    assert config.get("is_causal", True) is not non_causal
    assert config["sliding_window"] == 131072
    assert config["draft_vocab_size"] == 48000
    assert config["tie_word_embeddings"] is False
    assert sorted(path.name for path in draft_dir.iterdir()) == sorted(
        ("config.json", *weight_files)
    )
    for weight_file in weight_files:
        assert (draft_dir / weight_file).resolve() == (tmp_path / weight_file).resolve()


@pytest.mark.parametrize(
    "config_fields",
    [
        {"speculators_model_type": "eagle3"},
        {"sample_from_anchor": True},
        {"sliding_window_non_causal": True},
    ],
)
def test_checkpoint_the_draft_cannot_reproduce_is_refused(
    tmp_path: Path, config_fields: dict[str, object]
) -> None:
    """Each of these loads without error and then proposes tokens it was not trained for."""
    write_speculators_checkpoint(tmp_path, **config_fields)

    with pytest.raises(ValueError):
        materialize_draft_checkpoint(str(tmp_path), None)


def test_checkpoint_without_anchor_sampling_setting(tmp_path: Path) -> None:
    """Early DFlash checkpoints omit this field and never sample the anchor row."""
    write_speculators_checkpoint(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config.pop("sample_from_anchor")
    config_path.write_text(json.dumps(config))

    with materialize_draft_checkpoint(str(tmp_path), None) as draft_directory:
        flattened = json.loads((Path(draft_directory) / "config.json").read_text())
        assert flattened["dflash_config"]["sample_from_anchor"] is False


def make_draft_model() -> ReducedVocabularyDFlashDraftModel:
    draft_model = ReducedVocabularyDFlashDraftModel.__new__(
        ReducedVocabularyDFlashDraftModel
    )
    nn.Module.__init__(draft_model)
    draft_model.projector_type = None
    draft_model.is_nemotron_35_draft = False
    draft_model.fc = nn.Linear(2, 2, bias=False)
    draft_model.lm_head = nn.Linear(2, 3, bias=False)
    draft_model.register_buffer(
        "draft_to_target_token_ids", torch.empty(3, dtype=torch.long)
    )
    draft_model.load_weights(
        [
            ("fc.weight", torch.tensor([[2.0, 3.0], [5.0, 7.0]])),
            ("lm_head.weight", torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])),
            ("d2t", torch.tensor([2, 4, 7])),
            ("t2d", torch.zeros(12, dtype=torch.bool)),
        ]
    )
    return draft_model


def test_captured_and_eager_proposals_map_the_draft_head_to_target_tokens() -> None:
    """Using the target head, or a draft row as a target id, proposes the wrong token."""
    draft_model = make_draft_model()
    assert torch.equal(draft_model.fc.weight, torch.tensor([[2.0, 3.0], [5.0, 7.0]]))
    worker = ReducedVocabularyDFlashWorker.__new__(ReducedVocabularyDFlashWorker)
    worker.draft_model = draft_model
    # One three-token block: the anchor row, then rows picking draft ids 0 and 1.
    block_hidden_states = torch.tensor([[-1.0, -1.0], [2.0, 0.0], [0.0, 2.0]])
    zero_target_head = nn.Linear(2, 12, bias=False)
    nn.init.zeros_(zero_target_head.weight)

    eager_token_ids = worker._greedy_sample_from_vocab_parallel_head(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        hidden_states=block_hidden_states[1:], lm_head=zero_target_head
    )
    sampler = DraftBlockSampler(draft_model, block_size=3, max_batch_size=2)
    sampler(block_hidden_states)

    assert eager_token_ids.tolist() == [2, 5]
    assert sampler.out[:2].tolist() == [2, 5]


def test_eager_sampler_setting_keeps_the_head_out_of_the_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The setting would otherwise have no effect on this worker."""
    worker = ReducedVocabularyDFlashWorker.__new__(ReducedVocabularyDFlashWorker)
    worker.block_size = 8
    monkeypatch.setenv("SGLANG_DFLASH_EAGER_DRAFT_SAMPLER", "1")

    assert (
        worker._maybe_build_draft_sampler()  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        is None
    )


def test_draft_sampler_reuses_storage_across_changed_inputs_and_batch_sizes() -> None:
    draft_model = make_draft_model()
    sampler = DraftBlockSampler(draft_model, block_size=3, max_batch_size=2)
    output_address = sampler.out.data_ptr()
    for batch_size, proposal, target_token in [
        (2, [2.0, 0.0], 2),
        (1, [0.0, 2.0], 5),
        (2, [-2.0, -2.0], 9),
    ]:
        hidden = torch.tensor([[9.0, 0.0], proposal, proposal] * batch_size)
        sampler(hidden)
        assert sampler.out[: batch_size * 2].tolist() == [target_token] * (
            batch_size * 2
        )
        assert sampler.out.data_ptr() == output_address


@pytest.mark.accelerator
def test_draft_sampler_graph_replay_updates_mapped_tokens_in_shared_storage() -> None:
    """Replayed proposals must use new hidden states and retain the target-id mapping."""
    # Graph capability is independent of the narrower audio staging policy.
    device = torch.device(current_platform.device_type)
    backend = current_platform.get_device_graph_backend(device)
    if backend is None:
        pytest.skip("The selected accelerator has no graph backend")
    device_module = torch.get_device_module(device)
    device = torch.device(device.type, device_module.current_device())
    draft_model = make_draft_model().to(device=device, dtype=torch.bfloat16)
    sampler = DraftBlockSampler(draft_model, block_size=3, max_batch_size=2)
    output_address = sampler.out.data_ptr()
    inputs = {
        batch_size: torch.zeros(batch_size * 3, 2, device=device, dtype=torch.bfloat16)
        for batch_size in (1, 2)
    }
    graphs: dict[int, ReplayableGraph] = {}
    pool = backend.graph_pool_handle()
    stream = device_module.Stream(device=device)
    stream.wait_stream(device_module.current_stream(device))
    try:
        with torch.no_grad(), device_module.stream(stream):
            for batch_size, hidden in inputs.items():
                for _ in range(2):
                    sampler(hidden)
                with backend.capture(pool=pool, stream=stream) as graph:
                    sampler(hidden)
                graphs[batch_size] = graph
        device_module.current_stream(device).wait_stream(stream)
        for batch_size, proposal, target_token in [
            (2, [2.0, 0.0], 2),
            (1, [0.0, 2.0], 5),
            (2, [-2.0, -2.0], 9),
        ]:
            hidden = inputs[batch_size]
            hidden.copy_(
                torch.tensor(
                    [[9.0, 0.0], proposal, proposal] * batch_size,
                    device=device,
                    dtype=hidden.dtype,
                )
            )
            graphs[batch_size].replay()
            device_module.synchronize(device)
            assert sampler.out[: batch_size * 2].tolist() == [target_token] * (
                batch_size * 2
            )
            assert sampler.out.data_ptr() == output_address
    finally:
        device_module.synchronize(device)
        for graph in graphs.values():
            graph.reset()
