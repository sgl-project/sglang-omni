# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.minimax_music3 import acoustic
from sglang_omni.models.minimax_music3.dit import (
    ContinuousTransformer,
    MiniMaxMusic3DIT,
    RotaryEmbedding,
)
from sglang_omni.models.minimax_music3.model_runner import (
    MiniMaxMusic3ModelRunner,
    _ARState,
)


class SmallDIT(MiniMaxMusic3DIT):
    def __init__(self, *, compute_dtype, attention_backend):
        torch.nn.Module.__init__(self)
        self.diffusion_transformer = torch.nn.Module()
        transformer = ContinuousTransformer.__new__(ContinuousTransformer)
        torch.nn.Module.__init__(transformer)
        transformer.rotary_pos_emb = RotaryEmbedding(32)
        transformer.project_in = torch.nn.Linear(4, 4, bias=False)
        transformer._rotary_cache = {}
        self.diffusion_transformer.transformer = transformer
        self.sr_input = 24000
        self.sr_output = 44100
        self.hop_size_input = 960
        self.hop_size_output = 512

    def enable_compiled_blocks(self, *, warmup_mel_length):
        transformer = self.diffusion_transformer.transformer
        weight = transformer.project_in.weight
        self.warmup_tables = transformer._rotary_cos_sin(
            warmup_mel_length + 1, dtype=weight.dtype, device=weight.device
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_loaded_rotary_preserves_fp32_phases_before_warmup(
    tmp_path, monkeypatch, dtype
):
    original = SmallDIT(compute_dtype=torch.float32, attention_backend="torch_sdpa")
    inverse = original.diffusion_transformer.transformer.rotary_pos_emb.inv_freq.clone()
    checkpoint = tmp_path / "dit.pth"
    torch.save(original.state_dict(), checkpoint)
    monkeypatch.setattr(acoustic, "MiniMaxMusic3DIT", SmallDIT)
    decoder = acoustic.MiniMaxMusic3AcousticDecoder.__new__(
        acoustic.MiniMaxMusic3AcousticDecoder
    )
    decoder.device = torch.device("cpu")
    decoder.dtype = dtype
    decoder.attention_backend = "torch_sdpa"
    decoder.compile_acoustic = True
    decoder.cache_dit = False
    decoder.breakable_cuda_graph_requested = False

    decoder._build_dit(
        str(checkpoint),
        cache_dit_fn_compute_blocks=1,
        cache_dit_bn_compute_blocks=1,
        cache_dit_max_warmup_steps=1,
        cache_dit_residual_diff_threshold=0.08,
        cache_dit_max_continuous_cached_steps=1,
    )

    transformer = decoder.dit.diffusion_transformer.transformer
    torch.testing.assert_close(
        transformer.rotary_pos_emb.inv_freq, inverse, rtol=0, atol=0
    )
    assert transformer.project_in.weight.dtype == dtype
    for length in (690, 345):
        phases = torch.arange(length, dtype=torch.float32)[:, None] * inverse[None, :]
        phases = torch.cat((phases, phases), dim=-1)
        expected = (phases.cos().to(dtype), phases.sin().to(dtype))
        actual = transformer._rotary_cos_sin(length, dtype=dtype, device=decoder.device)
        for value, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        assert (
            transformer._rotary_cos_sin(length, dtype=dtype, device=decoder.device)
            is actual
        )
        if length == 690:
            assert decoder.dit.warmup_tables is actual


def make_requests(state, prompt_ids, outputs):
    requests = []
    for index, ids in enumerate(prompt_ids):
        data = SimpleNamespace(
            ar_state=state,
            prompt_token_ids=torch.tensor(ids),
            req=SimpleNamespace(
                output_ids=list(outputs),
                prefix_indices=[],
                extend_range=SimpleNamespace(length=len(ids) + len(outputs)),
            ),
        )
        requests.append(SimpleNamespace(request_id=f"row{index}", data=data))
    requests[0].data.cfg_uncond = requests[1].data
    return requests


def test_replay_reconstructs_every_codebook_for_both_prompt_rows():
    torch.manual_seed(23)
    tokens = torch.nn.Embedding(80, 4)
    extra = torch.nn.Embedding(7 * 16, 4)
    model = SimpleNamespace(
        get_input_embeddings=lambda: tokens,
        audio_embeddings=extra,
        audio_embedding_offsets=torch.arange(7) * 16,
        mel_token_offset=32,
        frame_embedding_scale=8**-0.5,
    )
    codes = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [9, 8, 7, 6, 5, 4, 3, 2]])
    state = _ARState(sampling_seed=7, seed=7, decode_limit=50)
    state.code_history = list(codes.unbind())
    state.sampling_position = 16
    state.generated_frames = 1
    requests = make_requests(state, [[1, 2, 3], [1, 5, 3]], [33, 41])
    runner = MiniMaxMusic3ModelRunner.__new__(MiniMaxMusic3ModelRunner)
    runner.model = model
    batch = SimpleNamespace(input_ids=torch.zeros(10, dtype=torch.long))

    runner.before_prefill(batch, None, requests)

    feedback = torch.stack(
        [
            (
                tokens.weight[32 + int(frame[0])]
                + sum(
                    extra.weight[16 * book + int(frame[book + 1])] for book in range(7)
                )
            )
            * 8**-0.5
            for frame in codes
        ]
    )
    expected = torch.cat(
        [torch.cat((tokens(req.data.prompt_token_ids), feedback)) for req in requests]
    )
    torch.testing.assert_close(batch.input_embeds, expected, rtol=1e-6, atol=1e-6)
    assert requests[0].data.ar_state is state
    assert state.sampling_position == 16
    assert state.generated_frames == 1
    torch.testing.assert_close(torch.stack(state.code_history), codes)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_replay_prefill_emits_one_new_frame_and_eos_emits_none():
    device = torch.device("cuda")
    state = _ARState(sampling_seed=7, seed=7, decode_limit=50)
    requests = make_requests(state, [[1], [1]], [])

    def depth_graph(hidden, c0, seeds, positions, forced, replay):
        codes = c0[:, None] + torch.arange(8, device=device)[None, :]
        return codes, hidden.repeat(1, 7), torch.zeros_like(codes[:, 1:])

    runner = MiniMaxMusic3ModelRunner.__new__(MiniMaxMusic3ModelRunner)
    runner.model = SimpleNamespace(
        num_codebooks=8,
        c0_logit_ids=torch.arange(64, device=device),
        rvq_depth_graph=depth_graph,
    )
    runner._forced_codes_dir = None
    logits = torch.full((2, 64), -torch.inf, device=device)
    logits[:, 1] = 0
    result = SimpleNamespace(
        logits_output=SimpleNamespace(
            hidden_states=torch.ones((2, 4), device=device), next_token_logits=logits
        )
    )

    runner._advance(result, requests, emit=False)
    assert len(state.code_history) == 1 and state.generated_frames == 0
    runner._advance(result, requests, emit=True)
    assert len(state.code_history) == 2 and state.generated_frames == 1
    runner._advance(result, requests, emit=False)
    assert len(state.code_history) == 3 and state.generated_frames == 2
    assert state.frames.end_frame == 2

    logits[:, 1] = -torch.inf
    logits[:, 0] = 0
    runner._advance(result, requests, emit=False)
    assert state.finish_reason == "stop"
    assert len(state.code_history) == 3 and state.frames.end_frame == 2
    assert state.sampling_position == 32
    assert torch.equal(
        result.next_token_ids, torch.zeros(2, device=device, dtype=torch.long)
    )
