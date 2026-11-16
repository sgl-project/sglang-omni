# SPDX-License-Identifier: Apache-2.0
"""The ELLSA 2 model path: features, the generator's block layout, the RVQ decoders and the motion rows they give."""

import numpy as np
import torch
import torchaudio
from transformers import LlamaConfig, Qwen2Config

from sglang_omni.models.ellsa2.batching import BatchedBlocks, BlockRequest
from sglang_omni.models.ellsa2.fbank import NUM_MEL_BINS, LogMelFilterbank, mel_banks
from sglang_omni.models.ellsa2.formats import FEATURES_PER_FRAME, POSE_FEATURES
from sglang_omni.models.ellsa2.generator import (
    BLOCK_SAMPLES,
    BLOCK_STEPS,
    MAX_HISTORY_BLOCKS,
    NUM_CODEBOOKS,
    Ellsa2Generator,
    HistoryBlock,
    PromptTokens,
    retained_history,
)
from sglang_omni.models.ellsa2.motion import (
    UPPER_JOINTS,
    MotionDecoder,
    Normalization,
    Tokenizers,
)
from sglang_omni.models.ellsa2.rvq import (
    CODEBOOK_SIZE,
    RVQVAE,
    FlameTokenizer,
    ResidualQuantizer,
    StreamingDecoder,
)

AUDIO_PER_BLOCK = 10
FEATURE_FRAMES_PER_BLOCK = 80


def tiny_generator() -> Ellsa2Generator:
    torch.manual_seed(0)
    backbone = Qwen2Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=16,
        max_position_embeddings=512,
    )
    encoder = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        vocab_size=1,
        max_position_embeddings=512,
        hidden_act="gelu",
        attn_implementation="sdpa",
    )
    generator = Ellsa2Generator(backbone, encoder)
    generator.restore_rope_precision()
    return generator.eval()


def random_codebooks(model: torch.nn.Module) -> None:
    for name, buffer in model.named_buffers():
        if name.endswith("codebook"):
            buffer.normal_()
        else:
            pass


def test_the_filterbank_gives_one_frame_per_10_ms_on_the_kaldi_mel_scale():
    filterbank = LogMelFilterbank()
    for samples in (12800, 3 * 12800, 1000):
        assert filterbank(torch.randn(samples)).shape == (
            (samples + 80) // 160,
            NUM_MEL_BINS,
        )
    kaldi, _ = torchaudio.compliance.kaldi.get_mel_banks(
        NUM_MEL_BINS, 512, 16000.0, 20.0, -400.0, 100.0, -500.0, 1.0
    )
    assert torch.allclose(mel_banks()[:-1].T, kaldi, atol=1e-4)


def test_a_block_reads_the_style_then_each_block_s_audio_and_the_motion_of_completed_blocks():
    generator = tiny_generator()
    prompt = PromptTokens(head=torch.tensor([1, 2, 3]), audio_start=4, audio_end=5)
    target = torch.randn(1, 2 * AUDIO_PER_BLOCK, 64)
    partner = torch.randn(1, 2 * AUDIO_PER_BLOCK, 64)
    known = torch.randint(0, CODEBOOK_SIZE, (12 + BLOCK_STEPS, NUM_CODEBOOKS))
    with torch.no_grad():
        embeds = generator.block_inputs(prompt, target, partner, known, 12, 2)
        motion = generator.motion_embeddings(known.unsqueeze(0))[0]
    block = 2 * (AUDIO_PER_BLOCK + 2)
    assert embeds.shape == (1, 3 + 12 + 2 * block + BLOCK_STEPS, 64)
    assert torch.equal(embeds[0, 3:15], motion[:12])
    assert torch.equal(embeds[0, 16:26], target[0, :AUDIO_PER_BLOCK])
    assert torch.equal(embeds[0, 28:38], partner[0, :AUDIO_PER_BLOCK])
    assert torch.equal(embeds[0, 15 + block : 15 + block + BLOCK_STEPS], motion[12:])
    assert torch.equal(embeds[0, -11:-1], partner[0, AUDIO_PER_BLOCK:])


def test_a_negative_code_adds_nothing_to_the_motion_embedding():
    generator = tiny_generator()
    codes = torch.randint(0, CODEBOOK_SIZE, (1, 1, NUM_CODEBOOKS))
    masked = codes.clone()
    masked[..., 0] = -100
    with torch.no_grad():
        full = generator.motion_embeddings(codes)
        partial = generator.motion_embeddings(masked)
        first = (
            generator.motion_embeds[0](codes[..., 0])
            * generator.motion_embed_weights[0]
        )
    assert torch.allclose(full - partial, first, atol=1e-6)


def test_a_block_is_six_greedy_steps_of_eighteen_codes():
    generator = tiny_generator()
    features = torch.randn(1, 2 * FEATURE_FRAMES_PER_BLOCK, NUM_MEL_BINS)
    prompt = PromptTokens(head=torch.tensor([1, 2, 3]), audio_start=4, audio_end=5)
    known = torch.randint(0, CODEBOOK_SIZE, (12 + BLOCK_STEPS, NUM_CODEBOOKS))
    first = generator.generate_block(prompt, features, features, known, 12, 2)
    again = generator.generate_block(prompt, features, features, known, 12, 2)
    assert first.shape == (BLOCK_STEPS, NUM_CODEBOOKS)
    assert torch.equal(first, again)
    assert 0 <= int(first.min()) and int(first.max()) < CODEBOOK_SIZE


def test_sessions_generated_together_from_their_kept_caches_get_the_codes_each_gets_alone():
    generator = tiny_generator().double()
    filterbank = LogMelFilterbank()
    prompt = PromptTokens(head=torch.tensor([1, 2, 3]), audio_start=4, audio_end=5)
    style = torch.randint(0, CODEBOOK_SIZE, (12, NUM_CODEBOOKS))
    batched = BatchedBlocks(generator, filterbank, max_batch=4, cache_positions=512)
    blocks = MAX_HISTORY_BLOCKS + 3
    # Sessions start a block apart, so a batch mixes Sessions starting over (their first block, or past twelve seconds) with Sessions reusing their caches.
    starts = (0, 1, 3)
    audio = [
        (
            torch.randn(blocks * BLOCK_SAMPLES).numpy() * 0.1,
            torch.randn(blocks * BLOCK_SAMPLES).numpy() * 0.1,
        )
        for _ in starts
    ]
    together = [[] for _ in starts]
    alone = [[] for _ in starts]
    caches = [batched.acquire(set()) for _ in starts]
    for round_ in range(blocks + max(starts)):
        requests, rows = [], []
        for row, start in enumerate(starts):
            block = round_ - start
            if not 0 <= block < blocks:
                continue
            else:
                rows.append(row)
            history = retained_history(alone[row])
            first = block - len(history)
            window = slice(first * BLOCK_SAMPLES, (block + 1) * BLOCK_SAMPLES)
            target, partner = (
                stream[window].astype(np.float32) for stream in audio[row]
            )
            known = torch.cat([style] + [codes for codes in history])
            requests.append(
                BlockRequest(
                    target_audio=np.split(target, len(history) + 1),
                    partner_audio=np.split(partner, len(history) + 1),
                    known_codes=known,
                    style_steps=12,
                    num_blocks=len(history) + 1,
                    cache=caches[row],
                )
            )
            alone[row].append(
                generator.generate_block(
                    prompt,
                    filterbank(torch.from_numpy(target))[None].double(),
                    filterbank(torch.from_numpy(partner))[None].double(),
                    known,
                    12,
                    len(history) + 1,
                )
            )
        for row, codes in zip(rows, batched.generate(prompt, requests)):
            together[row].append(codes)
    for row in range(len(starts)):
        assert len(together[row]) == blocks
        assert all(torch.equal(a, b) for a, b in zip(together[row], alone[row]))


def test_decoding_sessions_together_equals_decoding_each_alone():
    torch.manual_seed(0)
    model = RVQVAE(9, 2, 40).eval()
    random_codebooks(model)
    codes = torch.randint(0, CODEBOOK_SIZE, (3, 3 * BLOCK_STEPS, 2))
    alone = [StreamingDecoder(model) for _ in range(3)]
    together = [StreamingDecoder(model) for _ in range(3)]
    with torch.no_grad():
        # The third Session starts a block late, so its kept latents are shorter for a block.
        for start in range(0, codes.shape[1], BLOCK_STEPS):
            block = slice(start, start + BLOCK_STEPS)
            rows = [0, 1] if start == 0 else [0, 1, 2]
            joint = StreamingDecoder.decode_together(
                [together[row] for row in rows], codes[rows, block]
            )
            for index, row in enumerate(rows):
                assert torch.allclose(
                    joint[index],
                    alone[row].decode(codes[row : row + 1, block])[0],
                    atol=1e-5,
                )


def test_past_twelve_seconds_the_history_keeps_the_last_three_blocks():
    blocks = [
        HistoryBlock(target_audio=np.zeros(1), partner_audio=np.zeros(1))
        for _ in range(MAX_HISTORY_BLOCKS)
    ]
    assert retained_history(blocks[:-1]) == blocks[:-1]
    assert retained_history(blocks) == blocks[-3:]


def test_the_residual_quantizer_takes_the_nearest_code_of_each_level():
    quantizer = ResidualQuantizer(2)
    random_codebooks(quantizer)
    quantizer.layers[1].codebook.mul_(0.01)
    latents = (quantizer.layers[0].codebook[5] + quantizer.layers[1].codebook[9]).view(
        1, -1, 1
    )
    assert quantizer.quantize(latents).tolist() == [[[5, 9]]]


def test_decoding_block_by_block_equals_decoding_all_blocks_at_once():
    torch.manual_seed(0)
    model = RVQVAE(9, 2, 40).eval()
    random_codebooks(model)
    codes = torch.randint(0, CODEBOOK_SIZE, (1, 4 * BLOCK_STEPS, 2))
    blockwise, whole = StreamingDecoder(model), StreamingDecoder(model)
    with torch.no_grad():
        streamed = torch.cat(
            [
                blockwise.decode(codes[:, start : start + BLOCK_STEPS])
                for start in range(0, codes.shape[1], BLOCK_STEPS)
            ],
            dim=1,
        )
        at_once = whole.decode(codes)
    assert streamed.shape == (1, 4 * BLOCK_STEPS * 4, 9)
    assert torch.allclose(streamed, at_once, atol=1e-5)


def test_decoded_rows_move_the_upper_body_and_hands_and_keep_root_and_legs_at_rest():
    torch.manual_seed(0)
    flame = FlameTokenizer(
        torch.linalg.qr(torch.randn(100, 8))[0], torch.zeros(100, 1), 4, 2, 240
    )
    tokenizers = Tokenizers(
        upper=RVQVAE(78, 6, 300), hands=RVQVAE(180, 6, 300), flame=flame
    )
    for model in (tokenizers.upper, tokenizers.hands, tokenizers.flame):
        random_codebooks(model.eval())
    normalization = Normalization.release()
    rows = MotionDecoder(tokenizers, normalization).decode(
        torch.randint(0, CODEBOOK_SIZE, (BLOCK_STEPS, NUM_CODEBOOKS))
    )
    pose = rows[:, :POSE_FEATURES].reshape(-1, 52, 3)
    legs = [joint + 1 for joint in range(21) if joint not in UPPER_JOINTS]
    assert (
        rows.shape == (BLOCK_STEPS * 4, FEATURES_PER_FRAME) and rows.dtype == np.float32
    )
    assert not pose[:, 0].any() and not pose[:, legs].any()
    assert pose[:, [joint + 1 for joint in UPPER_JOINTS]].all(axis=-1).any()
    assert pose[:, 22:].any()
    assert np.allclose(
        rows[:, POSE_FEATURES + 100 : POSE_FEATURES + 103],
        normalization.flame_mean[100:103],
    )
