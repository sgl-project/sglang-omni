# Ordered multimodal frontend verification

## Frozen inputs and runtime

- Omni baseline: `b9aa02d411c27f1509711cfc90f43a3f6c384ac5`.
- MiniCPM-o-4_5: `503e754207c94da6bb26850b4469f367c9ea3582`.
- Qwen3-Omni: `26291f793822fb6be9555850f06dfe95f2d7e695`.
- Recommended image: `docker.io/hongccc/sglang-omni@sha256:ebe4239e29a764ee3a2806385c061c5fd438a26f01458e503d3822dcba5790df`.
- Omni runtime: SGLang 0.5.21, Transformers 5.12.1, Torch 2.13 CUDA 13.0.
- MiniCPM reference: Transformers 4.51.0, Torch 2.8, librosa 0.11.0, minicpmo-utils 1.0.5. Qwen reference uses the Omni environment.
- The image's initial SGLang 0.5.19 lacked SpawnRanks; it was replaced with 0.5.21. MiniCPM reference runs in a separate environment. Video final verification fixes librosa 0.11.0 and Decord 0.6.0 on both sides.
- Decode: greedy, 256 new tokens, repetition penalty 1, text output. MiniCPM thinking remains disabled; requests containing audio retain the existing TTS template. Qwen uses its existing template default.

Fixture manifests retain sample IDs, source revisions, media SHA256, messages and references. Local manifests are `parity-fixtures/manifest.json` (350 samples) and `video-parity-fixtures/manifest.json` (50 questions across 17 videos). Raw results, punctuation, prompt IDs and bounds remain in `benchmarks/results/input-parity/`.

| Dataset | Revision | Count |
| --- | --- | ---: |
| MMMU CI50 | ff72fd69cc7e0719e04a0ddb12d160de89c6fefe | 50 |
| MMSU CI2000 | 5ae6ed4343a89566be6dd90023d529f1bd97802d | 100 |
| SeedTTS EN/ZH reference audio | 81d1901582dee1293a537a6d945d084301712c41 | 100 each |
| Video_MME_ci | 833bd815c628ff277911bea3b1563545b21d5e27 | 50 |

## First differing boundary

API schema and client extraction retain content dictionaries and message order. The first difference is model-local preprocessing:

- MiniCPM filters inline non-text parts and joins text parts without the official newline separator. Bulk top-level placement cannot represent inline positions or earlier-turn ownership.
- Qwen's shared normalization JSON-encodes structured content; the local processor therefore receives text instead of media placeholders.
- MiniCPM video previously grouped all visual and audio inputs, losing frame/audio alternation. Decoder choice also changed raw pixels and samples. The final video frontend uses Decord frames and the reference audio decode priority.
- Qwen embedded video tracks previously preceded explicit audio features even when inline audio placeholders came first. Features now follow placeholder traversal.

Inline plus top-level input media is explicitly rejected; TTS reference audio is independent. Top-level media remains supported. Unsupported MiniCPM stack_frames values fail explicitly. No shared normalization or thinking/TTS defaults were changed.

## Input comparisons

The eight fixtures cover text, text parts, image, image/text alternation, image turns, audio, multiple audio parts and audio turns. MiniCPM and Qwen after prompts and IDs equal the reference for all eight. MiniCPM image/audio bounds and audio features also equal the reference, including the separate Transformers 4.51 run.

The synthetic timeline and real Video-MME videos were compared using official native frame/audio content against inline video input. For the real Christmas video, prompt, IDs, visual/audio bounds, pixels and audio features all match after fixing decoder dependencies. The video pipeline has explicit CPU tests for text surrounding video, frame/audio alternation and tail intervals. Qwen tests use the real processor to verify two embedded video tracks with explicit audio before or after them.

## Correctness results

Percentages below use strict option extraction; absent final options are failures. ASR uses NFKC, lowercase and removal of Unicode punctuation/symbols; English whitespace is collapsed, Chinese whitespace removed. Raw generated punctuation is preserved separately. All final generation groups have zero request errors.

| Model/task | Official | Before | After |
| --- | ---: | ---: | ---: |
| MiniCPM MMMU accuracy | 22% | 14% | 20% |
| MiniCPM MMSU accuracy | 69% | 42% | 68% |
| MiniCPM EN raw WER | 0.6865% | 100% | 44.9275% |
| MiniCPM EN extracted transcript WER | 0.6865% | 100% | 3.8902% |
| MiniCPM ZH CER | 0.5405% | 99.4595% | 0.8108% |
| Qwen MMMU accuracy | 36% | 40% | 36% |
| Qwen MMSU accuracy | 66% | 34% | 68% |
| Qwen EN WER | 0.9153% | 97.4828% | 0.6102% |
| Qwen ZH CER | 0.1351% | 128.3784% | 0.1351% |
| MiniCPM audio/video accuracy | 74% | 44% | 72% |

MiniCPM produces an extra `|||` JSON metrics suffix on 45 English ASR samples. The extracted metric removes that suffix only; raw WER still penalizes it. This behavior remains unresolved. A frozen-weight audio encoder probe confirms equal mel inputs and lengths but bf16 output differences (mean absolute error 0.001486, maximum 0.03125 on asr_en-0064). Replacing its mask with the official mask did not change that probe's error. This identifies an execution difference but does not prove the cause of the generation suffix.

MMMU often reaches the 256-token limit before an option: MiniCPM official/after have 39 missing answers each; Qwen has 25/26. The Qwen before score uses JSON-encoded image descriptions and a different answering pattern, so its apparent 4-point drop cannot establish visual quality regression. Longer-decoding evaluation is outside this fixed comparison.

Video official and final after outputs differ only on video-020-2: official D, after C, reference D. That video's prompt, IDs, pixels, audio features and both modality bounds also match exactly. Generation identity is not required. The one-question gap and MiniCPM ASR execution differences mean complete correctness parity is not yet established.

## Verification and reproduction

### PR review and refactor verification

The test review is recorded in bb354f2a and 8213b54c. The 36 frontend cases cover actual inline decoding, ordered parts, media turn ownership, supported top-level aliases, timed video sampling, pixel budgets and the audio decoder fallback. Duplicate layout assertions and processor keyword snapshots were removed. Tests were frozen after the second commit and remained byte-identical during production refactoring.

The reviewed production frontend uses typed video options, explicit unit names and reuses decoded inline media. Nine complete preprocessing snapshots (including video) and eight independently captured prompt/ID/bounds fixtures are identical before and after refactoring. The MiniCPM/pipeline/API regression run passed 1031 tests with 15 environment-dependent skips. The final frontend/API run and two enabled live transcription cases also pass. Full pre-commit checks pass.

The same frozen 350 generation cases and 50 video questions were rerun before and after refactoring. All metrics in the MiniCPM table remain exactly unchanged, with zero request errors. All 50 video outputs are identical. Six of the 350 text outputs differ: five MMMU explanations and one English comma; option accuracy, missing-answer counts, raw and extracted WER, and CER are unchanged.

This PR claims correctness improvements, not generation throughput or latency improvements. A warmed frontend profile with 31 measurements per fixture shows identical output snapshots and small timing variation: the final video median is 3.47% lower, image 1.99% lower, while other fixtures range from 0.54% lower to 7.57% higher (the latter is about 0.01 ms on text). These measurements do not establish a stable speedup or a material processing regression. Local evidence is stored under pr2635-before-refactor-*, pr2635-after-refactor-* and pr2635-profile-*.

The model frontend, Qwen pipeline and API suite passed 281 tests; two environment-gated live tests skipped. Enabled live inline/top-level audio transcription checks passed two tests. Full pre-commit validation includes Rust formatting. Large evidence files and media are intentionally outside tracked source.

```bash
python -m benchmarks.eval.minicpm_input_parity --family minicpm --backend official --model-path MODEL --output official-input.json
python -m benchmarks.eval.minicpm_input_parity --family minicpm --backend omni --model-path MODEL --output omni-input.json
python -m benchmarks.eval.omni_frontend_parity --prepare FIXTURES
python -m benchmarks.eval.omni_frontend_parity --prepare VIDEO_FIXTURES --video-dataset FROZEN_VIDEO_DATASET
python -m benchmarks.eval.omni_frontend_parity --manifest FIXTURES/manifest.json --family minicpm --backend official --model-path MODEL --output official-generation.json
python -m benchmarks.eval.omni_frontend_parity --manifest FIXTURES/manifest.json --family minicpm --backend omni --api-url http://127.0.0.1:18000 --output omni-generation.json
python -m benchmarks.eval.frontend_parity_metrics official-generation.json omni-generation.json
OMNI_PARITY_API_URL=http://127.0.0.1:18000 OMNI_PARITY_MANIFEST=FIXTURES/manifest.json pytest tests/unit_test/serve/test_openai_api.py -k live_chat_frontend
```

Use the untouched baseline checkout for before generation. For frozen reruns reuse manifests and checkpoint directories; preparation resolves current dataset heads. Use --family qwen for Qwen and the Transformers 4.51 environment for MiniCPM reference. The official Qwen harness casts floating processor tensors to model dtype; an initial run without that cast failed audio tasks and was replaced by a complete successful rerun.
