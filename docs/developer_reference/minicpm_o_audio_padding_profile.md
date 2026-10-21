# MiniCPM-o offline audio padding performance

## Change and scope

Offline MiniCPM-o audio preprocessing previously padded every segment to 30 seconds before mel extraction and discarded the unused frames. Interleaved video audio contains many short segments, so this work remains even when audio encoder outputs are cached. The bounded adapter uses the checkpoint extractor with padding limited to the longest segment plus one FFT window, capped at 30 seconds. The extra zeros preserve STFT boundary frames and the normalization floor. Dither retains the original padding. Native full-duplex streaming is unchanged.

The measured workload is repeated videos with interleaved source audio and warm encoder caches. It does not measure general video, first-use, or TTS throughput.

## Method

- Baseline: `b658fa671d6a8a01d9b973754300837dff49d22d`; candidate: the bounded audio adapter and its integration.
- One H100 NVL, GPU 0, the same 12 CPU affinity cores, `OMP_NUM_THREADS=4`; torch 2.13.0+cu130, sglang 0.5.21, transformers 5.12.1.
- Checkpoint: `openbmb/MiniCPM-o-4_5` at revision `503e754207c94da6bb26850b4469f367c9ea3582`.
- Videos from `zhaochenyang20/Video_MME_ci`: `qvwfWyU9Gfc.mp4`, `24i4ncHuf6A.mp4`, and `N1cdUjctpG8.mp4`.
- Ordered inline video with interleaved audio, 1 fps, at most 64 frames and 401408 pixels per frame. Each request asks for a one-sentence description with temperature 0, seed 42, and at most 32 output tokens.
- Four fresh services ran in A1 → B1 → B2 → A2 order on the same GPU. Each had independent warmup, then 17 measured requests: two first-use videos, two sequential three-video windows, six requests at concurrency 3, one speech-output request, and two single-audio requests.

## Results

| Metric | A1 | B1 | B2 | A2 |
|---|---:|---:|---:|---:|
| Warm sequential mean, three videos (s) | 2.4529 | 1.2786 | 1.2045 | 2.2089 |
| Concurrency-3 throughput, six requests (req/s) | 0.4626 | 0.9332 | 0.9678 | 0.5011 |
| Concurrency-3 p95 (s) | 6.3003 | 3.5862 | 3.5391 | 6.4043 |
| End-of-run GPU memory (MiB) | 63513 | 62509 | 63501 | 62513 |

Both candidate visits outperform both baseline visits: mean concurrency-3 throughput rises by about 97%, and warm sequential latency falls by about 47%. The six-request p95 is descriptive, and end-of-run memory is not peak memory. A checkpoint feature-extraction screen measured 16 one-second segments at 92.67 → 3.89 ms and 64 segments at 586.57 → 11.87 ms on four CPU threads; these component timings are not end-to-end speedups.

All 68 measured requests succeeded. Matching video requests returned the same text and token counts across all four services. A 33-case checkpoint extractor screen found bitwise-equal valid mel frames, and 125 related tests passed with the pinned checkpoint, including the full processor comparison. Speech output had matching transcripts and 156480-frame, 24 kHz mono WAVs. Cold and first speech requests include variable lazy initialization, so no stable gain is claimed for them. The second single-audio request was about 0.26–0.28 s in every arm.

## Reproduce

Install the pinned project dependencies, including decord, and download the checkpoint revision and three videos above. Run baseline and candidate on the same GPU and CPU affinity with identical settings:

```bash
python -m sglang_omni.cli serve \
  --model-path "$CHECKPOINT" --host 127.0.0.1 --port 18129 \
  --thinker.factory.max_seq_len 32768 \
  --thinker.engine.mem_fraction_static 0.55 \
  --talker.engine.mem_fraction_static 0.15
```

Send identical `/v1/chat/completions` requests with an inline `video_url`, `use_audio_in_video=true`, `video_fps=1`, `video_max_frames=64`, `video_max_pixels=401408`, and the generation settings above. Warm each service independently. For throughput, submit the three videos twice with three client workers and divide six completed requests by total wall time. Keep first-use results separate and reverse the service order.

```bash
MINICPMO_CHECKPOINT="$CHECKPOINT" python -m pytest \
  tests/unit_test/minicpm_o/test_preprocessor_prompt.py \
  tests/unit_test/minicpm_o/test_preprocessor_video.py \
  tests/unit_test/minicpm_o/test_sglang_talker.py \
  tests/unit_test/minicpm_o/test_thinker_model_runner.py -q
```

This small repeated-input workload does not establish production tail latency or broader language and long-form quality. No ASR, speaker-similarity, MOS, or waveform bitwise comparison was performed.
