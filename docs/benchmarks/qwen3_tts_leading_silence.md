# Qwen3-TTS Base: Leading Silence in X-Vector Voice Cloning

A Base voice clone without a reference transcript (x-vector mode) often opens with
several frames of silence before any speech. The silence is part of the generated
codec, so it adds directly to the time to first audible sample, in streaming and
non-streaming output alike. The engine now excludes silence codec ids from the first
two frames of x-vector-only requests. This page records how the cause was found,
how the default was chosen, and what it costs.

## Where the silence comes from

The talker samples one codebook-0 id per 80 ms frame. Logging the ids the talker
actually sampled for 360 x-vector requests on 1.7B Base (3 references, 12 prompts,
10 seeds) shows that the silence is decided on frame 0:

- Every one of the 322 requests that started with silence sampled id 1995 on frame 0.
  The requests that started speaking immediately sampled speech ids (404, 1221, 9, ...).
- Once frame 0 is 1995, the talker stays in room tone (about -60 dBFS) for four to
  seven frames, so onsets are bimodal: frame 0, or 320 to 640 ms.
- With the silence ids masked on frame 0, the talker's next choice is id 1221, a soft
  onset id. It is followed by speech in about 30% of cases (61 of 213) and by a
  quieter pause in the rest.

ICL requests are different. Their onset follows the pause at the end of the
reference clip, which the talker continues, so they are left unmasked.

## Method

- Rows flagged x-vector-only have the silence ids set to -inf while
  `len(output_codes) < leading_silence_mask_frames`. The mask is applied before
  temperature and top-k/top-p, so top-k is taken over the masked distribution.
  A retracted request keeps its `output_codes`, so re-prefill does not re-mask.
- The silence ids are derived at engine startup from the checkpoint's own codec:
  white, pink, and brown noise from digital silence up to -50 dBFS in 5 dB steps,
  8 s each, encoded, and the codebook-0 ids collected (24 ids on 1.7B Base). The
  share of generated leading-silence frames this set covers saturates at a -50 dBFS
  ceiling on both 1.7B and 0.6B, and no speech frame falls in it up to -40 dBFS.
- `leading_silence_mask_frames` is a `tts_engine` factory argument; 0 turns the
  mask off.

## Results

Hardware: 4x H200, one server per GPU. Each row is one server.

- **Grid**: 360 non-streaming x-vector requests (3 references from
  `zhaochenyang20/seed-tts-eval-mini`, 12 prompts, seeds 0 to 9, concurrency 8).
- **SeedTTS**: the full SeedTTS EN set (1088 clips) through
  `benchmarks/eval/benchmark_tts_seedtts.py --no-ref-text --seed 0 --concurrency 16`,
  WER from the CI ASR model, speaker similarity from the CI WavLM scorer.
- **Onset**: the first 5 ms frame whose peak reaches 0.02 (-34 dBFS). A 10 ms RMS
  detector at -40 dBFS gives medians within 25 ms of it.
- **Runaways**: clips that ran to `max_new_tokens` (163.84 s).

| Checkpoint | Frames masked | Grid median | Grid p90 | Grid > 160 ms | SeedTTS WER | SIM | SeedTTS median | SeedTTS p90 | SeedTTS > 160 ms | Runaways |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.7B | 0 | 445 ms | 635 ms | 89% | 0.854% | 61.00 | 495 ms | 745 ms | 95% | 0 |
| 1.7B | 0 (repeat) | 445 ms | 635 ms | 89% | 0.846% | 61.00 | 495 ms | 745 ms | 95% | 0 |
| 1.7B | 1 | 135 ms | 520 ms | 46% | 0.888% | 60.94 | 470 ms | 737 ms | 73% | 0 |
| 1.7B | 2 | 115 ms | 455 ms | 34% | 0.904% | 60.87 | 155 ms | 607 ms | 47% | 0 |
| 1.7B | 3 | 115 ms | 455 ms | 34% | 0.888% | 61.00 | 155 ms | 610 ms | 46% | 0 |
| 0.6B | 0 | 375 ms | 580 ms | 69% | 1.532% | 58.41 | 480 ms | 837 ms | 86% | 9 |
| 0.6B | 1 | 75 ms | 521 ms | 20% | 1.089% | 58.43 | 75 ms | 562 ms | 34% | 2 |
| 0.6B | 2 | 75 ms | 236 ms | 12% | 1.072% | 58.10 | 75 ms | 490 ms | 23% | 3 |

The 3-frame 1.7B row used an earlier, smaller silence set (white noise only, 11 ids).
At 1 and 2 frames that set gave the same grid request for request as the shipped set,
because both contain 1995 and neither contains 1221.

Reading the table:

- Run-to-run noise: the two 1.7B baselines differ by 0.008 WER points and not at all
  in SIM; the grid reproduces request for request.
- Two frames is the default. The third frame never samples a silence id, so it
  changes nothing.
- 1.7B costs about 0.05 WER points and 0.13 SIM for a median onset of 155 ms
  instead of 495 ms. 0.6B improves WER, mostly because fewer clips run away: a clip
  that starts silent is more likely never to stop.

## Masking the soft-onset id as well

Adding 1221 to the mask removes the leading silence almost entirely, at a visible
quality cost, so it is not shipped:

| Checkpoint | Frames masked | Grid median | Grid > 160 ms | SeedTTS WER | SIM | SeedTTS > 160 ms |
|---|---|---|---|---|---|---|
| 1.7B | 1 | 35 ms | 1% | 1.114% | 59.92 | 0% |
| 0.6B | 1 | 40 ms | 0% | 1.315% | 57.24 | 0% |

1221 is an ordinary soft onset, and forbidding it forces a hard attack on every
utterance. It was also found by probing one checkpoint rather than derived, so it
would be a hardcoded id.

## Onset detector

```python
import numpy as np


def peak_onset_ms(audio: np.ndarray, sample_rate: int) -> float | None:
    frame = sample_rate // 200
    count = len(audio) // frame
    peaks = np.abs(audio[: count * frame]).reshape(count, frame).max(axis=1)
    hits = np.flatnonzero(peaks >= 0.02)
    return float(hits[0] * 5) if hits.size else None
```

## Not yet evaluated

Blinded listening and first-phoneme error rate, attack clicks, long utterances
(30 s and up), languages other than English, streaming first-chunk timing, and
confidence intervals clustered by reference.
