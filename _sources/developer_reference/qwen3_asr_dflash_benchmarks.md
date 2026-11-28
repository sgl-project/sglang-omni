# Qwen3-ASR DFlash Benchmarks

These measurements were moved from the [Qwen3-ASR cookbook](../cookbook/qwen3_asr.md#dflash-speculative-decoding). Each experiment has its own hardware, checkpoint, cache, and scheduling conditions; do not combine their numbers into one baseline. For current launch options and platform restrictions, use the cookbook.

## 1.7B throughput and latency

On one H200 over SeedTTS clips, with streamed responses and the prefix cache off, the draft on its defaults raised throughput 1.72x at 1 concurrent request, 1.42x at 4 and 1.22x at 16 on English clips and 1.58x, 1.38x and 1.17x on Chinese clips. It lowered p95 latency by 22–51% from 1 to 16 concurrent requests, with the same WER and CER. On one H100 over SeedTTS English clips with synchronous scheduling, the draft raised throughput 1.90x at 1 concurrent request, 1.56x at 4 and 1.39x at 16. It lowered p95 latency by 26–42% and the p95 of the first streamed text by 9–26%, with the same WER. With the clips repeated and the prefix cache on, it raised throughput 1.90x, 1.51x and 1.40x.

## 0.6B on RTX 3090

One RTX 3090 (24 GB, SM86), SGLang-Omni `c15bfc80`, SGLang 0.5.23 and PyTorch 2.14.1+cu130 measured the target `Qwen/Qwen3-ASR-0.6B@5eb144179a02acc5e5ba31e748d22b0cf3e303b0` against `wilsonzheng0327/Qwen3-ASR-0.6B-dflash@fbf0ce97e3b76d8cd9f3da1791a8872c3d4c3a45`. The SeedTTS revision was `27f4c1adee83b5b29b7c4b375f6b976324bda308`. Each side ran twice in ABBA order on fresh servers; each leg used one warmup and three measured repeats at every load, unique audio (666 EN clips and 1,010 ZH clips), streamed responses and the prefix cache off. Across all loads, repeats and legs, this produced 140,784 measured requests and 46,928 warmup requests.

| concurrency | EN throughput gain | EN p95 latency | EN first output p95 | ZH throughput gain | ZH p95 latency | ZH first output p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.79x | -36.9% | +0.6% | 1.60x | -33.0% | +1.7% |
| 2 | 1.79x | -52.5% | -57.3% | 1.62x | -47.8% | -57.0% |
| 4 | 1.59x | -42.2% | -55.6% | 1.40x | -32.5% | -53.0% |
| 8 | 1.47x | -38.4% | -43.3% | 1.32x | -31.3% | -47.3% |
| 16 | 1.27x | -27.7% | -25.1% | 1.21x | -24.8% | -34.5% |
| 32 | 1.74x | -33.6% | -20.0% | 1.62x | -32.8% | -22.4% |
| 64 | 1.72x | -37.4% | -34.4% | 1.63x | -35.5% | -32.8% |

The c32 and c64 points set client concurrency above the profile's 16-request running cap. At c16, the client offers only enough requests to fill that cap. After a response completes, its replacement must pass through the client and HTTP path before reaching the freed slot. These gaps cost the faster draft side more because it turns requests over more often. At c32, up to 16 requests are already waiting in the server's admission queue, so a freed slot can be refilled without a client round trip. Server logs showed at most 16 running requests, with as many as 16 queued at c32 and 48 at c64. The client measured p95 from immediately before the HTTP POST through response completion. Server admission-queue time is included; time waiting for the benchmark client's own concurrency semaphore is outside the measurement. From c32 to c64, throughput stayed nearly flat while p95 rose with the longer queue. The server cap remained 16, and the roadmap's not-worse gate ends at c16.

A second clean sweep at `2e35ab6c` repeated the same prompts with the prefix cache on. It used the same ABBA order, loads, repeats and request counts. The cache served 95.20% of prompt tokens in every EN leg and 95.21% in every ZH leg.

| concurrency | EN throughput gain | EN p95 latency | EN first output p95 | ZH throughput gain | ZH p95 latency | ZH first output p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.87x | -38.8% | +0.8% | 1.67x | -34.8% | +0.6% |
| 2 | 1.93x | -55.6% | -61.6% | 1.72x | -51.8% | -63.8% |
| 4 | 1.72x | -47.7% | -59.8% | 1.54x | -41.8% | -61.2% |
| 8 | 1.70x | -44.7% | -48.4% | 1.46x | -37.4% | -51.7% |
| 16 | 1.45x | -33.5% | -28.9% | 1.30x | -27.8% | -32.6% |
| 32 | 1.79x | -36.4% | -22.4% | 1.61x | -34.6% | -21.8% |
| 64 | 1.80x | -38.0% | -36.2% | 1.59x | -32.1% | -27.8% |

No request failed in either sweep. With the prefix cache off, aggregate English WER was 1.6393% without the draft and 1.6295% with it; aggregate Chinese CER was 1.0135% and 1.0096%. With repeated prompts, WER ranged from 1.65% to 1.68% without the draft and 1.61% to 1.66% with it; CER ranged from 1.00% to 1.02% and from 1.00% to 1.01%. Mean acceptance length was 5.08 on English and 4.38 on Chinese. In the prefix-cache-off sweep, peak GPU memory rose from 18,485 to 19,187 MiB on English and from 18,527 to 19,243 MiB on Chinese. Every load completed, and neither prefix-cache mode was worse than plain decoding from c1 to c16 by the roadmap bounds.

## Sampling and repetition penalties

On CUDA, DFlash serves any `temperature`. Above 0, verification accepts each drafted token with the target model's probability for it and otherwise samples from the target's remaining probabilities. ASR transcripts are confident, so sampling barely lowers acceptance: on one H100 with SeedTTS English and Chinese clips at 16 concurrent requests, each verification step committed 5.66 tokens at `temperature` 0, 5.66 at 0.7 and 5.62 at 1.0. On the same server at the default cap of 64 running requests, the speedup at 0.7 and 1.0 stayed close to its greedy speedup at every load tested, from 1 to 64 concurrent requests. DFlash retains CUDA graphs.

On CUDA, DFlash applies `repetition_penalty` as ordinary decoding does: each verified position is penalized over the transcript so far and the drafted tokens before it. On one H200 with every SeedTTS clip sent at 1.2, no request failed, transcripts differed from a synchronous server without the draft about as often as they do with no penalty (5 of 666 English, 2 of 1,010 Chinese, 1 request at a time), and the logged acceptance length fell by about 1%. The draft does not see the penalty, so a strong one lowers acceptance more: by about 15% at 2.0.

## Async scheduling

The draft overlaps scheduling with host work by default; add `--asr.factory.enable_async_decode false` to schedule synchronously. On one H100 with greedy decoding and the prefix cache off, overlap ran at 1.12–1.26x synchronous scheduling on 60-second recordings, 1.01–1.15x on SeedTTS English clips and 0.99–1.15x on SeedTTS Chinese clips, from 1 to 64 concurrent requests, with the same WER and CER. On short clips it also removed the draft's loss under load: at 64 concurrent requests the synchronous draft ran at 0.94x the server without a draft on English clips and 0.91x on Chinese, and the overlapped draft at 1.08x and 1.04x. Overlap runs one extra verify round per request, so the logged acceptance length drops (5.9 to 4.6 tokens on English clips), and streamed first text arrives about 2 ms later for one short clip at a time. A decode batch smaller than `--asr.factory.async_decode_min_batch_size`, or a prefill after which fewer requests than it are decoding, has its result processed before the next batch is chosen; other batches overlap. With the draft it defaults to 2, so a lone request decodes synchronously; without a draft it defaults to 1. On one H200 with SeedTTS clips sent once each and streamed, the prefix cache off, first streamed text p95 for one clip at a time matched the server without a draft at 2 (1.00x on English and Chinese), against 1.16x and 1.19x with overlap at every batch size. That costs about 5% of single-request throughput on English and 4% on Chinese; from 2 to 64 concurrent requests the two settings stayed within 3% of each other in throughput. At 4 on one H100, first streamed text for one clip at a time also matched the server without a draft, and throughput stayed within 5% of overlap except at 2 requests, where it was about 7% lower. Set it to 1 to overlap every batch.

## High-concurrency regression

The H200 that measured 1.72x differed from the H100 under load: at the default cap of 64 running requests with the prefix cache off, the draft on its defaults gave no speedup at 32 concurrent requests and ran about 10% slower at 64, on both English and Chinese clips, with first streamed text p95 11–21% later than without the draft at both loads. Overlapping every batch had the same throughput, and synchronous scheduling lost the same throughput on English clips in an earlier sweep ([#3454](https://github.com/sgl-project/sglang-omni/pull/3454)).

## Realtime latency and memory

Realtime transcription sessions (`--enable-realtime`) use the draft too. On one H100 at the default memory fraction and cap of 64 running requests, SeedTTS English and Chinese clips and 60-second LongLibriHeavy recordings streamed at wall-clock pace, 1 to 16 sessions at a time, with server VAD or manual commit. No session hit an error or a protocol violation, WER stayed within 0.03 percentage points of the server without the draft, and the median time to a segment's first partial was 11 to 28% lower with the synchronous draft, least at a 1000 ms decode interval, and to its final transcript 17 to 27% lower. Overlap on every batch gave the same transcripts and median latencies as the synchronous draft on SeedTTS English sessions with server VAD, 1 and 16 at a time, with no errors or protocol violations.

The draft's weights and KV cache come out of the stage's memory fraction. On an 80 GB H100 at `--mem-fraction-static 0.08`, as in the streaming cascade profiles, the KV cache holds 11,501 tokens with the draft and 20,705 without. At that fraction, 640 SeedTTS Chinese sessions, 16 at a time, ran with no errors.
