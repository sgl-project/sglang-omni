# Qwen3-TTS Base runtime research

This port recovers the Qwen3-TTS Base 1.7B candidate evaluated in OpenRSI
submission 6 and the Base 0.6B candidate evaluated in submission 10. The historical complete candidates improved non-streaming
concurrency-16 throughput by **44.36% (0.6B)** and **26.15% (1.7B)**;
streaming throughput improved by **7.17%** and **6.20%**, respectively.
These measurements use the historical source and are not a new benchmark of
the current upstream port, or an isolated measurement of any one component.
The older upstream comparisons below cover the exact-frame implementation.
Their source revisions remain explicit. Separate Radix comparisons isolate
the review-time bucketing candidate within the PR runtime. The final policy
retains bucketing for 1.7B and exact-frame captures for 0.6B.

## Runtime changes

- Base cloning requests use the existing full prefill graph backend, with
  token buckets through 1,024 for 1.7B and the 8,192-token prefill budget for 0.6B. CustomVoice retains its existing 512-token
  ladder; VoiceDesign and unknown variants keep the breakable default.
- For 1.7B, whole-utterance CUDA vocoding uses 16-frame buckets through 256
  frames and batch sizes 1–4: up to 64 graphs. Requests are zero-padded to
  the smallest captured frame bucket and decoded audio is trimmed to the
  original length. Unsupported shapes, deterministic decoding and disabled
  graphs use the original tokenizer path.
- For 0.6B, whole-utterance CUDA vocoding preserves the native chunk/context
  boundaries (300/25 codec frames), uses exact lengths 1–325 for one or two
  requests (650 graphs), and uses a dedicated stream. Default non-streaming vocoder batching
  becomes two requests with no collection wait; explicit overrides win.
  Whole-utterance graphs require a recognized Base checkpoint size.
- Capture larger shapes first within a shared graph pool to limit reserved
  memory fragmentation. This adds startup work and resident graph memory.
- The initial incremental decoder launches its next cohort before resolving
  the previous one, retaining at most two in flight. Existing tail-decode
  routing, invalid-row handling and cancellation ownership remain in use.
- Fence each codec arena host index slot's outstanding upload before ring
  reuse. The four-slot ring and index values are unchanged.

Upstream already contains the historical invalid-row-mask transport mechanism
in [#2548](https://github.com/sgl-project/sglang-omni/pull/2548). This port uses
that implementation. It also preserves the follow-up scheduling in
[#2400](https://github.com/sgl-project/sglang-omni/pull/2400) and the first-chunk
tail runner in [#2658](https://github.com/sgl-project/sglang-omni/pull/2658).

## Bucketed whole-utterance graphs

The review-time change adopts [db-ol's bucketing implementation](https://github.com/db-ol/sglang-omni/commit/ffd8a22f9008ac9b6a0dab03ffaf25896e22dda8),
retaining the original author attribution. For 1.7B it replaces up to 1,024
exact-frame captures with 64 graphs. The candidate also reduced 0.6B to 42
graphs, but the independent three-seed check below showed higher corpus WER
in all three pairs. Commit `6075e99c902d1e9565110c20b5f155139dd7b4ad`
therefore restores the 650 exact-frame 0.6B captures and disables frame
padding for that holder. This is a conservative default-policy decision;
the serving measurements do not establish that padding caused the shift.

Capture remains in descending shape order within a shared pool. Only the
1.7B whole-utterance holder enables frame padding; initial and follow-up
streaming graph policies are unchanged. The 0.6B native 300/25-frame
chunk/context boundaries and reference-prefix trimming remain in place.
Static inputs are cleared before every replay; returned audio is cloned and
trimmed to the original frame count.

The tokenizer already zero-pads shorter requests to the longest request in
a native batch. Batch composition can therefore change tensor shapes and
floating-point results even without CUDA graphs. The causal decoder permits
trailing padding, but bitwise equality with a narrower native call is not
the contract for the bucketed path. The appropriate replay check compares
the graph with eager execution at the same padded shape, then checks
original-length trimming and fallback behavior.

[The reviewer reported](https://github.com/sgl-project/sglang-omni/pull/2779#discussion_r4286208221)
H200 SeedTTS EN non-streaming C16 results with interleaved seeds 42/43/44:
mean startup decreased from 333 to 157 seconds for 1.7B and 317 to 150
seconds for 0.6B over four launches each, with mean latency and WER within
their observed run-to-run variation. These are reviewer-reported results;
they were not independently recomputed from raw records in this evidence
archive and are not pooled with the measurements below.

### Initial both-size bucket candidate validation (2026-10-15)

The intermediate both-size bucket implementation in `6718550d76cc498fce830819879a7c5a6491ea88`
was tested on one Radix H200 with Python 3.12, SGLang 0.5.21, PyTorch
2.13.0+cu130, Transformers 5.12.1 and qwen-tts 0.1.1. The added tests
passed: **462 unit tests**, with three Ascend-only skips and 193
benchmark/accelerator cases deselected; **10 real-checkpoint CUDA tests**,
with no skips. The GPU cases cover padded graph versus padded eager
execution for both Base sizes, plus the existing 0.6B native chunk,
reference-trim and fallback cases. Full pre-commit passed.

Fresh bucketed servers reached health in **240.8 seconds (1.7B)** and
**180.5 seconds (0.6B)** with `SGLANG_OMNI_STARTUP_TIMEOUT` unset, using
the upstream 600-second limit. Each served two non-streaming and two
streaming requests successfully after 16 excluded warmups. The observed
health times have a 10-second polling resolution and include warm compiler
caches. These single-launch observations verify startup in this Radix
environment; they are not a paired startup-speedup estimate or a serving
quality benchmark. The earlier VISION startup limitation remains documented
below with its original source and environment.

The 1.7B snapshot is `fd4b254389122332181a7c3db7f27e918eec64e3`; the
0.6B snapshot is `5d83992436eae1d760afd27aff78a71d676296fc`. Both use
BF16 and the default pipeline configurations, with Torch compile disabled
uniformly. The runtime file SHA-256 is
`84203399a0e92a471755097bb7a9a7b300cbe994a6d94363d91990456824b444`.
These startup measurements precede the final 0.6B policy restoration and do
not describe its current default. Final source validation is recorded below.

### Independent exact-frame versus bucket comparison (2026-10-15)

This additional pair isolates frame bucketing within the existing PR runtime:
exact-frame `5f4de745afc9c2e7a6e5dea19513a9aaff959b82` versus the bucketed
runtime from `6718550d76cc498fce830819879a7c5a6491ea88`, published with
tests/documentation in `b6ece10ee6e77141b46dcb6efd307be33a0c21de`.
It does not compare the full PR with its upstream branch point.
Both Base sizes use one paired seed, 42, in non-streaming C16 mode. The
0.6B bucket arm is the rejected candidate, not the final default. The final
1.7B runtime path is unchanged from this measured bucket arm.

| Base size (non-streaming C16) | Exact requests/s | Bucket requests/s | Change | Exact WER (%) | Bucket WER (%) | Exact SIM | Bucket SIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6B | 29.545 | 28.703 | -2.85% | 1.6328 | 2.6375 | 70.3556 | 70.1101 |
| 1.7B | 30.215 | 30.100 | -0.38% | 1.1722 | 1.0550 | 71.5250 | 71.0479 |

0.6B: ΔWER +1.0048 percentage points; ΔSIM -0.2455; exact/bucket token-cap hits 1/3, samples with WER >50% 1/2.

1.7B: ΔWER -0.1172 percentage points; ΔSIM -0.4771; exact/bucket token-cap hits 0/0, samples with WER >50% 1/1.

All four cohorts contain 1,088 unique successful requests and matching
generation/WER/SIM/speed IDs, with zero failed requests or skipped scores.
Full-corpus WER and mean cosine similarity times 100 were independently
recomputed from checksum-verified archives. Token-cap hits and high-WER
outliers are retained. One paired seed does not establish that observed
differences are noise, prove quality equivalence or show statistical
significance. These values are separate from the earlier three-seed study.

| Base size | Exact / bucket startup (s) | Exact / bucket mean full latency (s) |
| --- | ---: | ---: |
| 0.6B | 333.1 / 180.1 | 0.439 / 0.462 |
| 1.7B | 340.5 / 181.8 | 0.525 / 0.528 |

Both arms use the upstream 600-second startup limit, with no environment
override. Health was polled every 10 seconds. Startup observations include
warm compiler caches on a shared host, so the single pairs describe this
environment and are not a controlled cold-start study. The separate
default-timeout smoke checks above also exercised streaming requests.

The four cohorts ran sequentially on the same Radix H200 and the pinned
environment described above. Order: 0.6B exact/bucket, then 1.7B exact/bucket.
Each starts a fresh server, excludes 16 warmup requests, and evaluates all
1,088 SeedTTS EN samples in native order with concurrency 16, seed 42 and
a 2,048-token cap. Client and scorer source is fixed at `94730a87`, identical
for both arms. All inputs use the pinned revisions below; SIM uses batch
one with complete waveforms and ASR retains its default concurrency 32.
Server/client affinity is 16 physical CPU cores (0–15), without exclusive
reservation. The bucket runtime/test hashes match `b6ece10e`; the later
`6075e99c` restores the 0.6B policy as described above.

Use the generation/scoring commands below with these two runtime sources,
`qwen3_tts_0_6b.yaml` or `qwen3_tts_1_7b.yaml`, seed 42, and startup-timeout
override unset. [0.6B audited numeric records and protocol](https://github.com/OpenRSI-Foundation/sglang-omni-qwen3-tts-base-1.7b-runtime-d154/tree/a6a74c5688a1ff948b5740508cba62299e213e63/review-evidence/pr2779/20261015/radix-qwen06), [1.7B audited numeric records and protocol](https://github.com/OpenRSI-Foundation/sglang-omni-qwen3-tts-base-1.7b-runtime-d154/tree/b14382514c885136769258e6f46677d586dfcc53/review-evidence/pr2779/20261015/radix-qwen17)
are backed up with remote byte verification; the evidence repositories
require access. The public tables and protocol are self-contained.

### Three-seed 0.6B bucket candidate check (2026-10-15)

After the seed-42 shift, seeds 43 and 44 were declared before execution.
The order was exact/bucket for 42, bucket/exact for 43, and exact/bucket for
44. Sources, settings, scorers and inputs are the same as the preceding
pair. The original seed 42 remains included. Each of the six cohorts has
1,088 successful requests and matching generation/WER/SIM/speed IDs, with
zero failures or skipped scores. All capped and high-WER samples remain.

| Seed | Exact WER (%) | Bucket WER (%) | Δ WER (pp) | Exact SIM | Bucket SIM | Exact / bucket requests/s | Exact / bucket cap hits | Exact / bucket WER >50% samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 1.6328 | 2.6375 | +1.0048 | 70.3556 | 70.1101 | 29.545 / 28.703 | 1 / 3 | 1 / 2 |
| 43 | 1.2392 | 1.3648 | +0.1256 | 70.2263 | 70.3008 | 28.164 / 34.882 | 2 / 1 | 2 / 1 |
| 44 | 1.1555 | 1.2141 | +0.0586 | 70.0344 | 70.0432 | 35.203 / 33.446 | 1 / 1 | 0 / 1 |

Equal-weight mean corpus WER is **1.3425% exact / 1.7388% bucket**. Mean
paired ΔWER is **+0.3963 percentage points**, sample SD **0.5280**. Mean SIM
is **70.2055 / 70.1514**; mean paired ΔSIM is **-0.0541**, sample SD
**0.1690**. WER is higher for the bucket candidate in all three pairs;
throughput and SIM deltas change sign. SD is descriptive over three pairs,
not a significance or equivalence test.

For seed 42, two capped samples account for 117 of the 120 additional word
errors. Generated token counts differ for 1,005 of 1,088 request IDs, so
this is not a fixed-codec decoder comparison and cannot attribute the WER
shift to frame padding alone. The final policy restores the pre-bucketing
0.6B capture shapes while retaining 1.7B bucketing. This removes the newly
introduced 0.6B default change without discarding its unfavorable evidence.

[Audited per-sample numeric records, all six cohort summaries, predeclared
seed order and paired statistics](https://github.com/OpenRSI-Foundation/sglang-omni-qwen3-tts-base-1.7b-runtime-d154/tree/e003cf994a71befee4f3b44acb490b5f5710c88a/review-evidence/pr2779/20261015/radix-qwen06-three-seed)
are backed up with remote byte verification; repository access is required.

### Final source validation (2026-10-15)

Final runtime `6075e99c902d1e9565110c20b5f155139dd7b4ad` passed **462
unit tests** (three Ascend-only skips, 193 benchmark/accelerator cases
deselected) and **10 real-checkpoint CUDA tests with no skips** on the same
Radix H200 environment. Four GPU cases check exact waveform equality for
the production 0.6B native chunk/reference-trim and fallback paths. Six
utility cases compare padded graph replay with padded eager decoding across
both sizes; those tests explicitly enable padding for the 0.6B utility,
whose production default is now disabled.

Fresh final-source servers reached health in **311.0 seconds (0.6B,
exact frames)** and **190.1 seconds (1.7B, buckets)**. Both retained the
upstream 600-second timeout with no environment override and successfully
served two non-streaming and two streaming requests after 16 excluded
warmups. Health polling resolution is 10 seconds, with warm compiler caches
on a shared host. These checks establish startup and serving in this Radix
environment, not a paired speedup or a new quality measurement. Full
pre-commit passed on the final source and this benchmark document.

Runtime SHA-256:
`57bde622dad03afdd71a608c4a50507d999e3df2c7237cd06b282ec8cd6ac3be`.
Test-file SHA-256:
`5e3177f123e41d1183db3e59b05db5efa8d26338bc2d4c471b5b95feb554557e`.

### Earlier native-padding diagnostic

A native-decoder padding diagnostic on 2026-10-15 provides a direct check of
that contract. It used the pinned 1.7B tokenizer below, BF16, the configured
29 fused SnakeBeta modules, H200, PyTorch 2.13.0+cu130, Transformers 5.12.1
and qwen-tts 0.1.1, without the optional `flash-attn` package. The tokenizer
used its native PyTorch fallback. The 66 cases combined frame lengths
31/32/33/63/64/65/127/128/129/255/256, batch sizes 1/2/4 and fixture seeds
7/11. Each batch retained its native longest length; shorter rows used the
native zero-code padding. The alternative padded that longest length to the
next power of two. Both outputs were trimmed to each row's original length
and reference-prefix boundary.

All 66 identical-shape repeats and all 24 cases needing no extra padding
were bitwise equal. Of the 42 cases with additional padding, 34 changed
retained waveform samples. For example, a single 31-frame request padded to
32 frames had maximum absolute difference 0.0541992 and mean absolute
difference 0.00139271 after trimming five reference frames. Differences were
also present outside its last two frames. The largest absolute difference
over the matrix was 0.223145.

These are synthetic valid codec IDs decoded by the real tokenizer, with
eager execution on both sides. They demonstrate that padding and trimming
do not preserve byte parity for this backend. They do not measure perceptual
quality, serving performance or a CUDA-graph bucketing implementation.
Result JSON SHA-256:
`d4e9d5da648c5e5ecfd4c6030845736a1c8f3c7fef88135060370dcb4635c557`.
[The complete fixture matrix and executed script](https://github.com/OpenRSI-Foundation/sglang-omni-qwen3-tts-base-1.7b-runtime-d154/tree/cf4a50d447d659624bdb32ba4591b0287f272ee6/review-evidence/pr2779/20261015/frame-padding)
are retained in the evidence repository, which requires access.

The first counterexample can be reproduced from this checkout with
`MODEL_PATH` pointing to the pinned 1.7B snapshot:

```python
import os

os.environ["SGLANG_OMNI_TORCH_COMPILE_DEFAULT"] = "0"

import torch

from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
tokenizer = load_qwen3_tts_tokenizer(
    os.environ["MODEL_PATH"],
    device="cuda:0",
    dtype="bfloat16",
    attn_implementation=None,
)
decoder = tokenizer.model.decoder.eval()
fuse_vocoder_decoder(decoder)
quantizers = tokenizer.model.config.decoder_config.num_quantizers
generator = torch.Generator().manual_seed(7 * 10000 + 31 * 10 + 1)
codes = torch.randint(0, 128, (1, quantizers, 31), generator=generator).cuda()
padded = torch.nn.functional.pad(codes, (0, 1))
with torch.inference_mode():
    decoder(torch.zeros_like(padded))
    native = decoder(codes).clone()
    repeat = decoder(codes).clone()
    bucketed = decoder(padded).clone()
    assert torch.equal(native.view(torch.uint8), repeat.view(torch.uint8))
    native = native[..., 5 * 1920 : 31 * 1920].float()
    bucketed = bucketed[..., 5 * 1920 : 31 * 1920].float()
    print((native - bucketed).abs().max().item())
```

The diagnostic above checks narrower versus padded eager execution; it does
not establish that the numerical difference is larger than the native
batch-composition effect, or that frame bucketing is unsuitable. The new
replay tests compare padded graphs with padded eager execution for both
Base sizes. Disabling the existing initial_cuda_graph option retains the
tokenizer fallback and also disables initial streaming graphs.

## Pre-bucketing three-seed quality follow-up (2026-10-15)

For Base 1.7B, this follow-up adds paired seeds 43 and 44 and reruns seed
42 in the same new environment. It does not pool the October 14 single
pairs with these repetitions. Base remains the fixed branch point
`fa4e6d1f1b4c962ecc223e1422f6e879f5ce1141`; evaluated head is
`d8977e6f8814320bdbc3549218c872450912caa5`. These measurements predate the bucketing change and describe the
exact-frame runtime, not the current bucketed implementation.

Every arm/mode/seed used a fresh server, all 1,088 SeedTTS EN requests in
native order, concurrency 16, a 2,048-token cap and 16 excluded warmups.
Base ran first for seeds 42/44; head ran first for seed 43. All generation,
WER, SIM and speed records have the same 1,088 IDs per cohort, zero failed
requests and zero skipped scores. WER below is the full corpus edit-count
ratio; SIM is mean WavLM cosine similarity multiplied by 100. Both were
independently recomputed from the checksum-verified raw archives. All
outliers and token-cap hits remain included.

| Mode (C16) | Seed | Base WER (%) | Head WER (%) | Δ WER (pp) | Base SIM | Head SIM | Δ SIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Non-streaming | 42 | 0.9797 | 1.1806 | +0.2010 | 71.2558 | 71.2574 | +0.0016 |
| Non-streaming | 43 | 1.0215 | 1.0718 | +0.0502 | 71.4009 | 71.3138 | -0.0872 |
| Non-streaming | 44 | 1.0466 | 1.0550 | +0.0084 | 71.4951 | 71.2800 | -0.2150 |
| Streaming | 42 | 1.1471 | 1.1722 | +0.0251 | 71.5253 | 71.3977 | -0.1276 |
| Streaming | 43 | 1.0299 | 1.0634 | +0.0335 | 71.1812 | 71.3749 | +0.1936 |
| Streaming | 44 | 0.9964 | 0.9964 | +0.0000 | 71.3606 | 71.3682 | +0.0076 |

| Mode | Mean base/head WER (%) | Mean Δ WER ± sample SD (pp) | Mean base/head SIM | Mean Δ SIM ± sample SD |
| --- | ---: | ---: | ---: | ---: |
| Non-streaming | 1.0159 / 1.1025 | +0.0865 ± 0.1013 | 71.3839 / 71.2837 | -0.1002 ± 0.1089 |
| Streaming | 1.0578 / 1.0773 | +0.0195 ± 0.0174 | 71.3557 / 71.3802 | +0.0245 ± 0.1613 |

Head WER was higher in 3/3 non-streaming and 2/3 streaming seeds.
The means weight each seed equally; SD describes the three paired deltas.
These repetitions show the observed variability but do not establish that
a quality shift is noise, prove non-inferiority, or measure perceptual
equivalence. No significance or quality-pass claim is made from three seeds.

| Mode | Seed | Base/head max-token hits | Base/head samples with WER >50% |
| --- | ---: | ---: | ---: |
| Non-streaming | 42 | 0 / 1 | 0 / 1 |
| Non-streaming | 43 | 0 / 0 | 0 / 0 |
| Non-streaming | 44 | 0 / 0 | 0 / 0 |
| Streaming | 42 | 0 / 0 | 0 / 1 |
| Streaming | 43 | 0 / 0 | 0 / 0 |
| Streaming | 44 | 0 / 0 | 0 / 0 |

### Follow-up environment and reproduction

The runs used one H200 per cohort on VISION, 16 allocated physical benchmark
cores plus one separate monitoring core, and a 192 GiB host-memory allocation.
The fixed Pyxis image SHA-256 is
`319cb4ed277c809ac52bd422802efc4bb2e316fcb9fc01f33e36d1a6276fe86e`
(derived from `lmsysorg/sglang` digest
`06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f`).
It uses SGLang 0.5.21, Python 3.12, PyTorch 2.13.0+cu130 and Transformers
5.12.1. This is a different container from the October 14 comparison.
All model, dataset, ASR and SIM input revisions/hashes remain those in the
reproduction table below. Clients and scorers use the unchanged base source;
SIM batch size is uniformly one with complete waveforms.

The seed-42 non-streaming baseline completed in the preceding allocation
on another H200 of the same host/image; its quality records are retained.
Its health-probe delay makes its startup time invalid, and no throughput
comparison is made for that pair across allocations. The remaining 11
cohorts share the resumed allocation. These repetitions support the
quality table, not a new pooled throughput claim.

**Pre-bucketing startup limitation:** an initial exact-frame head launch exceeded the upstream
600-second startup timeout before any formal requests. After preserving
that failure, both arms used `SGLANG_OMNI_STARTUP_TIMEOUT=1800` for all
new launches. The imported successful baseline above used the default
600 seconds. Extending the evaluation wait is not an upstream startup fix.

The five valid baseline launches took about 180 seconds;
the six head launches took 841–992 seconds. Health was polled every
10 seconds, so these are first-observed-health times with that polling
resolution. The previous 139/306-second warm-cache pair is specific to
its October 14 environment and does not describe this deployment.

Use the reproduction commands below with `--seed 42`, `--seed 43`, and
`--seed 44` for both modes and both sources. For this allocation, replace
the affinity with `0-3,11-12,14-19,26-29`, set the startup timeout to 1,800
seconds and `SPEAKER_SAMPLES_DIR` to a writable node-local directory. Use
a fresh available port for each TTS/ASR process and stop only the owned TTS
process before scoring. Keep the pinned inputs and the remaining
generation/scoring settings unchanged.

[Audited per-seed summaries, compact per-sample scores and protocol](https://github.com/OpenRSI-Foundation/sglang-omni-qwen3-tts-base-1.7b-runtime-d154/tree/f156f33b617a1f4d4871636009db26b6720178e1/review-evidence/pr2779/20261015/paired-quality)
are backed up in the evidence repository, which requires access. The tables
above are self-contained; the backup excludes dataset text, reference audio
and generated WAVs. Audited summary SHA-256:
`832b67d44483776d0659578a464a948cee1c42c0cdffd5be504604ce376ee047`.

## Pre-bucketing paired port measurements (2026-10-14)

Both revisions ran sequentially on the same allocated NVIDIA H200
(143,771 MiB, driver 595.71.05). Base is the original branch point
`fa4e6d1f1b4c962ecc223e1422f6e879f5ce1141`; the measured PR head is
`8becbc36f1fd78b1c3cd99d1b075201326b4e240`. This comparison does not use a
moving `main`. Both source trees were clean, and all clients and scorers ran
from the unchanged base checkout.

Server and client affinity was CPU 0–15: 16 physical cores on the GPU-local
NUMA node. Those CPUs were not exclusively reserved, and no 128 GiB host-memory
cap was imposed for this follow-up. The container image was
`docker.io/hongccc/sglang-omni@sha256:ebe4239e29a764ee3a2806385c061c5fd438a26f01458e503d3822dcba5790df`,
with SGLang upgraded to 0.5.21 as required by this checkout, Python 3.12,
PyTorch 2.13.0+cu130 and Transformers 5.12.1.

Each of the eight cohorts used the full 1,088-request SeedTTS EN split in
native order, concurrency 16, seed 42 and a 2,048-token cap. A fresh server
was started for each mode. Sixteen requests using the first reference were
excluded as warmup; normal reference caching and default KV-cache sizing
were retained. Weights were already downloaded. These are single runs,
not three-seed means or estimates with confidence intervals.

### Throughput and quality

WER is the full corpus edit-count ratio, including outliers. SIM is mean
WavLM cosine similarity multiplied by 100, not a percentage of passing
samples. Each row's base and head generated 1,088 successful requests, and
each WER/SIM calculation covered those same 1,088 IDs with zero failures or
skips. The corpus WER and SIM mean were independently recomputed from the
per-sample records.

| Model | Mode (C16) | Base requests/s | Head requests/s | Change | Base WER (%) | Head WER (%) | Base SIM | Head SIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6B | Non-streaming | 21.058 | 31.580 | +49.97% | 1.8923 | 1.3816 | 70.1584 | 70.0585 |
| 0.6B | Streaming | 27.073 | 35.671 | +31.76% | 1.5323 | 1.1639 | 70.3511 | 70.4384 |
| 1.7B | Non-streaming | 24.430 | 30.245 | +23.80% | 0.9462 | 1.1387 | 71.5595 | 71.4940 |
| 1.7B | Streaming | 32.223 | 32.459 | +0.73% | 1.1304 | 1.2560 | 71.5773 | 71.1164 |

The 1.7B WER increased by 0.1926 percentage points in non-streaming and
0.1256 points in streaming. Its streaming throughput difference is small,
and its SIM decreased by 0.4609 points. These observations do not establish
quality equivalence or a repeatable streaming speedup.

| Model | Mode | Base/head max-token hits | Base/head samples with WER >50% |
| --- | --- | ---: | ---: |
| 0.6B | Non-streaming | 1 / 3 | 2 / 2 |
| 0.6B | Streaming | 1 / 0 | 1 / 0 |
| 1.7B | Non-streaming | 0 / 0 | 0 / 1 |
| 1.7B | Streaming | 0 / 1 | 1 / 1 |

No capped or high-WER sample was dropped. SIM used batch size 1 for every
cohort with complete waveforms and the same scorer. The initial batch-8
attempt exhausted memory while padding long 0.6B baseline outputs, including
a 163.84-second sample; batch size was reduced uniformly without changing
audio, scoring models or thresholds.

### Startup and resident memory

Startup is process spawn to the first healthy HTTP response, measured before
the 16 excluded warmups. The second successful launch of each revision
(streaming mode) provides the warm-cache comparison: compiler caches were
warm, but each new process still captured its CUDA graphs.

| Model | Base startup (s) | Head startup (s) | Additional time (s) |
| --- | ---: | ---: | ---: |
| 0.6B | 179.079 | 248.180 | 69.101 |
| 1.7B | 138.569 | 306.147 | 167.577 |

The preceding non-streaming launches took 329.675/243.170 seconds
(base/head, 0.6B) and 154.100/349.664 seconds (1.7B). The first 0.6B base
launch had cold compiler caches, so its longer time is not evidence that
this PR reduces startup time. Each timing is one observation.

Resident memory below is serving-process `nvidia-smi used_gpu_memory` after
the 16 excluded warmup requests, rather than a peak or an isolated graph
allocation measurement.

| Model | Mode | Base resident memory (MiB) | Head resident memory (MiB) |
| --- | --- | ---: | ---: |
| 0.6B | Non-streaming | 126068 | 124386 |
| 0.6B | Streaming | 123430 | 124390 |
| 1.7B | Non-streaming | 126494 | 127638 |
| 1.7B | Streaming | 123514 | 124676 |

Default automatic KV-cache sizing was retained, so total resident memory
mixes weights, KV cache, graphs and allocator reservations. A lower total
does not imply that graph capture is free. These measurements do not
qualify the graph ladders for smaller GPUs.

### Reproduction

Pinned inputs:

| Input | Revision or SHA-256 |
| --- | --- |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | `5d83992436eae1d760afd27aff78a71d676296fc` |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | `fd4b254389122332181a7c3db7f27e918eec64e3` |
| `zhaochenyang20/seed-tts-eval-arrow` (EN) | `27f4c1adee83b5b29b7c4b375f6b976324bda308` |
| `Qwen/Qwen3-ASR-1.7B` | `7278e1e70fe206f11671096ffdd38061171dd6e5` |
| SIM `wavlm_large_finetune.pth` | `51f07e3b94d9e0262a6a675ef5a087be3dd09e8c62e9d886827f44f82fe7f94b` |
| SIM `wavlm_large.pt` | `6fb4b3c3e6aa567f0a997b30855859cb81528ee8078802af439f7b2da0bf100f` |

The unchanged benchmark loader pins the dataset revision above. Set
`MODEL_PATH` and `ASR_PATH` to the local snapshots of the listed revisions,
`MODEL_ID` to the checkpoint ID, `OUT` to a separate directory per cohort,
and `PORT` to an unused port. Use `qwen3_tts_0_6b.yaml` or
`qwen3_tts_1_7b.yaml` as `CONFIG`. Run the server from the selected base/head
checkout, with the following environment also applied to the clients:

```bash
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/work/hf-cache
export SEEDTTS_SIM_CACHE_DIR=/work/sim-cache
export SGLANG_OMNI_TORCH_COMPILE_DEFAULT=0
export SGLANG_OMNI_STRICT_PORT=1
unset OMP_NUM_THREADS
taskset -c 0-15 python -m sglang_omni.cli serve \
  --model-path "$MODEL_PATH" --host 127.0.0.1 --port "$PORT" \
  --config "examples/configs/$CONFIG"
```

From the base checkout in another shell, wait for `/health`, then use the
same mode for both warmup and measurement (`MODE=()` for non-streaming,
`MODE=(--stream)` for streaming; Bash array syntax):

```bash
MODE=()
COMMON=(--model "$MODEL_ID" --meta zhaochenyang20/seed-tts-eval-arrow
  --lang en --concurrency 16 --port "$PORT" --seed 42 --max-new-tokens 2048)
taskset -c 0-15 python -m benchmarks.eval.benchmark_tts_seedtts \
  "${COMMON[@]}" "${MODE[@]}" --generate-only --use-existing-server \
  --max-samples 1 --warmup 15 --output-dir "$OUT/excluded-warmup"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv,nounits
taskset -c 0-15 python -m benchmarks.eval.benchmark_tts_seedtts \
  "${COMMON[@]}" "${MODE[@]}" --generate-only --use-existing-server \
  --max-samples 1088 --warmup 0 --fingerprint --output-dir "$OUT"
```

Stop the owned TTS server before scoring its saved WAVs. Run both scorers
from the base checkout; ASR uses the default concurrency of 32. The runtime
SIM override below leaves the repository and waveform lengths unchanged:

```bash
taskset -c 0-15 python -m benchmarks.eval.benchmark_tts_seedtts \
  "${COMMON[@]}" "${MODE[@]}" --transcribe-only --skip-gpu-cleanup \
  --asr-model-path "$ASR_PATH" --server-timeout 3600 --output-dir "$OUT"
taskset -c 0-15 python -c \
  "import runpy; import benchmarks.tasks.tts as t; t.SPEAKER_SIMILARITY_BATCH_SIZE = 1; runpy.run_module('benchmarks.eval.benchmark_tts_seedtts', run_name='__main__')" \
  "${COMMON[@]}" "${MODE[@]}" --similarity-only --device cuda:0 \
  --output-dir "$OUT"
```

Saved evidence per cohort includes command arguments, source identities,
startup/memory snapshots, `speed_results.json`, `generated.json`, and
per-sample `wer_results.json` and `similarity_results.json`. This fixed
English workload on one H200 does not establish multilingual,
low-concurrency or smaller-device behavior.

## Historical 1.7B experiment identity

The work was developed by an autonomous research agent using
[RSI-Harness](https://github.com/OpenRSI-Foundation/OpenRSI-Index/tree/main/RSI-Harness),
from [OpenRSI proposal #154](https://github.com/OpenRSI-Foundation/OpenRSI-Index/discussions/154).

- Initial source: `1523543056bf334bc2f4d69db23aad0a002cedf2`.
- Research run: `51d85322a9484e83bcf8242a5c020dad`, submission `agent-6`.
- Evaluated source inventory SHA-256:
  `228108a4b3e11d9e81ecfd109cb504921e732a711f98e77717f070c5f9842279`.
- Checkpoint: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, revision
  `fd4b254389122332181a7c3db7f27e918eec64e3`.
- Dataset: full SeedTTS EN, 1,088 requests, revision
  `27f4c1adee83b5b29b7c4b375f6b976324bda308`, in native order.
- One NVIDIA H200, 16 physical CPU cores on one NUMA node, 128 GiB host
  memory. SGLang 0.5.20, PyTorch 2.13.0+cu130.
- Non-streaming and streaming at concurrency 16, each using seeds 42/43/44;
  16 TTS warmups per run excluded from scoring. Native engine admission and
  decode graph limits remain 64/64; the maximum token cap is 2,048.
- Stock diagnostic ASR followed by independent original-source authoritative
  Qwen3-ASR-1.7B on the generated WAVs, concurrency 32 and 64 ASR warmups.

## Historical 1.7B complete results

Values are arithmetic means of all three per-run summaries, including latency
percentiles. They are not percentiles pooled across runs. Baseline refers to
the initial pristine version, rather than the preceding accepted candidate.

| Mode | Metric | Pristine | Submission 6 |
| --- | --- | ---: | ---: |
| Non-streaming C16 | Requests/s | 21.1670 | 26.7017 (+26.15%) |
| Non-streaming C16 | Mean full latency (s) | 0.7530 | 0.5950 |
| Non-streaming C16 | p99 full latency (s) | 1.5030 | 0.9927 |
| Non-streaming C16 | p99 RTF | 0.4231 | 0.1780 |
| Non-streaming C16 | Corpus WER (%) | 1.0243 | 1.0383 |
| Streaming C16 | Requests/s | 22.7613 | 24.1733 (+6.20%) |
| Streaming C16 | Mean first audio (ms) | 89.87 | 75.13 |
| Streaming C16 | p95 first audio (ms) | 127.00 | 114.57 |
| Streaming C16 | p99 first audio (ms) | 997.63 | 252.80 |
| Streaming C16 | Mean full latency (s) | 0.6997 | 0.6593 |
| Streaming C16 | p99 full latency (s) | 1.4773 | 1.0790 |
| Streaming C16 | p99 RTF | 0.3554 | 0.2006 |
| Streaming C16 | Corpus WER (%) | 1.0466 | 1.0438 |
| Streaming C16 | C50 (%) | 99.85 | 99.91 |

All prescribed throughput repetitions:

| Mode | Seed | Pristine requests/s | Submission 6 requests/s |
| --- | ---: | ---: | ---: |
| Non-streaming C16 | 42 | 21.246 | 26.707 |
| Non-streaming C16 | 43 | 21.116 | 26.774 |
| Non-streaming C16 | 44 | 21.139 | 26.624 |
| Streaming C16 | 42 | 23.035 | 24.313 |
| Streaming C16 | 43 | 22.565 | 24.190 |
| Streaming C16 | 44 | 22.684 | 24.017 |

The complete candidate passed all 27 prescribed quality/performance gates,
resource and cleanup checks. All 6,528 scored TTS requests and 96 TTS warmups
completed with no failures or skips. Maximum observed GPU memory was
135,553 MiB. The non-streaming WER increase of 0.0140 percentage points is
retained; WER does not establish perceptual or speaker equivalence.

## Evidence and limits

[Complete Judge output](https://github.com/OpenRSI-Foundation/sglang-omni-qwen3-tts-base-1.7b-runtime-d154/blob/main/logs/runs/51d85322a9484e83bcf8242a5c020dad/sglang-omni-qwen3-tts-base-1.7b-runtime/submissions/agent-6/test_output.txt),
[frozen accepted metrics](https://github.com/OpenRSI-Foundation/sglang-omni-qwen3-tts-base-1.7b-runtime-d154/blob/main/recovery/radix-20261010/latest/rolling-current.json)
and the
[research snapshot](https://github.com/OpenRSI-Foundation/sglang-omni-qwen3-tts-base-1.7b-runtime-d154/blob/main/recovery/radix-20261010/latest/workspace.tar.gz)
retain the original feedback, hypothesis history and source diff. The evidence
repository requires access; the tables above are self-contained.

The historical experiment used a fixed public English workload and one
allocation. The separate current-port comparison above does not provide
low-concurrency, multilingual or smaller-device serving qualification. Graph capture can
increase startup time and memory use. The port also retains newer upstream
tail-decode and device abstractions, so byte identity with the historical
candidate is not claimed. Current-port regression coverage and the exact
validation environment are reported in the pull request.

## Historical 0.6B submission 10

Research run: `62937fcd31464d01b46ce685e6ca78e3`. Evaluated source inventory SHA-256: `96b563e447413d24e88a8ee9eeb75834ca2e6f4d9608c6bf66450eb9a159b1a4`.

The 0.6B experiment used the same six-run workload and hardware envelope above. Its checkpoint was Qwen/Qwen3-TTS-12Hz-0.6B-Base. All 27 gates passed, with zero scored failures or skips.

| Mode | Metric | Pristine | submission 10 |
| --- | --- | ---: | ---: |
| Non-streaming C16 | Requests/s | 21.3553 | 30.8277 (+44.36%) |
| Non-streaming C16 | Audio seconds/s | 91.8010 | 134.2343 |
| Non-streaming C16 | Mean full latency (s) | 0.7453 | 0.5150 |
| Non-streaming C16 | p99 full latency (s) | 1.6123 | 0.8643 |
| Non-streaming C16 | p99 RTF | 0.4133 | 0.1525 |
| Non-streaming C16 | Corpus WER (%) | 1.2029 | 1.2420 |
| Streaming C16 | Requests/s | 22.5917 | 24.2120 (+7.17%) |
| Streaming C16 | Audio seconds/s | 98.1717 | 106.4243 |
| Streaming C16 | Mean first audio (ms) | 93.0667 | 74.0000 |
| Streaming C16 | p95 first audio (ms) | 121.0333 | 114.1667 |
| Streaming C16 | p99 first audio (ms) | 1370.3000 | 270.0000 |
| Streaming C16 | Mean full latency (s) | 0.7040 | 0.6573 |
| Streaming C16 | p99 full latency (s) | 1.8830 | 1.0883 |
| Streaming C16 | p99 RTF | 0.4273 | 0.1965 |
| Streaming C16 | Corpus WER (%) | 1.2308 | 1.2755 |
| Streaming C16 | C50 (%) | 99.6633 | 99.9400 |

All prescribed 0.6B throughput repetitions:

| Mode | Seed | Pristine requests/s | Submission 10 requests/s |
| --- | ---: | ---: | ---: |
| Non-streaming C16 | 42 | 21.709 | 31.666 |
| Non-streaming C16 | 43 | 21.014 | 31.319 |
| Non-streaming C16 | 44 | 21.343 | 29.498 |
| Streaming C16 | 42 | 22.899 | 25.217 |
| Streaming C16 | 43 | 22.759 | 24.068 |
| Streaming C16 | 44 | 22.117 | 23.351 |

[Frozen accepted metrics](https://github.com/OpenRSI-Foundation/sglang-omni-qwen3-tts-base-0.6b-runtime-d154/blob/main/recovery/radix-20261010/latest/rolling-current.json), [pristine baseline](https://github.com/OpenRSI-Foundation/sglang-omni-qwen3-tts-base-0.6b-runtime-d154/blob/main/recovery/radix-20261010/latest/baseline.json), [complete Judge output](https://github.com/OpenRSI-Foundation/sglang-omni-qwen3-tts-base-0.6b-runtime-d154/blob/main/logs/runs/62937fcd31464d01b46ce685e6ca78e3/sglang-omni-qwen3-tts-base-0.6b-runtime/submissions/agent-10/test_output.txt), and [research snapshot](https://github.com/OpenRSI-Foundation/sglang-omni-qwen3-tts-base-0.6b-runtime-d154/blob/main/recovery/radix-20261010/latest/workspace.tar.gz). The evidence repository requires access; the tables above are self-contained.

Submission 14 reached a higher non-streaming score but failed a streaming guard; the retained candidate is submission 10. Its static-input padding optimization is independently covered by [#2754](https://github.com/sgl-project/sglang-omni/pull/2754), so it is not duplicated here. The combined port also shares the 1.7B capture order and initial-cohort overlap changes. Consequently, neither historical result is a new measurement of this combined patch.
