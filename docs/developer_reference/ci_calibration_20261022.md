# ASR, TTS and Omni calibration: October 22, 2026

This report records repeated CI measurements at
[`e5606aa62bfbfa69c1bb6bbd6a0420499d2ff35f`](https://github.com/sgl-project/sglang-omni/commit/e5606aa62bfbfa69c1bb6bbd6a0420499d2ff35f),
the head of [PR #3029](https://github.com/sgl-project/sglang-omni/pull/3029).
The native calibration completed on October 22 at 19:28:29 UTC with at least
five accepted full observations for each of 104 stage keys. Full-duplex
benchmarks (FDB) were excluded.

**The accompanying thresholds are historical candidates only.** No executable
threshold constants change. The comparison base,
[`1e8dcd26f9b960a0fa8739f89d0f4ea68e77d195`](https://github.com/sgl-project/sglang-omni/commit/1e8dcd26f9b960a0fa8739f89d0f4ea68e77d195),
has changes to the shared scheduler, MOSS-TTS model runner, Qwen3-Omni codec
and Qwen3-TTS codec. It is not measurement-equivalent to the measured source.
These results do not establish current-main performance or justify applying
the candidates to it.

## Coverage

| Suite | Complete stages | Accepted toward five per stage | Accepted records | Numerically excluded records | All canonical records |
| --- | ---: | ---: | ---: | ---: | ---: |
| ASR | 17/17 | 85/85 | 88 | 19 | 107 |
| TTS | 40/40 | 200/200 | 200 | 42 | 242 |
| Omni | 47/47 | 235/235 | 240 | 20 | 260 |
| **Total** | **104/104** | **520/520** | **528** | **81** | **609** |

Eight stage keys have six accepted observations: three FunASR keys, two Qwen
MMMU keys and three Qwen VideoAMME Talker TP2 keys. All accepted observations
are retained when calculating each metric's worst value; counts in the
five-per-stage column are capped only for reporting coverage.

The accepted records represent 163 full pytest invocations (21 ASR, 50 TTS,
92 Omni). Of these, 32 exited with assertions against the old thresholds
(one TTS, 31 Omni) while satisfying the native metric-extraction and
completeness contract. Thus, complete calibration coverage does not mean
all old CI thresholds passed. Interrupted attempts and completed request
counts were not promoted into accepted full observations.

## Published data and interpretation

| File | Contents |
| --- | --- |
| [observations.csv](ci_calibration_20261022/observations.csv) | All 609 canonical stage observations, including metrics, sample counts, return codes, exclusion evidence, source revision, timestamps and seed policy. |
| [threshold-candidates.csv](ci_calibration_20261022/threshold-candidates.csv) | 265 read-only metric candidates with effective sample counts, target symbols, historical reference values and worst-value operations. |
| [attempts.csv](ci_calibration_20261022/attempts.csv) | 23 canonical native attempt archives, including contention discards and incomplete attempts. |
| [asset-revisions.csv](ci_calibration_20261022/asset-revisions.csv) | Exact recorded model and dataset revisions. |

An observation is accepted in this dataset when both `numerically_excluded`
and `block_discarded` are `False`. The `strict_cell` field preserves the native
audit notation. Multiple stage records can originate from one pytest
invocation, so rows are not independent executions. Missing optional fields
are not false values; in particular, 150 TTS records lacked an optional
observation-validity field under their native schema.

The `*_json` CSV columns contain JSON. Non-finite exclusion diagnostics are
encoded as strings such as `"inf"`. Metric values retain their original units:
WER fractions and percentage-valued metrics must not be interchanged.
The candidate `display_scale` is a report-display multiplier, not a conversion
for writing constants. `digits` is the native rounding precision.
Candidates are pre-slack references; existing assertion slack is not included.
There are 161 tighter, 37 equal and 67 looser candidates relative to the
measured revision, with no duplicate target symbols.

`original_record_sha256` and `original_archive_sha256` identify the original
native files retained outside this repository. These public CSVs are
projections, not byte-for-byte copies of those files. Their original hashes
cannot be recomputed from the projections. Raw logs, environment receipts,
native engine files and audio are not bundled here, so the repository alone
does not provide a complete execution replay or independently verify every
operational gate.

The coverage counts can be reproduced from the repository root with Python's
standard library:

```python
import csv
from collections import Counter

with open("docs/developer_reference/ci_calibration_20261022/observations.csv") as f:
    rows = list(csv.DictReader(f))
accepted = [
    row
    for row in rows
    if row["numerically_excluded"] == "False"
    and row["block_discarded"] == "False"
]
counts = Counter((row["suite"], row["stage_key"]) for row in accepted)
print(len(rows), len(accepted), len(counts), sum(min(n, 5) for n in counts.values()))
# 609 528 104 520
assert min(counts.values()) >= 5
```

For each candidate, take `max` or `min` (the `worst_op` column) over its
accepted stage's `metrics_json[metric]` values to reproduce `worst_raw`.
The frozen engine then derives `candidate_write_value` by `stage_group`:

- `wer`: round upward to four decimal places.
- `reliability`: round upward to an integer.
- `accuracy`, `diarization`, `similarity`, `utmos`: retain the raw worst value.
- Other groups: use Python's `round(worst_raw, digits)` unless that would
  tighten beyond the raw worst value (raise a minimum or lower a maximum);
  in that case, retain the raw value.

Do not apply `display_scale` to these calculations. This checks the published
calculation, not applicability to a different source revision.

## Exclusions and quality variability

The frozen engine applied its existing median-absolute-deviation and gap
policy: a scaled deviation above 3.5 and a nearest-neighbor gap of at least
20% of the remaining median, subject to resolution floors. Rejection applies
to every stage key in the affected invocation. It schedules two replacement
rounds per rejected round, with at most one full block restart for a
degenerate population. All excluded values are retained. Final review did not
use the fallback that keeps an unstable block with a warning.

These are numerical selection decisions, not independent proof of broken
executions. In particular, selection on correctness metrics can hide real
variation or defects.

| Scope | Excluded rounds / records | Trigger and surviving evidence |
| --- | --- | --- |
| ASR multi-speaker | Rounds 3 and 6, eight keys each: 16 | AISHELL4 delta-CER 0.598175 and 0.593740 versus remaining medians 0.192647 and 0.202903. |
| ASR FunASR | Round 1, three keys: 3 | English latency p95 0.414039 s and RTF p95 0.0992; native gaps 26% and 31%. Six observations per key survive. |
| TTS mixed serving | Rounds 1–5, 8 and 12, six keys each: 42 | Initial completion-token split 95/95/95/160/152; later token/audio shifts and REST-stream p95 0.857379 s versus median 0.681294 s. Surviving rounds: 6, 7, 9, 10, 11. |
| Qwen MMMU | Round 3, two keys: 2 | Accuracy 70% versus remaining median 55%. Surviving accuracies: 58%, 54%, 56%, 54%, 62%, 54%. Excluding the higher value changes the quality distribution. |
| Qwen VideoAMME Talker TP2 | Round 2, three keys: 3 | WER fraction 0.0245232 versus remaining median 0.00815223. Replacement WER fractions: 0.0135593 and 0.00526316. |
| Qwen VideoMME Talker | Original rounds 1–5, three keys each: 15 | Counts of samples above 50% WER were [5, 3, 0, 3, 0], triggering the degenerate-block rule. Replacement rounds 6–10 each completed 20/20, accuracy 60%, with counts [1, 1, 1, 0, 1]. |

The 23 native attempt archives contain nine CPU-contention discards, five
startup/crash classifications and nine archives without a completed native
record. Additional manual stop and diagnostic archives are outside that
count. Four of the five startup/crash classifications were actually complete
quality-assertion runs caught by a generic live crash detector:

- Qwen MMMU round 2/attempt 1 (60%), round 4/attempt 3 (54%) and round 5/attempt 1 (56%).
- Qwen MMSU Talker round 5/attempt 1 (40/40 samples, accuracy 65%, WER fraction 0.0281843).

Those four runs remain excluded; no classifier patch or manual promotion was
used. Timing-dependent classification of the same assertion type creates
another selection limitation. The remaining startup failure was a CosyVoice
address-in-use error.

Accepted old-threshold failures include Qwen VideoAMME accuracies
66%, 66%, 64%, 64%, 64% against the old 68% reference; MiniCPM-O VideoAMME
Talker WER fraction 0.0080645161 against 0.0051; and Qwen VideoMME Talker WER
assertions. These outcomes warrant review before any quality threshold is
changed.

## Speed dispersion

Five of 165 speed metric series have `(max - min) / abs(median) >= 20%`.
This is a descriptive flag, not another acceptance gate. Ordinary tails
remain in the data and candidates.

| Stage / metric | Min | Median | Max | Relative range |
| --- | ---: | ---: | ---: | ---: |
| Qwen ASR English latency p95 (s) | 0.149768 | 0.155565 | 0.195011 | 29.08% |
| Qwen ASR English RTF p95 | 0.0349 | 0.0363 | 0.0429 | 22.04% |
| Whisper ASR English latency p95 (s) | 0.354422 | 0.389745 | 0.437727 | 21.37% |
| Whisper ASR English RTF p95 | 0.0797 | 0.0834 | 0.0972 | 20.98% |
| Qwen VideoAMME Talker TP2 mean RTF | 2.5462 | 2.8093 | 3.2189 | 23.95% |

## Environment and continuity

| Component | Recorded identity |
| --- | --- |
| Measured source | `e5606aa62bfbfa69c1bb6bbd6a0420499d2ff35f` |
| Calibration tools base | `63deaa330b6fc9bed758ed5362fa85ef53b31612` |
| Frozen native engine | v0.8.0; SHA256 `8127e28a29643de2e17f3eb58ae7ffc00cee976ed1ae552fe0faa26f34b98d81` |
| Image | `docker.io/hongccc/sglang-omni@sha256:06ed306707bcf3d9e9beb5ba7ce66f09a666cac11c23455ece50c34199c0e985` |
| Hardware | 2 × H100 80 GB, NV18; driver/NVML 580.126.20 |
| Python / PyTorch / SGLang | 3.12.15 / 2.14.1+cu130 / 0.5.22 |
| Other dependencies | transformers 5.19.0; flashinfer-python 0.7.1; numpy 2.3.5; nvidia-ml-py 13.615.71; pytest 9.1.1 |
| Dependency freeze SHA256 | `92869ceb221b3615922ac6e60e121452b564b0b2f04cdad5569d18f3be098ab7` |
| Seed policy | Test-default/unset; calibration and PYTHONHASH seeds null |

All 21 required exact-pin/import checks passed. Optional metadata conflicts
remained (14 in the virtual environment versus 15 in the image); this is not
a claim of full `pip check` success. Model and dataset revisions are listed
in the accompanying CSV.

The runtime was lost after an operational stop and restored before completing
the remaining Omni observations. The restored immutable image, measured
source, dependency freeze and nine Omni assets matched; 2,889 restored native
files were hash-verified. The same GPU pair and CPU lane were reused, ten
quiet CPU windows and the original precheck passed, and execution resumed at
17:59:42 UTC on October 22. Restoring provenance does not recover the lost
audio or remove the isolation limitations below.

### CPU isolation limits

The lane was CPUs 48–63 and 112–127: 32 logical / 16 physical CPUs on NUMA 1.
It was not exclusively reserved. The rootless container had no enforceable
cpuset or CPU quota; it used `taskset`, inherited affinity masks and
five-second telemetry. The unchanged contention gate rejected three
consecutive windows above two foreign busy cores. Recovery required ten
consecutive five-second windows below two busy cores and the original
precheck. A single peak above two cores was not itself grounds for rejection.

A watchdog observed an ASR-evaluation startup thread outside the lane. A
separate startup-only syscall trace reproduced NVML 580.126.20 probing CPUs
outside the inherited mask and then restoring it. The original process was
not traced, and the diagnostic contributed no measured requests. This does
not establish steady-state inference escape or invalidate all prior data;
five-second samples also cannot prove uninterrupted affinity.

**Continuous hard CPU isolation remains unqualified.** This environment is
not an exact reproduction of CI's conservative 28-CPU lane and autoscaler
isolation.

### Media preservation limits

The earlier compact collection preserved 7,148 files plus 295 router TOMLs.
However, **23,690 original WAV paths, totaling 9,162,637,240 bytes (8.53 GiB),
were not copied and their remote originals are now absent**. Another 31
original non-media files outside the compact collection were not copied.
This prevents complete retrospective listening or re-evaluation of the
original audio.

The final restored-runtime inventory contained 4,093 regular files totaling
698,512,873 bytes. All were copied and SHA256-verified, including 340 new WAV
files totaling 215,574,920 bytes. None of that final inventory was left
uncopied. New audio does not replace the lost original audio. No generated
video appeared in these inventories; symlinks and model/dataset caches were
outside their scope. Path counts can include archived copies, and the two
inventory counts must not be added as a unique-file total.

## Requirements before threshold changes

Review the correctness distributions, classifier-dependent exclusions,
variable speed series and CPU-isolation limits. Establish source and
environment comparability for the intended target; the comparison base above
does not satisfy that condition. Any new measurement method or stronger
isolation must be reported as a separate cohort. Retain the existing
thresholds until that evidence and the proposed changes have been reviewed.
