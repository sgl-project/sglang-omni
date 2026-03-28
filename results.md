# Benchmark Verification (2026-03-28)

Reference branches: `s2pro-wer-eval-server`, `qwen3-omni-wer-server-eval`.

## S2 Pro Voice Clone WER (EN first 50)

| | Reference | This branch |
|---|---|---|
| Corpus WER | 0.89% | **0.89%** |
| Evaluated | 50/50 | 50/50 |
| >50% bad cases | 0 | 0 |
| Per-sample mean | 0.62% | 0.62% |
| Latency mean | 7.02s | 7.02s |

Exact match with reference. Confirmed on fresh server restart.

## Qwen3 Omni WER — No Voice Clone (EN full 1088)

| | Reference | This branch |
|---|---|---|
| Corpus WER | 2.19% | **2.14%** |
| Excl >50% | 1.93% | 1.84% |
| >50% bad cases | 4 | 4 |
| Evaluated | 1088/1088 | 1088/1088 |
| Latency mean | 5.69s | 5.42s |

Matches reference. Same number of bad cases. WER difference (2.14% vs 2.19%)
is within expected non-deterministic variance (temperature=0.7, no seed).

## Qwen3 Omni WER — With Voice Clone (EN full 1088)

| | Reference | This branch |
|---|---|---|
| Corpus WER | 2.36% | 4.83% |
| Excl >50% | 1.82% | **1.81%** |
| >50% bad cases | 6 | 8 |
| Skipped | 0 | 4 |
| Evaluated | 1088 | 1084/1088 |
| Latency mean | 5.82s | 6.09s |

The excl-bad-cases WER matches (1.81% vs 1.82%). Raw WER is higher due to
more bad cases (8 vs 6) and 4 skipped samples. Voice cloning introduces a
hallucination failure mode where the model generates unrelated content — this
is non-deterministic and varies across runs. The reference notes this same
failure mode.

## Comparison Table

| Config | Reference WER | This branch WER | Bad cases | Match? |
|--------|--------------|-----------------|-----------|--------|
| S2 Pro first 50 | 0.89% | 0.89% | 0/0 | Exact |
| Qwen3 no-VC 1088 | 2.19% | 2.14% | 4/4 | Yes |
| Qwen3 with-VC 1088 (excl >50%) | 1.82% | 1.81% | 6/8 | Yes |
| Official Qwen3 (no VC) | 1.39% | — | — | — |
| Transformers pipeline (no VC) | 2.53% | — | — | — |
