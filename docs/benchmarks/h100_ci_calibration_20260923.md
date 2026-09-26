# H100 CI calibration: September 23, 2026

This change uses the completed full calibration at
`73e443bbacd5972235003ccd382d762529bb5bc4`. It updates **118 reference
fields**: 47 ASR and 71 Omni. It does not repeat calibration on a newer main
revision, change gate formulas, change sample counts, or weaken disputed quality
gates. TTS reference updates are deferred because its measured environment differs
from the production preparation path.

The code branch starts from main at
`6307e2fac13739bd77aa93f3696267906e411746`; the measurements remain tied to
`73e443bbacd5972235003ccd382d762529bb5bc4`.

Sampling and the evidence/recovery audit are complete. The 118 selected reference
updates are accepted for this historical run; that does not imply every quality
observation passed. The
[complete candidate inventory](h100_ci_calibration_20260923.csv) records all 255
mappings, original/candidate references, actual gates, and whether this patch
includes each field. Candidate references in deferred rows are not applied.

## Measurement scope

| Suite | Complete stages | Mapped metrics | Final retained stage observations |
| --- | ---: | ---: | ---: |
| ASR | 17/17 | 80 | 85 |
| TTS | 34/34 | 72 | 170 |
| Omni | 47/47 | 103 | 252 |
| Total | 98/98 | 255 | 507 |

Each stage has at least five valid observations using its complete configured CI
sample count. Stages can share one pytest invocation. The full scope contains 29
execution units; it is not a claim to run every item in the original benchmark
datasets or every correctness-only CI module. New presets introduced after the
measured commit are outside this run.

- Run identity: `20260922T191610Z_73e443b`.
- Two H100 80 GB GPUs; three other two-GPU CI lanes remained available.
- 28 logical CPUs with the explicitly selected process-affinity contract; this
  is not cgroup cpuset confinement.
- Image: `docker.io/hongccc/sglang-omni@sha256:ebe4239e29a764ee3a2806385c061c5fd438a26f01458e503d3822dcba5790df`.
- ASR and Omni used the original base environment; TTS used the separate
  dependency-recovery environment described below. `NCCL_NVLS_ENABLE=0`.
- Native run, status, strict audit, report and read-only apply-plan completed.
  All three final native readiness checks reported no missing stages.
- Formal start: September 22, 20:03:49 UTC. Completion: September 23,
  14:31:03 UTC. Elapsed time: 18 h 27 min, including TTS recovery and resampling.
- Container exit code 0; reservation released with no cleanup errors.

The native engine SHA-256 is
`93fc74f464ae1d0a84207d7c999c018ee61dcc146b6bcb247a8fa298a29cf1ea`.
The patch copies the engine's raw pre-slack candidate values exactly for the
selected fields. It does not replace them with derived gate values or introduce
a different estimator. Native excluded rounds remain in the evidence.

## Included and deferred references

The conservative selection excludes quality-gate loosening, candidate gates that
reject observed excluded rounds, and unresolved environment-applicability changes.
Unchanged candidate values require no source edit. The numerical failure review
uses the actual frozen CI slack and rounding formulas, not reference-only
comparisons.

### ASR

ASR completed 29 pytest attempts, producing 157 stage observations before native
selection. Multi-speaker sampling ran 14 rounds and retained rounds 10–14 after a
block reset; the other three execution units retained all five rounds.

The review replayed 406 multi-speaker quality/count comparisons and the 12
SeedTTS WER mappings over five rounds. Those observed values satisfy the proposed
quality gates. Quality-gate loosening is nevertheless deferred in this patch.
Two delta-CER references are report-only in the measured source and are left
unchanged. Long90 attribution remains a reported limitation, not a newly enabled
gate.

The first multi-speaker round has four speed measurements that fail candidate
gates: throughput, mean latency, p95 latency and mean RTF. Their references are
left unchanged. The first-batch delay has not been causally attributed to JIT or
contention. Excluding it statistically does not establish first-run equivalence.

### Omni, including MiniCPM-o 4.5

The raw JSON/JUnit subset covers 100 immutable attempts and all 47 stages. Its
103 proposed values match the native apply-plan. Across all retained and excluded
observations, 589 gate comparisons found three quality failures and one speed
failure against candidates. Those four mappings are deferred, as are ten quality
gate loosenings. Of the remaining mappings, 71 values change in this patch;
unchanged values are left alone.

MiniCPM-o 4.5 accounts for 24 stages and 52 mappings across 51 attempts. All 184
speed comparisons for its 32 speed mappings pass both original and candidate
gates. Maximum observed `(max-min)/median` across these speed metrics is 10.26%;
this is repeated-run evidence, not a tail-reliability guarantee.

Its quality limitations remain explicit:

- MMMU talker repeats the same 20 questions five times. There are 37/100 repeated
  sample observations with WER above 50%; unfiltered corpus WER ranges from
  **38.90% to 48.38%**, while the filtered subset ranges from 16.39% to 25.96%.
  Both old and candidate gates permit up to eight high-WER samples per round.
  A passing filtered gate is not evidence of low overall speech error.
- The WER reference is generated thinker text and the hypothesis is ASR of
  generated speech. The relative contributions of synthesis, ASR and text
  normalization have not been established. Missing duration defaults in the WER
  adapter must not be interpreted as empty audio.
- TTS round 5 has speaker similarity 42.770416, below the original 42.868747
  floor and candidate 43.415429 floor. It was excluded together with its WER
  round; the similarity reference is not changed.
- TTS round 7 has two samples above 50% WER (100% and 60%). The proposed tail
  allowance increase from one to two is not applied.
- VideoAMME talker round 3 has WER 0.806452%, above both old and candidate 0.51%
  gates. It is preserved and the gate is not changed. MiniCPM runs DP2 in this
  shared test file; the Qwen-specific TP2 assertion is intentionally skipped.
- Accuracy decreases and other quality-gate loosenings remain deferred. The
  small fixed sample sets do not prove persistent regression against a
  controlled historical baseline.

For Qwen3-Omni, the excluded TTS WER observation passes the old gate but fails
the proposed tighter gate, so that update is deferred. The VideoAMME talker
candidate WER gate increase from 1.41% to 4.80% is also deferred.

### TTS

All 34 stages completed in a separate environment after CosyVoice source
prerequisites were found missing. Fixed CosyVoice/Matcha revisions were supplied,
setuptools was pinned to 79.0.1, and the conflicting optional ModelOpt package was
excluded from that isolated environment. Original ASR/Omni environments were not
modified. Existing image metadata issues remain documented; this was not a clean
pip-check result.

The fresh TTS cohort is independent of the archived partial cohort. It completed
44 attempts and 164 testcases: seven numerical failures, no errors or skips.
Those failures include Higgs/MOSS speaker-similarity floors and one serving
latency tail. Serving retained rounds 10–14 after exclusions involving both
latency and generated output length. These are not all attributable to CPU noise.
The earlier Qwen3-TTS 163.84-second repetitive output remains recorded separately.

All 72 TTS mappings are reported in the inventory, but none is applied here.
Production environment alignment and quality disposition require separate work.

## Validation and limitations

Completed validation:

- Native full-scope status and strict readiness: ASR 17/17, TTS 34/34, Omni 47/47.
- Original native observations and numerical assertion failures retained;
  infrastructure failures are not counted as complete observations.
- Raw TTS and Omni JSON/JUnit subsets independently checked against SHA-256
  manifests; ASR has separate raw, retention and CPU review records.
- Omni JUnit: 193 passed, 11 numerical failures, 16 expected model-specific skips,
  zero errors. TTS fixed structural/canonical checks reviewed separately.
- Changed-file pre-commit hooks pass, including formatting and Python AST checks.
  The all-files run passed its non-Rust checks after formatting one changed file;
  the Rust formatting hook could not run because local cargo is unavailable. No
  Rust files are changed. GitHub lint, documentation build and test-layout checks
  passed on the initial PR revision.
- Selected patch values match all 118 corresponding native candidate values;
  source AST is unchanged after masking numeric literals. Deferred and fixed
  fields remain at their original values.
- Complete remote evidence inventory: 46,057 entries and 16,952,013,586 regular
  bytes. Before/after inventories match. Current-cohort archive checks cover
  173 archives and 650 raw stage observations, with no missing, partial or
  unreferenced archives. These observations include native excluded rounds;
  the final retained count remains 507.
- Four CPU-monitor epochs have no observed identity or affinity anomalies.
  Invalid samples remain recorded: ASR 122/3,265, archived partial TTS 79/1,259,
  fresh TTS 199/4,637 and Omni 199/3,483. Process/thread read gaps prevent a
  claim of continuous observation or zero foreign load.
- Recovery partition and bootstrap checks pass: 3,228 original archive entries,
  57 stop-snapshot entries, and three bootstrap archives containing 24, 84 and
  48 entries all match their manifests. Original ASR observations are preserved;
  fresh TTS and Omni observations do not reference the discarded partial cohort.
- Native status, strict audit and apply-plan agree on the measured commit,
  tool identity and all 98 stages, each with at least five clean full-sample
  observations. Terminal records confirm idle calibration GPUs, released lane
  reservation and no cleanup errors.

The acceptance covers the 118 selected historical references only. TTS environment
alignment and the disclosed quality issues remain separate work. The full media
backup download is separate from the completed remote hash and archive audit.
Full-scope sampling is not described as all pytest jobs passing, and GPU CI on
the newer main revision is a separate validation.

No quality anomaly or excluded observation has been removed to obtain a green
report. Later code changes can be calibrated in a separate run; they are not
claimed to have been measured by this historical calibration.
