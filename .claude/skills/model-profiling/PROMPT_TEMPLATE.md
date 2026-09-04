# Agent prompt template

Fill every `{PLACEHOLDER}` before sending this as the background agent's
prompt. Do not paste the methodology's prose into the filled prompt —
reference section names only, the agent has its own access to the repo.

```
You are profiling {MODEL_NAME} following the methodology in
.claude/skills/model-profiling/METHODOLOGY.md. Read the current version of
that doc first — it gets corrected as new models are profiled, do not rely
on any summary of it from elsewhere.

Benchmark entrypoint: {BENCHMARK_ENTRYPOINT}
Candidate GPU pool: {GPU_POOL}
Output doc (new or resume): {OUTPUT_DOC_PATH}
  Save every raw artifact a reported number comes from (py-spy outputs,
  benchmark logs, nvidia-smi captures) in the same directory as this doc,
  and cite the artifact path plus the exact command next to the number.
Methodology doc to fold generalizable findings into: {METHODOLOGY_DOC_PATH}

Scope for this run: {LAYERS_IN_SCOPE}
  (e.g. "discovery only — Layer 1 first, then branch per §2; no confirmed
  Layer 4 hypothesis yet" or "Layer 4 on <specific hypothesis>, then Layer 5
  — discovery was already approved in a prior message")

Do:
1. Follow .claude/skills/model-profiling/METHODOLOGY.md §1's pre-check checklist yourself —
   confirm the GPU pool is actually free. If a process from a prior run is
   still holding a GPU you need, do NOT kill it yourself (see "Do not"
   below) — report its PID, command line, and how you verified it with
   `ps`, and stop for explicit confirmation before anyone kills it. Once
   the pool is confirmed clear, record a baseline environment fingerprint
   per .claude/skills/model-profiling/METHODOLOGY.md §3 item 1 — including
   the repo's own commit SHA and dirty status.
1a. If {OUTPUT_DOC_PATH} already exists (resume run): before reusing any
   prior layer's evidence, compare its recorded baseline fingerprint
   against the current environment (commit SHA, GPU model, dependency
   pins). If anything drifted, report the drift and which prior layers it
   may invalidate — do not silently build on stale evidence.
2. Layer 1 first, always. Then branch per
   .claude/skills/model-profiling/METHODOLOGY.md §2's own routing rule —
   1 -> 3 -> 2 -> 4 -> 5 is NOT a fixed sequence:
   - Busy ratio well below saturation (CPU/orchestration-bound): go to
     Layer 2 next, then Layer 3 (concurrency scan), then Layer 4 (A/B on
     what Layer 2 found), then Layer 5 only if Layer 4 actually changed a
     default.
   - Busy ratio already near saturation (GPU-kernel-bound): per the
     methodology, Layer 2-4's CPU-side digging is low-ROI — skip it, note
     why in the report, and flag kernel-level analysis as the candidate
     next step for a follow-up scope; do not start kernel-level analysis
     on your own. Only run Layer 5 if you actually changed something.
   Record evidence at each layer you do enter rather than jumping straight
   to a conclusion.
2a. If your scope is discovery only (no confirmed Layer 4 hypothesis yet):
   stop once Layer 2 (or the saturation branch above) is done. Report the
   candidate hypothesis/hypotheses — do not pick one and run Layer 4 on
   your own judgment. Wait for a follow-up message confirming which one (if
   any) to A/B before touching Layer 4. On the saturation branch there is
   no Layer 2 hypothesis to report — report the saturation evidence itself
   and state explicitly that the remaining option is kernel-level analysis,
   then stop and wait the same way.
3. Before trusting any aggregate throughput/latency number, check the max
   latency and the raw per-sample distribution, not just percentiles/means —
   a single pathological sample can dominate wall-clock and look like a
   regression that isn't real.
4. If nvidia-smi utilization sampling looks low, check whether the sampling
   window covers ramp-up — a short window can read far below true
   steady-state.
4a. If the model has distinct short-form and long-form workload shapes (check
   for a second eval entrypoint, e.g. a `*_longform.py` sibling of the main
   benchmark script), run Layer 1 and Layer 3 on each shape separately —
   see .claude/skills/model-profiling/METHODOLOGY.md's Layer 3 dataset-shape
   note. If only
   one shape is actually in scope for this model, say so explicitly in the
   output doc rather than silently testing one and calling it complete.
5. Before starting a server and running a benchmark client separately,
   check whether the benchmark script has a flag to attach to an existing
   server (e.g. `--use-existing-server`) instead of launching its own.
6. If you observe GPU memory/utilization you can't attribute to any PID you
   control, say so explicitly and try `nvidia-smi --query-compute-apps` —
   do not report a specific PID as the cause unless you actually verified it
   with `ps`. Treat "not found" as inconclusive, not as proof of absence.
7. Write {OUTPUT_DOC_PATH} with the report shape defined by
   .claude/skills/model-profiling/METHODOLOGY.md §3 item 8: baseline
   environment (including commit SHA), one section per layer you entered
   (method used, what was found, evidence strength graded against §3 item
   8's rubric — not by feel — with negative results stated explicitly), a
   findings-evidence-recommendation table, and a cleanup section. Every
   number must cite the command and raw-artifact path that produced it,
   per the Output doc note above. This path is gitignored on purpose —
   profiling results are working artifacts, not maintained docs. Do not
   move it under docs/ and do not commit it; the human files the durable
   record from your report.
8. Fold only the cross-model, generalizable lessons (not this model's
   specific numbers) into {METHODOLOGY_DOC_PATH}.
9. Clean up every process and GPU allocation you started before finishing.
   Confirm cleanup with nvidia-smi / ps, don't just assume your own
   teardown code worked.
10. Report back: what layers you completed, what you found, what's still
    open, and explicit confirmation that cleanup succeeded. If you get
    stuck (e.g. a server won't finish booting), diagnose via logs — don't
    silently poll forever — and report the partial state clearly if you
    can't recover.

Do not:
- Kill any process you didn't start yourself, including orphans left over
  from a prior run — report and wait for confirmation instead.
- Guess at or report a specific PID/process attribution you haven't
  verified with `ps`/`nvidia-smi --query-compute-apps`.
- Put this model's specific numbers into the pure methodology doc.
- Decide on your own which doc an ambiguous generalizable-vs-model-specific
  finding belongs in — flag it in your report instead of picking one.
- Run Layer 4 right after Layer 2 on your own judgment when your scope was
  discovery-only — stop and report the candidate hypothesis, then wait for
  a follow-up message before A/B-ing anything.
```

## Placeholder reference

| Placeholder | Meaning |
|---|---|
| `{MODEL_NAME}` | Directory name under `sglang_omni/models/` |
| `{BENCHMARK_ENTRYPOINT}` | Path(s) to the benchmark script/config to drive the run — list more than one if the model has separate short-form/long-form eval entrypoints |
| `{GPU_POOL}` | Candidate free GPU id(s), from the skill's own pre-check |
| `{OUTPUT_DOC_PATH}` | `.profiling-runs/<model>/profile.md` — gitignored working artifact; raw evidence artifacts live in the same `.profiling-runs/<model>/` directory. The durable record of results is a sub-issue under the tracking issue (see SKILL.md "Result tracking"), not a committed doc |
| `{METHODOLOGY_DOC_PATH}` | `.claude/skills/model-profiling/METHODOLOGY.md` |
| `{LAYERS_IN_SCOPE}` | Which of Layer 1/2/3/4/5 this run covers. A fresh run without an existing `.profiling-runs/<model>/profile.md` must scope to discovery only (Layer 1 first, then branch per METHODOLOGY.md §2's routing rule) — Layer 4/5 get filled in and confirmed separately once Layer 2 surfaces a concrete hypothesis. A resume run may cover Layer 4/5 directly if the hypothesis was already confirmed. |
