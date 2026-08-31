# Agent prompt template

Fill every `{PLACEHOLDER}` before sending this as the background agent's
prompt. Do not paste the methodology's prose into the filled prompt —
reference section names only, the agent has its own access to the repo.

```
You are profiling {MODEL_NAME} following the methodology in
docs/profiling_methodology.md (and its English mirror
docs/profiling_methodology_en.md). Read the current version of that doc
first — it gets corrected as new models are profiled, do not rely on any
summary of it from elsewhere.

Benchmark entrypoint: {BENCHMARK_ENTRYPOINT}
Candidate GPU pool: {GPU_POOL}
Output doc (new or resume): {OUTPUT_DOC_PATH}
Methodology docs to fold generalizable findings into: {METHODOLOGY_DOC_PATHS}

Scope for this run: {LAYERS_IN_SCOPE}
  (e.g. "Layer 1, 3, 2, 4 fresh; Layer 5 only if Layer 4 changes a default")

Do:
1. Follow docs/profiling_methodology.md §1's pre-check checklist yourself —
   confirm the GPU pool is actually free (report any busy PID you can't
   attribute via `ps`, don't assume it's safe to ignore), record a baseline
   environment fingerprint, clean up any orphaned processes from prior runs.
2. Work through the layers in scope, in the methodology's documented order
   (Layer 1 -> Layer 3 -> Layer 2 -> Layer 4 -> Layer 5), stopping to record
   evidence at each layer rather than jumping straight to a conclusion.
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
   see docs/profiling_methodology.md's Layer 3 dataset-shape note. If only
   one shape is actually in scope for this model, say so explicitly in the
   output doc rather than silently testing one and calling it complete.
5. Before starting a server and running a benchmark client separately,
   check whether the benchmark script has a flag to attach to an existing
   server (e.g. `--use-existing-server`) instead of launching its own.
6. If you observe GPU memory/utilization you can't attribute to any PID you
   control, say so explicitly and try `nvidia-smi --query-compute-apps` —
   do not report a specific PID as the cause unless you actually verified it
   with `ps`. Treat "not found" as inconclusive, not as proof of absence.
7. Write {OUTPUT_DOC_PATH} following the section shape of the most complete
   existing example under docs/developer_reference/ (check which one is most
   complete — currently moss_transcribe_diarize_profile.md): baseline
   environment, per-layer sections with evidence, a
   findings-evidence-recommendation table, and a cleanup section.
8. Fold only the cross-model, generalizable lessons (not this model's
   specific numbers) into BOTH {METHODOLOGY_DOC_PATHS}, kept in sync with
   each other.
9. Clean up every process and GPU allocation you started before finishing.
   Confirm cleanup with nvidia-smi / ps, don't just assume your own
   teardown code worked.
10. Report back: what layers you completed, what you found, what's still
    open, and explicit confirmation that cleanup succeeded. If you get
    stuck (e.g. a server won't finish booting), diagnose via logs — don't
    silently poll forever — and report the partial state clearly if you
    can't recover.

Do not:
- Kill any process you didn't start yourself.
- Guess at or report a specific PID/process attribution you haven't
  verified with `ps`/`nvidia-smi --query-compute-apps`.
- Put this model's specific numbers into the pure methodology docs.
- Decide on your own which doc an ambiguous generalizable-vs-model-specific
  finding belongs in — flag it in your report instead of picking one.
```

## Placeholder reference

| Placeholder | Meaning |
|---|---|
| `{MODEL_NAME}` | Directory name under `sglang_omni/models/` |
| `{BENCHMARK_ENTRYPOINT}` | Path(s) to the benchmark script/config to drive the run — list more than one if the model has separate short-form/long-form eval entrypoints |
| `{GPU_POOL}` | Candidate free GPU id(s), from the skill's own pre-check |
| `{OUTPUT_DOC_PATH}` | `docs/developer_reference/<model>_profile.md` |
| `{METHODOLOGY_DOC_PATHS}` | `docs/profiling_methodology.md` and `docs/profiling_methodology_en.md` |
| `{LAYERS_IN_SCOPE}` | Which of Layer 1/2/3/4/5 this run covers (fresh vs. resume) |
