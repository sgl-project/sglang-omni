# Calibration operations

## GPU groups and layouts

Pin every calibration process with `TUNE_GPU_INCLUDE`. Use a separate run
directory per process. Do not point two processes at one plan.

Layouts (see `SKILL.md` for Mode A / B / C):

```bash
# Mode A — one group
TUNE_GPU_INCLUDE=0,1 python tune.py --model omni run --stages ALL --repeats 5 ...

# Mode C — N independent full calibrations (preferred multi-GPU default)
TUNE_GPU_INCLUDE=0,1 python tune.py ... --output-dir "$RUN_G01"
TUNE_GPU_INCLUDE=2,3 python tune.py ... --output-dir "$RUN_G23"
TUNE_GPU_INCLUDE=4,5 python tune.py ... --output-dir "$RUN_G45"

# Mode B — shared scope, disjoint stages, then merge-runs
TUNE_GPU_INCLUDE=0,1 python tune.py ... --stages <A> --output-dir "$RUN_A"
TUNE_GPU_INCLUDE=2,3 python tune.py ... --stages <B> --output-dir "$RUN_B"
```

Do not hard-code a fixed “two groups share ASR/TTS/Omni” partition table in
agent plans. Choose Mode C unless the user explicitly wants one combined
worst-of-five via Mode B.

## Concurrent isolation (required for any multi-group run)

These rules apply whether groups run Mode B or Mode C.

1. **Disjoint GPUs.** Include sets must not overlap. Respect `TUNE_GPU_EXCLUDE`
   for host-reserved devices; never launch on or clean those GPUs.
2. **Per-group cache root.** Give each group a distinct `XDG_CACHE_HOME` and/or
   `HOME` (or equivalent `OMNI_CI_HOME` partition). FlashInfer cleanup wipes
   only this job’s first cache dir; wiping every candidate path races live
   workers on other groups.
3. **Scoped cleanup only.** Every cleanup path must pass physical GPU ids via
   `CUDA_VISIBLE_DEVICES`. See Cleanup below.
4. **No interactive shell pollution.** Bootstrap from
   `.github/scripts/ci_env.sh` (and explicit exports). Do **not** `source`
   `~/.zshrc` / `~/.bashrc` for calibration — they often force
   `CUDA_VISIBLE_DEVICES` and break multi-group pinning.
5. **Secrets.** Keep `HF_TOKEN` in a dedicated file with mode `600`, or the
   process environment. Do not rely on interactive shell state.
6. **Auditable cleanup.** Keep cleanup stdout/stderr visible so cross-group
   mis-kills are diagnosable.

## Cleanup

`tune.py` passes the actual physical GPU indices from the pytest launch to
`delete_gpu_process.sh --kill-orphans`. Invariants:

- `CUDA_VISIBLE_DEVICES` selects **physical** GPU indices to clean.
- The script then **unsets** CVD before calling `nvidia-smi`, so `--id=N` is
  never remapped by a visible-device subset.
- Unscoped cleanup is refused unless `GITHUB_ACTIONS=true`,
  `OMNI_CI_ALLOW_UNSCOPED_GPU_CLEAN=1`, or CVD is explicitly `all`.
- Orphan kill matches only `/dev/nvidiaN` for selected ids (not nvidiactl/uvm).
- Skip processes whose own CVD is **disjoint** from the cleanup scope.
- Skip ephemeral version probes (`importlib.metadata`, `import sglang`,
  `m.version(...)`) so one group’s cleanup cannot SIGKILL another group’s
  precheck.
- Use `/usr/bin/tr` inside the script; interactive shells may alias `tr`→`tree`.

`benchmarks/benchmarker/utils.wait_for_gpu_memory_release` requires
`CUDA_VISIBLE_DEVICES` when not on GitHub Actions.

Manual cleanup:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash .github/scripts/delete_gpu_process.sh --kill-orphans
```

Never use unscoped `pkill -9`, `killall`, or a cleanup command without an
explicit target set on a shared host.

## Monitoring

Create exactly one Tab A and one Tab B for every configured GPU group. Three
groups require three Tab A terminals and three Tab B terminals, even when a
group is temporarily idle between queued runs. Watcher count follows the number
of GPU groups, not the Mode A/B/C choice.

### IDE-visible terminals (required)

Tab A and Tab B are **operator-facing**. They must run in Cursor / IDE terminal
panels so the user can see live progress without opening log files.

- Launch each watcher in a dedicated IDE terminal (agent background shell that
  streams into the Terminal panel, or a user-opened tab).
- **Do not** treat `nohup … > /tmp/*.log &` as having started Tab A/B. Durable
  tees under `/tmp` are a backup only; they do not replace visible terminals.
- Keep **exactly one** Tab A and **exactly one** Tab B per GPU group. Kill
  duplicate watcher processes before restarting.
- If the user closes an IDE tab, restart that watcher in a new visible terminal;
  do not assume a leftover background process is still visible to them.

Tab A — aggregate strict progress:

```bash
bash .claude/skills/tune-ci-thresholds/watch_calibration_group.sh \
  <gpu-group> <group-run-1> [<group-run-2> ...]
```

Tab B — active server / pytest logs:

```bash
bash .claude/skills/tune-ci-thresholds/watch_calibration_servers.sh \
  <gpu-group> <group-run-1> [<group-run-2> ...]
```

Behavior:

- Resolves the active pytest from its process and `--basetemp`.
- Prefers `server.log` under that basetemp. Locally (non-CI),
  `server_log_file()` returns `None`, so router/worker stdout is multiplexed
  into the sibling pytest `runN.log`; Tab B falls back to that file.
- Detaches when cleanup kills a server or pytest exits; attaches the next
  launch in the same Tab B (must not stay stuck on a completed log).
- IDE terminals truncate around ~1MiB. Tab B filters Decode/Prefill batch spam
  by default and tees a durable copy to
  `/tmp/calibration_tabB_<gpu-group>.log` (example:
  `/tmp/calibration_tabB_0_1.log`). Set `CALIBRATION_SERVER_WATCH_VERBOSE=1`
  for the raw terminal stream.

Start both watchers before that group’s first job. Pass all run directories
assigned to the group, including queued dirs whose `plan.json` does not exist
yet. Keep watchers alive until the group’s queue is complete.

Also poll `status`, `strict-audit`, and `nvidia-smi` at least every 120 seconds
while work is active. The legacy `tail_calibration_pytest.sh` is a debugging
fallback only.

## Ghost GPU memory / invisible clients

Before pinning a group, confirm selected GPUs are actually free
(`memory.used` near idle and no compute apps). On shared hosts you may see:

- high `memory.used` with **no** processes in `nvidia-smi`;
- `nvidia-smi --gpu-reset` failing with *In use by another client*;
- no matching `/dev/nvidiaN` holders inside this container.

Do **not** block forever on that pair. Prefer another idle two-GPU pair, update
`TUNE_GPU_INCLUDE` / `TUNE_GPU_EXCLUDE`, and tell the user which GPUs were
substituted. Never broaden cleanup onto reserved or foreign GPUs to chase
ghost memory. Host-side reset (Fabric Manager / container restart) is outside
this skill’s scope unless the user asks.

## Contaminated-run recovery

When the user rejects a calibration (for example concurrent interference,
implausible speed regressions, or mixed environments), follow this checklist:

1. Stop the old queue and all Tab A/B watchers for that session.
2. Roll back applied threshold edits if any (restore pre-apply test constants /
   `stages.yaml`). Keep unrelated skill or infra fixes unless the user asks to
   revert those too.
3. Create a **new** UTC run directory on current `HEAD` (do not resume the
   contaminated dir).
4. Relaunch with an explicit layout and a free GPU pair; start IDE-visible
   Tab A/B before the first pytest.
5. After the new run is strict-ready, generate `report` / `apply-plan` from the
   new directory only. Do not mix artifacts from the contaminated session.

## CUDA recovery

If `nvidia-smi` works but PyTorch cannot initialize CUDA, stop the affected
group. Do not loop retries or broaden cleanup. Host recovery may require
restarting Fabric Manager and the container. Re-run the CUDA smoke and precheck
before resume.

## Split-run reporting (Mode B only)

Merge strict-ready stage partitions with:

```bash
python tune.py merge-runs --run-dir "$RUN_A" --run-dir "$RUN_B" \
  --output-dir "$RUN_COMBINED"
```

The command validates:

- identical commit and schema hashes;
- compatible environment fingerprints;
- identical repeat policy;
- disjoint and complete stage ownership;
- strict readiness of both inputs.

Independent full calibrations (Mode C) remain separate replications. Comparing
their distributions is useful; silently combining them changes N and the
statistical policy.
