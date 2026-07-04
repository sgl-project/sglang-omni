# H100 CI Runner Capacity

This note documents how Omni CI uses the H100 self-hosted GitHub Actions
runner and what must be true before adding more runner capacity.

## Current Runner Lane

Omni GPU CI jobs use the GitHub Actions label set `self-hosted,h100`.
The current H100 lane is served by the `omni-runner-h100` worker and the
workflow containers pin themselves to GPU devices `6,7` through
`NVIDIA_VISIBLE_DEVICES=6,7`.

That means queue time is controlled by GitHub Actions runner-worker
availability, not by raw GPU idleness. A host can show idle H100 GPUs while
Actions still keeps jobs queued if no online worker with the required labels is
available.

The current pinning is intentional: all Omni CI stages assume the same
two-GPU slice and the shared `/github/home` plus Hugging Face cache layout. Do
not start more `self-hosted,h100` workers on the same host unless each worker
has an explicit non-overlapping GPU slice and the workflows route jobs to that
slice.

## Exclusivity Requirement

Omni CI benchmark and accuracy gates must treat the assigned GPU slice as
exclusive for the full job lifetime. A clean pre-stage GPU is not enough: if a
new unrelated process attaches to one of the same GPUs after the benchmark has
started, throughput, latency, outlier WER, and CUDA memory behavior can become
untrustworthy.

Without Slurm or another resource scheduler, exclusivity has to be enforced at
the host and runner layers:

- Host policy reserves the CI GPU slice. Non-CI users and ad-hoc containers
  must not be able to allocate those GPU device nodes while the runner is
  online.
- GitHub runner labels map to one specific slice. Actions should never be able
  to schedule two workers onto the same physical GPUs.
- CI preflight and post-stage cleanup remain useful, but they are verification
  and recovery tools. They do not provide exclusive scheduling by themselves.

`nvidia-smi` cleanup and memory thresholds catch stale processes before or after
a stage. They cannot stop a new process from starting mid-stage.

## Queue Diagnosis

Use the run jobs API to distinguish runner queue time from benchmark runtime:

```bash
gh api /repos/sgl-project/sglang-omni/actions/runs/<run-id>/jobs \
  --jq '.jobs[] | {id,name,status,conclusion,labels,runner_name,started_at,completed_at}'
```

Interpretation:

- `status=queued`, empty `runner_name`: GitHub Actions has accepted the job but
  has not assigned a matching runner worker.
- `status=in_progress`, non-empty `runner_name`: the job has an Actions worker;
  inspect the runner tmux or the job log for runtime issues.
- Long queue time followed by a short job duration is a runner-capacity problem,
  not a benchmark regression.

For a representative case, the baseline MOSS run in
[#943](https://github.com/sgl-project/sglang-omni/issues/943) waited about 49
minutes for `setup - omni venv` to start on `omni-runner-h100`, then completed
the setup in about 1 minute. Its TTS streaming and non-streaming benchmark
stages each took about 5 minutes once assigned.

## Safe Scaling Options

Adding capacity requires both runner registration and GPU placement discipline.

1. Add another H100 Actions worker on a separate host that can safely expose the
   same GPU devices `6,7` to Omni CI containers. This preserves the current
   workflow pinning and is the lowest-risk way to increase `self-hosted,h100`
   concurrency.
2. Split one host into multiple runner workers only after adding slice-specific
   routing. Example labels are `h100-gpu01`, `h100-gpu23`, `h100-gpu45`, and
   `h100-gpu67`. Each runner must have a separate actions-runner directory,
   unique runner name, and a container GPU binding that cannot overlap another
   worker's binding.
3. If short TTS gates should not wait behind longer Qwen3-Omni stages, add
   job-class labels such as `h100-tts` and `h100-qwen3` and update the relevant
   reusable workflows to target those labels deliberately.

Avoid registering extra workers with only the generic `h100` label on the same
host while the workflows still hard-code `NVIDIA_VISIBLE_DEVICES=6,7`; that
can schedule concurrent jobs onto the same GPUs.

## Non-Slurm Isolation Options

Use these options in order of strength.

1. **Dedicated host or dedicated device nodes.** Reserve the CI GPUs for the
   runner at the host policy level. On shared hosts, use device permissions,
   udev rules, container runtime policy, or an equivalent access-control layer
   so normal users and non-CI containers cannot open the reserved `/dev/nvidia*`
   nodes. This is the only practical way to protect against non-cooperative
   processes without Slurm. It does not protect against root or users with
   unrestricted Docker access.
2. **One runner per non-overlapping slice.** Register one Actions worker per
   reserved slice and give it a slice-specific label. Update workflows so the
   label and `NVIDIA_VISIBLE_DEVICES` agree. For example, a `h100-gpu67` runner
   should be the only worker allowed to expose devices `6,7`.
3. **Host lock as defense in depth.** A shared `flock` file can serialize CI
   jobs that run on the same host when every runner wrapper and maintenance
   script respects it. This prevents accidental overlap among cooperative CI
   workers, but it does not stop an external process that ignores the lock.
4. **Fail-fast contamination checks.** At job start, periodically during long
   benchmarks, and during teardown, inspect `nvidia-smi --query-compute-apps`
   for unexpected PIDs on the assigned GPUs. If a foreign PID appears, fail the
   job and mark the run contaminated instead of publishing performance numbers.

Avoid GPU `EXCLUSIVE_PROCESS` compute mode for the current Omni CI lane unless
each stage has been audited to use only one CUDA process per GPU. The staged
router/worker topology can involve multiple processes, so driver-level
exclusive-process mode may reject legitimate CI children.

Do not use MPS as an isolation mechanism. MPS improves sharing; it does not
make benchmark results exclusive.

## Runner Maintenance Checklist

Before marking runner capacity as healthy, verify:

- The intended runner workers are online in GitHub Actions and have the expected
  labels.
- Each worker has a distinct runner name and actions-runner directory.
- Each worker's container GPU slice is explicit and non-overlapping.
- The assigned GPU slice is reserved against non-CI processes for the whole job
  lifetime, not just cleaned before stage start.
- The runner can mount `/dev/shm`, `/github/home`, and the shared Hugging Face
  cache with the same paths expected by Omni CI.
- A manual MOSS Omni CI dispatch can reach `setup - omni venv` without waiting
  tens of minutes when the assigned GPU slice is idle.
- A contamination drill that starts a foreign process on the slice either fails
  to acquire the GPU or causes the CI guard to fail the run before any numbers
  are treated as valid.

If the project intentionally keeps a single H100 worker slot, treat long
`self-hosted,h100` queues as expected capacity limits and say so in PR reviews
instead of interpreting them as model or test failures.
