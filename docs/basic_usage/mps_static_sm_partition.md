# Per-process SM caps with CUDA MPS static partitioning

`processes.<name>.sm_cap` reserves a fixed partition of GPU streaming
multiprocessors (SMs) for a logical process. All replicas of that process on
the same physical GPU share the partition. Different processes receive disjoint
partitions. This can improve end-to-end throughput when a compute-hungry stage
otherwise crowds out a latency-sensitive stage sharing the GPU through MPS.

For example, Qwen3-TTS AR decode can approach saturation with roughly half a
GPU. Giving it more SMs adds little, while the vocoder can use the whole GPU and
stretch the gaps between AR decode steps. Bounding the vocoder can protect those
steps and improve pipeline throughput.

## Configuration

Use a driver and `nvidia-cuda-mps-control` that support static partitioning. The
mechanism has been verified with MPS control 13020 (CUDA 13.2) on H200.

```yaml
mps: on
processes:
  engine:
    num_replicas: 4
    replica_devices: [0, 0, 0, 0]
    sm_cap: 40
  vocoder:
    num_replicas: 4
    replica_devices: [0, 0, 0, 0]
    sm_cap: 80
```

This is a partition-allocation example, not a performance recommendation.
The stage declarations must assign stages to these process names. To cap a
single stage, give it its own process; `StageConfig` has no `sm_cap` field.
The usual config CLI accepts `--processes.vocoder.sm_cap 80`. Every other GPU
process on that physical card must also declare a cap.

`mps: auto` enables MPS even for a single capped process. If MPS cannot run,
startup fails instead of ignoring the cap. `mps: off` with a cap is invalid.
CPU-only and tensor-parallel processes cannot request a cap in the native MPS
runtime. A CPU stage colocated inside a GPU process shares that process's policy.

## Choose the stage and size by measurement

Measure each stage's single-invocation latency across several SM limits in
isolation. The stage that nearly saturates within half a GPU is the one to
protect from interference. The stage that keeps getting faster toward the full
GPU is the candidate to limit. Then measure the complete pipeline, including
throughput, latency, and streaming continuity at the intended replica count and
concurrency. Static mode still requires an explicit partition for the protected
process; give it enough SMs to reach its saturation region.

Prefer the wider end of the useful range: the upper range is broad, while the
lower edge can be steep. In a Qwen3-TTS four-replica sizing experiment, results
between 72 and 96 SM differed by only 1.3%. At 64 SM the improvement over uncapped
execution fell to 2.6%; at 56 SM execution was 9.3% worse than uncapped. These
observations describe the sizing tradeoff, not a portable safe-SM list or a
guarantee for another driver, partition layout, or workload.

There is no portable default. Qwen3-TTS benefits from bounding the vocoder;
MOSS-TTS-Local's candidate is the engine. Both Higgs stages want more than half
the GPU, and limiting either gave a negative return in the tested workload.

## Allocation and lifecycle constraints

- On a physical GPU, either all resident GPU processes declare `sm_cap` or none
  do. In static mode, a client without a partition fails CUDA initialization.
- Each cap must be a positive integer multiple of the device's chunk size. The
  runtime creates a temporary one-chunk partition, reads its SM count from
  `lspart`, and deletes it. Invalid caps fail with the legal values.
- Caps are summed once per logical process per physical GPU, not per replica.
  Replicas on different GPUs receive a separate partition on each GPU.
- The sum cannot exceed the allocatable SM count. When joining a shared daemon,
  partitions belonging to other owners reduce the available capacity further.
- SMs that cannot form a complete allocatable chunk are unavailable. On the
  verified H200, 15 chunks of 8 SM yield 120 allocatable SMs out of 132 physical
  SMs. The implementation does not assume an 8-SM chunk size.
- Static mode is a daemon-level setting (`-S`). A running daemon with a different
  mode is not replaced. Startup fails with the existing native-MPS cleanup
  guidance; stop its workloads and owners before changing modes.

The runtime holds the existing per-GPU lock during probing and allocation,
and records partitions in the owner lease before spawning workers. Each worker
checks its actual SM count in the numerical startup gate. A static client receives
`CUDA_MPS_SM_PARTITION=<GPU-UUID>/<partition-id>` and
`CUDA_VISIBLE_DEVICES=0`, relative to the daemon's single visible GPU. Partition
IDs can contain multiple `/` characters and must be retained in full.

Shutdown waits for the owner's clients to detach before removing its partitions.
Failed removal is reported and leaves retained state with cleanup instructions.
The last owner quits the daemon only after clients have drained. Do not supply
external partition or active-thread-percentage environment overrides alongside
native static partitioning.

## Numerical startup gate

Static partitions change the `multiProcessorCount` visible inside a process.
cuBLAS uses this value to choose kernels, and some SM counts select defective
BF16 GEMM kernels in certain software versions. Verified H200 observations with
PyTorch 2.13.0+cu130 included relative error around 0.92 at 56 SM, versus roughly
0.003 normally. Other counts worked in that environment; they are not guaranteed
safe in another cuBLAS version. Related failures also occurred with MPS active
thread percentage limits, so this is not unique to static partition allocation.

Every capped OS process runs a BF16 GEMM with M=512, K=1024, N=1024 after CUDA
initialization and before loading models. It compares against a CPU FP32
reference using the same BF16-rounded inputs. Relative Frobenius error above
`1e-2`, non-finite error, or a reported SM count different from the request
fails startup. The numerical error identifies the actual SM count and asks for a
different `sm_cap`. There is no hardcoded safe list, and uncapped processes do
not run the gate. This small check detects the observed failure class; it does
not replace model-level correctness validation.
