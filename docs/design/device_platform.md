# Device platform architecture

## Status and scope

This document specifies the device boundary introduced by the ROCm platform
foundation PR. The implementation deliberately covers only CUDA, ROCm, and a
CPU fallback. Transport backends, device graphs, model-specific kernels, and
broader accelerator support are separate changes.

The refactor follows SGLang's `DeviceMixin`, `PlatformEnum`, and lazy
`current_platform` conventions to minimize the amount of new vocabulary in
SGLang-Omni.

## Motivation

SGLang-Omni historically used CUDA as both a hardware identity and a PyTorch
device type. Shared process code consequently constructed `"cuda:N"` values
and called `torch.cuda.*` directly.

That distinction matters for ROCm:

- its platform identity is `rocm`;
- its PyTorch device type remains `cuda`;
- its device operations remain under `torch.cuda`;
- its recommended Linux visibility variable is `ROCR_VISIBLE_DEVICES`.

`torch.device("rocm")` is not valid. Platform identity must therefore remain
separate from the device string accepted by PyTorch.

SGLang-Omni currently pins `sglang==0.5.12.post1`, so it cannot yet delegate
this complete boundary to the newer SGLang platform implementation. Omni also
must not register itself as an SGLang SRT platform: attention backends, KV
pools, graph runners, and allocators remain owned by SGLang.

## Architecture

```text
DeviceMixin
|- CudaDeviceMixin
|  `- RocmDeviceMixin
`- CpuDeviceMixin

Platform(DeviceMixin)
|- CudaPlatform(CudaDeviceMixin, Platform)
|- RocmPlatform(RocmDeviceMixin, CudaPlatform)
`- CpuPlatform(CpuDeviceMixin, Platform)
```

Responsibilities are split as follows:

- `DeviceMixin` defines platform identity and basic device operations.
- Vendor device mixins implement the corresponding PyTorch operations.
- `Platform` adds Omni process-level hooks and visibility configuration.
- Concrete platforms compose those two roles.

ROCm inherits CUDA device operations because PyTorch exposes HIP through its
CUDA-compatible API. This inheritance does not imply that CUDA IPC, NVML,
NCCL policy, CUDA graphs, or NVIDIA kernels are portable to ROCm.

## Platform resolution

`sglang_omni.platforms.current_platform` is a module-level lazy singleton,
following SGLang's convention.

Resolution selects one usable platform:

1. If `torch.cuda.is_available()` is true and `torch.version.hip` is set,
   resolve `RocmPlatform`.
2. If `torch.cuda.is_available()` is true without a HIP build, resolve
   `CudaPlatform`.
3. Otherwise resolve `CpuPlatform`.

Availability is a resolver concern, not part of the resolved platform API.
Callers therefore never combine `is_available()` with platform identity.

## Identity and device values

| Platform | `PlatformEnum` | `device_name` | `device_type` | Runtime API |
|---|---|---|---|---|
| NVIDIA CUDA | `CUDA` | `cuda` | `cuda` | `torch.cuda` |
| AMD ROCm | `ROCM` | `rocm` | `cuda` | `torch.cuda` |
| CPU fallback | `CPU` | `cpu` | `cpu` | no device runtime |

Pipeline configuration still uses the existing `gpu_id` contract. TP process
startup may remap a configured GPU ID to process-local device zero by changing
the platform visibility variable before child initialization.

This foundation intentionally preserves the original integer-to-device flow:

```python
if gpu_id is not None:
    current_platform.set_device(current_platform.get_device(int(gpu_id)))
```

It does not introduce a second placement object or change factory arguments.
Separating global placement IDs from process-local IDs remains a possible
follow-up, after the platform refactor is verified independently.

## Platform contract

The base contract contains only operations currently used by shared Omni code.

| API | Meaning | CUDA and ROCm | CPU fallback |
|---|---|---|---|
| `_enum` and `is_*()` | SGLang-compatible platform identity | distinct CUDA/ROCm identity | CPU identity |
| `device_name` | Human-readable platform name | `cuda` or `rocm` | `cpu` |
| `device_type` | Type accepted by `torch.device` | `cuda` for both platforms | `cpu` |
| `get_device(device_id)` | Construct a process-local `torch.device` | `torch.device("cuda", id)` | `torch.device("cpu")` |
| `set_device(device)` | Select the current device for the calling thread | `torch.cuda.set_device(device)` | no-op |
| `device_count()` | Number of process-visible accelerator devices | `torch.cuda.device_count()` | `0` |
| `get_device_properties(device_id)` | Native properties used by diagnostics | `torch.cuda.get_device_properties(id)` | unsupported |
| `synchronize()` | Wait for current-device work | `torch.cuda.synchronize()` | no-op |
| `empty_cache()` | Release unused allocator blocks | `torch.cuda.empty_cache()` | no-op |
| `reclaim_process_memory(device, suppress_errors=False)` | Failed-process cleanup hook | implemented by `CudaPlatform` and inherited by ROCm | no-op base stub |
| `device_control_env_var` | Single process visibility variable | CUDA: `CUDA_VISIBLE_DEVICES`; ROCm: `ROCR_VISIBLE_DEVICES` | unset |
| `visible_device_value(env)` / `visible_devices(env)` | Read and parse the platform-owned visibility mask | parses indices, UUIDs, and other opaque selectors | empty |
| `worker_device_env(id, env)` | Validate a logical device ID and map it to one child-process selector | returns the platform visibility override | rejects accelerator placement |
| `compatibility_env_defaults(env)` | Vendor runtime compatibility policy applied before worker initialization | CUDA owns NVIDIA/SM-specific defaults; ROCm returns none | none |

CPU behavior is intentionally polymorphic. Shared callers invoke
`set_device()`, `empty_cache()`, or `reclaim_process_memory()` without an
`is_cpu()` guard. The CPU implementation absorbs those operations as no-ops.
This keeps platform branching at capability boundaries instead of scattering
CPU checks throughout core code.

### Memory reclamation

`Platform.reclaim_process_memory()` is a stub. CUDA owns the implementation
because its cleanup sequence is runtime-specific:

1. select the device;
2. synchronize it;
3. empty the allocator cache;
4. collect CUDA IPC allocations.

Normal callers use the default `suppress_errors=False`, so failures propagate.
Failed-worker cleanup passes `suppress_errors=True`. In that mode,
`synchronize()` and `ipc_collect()` failures are suppressed independently,
preserving the original behavior: `empty_cache()` is still attempted if
synchronization fails. An `empty_cache()` failure is not hidden by this flag
and is handled by the worker cleanup boundary.

### Device visibility

Each accelerator platform exposes exactly one `device_control_env_var`:

```python
CudaPlatform.device_control_env_var = "CUDA_VISIBLE_DEVICES"
RocmPlatform.device_control_env_var = "ROCR_VISIBLE_DEVICES"
```

The platform parses that variable, validates logical device assignments, and
produces the child-process selector override. The launcher adds only
Omni-specific TP settings. It does not implement precedence or compatibility
behavior for `HIP_VISIBLE_DEVICES` or other ROCm aliases. AMD recommends
`ROCR_VISIBLE_DEVICES` on Linux, and this unmerged platform contract adopts
that convention directly.

Diagnostics use the same platform property. They do not read
`CUDA_VISIBLE_DEVICES` as a universal visibility source. Structured diagnostic
output reports `device_control_env_var` and `visible_devices` generically.

NVIDIA architecture compatibility defaults are also platform-owned. CUDA
delegates their calculation to the existing topology helper and applies them
before accelerator-dependent stage construction. ROCm overrides this hook with
a no-op, so shared worker code contains no CUDA/ROCm branch.

## APIs deliberately outside the contract

The following do not belong in the base device abstraction:

- `is_cuda_alike()`: a shared PyTorch namespace does not prove compatible IPC,
  graphs, kernels, or communication behavior.
- graph or native-IPC capability booleans: support depends on runtime version,
  topology, model, and selected backend.
- streams, events, graph objects, communicators, and IPC handles: these require
  subsystem-specific lifecycles and implementations.
- a public vendor runtime property: callers should not escape the abstraction
  to make arbitrary runtime calls.
- a generic tensor movement wrapper: `Tensor.to()` and `Module.to()` already
  express dtype conversion, copy semantics, and the concrete destination.

Memory accounting beyond the native properties record also remains
subsystem-specific because process memory, allocator memory, and physical
device memory are different quantities.

## Communication and mixed-device payloads

Communication transport selection is separate from platform identity.
`CommRouter` currently accepts payloads containing any combination of CPU and
CUDA-device tensors. On ROCm, PyTorch still reports accelerator tensors as
device type `cuda`, so the same payload classification applies.

Packed payload metadata records the original device of each tensor. This is
enough for CUDA IPC transfers to preserve mixed placement:

- accelerator-origin tensors remain on the receiving accelerator;
- CPU-origin tensors are restored to CPU.

Host shared-memory staging requires additional receiver context:

```text
sender accelerator -> CPU SHM -> receiving accelerator
sender CPU         -> CPU SHM -> receiving CPU
```

The serialized tensor device records provenance, while the receiver device
identifies where accelerator-origin tensors must be restored. A single global
platform device cannot replace this per-tensor decision. Receiver-aware SHM
restoration belongs to the communication/transport phase, not to
`current_platform`.

Transport implementations must therefore own:

- CUDA IPC and future verified HIP IPC;
- host SHM staging and receiver placement;
- remote transports;
- peer-access and topology checks.

## Device graphs and model integration

The existence of graph APIs under `torch.cuda` does not prove a model or kernel
is capture-safe on ROCm. Graph selection must remain a separate backend/model
decision verified on real hardware.

Likewise, model factories still accepting `gpu_id` or constructing CUDA device
strings are not changed wholesale in this foundation PR. They should migrate
in separate, reviewable changes after the core lifecycle boundary is stable.

## CUDA-leak inventory

Direct CUDA spelling falls into distinct categories:

| Area | Resolution |
|---|---|
| Core device lifecycle | Use `current_platform`; guarded by `test_core_platform_boundary.py` |
| Existing model placement and factory arguments | Preserve in this PR; migrate model by model later |
| CUDA IPC, NCCL, CUDA graphs, streams, and events | Keep explicit inside vendor/subsystem implementations |
| NVML, compute capability, and NVIDIA topology policy | Query only for `current_platform.is_cuda()` and add ROCm-specific implementations separately |

NVML reports NVIDIA devices only. Diagnostics query it only for a CUDA
platform so they do not combine an unrelated NVIDIA inventory with ROCm
devices. This prevents misleading diagnostic output; it is not a runtime or
inference correctness issue.

## Verification

Unit tests without hardware cover:

- CUDA, ROCm, and CPU resolution;
- platform identity and PyTorch device construction;
- CPU no-op behavior;
- CUDA/ROCm lifecycle dispatch;
- strict versus suppressed failed-worker cleanup;
- CUDA and ROCm TP visibility remapping;
- generic diagnostic visibility reporting;
- the core boundary against new direct lifecycle calls.

Real-device verification is still required for:

- discovery and current-device selection;
- TP worker visibility and local-device remapping;
- tensor and module placement;
- cleanup following startup failure;
- subprocess startup;
- eager model correctness;
- every separately enabled transport, graph, compiler, and kernel backend.

NVIDIA regression testing remains required because the platform refactor must
preserve existing CUDA behavior.

## Follow-up phases

1. Transport: preserve CUDA IPC, add verified HIP IPC, and complete
   receiver-aware SHM restoration.
2. Model integrations: pass concrete devices through ASR, TTS, and Omni
   factories and remove model-specific hardcoded placement incrementally.
3. Graph and kernel enablement: opt in only after correctness testing on real
   ROCm hardware.
4. Broader platforms: add new vendor `DeviceMixin` implementations without
   expanding this PR's support claims.

## References

- Local SGLang reference: `../sglang/docs_new/docs/hardware-platforms/plugin.mdx`
- Local SGLang device contract:
  `../sglang/python/sglang/srt/platforms/device_mixin.py`
- [PyTorch accelerator API](https://docs.pytorch.org/docs/stable/accelerator.html)
- [ROCm environment variables](https://rocm.docs.amd.com/en/latest/reference/environment-variables/index.html)
