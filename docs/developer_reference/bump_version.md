# Bumping the SGLang Pin

SGLang-Omni pins one SGLang release and the stack that release pins. Moving
that pin is a version bump PR. This page covers what moves together, where
Omni depends on SGLang beyond its public API, how the CI image is updated,
and what has to be measured before the PR is credible. Read
[main.md](./main.md) first for the stage / scheduler / model-runner picture;
the seams named below are the ones that page introduces.

Omni is not a thin caller of SGLang. `OmniScheduler` borrows the upstream
`Scheduler` methods it does not override and runs them on itself,
`SGLModelRunner` subclasses `ModelRunner`, the execution bridge drives
`ForwardBatch` from Omni's own event loops, and the engine builders mutate
`ServerArgs` at fixed points of the engine lifecycle. Each of those is a
contract with one SGLang release, and upstream can move any of them without
touching a public signature. That is why a bump is reviewed as a contract
change, not as a pin edit.

## What moves together

The pin set is whatever upstream's `python/pyproject.toml` pins at the target
tag. Diff it against the current tag first, and resolve the `lmsysorg/sglang`
image for the tag to a digest; the size of the bump follows from whether
torch, CUDA or Python moved, not from the SGLang version number.

Omni mirrors these pins in `pyproject.toml`: `sglang`, `torch`,
`torchvision`, `torchcodec`, `flashinfer_python[cu13]`, `flash-attn-4`,
`kernels`, `numba`, `transformers`, and `torchaudio` when upstream moves it.
`sglang-kernel` is not pinned by Omni; it comes with the image. Every
comment that says a pin matches the sglang stack marks a line to revisit.

The other places a version lives:

| File | What it carries |
|---|---|
| `docker/Dockerfile` | `SGLANG_IMAGE` (digest of the new tag's cu13 manifest), the FlashInfer reinstall version, the JIT cache path `/root/.cache/flashinfer/<version>`, `FLASHINFER_CACHE_IMAGE` |
| `.github/workflows/*.yaml` | Every `image:` line, pinned by digest |
| `docs/get_started/installation.md`, `docs/basic_usage/tts.md`, `docs/cookbook/*.md`, model READMEs | Version names in install instructions |
| Comments in `sglang_omni/` | Never name a version; state the invariant the code relies on so the text survives the next bump |

Search the tree for the old versions and the old image digest; the table is
what past bumps touched.

The ROCm, XPU, NPU and MUSA stacks (`docker/rocm.Dockerfile`,
`docker/xpu.Dockerfile`, `pyproject_rocm.toml`, `pyproject_xpu.toml`,
`docs/get_started/installation_xpu.md`) pin their own SGLang tag and base
images. No project CI builds them, so a bump PR leaves them alone, says so in
its description, and hands the provider owners the new tag, the matching
provider image digest if one exists, and any platform dispatch change made in
`sglang_omni/platforms/`.

## Where Omni depends on SGLang

Omni's contracts with SGLang are not all visible from the import list. The
surfaces below are the ones past bumps had to revisit, each with an example
of how it broke. Check every one of them, and expect a release to add a
surface this list does not name.

**Imports.** Every `from sglang... import` in `sglang_omni/` and `tests/`.
Most files import upstream directly; `sglang_omni/vendor/sglang/` re-exports
the layers, distributed helpers and core types that Omni patches or wants one
import site for. Check that the module and symbol still exist, that a
re-export resolves to the same origin, and that the signature or dataclass
fields are unchanged. An import behind `except ImportError` gets the
same check: the MOSS-TTS flash attention import was guarded that way, and
when `sglang.jit_kernel` became `sglang.kernels.ops` it would have fallen
back to SDPA without a word.

**Borrowed and subclassed classes.** `OmniScheduler` looks up the upstream
`Scheduler` methods it does not override through `__getattr__` and runs them
with itself as `self`, and builds the scheduler components those methods
expect (`SchedulerDPAttnAdapter`, `SchedulerLoadInquirer`, the logprob
processor, `ParallelState`, `NewTokenRatioTracker`) with upstream's own
kwargs; `SGLModelRunner` subclasses `ModelRunner`. Diff the body of every
method Omni overrides and every borrowed method it calls, and look for
`self.<attr>` reads the new upstream bodies make that `OmniScheduler.__init__`
never assigns. When a constructor gains or loses fields, pass the new shape;
a helper that filters kwargs by signature or branches on field layout keeps
two versions alive.

**The vendor layer.** `sglang_omni/vendor/sglang/layers.py` patches
`RMSNorm.forward_cuda` and `models.py` patches `apply_qk_norm`; the module
docstrings state what each patch does, and `models.py` the condition for
removing it. Check that the wrapped upstream body still has the shape the
patch assumes (a dispatch rewrite upstream can route around a patched method
without an error) and that `tests/unit_test/vendor/` still pins it.

**`ServerArgs` mutation.** Omni changes engine configuration after
`build_sglang_server_args` through one seam,
`sglang_omni/vendor/sglang/server_args.py::override_server_args`. Upstream
decides what a mutation means at each lifecycle phase; today a record that
is not yet published resolves in place, and a published record is read-only
with its values living on the runtime-context bags. Every call site has a
phase, and every later reader has to read from where the current release
stores the value. The bump that introduced the read-only record turned
several write-then-read-back sites into hard errors.

**Compat overlays.** `sglang_omni/models/dots_tts/compat.py`,
`sglang_omni/models/qwen3_tts/compat.py` and
`sglang_omni/models/qwen3_omni/components/vision_compat.py` bridge a pinned
third-party package to the pinned stack, and each carries its removal
condition in its docstring. Read the condition against the new stack and
delete the overlay when it is met. A new overlay is one module that every
import of the package goes through, with the condition written down.

**Copies of upstream code.** Omni re-implements a few upstream helpers where
it needs a different shape, for example the seeded sampling transform in
`moss_tts/sampling_kernels.py` and `qwen3_tts/sampling_kernels.py`. A copy
imports nothing, so no import or signature diff flags it; find them through
the comments that name the upstream source and diff them against the new tag
by hand. When upstream changes a numerical detail (a clamp, an accumulation
order), mirror it in every copy and pin the boundary with a test per copy.

**Test doubles and monkeypatch targets.** `tests/unit_test/fakes.py` and the
test-local fakes model resolved upstream shapes. A test that patches an
upstream name fails loudly when the name is gone; a fake that still accepts
a field upstream removed does not, so the test passes and the code does
not.

**Defensive access.** `getattr`, `hasattr` and `except AttributeError`
against state the pinned release defines statically are a second version
kept alive by accident. The bump adds none and converts the ones in files it
touches.

## Reading the upstream delta

SGLang squash-merges, so every upstream PR is one first-parent commit and the
first-parent log between the two tags is the complete delta. Most of it
touches files Omni never reaches. Scope the log to the files behind the
import and subclass surfaces above and read every commit that remains,
noting the old behavior, the new behavior and the first place in Omni where
the difference is visible. Check separately for release-line patches that
the previous tag carried and the new tag does not.

Some upstream modules are borrowed as behavior, not as names, and are read
as full diffs regardless of what the scoped log says: `managers/scheduler.py`
and `scheduler_components/`, `schedule_batch.py`, `schedule_policy.py`,
`model_executor/model_runner.py`, `forward_batch_info.py`, the CUDA graph
runners, `server_args.py` with the runtime context, `sampling/`, the layers
named in the vendor module, `mem_cache/` where Omni sizes KV, and
`environ.py` for changed defaults.

Changes that reach Omni live through borrowed `Scheduler` methods (admission
order, tree-cache eviction, memory reservations subtracted from the KV
budget, which of stop and `max_new_tokens` wins on the same step) are
upstream behavior. Omni does not pin or patch them; they go in the PR as
inherited changes with the user-visible effect stated, and the A/B measures
them.

## Consequences of the stack, not of SGLang

A torch or Python move brings changes no upstream commit describes. Examples
that have cost time:

- Transitive pins. torch pins its own NCCL; one NCCL release made a failed
  NVLS multicast bind fatal at communicator creation where the previous one
  logged and fell back, so every TP>=2 engine died on hosts where the bind
  fails. The fix was an environment variable in the TP workflow with the
  cause in a comment, but finding it needed the torch release's dependency
  list, not SGLang's.
- Kernel hubs. Transformers serves some attention implementations from the
  `kernels` hub, which builds per torch and CUDA version. When no build
  exists for the new pair the model falls back to eager without an error.
  For the paths Omni captures into CUDA graphs, SGLang's own attention
  classes ship with the pin and do not have this problem.
- Import guards in model packages. A package that checks the torch and
  torchaudio versions at import refused a torch that torchaudio never
  matched. That is what a compat overlay is for.
- Floating-point programs. A Transformers upgrade changed the intermediate
  dtype and reduction order of a vision positional-embedding interpolation.
  Every API and shape matched and a benchmark score dropped. For a
  pretrained model the arithmetic that interprets its weights is part of
  the contract; the overlay preserves the old sequence and is verified on
  intermediates, not only on the final score.
- Caches. New Inductor, Triton and FlashInfer versions invalidate every
  compiled artifact once, so the first pass in a fresh image measures
  compilation, not serving.

## The CI image

GPU CI runs inside `hongccc/sglang-omni`, pinned by digest in every
workflow. The CI virtualenv is built on the image's Python with system site
packages, so torch, FlashInfer and SGLang come from the image and only what
the image lacks is installed on top; `verify_omni_installed_pins.py` then
checks every exact pin in `pyproject.toml` against what is installed.

A bump therefore ships a new image: build `docker/Dockerfile` on the
`lmsysorg/sglang` digest for the new tag, populate the FlashInfer JIT cache
on a GPU for the architectures CI runs on (Docker builds have none, so the
Dockerfile copies the cache from a previous image), push it, and put the new
digest in the Dockerfile and the workflows. CI on the branch means nothing
until the workflows point at the new image: on the old one the setup step
installs the new torch into the virtualenv and nothing after that reflects
the shipped stack.

## Validation

Two things prove the bump: the complete unit suite on the new stack, and an
A/B of every model between the current pin and the target pin. Profiling
comes in when the A/B moves a number.

Run the complete unit suite, default and accelerator selections, inside the
new image. A failure is one of three things: a test that patches an upstream
name that no longer exists, a fake that models the old shape, or a real
contract break. The first two are test fixes; the third is a surface above
that was missed.

The A/B compares `main` at the merge base on the current image with the
branch on the new image, on the same host with the same datasets and
concurrency, from a warm server (the first pass in a fresh image is
discarded). Run the CI presets and gates for the families the workflows
cover, and a manual launch with one non-streaming and one streaming request
for the families they do not, at the CI concurrency and at concurrency 1,
where per-request host cost shows. The families that consume something
nothing else does, such as the diffusion runtime, the dLLM scheduler or the
weight-share topology, are the ones most likely to break without a test
noticing.

Compare accuracy gates per sample rather than by aggregate score: a sample
that flips between two runs of one arm is noise, a sample that is stable on
each arm and differs across arms is the bump. Compare the startup logs too:
CUDA graphs captured for the families that enable them, the same set of
fallback warnings, the KV pool size, and time to ready from a warm cache. A
gate that `main` also fails is not the bump's to retune; it belongs to a
calibration PR of its own.

When a number moves, the request-level event recorder says which stage owns
the difference, a torch trace per stage separates kernel time from host
time, and a microbenchmark of the kernel the trace points at settles it;
[profiler.md](./profiler.md) has the mechanics. A regression belongs to the
bump when the same-host A/B reproduces it with the Omni commit held fixed.
One TTS regression blamed on a bump bisected to an unrelated PR merged three
days before the branch existed; the bump was just the first change to run
that configuration at the CI concurrency.

## The PR

The description carries the upstream delta, not the diff: the pins that
move and the one upstream change that dominates the adaptation, the
modifications grouped by the surfaces above, a separate list of every
inherited upstream change a user can observe, and the A/B with both arms'
commits and image digests, sample counts and the profiler attribution for
any delta outside noise. Measurements and inferences are labeled as what
they are.

GPU CI needs the `run-ci` label plus one selector per family (`run-higgs`,
`run-moss`, `run-qwen3-tts`; `run-fun-asr`, `run-qwen3-asr`,
`run-whisper-asr`), applied with `/tag-and-rerun-ci <selectors>`. The
selectors within a family are exclusive, so each preset gets its own run on
the new image before merge.

After the merge, everyone pulls the new image and rebuilds their
virtualenvs; an environment built against the old pins does not run `main`.

A full CI calibration must follow the merge. The new stack moves throughput
and latency, so the thresholds in the workflows still describe the old image.
Recalibrate every family on the merged commit, in its own PR.
