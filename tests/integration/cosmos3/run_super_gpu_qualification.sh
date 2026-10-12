#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run from an idle GPU sandbox with the project's dependencies installed.
# Default allocation: 2 x H100 80GB, with the committed layerwise-offload recipe.
# Export HF_TOKEN for the gated checkpoint (or populate the model cache first).
# Native SGLang must register Super and support the generation HTTP lifecycle:
# AsyncSchedulerClient.initialize(..., worker_failure=...) and scheduler_failure.
# The historical 4e9e407 runtime does not satisfy that HTTP prerequisite.
#
# Example:
#   export COSMOS3_SUPER_NATIVE_REVISION=<installed-native-git-sha>
#   bash tests/integration/cosmos3/run_super_gpu_qualification.sh
# Optional: COSMOS3_SUPER_GPU_IDS=0,1,2,3 for a separate four-GPU campaign.
# Optional: COSMOS3_SUPER_TRANSPORTS=sdk for SDK-only qualification.
# Resident FSDP is an unqualified experiment: set COSMOS3_SUPER_RESIDENCY=resident.
# Do not use pytest-xdist; each deployment owns the selected GPU allocation.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
: "${COSMOS3_SUPER_NATIVE_REVISION:?Set this to the installed native SGLang git SHA}"
export COSMOS3_SUPER_GPU_IDS="${COSMOS3_SUPER_GPU_IDS:-0,1}"
export COSMOS3_SUPER_TRANSPORTS="${COSMOS3_SUPER_TRANSPORTS:-sdk,http}"
export SGLANG_OMNI_STARTUP_TIMEOUT="${SGLANG_OMNI_STARTUP_TIMEOUT:-1800}"
export COSMOS3_SUPER_RUN_GPU=1

# mktemp prevents pytest's basetemp cleanup from removing earlier campaigns.
campaign_dir=$(mktemp -d "${TMPDIR:-/tmp}/cosmos3-super-qualification.XXXXXXXX")
printf 'Saving qualification results to %s\n' "$campaign_dir"
git rev-parse HEAD > "$campaign_dir/omni-revision.txt"
printf '%s\n' "$COSMOS3_SUPER_NATIVE_REVISION" > "$campaign_dir/native-revision.txt"
git status --short > "$campaign_dir/worktree-status.txt"
python -m pip freeze > "$campaign_dir/packages.txt"
nvidia-smi > "$campaign_dir/gpus.txt"

# Shared runtime accelerator checks: IPC, stream routing, and worker placement.
python -m pytest -v -s -m 'accelerator and not benchmark' \
  tests/unit_test/pipeline tests/unit_test/config tests/unit_test/mps \
  --junitxml="$campaign_dir/shared-accelerator.xml" \
  2>&1 | tee "$campaign_dir/shared-accelerator.log"

# Eight synthetic generation modes; text/image/video understanding; cancellation,
# stage-owner death, process-tree cleanup, and a fresh Reasoner deployment.
python -m pytest -v -s tests/integration/cosmos3/test_super_gpu.py \
  --basetemp="$campaign_dir/smoke" --junitxml="$campaign_dir/smoke.xml" \
  2>&1 | tee "$campaign_dir/smoke.log"

# YAML -> SDK and HTTP: all seven full-step generation recipes plus policy,
# inverse/forward dynamics transport checks and a short Reasoner request.
python -m pytest -v -s \
  tests/test_model/test_cosmos3_super_generator_ci.py \
  tests/integration/cosmos3/test_super_transports.py \
  --basetemp="$campaign_dir/transports" --junitxml="$campaign_dir/transports.xml" \
  2>&1 | tee "$campaign_dir/transports.log"

# YAML -> SDK and HTTP: identical MMMU/VideoMME samples and shared scoring.
python -m benchmarks.dataset.prepare --dataset mmmu-ci-50 \
  2>&1 | tee "$campaign_dir/prepare-mmmu.log"
python -m benchmarks.dataset.prepare --dataset videomme-ci-50 \
  2>&1 | tee "$campaign_dir/prepare-videomme.log"
python -m pytest -v -s tests/test_model/test_cosmos3_super_reasoner_ci.py \
  --basetemp="$campaign_dir/accuracy" --junitxml="$campaign_dir/accuracy.xml" \
  2>&1 | tee "$campaign_dir/accuracy.log"

# Review saved images/videos and listen to sound; structural checks alone do not
# establish semantic quality. Historical HTTP scores were MMMU 30/50, VideoMME
# 28/50; compare like-for-like prompts, runtime, checkpoint, and sampling settings.
# Native-rank failure, partial-rank startup recovery, and matched native quality
# comparisons remain separate, unqualified work; these commands do not cover them.
# Additional upstream CUDA-MPS preflight, to check separately on this GPU host:
# python -m pytest -v \
#   tests/unit_test/serve/test_mps_dp_support.py::TestLaunchFailsClosedBeforeResources::test_unpinned_kv_budget_leaves_no_state
# That test also fails on unmodified main on a Mac without nvidia-smi.
printf 'Qualification finished. Review artifacts in %s\n' "$campaign_dir"
