# MLX Audio Dependency Policy

Source/lockfile status checked against app commit `81c04a3`. Distinguish **the app's current dependency**, **the published fork**, and **pending app validation**. A passing library workflow or static lockfile audit is not an app or model-quality release certification.

## Current Voxt working-tree pins

`Voxt.xcodeproj/project.pbxproj` now references the published fork and matching LM revision:

- Audio URL: `https://github.com/hehehai/mlx-audio-swift.git`
- Audio revision: `2a6e75d28ae6a399ba7c7aec842384ef7a3142b5`
- LM revision: `c6446cf7bfb7cea76408013b614d4b2c530eaa03`
- MLX runtime: fork exact `0.31.6`
- Tokenizer / hub: fork exact `swift-transformers 1.3.4` / `swift-huggingface 0.10.2`

This replaces `.12` / runtime `0.31.4` / LM `d242429` in the project requirements. The tracked app `Package.resolved` now matches the upgraded compatibility set and passes `audit_model_stack.py --resolved`. The earlier mismatch caused [PR CI 35322684144](https://github.com/hehehai/voxt/actions/runs/35322684144) to fail before app compilation or tests; that is historical evidence, not the current lockfile status. Static pin consistency does not prove successful macOS resolution, compilation or model behavior.

The new fork preserves `.12`'s structured `TranscriptionEvent.ended(STTOutput)`, language provenance, Qwen KV/language controls, incremental Nemotron streaming, and streaming/VAD failure propagation.

FluidAudio has been removed from Voxt's package references and speaker-analysis implementation. Sortformer remains in `MLXAudioVAD`. FluidAudio removal does not establish the provenance or absence of ONNX; audit the actual app graph and artifacts separately.

## Published fork and evidence

- Repository: `hehehai/mlx-audio-swift`, remote and local branch `main`.
- Integration revision: `2a6e75d28ae6a399ba7c7aec842384ef7a3142b5` (already pushed).
- Includes upstream history through `3e978558404df4ad1bbb0a5634a03df2b0f9dfa5`; no missing upstream commits at the time of this audit.
- The earlier squash synchronization was corrected by a normal ancestry merge. Future upstream synchronization must preserve upstream ancestry.
- Passing CI revision: `33e72855c0169377ae5ddae83b0ce5da0673e920`.
- Both revisions have tree `4f1c756bff39e3ad320e2b12ce3f1726dd2a268b`; the final ancestry-only merge deliberately did not repeat CI.
- [Actions run 35309904287](https://github.com/hehehai/mlx-audio-swift/actions/runs/35309904287): Xcode 26.5 / Swift 6.3.2, dependency resolution, build, 669 Swift Testing tests and 2 XCTest tests passed.
- The workflow excludes `SmokeTests`; environment-gated network / model tests are not all exercised. This is not proof that all Voxt checkpoint weights, long meetings, or accuracy/latency targets pass.
- No new release tag was created. The temporary `chore/voxt-model-stack-modernization` and `codex/moss-asr-configuration` branches were removed after merging. Do not reference them.

The fork additionally fixes Parakeet's compiled closure captures using weight-aware `CompiledTrace`. Regression tests verify weight updates and absence of an owner retention cycle. The runner's Xcode selection, SDK and compiler paths were aligned to fix MLX CPU JIT subprocesses mixing Xcode 26.5 and 26.6.

## Integrated compatibility set (app validation pending)

Keep these as a single compatibility set; do not revert or update just one component:

| Dependency | Target |
|---|---|
| Audio fork | revision `2a6e75d28ae6a399ba7c7aec842384ef7a3142b5` |
| mlx-swift | exact `0.31.6` via Audio |
| mlx-swift-lm | revision `c6446cf7bfb7cea76408013b614d4b2c530eaa03` |
| swift-transformers | exact `1.3.4` via Audio |
| swift-huggingface | exact `0.10.2` via Audio |

Swift 6.3 is required and was verified in the fork CI. There is no need to wait for another fork sync or tag to integrate the published immutable revision. Do not point the shared app project at floating `main` or a developer's local path.

The app source now includes model preparation after weight installation, effective EOS, model-declared chat conventions, typed prefill (`.remainder` initially retained), and a concurrent loader entry point. The prequantized loader remains. One-time patch/snapshot tools have been removed; application sources are the single implementation.

Before release:

1. Validate concrete model / tokenizer conventions, thinking suppression, EOS, OptiQ mixed quantization, VLM inputs, cancellation and final chunk delivery. Source adaptation does not prove behavior parity.
2. Verify the **whole app** resolves strictly on macOS using the committed `Voxt.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved`. Regenerate it only for an intentional dependency update. The audio CI graph is reference evidence, not a substitute app lockfile.
3. Run the complete local validation batch before pushing for CI confirmation.

## Toolchain and local-first validation

Use one Xcode for the build, XCTest and MLX CPU JIT child processes. On a dedicated Mac runner select Xcode 26.5 at system level as well as through `DEVELOPER_DIR`, resolve `SDKROOT` with that Xcode, and put its default compiler toolchain first in `PATH`. On a shared developer machine coordinate a system `xcode-select` change rather than silently changing another user's toolchain.

Complete the related API changes and regression coverage locally before pushing. Run dependency audit, Debug/Release build, XCTest and relevant downloaded-model replay on Apple Silicon, then push a complete batch for a single CI confirmation. Do not use one commit/Actions run per compiler error.

Earlier integration work exercised pure Foundation state machines using a temporary Swift 6.3.2 toolchain on Linux. The current refactoring audit environment has neither Swift nor Xcode. Neither isolated harnesses nor Python/static checks type-check the complete macOS Swift/Metal app. When no local Mac is available, record that limitation and leave verification pending; a passing fork workflow does not waive the app validation gate.

### Complete local app validation

After selecting the matching Xcode on a Mac:

```bash
bash tools/run_model_stack_validation.sh
# For an intentional dependency upgrade only:
# bash tools/run_model_stack_validation.sh --update-lock
# With the integration-test checkpoints already installed:
VOXT_RUN_MODEL_TESTS=1 VOXT_MODEL_STORAGE_ROOT="/absolute/path/to/existing/models" \
  bash tools/run_model_stack_validation.sh
```

Normal runs use the reviewed lockfile strictly. `--update-lock` explicitly permits dependency re-resolution for an intentional upgrade. The script builds Debug/Release, runs XCTest, audits the resolved graph and release binary/link maps, and keeps logs/results under `build/model-stack-validation.*`. It does not push or dispatch Actions. Review skipped tests and separately measure real model quality, latency and memory.

## Reproducibility and audits

- `bash tools/resolve_dependencies.sh` verifies an existing lockfile strictly or bootstraps a missing one on macOS. Review and commit actual resolver output; never fabricate transitive pins.
- During an intentional upgrade, regenerate the existing lockfile explicitly in that upgrade workspace before returning to strict resolution. A stale lockfile is not repaired by silently weakening release resolution.
- `tools/audit_model_stack.py` now checks the new Audio/LM source revisions and the full upgraded model dependency set. Update it in the same change as any future project pin changes.
- Source audit: `python3 tools/audit_model_stack.py`.
- Resolved graph audit: add `--resolved <app Package.resolved>`.
- Release audit: `--app <Voxt.app>` and `--link-map <map>` on macOS. Inspect static/dynamic provenance and resources, not just text grep.
- CI preserves actual resolved graphs and release link maps are audited. Production releases require the app lockfile in source control.

## Fork synchronization and release rules

1. Prefer official upstream releases when they include every required product contract and fix.
2. Otherwise use this fork with a small, documented patch set. Upstream synchronization must use a normal merge preserving ancestry, not squash; review conflicts and verify retained contracts.
3. Do not reset the fork to upstream and lose Voxt changes. An `ahead` count is expected for the patch set; a `behind` count requires investigation.
4. Pin an immutable published revision in Voxt while preparing a release. A fork tag is optional for initial integration.
5. After validation, annotated fork tags use `v<upstream-version>-voxt.<n>`. Never move an existing tag or reuse an upstream tag for different code.
6. A branch merge is not a quality certification. Do not label a release stable until app build, model quality, memory and performance gates pass.

See [the updated plan](ModelStackModernizationPlan.zh-CN.md) and [implementation status](ModelStackModernizationImplementation.zh-CN.md).
