# MLX Audio Dependency Policy

## Current boundary

Voxt no longer imports, loads or links MLXAudioSTT, MLXAudioVAD or MLXAudioCore. The Audio package and resolved pin are removed. Local ASR, Silero and Sortformer use the existing Native HTTP/WebSocket server subprocesses; unavailable Native reports an explicit error rather than silently returning to Swift Audio.

The shared `sglang_omni_mlx/native` package must remain the officially merged implementation. XPC and the task-added C ABI/engine/lifetime changes are withdrawn, not future production gates. App-side protocol conversion, process ownership and cooperating model-directory readers remain. Preserve the original family options/longform behavior, model catalog, saved choices, installations and `model-storage/mlx-audio/...` paths. That storage name is not a package dependency.

Removing Audio does not mean removing MLX. Keep MLX Swift LLM/VLM/MLX, llama.swift, Tokenizers/HuggingFace and Remote ASR/LLM. Identity-only model fixtures in lifecycle tests cannot perform inference and are not numerical fallbacks.

## Retained compatibility set

| Dependency | Requirement |
| --- | --- |
| mlx-swift-lm | revision `c6446cf7bfb7cea76408013b614d4b2c530eaa03` |
| mlx-swift | direct exact `0.31.6` |
| swift-transformers / Tokenizers | direct exact `1.3.4` |
| swift-huggingface / HuggingFace | direct exact `0.10.2` |
| OmniModelArtifacts | independent local Foundation/POSIX package; no numerical model loading |

The direct products replace previously borrowed Audio transitive dependencies. Real macOS STT unlink initially exposed LocalTokenizerLoader's missing Tokenizers/Hub/yyjson graph; explicit linkage repaired it without upgrading retained pins. The actual subsequent Audio resolver removed only that pin. Do not fabricate lockfile entries or point shared requirements at a developer's local checkout/floating branch.

FluidAudio and Audio library runtime products remain forbidden. Inspect source, resolved graph, final bundle and link-map provenance separately; a source grep is not a complete binary audit.

## Numerical and lifecycle verification

- `ASRAudioSampleRateConverter` preserves the pinned AVAudioConverter path and caller rounding, not a replacement linear resampler. Its committed 42-output fixture was captured from the real pre-unlink compiled AudioCore product; provenance, sample formulas and Float32 bits are recorded beside the binary. A replacement-generated expectation is not a reference. Platform provenance does not certify every OS/SDK.
- Silero offline continues whole-audio processing, including audio longer than the abandoned validation transport's 30-second bound. Streaming keeps its original completed-frame/nil/tail behavior and serializes reset/unload with active exchanges.
- Ordinary Sortformer keeps its existing feed geometry and scheduler. The uncalled legacy realtime annotation actor was deleted after caller/selector/resource audit; an attempted 24-second whole-window replacement did not ship and existing Native admission caps were not changed.
- Numerical parity requires independently captured legacy references with matching model, decoded samples and profile/settings. Missing Silero/Sortformer references are explicit skips, not Native self-comparisons or parity acceptance. Real Sortformer weights/reference acceptance remains pending.
- App readers span model construction/process launch through actual child exit. Cancelled network/task presentation is not GPU retirement proof. Uninterruptible children and App-crash/orphan cleanup remain separate risks.

Previous Mac runs belong to their frozen source snapshots; earlier Core/helper/XPC results do not certify the corrected route. Fresh no-XPC validation passed the Mac App/TestDebug suite (1975: 1933 passed, 42 skipped, zero failures), official runtime rebuild/inventory, actual App artifact audit, public-fixture Qwen CLI comparison and ten real Silero HTTP/parity/reader checks. Full model/Release/GUI/performance acceptance remains separate.

## Local-first validation

Use one Xcode for App builds, XCTest and MLX CPU JIT child processes. Resolve SDK/toolchain paths through that Xcode. Do not silently change system `xcode-select` on a shared machine.

```bash
python3 tools/audit_model_stack.py \
  --resolved Voxt.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved
swift test --package-path Packages/OmniModelArtifacts
bash tools/run_model_stack_validation.sh
```

For an intentional dependency update only, `run_model_stack_validation.sh --update-lock` permits re-resolution. Normal runs use the reviewed lockfile strictly. Optional installed-model testing requires `VOXT_RUN_MODEL_TESTS=1` and an explicit `VOXT_MODEL_STORAGE_ROOT`; it does not authorize new model downloads or user-recording access.

The model-stack script builds Debug/Release, runs XCTest and audits resolved/binary/link-map inputs. It does not push or dispatch Actions. Review skips separately and measure real model quality, memory/latency and long meetings. Portable Foundation tests and Python/static checks do not typecheck the complete macOS Swift/Metal application.

- `tools/resolve_dependencies.sh` verifies/bootstraps real resolver output on macOS.
- `tools/audit_model_stack.py` checks fixed non-Audio dependencies and rejects retired Audio/XPC products.
- `--app <Voxt.app>` and `--link-map <map>` inspect final macOS artifacts. Do not exempt library modules merely because symbols are stripped.
- Preserve CI/release opt-in gates. App compilation alone is not release, signing, GUI or full model acceptance.

## Historical library baseline

Before retirement, Voxt used `hehehai/mlx-audio-swift` revision `2a6e75d28ae6a399ba7c7aec842384ef7a3142b5`, with tree `4f1c756bff39e3ad320e2b12ce3f1726dd2a268b`. Its ancestry merge followed CI revision `33e72855c0169377ae5ddae83b0ce5da0673e920` with the same tree. [Actions run 35309904287](https://github.com/hehehai/mlx-audio-swift/actions/runs/35309904287) passed library build/tests, excluding gated model/network smoke. This is historical provenance for captured references, not a current App dependency or release certification.

Earlier `.12`/MLX0.31.4/LM `d242429` lockfile mismatch and [CI 35322684144](https://github.com/hehehai/voxt/actions/runs/35322684144) pre-compilation failure remain historical failures. Current retained LM/tokenizer/runtime/hub pins must be reviewed together; no fork reset, squash synchronization, moved tag or new library release is part of the Native HTTP migration.

See the [Native HTTP migration plan](../plan/NativeHTTPMigrationPlan.zh-CN.md) and [earlier model-stack implementation record](ModelStackModernizationImplementation.zh-CN.md).
