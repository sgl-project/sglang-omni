# Repository Guidelines

## Project Shape

Voxt is a macOS menu bar voice input and translation app written in Swift with an Xcode project at `Voxt.xcodeproj`.

- Main app sources live under `Voxt/`.
- Tests live under `VoxtTests/`, with shared helpers in `VoxtTests/TestSupport/`.
- User-facing docs live under `docs/`.
- Local build products, derived data, and scratch references live under `build/` and `tmp/`; avoid editing or reviewing generated files there.
- The root `.codex/` directory is local workflow state and is gitignored.

## Build And Test

Use the shared `Voxt` scheme.

```bash
xcodebuild -list -project Voxt.xcodeproj
xcodebuild build -project Voxt.xcodeproj -scheme Voxt -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO
xcodebuild test -project Voxt.xcodeproj -scheme Voxt -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO
```

For focused test runs, prefer `-only-testing:VoxtTests/TestClassName` or `-only-testing:VoxtTests/TestClassName/testMethodName`.

CI runs tests with explicit SwiftPM cache paths and `CODE_SIGNING_ALLOWED=NO`; mirror `.github/workflows/tests.yml` when reproducing CI package resolution issues.

## Signing

Shared signing defaults are in `Config/Signing.shared.xcconfig`.
For local signing, copy `Config/Signing.local.xcconfig.example` to `Config/Signing.local.xcconfig`; the local file is gitignored. Do not commit personal signing settings.

## Coding Conventions

- Follow existing Swift style in nearby files.
- Keep UI work consistent with the existing SwiftUI/AppKit split under `Voxt/Windows`, `Voxt/Settings`, and `Voxt/App`.
- Split large files by responsibility, not arbitrary line counts. Preserve actor isolation and keep file-local helpers private when extracting code.
- Treat unreferenced-symbol scans as candidates only: check protocol/selector callbacks, resource references, migrations, and tests before deletion.
- Prefer existing managers, stores, and support types before adding new abstractions.
- Keep tests deterministic by using `UserDefaults` suites and temporary directories from `VoxtTests/TestSupport` where applicable.
- Do not modify audio fixtures in `VoxtTests/Fixtures/Audio/` unless the task explicitly requires fixture changes.

## Dependencies

Swift package dependencies are resolved through the Xcode project. Direct package references include MLXAudio, mlx-swift-lm, llama.swift, GRDB, swift-log, Sparkle, PermissionFlow, and FaviconFinder; mlx-swift is part of the model dependency graph. The MLX Audio dependency policy is documented in `docs/MLXAudioDependency.md`.

## Architecture And Documentation

- Documentation index: `docs/index.md`.
- Current source map: `docs/Architecture.md`.
- Refactoring baseline and deletion evidence: `docs/RefactoringAssessment.zh-CN.md`; staged implementation and pending validation: `docs/RefactoringProgress.zh-CN.md`.
- Keep current behavior guides distinct from historical plans; do not present pending verification as completed.

## Release Workflow

Release automation is in `.github/workflows/release.yml`. Stable release tasks may also use the local `voxt-release` skill under `.codex/skills/voxt-release/`.
