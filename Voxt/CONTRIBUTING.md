# Contributing to Voxt

Thank you for helping improve Voxt. Contributions of code, tests, documentation, translations, and bug reports are welcome.

Please read this guide before opening an issue or pull request. For security vulnerabilities, do not disclose details in a public issue; use GitHub's private vulnerability reporting flow when it is available for the repository, or contact a maintainer privately through GitHub.

## Before you start

- Check existing issues and pull requests so that you do not duplicate ongoing work.
- For a larger feature or behavior change, open an issue first and describe the problem, proposed approach, and user impact.
- Keep each pull request focused on one change whenever possible.
- Do not include credentials, API keys, signing certificates, private user data, or generated build artifacts in a contribution.

## Development environment

Voxt is a Swift macOS menu bar app. The project currently targets macOS 15.0 or later and uses the shared `Voxt` Xcode scheme.

You will need:

- A Mac running macOS 15.0 or later.
- Xcode with Swift and macOS development support. CI currently runs with Xcode 26.5.
- Git and network access for Swift package dependencies.

Clone the repository and open the project:

```bash
git clone <repository-url>
cd Voxt
open Voxt.xcodeproj
```

Most contributors can build and test without signing credentials. Local signing overrides are optional and must remain uncommitted.

## Build and test

Before submitting a pull request, run the checks relevant to your change. The standard commands are:

```bash
xcodebuild -list -project Voxt.xcodeproj

xcodebuild build \
  -project Voxt.xcodeproj \
  -scheme Voxt \
  -destination 'platform=macOS' \
  CODE_SIGNING_ALLOWED=NO

xcodebuild test \
  -project Voxt.xcodeproj \
  -scheme Voxt \
  -destination 'platform=macOS' \
  CODE_SIGNING_ALLOWED=NO
```

For a focused test run, use `-only-testing`, for example:

```bash
xcodebuild test \
  -project Voxt.xcodeproj \
  -scheme Voxt \
  -destination 'platform=macOS' \
  -only-testing:VoxtTests/SQLiteStorageRepositoryTests \
  CODE_SIGNING_ALLOWED=NO
```

If Swift package resolution behaves differently locally than in CI, reproduce the package settings from [`.github/workflows/tests.yml`](.github/workflows/tests.yml), including the resolved package file and isolated cache paths.

## Making changes

Start with the [source map](docs/Architecture.md) and [documentation index](docs/index.md). The [refactoring checklist](docs/RefactoringAssessment.zh-CN.md) records known hotspots and validation requirements.

- Follow the existing Swift style and the nearby SwiftUI/AppKit structure.
- Prefer existing managers, stores, and support types before introducing new abstractions.
- Keep behavior changes covered by deterministic tests where practical.
- Split by responsibility; prefer roughly 200–500 lines for new files and review files exceeding 800 lines. These are review guidelines, not a reason to scatter shared mutable state across extensions.
- Remove code only after checking production/test references, protocol and selector callbacks, resources, and migrations. Similar-looking provider or failure-path tests are not automatically duplicates.
- Preserve actor isolation, cancellation, and access control when moving Swift declarations. Update focused-test scripts when renaming suites.
- Use the shared test helpers in `VoxtTests/TestSupport/` and isolated `UserDefaults` suites or temporary directories for stateful tests.
- Do not modify audio fixtures in `VoxtTests/Fixtures/Audio/` unless the change specifically requires a fixture update.
- For UI changes, verify the affected menu bar, settings, window, and accessibility behavior on macOS.
- For audio, permissions, model, or remote-provider changes, document any required hardware, permissions, downloads, credentials, or manual verification in the pull request.
- Update the relevant documentation when a public behavior, setup step, or user-facing option changes.

Generated files, local settings, build products, and downloaded model/runtime artifacts should stay out of commits. Review `git status` before creating a pull request.

## Branches and commits

Create a topic branch from the current default branch (`main`):

```bash
git switch main
git pull --ff-only
git switch -c fix/short-description
```

Use a short, descriptive branch name such as `feat/...`, `fix/...`, `docs/...`, or `test/...`. Write commit messages in the imperative mood and keep them specific, for example:

```text
Fix duplicate final transcription commit
Add tests for meeting retention limits
Update remote model documentation
```

Release tags are maintained by project maintainers. Do not create or push `vX.Y.Z` or `vX.Y.Z-beta.N` tags as part of a normal contribution.

## Pull requests

A good pull request should:

1. Explain the problem and the intended behavior.
2. Summarize the implementation and call out important trade-offs.
3. List the tests and manual checks you ran, including the exact command when useful.
4. Include screenshots or a short recording for visible UI changes.
5. Mention follow-up work, known limitations, migrations, or required configuration.
6. Keep unrelated formatting or refactoring out of the change.

Pull requests are reviewed for correctness, regression risk, test coverage, user experience, privacy/security impact, and maintainability. CI must pass before merging. Be prepared to update the branch in response to review feedback.

## Reporting bugs and requesting features

When reporting a bug, include:

- What you expected and what happened instead.
- Steps to reproduce, including the relevant app mode, provider, model, shortcut, or input device.
- macOS and Voxt versions, Mac model/architecture, and relevant configuration.
- Logs or screenshots with personal data and secrets removed.
- Whether the issue reproduces in the latest release or current `main`.

Feature requests are most useful when they describe the user problem, the proposed outcome, and alternatives considered rather than only prescribing an implementation.

## License

By contributing to Voxt, you agree that your contribution is provided under the repository's [Apache License 2.0](LICENSE).
