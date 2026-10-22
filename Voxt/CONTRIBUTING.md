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

## Local packages and release automation

The root `Voxt Mac CI` workflow runs the native runtime goldens, the server
API tests, Voxt's Omni unit tests and the tooling tests on the Apple Silicon
runner for pull requests that touch Voxt or `sglang_omni_mlx`. Its package job
additionally requires the repository variable `VOXT_ENABLE_CI_PACKAGE=true`.

For local development, build the Native Omni runtime and package the app with:

```bash
runtime_dir="$PWD/Voxt/build/omni-runtime"
sglang_omni_mlx/native/scripts/build_runtime.sh "$runtime_dir"
Voxt/tools/package_local_app.sh \
  --optimized \
  --native-runtime "$runtime_dir" \
  --output-dir "$PWD/Voxt/build/local-package" \
  --no-sign
```

The command creates a DMG, ZIP, PKG, and SHA-256 files under a timestamped
directory. To install the locally built app for manual verification, append
`--install --open`. `--universal` also builds the app for Intel Macs; the
runtime stays arm64, so Intel Macs use remote speech recognition. Signed
packaging requires an explicitly supplied, matching Developer ID provisioning
profile via `--provisioning-profile PATH` or `VOXT_PROVISIONING_PROFILE`. It
preserves the authorized application identifier and private Keychain group;
missing, expired or mismatching authorization fails instead of stripping
credential access. Pass an App Store Connect API key with `--notary-key`,
`--notary-key-id` and `--notary-issuer` to notarize the app, the DMG and the
PKG and staple their tickets.

### Releases

A maintainer releases Voxt by raising `MARKETING_VERSION` on `main` and pushing
a `voxt-vX.Y.Z` tag that matches it:

1. `Voxt Release` builds the runtime and the universal app on GitHub's macOS
   runner, signs and notarizes them as above, and uploads the ZIP, DMG, PKG and
   their SHA-256 files to a draft release. The build number follows the
   original Voxt scheme (1.16.2 is 101600299). A manual run without a tag only
   builds an unsigned package.
2. The draft's notes already hold the requirements, install steps, build
   source and the pull requests that touched `Voxt/` or
   `sglang_omni_mlx/native` since the previous `voxt-v` tag. The maintainer
   installs the draft's DMG, checks it, replaces each TODO line (a one-sentence
   summary, Changed and Fixed; drop a section with nothing in it), and
   publishes the release.
3. Publishing runs `Voxt Appcast`, which signs the ZIP with the Sparkle key,
   uploads the ZIP and DMG to the Voxt update service, and makes the version the
   latest entry of the stable appcast, so installed copies offer it as an update.
   The in-app notes are the release notes; sections titled `English`,
   `简体中文` or `日本語` are shown to users of that language. It refuses notes
   that still contain a TODO line. It then points the Homebrew cask in
   `hehehai/homebrew-tap` at the release. A manual run with a tag republishes
   that release.

Both workflows read these secrets from the `voxt-release` environment, which
only `voxt-v*` tags and `main` may use:

- `DEVELOPER_ID_APP_CERT_P12`: base64-encoded Developer ID Application certificate.
- `DEVELOPER_ID_APP_CERT_PASSWORD`: password for that certificate.
- `DEVELOPER_ID_INSTALLER_CERT_P12`: base64-encoded Developer ID Installer certificate.
- `DEVELOPER_ID_INSTALLER_CERT_PASSWORD`: password for that certificate.
- `DEVELOPER_ID_INSTALLER_IDENTITY`: exact Developer ID Installer identity.
- `DEVELOPER_ID_PROVISIONING_PROFILE`: base64-encoded Developer ID provisioning profile.
- `APPLE_NOTARIZATION_KEY`: App Store Connect API key contents (`.p8`).
- `APPLE_NOTARIZATION_KEY_ID`: App Store Connect API key ID.
- `APPLE_NOTARIZATION_ISSUER`: App Store Connect API issuer ID.
- `SPARKLE_PRIVATE_KEY`: Sparkle EdDSA private key matching `SUPublicEDKey`.
- `VOXT_UPDATE_API_BASE`: base URL of the Voxt update service.
- `VOXT_RELEASE_API_KEY`: API key of the Voxt update service.
- `HOMEBREW_TAP_TOKEN`: token that can push to `hehehai/homebrew-tap`.

On a Mac, the certificates and the profile can be encoded for a secret with:

```bash
base64 -i developer-id-application.p12 | tr -d '\n'
base64 -i developer-id-installer.p12 | tr -d '\n'
base64 -i voxt.provisionprofile | tr -d '\n'
```

The model weights are not included in the DMG. After installation, Voxt
downloads the selected MLX model into the user's model cache, so testers can
install the same notarized app and download models independently.

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

Release tags are maintained by project maintainers. Do not create or push `voxt-vX.Y.Z` tags as part of a normal contribution.

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
