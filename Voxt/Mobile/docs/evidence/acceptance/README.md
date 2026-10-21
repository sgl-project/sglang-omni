# Acceptance run

What this checks: a fresh clone of this branch builds and passes every test command in `Voxt/Mobile/README.md` and the `voxt-mac-ci.yaml` jobs on a Mac with only the documented prerequisites, and each missing prerequisite stops with a message that says what to do.

## How it was run

- Clone: `git clone --depth 1 --branch feat/voxt-mobile` into a new folder (the branch at 80e4b509; the only later changes are the Keychain-test gating in VoxtMobileKit, rerun on the final commit, and the rebase onto current `main`). No untracked or ignored files.
- Isolation: a new macOS user account needs admin rights, so the run used its own `HOME`, `TMPDIR`, XDG and clang module caches in the run folder, SwiftPM's default caches under that `HOME`, Go's caches inside the clone, and `PATH` limited to Homebrew and the system. Simulator devices and `xcodebuild`'s error result bundles still come from the account (CoreSimulator and the system temp folder ignore `HOME`/`TMPDIR`).
- No Keychain: `swift … --disable-keychain --disable-netrc`, `xcodebuild … -packageAuthorizationProvider netrc`, `CODE_SIGNING_ALLOWED=NO` or ad-hoc signing only.
- Machine: Apple silicon, macOS 27.0, Xcode 27.0, Go 1.26.3 (the scripts downloaded 1.27.1 from `go.mod`), XcodeGen 2.45.4, Python 3.14.

## Results

| Command | Result | Time |
| --- | --- | --- |
| Voxt `build-for-testing` (TestDebug), then `voxt-unit`'s 27 classes | 344 tests, 0 failures | 363 s + 45 s |
| `go test ./...` in `tailcat/native` and `tailcat/host` | ok | 76 s |
| `tools/build-tailcat-native.sh` | built | 90 s |
| `swift test`: VoxtMobileKit (on the final commit) | 347 passed, 1 skipped (`VOXT_KEYCHAIN_TESTS` unset) | 50 s |
| `swift test`: VoxtTailcat, TypingInput, KoreanInput, SpanishInput | 12, 8 + 14, 7, 5 | 181 s with the next row |
| `swift test`: JapaneseInput, PinyinInput, ZhuyinInput | 12 + 15, 51, 24 | |
| `tools/test-package.sh` (VoxtConnectionKit) | 13 + 82 | 45 s |
| Voxt Connector `xcodebuild build`, unsigned | succeeded | 31 s |
| VoxtMobileTests on an iPhone 17 Pro simulator | 147 run, 4 skipped, 0 failures | 301 s |
| `tools/test_*.py`, `tools/test-design-tour-fixture.py` | 13 + 2 + 4 tests, 6 fixture checks | 15 s |
| `ios/tools/test-input-methods-optional.sh` | 4 of 4 checks | 47 s |
| `VOXT_CONNECTOR_SIGNING_IDENTITY=- tools/build-connector.sh` | built, signed ad hoc | 45 s |
| `tools/tailcat-go-bootstrap-test.sh` | helper built with Go 1.27.1 | 60 s |

Result lines of every command: `results.txt`; step times: `steps.txt`.

## If something is missing

| Missing | Message |
| --- | --- |
| Go | `system go was not found on PATH` |
| Go 1.27.1 and no network | `go1.27.1 toolchain was not found: install Go 1.27.1 or later, or let go download it (needs network)` (an unreachable proxy fails after about 30 s) |
| XcodeGen, Connector build | `xcodegen is needed to build Voxt Connector: brew install xcodegen` |
| XcodeGen, build without input methods | `FAIL xcodegen` |
| An iPhone simulator, CI job | `::error::No iPhone simulator on the runner; install one with: xcodebuild -downloadPlatform iOS` |
| The named simulator, `xcodebuild` | `Unable to find a device matching the provided destination specifier` |
| A signing identity, Connector build | names `VOXT_CONNECTOR_SIGNING_IDENTITY=-` (ad hoc) and that the port and token are then entered by hand |

Full output: `negative-checks.txt`.

Found by this run: `PairingStoreTests.keychainRoundTripUpdatesAndDeletesOnlyItsOwnItem` (VoxtMobileKit) writes to the login Keychain from `swift test`. With no login keychain (the isolated `HOME`) it blocked on a Keychain dialog until the run's 10-minute watchdog stopped it; with the account's own `HOME` it passes. It now runs only when `VOXT_KEYCHAIN_TESTS=1` is set, and the VoxtMobileKit row above is from the rerun.

Not run unattended, because each needs a person at the Mac: a device build without `Signing.local.xcconfig` (Xcode looks up identities in the login Keychain) and launching Voxt.app (macOS asks for Accessibility at startup).
