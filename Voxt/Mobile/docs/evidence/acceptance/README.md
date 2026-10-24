# Acceptance run

What this checks: a fresh clone of this branch builds and passes every test command in `Voxt/Mobile/README.md` and the `voxt-mac-ci.yaml` jobs on a Mac with only the documented prerequisites, and each missing prerequisite stops with a message that says what to do.

## How it was run

- Clone: `git clone --no-local --depth 1 --branch <branch>` into a new folder, of an earlier build of this change; the PR description lists what changed after it. No untracked or ignored files.
- Isolation: a new macOS user account needs admin rights, so the run used its own `HOME`, `TMPDIR`, XDG and clang module caches in the run folder, SwiftPM's default caches under that `HOME`, Go's caches inside the clone, and `PATH` limited to Homebrew and the system. Simulator devices and `xcodebuild`'s error result bundles still come from the account (CoreSimulator and the system temp folder ignore `HOME`/`TMPDIR`).
- No Keychain: `swift … --disable-keychain --disable-netrc`, `xcodebuild … -packageAuthorizationProvider netrc`, `CODE_SIGNING_ALLOWED=NO` or ad-hoc signing only.
- Machine: Apple silicon, macOS 27.0, Xcode 27.0, Go 1.26.3 (the scripts downloaded 1.27.1 from `go.mod`), XcodeGen 2.45.4, Python 3.14.
- The iPhone unit tests waited for the simulator, which another run was using, and then ran on the same clone; the wait is in `steps.txt`.

## Results

| Command | Result | Time |
| --- | --- | --- |
| Voxt `build-for-testing` (TestDebug), then `voxt-unit`'s 30 classes | 445 tests, 0 failures | 317 s + 45 s |
| `go test ./...` in `tailcat/native` and `tailcat/host` | ok | 91 s |
| `tools/build-tailcat-native.sh` | built | 90 s |
| `swift test`: VoxtMobileKit | 347 passed, 7 skipped by their gates (the live-server suites; the Keychain round trip without `VOXT_KEYCHAIN_TESTS`) | 271 s with the next two rows |
| `swift test`: VoxtTailcat, TypingInput, KoreanInput, SpanishInput | 12, 8 + 14, 7, 5 | |
| `swift test`: JapaneseInput, PinyinInput, ZhuyinInput | 12 + 15, 51, 24 | |
| `tools/test-package.sh` (VoxtConnectionKit) | 13 + 82 | 45 s |
| Voxt Connector `xcodebuild build`, unsigned | succeeded | 46 s |
| VoxtMobileTests on an iPhone 17 Pro simulator | 147 run, 4 skipped, 0 failures | 304 s |
| `tools/test_*.py`, `tools/test-design-tour-fixture.py` | 13 + 2 + 4 tests, 6 fixture checks | 15 s |
| `ios/tools/test-input-methods-optional.sh` | 4 of 4 checks | 45 s |
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

Full output: `negative-checks.txt`. Most of it is from a run on a build before the one in `steps.txt`; the XcodeGen check for the Connector build is from the run in `steps.txt`, which also repeated the simulator-name check (`results.txt`).

Found by the earlier run: `PairingStoreTests.keychainRoundTripUpdatesAndDeletesOnlyItsOwnItem` (VoxtMobileKit) writes to the login Keychain from `swift test`. With no login keychain (the isolated `HOME`) it blocked on a Keychain dialog until the run's 10-minute watchdog stopped it; with the account's own `HOME` it passes. It now runs only when `VOXT_KEYCHAIN_TESTS=1` is set.

Not run unattended, because each needs a person at the Mac: a device build without `Signing.local.xcconfig` (Xcode looks up identities in the login Keychain) and launching Voxt.app (macOS asks for Accessibility at startup).
