# Chrome accessibility validation

Measured on 2026-11-27, production source `5c87b585d4471032cb3c1176e4e72e54e5baf5d2`. Features: **Meeting detection** and **App enhancement** (`testdata/behaviours.tsv`). The follow-up commit adds this evidence and a manual runner; it does not change the measured production sources.

## Scope and method

Apple M4, 10 CPU cores, 24 GiB RAM, macOS 26.6.2 (25G83). Independent, disposable Chrome profiles; the daily Chrome process started and ended with its enhanced accessibility attribute already enabled. No microphone/camera capture, login, joined call or installed Voxt replacement was used.

The release helper reads `browser.url` with the production 1 s, 2 s, then 5 s retry spacing during the active phase. A temporary XCTest drives the real `FrontApps.manual`, `setManualAccessibility` and `releaseUnusedAccessibility` implementations against the specified live Chrome PID. This isolates the native attribute lifecycle from foreground-app routing; it does not verify the installed app's microphone-to-configuration wiring. The existing unit/integration tests cover that wiring.

The local page has 1,000 or 10,000 document rows. Each animation frame updates 24 text cells and a marker. Each sample lasts six seconds, after a 2.2-second mode settle. For each size, repeat **off → on → restored off three times** (18 samples per browser). The browser remains foreground and visible, with a 1,100 × 740 point window. Read-only AX probes verify the actual attribute for every phase. The JSON records samples, not just averages.

No UI automation observes the test browser during these render samples: requesting a native accessibility tree can itself enable Chrome's enhanced accessibility attribute and invalidate the off baseline. Native window operations are a separate, intentionally enabled test.

CPU is the browser plus its descendant processes, calculated from process CPU-time deltas; **100% means one CPU core**, not the whole M4. Memory is the sum of their RSS, which counts shared pages more than once and is not physical memory footprint. These are short local synthetic samples, not statistical proof for arbitrary sites or a memory-leak test. Background system activity and cache warm-up remain possible confounders.

## Measured results

Each cell is **p95 frame ms / CPU % of one core / RSS sum MiB**, using the median of three samples.

| Browser | Rows | Off | On + URL retries | Restored off |
| --- | --- | --- | --- | --- |
| Chrome 158.0.8127.37 | 1000 | 17.2 / 37.03 / 945.0 | 17.1 / 38.67 / 950.3 | 17.2 / 39.07 / 950.5 |
| Chrome 158.0.8127.37 | 10000 | 17.5 / 71.21 / 1592.3 | 17.5 / 79.18 / 1598.7 | 17.5 / 69.26 / 1633.9 |
| Chrome for Testing 158.0.8127.36 | 1000 | 16.7 / 36.58 / 1183.0 | 16.8 / 40.43 / 1192.4 | 16.7 / 37.06 / 1148.6 |
| Chrome for Testing 158.0.8127.36 | 10000 | 16.7 / 66.86 / 1538.1 | 16.8 / 78.08 / 1578.1 | 16.8 / 70.50 / 1499.6 |

There was no material frame-time regression in these samples; CPU and memory costs were measurable. Paired on-minus-preceding-off RSS changes were Chrome 1,000 rows: +22.7, +5.4, +10.9 MiB; Chrome 10,000 rows: +228.5, +20.0, -24.0 MiB; Chrome for Testing 1,000 rows: +55.2, +4.5, +9.3 MiB; Chrome for Testing 10,000 rows: +317.9, +38.1, -8.1 MiB. First activations can cost more than phase medians suggest. Raw data: [`results-20261127.json`](../testdata/chrome-accessibility/results-20261127.json).

The Chrome for Testing run additionally exercised the 10,000-row page while idle. After ten seconds of activation settle, a 20-second sample used 0.15% of one core; off and restored ten-second samples used 0.00% and 0.10%. RSS sums were 1,404.2 → 1,442.8 → 1,418.5 MiB. An earlier five-second activation sample reached 9.28% CPU, so the low steady figure does not describe activation cost.

Five further idle on/off cycles all restored the attribute to false. Their on RSS sums were 1,404.3 / 1,440.3 / 1,449.3 / 1,456.2 / 1,438.4 MiB, and off sums were 1,357.3 / 1,363.0 / 1,384.9 / 1,397.3 / 1,404.3 MiB. Memory did not immediately return to the first baseline; these samples do not establish absence of a leak.

## Window operations and limits

Native operations were exercised on official Chrome for Testing **158.0.8127.36**, with enhanced accessibility on. It shares the `158.0.8127` version branch with installed Chrome **158.0.8127.37**; this is not an exact-version window-manager compatibility claim. The official build is available from [Chrome for Testing](https://googlechromelabs.github.io/chrome-for-testing/).

| Operation | Observed result |
| --- | --- |
| Type into page | The full `voxt4573 responsiveness check` text appeared in the field and echo. |
| Tile left | Window bounds became x=0, y=33, w=736, h=867 points. |
| Tile right | Bounds became x=736, y=33, w=734, h=867. |
| Zoom and restore | Bounds became x=0, y=33, w=1470, h=867, then returned to the right-hand bounds. |
| Minimize and restore | Read-only window metadata reported the test window offscreen after minimizing; reactivation restored it and preserved the typed text. |
| Restore previous size | Returned to x=100, y=60, w=1100, h≈740. |
| Title-bar drag | **Inconclusive.** Coordinate automation did not move the window with the attribute either on or verified off. This is not counted as a pass or attributed to the feature. |

Geometry was verified independently with read-only `CGWindowListCopyWindowInfo`. The native UI observer raises a minimized app on inspection, so the offscreen check was taken before requesting another UI tree. These observations show completed operations, not a measured animation-latency comparison. Freehand resizing, VoiceOver, third-party window managers, multi-display behavior and long browsing sessions remain manual checks. No claim of universally unaffected window handling is made.

The published manual runner was smoke-tested against Chrome 158.0.8127.37: its actual off → on → off checks passed, followed by a passing opt-in native XCTest with no compiler warnings. The copied test was removed afterward. Changed-file pre-commit, `scripts/check.sh`, and the behaviour/report/protocol-document tests passed; the production suites at the source head are recorded in the PR.

## Manual runner

A maintainer runs this before a release that changes Chromium accessibility activation/restoration, on an otherwise idle Mac. Sources are in `testdata/chrome-accessibility/`; they are test-only, use loopback, and must not be bundled. Python 3 and the existing macOS Accessibility authorization for the helper/test process are required. The runner fails if the target browser is not foreground. Do not attach a UI accessibility observer until render measurements finish.

From `Voxt/`, build the release helper and prepare a scratch directory. Copy the opt-in test only when the destination does not already exist:

```sh
swift build --package-path helper -c release -j 2
scratch=$(mktemp -d /tmp/voxt-chrome-ax.XXXXXX)
export CHROME_NATIVE_PROBE_DIR="$scratch"
swiftc testdata/chrome-accessibility/ax-observe.swift -o "$scratch/ax-observe"
export CHROME_AX_OBSERVER="$scratch/ax-observe"
probe_test=helper/Tests/VoxtNewHelperTests/ChromeNativeProbeTests.swift
test ! -e "$probe_test" && cp testdata/chrome-accessibility/ChromeNativeProbeTests.swift "$probe_test"
swift test --package-path helper -j 2 --filter ChromeNativeProbeTests > "$scratch/native-test.log" 2>&1 &
native_test_pid=$!
open -n -a '/Applications/Google Chrome.app' --args --user-data-dir="$scratch/chrome-profile" --no-first-run --no-default-browser-check --new-window --window-size=1100,740 --window-position=100,60 about:blank
```

Find the main Chrome PID for **that exact scratch profile** in Activity Monitor or `ps`, then start the server (replace `TEST_PID`). Wait for `native-probe.ready` and the server's initial false attribute before continuing. Use Chrome's executable with the same profile to route the page to the intended process; `open -a` alone can route it to a daily Chrome instance and invalidates the comparison.

```sh
python3 testdata/chrome-accessibility/harness.py --pid TEST_PID --bundle com.google.Chrome > "$scratch/server.log" 2>&1 &
server_pid=$!
'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' --user-data-dir="$scratch/chrome-profile" 'http://127.0.0.1:18773/probe?tag=current&auto=1'
```

Wait for all 18 samples and the final released transition in `$scratch/TEST_PID-measurements.json`. Reject a run with errors, hidden pages, wrong foreground PID or mismatched AX flags. Run window checks separately with `?tag=current&window=1`, recording geometry and any observer-induced attribute changes. Do not combine those samples with the clean render data.

To end the opt-in native test, atomically write `native-command.json` in the scratch directory with a new unique string `id`, the target integer `pid`, `bundle: "com.google.Chrome"`, and `phase: "quit"`. Wait for the matching response, then `wait "$native_test_pid"` and inspect the test log. This restores attributes that the test owns. Stop the server with `kill -INT "$server_pid"`, close only the disposable Chrome instance, and remove the copied test file. Retain the JSON and logs for review; remove the scratch profile/build download after review. A pre-existing true attribute is preserved; restoration remains best effort and a crash cannot restore it.
