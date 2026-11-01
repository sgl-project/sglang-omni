# Helper protocol 1

How VoxtNew (Go) talks to `voxtnew-helper` (Swift, [helper/](../helper/README.md)). Go types: `internal/helper/proto`. Swift types: `helper/Sources/VoxtNewHelperKit/Wire*.swift`. Golden transcripts: `helper/Fixtures/*.jsonl`, decoded by both sides' tests. Supervisor: `internal/helper`.

It is a subset of AudioSmith's engine protocol 2 (`docs/protocol-v2.md` there): the same envelope, handshake and error shape, with one helper process instead of an engine per role, and methods of its own. Their shapes come from the M0 spikes on a Mac (the project's `research/spikes.md`); the differences from the spikes' draft are in §6.

**Change it in all places at once**: the Go types, the Swift types, the fixtures, the fake helper (`internal/helper/helpertest`) and this page. `go test ./internal/helper/proto` and `cd helper && swift test` fail when they disagree.

## 1. Transport

- The app starts the helper with no arguments; its working directory is the executable's. It is `VoxtNew.app/Contents/Resources/helper/voxtnew-helper`, or `$VOXTNEW_HELPER` (development: README, "Signing during development").
- One JSON message per line (UTF-8, `\n`), at most 4 MiB (`proto.MaxLine`); app → helper on stdin, helper → app on stdout. stderr is free-form text: the supervisor writes it to `~/Library/Logs/VoxtNew/helper.log` and to the logs panel (source `helper`).
- The helper exits when stdin closes or on `helper.shutdown`.
- macOS checks the helper's permissions (Microphone, Accessibility) against VoxtNew.app, its responsible process: the prompts and System Settings name VoxtNew, and a grant reaches the running helper (the event tap installs within 2 s, no restart).

## 2. Messages

| Shape | Members | Direction |
|---|---|---|
| request | `id`, `method`, `params`? | app → helper |
| notification | `method`, `params`? (no `id`) | helper → app |
| response | `id` + `result` or `id` + `error` | helper → app |

No `jsonrpc` member; ids are strings (`c1`… for control requests, the session's ULID for a capture). Unknown members are ignored. A member Go marks `omitempty` is absent when zero (Swift: an optional left nil); the others are always present.

Errors are `{"code": "…", "message": "…", "data"?: {…}}` with a string code:

| Code | When | data |
|---|---|---|
| `bad_request` | malformed params, an unknown permission, a capture.stop without job | |
| `method_not_found` | unknown method | |
| `not_initialized` | a request before `helper.hello` | |
| `protocol_mismatch` | hello with another protocol | `{"supported":[1]}` |
| `busy` | a recording runs already | |
| `cancelled` | the job was cancelled | |
| `permission_denied` | macOS refused | `{"permission":"microphone"\|"accessibility"}` |
| `device_unavailable` | no input started, or none gave audio | `{"device":"…"}` |
| `unsupported` | the helper cannot do it on this Mac | `{"capability":"…"}` |
| `timeout` | the helper gave up waiting on macOS | |
| `internal` | anything else | |

## 3. Sequencing

- `helper.hello` comes first; until it is answered every other request fails with `not_initialized`. The app refuses a helper whose `protocol` is not 1 (it is not restarted: rebuild it).
- Requests are answered in any order; each control request runs on its own, so an app slow to answer Accessibility does not hold up `capture.stop`.
- **Jobs** (`capture.start`) send notifications naming them in `job` (`capture.started`, `capture.level`, `capture.pcm`, `capture.restarted`, `log`) before their response; `job.cancel` makes them answer `cancelled`. `capture.stop` and `job.cancel` are answered after the job they end, so every frame has arrived when `capture.stop` answers.
- Notifications without `job` belong to no request: `hotkey.event`, `hotkey.state`, `app.activated`, `output.changed`, `output.released` (and `log`). The supervisor hands them to `Config.OnNotify`; a notification naming a job that already answered is dropped.
- If the helper exits, its calls fail (`core.ErrHelper`) and the supervisor restarts it after 1 s, 2 s, 4 s… (at most 3 times in 10 minutes).
- **Timing.** The supervisor times each call's phases (`helper.Trace`: waiting for a ready helper, writing, the helper's answer, waking the caller) and logs a control request slower than 500 ms with them; the helper logs its own side of a request slower than 250 ms on stderr (queued, ran, and the time it waited for its main thread, where the clipboard and NSWorkspace run). The spikes saw 1–2 s round trips while the helper worked for tens of milliseconds; the two lines tell where such a call waited.

## 4. Methods

### Control

| Method | Params → result |
|---|---|
| `helper.hello` | `{protocol, app, logLevel?}` → `{protocol, version, pid, methods, device{name, os}}` |
| `helper.shutdown` | → `{}`, then exit 0 (a recording is cancelled) |
| `job.cancel` | `{job}` → `{cancelled}` |
| `permissions.status` | → `{microphone: "granted"\|"denied"\|"notDetermined"\|"restricted", accessibility}` |
| `permissions.request` | `{permission: "microphone"\|"accessibility"}` → as `permissions.status` |
| `capture.devices` | → `{devices: [{id, name, channels, sampleRate, default, transport?}]}` |
| `capture.start` **job** | `{devices?, sampleRate?, frameMs?, levelHz?, maxSeconds?}` → `{device, reason, frames, samples, seconds, restarts, peak, firstFrameMs?, cpuSeconds}` |
| `capture.stop` | `{job}` → `{stopped}` |
| `hotkey.configure` | `{bindings: [{id, modifiers, key?, behavior, swallow?}], doubleTapMs?, longPressMs?, tapMaxMs?, swallowFn?, active?}` → hotkey status |
| `hotkey.status` | → `{installed, accessibility, bindings, events, disabledByTimeout, missedEvents, maxCallbackMicros, error?}` |
| `app.frontmost` | → `{pid, bundleId?, name?}` |
| `app.list` | `{installed?}` → `{apps: [{bundleId, name, running, pid?, path?}]}` |
| `app.configure` | `{manualAccessibility?: [bundleId]}` → `{}` |
| `app.icons` | `{apps: [{bundleId, path?}], size?}` → `{icons: [{bundleId, png?}]}` |
| `textio.focus` | → `{pid, app?{pid, bundleId?, name?}, role?, subrole?, editable, selection?{location, length}, valueLength, error?}` |
| `textio.selection` | `{allowCopy?, copyTimeoutMs?}` → `{text?, via: "ax"\|"ax-range"\|"ax-marker"\|"copy"\|"none", ms, steps?}` |
| `textio.paste` | `{text, keepClipboard?, restoreDelayMs?, targetPid?, press?: [{key, modifiers?}], pressDelayMs?, verifyMs?}` → `{posted, restored, clipboardTaken, ms, front?, check?: "inserted"\|"unchanged"\|"unknown", checkMs?, field?{role?, before?{length, selection?}, after?}, secureInput?{pid, bundleId?, name?}, pressSkipped?, kept?}` |
| `browser.url` | `{bundleId?, pid?}` → `{bundleId, url?, via: "ax-focus"\|"ax-document"\|"ax-weburl"\|"none", ms, error?, retried?}` |
| `output.watch` | `{on}` → `{device?{id, uid, name?}, muted?, settable}` (the default output now) |
| `output.mute` | `{device{id, uid, name?}, journal?}` → `{took, state}` |
| `output.release` | `{device, unmute?}` → `{unmuted}` |
| `output.recover` | `{journal}` → `{restored?: [device]}` |

### Notifications

| Notification | Params |
|---|---|
| `capture.started` | `{job, device, deviceName, fellBack, fallbackReason?, deviceSampleRate, deviceChannels, sampleRate, startMs}` |
| `capture.level` | `{job, seconds, rms, peak}` (dBFS, -120 is silence; ~20 Hz) |
| `capture.pcm` | `{job, seq, offset, data}` (`data`: base64 of mono PCM16 LE at the started rate; `seq` from 0 and `offset` in samples, without gaps; ~100 ms each, the last one shorter) |
| `capture.restarted` | `{job, reason: "noCallback"\|"configurationChange"\|"deviceLost", attempt, device}` |
| `hotkey.event` | `{binding, phase: "trigger"\|"down"\|"up", behavior: "tap"\|"doubleTap"\|"longPress", heldMs?, t, front?{pid, bundleId?, name?}}` |
| `hotkey.state` | the hotkey status, when the tap was installed (Accessibility granted) or removed (taken away) |
| `app.activated` | `{pid, bundleId?, name?}` |
| `output.changed` | as `output.watch`'s result, when the default output device or its mute changed (while the watch is on) |
| `output.released` | `{devices?, reason: "sleep"}`: the helper gave its claims back itself |
| `log` | `{level, message, job?}` |

### Notes

**capture.start.** It tries `devices` (Core Audio UIDs, the user's order) in turn, then the default input (`fellBack`, `fallbackReason`); none starting is `device_unavailable`, the microphone not allowed `permission_denied` (not asked yet: macOS asks, and the job waits for the answer). The audio is resampled to `sampleRate` (16000) mono and cut into `frameMs` (100) frames. `capture.started` comes with the first audio from the input, 200–400 ms after the request (`startMs`): before it, the user's words are not heard. Once it took 5.9 s (10-20, the first session after the app started, while three models loaded; the swap was 90% full that morning): the app logs the result's `firstFrameMs` with each recording, warns past 1 s, and the overlay says "Starting the microphone…" past 0.4 s (`session.WaitMic`). A watchdog rebuilds an engine that gives no audio within 1.2 s, the second time on the default input; a third silence ends the job (`device_unavailable` when no audio came at all, else the recording so far with reason `noCallback`). A configuration change rebuilds the engine (`capture.restarted`); the chosen device going away ends the recording with reason `deviceLost`, the default input follows the new default. `maxSeconds` ends it on its own (reason `maxSeconds`), a guard against a lost `capture.stop`. The app writes the WAV it keeps, from the frames.

**capture.start from a file (tests only).** With `VOXTNEW_HELPER_CAPTURE_FILE=<audio file>` in its environment the helper records that file instead of the microphone: any rate and channels AVAudioFile reads, through the same downmix, resampling, `PCMFramer`, `capture.started` (`device` is `file:<name>`, `deviceName` the file's name, `deviceSampleRate` and `deviceChannels` the file's), `capture.level`, `capture.pcm` and endings (`capture.stop`, `job.cancel`, `maxSeconds`) as a microphone, at a microphone's pace (`VOXTNEW_HELPER_CAPTURE_SPEED`, default 1, plays it faster), then silence until it is stopped. It never asks for the microphone, and has no watchdog or device watcher. **Only for tests** (`scripts/mac-e2e.sh`, `mac_e2e_test.go`): the app never sets it, and the helper says so on stderr when it is set.

**hotkey.configure.** It replaces every binding; an empty list removes the tap. A binding is a modifier-only chord (`key` absent: `{"fn":"any"}`, `{"fn":"any","shift":"any"}`, `{"command":"right"}`) or a key (`key`: a kVK code, 0 is A) with modifiers (`{"control":"any","command":"any"}` and 9 is ⌃⌘V); families are `fn`, `shift`, `control`, `option`, `command`, sides `any`, `left`, `right` (fn: `any`). Families not listed must be up. The recognizer judges a chord from the first modifier pressed to the last released: everything held in between is the chord (fn then ⇧ is fn+⇧), another key or a mouse button (an ⌥-click, an ⌥-drag) spoils it, a hold longer than `tapMaxMs` (1500) is no tap. Modifier-only chords are never swallowed (fn's own events aside, with `swallowFn`): a modifier held to type or click does what it always does and fires nothing. Sides come from the flags' device bits, so `{"option":"right"}` (VoxtNew's default, right ⌥) never matches the left ⌥, and ⌥-characters typed with the left key are untouched; an event without a device bit (a synthetic one) counts as the left key. The keys the helper posts itself (⌘V, ⌘C, `press`) carry `eventSourceUserData` 0x564F5854 ("VOXT"; Auralink's are 0x41555241, "AURA") and pass the tap unjudged; another app's posted keys are judged as typing. A chord bound to both a tap and a double tap fires the tap after `doubleTapMs` (the system's double-click interval, at most 800, else 500) unless a second press starts within it. `longPress` sends `down` after `longPressMs` (300) and `up` on release. Key bindings fire on key down and swallow their key down and up (unless `swallow` is false); modifier-only chords are never swallowed, but fn and the 🌐 key's events with `swallowFn`. With `active` (a session records) a tap fires at once and Escape fires `{"binding":"escape"}` and is swallowed. A modifier whose event comes without its state changing means macOS dropped one (seen: fn's release after fn+⌃): the chord in progress is dropped (`missedEvents`). The app flips `active` by sending the same bindings: a chord in progress survives a change of timings or `active`. Without Accessibility `installed` is false and the helper retries every 2 s, sending `hotkey.state` when it gets in; meanwhile the app falls back to MyGo's Carbon shortcuts for plain key combinations. `front` is the app in front as the key went down, the start of the session's context snapshot (plan §4.2); `t` is the helper's monotonic clock in ms.

**app.\*.** The helper keeps the app in front from NSWorkspace's activations and tells each change (`app.activated`), so the app need not poll. On activation it gives `AXManualAccessibility` to Electron apps (it finds them by their framework) and to the bundle ids of `app.configure`, once per process (Chromium builds a page's accessibility tree only when a client asks, a moment later), so the selection and the tab's address can be read when a shortcut comes; the outcome is a `debug` log line. `app.configure` replaces the list (case does not matter) and gives the app in front the attribute at once when it is listed (it came to the front before the list); a new helper has an empty list, so the app sends it after each start: its Chromium browsers (`appctx.TreeOnDemand`: the built-in Chromium ones and the user's) while app enhancement is on, none while it is off. The list comes from the app because the user adds browsers; the helper keeps it only for the activations. `app.list` lists the regular running apps and, with `installed`, the bundles in /Applications, /System/Applications and ~/Applications (one folder down too), each bundle id once, by name: the apps App enhancement offers as members.

**app.icons.** The icons the Finder shows (`NSWorkspace.icon(forFile:)`) of at most 48 apps (`proto.MaxIcons`; more is `bad_request`), each by its bundle's `path` when that is still there (app.list's), else the bundle the system finds for `bundleId` (`urlForApplication(withBundleIdentifier:)`): App enhancement's members carry only their bundle ids. Each is drawn off the main thread into a `size` × `size` pixel PNG (default 64, at most 256: 32 points at 2×), base64 in `png`; `png` is absent when no bundle was found (a removed app), and the app shows a generic icon. The answer has one icon per app asked for, in order. The app asks for the rows a list shows, 24 at a time, and keeps the PNGs (`internal/appicons`: `~/Library/Caches/VoxtNew/icons/<bundle id>-64.png`, redrawn after a week); a helper without the method (it is not in hello's `methods`) only means generic icons, so adding it kept protocol 1.

**textio.\*.** `textio.focus` reads the system-wide focused element (else the app in front's): `pid` and `app` are the process it belongs to (not always the app in front: a panel of another process), or the app in front with `error: "no focused element"`; `editable` when its value can be set, its role is a text role (`AXTextArea`, `AXTextField`, `AXComboBox`, `AXSearchField`), or it sits in a web page's editable region (`AXEditableAncestor`, WebKit and Chromium); `subrole: "AXSecureTextField"` is a password field. The app decides where a result goes (`internal/deliver`: the app in front when the result is ready, never a password field). `textio.selection` reads the focused element's selected text, the text of its selected range, then WebKit's and Chromium's text markers; a caret without a selection is `none` (no ⌘C: it would copy nothing, or a whole line). Otherwise, with `allowCopy`, it posts ⌘C, waits up to `copyTimeoutMs` (300) for the clipboard to change, reads it and puts the old clipboard back. The app decides `allowCopy` (not in apps that copy a line on an empty selection). `textio.paste` snapshots the clipboard (every item and type), writes the text (marked `org.nspasteboard.TransientType` and `AutoGeneratedType`, which clipboard managers skip), posts ⌘V, then the keys of `press` after `pressDelayMs` (120; app enhancement's auto key: Return is `{"key":36}`), and after `restoreDelayMs` (500) writes the snapshot back unless the clipboard changed meanwhile (`clipboardTaken`). With `targetPid` it pastes only when that app is in front, else answers `posted: false` with the app in `front`; the app reactivates the target itself when it must (`macx.Activate`: only the active app can hand activation over). Posted keys carry the helper's marker, so its own tap lets them through. Without Accessibility, `permission_denied`.

**textio.paste's check** (`verifyMs`, 10-20; the app sends 1000, `deliver.DefaultVerifyMS`). A posted ⌘V says nothing about where the text went: on 10-20 four pastes into a web page in Arc were all `posted: true` and the user saw at least one of them do nothing. With `verifyMs` the helper reads the focused element before ⌘V (its `AXValue`: length in UTF-16 and a digest, only `AXNumberOfCharacters` past 200 000; its `AXSelectedTextRange`; the bytes of its `AXSelectedTextMarkerRange`, WebKit's and Chromium's caret, which a web area has without value or selection) and again every 25 ms after it, up to `verifyMs`, until a reading differs (`PasteChecker`, VoxtNewHelperKit, tested by `PasteCheckTests`). `check` is `inserted` (`checkMs`: when the change showed), `unchanged` (nothing changed by the end: `checkMs` is the wait), or `unknown` (the element has neither value, selection nor text marker, or another element has the focus at the end). `field` gives the role and both readings (`marker: true` when the text marker was read), for the log (no text). Then: `inserted` presses `press` once `pressDelayMs` has passed since ⌘V and restores the clipboard 150 ms after ⌘V (the app read it: its field changed); `unknown` does as without the check; `unchanged` skips `press` (`pressSkipped`: Return must not send what was there before) and leaves the text on the clipboard (`kept`; `restored` false), so that the user can paste it and a paste the app reads late still gets the text. `secureInput` is the app that had secure event input on when ⌘V was posted (`IsSecureEventInputEnabled`, the login session's `kCGSSessionSecureInputPID`; `pid` 0 when unknown), which may keep posted keys from the app in front; absent when it was off. The app decides what an `unchanged` paste means (`internal/deliver`: not inserted, the answer card).

**browser.url.** Accessibility only (no AppleScript, so no Automation consent per browser; the user's decision, 2026-10-15), the same way for every app: the `AXURL` of the outermost `AXWebArea` around the focused element (`ax-focus`: the tab being typed in, also in a split view, not an iframe inside it), else the focused window's `AXDocument` (`ax-document`: Chromium browsers and Safari), else the `AXURL` of the window's first web area, breadth first over at most 400 nodes (`ax-weburl`: Arc). `pid` (the shortcut's `front`) picks the process, else `bundleId`'s first running one, else the app in front. No URL is `via: "none"` with `error` saying why (not running, no focused window, no web area: Safari in the background has none). When no web area has an address in an Electron app or one `app.configure` listed, the helper sets `AXManualAccessibility` on the process (once per process, besides the activation's), waits 150 ms and reads once more (the same 400 nodes): the result is the second read's, with `retried: true`. Arc in front before the helper started had no page tree at all (2026-10-15: `none` after 400 nodes in 0.7 s; retried, `ax-focus`; then 2 ms). `AXEnhancedUserInterface` is not set: Chromium needs only the manual attribute, and the enhanced one slows window animations. Each element answers within 0.3 s; the app adds its own timeout. Which apps are browsers is the app's business (`internal/appctx`); a URL is used only to match app enhancement rules, never sent to a model.

**output.\*.** The system's output muted while a session records (Settings → Audio → Mute system audio while recording; `internal/outmute` decides, docs/behavior.md §9.3; ported from Auralink). Only `kAudioDevicePropertyMute` (output scope) of the default output device changes: nothing of the system's sound is read, so no permission is needed. A device is found by its `uid` (Core Audio's `DeviceUID`): its `id` is reused by another device after it is unplugged and changes when it is reconnected; `name` is for the logs of the helper only. `output.watch {on: true}` answers the default output and its mute (`muted` absent: it has none to read, or the device is gone; `settable`: the helper can set it) and from then on sends `output.changed` when the default device or its mute changes (Core Audio's listeners, read 20 ms after the first call, sent only when the state differs from the last one sent); `on: false` stops. `output.mute` mutes the device and takes a **claim** on it, unless it is muted already (`took: false`: a mute the user made, or another app's, is never claimed, so never undone); `unsupported` when it has no mute the helper can set, `device_unavailable` when no device has the UID now. The claim is written to `<journal>/<pid>.json` (`{"pid", "devices"}`) before the mute; a journal that cannot be written mutes nothing (`internal`). `output.release` gives a claim back, with `unmute` unmuting the device if it is still muted; without a claim on it the helper does nothing, so the app may release a mute whose answer it never got. The helper undoes its claims (unmuting each device still muted) and removes its journal when stdin closes, on `helper.shutdown`, on SIGTERM, SIGINT or SIGHUP, and before the Mac sleeps (`NSWorkspace.willSleepNotification`: it does not wait for the app, and sends `output.released`). `output.recover` undoes the journals of helpers that are gone (killed by the supervisor's restart, crashed, or killed with the app): a file whose `pid` is still a running process is left alone (another VoxtNew, a development build); only `<number>.json` files count; each device still muted is unmuted (`restored`), and the files are removed. The app sends it after each `helper.hello`, before any `output.mute`, with the journal in `~/Library/Caches/VoxtNew/output-mute` (MyGo's cache directory). A helper without these methods (not in hello's `methods`) only means the output is never muted, so adding them kept protocol 1. Rules and journal are the same in the Swift helper (`MuteClaims`, `VoxtNewHelperKit/OutputMute.swift`) and the fake (`helpertest/output.go`).

## 5. Testing

- `go test ./internal/helper/proto`: every fixture line decodes strictly with the Go types and encodes back to the same JSON; the sequencing rules hold (one response per request, a job's notifications before its response and after `capture.started`, frames without gaps, the result counting them); every method has a fixture; the handshake lists every request.
- `cd helper && swift test` (on a Mac): every fixture line decodes with the Swift types and encodes back to the same JSON; the hotkey recognizer and the PCM framer.
- `VOXTNEW_MAC_E2E=1 go test -run TestMac -v .` (on a Mac; `scripts/mac-e2e.sh`): the real supervisor and the real helper — its read-only methods, the file input above against the speech sample, and a whole session over the fake local models (`TestMacE2ESessionLocal`; the real models: `TestMacE2EOmni`, `scripts/mac-regress.sh`) — without a window, a key, a tap or the clipboard (manual-test.md, "自动真机测试（无界面）").
- `go test ./internal/outmute`: the mute's rules over a helper in process (`helpertest.OutputHelper`), and the real supervisor against the fake helper process: the app going away (stdin closed: the helper unmutes), the helper crashing right after a mute (`crash-mute`: the next one recovers its journal), both dying, the output changing (`out-switch=`), the user unmuting (`out-unmute=`), devices without a mute (`out-default=`), an older helper (`no-output`). `cd helper && swift test` runs the same rules against a pretend Core Audio (`OutputMuteTests`). `VOXTNEW_MAC_E2E=1 VOXTNEW_MAC_OUTPUT=<UID> go test -run TestMacHelperOutputMute .` (opt-in, on a Mac): the real helper and Core Audio, each mute under a second (manual-test.md §10).
- `go test ./internal/helper`: the supervisor against `helpertest`, the fake helper the test binary becomes (`VOXTNEW_FAKE_HELPER`): handshake, capture with PCM frames (a tone, or a WAV played back), cancel, a stuck cancel, crash and restart, protocol mismatch, an app updated on disk (no helper started), missing executable, permissions, hotkey events and scripted notifications, text I/O, call traces, fixture replay.

## 6. Differences from the spikes' draft

| Draft (research/spikes.md) | Protocol 1 | Why |
|---|---|---|
| `capture.start` answers at once; `capture.stop` returns the statistics; `capture.ended` reports a failure | `capture.start` is a job: `capture.started` at the first audio, its response when it ends (stopped, maxSeconds, deviceLost, noCallback), `job.cancel` | The supervisor's job model (routing by `job`, cancel, a crash failing the call); one place says how a recording ended |
| `{session, device?}` | `{devices?}`, the user's preference list (settings `Audio.Devices`) | The settings keep a list; the helper falls back along it |
| `capture.pcm {session, seq, offset, samples, pcm}` | `{job, seq, offset, data}` | `samples` is `len(data)/2`; `data` like AudioSmith's chunks |
| `capture.started` was the start's answer, timed to `engine.start()` | sent with the first audio | That is when the user's words are heard |
| `watchdogSeconds`, `simulateSilentStarts` | fixed 1.2 s; no test hooks | The fake helper covers the cases on Linux |
| `hotkey.event {…, lagMs}` | no `lagMs` | `CGEvent.timestamp` is not on `CLOCK_UPTIME_RAW`'s base: the value was always out of range |
| `hotkey.event` without context | `front`, as the draft recommended | The session's context snapshot |
| defaults `doubleTapMs` 400, `tapMaxMs` 1000 | 500 (or the system's, at most 800), 1500 | The real-key retest (spike e01bb05): double taps judged by the second press; ⇧+fn held 983 ms |
| `helper.ping`, `helper.usage`, `helper.quit`, `debug.*`, `hotkey.raw` | `helper.shutdown`; no ping, usage or debug methods | Spike instrumentation; AudioSmith's handshake and shutdown |
| `permissions.status {…, listenEvent, postEvent}` | `{microphone, accessibility}` | Input Monitoring is not needed; `CGPreflightPostEventAccess` stays false while posting works |
| `textio.frontmost` | `app.frontmost`, and `app.activated` | Grouped with the activation notifications |
| `textio.activate`, `textio.paste {targetPid}` activating the target | no activation in the helper; `targetPid` only checks the app in front | macOS 14 activates cooperatively: the active app (VoxtNew) must yield, so `macx.Activate` runs in the app |
| `textio.paste {restore, pressReturn, returnDelayMs, source, tap}` | `{keepClipboard, press: [{key, modifiers}], pressDelayMs}`; HID source and tap fixed | Auto keys other than Return (⌘Return); the spike's variants were measurements |
| `textio.selection {enableChromiumAX, skipAX}` | done at activation (Electron apps); no `skipAX` | The tree needs time to appear: at activation it is ready when the shortcut comes |
| `browser.url {appleScript, ax, noCache, enableChromiumAX}` | `{bundleId?, pid?}`, Accessibility only (M5): the web area around the focus first | No Automation consent to ask per browser (the user's decision); the focused web area is the tab being typed in |
| `automation.probe`, `automation.request` | removed (M5) | Nothing runs AppleScript any more |
| — | `app.list` (M5) | App enhancement picks member apps from the running and installed ones |
| — | `app.icons` (10-15) | The lists that name apps show their icons; the app has no AppKit of its own (no cgo), so the helper draws them |
| — | `app.configure {manualAccessibility}`; `browser.url`'s `retried` | Chromium browsers build a page's tree only when asked (Arc read none); the browser list is the app's (the user adds some) |
| — | `output.watch`, `output.mute`, `output.release`, `output.recover`; `output.changed`, `output.released` (Auralink's, 10-17) | The output muted while recording (Voxt's `SystemAudioMuteController`), with the policy in Go and claims the helper keeps in a journal, so that no crash leaves the output muted |
