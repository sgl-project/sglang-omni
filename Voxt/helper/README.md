# voxtnew-helper

The small Swift process VoxtNew runs beside its window, for what MyGo cannot do from Go:

- **capture**: record the microphone (AVAudioEngine) and stream mono PCM16 to the app (`capture.*`), falling back along the user's devices to the default input, with a watchdog for inputs that start without audio;
- **hotkeys**: watch the keyboard with a CGEventTap, so that `fn`, a single modifier, chords, double taps and long presses can be shortcuts, deciding on the tap's thread whether to swallow each key (`hotkey.*`);
- **text in and out**: what is focused in the app in front, its selected text (Accessibility, else ⌘C when the app allows), and pasting through the clipboard with keys after it and the clipboard put back (`textio.*`);
- **the app in front**: kept from NSWorkspace and told to the app (`app.frontmost`, `app.activated`, and `front` in each `hotkey.event`);
- **browser URLs**: the front tab's URL from the Accessibility tree only, with no Automation consent to ask (`browser.url`), and the apps to pick from for app enhancement (`app.list`) with their icons (`app.icons`);
- **permissions**: the Microphone and Accessibility as the helper sees them, and macOS's prompts (`permissions.*`);
- **the output's mute**: the default output device's mute while VoxtNew records (`output.*`), the mutes it made kept in a journal and undone when the app goes away, before sleep, or by the next helper.

It speaks [helper protocol 1](../docs/helper-protocol.md) over stdin and stdout; `internal/helper` supervises it (handshake, restarts, stderr to `~/Library/Logs/VoxtNew/helper.log`, call timings). The helper keeps no settings: Go decides (which device, when ⌘C may run, the delays, the keys) and passes it in the params. Its shapes come from the M0 spikes on a Mac (the project's `research/spikes.md`).

macOS checks the helper's permissions against VoxtNew.app, its responsible process: the prompts and System Settings name VoxtNew, and a helper outside the bundle (`VOXTNEW_HELPER`) has the app's grants too.

## Layout

```
helper/
  Package.swift              Swift 6, macOS 15, no third-party dependencies
  Sources/VoxtNewHelperKit/ pure logic, tested without permissions:
    Wire.swift, WireTypes.swift   the protocol (mirror of internal/helper/proto)
    HotkeyRecognizer.swift        chords, taps, double taps, long presses, swallowing
    PCMFramer.swift               PCM16 frames and levels
    CaptureSource.swift           downmix and resampling; the test-only file input
    Apps.swift                    app.list's merge; app.icons' sizes; the web area around the focus
    ManualAccessibility.swift     which apps get AXManualAccessibility, once per process
    OutputMute.swift              the output's claims and their journal (MuteClaims), the watch's changes
  Sources/VoxtNewHelper/    the executable:
    main.swift, Server.swift      stdin/stdout, hello, routing, the capture job, slow-request lines
    Capture.swift                 AVAudioEngine, devices, watchdog
    Hotkey.swift                  the event tap
    FrontApps.swift               the app in front, AXManualAccessibility (Electron apps, app.configure's browsers)
    TextIO.swift                  focus, selection, paste
    Browser.swift                 browser URLs (AX), app.list
    AppIcons.swift                app.icons: apps' icons as PNGs
    Permissions.swift             Microphone, Accessibility
    Output.swift                  Core Audio's output devices, their listeners, undoing on exit and sleep
  Tests/VoxtNewHelperKitTests/  FixtureTests (helper/Fixtures), the recognizer, the framer, the file input, ManualAccessibility, OutputMute
  Fixtures/                  golden transcripts, shared with Go; audio/ for the file input's tests
  Support/                   Info.plist (linked into the binary), helper.entitlements
```

## Build and test

On a Mac with Xcode (from Linux through `scripts/mac.sh`):

```sh
scripts/build-helper.sh                  # release build, signed, into resources/darwin-arm64/helper/
scripts/build-helper.sh --out DIR        # elsewhere: a helper outside the app (VOXTNEW_HELPER)
cd helper && swift test                  # the Kit: fixtures, recognizer, framer
```

`go tool mygo build` ships `resources/darwin-arm64/helper/voxtnew-helper` as `VoxtNew.app/Contents/Resources/helper/voxtnew-helper`, signed with `Support/helper.entitlements` (mygo.json `macos.helperEntitlements`: the microphone under the hardened runtime). The version in `Support/Info.plist` follows mygo.json's (the build script checks).

Without Xcode, the fake helper stands in (it records a tone): `go build -o /tmp/fakehelper ./internal/helper/helpertest/cmd/fakehelper`, then `VOXTNEW_HELPER=/tmp/fakehelper`.

## Recording a file instead of the microphone (tests only)

`VOXTNEW_HELPER_CAPTURE_FILE=<audio file>` makes `capture.start` play that file (any rate and channels) through the capture path a microphone takes — downmix, resampling, framing, `capture.level` and `capture.pcm`, the same endings — at a microphone's pace (`VOXTNEW_HELPER_CAPTURE_SPEED=4` plays it four times as fast), then silence until `capture.stop`. The microphone is never asked for, so it runs on a Mac nobody sits at, from an ssh session. It exists for the headless Mac tests (`scripts/mac-e2e.sh`, `mac_e2e_test.go`; docs/helper-protocol.md §4); the app never sets it.

```sh
printf '%s\n' '{"id":"1","method":"helper.hello","params":{"protocol":1,"app":"cli"}}' \
  '{"id":"s1","method":"capture.start","params":{"maxSeconds":2}}' \
  | VOXTNEW_HELPER_CAPTURE_FILE=testdata/speech/sample-zh-en.wav helper/.build/debug/voxtnew-helper | cut -c1-120
```

## Trying it by hand

The helper answers on a terminal too (its permissions are then the terminal's, not VoxtNew's):

```sh
printf '%s\n' '{"id":"1","method":"helper.hello","params":{"protocol":1,"app":"cli"}}' \
  '{"id":"2","method":"capture.devices"}' '{"id":"3","method":"browser.url","params":{"bundleId":"com.google.Chrome"}}' \
  | helper/.build/debug/voxtnew-helper
```

In the app, the shortcuts work as soon as the helper is ready: fn (or what Transcribe's page binds) records, and the logs panel (source helper, debug level) lists each event with the app in front.
