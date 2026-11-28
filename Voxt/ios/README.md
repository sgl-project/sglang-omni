# Voxt for iPhone

The iOS app of Voxt, milestone 1 of [#4684](https://github.com/sgl-project/sglang-omni/issues/4684): a standalone app that records, recognises the speech and then transcribes, translates or rewrites it with the user's language model. Everything for iOS lives in this directory; the macOS app's code is unchanged.

## What it shares with the macOS app

The features' logic is the macOS app's Go code, not a port of it. `voxtcore/` calls the same packages the macOS session uses:

| Step | Package | Behaviour kept |
|---|---|---|
| Prompts | `internal/compose` | the layered prompt (role, input policy, clean-up, structure, task, dictionary notes, glossary, output contract), English and Chinese templates, long speech cleaned up in parts |
| Language models | `internal/llm` | the presets (OpenAI, Anthropic, Gemini, DeepSeek, Ollama, ... and any OpenAI-compatible server), retries and timeouts, the connection test, thinking stripped from answers |
| Answers | `internal/compose` | `CleanOutput`, `CleanRewrite`, `ParseAnswer`: Translate asks again strictly when the answer looks untranslated; Rewrite from scratch answers with a title |
| Snippets | `internal/snippet` | a trigger alone gives its text as saved; triggers inside speech are marked for the model and restored after it |
| Remote speech | `internal/asr/openai` | any OpenAI-compatible `/v1/audio/transcriptions`, with the dictionary's terms as hints |
| Languages | `internal/lang` | the language table with English and Chinese names |

The steps follow `internal/session` (`run_transcribe.go`, `cleanup.go`, `write.go`, `snippets.go`); `voxtcore/process.go` names the function each one mirrors.

## Layout

```
ios/
├── voxtcore/          Go: the shared core behind one JSON call per operation (Calls), tested on any platform
│   └── cabi/          Go: its C ABI (VoxtCoreCall, VoxtCoreCancel, VoxtCoreFree), built as a static library
├── VoxtKit/           Swift package
│   ├── Sources/CVoxtCore/   the C ABI's header
│   ├── Sources/VoxtKit/     the bridge (VoxtCore), settings, Keychain, recording, speech engines, the session, history
│   ├── Sources/VoxtUI/      the SwiftUI screens and their strings (en, zh-Hans)
│   └── Tests/               Swift Testing: the session with fakes, the stores, the bridge against the real Go core, the strings
├── App/               the app target: project.yml (XcodeGen), the entry point, the permission prompts
│   └── UITests/             XCUITest: every feature driven in the simulator against fake_server.py
└── scripts/           build-core.sh (the Go core per SDK), test-kit.sh (VoxtKit's tests on a Mac)
```

The session (`VoxtKit/Session/VoiceSession.swift`) is the iOS counterpart of `internal/session`: idle → recording → recognising → composing → done or failed. The speech engines are behind `SpeechEngine`:

- **On this iPhone**: Apple's speech recognition (`SFSpeechRecognizer`), with "Only on this iPhone" on by default, so the audio never leaves the device; a language the iPhone cannot recognise offline fails with a message instead of going to Apple.
- **Remote speech service**: an OpenAI-compatible transcription API, through the Go core.

Keys stay in the Keychain (`llm.<preset>`, `asr.remote`, the macOS app's account names), readable after the first unlock and never synced off the device. Settings and history are JSON files in Application Support with file protection.

## Build and run

Needs Xcode 16 or later, Go (`GOTOOLCHAIN=auto` fetches the version in `../go.mod`) and [XcodeGen](https://github.com/yonaskolb/XcodeGen).

```sh
cd Voxt/ios/App
xcodegen
open Voxt.xcodeproj
```

Pick a signing team and run on an iPhone or the simulator. The pre-build step runs `scripts/build-core.sh xcode`, which builds the Go core for the SDK being built into `ios/build/<sdk>/`. On first launch, set up a language model in Settings → Language models. The Go core is built for arm64 only, so the simulator needs an Apple silicon Mac.

## Tests

```sh
cd Voxt
CGO_ENABLED=0 go test ./ios/...     # the core: every feature's step against a fake OpenAI-compatible server on loopback
ios/scripts/test-kit.sh             # builds the core for this Mac, then VoxtKit's Swift tests against it
```

`test-kit.sh` also runs with only the Command Line Tools: it adds their Swift Testing and builds against their macOS 26 SDK, because the newer SDK's SwiftUI needs macro plugins that only Xcode has.

The UI tests (`App/UITests`) drive the app in the simulator: the three screens, the language model's connection test, translating typed text into the history, transcribing a recording through a remote speech service, and Rewrite over a text and from scratch. They talk to a fake OpenAI-compatible server on this Mac, so they need no account or key:

```sh
cd Voxt/ios
python3 App/UITests/fake_server.py &
cd App && xcodegen
xcodebuild test -project Voxt.xcodeproj -scheme Voxt -destination 'platform=iOS Simulator,name=iPhone 17'
TEST_RUNNER_VOXT_UI_LANGUAGE=zh-Hans xcodebuild test -project Voxt.xcodeproj -scheme Voxt -destination 'platform=iOS Simulator,name=iPhone 17'
```

The recording tests use this Mac's microphone: allow Simulator in System Settings → Privacy & Security → Microphone. Keep code signing on (the simulator signs locally): without it the Keychain refuses the keys.

## Not in milestone 1

- **Voxt's own speech models on the iPhone** (Whisper small, Qwen3-ASR 0.6B on the Omni native MLX runtime). iOS apps cannot start the `*_server` processes the macOS app supervises, so this needs the runtime built for iOS and served in the app's process; it plugs in as another `SpeechEngine`.
- **A keyboard extension** for dictation in other apps: its memory budget is small and it cannot use the GPU (#4202), so it would record and hand off to the app or a remote service.
- **Sync with the macOS app** (settings, dictionary, prompts, snippets): `AppSettings` keeps the macOS field names where they mean the same, for the format still to agree.
- **Streaming answers**, App enhancement, meetings, the ChatGPT sign-in (its preset is left out) and the app icon.
