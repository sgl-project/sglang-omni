# Meeting Notes

This document explains Voxt's Meeting Notes mode: what it does, how to configure it, which engines it supports, and what to expect from the current beta implementation.

## Overview

Meeting Notes is a separate long-running capture mode for meetings, calls, interviews, and podcasts.

<video src="https://github.com/user-attachments/assets/1ede04a2-1348-483e-b487-22561ad02a77" controls preload="none" width="100%"></video>

- Starts with the dedicated meeting shortcut
- Opens a floating meeting card instead of the normal transcription card
- Does not paste text into the focused input
- Saves results into a dedicated `Meeting` history type
- Can open a detail window for timeline review, translation, and export

Current beta uses source-based separation:

- microphone audio is labeled as `Me`
- system audio is labeled as `Them`

This is not true diarization yet.

## How To Enable It

1. Open the main window, then go to `General > Output`
2. Turn on `Meeting Notes (Beta)`
3. Grant the required permissions:
   - microphone access when using a microphone source
   - Accessibility for global shortcuts, not Input Monitoring
   - macOS requests system audio capture when starting a meeting with system/mixed audio. If denied, enable access in System Settings and retry. Microphone-only recording, file import and ordinary output-device mute do not need it

After enabling it:

- the meeting shortcut appears in the main window's `Hotkey` page
- the `Permissions` page explains meeting system audio and links to System Settings; it does not infer authorization through private APIs
- meeting history becomes available in History filters

## Supported ASR Engines

Meeting Notes follows the current global transcription engine.

### Whisper

- Supported
- Uses the current Whisper model
- Uses the existing Whisper `Realtime` setting
  - `Realtime ON`: lower-latency meeting updates
  - `Realtime OFF`: quality-first chunked meeting transcription

### MLX Audio

- Supported
- Uses the current MLX model
- Realtime-capable MLX models use lower-latency meeting updates
- Other MLX models use chunked meeting transcription

### Remote ASR

- Supported
- Uses the currently selected Remote ASR provider and configuration
- Meeting behavior now depends on provider family:
  - OpenAI: existing chunk pseudo-realtime path when enabled
  - Doubao ASR: uses a dedicated `Meeting ASR` model and chunk/file transcription path
  - Aliyun Bailian ASR: uses a dedicated `Meeting ASR` model and async/file transcription path
  - GLM ASR: chunked upload flow

For `Doubao ASR` and `Aliyun Bailian ASR`:

- `Meeting Notes` does not use the provider's live websocket path
- configure `Meeting ASR` separately in the main window under `Model > Remote ASR > [Provider]`
- if the meeting model is missing, meeting start is blocked
- use `Test Meeting ASR` to verify the meeting-specific request path

### Direct Dictation

- Not supported for meetings in the current beta

## Meeting Card

The meeting card is optimized for long-running capture.

- collapsible header-only mode
- pause / resume
- close with secondary confirmation
- per-segment timestamp list
- click to copy a segment
- optional realtime translation for `Them`
- live detail window

If a meeting already contains transcript content, ending it from a collapsed card will auto-expand first so the confirmation dialog is easier to use.

## Realtime Translation

Meeting realtime translation follows the app's existing translation model/provider settings.

- uses the same translation provider selection as normal translation
- uses the same fallback rules
- uses its own remembered target language for meeting mode
- translates only `Them` segments in the current UI

If the selected translation provider cannot be used directly for text translation in meeting mode, Voxt falls back through the normal resolver behavior.

## History, Detail Window, And Export

When a meeting finishes normally:

1. the meeting card closes
2. a `Meeting` history entry is saved
3. the meeting detail window opens automatically

The detail window supports:

- transcript review
- click a transcript segment to seek to its audio position
- click or drag the waveform timeline
- speaker ranges and speech-start markers on the waveform
- per-segment editing, deletion, and undo for the latest deletion
- persistent highlights for important segments
- translation on existing meeting segments
- export when the current mode allows it

Editing a segment clears its previous translation so it can be recalculated using the current translation settings. Existing summaries remain available but are marked as stale until regenerated. Deleting transcript text never deletes the meeting audio.

## Notes And Limitations

- Current beta is source-based (`Me` / `Them`), not full speaker diarization
- Meeting mode is isolated from normal transcription / translation / rewrite sessions
- Meeting mode uses a dedicated history type and detail flow
- Cold local-model startup can still take time on first use; Voxt shows model initialization state in the overlay
- For remote providers, Voxt keeps the meeting UI uniform, but transport differs by provider capability:
  - `Doubao` / `Aliyun` meetings use provider-specific chunk/file transcription models
  - other providers stay on their existing chunk-based meeting path
