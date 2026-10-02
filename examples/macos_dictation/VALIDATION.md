# Validation and contribution notes

## Deterministic checks

Run `bash examples/macos_dictation/local_setup_test.sh` for the shell installer and
launcher. It uses temporary paths, fake HTTP responses and command boundaries;
it does not install packages, download models or launch the real app. It covers
read-only dry runs, native Homebrew prefix validation, cached model lookup,
service/model validation, process ownership and shortcut conflict preservation.
A clean-machine installation with real package registries, model downloads and
macOS permissions remains a separate manual check.

Run `bash examples/macos_dictation/verify_all_test.sh` on an Apple Silicon Mac.
The suite fails on the first failed check. It does not download models, record
microphone audio, read another application's fields, or send real key events.
Clipboard tests use isolated named pasteboards; preferences tests use disposable
UserDefaults suites. The tests do not read the application's saved personal background.
The runner compiles all offline fixtures with Swift 5, arm64 and a macOS 14 deployment
target, sharing one module cache. It accepts optional fixture names for focused runs,
such as `bash examples/macos_dictation/verify_all_test.sh ClientComposition HTTPTransport`.
Original individual scripts and the older `run_test.sh` subset remain unchanged.

For the polishing and service-readiness regressions, run
`bash examples/macos_dictation/review_fixes_test.sh`. It selects the relevant
fixtures from the same offline suites used by CI, including their HTTP fallback
and configuration integration checks.

Run `bash examples/macos_dictation/lifecycle_test.sh` for warmup and backend
lifecycle regressions. The real `ClientState` and intercepted HTTP verify saved
startup preferences, preemption before foreground polishing, foreground takeover,
and a model change during recording. Shell tests use a simulated clock to check
ASR readiness beyond 300 seconds, configured budgets and timeout failures. Disposable
Python spawn/daemon workers exercise parent crashes, graceful launcher shutdown,
forced cleanup and isolation from unrelated processes. These fixtures do not load
models or measure Metal memory reclamation. CI runs this entry point separately.

The suite includes `Timing_test.swift`. It uses a simulated monotonic
clock to verify ASR/LLM/total boundaries, failed-request durations, skipped/disabled
stages and cancellation across rounds. Its numbers are fixtures, not performance
measurements. CI invokes the same complete entry point as local development.

`RecordingLimit_test.swift` checks one-minute capture
and import boundaries using synthetic audio, including retention after the previous
30-second boundary and rejection of oversized files. It exercises the automatic-stop
path with an injected short limit and never opens the microphone.

`CompletionFeedback_test.swift` checks that empty
recognition, imports and failures dismiss after a readable hold, while active work
stays visible. It covers hover/re-entry, cancellation, retained error details and
unchanged delivery state without any real input events or clipboard access.

`ClientComposition_test.swift` constructs the real `ClientState` with a fake recorder,
intercepted HTTP transport, disposable preferences and a fake text-target factory.
It covers start/stop through the menu/hotkey entry method, one insertion, target-mode
changes between rounds, cancellation during authorization and HTTP, empty results,
target-capture failures, and saving service configuration during a recording. It does
not register a real global shortcut or exercise the native event dispatcher.

`HTTPTransport_test.swift` covers success, non-2xx/non-HTTP responses, malformed ASR
JSON, ASR model-list readiness, timeout/connection errors, cancellation propagation
and raw-text fallback after HTTP, decoding or fidelity failures in Ollama.
It also starts a disposable HTTP fixture bound only to `127.0.0.1` on an
automatically allocated port. Actual URLSession requests verify
that 302/307 redirects are not followed, with a reachable destination as a positive
control. No existing model server is used. Sandboxed runners must permit loopback
sockets and the system pasteboard service; failures are not silently skipped.

Coverage includes:

- Session cancellation, stale responses, empty input, raw fallback and single delivery.
- Independent model/address validation, persistence, per-round configuration
  snapshots, configured health endpoints and warmup invalidation after model changes.
- Prompt quoting and prefix stability, repeated-word/ASCII-word-boundary preservation,
  matching question-mark presence, number/language/negation preservation,
  explicit spelling corrections, and preservation of code/path/operator syntax.
- Current-cursor and locked-target delivery, clipboard ownership, ambiguous paste,
  target changes, failures, and no automatic Return.
- Shortcut capture/registration rollback, settings persistence, and hover/fade
  cancellation, re-entry, fallback completion and reduced motion.

`build_client.sh` compiles the actual application through Swift Package Manager.
`build_compatibility_test.sh` checks the resulting arm64 architecture, macOS 14
minimum target and code signature. An ad-hoc signature is for a local build;
notarization and distribution signing are outside this example.

With full Xcode, also run:

```bash
swift test --package-path examples/macos_dictation --arch arm64
```

That XCTest target covers core session behavior, not the whole native app.
The standalone suite is available with Command Line Tools when XCTest is absent.
The GitHub workflow runs both on the documented arm64 `macos-15` runner; see
[GitHub runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).
It does not exercise microphone permission dialogs or real editor insertion.

## Voice correction checks

Voice-correction regressions run separately from the original suite:

```bash
bash examples/macos_dictation/correction_test.sh
bash examples/macos_dictation/revision_target_test.sh
bash examples/macos_dictation/correction_integration_test.sh
bash examples/macos_dictation/correction_feedback_test.sh
bash examples/macos_dictation/ollama_correction_test.sh
bash examples/macos_dictation/correction_plan_test.sh
bash examples/macos_dictation/spelling_correction_test.sh
bash examples/macos_dictation/correction_instruction_echo_test.sh
bash examples/macos_dictation/background_correction_test.sh
bash examples/macos_dictation/revision_confirmation_test.sh
bash examples/macos_dictation/revision_anchor_compatibility_test.sh
bash examples/macos_dictation/revision_selection_delay_test.sh
```

These use fake recording, intercepted HTTP and isolated pasteboards. They cover
UTF-16 edit spans, double-Option recognition, cancellation, changed targets,
clipboard ownership, shared-microphone exclusion, result-window transitions,
structured responses and the explicit whole-paragraph deletion guard.
The planner and Ollama fixtures cover the `scope`/`text` contract, static teaching
examples, exact quoted edits, literal/ordinal bounds, field-value preservation,
single-character checks, missing literal/contact evidence, exact homophone bounds,
local-versus-rewrite boundaries, one feedback retry,
clarification without delivery, and cancellation during either generation attempt.
The additional checks cover explicit spelling versus background suffixes and delayed
caret updates after the expected pasted text appears. A different draft still prevents
confirming an insertion.
The anchor compatibility regression also keeps a stale caret throughout confirmation,
checks an added final LF and a removed empty-editor LF placeholder, and verifies the
actual revision preserves the editor's suffix. It rejects a missing LF inside the
dictated span, a shifted span, unrelated edits and changed focus. These are simulated
editor checks, not evidence of compatibility with a particular external application.
It also covers a stale pre-paste snapshot predicting 20 UTF-16 units when the
changed field contains exactly the 15-unit dictated paragraph. Whole-field
confirmation resets the retained offset to zero and applies only the changed
fragment. Unchanged pre-existing text, duplicate paragraphs and unrelated text
do not qualify for this fallback.

The selection-delay regression models an AX setter returning before its selected
range is readable. It verifies one paste after acknowledgement and no paste when
the range never updates, the text/focus changes, or correction is cancelled.

The instruction-echo regression sends a contaminated model response through the
real correction session and checks that no draft write occurs. It also covers
punctuation/spacing changes, legitimate quoted replacement text, and the scoped
`S.G. Lang` / `L.A.N.G.` spelling case. With `OMNI_CORRECTION_LIVE_TEST=1`, its
runner additionally checks the same client-side spelling path without a model request. That is a functional check, not model-only accuracy evidence.

The optional local-model check uses synthetic text only:

```bash
OMNI_CORRECTION_LIVE_TEST=1 bash examples/macos_dictation/correction_live_test.sh
OMNI_CORRECTION_LIVE_TEST=1 bash examples/macos_dictation/background_correction_test.sh
OMNI_CORRECTION_LIVE_TEST=1 bash examples/macos_dictation/correction_accuracy_test.sh
OMNI_CORRECTION_LIVE_TEST=1 bash examples/macos_dictation/semantic_editing_live_test.sh
```

It checks name, number and negation edits against exact expected strings. It does
not establish broad model accuracy or test microphone, keyboard or editor delivery.
`OMNI_CORRECTION_URL` and `OMNI_CORRECTION_MODEL` override the documented defaults;
the URL must still pass the client's loopback-only validation. The accuracy and
semantic-editing scripts repeat their fixed development cases and report mean ±
sample standard deviation. Read every output: exact/keyword criteria are limited,
and a rejected edit only passes where a clarification was explicitly expected.
Service or response-format failures must not count as successful clarification.
These are diagnostic evaluations, not release gates or population accuracy estimates.

For two user-supplied audio files, run `audio_correction_test.sh` with
`OMNI_CORRECTION_AUDIO_TEST=1`, the original recording path, the correction recording
path and an optional expected full result. `--saved-settings` opts into reading the
installed app's saved configuration and enabled personal background. This test prints
transcripts, contacts local models, and applies the real revision to an in-memory draft;
it never sends input events or changes the user's editor, clipboard or preferences.
See [VOICE_CORRECTION.zh-CN.md](VOICE_CORRECTION.zh-CN.md) for an invocation example.

## Other opt-in live backend checks

Start the default local servers described in [README.md](README.md), then run:

```bash
bash examples/macos_dictation/polish_prompt_test.sh
bash examples/macos_dictation/personalization_live_test.sh
bash examples/macos_dictation/live_backend_test.sh /absolute/path/to/a-short-recording.wav
```

The first two use fixed synthetic utterances/backgrounds and do not load user
preferences. The last sends the supplied audio to the local ASR service; it does
not record new audio or paste the resulting text. Live scripts use the documented
default service configuration, independently of application preferences.

Report the exact checkout, macOS/Swift/backend versions, model IDs and precision
when publishing results. Distinguish raw ASR, accepted copyediting and fallback:
a correct fallback does not demonstrate that the LLM obeyed its prompt.
For performance measurements, report repeated observations as mean ± sample std,
with sample count, warm/cold conditions and audio duration. Mark single observations
explicitly; prefill/cache measurements are not end-to-end dictation latency.

## Manual acceptance before requesting merge

Voice correction has not yet passed end-to-end acceptance in Codex. A recent manual
run selected the intended word but stopped before paste because the selected range
was not confirmed. The bounded selection-acknowledgement fix passes simulated-editor
regressions; a fresh microphone-to-Codex replacement run remains pending. Do not treat
model output checks or in-memory draft tests as confirmation of external-editor delivery.

Use a non-sensitive scratch draft in each target application. Record which
application and version were tested, input mode, expected result and actual result.

| Scenario | Expected behavior |
| --- | --- |
| Start/stop using the configured hotkey | One recording, one paste, no automatic send |
| Existing draft and selection | Paste follows the editor's normal insertion/replacement behavior |
| Move focus while processing in current-cursor mode | Paste goes to the focus active at completion |
| Change original target in locked mode | Automatic insertion stops with a visible explanation |
| Cancel during recording/ASR/polishing | No later result or paste from that round |
| Hover before completion, during hold, or during fade | Remains visible; leaving fades; re-entry restores |
| Model failure or rejected copyedit | Original text is retained; completed fallback may fade |
| Empty recognition, file import, or failed/skipped insertion | Completed notice fades after a longer hold; hover keeps it; result details remain available |
| Finish with polishing enabled/disabled or with a failed request | Bar shows attempted stages; result view shows total and accurate statuses; no fake zero for unexecuted stages |
| Save model/background changes during a round | Current round unchanged, next round uses new configuration |
| Restart after changing settings | Shortcut, model configuration and explicit preferences restored |
| Backend unavailable or accessibility denied | Actionable status; no false claim of confirmed insertion |
| Double-tap the same Option key after dictation, speak an edit, then double-tap again | One correction recording and one update of the verified previous span; no automatic send |
| Use Option+Space, hold Option, or alternate left/right Option | Does not accidentally start correction |
| Edit the original paragraph or switch away during correction | Automatic replacement stops; corrected text remains available to copy |
| Edit a name, a number, or a negation with polishing disabled | Explicit correction still uses Ollama and may make the requested semantic change |
| Correct twice, including Unicode text and a deletion | Each edit uses the latest confirmed span and preserves surrounding text |
| Copy something else while correction is being delivered | The user's newer clipboard content is retained |

The original client has been exercised locally in Codex with user-confirmed
insertion and no automatic send. That evidence is not a guarantee of compatibility
with every editor, operating-system release, or future build. New UI/layout and
cross-application combinations need fresh manual acceptance.

## PR scope

Keep the contribution under `examples/macos_dictation`, its examples index entry,
and its scoped macOS workflow. Do not include build outputs, recordings, personal
backgrounds, local development diaries or signing credentials. Keep ASR runtime
changes in separate PRs.

Use the repository PR template. Link the Apple Silicon roadmap as `Related to #1967`;
this client does not complete that roadmap. Include the commands actually run,
manual coverage and any remaining limitations. A configured workflow is not a
successful GitHub Actions run until it has executed on the PR.
