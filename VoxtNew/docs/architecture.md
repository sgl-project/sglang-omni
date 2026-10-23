# VoxtNew architecture

*State: M1 (2026-10-17): the speech models run on the Omni native MLX servers (internal/omni), downloaded by internal/models; the language models run in voxtnew-llm, a Swift process of their own (internal/llmrt, M2). M0 was the skeleton: Auralink copied, renamed and stripped of its remote services. Plan and milestones: [plan.md](plan.md). Behaviour compared with Voxt: [behavior.md](behavior.md). Helper wire format: [helper-protocol.md](helper-protocol.md); the language models' process: [llm-protocol.md](llm-protocol.md), its models and their measurements: [llm-models.md](llm-models.md).*

VoxtNew is a Go app on MyGo (native UI, no cgo) with one small Swift helper process, on local models only. It is a copy of Auralink (@748cf55, the same author's app on remote models), whose code it keeps but for the remote layer (§5): sessions get their speech and language models from `internal/local`, which M1 (the Omni native MLX servers) and M2 (the `voxtnew-llm` process) fill in. Auralink itself reused AudioSmith's engineering (AudioSmith @f171c86, Apache-2.0): its window shell, UI kit and tokens, settings, store, i18n, logs and notification code, and its engine supervisor reduced to one helper. Milestone names up to M6 in comments are Auralink's; VoxtNew's are M0–M4 (plan.md).

## 1. Package map

```
github.com/sgl-project/sglang-omni/VoxtNew
├── *.go                  package main: the UI (views + view state) and the wiring
├── internal/
│   ├── core/             event bus, ULIDs, error sentinels, notifications and log lines        (leaf)
│   ├── settings/         the typed, versioned settings document (v1) + Service                 (leaf)
│   ├── store/            persistence contract: Documents, History, Dictionary
│   │   └── sqlitestore/  SQLite (MyGo plugin): migrations, backup, FTS5 trigram, maintenance
│   ├── helper/           supervisor of voxtnew-helper: handshake, restarts, stderr to the log, call traces
│   │   ├── proto/        the helper protocol's wire types (leaf); fixtures in helper/Fixtures/
│   │   └── helpertest/   a fake helper for tests (the test binary re-runs itself); cmd/fakehelper
│   ├── i18n/             T / N / X keyed by English source; locales/*.json; dates              (leaf)
│   ├── icons/            Solar icons (names.txt → solar_gen.go)
│   ├── logs/             log ring + rotated files
│   ├── notify/           the notification centre (kept in the store's documents)
│   ├── lang/             the language table, 652 languages                                     (leaf)
│   ├── permission/       Microphone and Accessibility (purego on darwin; Fake for tests)       (leaf)
│   ├── macx/             activating an app again before a paste (purego on darwin); the       (leaf)
│   │                     recording's sounds (afplay, Voxt's presets and how long each is
│   │                     heard); the system's web proxies (CFNetwork);
│   │                     a process's memory (proc_pid_rusage)
│   ├── asr/              the shape of speech recognition: FileTranscriber, Streamer/Stream, error
│   │                     kinds, WAV/PCM helpers (the models are internal/local's)
│   ├── llm/              chat models (from AudioSmith): request and response types, skills, Backend,
│   │                     Client (retries, JSON repair, thinking stripped), schemas, usage
│   ├── local/            the local models as sessions open them: Services (Installed, ASR, LLM),
│   │                     Library (the installed models of internal/models, opened by Openers: the
│   │                     runtimes' ASROpener and LLMOpener; ErrNoRuntime without one), Choose,
│   │                     ForFeature, ErrNotInstalled; NotInstalled; VoiceActivity (a speech
│   │                     provider that brings its VAD model)
│   ├── models/           the model catalogue (six curated models pinned to the family's commits,
│   │   │                 every file's SHA-256; two borrowed family LLMs), the models in the
│   │   │                 ActNow family's library ($HUB, the Hugging Face cache) through
│   │   │                 actnowkit's hfhub (downloads: mirrors raced, ranges, resume, the
│   │   │                 family's locks and claims) and models (claims, deletion decisions,
│   │   │                 tombstones, reuse survey, migration); the downloads' proxy
│   │   │                 (SystemProxy, ProxyFor); Manager                    (actnowkit only)
│   │   └── modelstest/   Seed: installed models of sparse blobs in a $HUB layout, for tests
│   ├── actnow/           VoxtNew in the ActNow family (spec v1.0.3, actnowkit): the family
│   │                     directory, the manifest, discovery of the other apps, shared settings
│   │                     (language, proxy, Hub endpoints, model folder: follow, migrate, mirror),
│   │                     shortcuts declared and their conflicts, the runtime file (loaded models,
│   │                     budget, memory requests, the paste flag)
│   ├── resident/         a model runtime's events (loaded, busy, idle, unloaded) in order on a
│   │                     goroutine of their own (Queue), from each process slot's Tracker:
│   │                     omni and llmrt report them, the app writes the family's runtime file
│   ├── omni/             the Omni native MLX servers (M1): Runtime (one qwen3_asr_server per model,
│   │                     ready line, start timeout, idle unload (SetIdleTimeout), Unload, restart after death,
│   │                     stall kill, Close; OnModel, BeforeLoad), ASR (asr.FileTranscriber + asr.Streamer: Voxt's
│   │                     Qwen3-ASR Final policy, the /v1/realtime preview; WithVAD), VAD
│   │                     (vad.Model: Silero), the curated models and their tuning (models.go)
│   │   └── omnitest/     a fake qwen3_asr_server for tests (the test binary re-runs itself)
│   ├── llmrt/            the language models (M2): Runtime (supervises voxtnew-llm: started on
│   │   │                 first use, handshake, started again after a crash, idle stop (SetIdleTimeout), Unload,
│   │   │                 UnloadIdle, Stop; OnModel, BeforeLoad: resident.go), Backend (an llm.Backend over llm protocol 1, deltas on the caller's
│   │   │                 goroutine), Open (local.Chat: llm.Client over it, warmed up), the curated
│   │   │                 models' generation settings (models.go), Hy-MT2's own prompt (hymt.go),
│   │   │                 the VOXTNEW_DEV_LLM hook (dev.go)
│   │   ├── wire/         llm protocol 1's wire types (leaf); fixtures in llm/Fixtures/
│   │   └── llmrttest/    a fake voxtnew-llm for tests (the test binary re-runs itself)
│   ├── session/          the one owner of a voice session: the state machine (reading →
│   │                     recording → recognizing → composing → delivering → done/cancelled/
│   │                     failed), the selection, Translate's and Rewrite's streaming answer, its
│   │                     snapshots, the history entry; adapters over the helper (helperio.go)
│   ├── vad/              local voice activity detection (energy over an adaptive noise floor):
│   │                     Analyze, Compact (file services), Gate (streams), Merge; Model, the
│   │                     interface of a VAD model (Silero, internal/omni) the gate can follow  (leaf)
│   ├── dsp/              the sound of what a speech service gets: a high-pass filter (4th-order
│   │                     Butterworth, 80 Hz), one gain from the speech (files) or a slow one
│   │                     (streams, Leveler), a peak limiter; pure Go
│   ├── compose/          the layered prompt composer: Transcribe, Translate, Rewrite;
│   │                     templates/{en,zh-Hans}; 112 golden prompts; Untranslated, ParseAnswer;
│   │                     long speech in pieces (Pieces, BuildParts, JoinParts)
│   ├── deliver/          the insert transaction: textio.focus, then textio.paste or the answer card
│   ├── hotkey/           the shortcuts' grammar, bindings for hotkey.configure and the Carbon
│   │                     fallback, conflict checks, routing an event to an action
│   ├── appctx/           app enhancement: groups (document appctx:rules), URL rules and their
│   │                     matching, the order URL → the browser's group → the app's group → none,
│   │                     the built-in browsers, auto keys
│   ├── appicons/         apps' icons for the lists that name apps: which are wanted, asked for,
│   │                     there or missing; the PNGs on disk (a week fresh); fetching a batch
│   ├── outmute/          the system's output muted while a session records: when (after the start
│   │                     cue, until the recording stops), a change of device, the user's unmute,
│   │                     a new helper undoing an earlier one's mutes (output.recover); over the
│   │                     helper's output.* (helperio.go; Auralink's)
│   ├── appbundle/        the version of the app bundle on disk (Info.plist), to tell an update      (leaf)
│   │                     installed while the app runs (updates.go, helper.Config.Stale)
│   ├── overlay/
│   │   ├── layout/       the overlay's geometry as numbers: the pill's, the live text's, a notice's and
│   │   │                 the answer card's sizes (Voxt's), the shadow's margin, the frame anchored to an
│   │   │                 edge of the display, the waveform's bars, the loader's dots (mygo.Rectangle only)
│   │   └── ink/          the overlay's ink check: the face and its strokes measured from a picture, against
│   │                     the layout (Tester renders on Linux, the window's captures on the Mac)
│   ├── uitext/           the words the UI shows from the domain's values (shortcuts, models' errors,
│   │                     a session's state, history entries, sizes and times, the menu bar's labels):
│   │                     i18n, no MyGo; the one internal package that imports i18n
│   ├── datadir/          where the data folder is: location.json, a move or an erase finished at the
│   │                     start (Resolve, before the library opens), copying and checking a tree
│   ├── recordings/       the history's recordings as files (the History settings dialog): their size,
│   │                     moving them to another folder (copy, check, switch, then delete), exporting them
│   ├── dict/             the dictionary's rules: cleaning a term, hints for speech services,
│   │                     aliases replaced in a transcript (Replace), CSV and text import and
│   │                     export, merging an import
│   ├── eval/             the models' quality evaluation: the eval set (testdata/eval: synthesised
│   │                     dictations, their words, the expected clean-up, checks), error rates,
│   │                     dictionary terms found, clean-up checks (docs/eval.md)
│   ├── archtest/         tests of the code's structure only: file sizes, import layering ("Code structure rules")
│   └── tools/            developer commands (standard library only): snapcmp (snapshot comparison),
│                         declhash (pure-move check), evalsynth (the speech fixtures, Qwen3-TTS on a Mac)
├── helper/               the Swift helper (helper/README.md): Sources/, Tests/, Fixtures/, Support/
└── third_party/mygo/     MyGo 0.5.5 with seven patches (scripts/vendor-mygo.sh)
```

The plan's `textio` is the helper's `textio.*` (session.HelperText adapts it).

### Import rules

Arrows point to what a package may import. No cycles. Leaves import nothing internal.

```
main ──▶ every internal package
store ──▶ core                 store/sqlitestore ──▶ core, store      datadir ──▶ store/sqlitestore    recordings ──▶ datadir, store    overlay/layout ──▶ session (Bars), settings (Overlay)    overlay/ink ──▶ overlay/layout
uitext ──▶ i18n and the domain's types (appctx, asr, compose, core, deliver, dict, helper, helper/proto, hotkey, lang, llm, local, models, permission, session, settings, store)
helper ──▶ core, helper/proto  helper/helpertest ──▶ helper/proto
notify, logs ──▶ core
asr ──▶ core                  llm ──▶ core                  local ──▶ asr, llm, models, vad     models/modelstest ──▶ models
omni ──▶ asr, lang, resident, vad      eval ──▶ asr
llmrt ──▶ asr, core, lang, llm, llmrt/wire, local, macx, resident    llmrt/llmrttest ──▶ llmrt/wire
session ──▶ appctx, asr, compose, core, deliver, dict, dsp, helper/proto, lang, llm, settings, store, vad
dsp ──▶ vad
compose ──▶ lang              deliver ──▶ helper/proto      hotkey ──▶ helper/proto, settings
appctx ──▶ helper/proto, hotkey, settings                    dict ──▶ store
appicons ──▶ helper/proto        outmute ──▶ helper/proto
actnow ──▶ settings
core, settings, helper/proto, llmrt/wire, i18n, lang, permission, macx, vad, appbundle, models, resident ──▶ (nothing internal)
actnow, models, local, uitext ──▶ github.com/hehehai/actnowkit (archtest: actnowUsers; its conformance package only in tests)
icons ──▶ mygo/ui
```

The domain packages (session, compose, deliver, hotkey, appctx, dict, asr, llm, local) are pure Go behind narrow interfaces, with the macOS parts behind them: recording, the event tap, the app in front, AX reads (text, browser addresses) and pasting in the helper; the permission reads before the helper runs and `macx.Activate` in purego code. The local models are behind `local.Services` (M1, M2: processes supervised from Go, over loopback). That keeps their logic testable on Linux with fakes: the fake helper plays recordings, scripted hotkey events and text I/O, and `local_fakes_test.go` stands in for the models.

Event types live next to their publisher, and their topic starts with the publisher's name. They satisfy `core.Event` (a `Topic() string` method) without importing `core`:

| Event | Topic | Publisher |
|---|---|---|
| `core.NotificationPosted`, `core.LogAppended` | `notification.posted`, `log.appended` | notify, logs |
| `settings.Changed` | `settings.changed` | the settings store, after saving |
| `helper.StatusChanged` | `helper.status` | the supervisor |

### Why the UI stays in package main

MyGo builds the root `main` package (`mygo dev`, `mygo build`, `ui.View`); views need the window and its main thread. The domain, which must be goroutine-safe and testable, is in `internal/`. Each area of the UI keeps its fields in a struct of its own, embedded in `app` (app.go).

### Code structure rules

`internal/archtest` checks these with every `go test ./...`; each failure says how to fix it (the rules from AudioSmith's refactor, the plan of refactor R0–R8).

- **File size.** A Go file has at most 600 lines, a test file 800; code should keep to about 400 (files over it are reported by `go test -v -run TestFileSizes ./internal/archtest`, not failed). Generated files, `third_party/`, `testdata/`, `build/` and the Swift helper are not counted. The files over the limit today are listed in `internal/archtest/allowlist.txt` with their size as a ceiling, why they are long and the batch that splits them: they may shrink, not grow, and leave the list once split. Functions over 150 lines are reported (`-run TestLongFuncs`).
- **Layering.** Leaves (above) import nothing internal; only `icons` uses `mygo/ui`, and only `store/sqlitestore` (`plugins/sqlite`) and `overlay/layout` (`mygo`, for `mygo.Rectangle`) another MyGo package; only `uitext` may import `i18n` (the domain returns codes and errors, package main and `uitext` say them); fakes (packages whose name ends in `test`) are for tests; `internal/tools` is standard library only. The rules and their reasons are in `internal/archtest/imports_test.go`.
- **Every internal → internal import is listed** in `internal/archtest/imports.txt` (`from to`, paths under `internal/`). A new dependency adds a line there, so that review sees it; a line no import needs fails until removed.
- **Refactors that only move code** prove it: `go run ./internal/tools/declhash . ./internal/...` on the base and on the branch print the same lines (a hash per top-level declaration, comments and build constraints included; `-nocomments` tells a move that also updated the file names comments mention), and `scripts/snapshots.sh check` renders every set of UI snapshots of both and compares them pixel for pixel (§6). Snapshots are reproducible: `snapshot_env_test.go` pins the clock (a `testing/synctest` bubble), the data directories, the free disk space and the sizes Settings → Storage measures.

## 2. Processes and data flow

```
          MyGo main thread                          goroutines
┌──────────────────────────────┐  calls (go …)  ┌──────────────────────────────────────────┐
│ views (build pass: no I/O)   │──────────────▶ │ library: store (SQLite) · bus · settings │
│ app view state + snapshots   │                │ helper.Supervisor ──stdio JSON lines──▶ voxtnew-helper (Swift)
│        ▲ Window.Update       │                │    capture.* · hotkey.* · textio.* · app.* · browser.url · permissions.* · output.*
│                              │                │ llmrt.Runtime ──stdio JSON lines──▶ voxtnew-llm (Swift: MLX, llama.cpp)
│                              │                │    model.load · chat (chat.delta, job.cancel) · model.status; started on use, idle stop
└────────┼─────────────────────┘                └──────────────────────────────────────────┘
   core.Bus subscribers ◀── Publish ─────────────┘
   (wiring code, once)
```

main.go wires, in order: the single-instance lock → the updater plugin (`mygo.Use(native.New(…))`, updates.go; release builds only can update, docs/release.md) → `datadir.Resolve` (finishes a pending move or erase of the data folder) → `openLibrary` (store migrated, bus, settings) → `newHelperHost` (finds the helper) → `newApp` → `connectShell` (log files, deep links) → `connectHelper` (the supervisor, its status and notifications) → `connectUpdates`. When MyGo is ready: the window, `applyBeta` (Beta updates, before the first check), `startHelper`, the menu bar icon, `watchBundle` (an update installed while it runs). On quit: the tray, the helper (`helper.shutdown`), the notifications and logs, then the database.

**The family's runtime file** (spec §11.1, `runtime/dev.hehehai.voxtnew.json` in the family directory, `internal/actnow`'s `Runtime`; family_runtime.go wires it): the runtimes say what they hold through `Config.OnModel` (`internal/resident` events: omni's servers when ready / first lease / last lease / process exit, voxtnew-llm's model after a load or chat / during a call / after / on a switch or the process's exit), on their own event goroutine; the app maps each to `Runtime.Loaded(id, "omni"|"llm", repo, commit, bytes)` (bytes: `macx.Memory` of the process, else what voxtnew-llm reported, else the model's size), `Busy`, `Idle` (`idleSince`), `Unloaded`, and each change rewrites the file. Before a server or the process loads a model, `Config.BeforeLoad` calls `Runtime.Reserve` (the budget against the other apps' files: VoxtNew's models idle for a minute go first, then TopicMemory, then it loads anyway). TopicMemory from another app calls `Runtime.OnMemory`, which unloads through `a.unloadIdleModel` (omni `Unload`: after the uses in progress; llmrt `UnloadIdle`: only without a call). Every paste goes through `a.textIO()` (`familyText`: `Runtime.Paste` waits for another app's `pasting`, at most 500 ms, and sets VoxtNew's while `textio.paste` runs, which answers once the clipboard is restored). Quit stops the runtimes, then `closeFamily` removes the file; later events write nothing.

There is no job queue (plan §5): a session lives in memory, owned by one state machine (`internal/session`), and writes one history entry when it ends.

### A session

```
hotkey.event (helper) ─▶ helperNotified ─▶ hotkeyFired ─▶ hotkey.Route ─▶ startSession (session_control.go)
   sessionBlocker: helper ready? the models the feature needs installed? (memory only)  ─ no ─▶ a notice in the overlay
   (Rewrite's shortcut on a Rewrite answer card: followUpAnswer, the card's turns as Previous)
   session.Manager.Start ─▶ goroutine:
      capture.start (job, the session's id) ─▶ capture.pcm ─▶ memory ─▶ Stream.Send (streaming services)
      Resolver (local.Services: ASR, LLM; a model not installed stays nil) and the dictionary, meanwhile
      app enhancement, meanwhile: a browser's address (browser.url by the front's pid, 800 ms),
         appctx.Resolve(rules, front, address, feature) ─▶ the snapshot's Scene (the overlay's label)
   Stop (a shortcut, the menu bar) ─▶ capture.stop ─▶ Finish / upload ─▶ the text
         (internal/dsp: high-pass 80 Hz, then the level: one gain for an upload, a slow one after a stream's gate; a limiter)
         (internal/vad: a stream gets nothing before the speech, an upload no silence around it; no speech: nothing is sent)
         (the recording kept in the history is the original)
   dict.Replace: the dictionary's aliases ─▶ their terms (the history keeps the text as recognised)
   compose.BuildParts (+ the Scene layer, the glossary): one request, or one per piece where the language changes or over 300 characters
      ─▶ a chat per part (progress deadlines: first token 15 s, 5 s between tokens, a cap growing with the piece; behavior.md 5.5), streaming into the draft ─▶ compose.Clean, compose.JoinParts
         (a part fails: its text as recognised, a warning)
   deliver.Insert: textio.focus (the app in front now) ─▶ textio.paste {targetPid: that app, press: the group's auto key only in the start app, verifyMs: 1000}   (or the answer card)
         (the field did not change after ⌘V: not inserted, the text stays on the clipboard, the answer card; behavior.md 5.8a)
Translate: textio.selection {allowCopy: session.AllowCopy(app)} first ─ text ─▶ compose (no recording)
           ─ none ─▶ record and recognise as above ─▶ compose
Rewrite:   textio.selection while recording; the instruction (+ the selection) ─▶ compose
   deliver.Check (replacing a selection only) ─ would not insert ─▶ the answer streams into the card
   one streaming chat (the same progress deadlines, a cap of at least 60 s; Rewrite without a selection: JSON {title, content}) ─▶ Draft
   Translate: compose.Untranslated? ─▶ once more with the Strict layer
   fails ─▶ failed, in the card with Try again (retryAnswer: StartRequest.Text)
   history entry (+ audio/<id>.wav) ─▶ the end snapshot
snapshots ─▶ a.post ─▶ sessionChanged ─▶ (events.go, in order) the setup guide, the start and stop sounds;
                                         hotkey.configure {active}; the menu bar (its menu and recording icon), the overlay;
                                         at the end: Home, the card or a notice (then the memory notice), then onHistory (Home, the menu bar, History)
```

While a session is live (or the answer card shows) the bindings go to the helper with `active`: a tap fires at once and Escape is reported. Without Accessibility the event tap is not installed; the bindings Carbon can express are registered with `mygo.GlobalShortcut` until it is.

## 3. Threading and state rules

1. **The build pass does no I/O.** A view reads only `app` fields, snapshots and memory-only getters (`settings.Service.Get`, `i18n.T`). Store, model runtime and helper calls happen in handlers that start a goroutine (`a.async`, `helperCall`).
2. **`app` is main-thread only.** Goroutines reach it through `a.post(fn)` (`Window.Update`). In tests `post` runs fn at once, or queues it (`postQueue`) in tests that run real goroutines (the supervisor against the fake helper process).
3. **Bus subscriptions are set up once, in wiring code** (`openShell`, `connectHelper`), never in a build pass. Handlers must not block. Publishers publish after their change is durable, never while holding a lock.
4. **Domain packages are goroutine-safe**, take a `context.Context` when they may block, return copies, and wrap `core.Err*` sentinels.
5. **MyGo lifetime rules hold** (`.agents/skills/mygo-maintenance`): keys before stateful widgets; never store elements; dialogs on goroutines; persistent callbacks capture `c.Services()`, not the `Context`.
6. **The helper is stateless.** Truth (history, dictionary, settings, rules) lives in Go. A helper crash fails its calls (`core.ErrHelper`) and the supervisor restarts it; nothing is lost. Late notifications of a finished job are dropped.

### Seams inside package main

| Seam | Defined in | Used by |
|---|---|---|
| `a.lib` (`*library`: store, bus, settings) | library.go | every area |
| `a.navigate(path)`, `a.showPage(name)`, `a.currentPage()`, `a.shownPath()`; `movedPaths` (old paths, e.g. `/permissions` → `/settings/permissions`) | shell.go, shell_pages.go, shell_settings.go | palette, deep links, notifications, tests |
| `navGroups`, `shortcutPages` (⌘1–⌘6), `pageTitle`, `a.navBadge` | shell_pages.go | sidebar, rail, palette, Go menu |
| `a.page(c, p)`, `a.pageShown(path)` | pages.go | the router: each page's view, and what it reads when it shows |
| `a.updateSettings(fn)`, `a.prefs`, `settingsSections()`, `a.settingsShown(sub)`, `a.setMainLanguage(code)`, `a.syncOpenAtLogin()` | shell_settings.go (the dialog, `settingsState`), settings_general.go, settings_locale.go | every area that changes a setting; sections that read something when they open; the window's opening (shell.go) |
| `a.railMode()` (the only place that decides: the user's choice, else a window narrower than 740), `a.setRail(on)` | shell_nav.go | the sidebar, `pageWidth`, Settings → Appearance |
| `a.audio` (the inputs `capture.devices` listed), `a.loadDevices()`, `a.audioShown()`, `deviceInUse(list, devs)`, `a.playSound` (`macx.PlayWait`, on a goroutine; tests record), `recordingCue(was, s)` (start, stop or none), `cueSound(cue, preset)`; `a.trySound(preset)` (a preview, `a.startSound`: `macx.Start`; tests record), `a.soundsRow(c)` (the presets' menu and Try) | settings_audio.go, settings_sounds.go | Settings → Audio, helperlink.go (a helper that became ready), mute_link.go (`recordingAudio` plays the cues) |
| `a.mute` (`*outmute.Controller`), `a.muting`, `a.recordingAudio(was, s)` (the cues and the mute around a recording; `startCueWait(sound)`), `a.muteHelperChanged`, `a.muteNotified`, `a.muteSettingChanged`, `a.pillSession()` (the pill's text line with the output's note), `a.closeMute()` | mute_link.go | events.go (onSession), helperlink.go, settings_audio.go, overlay_face.go, overlay_size.go, app.go (quit) |
| `faceStyle(e, o, mode, h)`, `overlayPill(c, s, mode, look)`, `a.overlayRestyle()` | overlay_face.go, overlay.go | the overlay, the preview of Settings → Overlay style (settings_overlay.go), `settingsChanged` |
| `uitext.TriggerText(trigger)` (its name; the setup guide's `uitext.TriggerHowText` says how to use it), `featurePath(feature)` | internal/uitext (keys.go), settings_shortcuts.go | Settings → Shortcuts |
| `a.trayItems()`, `a.loadTrayRecent()` (the last three results), `trayicon.Icon(recording)` (Voxt's microphone, `internal/trayicon`), `a.copyText` (tests) | tray.go | `historyChanged`, the menu bar |
| `a.helperStatus()`, `helperCall[T](a, method, params, timeout, then)`, `a.helperNotified(m)`, `a.restartHelper()`, `a.hotkeys` (hotkey.state), `a.front` (app.activated) | helperlink.go | the helper's card (shell_notices.go), Home, permissions, Diagnostics, sessions |
| `a.mgr` (`*session.Manager`), `a.sess` (the newest snapshot), `a.startSession(req)`, `a.startTranscribe(target, hold, try)`, `a.sessionBlocker(kind)`, `a.stopSession()`, `a.cancelSession()`, `a.toggleTranscribe()`, `a.pasteLast(target)`, `a.answer` + `openAnswer` / `insertAnswer` / `copyAnswer` / `closeAnswer` / `retryAnswer(lang)` / `followUpAnswer(hold)`, `a.showNotice`, `sessionProblemText(p)`, `a.syncHotkeys()`; the short-of-memory notice (`a.onPace`, llmrt's `Config.OnPace`, → `a.chatPaced` → `a.memoryNoticeAfter`; `llmrt.ShortOfMemory` decides) | session_link.go (the snapshots, notices), memory_notice.go, session_control.go (start, stop, paste the last), session_answer.go (the card's actions), session_hotkeys.go | the overlay, the feature pages' Try areas, the menu bar, helperlink.go |
| `a.overlaySync()`, `a.ov` (mode, size, visible), `a.later(d, fn)` (timers; `laterHook` in tests), `overlayView`, `waveform` / `layout.WaveBars`, `overlayLoader`, `uitext.WorkingText(s)`, `kindIcon`, `uitext.KindTitle`, `cardNow`, `uitext.TargetLanguageLabel`, `noticeActionLabel` | overlay.go (state, sync), overlay_size.go, overlay_face.go, overlay_wave.go, overlay_card.go; the texts in internal/uitext (session.go) and session_text.go | session_link.go; the feature pages and History reuse the texts |
| `a.featurePage(c, feature)`, `a.fui(feature)` (each page's controls), `featureKind`, `uitext.HotkeyText`, `uitext.FeatureTitle` | feature_page.go; feature_keys.go, feature_models.go, feature_prompts.go; internal/uitext (keys.go: shortcuts' words) | pages.go (Transcribe, Translate, Rewrite) |
| `a.historyChanged()` | history_page.go (history_detail.go: an entry's details; internal/uitext history.go: its words) | session_link.go, after an entry is kept or changed |
| `a.aeBook` (`ruleBook`: the groups for sessions, goroutine-safe), `a.aeRules` + `a.setRules(r)`, `a.sceneConfig()`, `a.urlReads`, `a.frontChanged(app)`, `a.lastOther`, `a.previewApp()`, `a.refreshPreview(force)`, `a.previewMatch()` | appenhance.go | session_link.go (the manager's Scene), helperlink.go (app.activated), the App enhancement page |
| `uitext.SourceText(source, pattern)` | internal/uitext (history.go) | History's details, App enhancement's preview |
| `a.loadDict()`, `a.openTerm(term)`, `chooseImportFile`, `chooseExportFile` (test seams) | dictionary_page.go | the Dictionary page |
| `a.permSys` (`permission.System`), `a.recheckPermissions()`, `a.askPermission(which)` | permissions.go | Settings → Permissions, Diagnostics, Home, the setup guide |
| `a.tryCard(c, tryOpts)` (a Try area), `a.tryIntro(feature)`, `a.startTry(kind)`, `a.startSessionID(req)` (the id, to tell a try from another session) | try_card.go, session_control.go | the feature pages, Home, the setup guide |
| `a.tryPlace(kind, target)` (where a shortcut pressed while VoxtNew is in front runs as a try), `a.tryStarted`, `a.homeChanged()`, `readHome` (today's numbers and the recent results, off the main thread), `a.readinessList()`, `a.openHistoryEntry(e)` | home_page.go, home_ready.go (readiness), home_try.go (try places) | session_hotkeys.go (`hotkeyFired`), history_page.go (`historyChanged`) |
| `a.openOnboarding()`, `a.maybeOnboard()` (first start), `a.obSessionChanged(s)` (a step passes on its own session's result), `a.obOwns(id)`, `a.onboardingRow(c)` | onboarding.go (onboarding_view.go: the dialog; onboarding_steps.go: the steps) | shell.go (`openWindow`), session_link.go, Settings → General, Home, ⌘K |
| `floatingWindowOptions(…)`, `showFloating(w, x, y)` (non-activating windows, mygo patch 0003), `hudContent(view)` (their text unsmoothed, patch 0005), `activateTarget(pid)` (`macx.Activate`) | floating.go | the recording pill and the answer card (M3, M4) |
| `a.upd` (`updateService`: MyGo's updater plugin; `fakeUpdates` in tests), `a.checkForUpdates()`, `a.updateMenuItems(act)`, `a.bundleStale` (`appbundle.Updated`), `a.appUpdated()` (the restart card), `a.relaunch()` (`appRelaunch`) | updates.go | the app menu, the menu bar, ⌘K, Settings → About, the helper's card (FailUpdated), Home |
| `a.toast(msg)`, `a.notify(…)`, `a.post(fn)`, `a.async(fn)` | app.go, shell_notices.go | everyone |
| `a.onSession`, `a.afterSession`, `a.onSessionEnd`, `a.onHistory` (observers, in order; `connectEvents` lists them all), published by `sessionChanged` / `sessionEnded` (session_link.go) and `historyChanged` (history_page.go) | events.go | the setup guide, the sounds, the menu bar, the overlay, Home, History: an area that cares about sessions or the history adds itself there, instead of the publisher calling it |
| `a.local` (`local.Services`; `newLocalServices`, tests swap in fakes), `a.localOpeners()` (the runtimes: M1's speech models through `a.openOmniASR` and `a.unloadOmni`, models_omni.go, which also stops the servers on quit, `a.closeOmni`; M2's language models through `a.openLLM`, llm_link.go, whose `a.llmRuntime()` is made on first use and stopped on quit, `a.stopLLM`; `a.unloadModel` sends a delete's Unload to the runtime of the model's kind; `a.withDevLLM` adds VOXTNEW_DEV_LLM's models), `a.models` (`*models.Manager`), `a.modelStatus` (the statuses' copy, `modelsChanged`), `a.downloadModel(id)` / `cancelModel` / `askDeleteModel` (the family's delete rules: `Manager.Decide`, then the plain or the "Remove from VoxtNew" dialog, models_delete.go) / `deleteModel` / `useModel` (another app's model: `Manager.Use`), the family's line of a card (`modelFamilyText`: Shared with, Downloading in, Deleted in, Installed by; models_cards.go), the Network… dialog (`openNetwork`, the shared proxy and Hub endpoints, models_network.go), the Advanced section (`modelsAdvanced`: how long models stay loaded, `settings.Models`; `keepIdle` maps it to the runtimes' idle times, `a.applyKeepLoaded` gives them a change at once, models_keep.go), `a.modelMem` (the loaded models' memory: `a.loadedModels()` lists the runtimes' processes, `omni.Runtime.Processes` and `llmrt.Runtime.Status`, measured by pid with `macx.Memory` off the main thread while the page shows, models_memory.go), `a.installed(kind)` (memory only), `a.featureModel(kind, feature)` (only models that serve the feature: the translation model only translates), `featureOf(s, feature)`, `setFeatureModel`, `modelsPath`; `uitext.ModelName` / `ModelAbout` / `ModelLanguages` / `DownloadErrorText` | models_page.go, models_cards.go, models_delete.go, models_network.go, models_memory.go, models_keep.go, models_runtimes.go, models_omni.go, llm_link.go | the Local models page, the feature pages' Models section (feature_models.go), Home's readiness, the setup guide, `sessionBlocker`, `checkModels`, History's details; sessions open models through `resolver(a.local)` off the main thread (session_link.go) |
| `a.textIO()` (the text I/O every paste goes through: the family's paste rule), `a.modelEvents(worker)` and `a.beforeLoad()` (a runtime's `OnModel` and `BeforeLoad`), `a.unloadIdleModel(id)` (the family's memory requests), `a.famRuntime()` | family_runtime.go | session_link.go, session_control.go, session_answer.go; models_omni.go, llm_link.go; family.go (`openFamily`) |
| `a.notices` (the centre's, newest first), `a.noticeStack(c)` (the cards at the foot of the sidebar; the rail's popover), `a.setProblem(id, code, n)` (a problem's card: shown once, withdrawn when solved, kept away this run once dismissed), `a.checkModels()`, `a.checkHelper()` | shell_notices.go | the sidebar, the rail; `openModels`, `settingsChanged`, `helperChanged` |
| `uitext.HotkeyGlyphs(keys)` | internal/uitext (keys.go) | ? sheet, menu bar, feature pages |
| Design tokens: `tok(t)`, `px`, `fs*`, `sp*`, `radius*`; `accentName` (teal) | ui_tokens.go | every view |
| UI kit (AudioSmith 0.3): buttons by rank, `pageHeader`, `card`, `formRow`, `settingRow`, `emptyState`, `notice`, `sidePanel`, `dialogFrame`… | ui_kit.go (its overview), ui_buttons.go, ui_marks.go, ui_layout.go, ui_panels.go, ui_forms.go (AudioSmith's split), ui_rows.go (`settingRow`, `sectionTitle`) | every view |

## 4. Data on disk (macOS)

| What | Where |
|---|---|
| Database: history (+ `history_fts`; `rule_source` since migration 2), dictionary terms, documents (`settings:app`, `appctx:rules`, `notifications`) | `~/Library/Application Support/VoxtNew/voxtnew.db` (WAL; `voxtnew.db.bak-<v>` before each migration) |
| Local models | the ActNow family's model library `$HUB` (family spec §7.1: the family's `models.root`, else `$HF_HUB_CACHE`, `$HF_HOME/hub`, `~/.cache/huggingface/hub`), the Hugging Face layout `models--<owner>--<name>/{blobs,snapshots/<commit>,refs}`, shared with the other ActNow apps (and huggingface_hub); the runtimes load `snapshots/<commit>` (`models.Manager.Dir`); locks in `$HUB/.locks/models--…/` |
| The ActNow family | `~/Library/Application Support/ActNow/` (`$ACTNOW_FAMILY_DIR`): `family.json`, `apps/dev.hehehai.voxtnew.json` (the manifest: version, path, capabilities, shortcuts, links, `models.root`), `models/claims/dev.hehehai.voxtnew.json` (the models VoxtNew uses), `models/removed.json` (deleted for all apps), `runtime/dev.hehehai.voxtnew.json` (while it runs: loaded models, the paste flag), `settings.json` (shared settings; VoxtNew's follow flags are in its own `settings:app`, `family`) |
| Recordings kept with history entries | `…/VoxtNew/audio/<id>.wav` (only with Settings → Privacy → Keep the recordings) |
| Data folder location | `…/VoxtNew/location.json` when the user moved it (Settings → Storage); `move.json`, `reset.json` while a move or an erase waits for the next start |
| The output's mutes (crash recovery) | `~/Library/Caches/VoxtNew/output-mute/<helper pid>.json`: the devices a helper muted, written before each mute, removed when it gives them back; the next helper undoes a dead one's (`output.recover`). Empty while nothing records |
| Apps' icons (App enhancement's lists) | `~/Library/Caches/VoxtNew/icons/<bundle id>-64.png`, drawn by the helper (`app.icons`), redrawn after a week (`internal/appicons`) |
| Logs | `~/Library/Logs/VoxtNew/voxtnew.log` (app, rotated at 5 MB), `helper.log` (the helper's stderr) and `llm.log` (voxtnew-llm's stderr) |
| Helper | `VoxtNew.app/Contents/Resources/helper/voxtnew-helper` (or `$VOXTNEW_HELPER`) |
| Language models' runtime | `VoxtNew.app/Contents/Resources/llm/`: `voxtnew-llm`, `mlx.metallib`, `llama.framework`, `swift-transformers_Hub.bundle` (`scripts/build-llm.sh`; or `$VOXTNEW_LLM`) |
| Omni runtime | `VoxtNew.app/Contents/Resources/omni/bin/`: `qwen3_asr_server`, `libmlx.dylib`, `libjaccl.dylib`, `mlx.metallib` (`scripts/build-omni.sh`; or `$VOXTNEW_OMNI_RUNTIME`) |

## 5. What came from Auralink, and from AudioSmith before it

VoxtNew is Auralink @748cf55 with these changes (M0): renamed (module `github.com/sgl-project/sglang-omni/VoxtNew`, bundle `dev.hehehai.voxtnew`, scheme `voxtnew`, helper `voxtnew-helper`, `VOXTNEW_*` variables, its own data, cache and log folders); removed: `internal/asr/{openai,doubao,aliyun,asrtest}` and asr's WebSocket dial and endpoint policy, llm's OpenAI and Anthropic transports, presets and `Probe`, `internal/providers`, `internal/secrets` (the Keychain), `internal/netx` (the proxy), `internal/demo` and the demo build, the Model services page and dialog, Settings → Network, the setup guide's key cards, the README's media and the tests that called real services; new: `internal/local` and the Local models page (`models_page.go`), the feature pages' Models section, the in-process fake models of the tests (`local_fakes_test.go`). The settings document lost `Network` and `Providers`; `Feature.ASR`/`LLM` hold a local model's id.

What Auralink took from AudioSmith:

| VoxtNew | From AudioSmith @f171c86 | Change |
|---|---|---|
| `ui_kit.go` (split as AudioSmith's: `ui_buttons.go`, `ui_marks.go`, `ui_layout.go`, `ui_panels.go`, `ui_forms.go`), `ui_tokens.go` | same files | comments; `setAccent` (accent review shots) dropped |
| `shell*.go` | same files | VoxtNew's sidebar groups and pages (a first group without a title); a smaller window (800×520, at least 680×440; AudioSmith 1100×720 and 820×520); no status bar: the version beside Settings, notifications as stacked cards over it (the rail: a dot, then a popover), the services and the helper as cards while something is wrong, instead of the bell, engine, jobs and downloads; no ⌘E quick switch; ? sheet lists the global shortcuts from the settings |
| `shell_settings.go`, `settings_*.go`, `internal/datadir` (`storage_move.go`), `sysinfo.go` | same files | nine sections: General (app, main and other languages, menu bar, open at login), Permissions (permissions.go), Audio (the microphones' order, sounds), Overlay style (with a preview), Shortcuts (the table of every shortcut), Appearance (the sidebar: automatic, names and icons, icons only), Privacy (history, recordings, text in logs), Storage (VoxtNew's folders), About, 诊断与关于 in Chinese (helper instead of engine) |
| `permissions.go` | same file | a section of the settings dialog (after General): the Microphone and Accessibility only, through the helper; no browser Automation (addresses are read with AX) |
| `tray.go` | `dictation_tray.go` | VoxtNew's menu (the last three results, a click copies one); a sound-wave icon, with a dot while recording |
| `internal/core` | `core` | bus, ids, errors, notifications, log lines only; `Stream` dropped |
| `internal/settings` | `settings` | the Store kept; the document is new (v1, plan §3–§5) |
| `internal/store`, `sqlitestore` | same | Open, migrations, backup, `CheckFile`, `RewritePaths` kept; new schema and repositories; `Stats` instead of `Usage` |
| `internal/helper` (+ `proto`, `helpertest`) | `engine` (+ `proto`, `enginetest`) | one process instead of roles; no scheduler, memory budget or stats polling; notifications without a job go to `OnNotify`; protocol 1 with the helper's methods |
| `internal/permission` | `dictation/system_darwin.go` | Microphone and Accessibility only; the page asks the helper first |
| `helper/` (Swift) | `engine/Sources` (`Server.swift`, `Capture.swift`, the protocol target's `Wire.swift`) and the M0 spikes | no MLX; the M0 spike's capture, event tap, text I/O and browser code, made protocol 1 (docs/helper-protocol.md §6) |
| `internal/i18n`, `icons`, `logs`, `notify`, `lang` | same | zh-Hans rebuilt from what VoxtNew uses (304 strings at M1); the icons of `names.txt` (53 at M6); log file `voxtnew.log` |
| `third_party/mygo`, `scripts/vendor-mygo.sh`, `scripts/mac.sh`, CI | same | renamed; patch 0003 (non-activating windows) is VoxtNew's |

| `internal/llm` | `llm` | no engine backend, no dubbing skills; skills `transcribe_enhance`, `translate_text`, `rewrite`; in VoxtNew no transport at all (M2 adds the local one) |

Not taken: jobs, hub (downloads), catalog, licence, engine's MLX side, services, dsp (VoxtNew's `internal/dsp` is its own), translate, text, doc, dub, api, mcp, workflow, telephony, watermark, gallery, voicedesign, audio, and the dictation's paste and clipboard code (pasting is the helper's `textio.paste`).

## 6. Testing

| What | Command | Notes |
|---|---|---|
| All Go tests | `CGO_ENABLED=0 go test ./...` | Views in `ui.Tester` with a fake supervisor in process; the real supervisor against the fake helper process; protocol fixtures; the store over real SQLite (the plugin's library is downloaded and checked once, or `MYGO_SQLITE_LIBRARY`); i18n key parity |
| MyGo vet | `go tool mygo vet .` | `go vet ./...` and the UI-lifetime checks |
| Race | `CGO_ENABLED=1 go test -race ./...` | needs cgo |
| Snapshots, compared | `scripts/snapshots.sh check [BASE]` (`twice`: is it reproducible?; `render DIR`) | every set below (main, min, home, overlay, ink) of BASE (default: the merge base with origin/main) and of the working tree, compared pixel for pixel by `internal/tools/snapcmp`; about a minute; the pictures of what differs in `$SNAP_TMP/diff`. New snapshot tests call `runSnapshots` and make their directories with `testDir` (snapshot_env_test.go). The overlay's flows (`VOXTNEW_FLOW_SHOTS`) are left out: they run real goroutines, and their tests check the steps |
| Snapshots | `VOXTNEW_SNAPSHOTS=/tmp/shots go test -run TestSnapshots .` | every page, the settings sections and the overlays, light/dark, en/zh-Hans, 2×; the window's default size (800×520); `VOXTNEW_SNAPSHOTS_SIZE=680x440` (or `min`) for the smallest; the foot of the sidebar with none, one and three notifications and the rail's dot and popover (`sidebar-notices-*`, `rail-notices*`) |
| Swift helper | `scripts/mac.sh 'cd helper && swift test'` | the wire types against helper/Fixtures (the same lines the Go tests decode), the hotkey recognizer, the PCM framer; `scripts/mac-regress.sh` runs it on mac-air (CI is Linux only) |
| Mac checks | `scripts/mac.sh 'VOXTNEW_MAC=1 go test -run "TestNative|TestActivate|TestPlay|TestSystemProxies" -v ./internal/permission ./internal/macx'` | the permission calls, activation, the recording's sounds, the system's proxies (as `scutil --proxy` shows them) |
| System proxy (M4) | `scripts/mac.sh 'env -u https_proxy -u http_proxy -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u all_proxy VOXTNEW_MODELS_REAL=1 go test -count=1 -v -run TestRealSystemProxy .'` | the VAD from huggingface.co alone, every request through the system's proxy (mac-air: 127.0.0.1:7897) |
| Packaged app | `scripts/mac.sh 'scripts/build-helper.sh && scripts/build-llm.sh && go tool mygo build'` | the helper and voxtnew-llm (mygo.json's `helperEntitlements` name it), then the app; signing during development: README |
| Release build | `scripts/mac.sh 'scripts/release.sh --runtimes $HOME/voxtnew-runtime'`; `scripts/mac-sign-e2e.sh` | the helper, the Omni runtime and voxtnew-llm inside the app, checked by `release-verify.sh` (every nested binary signed, the runtimes start from the bundle); signed with a throwaway identity: the hardened runtime and timestamps everywhere ([release.md](release.md)) |
| CI | `.github/workflows/voxtnew-ci.yaml` (repository root) | on changes to `VoxtNew/**`: gofmt, the Go tests, MyGo vet, the update-check build's vet, `vendor-mygo.sh --check`, on Linux |
| Settings (M6) | `go test -run 'TestAudioSettings|TestDeviceInUse|TestRecordingCues|TestSessionSounds|TestSounds|TestPreviewPlayer|TestOverlayStyle|TestShortcutsSettings|TestOtherLanguages|TestAutoRail|TestStartHidden|TestTray' .`; `go test -run 'TestPlay|TestPresets|TestSoundFiles|TestStartStop' ./internal/macx` (TestStartStop on a Mac: a silent stand-in for afplay) | the microphones' list over the fake helper's `capture.devices`, the cues of each change of a session and of each preset, the presets' menu (shown while the sounds are on, a choice or a rest in it plays, Try's accent), the overlay's settings, the shortcut table's problems and links, the automatic rail at 680 and 800 wide, the menu bar's items and icons |
| App enhancement, dictionary | `go test ./internal/appctx ./internal/dict`; `go test -run 'TestScene' ./internal/session`; `go test -run 'TestAppEnhancement|TestDictionary|TestSessionAppEnhancement|TestSceneConfig' .` | URL rules and the matching order (table tests); sessions in a URL group, a browser's group, an app's group, none, an address that times out, the auto key; the dictionary's hints in each service's request (golden files) |
| Sound | `go test -v -run 'TestCompareVariants' ./internal/dsp` (the table of levels before and after; `VOXTNEW_DSP_OUT=dir` writes the WAVs) | the filter's response at 50, 60, 80, 100, 200 Hz and 1 kHz, a DC offset at the start and a jump of it, chunks against the whole byte for byte; the gain rule (table), quiet speech raised as a file and as a stream, noise alone never raised, a stream's gain holding through silence and moving smoothly, the limiter on full-scale squares and on speech raised by 30 dB, nothing, a byte, zeros; the speech sample quiet, under hum, white and fan noise, and clipped (no speech frame cut, never over -1 dBFS, the stream's gain near the file's) |
| Sessions | `go test ./internal/session ./internal/vad ./internal/compose ./internal/deliver ./internal/hotkey`; `go test -run 'TestSession|TestOverlay|TestAnswer|TestTranslate|TestRewrite' .` | the state machine with in-process fakes (both kinds of speech service × the language model on, off, failing, timing out × cancel at each stage × a target that takes no text × late callbacks; Translate and Rewrite: a selection read, missing, failing, by ⌘C, the strict retry, the card by choice, by a target that takes no text and by length, the JSON answer, the text given at Start, follow-ups; the voice activity detection on the speech sample with silence and noise added, a stream held until the speech, a recording without speech sent nowhere; the overlay leaving the screen after each way a session ends, on a fake clock); the app over the fake helper process (`selection=`, `copy-selection=`, `readonly`, `focus=`, `switch=`, `paste-log=`) with the fake local models of local_fakes_test.go (a transcriber, a streamer, a chat model behind `llm.Client`) |
| Headless on a Mac | `MAC_DIR=code/VoxtNew-mt scripts/mac.sh scripts/mac-e2e.sh` | the real supervisor and Swift helper (`mac_e2e_test.go`, `VOXTNEW_MAC_E2E=1`): the helper's read-only methods, its test-only file input (`VOXTNEW_HELPER_CAPTURE_FILE`) against the speech sample, a Transcribe session over the fake local models, the history in a temporary SQLite; text fit with Core Text. No window, key, tap, clipboard or microphone: it runs beside other work (manual-test.md, "自动真机测试（无界面）") |
| Omni runtime | `go test ./internal/omni`; `go test -run 'TestSessionOmni|TestOmniOpener' .` | the supervisor, Final, live preview and Silero against the fake server (`omnitest`: starts, failures, hangs, crashes, stalls, idle unload, Unload, a switch of model, Close); a whole session through the Library and `localOpeners` over it, a delete that unloads, quit |
| Omni on a Mac | `scripts/mac.sh 'scripts/build-omni.sh --out build/omni-stage'`, then `TestMacE2EOmni` with `VOXTNEW_MAC_E2E=1`, `VOXTNEW_HELPER`, `VOXTNEW_OMNI_RUNTIME`, `VOXTNEW_OMNI_MODELS` (mac_e2e_omni_test.go's header) | the real models: a Final cold and warm, a live session with the helper playing the sample, Silero's speech; it logs the times (`MEASURE` lines) |
| Loaded models' memory on a Mac | `VOXTNEW_MAC_E2E=1 VOXTNEW_OMNI_RUNTIME=… VOXTNEW_LLM=… go test -run TestMacModelMemory -v .` (mac_e2e_memory_test.go's header); `go test -run TestMemory -v ./internal/macx` | asr.fast, llm.fast and llm.translate loaded from the models in `$HUB`, measured by pid as the Local models page does, against voxtnew-llm's own report (`MEASURE` lines) |
| Language models | `go test ./internal/llmrt/...`; `go test -run TestLLM .` | the protocol's fixtures (`llm/Fixtures`, as the Swift tests); the runtime against the fake process (`llmrttest`: lazy start, streaming on the caller's goroutine, the profiles' parameters, idle stop, a crash retried, cancel, a cancel that hangs, mismatch, silence, no binary, a model that does not load, Stop, warm-up, Unload); Translate and Rewrite sessions and the opener through the Library over it (`llm_e2e_test.go`) |
| The ActNow family | `go test ./internal/actnow ./internal/models` | actnowkit's conformance scenarios (spec §16.1: reuse without downloading, deletion protection by claims, locks and loaded models, deleting for all apps, concurrent and resumed downloads against another app, settings that follow the family, shortcut conflicts, uninstalled apps, forward compatibility) through VoxtNew's Manager and Family; the runtime file (`internal/actnow` `TestRuntimeFile`, `TestOnMemory`, `TestReserve`, `TestPaste`; the runtimes' events, `internal/resident`, `TestModelEvents*` in omni and llmrt; across a session, `family_runtime_test.go`); the catalogue as hfhub entries and family pins (`TestFamilyPins`, `TestEntry`); the manifest. Every test uses temporary `$FAMILY`/`$HUB` (`conformance.New`; the app's tests put both in each test app's data folder, `testutil_test.go` `runIsolated`). On a Mac: `go run github.com/hehehai/actnowkit/cmd/actnow-conformance check|discover|claims|watch|lock` |
| Language models on a Mac | `scripts/mac.sh 'scripts/build-llm.sh && scripts/build-llm.sh --test'`, then `VOXTNEW_LLM_REAL=1 VOXTNEW_DEV_LLM=<id>=<dir>,… go test -run TestLLM -v .` | the Swift Kit's tests (fixtures, server, repetition guard, UTF-8); the real process and models: the sessions of llm_e2e_test.go, `TestLLMMeasure` (VoxtNew's prompts, each twice: first token, rate, memory, answers) and `TestLLMRealCancel`; [llm-models.md](llm-models.md) has the numbers |
| Eval on a Mac | `VOXTNEW_EVAL=1 … go test -run TestEval -v .` (eval_test.go's header; `VOXTNEW_EVAL_QUICK=1`, `VOXTNEW_EVAL_ONLY`, `VOXTNEW_EVAL_OUT`) | the eval set through the real runtimes and the catalogue's models in `$HUB`: both speech models under three language hints, with and without the dictionary, live (`TestEvalASR`); the clean-up with and without compose's example, spoken Translate, Hy-MT2 (`TestEvalLLM`); app enhancement and the dictionary in whole sessions (`TestEvalAppEnhancement`, `TestEvalDictionary`); [eval.md](eval.md) has the numbers |
| Mac regression | `MAC_DIR=code/VoxtNew-xyz scripts/mac.sh scripts/mac-regress.sh [--full]` | the helper's Swift tests, `TestMacE2EOmni`, `TestLLM*` (real), the eval (quick: floors), the app built with the runtimes, started with `--smoke` (smoke.go: its own data, no window, no shortcuts; each runtime answers once) and quit with nothing left running; a summary table |
| By hand on a Mac | [manual-test.md](manual-test.md) | what needs permissions, a GUI and the real models, in order |
| Text fit | `VOXTNEW_TEXTFIT_PROBE=1` / `VOXTNEW_TEXTFIT_AUDIT=1 go test -run 'TestTextFits|TestTranslationsFit' .` | meaningful on macOS (Core Text) |
