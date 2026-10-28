# App

Application orchestration layer for launch, menu bar ownership, recording entry points, and feature routing.

## Responsibilities

- Connects hotkeys, menu actions, recording sessions, transcription flows, translation, notes, and settings windows.
- Owns app-level startup policies, runtime synchronization, warmup, and development-only seeding.
- Uses AppDelegate flow extensions for coordination; substantial shared session state still lives in AppDelegate and is a refactoring boundary, not an isolated module.

## Status menu history and tracking

`MenuWindowCoordinator` routes the History submenu to the existing history filters. Notes now lives under History rather than as a duplicate top-level shortcut. Feature availability hides disabled categories; transcription remains available. Recent Transcriptions merges dictation and translation by creation time, newest first, with five entries total. Disabling translation excludes it from these shortcuts without deleting stored history.

`TranscriptionHistoryStore.loadRecentMenuEntries` reads bounded lightweight previews off the main thread, independently of the history window's current page. `StatusMenuHistorySupport` filters/sorts cached candidates and bounds preview titles. Selection loads the current full text asynchronously and copies through the shared pasteboard writer, without opening a window or injecting text.

Menu tracking does not fetch history or enumerate audio hardware. Asynchronous rebuild requests are coalesced until the menu closes; all enabled states are explicit rather than relying on AppKit responder-chain validation. The note corner-hover monitor skips screen sampling and transition scheduling while a menu is tracking, then resumes after dismissal.

These are static-review mitigations, not a confirmed diagnosis of the reported hover stall. macOS validation is still required (the implementation environment has no Xcode/AppKit runtime):

- Run `StatusMenuHistorySupportTests`, `TranscriptionHistoryStoreAsyncTests`, `VoxtNoteCornerHoverStateMachineTests`, and `LocalizationResourcesTests` with the shared Voxt scheme.
- Check mixed recent records, long/multiline full-text copying, empty history, deletion/clearing, and each feature toggle.
- Open the tray/history/microphone menus with the main window both open and closed; compare hover with Notes enabled/disabled. Trigger history/device/update changes while tracking: the menu should remain stable and refresh on the next open.
- Check that note-panel context menus keep the panel visible and corner reveal/hide resumes after dismissal. Use Instruments Time Profiler on macOS if hover still stalls; no measured performance improvement is claimed yet.

## Request and capture tasks

`LLMRequestLifecycle` owns request validity and outstanding LLM work. `TrackedTaskStore` retains cancelled work until it actually exits, so termination and idle reclamation can observe it. Capture-start replacement waits for prior starts to unwind before touching the shared audio engine. Cancellation is not resource-release confirmation.

## Output delivery

- `SessionOutputPreparation.swift`: normalization, dictionary correction and the prepared delivery snapshot.
- `SessionTextIO.swift`: committing, delivery destinations, session-checked history/evidence updates and answer-overlay interaction.
- `TextInputIO.swift`: read-only AX/input inspection; `TextOutputDelivery.swift`: target restoration, paste and follow-up key posting.
- `TextInjectionTransaction.swift`: validates queued injection at execution and completes once. Output generations survive normal teardown but not begin/cancel/dismiss.
- `SessionTimingLogging.swift`: timing snapshots and diagnostics.
- `Recording/SessionEndFlow.swift`: session-end orchestration.

The obsolete finalize-stage runner and unreachable preview/replacement chain have been removed. Preparation precedes delivery; history and dictionary evidence are updated only if that session still accepts the completion. Manual overlay delivery snapshots its target/history identity. Pasteboard restoration uses the shared text writer's change-count ownership; key posting is not an editor acknowledgement.
