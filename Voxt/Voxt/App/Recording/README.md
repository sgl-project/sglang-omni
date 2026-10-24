# Recording

Recording-session coordination for capture startup, overlay state, text routing, and finalization.

## Responsibilities

- Plans recording starts and validates the runtime context used by local or remote transcription.
- Routes live and final transcript text into overlays, input targets, history, and follow-up actions.
- Handles session end behavior, voice-end commands, capture handoff, and recording overlay updates.

`RecordingSessionLifecycle` owns session identity, cancellation, exactly-once output admission and end markers. Starting a session resets these together; cancellation rejects old output while still permitting cleanup for the cancelled ID. Late end requests/completions cannot clear a newer session.

`SessionEndFlow` executes its fixed cleanup sequence directly rather than through stage protocols. UI, capture, timing and external-editor delivery transactions remain in their existing components.
