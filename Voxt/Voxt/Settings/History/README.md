# History Settings

Settings UI for transcription history, audio retention, correction presentation, and detail sheets.

## Responsibilities

- Presents history-related settings and reusable history settings components.
- Configures audio archive behavior and correction display preferences.
- Supports detail sheet content used from settings and history workflows.

`HistorySettingsView` owns filters, selection and async list state; `HistorySettingsData` contains list/deletion values. `HistorySettingsComponents` holds shared history rows and toolbar presentation. Note filters/section controls live in `NoteHistoryControls`; `NoteHistoryRow` retains its own editing/focus state and private footer components.
