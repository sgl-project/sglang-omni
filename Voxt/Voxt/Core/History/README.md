# History

Transcription history domain logic for stored entries, archived audio, conversation text, and formatting.

## Responsibilities

- Persists history entries and resolves value updates used by settings and detail windows.
- Manages audio archive paths, playback metadata, and cleanup-safe storage behavior.
- Provides conversation assembly and relative timestamp formatting for history-facing UI.

`TranscriptionHistoryModels` contains entry/report values, CodingKeys and legacy decode fallbacks. `TranscriptionHistoryStore` remains the sole owner of published entries, pagination/reload state and persistence operations. Moving model declarations did not change raw values, stored fields, migration behavior or data paths.
