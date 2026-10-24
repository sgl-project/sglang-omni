# Core

Shared business and infrastructure layer used across app flows, settings, transcription, and windows.

## Responsibilities

- Provides persistence, stores, repositories, permissions, prompts, audio helpers, and app behavior services.
- Holds reusable transcript, session, dictionary, history, note, sync, and translation support.
- Keeps UI-independent logic separate from AppKit and SwiftUI presentation code.

Repositories live beside their domain stores in `History/` and `Dictionary/`. `Notes/` contains note storage, export records, and Obsidian/Reminders sync coordinators. `VoxtDatabase` remains shared infrastructure at the Core root. These source moves do not change database or user-data paths.
