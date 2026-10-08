# Meeting Detail Window

Meeting detail presentation for summaries, transcript views, icons, formatting, styles, and view model state.

## Responsibilities

- Presents saved meeting summaries, transcript sections, speaker metadata, and supporting controls.
- Formats meeting detail content for readable macOS window presentation.
- Keeps meeting-detail view state and styling separate from meeting processing logic.

## Incomplete file transcription

The file-task “View Transcription” action opens this same window with `MeetingDetailViewModel.Mode.fileDraft`, using the completed ASR checkpoint's segments (including timestamps), not a flattened string or a fake history entry. Draft mode allows timeline search/selection/copy, but not editing, speaker relabeling, export, summary/chat, translation, or playback from a deletable queue cache. The header explicitly says speaker analysis is incomplete. Closing it does not cancel analysis; removing the task closes only its draft window.

The queue's successful-completion callback promotes an existing draft controller to the saved history entry in the same NSWindow, without activating the application. A fresh hosting view resets the playback StateObject to the history-owned archive URL. Completed tasks do not open new windows in the background; repeat clicks reuse the registered window. No controller holds a writable draft persistence handler. Checkpoint-load completion rechecks task existence, attempt start time and final state before presentation.

`MeetingDetailFileDraftTests` covers draft capability gates, timestamps/search and AppKit window reuse/promotion/close. These require macOS/XCTest; Linux syntax checks do not validate window lifecycle or playback behavior.

## Components

- `MeetingDetailWindow`: window manager/controller ownership and AppKit setup.
- `MeetingDetailWindowView`: layout, transcript selection, dialogs and scroll synchronization.
- `MeetingDetailPlaybackController`: player/timer lifetime; still owned by the window view.
- `MeetingDetailPlaybackPane`: playback controls and local zoom/highlight/popover state; scrubbing is bound to the parent.
- `MeetingDetailViewModel`: published state, async operations and mutations.
- `MeetingDetailPresentation`: read-only display/configuration policy, without widening published setters.
- `MeetingTranscriptExporter`: AppKit save-panel integration.
- `MeetingTranscriptComponents`: shared transcript views, including the equatable virtual-list pane.

Verify playback, active-segment scrolling, search/editing and speaker presentation on macOS after UI extraction; source comparisons do not establish view-identity or accessibility correctness.
