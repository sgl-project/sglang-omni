# Utilities

Cross-cutting support code for localization, logging, updates, async persistence, and app utilities.

## Responsibilities

- Keeps small infrastructure helpers out of feature-specific folders.
- Provides shared browser automation script building, pasteboard, waveform, logging, and update support.
- `PasteboardTextWriter` owns temporary text writes and restoration tokens. Overlapping pastes inherit the original baseline; observed newer copies invalidate restoration. Snapshots remain text-only, and change-count checks are not cross-process atomic compare-and-swap.
