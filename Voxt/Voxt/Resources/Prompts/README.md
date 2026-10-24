# Prompts

Canonical prompt templates loaded by prompt resource stores and feature defaults.

## Responsibilities

- Keeps default rewrite, translation, dictionary, note, and meeting prompts outside compiled Swift logic.
- Organizes prompt resources by supported language code.
- Keeps model-specific, non-localized prompts under `shared/`.
- Makes prompt copy easier to review, localize, and update without touching flow code.

## Layout

- `en/`, `zh-Hans/`, and `ja/` contain localized prompt presets. File names use `<language>-<resource>.txt`.
- `shared/` contains model protocol prompts whose wording must not change with the app interface language.
- Every active resource must be registered by `LocalizedPromptResource`, `SharedPromptResource`, or the feature preset catalog. Do not add unregistered prompt files.
- Swift should only assemble runtime data, optional context blocks, and provider request envelopes; reusable fixed prompt wording belongs here.
