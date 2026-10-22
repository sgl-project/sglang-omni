# Voxt

macOS app target for Voxt, containing the application source, resources, and user-facing runtime surfaces.

## Responsibilities

- Starts and coordinates the menu bar app, recording flow, meeting mode, settings, and windows.
- Keeps shared runtime services, transcription engines, model support, and provider integrations together under the app target.
- Stores localized resources, prompt templates, visual assets, and macOS UI entry points used by the shipped app.

See the [source map](../docs/Architecture.md) for directory boundaries and runtime paths, and the [refactoring assessment](../docs/RefactoringAssessment.zh-CN.md) for known coupling and pending work.
