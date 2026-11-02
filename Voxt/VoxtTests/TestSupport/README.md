# Test Support

Shared helpers, factories, doubles, and assertions used by Voxt test targets.

## Responsibilities

- Provides deterministic fixtures and utilities for stores, temporary paths, and model-test gating.
- Centralizes reusable test doubles and factory helpers.
- Keeps test setup code out of individual test cases when it is shared across suites.

`HotkeyManagerTestCase`, `RemoteModelConfigurationTestCase`, `MLXModelManagerTestCase`, and `MeetingDetailViewModelTestCase` hold setup/restoration or construction helpers shared by the split suites. They intentionally define no test methods. Preserve their actor annotations, credential/defaults cleanup and controlled lifetime behavior when changing fixtures.
