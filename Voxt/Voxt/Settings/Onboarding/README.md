# Onboarding Settings

Onboarding UI, step data, permission guidance, and setup flow support.

## Responsibilities

- Presents first-run and settings-accessible onboarding screens.
- Tracks onboarding step data, completion state, and guidance components.
- Keeps onboarding presentation separate from persistent onboarding preferences.

`OnboardingGuideView` retains SwiftUI state and lifecycle assembly. Its sibling `OnboardingGuidePermissions`, `Practice`, `ModelSelection`, `Configuration`, `Modals`, and `Shortcuts` files implement focused parts of that same view. Model rows, style/shape definitions, presentation components and `SelectableGuideTextView` are separate.

Do not assume the extensions own independent state. Preserve view identity, focus handling, session-ID checks, draft confirmation and microphone/permission task cleanup when extracting child views.
