# Settings

Settings presentation layer for app configuration, feature tuning, permissions, models, and provider setup.

## Responsibilities

- Hosts the settings shell, general panes, dialogs, sheets, and shared settings controls.
- Separates large settings areas such as models, features, dictionary, history, onboarding, and enhancement.
- Keeps settings UI state and validation close to the screens that own it.

Remote provider sheets separate `RemoteProviderSheetSnapshot` (configuration/generation assembly) and `RemoteProviderSheetValidation` (endpoint and field validation) from model/credential controls and operations. Form state remains in the parent SwiftUI view; these extensions are not independent owners. Removed menus and old OpenAI state were unused; persisted provider compatibility fields and credential edit intent remain unchanged. `RemoteProviderSheetOperations` owns model-list/connection tasks and results. Close/replacement cancels and invalidates prior requests, tracks them until exit, and deferred view callbacks check current result identity.

Installation warnings distinguish an unknown/checking cache from confirmed absence. `ModelInstallationObservation` shares a debounced installation-revision stream between the settings shell and model catalog; scan completion refreshes badges without menu interaction or progress polling. Feature selectors show `Loading…` while validation is pending and remain non-selectable.

`PermissionsSettingsView` retains permission state, cancellable refresh/request/test tasks, and persistence. `BrowserAutomationPermissionProbes` separates their inputs/results and existing nonisolated native checks; extraction does not change the prompting or authorization policy.
