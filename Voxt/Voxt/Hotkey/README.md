# Hotkey

Global hotkey capture, interpretation, recording UI, action resolution, and runtime configuration.

## Responsibilities

- Registers and manages global shortcuts used to start recording and trigger app actions.
- Normalizes modifier keys, key events, recorder UI state, and user-facing hotkey labels.
- Resolves hotkey actions against app settings and runtime availability.

## Ownership

`HotkeyEventTapInstallation` owns one tap, source, callback context and dedicated `HotkeyEventTapRunLoop`. The manager detaches old installations under its routing lock and stops them outside the lock, so retirement cannot stop a replacement thread. The callback context weakly references the manager; source removal retains the context through callback-thread cleanup.

Queued events/recovery and main-queue actions carry generation checks. Application callbacks run outside the state lock. `HotkeyBusinessState` pairs the six fields for each shortcut business instead of maintaining parallel scalar state; routing priorities and gesture rules are unchanged.

`HotkeySupport` keeps preference values/defaults and matching policy; `SidedModifierFlags`, `HotkeyPreferencePersistence` and `HotkeyPreferencePresentation` separate hardware-side flags, stored preference migration and labels. Codable layouts and preset values are preserved; invalid signed/out-of-range integer preferences fall back without trapping. The manager's lock-protected gesture state machine remains one owner rather than being split by line count.

The run-loop tests use ordinary CF sources without installing global taps or requesting permissions. Real permission recovery and sleep/wake acceptance remain required.

Runtime and recorder taps verify that the newly installed, process-owned tap retains every requested event type. macOS can return a valid tap while filtering out ordinary key-down and key-up events; this must not be reported as a working global shortcut backend. An unverifiable or partial tap is invalidated, the next location is tried, and a failure records the Accessibility and Input Monitoring preflight states with recovery guidance. No explicit Input Monitoring permission request is made. The recorder retains its local event fallback when a global tap cannot be validated.

Escape tests also exercise production asynchronous callback dispatch, repeat/up suppression, injected-event and shortcut-recording exclusion, callback reentry, and invalidation of queued Escape actions after a routing reset. These tests do not grant macOS permissions or substitute for physical foreground/background cancellation checks after signing or installation changes.
