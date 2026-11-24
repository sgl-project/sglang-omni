---
name: mygo-maintenance
description: Maintain Go applications using github.com/egoist/mygo, including UI changes, crash fixes, and dependency upgrades. Apply MyGo-specific lifetime, identity, threading, and testing practices.
---

# MyGo app maintenance

Check the app's pinned MyGo version and existing conventions before changing APIs. Locate its source with `go list -m -f '{{.Dir}}' github.com/egoist/mygo`; read `ui/doc.go` and relevant implementation comments. Read `docs/ui/migration.md` only for major-version upgrades, including minor-version bumps while on `0.x` (for example, `0.2.x` to `0.3.x`, or `1.x` to `2.x`). Confirm API availability against the pinned source.

The native UI rules below apply where the app uses package `ui`; preserve a web app's frontend and typed Go/TypeScript bindings.

## UI lifetime and identity

- Views receive a stable `*ui.Context` for their window. `Children(func())` shares it and changes its parent scope during the callback. Keep Context out of app state and workers; capture `c.Services()` for persistent clipboard/URL callbacks and background redraws.
- Constructors return checked `ui.Element` values. Elements and custom parts expire before the next build pass, including another pass in the same frame. Store app data, widget state and `ui.Handle`, never an element or parts. Zero elements are safe and `Valid()` is a quiet check; stale operations panic in development/Tester and become empty results or no-ops in production. Fix the ownership rather than relying on validity guards or recovery.
- Bind persistent identity with `.Bind(&handle)`. `handle.Focused(c)` and `FocusWithin(c)` read committed focus independently of construction order. `handle.Focus(c)` requests focus and waits while hidden; `CancelFocus(c)` cancels it. Handles support several windows; select one with its Context, and keep widget state separate per window.
- Use `.FocusBind(&a.pane, files)` when app data should request focus. The field holds desired focus; `ui.FocusedValue(c, &a.pane)` reads actual focus while a hidden target waits. Use a comparable field, unique target values and its zero value for no focus.
- Use `handle.Bounds(c)` for committed geometry in input callbacks. `c.Resolve(handle)` returns only the current build's element. Do not capture an element in a later input callback to query its bounds.
- Give controls stable, unique identity before construction with `c.Key(key)`, including stateful widgets. Container `.Key(key)` must precede children and local state. Configure fluent lists before `.Rows`; use `.ItemKey` for stable item identity and `ui.ListRow.Selected()`/`ListFocused()` for row styling.

## Input and actions

- Bound values update after construction and configuration. `Changed`/`Submitted` notices appear in the following build pass. Bind directly to persistent model fields when possible. A local value recomputed each pass loses the edit before its notice; write derived checkbox or segmented values back in `OnClick`, after that build's bound input updates them.
- `OnClick`/`OnShortcut` run after construction and bound input; `OnChange`/`OnSubmit` observe the rebuilt view's notices. Prefer deferred actions when changing a collection being built. `Clicked()` and shortcut queries remain supported for local interactions.
- Declare scoped commands with `handle.OnShortcut(c, mods, key, fn)`, before or after Bind. Only an enabled control present in the current build takes its command. Use `c.OnShortcut` for commands in the current parent scope.

## State, work, and rendering

- Views may rebuild several times for one event. Trigger external side effects through handled actions or explicit guarded jobs, not unconditional rendering code.
- Keep slow I/O and expensive computation off the UI thread. Publish results through `Window.Update`; it is asynchronous. `Invalidate` requests redraws without synchronizing model writes. Cancel obsolete jobs or reject stale results after source changes or window closure.
- Keep drawing callbacks limited to painting; they can run repeatedly. Prefer MyGo widgets/base widgets to preserve keyboard, focus, disabled, and accessibility behavior. Follow the app's theme and sizing conventions.

## Verification

- Reproduce UI failures with `ui.NewTester`. Exercise extra frames while elements disappear and return, plus relevant focus, typing, and shortcut behavior.
- Run targeted regression tests, `go tool mygo vet .` for stored build values and goroutine captures, and `CGO_ENABLED=0 go test ./...`. Use race checks when changing shared state, and inspect rendered output or the native app when changing layout/platform behavior, respecting the user's verification preferences.
- For upgrades, preview with `go tool mygo migrate-ui .`, apply with `-write`, and review persistent identity/services, separately assigned keys and derived bound values manually. Verify affected API/lifecycle assumptions against the new pinned source and rerun relevant UI tests. Report remaining platform-specific verification limits.
