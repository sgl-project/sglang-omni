# W2 phase 0: UIA paste-verification probe

## Scope and reproduction

The probe is a Windows-only `main` package using the standard library and `golang.org/x/sys/windows`. It launches a uniquely named temporary file in Notepad and a PowerShell console, focuses the UIA element, types `seed ` with SendInput, snapshots the OLE clipboard, posts Ctrl+V, and polls the original focused element every 25 ms for 1 s. It prints UTF-16 lengths, SHA-256 prefixes, selection offsets, the first changed sample, and a verdict; it never prints field contents. A no-input Notepad pass checks the `unchanged` path. Notepad is saved only to its temporary file and its own window is closed; no existing Notepad process is killed.

Tested on Windows build 10.0.26200.0, amd64, Go 1.27.4, on 2026-11-13. The interactive scheduled-task launch worked; these are measured results, not a non-interactive extrapolation.

Commands run from Git Bash in `C:\Users\20192\git\sglang-omni\Voxt`:

```sh
CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build -o C:/Users/20192/AppData/Local/Temp/voxt-uia-spike.exe ./helper-windows/spike
CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go vet ./helper-windows/spike
CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go test ./helper-windows/spike
MSYS2_ARG_CONV_EXCL='*' schtasks.exe /Create /TN spike-uia /TR 'cmd.exe /c "C:\Users\20192\AppData\Local\Temp\voxt-uia-spike.exe > C:\Users\20192\AppData\Local\Temp\voxt-uia-spike.log 2>&1"' /SC ONCE /ST 00:00 /IT /F
MSYS2_ARG_CONV_EXCL='*' schtasks.exe /Run /TN spike-uia
```

Task Scheduler warns that 00:00 is in the past, but the explicit `/Run` starts the task immediately. Read `C:\Users\20192\AppData\Local\Temp\voxt-uia-spike.log` after it reaches the Ready state. Remove the task with `MSYS2_ARG_CONV_EXCL='*' schtasks.exe /Delete /TN spike-uia /F`.

## Measured results

| Target | Focused UIA element | Baseline | First post-paste reading | Verdict and timing |
|---|---|---|---|---|
| Notepad | `Document(50030)`, framework `Win32`, `IsPassword=false`, `ValueIsReadOnly=false` | ValuePattern and TextPattern both length 5 with the same digest; selection `5+0` | Both lengths 20 (+15), both digests changed, selection `20+0` | `inserted`, 27.6 ms, on the first sample. All three signals detected it independently. |
| Notepad, no-input control | Same element after insertion | Value/Text length 20, selection `20+0` | 40 unchanged samples | `unchanged`, 1018.8 ms. The slight overrun is the final sequential UIA read after the 1 s deadline. |
| PowerShell console | `Text(50020)`, framework `XAML`; no ValuePattern | TextPattern whole-buffer length 122, selection `0+0` | All 40 readings had length 122, the same digest, and selection `0+0` | Raw comparison `unchanged`, final assessment `unknown`, 1012.6 ms. A static whole-buffer reading with no meaningful caret cannot verify whether console input changed. |

Across four Notepad insert runs, the change was visible on the first sample, between 27.6 ms and 30.4 ms. The probe does not claim that the console paste succeeded or failed: UIA provided no evidence either way, which is exactly why `unknown` is safer than `unchanged` there.

An independent verification run on 2026-11-14 reproduced these outcomes on the same Windows build and Go version: Notepad changed on the first sample at 27.2 ms (all five signals), its no-input control was `unchanged` at 1009.9 ms, and the console was `unknown` at 1015.3 ms after 40 unchanged readings. Both clipboard restores returned `restored=true`, `taken=false`.

The final run restored the original clipboard after both targets (`restored=true`, `taken=false`). An earlier run returned `CLIPBRD_E_CANT_CLOSE` (`0x800401d4`) from `OleSetClipboard`; bounded retries while processing sent messages fixed it. A helper must not simply sleep through clipboard-manager contention: the other process may be waiting for this thread to service a delayed-rendering message.

## Recommended Windows mapping for `check`

1. Before posting, retain the focused `IUIAutomationElement`, its process ID, control type, framework, `IsPassword`, and ValuePattern read-only state. Read every available signal: ValuePattern value length and digest, TextPattern document length and digest, and TextPattern selection. Compute a selection's absolute offset by cloning DocumentRange, moving its end to the selection start, and measuring the prefix and selected text in UTF-16 units.
2. Post Ctrl+V, wait 25 ms, then read the retained element sequentially every 25 ms until `verifyMs`. Measure `checkMs` when the read completes. Do not use cached properties for the comparison.
3. Return `inserted` on the first difference in a signal present in both readings: value length, value digest, text length, text digest, or selection. The digest is required for a same-length replacement; selection catches caret movement when text does not change. As on macOS, this proves an observed field change, not that the requested bytes were inserted, and a concurrent user edit can produce the same verdict.
4. Return `unchanged` only after the full timeout when the same element still has focus and at least one trustworthy signal remained comparable and equal. For a TextPattern-only focus that is not Edit or Document, such as the console's Text element, downgrade a would-be `unchanged` to `unknown`; a whole buffer and a fixed `0+0` range are not a reliable negative result.
5. Return `unknown` when the baseline has no comparable signal, the last reading cannot be compared, the original element disappears or errors, focus moves, or the control shape cannot distinguish input from read-only output. Preserve the macOS timing convention: `checkMs` records the first change for `inserted` and the full watch for `unchanged`, and is omitted for `unknown`.

Secure and focus edge cases:

- `IsPassword=true` is the direct Windows password signal. Normal delivery should reject the field before reading or pasting. Never substitute masked ValuePattern text for the real value or infer `unchanged` from it. Secure desktops and credential prompts may also make UIA focus or SendInput unavailable; those paths are `unknown` (or a request error before posting), not proof of no insertion.
- A non-editable control with no value, document, or selection is `unknown`. A readable read-only editor can produce `unchanged`, but `textio.focus` should mark it non-editable so normal delivery never pastes there.
- Keep polling the original COM element, not whichever element is focused later. If no change is seen, reacquire the focused element and call `IUIAutomation.CompareElements`; a different element, no focus, or a failed identity check is `unknown`.

## Risks for `textio.selection` and `textio.read`

- UIA selection ranges have no absolute integer offset. The prefix-range method works in Notepad but adds document, prefix, and selection reads to every sample. Multiple selection ranges need a policy; comparing only the first range can miss changes in another range.
- `GetText(-1)` reads the whole document, so 25 ms polling can be expensive on large editors. Apply the existing 200,000 UTF-16-unit cap in the helper. Without ValuePattern, UIA has no cheap whole-document length property; a capped TextPattern-only document may need the existing `textio.read` error `unreadable` rather than a fabricated length.
- TextPattern providers can normalize line endings or expose stale/different text from ValuePattern. Offsets must be defined against the TextPattern document's UTF-16 text and tested with CRLF, surrogate pairs, Chromium, Electron, and browsers before relying on dictionary-learning ranges.
- UIA COM calls have no direct equivalent of the macOS per-element 250 ms messaging timeout. A hung provider must be isolated behind a helper call deadline; abandoning a blocked COM call on the same STA can leave the thread unusable.
- `textio.read.element` cannot be a COM pointer address because a later query returns a different pointer for the same field. Use a stable digest of the UIA runtime ID and validate its lifetime across focus changes and document replacement.
- This probe covers Notepad and one console provider only. It does not establish reliable verification for browser contenteditable controls, Word, terminal input, or secure fields.

## Helper-protocol consequence

Protocol 1's `textio.paste` result shape can be honored as-is: `posted`, `restored`, `clipboardTaken`, `ms`, `front`, `check`, `checkMs`, `field`, `pressSkipped`, and `kept` all have Windows implementations. Value/Text digest comparison remains internal, `field.before/after` carries length and selection, and `marker` is simply absent. The existing clipboard sequence number supports the `clipboardTaken` check, while OLE snapshot/restoration preserves the clipboard formats. No protocol-version bump is required for the probe's findings.

The following spec-level mappings still need an explicit decision; none forces a new result shape:

1. **Role vocabulary.** The app compares literal `AXSecureTextField`, `AXWebArea`, and other AX roles. Either map UIA control types to those existing values at the helper boundary (recommended, including `IsPassword` to `AXSecureTextField`) or make delivery platform-aware. Raw names such as `Button` and `Document` must not leak into the current AX role checks.
2. **Selection `via` vocabulary.** Decide whether UIA reads reuse `ax`/`ax-range` or add truthful `uia`/`uia-range` values. The latter changes protocol vocabulary and fixtures, although not the JSON shape.
3. **Secure-input reporting.** Windows has no direct equivalent of macOS global secure-event-input ownership. Recommended: omit `secureInput` on Windows, use UIA `IsPassword` for the focused field, and document that `ReasonSecureInput` is macOS-only. Reusing `secureInput` merely for a password field would change its meaning.
4. **Multiple selections.** For paste verification, compare a digest of the complete range array. For `textio.selection` and `textio.read`, choose and document either the first range (keeps protocol 1) or a new multi-range representation (requires a protocol decision and app changes).
5. **Element identity.** Define Windows `textio.read.element` as a stable runtime-ID digest, with the same process-lifetime expectation as the macOS CFHash; pointer identity is not acceptable.
