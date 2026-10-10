// SelectedTextSystemSelectionSupport.swift
// Capability-based policy for system selection probing (no AppKit UI).

import Foundation
import ApplicationServices

enum SelectedTextSystemSelectionSupport {
    static func ignoresSelectedTextFlow(bundleID: String?) -> Bool {
        bundleID == "md.obsidian"
    }

    /// Hard gate: a caret-only range (`length == 0`) must not count as a selection.
    static func hasNonEmptySelectedTextRange(length: CFIndex) -> Bool {
        length > 0
    }

    /// Editors that implement "Cmd+C with no selection copies the current line"
    /// (VS Code `editor.emptySelectionClipboard`, Sublime, Zed, JetBrains, etc.).
    ///
    /// This is a clipboard-semantics capability class. Many of these apps also
    /// fail AX focus lookup with `kAXErrorCannotComplete` (-25204), so
    /// "focused == nil" alone cannot mean "safe to Cmd+C" — that path remains
    /// necessary for AX-dead apps such as WeChat, but must stay denied here.
    static func copiesCurrentLineWhenSelectionEmpty(bundleID: String?) -> Bool {
        guard let bundleID, !bundleID.isEmpty else { return false }

        let exactIDs: Set<String> = [
            "com.microsoft.VSCode",
            "com.microsoft.VSCodeInsiders",
            "com.microsoft.VSCodeExploration",
            "com.visualstudio.code.oss",
            "com.visualstudio.code",
            "com.vscodium",
            "com.google.antigravity",
            "com.sublimetext.3",
            "com.sublimetext.4",
            "dev.zed.Zed",
            "dev.zed.Zed-Preview",
            "com.trae.app",
            "com.trae.solo.app",
            "cn.trae.app",
            "com.qoder.app",
            "md.obsidian"
        ]
        if exactIDs.contains(bundleID) {
            return true
        }

        let prefixes = [
            "com.todesktop.",
            "com.jetbrains.",
            "com.exafunction.",
            "com.trae.",
            "cn.trae.",
            "com.qoder."
        ]
        if prefixes.contains(where: bundleID.hasPrefix) {
            return true
        }

        let lowered = bundleID.lowercased()
        return lowered.contains("vscode") || lowered.contains("vscodium")
    }

    /// Whether Cmd+C may run after all non-destructive reads missed.
    ///
    /// Capability rule (not per-app special cases):
    /// - caret-only range (`length == 0`) → deny
    /// - non-empty range → allow (recover text when attributes are empty)
    /// - no focused element:
    ///   - deny when the frontmost app's clipboard class invents a line on empty selection
    ///   - allow otherwise (AX-dead apps that do not invent clipboard content)
    /// - focused element but no range attribute → deny
    ///   (including browsers: Cmd+C can copy residual page selection while a
    ///   focused input has only a caret; browser selection uses AppleScript JS)
    ///
    /// Line-copy editors with a total AX blackout use a separate filtered probe
    /// (`shouldAttemptAXBlackoutClipboardProbe`); do not widen this gate for them.
    static func shouldAttemptSimulatedCopy(
        focusedElementAvailable: Bool,
        selectedTextRange: CFRange?,
        isBrowser: Bool,
        copiesLineOnEmptySelection: Bool
    ) -> Bool {
        _ = isBrowser
        if focusedElementAvailable, let selectedTextRange {
            return selectedTextRange.length > 0
        }
        if !focusedElementAvailable {
            return !copiesLineOnEmptySelection
        }
        return false
    }

    /// Last-resort clipboard probe for the line-copy capability class when AX exposes
    /// neither a focused element nor any window candidates.
    ///
    /// Without this path, Electron editors that fail FocusedUIElement (-25204) and
    /// return an empty AX window list can never recover a real selection. Results must
    /// still pass `looksLikeEmptySelectionLineCopy` filtering.
    static func shouldAttemptAXBlackoutClipboardProbe(
        focusedElementAvailable: Bool,
        axWindowCandidatesAvailable: Bool,
        copiesLineOnEmptySelection: Bool,
        supportsAXBlackoutSelectionRecovery: Bool = true
    ) -> Bool {
        supportsAXBlackoutSelectionRecovery
            && copiesLineOnEmptySelection
            && !focusedElementAvailable
            && !axWindowCandidatesAvailable
    }

    /// Best-effort filter for `emptySelectionClipboard`-style payloads:
    /// exactly one line plus a trailing newline. Mid-line selections usually omit
    /// that trailing newline; multi-line selections contain interior newlines.
    ///
    /// Residual risk: last lines without EOL may false-accept; intentional whole-line
    /// selections that include EOL may false-reject.
    static func looksLikeEmptySelectionLineCopy(rawClipboardText: String) -> Bool {
        let normalized = rawClipboardText
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
        let trimmed = normalized.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return true }
        guard normalized.hasSuffix("\n") else { return false }
        let body = String(normalized.dropLast())
        return !body.contains("\n")
    }

    /// Range attribute present with `length == 0` means caret-only; further probes cannot help.
    static func isConfirmedCaretOnly(selectedTextRange: CFRange?) -> Bool {
        guard let selectedTextRange else { return false }
        return selectedTextRange.length == 0
    }

    /// AppleScript ran without an execution error and returned empty/whitespace selection text.
    /// Dialect retries are only useful when the script form itself fails.
    static func isDefinitiveEmptyBrowserSelection(output: String?, hadExecutionError: Bool) -> Bool {
        guard !hadExecutionError, let output else { return false }
        return output.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    /// Outcome of a browser AppleScript selection probe.
    ///
    /// `confirmedEmpty` means JS executed successfully and reported no selection
    /// (including caret-only focused inputs). Callers must not fall through to
    /// simulated Cmd+C, which can still copy residual page selection.
    enum BrowserSelectionProbeOutcome: Equatable {
        case selected(String)
        case confirmedEmpty
        case unavailable
    }

    enum BrowserSelectionSource: Equatable {
        case form
        case editor
        case page
        /// Untagged legacy/plain payload (treat like page for distrust rules).
        case unknown
    }

    struct BrowserSelectionPayload: Equatable {
        let source: BrowserSelectionSource
        let text: String
    }

    /// Chromium/Safari Apple Events often return JS strings as a quoted AppleScript
    /// string value, e.g. `"F::V::789"` (literal quote characters included).
    /// Production logs showed `rawChars=11 prefix="F::V::789"` for selected `789`.
    static func normalizeBrowserSelectionScriptOutput(_ raw: String?) -> String? {
        guard let raw else { return nil }
        var normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if normalized.count >= 2, normalized.first == "\"", normalized.last == "\"" {
            normalized.removeFirst()
            normalized.removeLast()
            normalized = normalized
                .replacingOccurrences(of: "\\\"", with: "\"")
                .replacingOccurrences(of: "\\\\", with: "\\")
        }
        return normalized
    }

    /// Parses `<F|E|P>::V::<text>` from `BrowserAutomationScriptBuilder.selectionJavaScript`.
    static func parseBrowserSelectionScriptOutput(_ raw: String?) -> BrowserSelectionPayload? {
        guard let raw = normalizeBrowserSelectionScriptOutput(raw) else { return nil }
        let separator = BrowserAutomationScriptBuilder.selectionSourceSeparator
        let minimumTaggedLength = 1 + separator.count
        guard raw.count >= minimumTaggedLength else {
            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return BrowserSelectionPayload(source: .unknown, text: "") }
            return BrowserSelectionPayload(source: .unknown, text: trimmed)
        }

        let sourceToken = String(raw.prefix(1))
        let remainder = raw.dropFirst()
        guard remainder.hasPrefix(separator) else {
            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return BrowserSelectionPayload(source: .unknown, text: "") }
            return BrowserSelectionPayload(source: .unknown, text: trimmed)
        }

        let text = String(remainder.dropFirst(separator.count))
        let source: BrowserSelectionSource
        switch sourceToken {
        case BrowserAutomationScriptBuilder.selectionSourceForm:
            source = .form
        case BrowserAutomationScriptBuilder.selectionSourceEditor:
            source = .editor
        case BrowserAutomationScriptBuilder.selectionSourcePage:
            source = .page
        default:
            source = .unknown
        }
        return BrowserSelectionPayload(source: source, text: text)
    }

    /// AX roles that mean "caret is in a text control", where a page-level
    /// `getSelection()` hit is usually a residual false positive.
    static func isBrowserTextControlRole(_ role: String?) -> Bool {
        guard let role, !role.isEmpty else { return false }
        let textControlRoles: Set<String> = [
            kAXTextFieldRole as String,
            kAXTextAreaRole as String,
            "AXSearchField",
            kAXComboBoxRole as String
        ]
        return textControlRoles.contains(role)
    }

    /// Definitive AppleScript decision plus a stable reason for logs/diagnostics.
    /// `nil` means "script form failed — try the next dialect".
    struct BrowserSelectionProbeDecision: Equatable {
        let outcome: BrowserSelectionProbeOutcome
        let reason: String
        let payload: BrowserSelectionPayload?
    }

    static func browserSelectionProbeOutcome(
        output: String?,
        hadExecutionError: Bool,
        axFocusedTextControl: Bool = false,
        axFocusAvailable: Bool = true
    ) -> BrowserSelectionProbeOutcome? {
        browserSelectionProbeDecision(
            output: output,
            hadExecutionError: hadExecutionError,
            axFocusedTextControl: axFocusedTextControl,
            axFocusAvailable: axFocusAvailable
        )?.outcome
    }

    static func browserSelectionProbeDecision(
        output: String?,
        hadExecutionError: Bool,
        axFocusedTextControl: Bool = false,
        axFocusAvailable: Bool = true
    ) -> BrowserSelectionProbeDecision? {
        if hadExecutionError {
            return nil
        }
        guard let payload = parseBrowserSelectionScriptOutput(output) else {
            return nil
        }

        let trimmed = payload.text.trimmingCharacters(in: .whitespacesAndNewlines)
        let sourceLabel = browserSelectionSourceLabel(payload.source)
        if trimmed.isEmpty {
            return BrowserSelectionProbeDecision(
                outcome: .confirmedEmpty,
                reason: "js-empty source=\(sourceLabel)",
                payload: payload
            )
        }

        let isPageLike = payload.source == .page || payload.source == .unknown

        // AX says a text field/area is focused, but JS only found a page-level
        // selection (common when Apple Events misses the real activeElement and
        // falls through to residual page ranges). Do not treat that as selected text.
        if axFocusedTextControl, isPageLike {
            return BrowserSelectionProbeDecision(
                outcome: .confirmedEmpty,
                reason: "reject-page-like-while-ax-text-control source=\(sourceLabel) chars=\(trimmed.count)",
                payload: payload
            )
        }

        // Untagged payloads are ambiguous. When AX focus is unavailable (common in
        // Chromium/Arc web content), only trust explicitly tagged F/E/P results from
        // our JS. Tagged page selections (`P`) must still be accepted so article
        // highlights work while the caret is not in an input — form-first JS already
        // returns `F` empty instead of residual page text when an input is focused.
        if !axFocusAvailable, payload.source == .unknown {
            return BrowserSelectionProbeDecision(
                outcome: .confirmedEmpty,
                reason: "reject-untagged-while-ax-focus-unavailable chars=\(trimmed.count)",
                payload: payload
            )
        }

        return BrowserSelectionProbeDecision(
            outcome: .selected(trimmed),
            reason: "accept source=\(sourceLabel) chars=\(trimmed.count) axFocus=\(axFocusAvailable) textControl=\(axFocusedTextControl)",
            payload: payload
        )
    }

    static func browserSelectionSourceLabel(_ source: BrowserSelectionSource) -> String {
        switch source {
        case .form: return "F"
        case .editor: return "E"
        case .page: return "P"
        case .unknown: return "U"
        }
    }

    /// Compact, privacy-safe debug summary of the AppleScript JS return value.
    /// Shows whether the printable tag survived the bridge, without dumping full text.
    static func browserSelectionRawDebugSummary(_ raw: String?) -> String {
        guard let raw else { return "raw=nil" }
        let normalized = normalizeBrowserSelectionScriptOutput(raw) ?? ""
        let separator = BrowserAutomationScriptBuilder.selectionSourceSeparator
        let hasSeparator = raw.contains(separator) || normalized.contains(separator)
        let quoted = raw.count >= 2 && raw.trimmingCharacters(in: .whitespacesAndNewlines).first == "\""
        let prefix = String(raw.prefix(24))
            .replacingOccurrences(of: "\n", with: "\\n")
            .replacingOccurrences(of: "\r", with: "\\r")
            .replacingOccurrences(of: "\t", with: "\\t")
        let payload = parseBrowserSelectionScriptOutput(raw)
        let source = payload.map { browserSelectionSourceLabel($0.source) } ?? "?"
        let textChars = payload?.text.trimmingCharacters(in: .whitespacesAndNewlines).count ?? -1
        return "rawChars=\(raw.count) normalizedChars=\(normalized.count) quoted=\(quoted) hasSep=\(hasSeparator) source=\(source) textChars=\(textChars) prefix=\(prefix)"
    }

    /// Stable deny reason for logs / diagnostics after copy policy rejects.
    static func denialDetail(
        allowCopy: Bool,
        focusedElementAvailable: Bool,
        selectedTextRange: CFRange?,
        isBrowser: Bool,
        copiesLineOnEmptySelection: Bool
    ) -> String {
        if allowCopy {
            return "copy-missed"
        }
        if !focusedElementAvailable, copiesLineOnEmptySelection {
            return "ax-focus-unavailable-line-copy-editor"
        }
        if focusedElementAvailable, selectedTextRange?.length == 0 {
            return "confirmed-caret-only"
        }
        if focusedElementAvailable, selectedTextRange == nil, !isBrowser {
            return "focused-without-range-non-browser"
        }
        return "copy-denied"
    }
}
