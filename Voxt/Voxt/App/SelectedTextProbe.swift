// SelectedTextProbe.swift
// System selection probe for dictionary / note / translation hotkeys.

import Foundation
import AppKit
import ApplicationServices

extension AppDelegate {
    private static let selectionAXFocusTimeout: Float = 0.2
    /// After primary FocusedUIElement failed, keep secondary AX cheap.
    private static let selectionAXFallbackTimeout: Float = 0.05

    /// Selected text for note / translation / rewrite hotkeys.
    /// Applies shared meaningful-content policy (rejects punctuation-only false positives).
    func selectedTextFromSystemSelection() -> String? {
        selectedContentTextFromSystemSelection()
    }

    /// Note / translation / rewrite: meaningful selected content only.
    func selectedContentTextFromSystemSelection() -> String? {
        SystemSelectionTextSupport.contentSelection(from: probeRawSelectedTextFromSystemSelection())
    }

    /// Dictionary hotkey: meaningful content + short-term caps.
    func selectedDictionaryTermFromSystemSelection() -> String? {
        SystemSelectionTextSupport.dictionaryCandidateTerm(from: probeRawSelectedTextFromSystemSelection())
    }

    /// Raw AX / AppleScript / clipboard probe result before hotkey content policies.
    func probeRawSelectedTextFromSystemSelection() -> String? {
        let appBundleID = NSWorkspace.shared.frontmostApplication?.bundleIdentifier ?? "unknown"
        let isBrowser = isBrowserBundleID(appBundleID == "unknown" ? nil : appBundleID)

        guard AccessibilityPermissionManager.isTrusted() else {
            logSelectionProbeResult(
                textSource: "nil",
                app: appBundleID,
                detail: "accessibility-not-trusted"
            )
            return nil
        }

        let focusResolve = resolveFocusForSystemSelection()
        let focusedElement = focusResolve.focusedElement
        VoxtLog.input(
            "selectionProbe.focus: available=\(focusedElement != nil) windows=\(focusResolve.candidateWindowCount) app=\(appBundleID)",
            verbose: true
        )

        let rawSelectedRange = focusedElement.flatMap {
            axRangeAttribute(
                kAXSelectedTextRangeAttribute as CFString,
                for: $0,
                timeout: Self.selectionAXFocusTimeout
            )
        }

        if SelectedTextSystemSelectionSupport.isConfirmedCaretOnly(selectedTextRange: rawSelectedRange) {
            logSelectionProbeResult(
                textSource: "nil",
                app: appBundleID,
                detail: "confirmed-caret-only"
            )
            return nil
        }

        switch readSelectedTextNonDestructively(
            focusedElement: focusedElement,
            rawSelectedRange: rawSelectedRange,
            isBrowser: isBrowser,
            appBundleID: appBundleID
        ) {
        case .text(let text):
            return text
        case .confirmedEmpty:
            logSelectionProbeResult(
                textSource: "nil",
                app: appBundleID,
                detail: "browser-confirmed-empty"
            )
            return nil
        case .miss:
            break
        }

        return readSelectedTextBySimulatedCopyIfAllowed(
            focusedElement: focusedElement,
            rawSelectedRange: rawSelectedRange,
            isBrowser: isBrowser,
            appBundleID: appBundleID,
            axWindowCandidatesAvailable: focusResolve.candidateWindowCount > 0
        )
    }

    // MARK: - Non-destructive reads

    private enum NonDestructiveSelectionRead {
        case text(String)
        case confirmedEmpty
        case miss
    }

    private func readSelectedTextNonDestructively(
        focusedElement: AXUIElement?,
        rawSelectedRange: CFRange?,
        isBrowser: Bool,
        appBundleID: String
    ) -> NonDestructiveSelectionRead {
        if let focusedElement,
           let text = probeAXSelectedText(from: focusedElement) {
            logSelectionProbeResult(
                textSource: "axSelected",
                app: appBundleID,
                detail: "chars=\(text.count)"
            )
            return .text(text)
        }

        if let focusedElement,
           let selectedRange = nonEmptySelectedTextRange(rawSelectedRange) {
            let rangeText = axParameterizedString(
                kAXStringForRangeParameterizedAttribute as CFString,
                range: selectedRange,
                for: focusedElement,
                timeout: Self.selectionAXFocusTimeout
            )
            let trimmedRangeText = rangeText?.trimmingCharacters(in: .whitespacesAndNewlines)
            if let trimmedRangeText, !trimmedRangeText.isEmpty {
                logSelectionProbeResult(
                    textSource: "stringForRange",
                    app: appBundleID,
                    detail: "chars=\(trimmedRangeText.count)"
                )
                return .text(rangeText ?? trimmedRangeText)
            }
        }

        if let focusedElement,
           let text = selectedTextFromAXTextMarker(startingAt: focusedElement) {
            logSelectionProbeResult(
                textSource: "textMarker",
                app: appBundleID,
                detail: "chars=\(text.count)"
            )
            return .text(text)
        }

        if isBrowser {
            let axRole = focusedElement.flatMap {
                axStringAttribute(
                    kAXRoleAttribute as CFString,
                    for: $0,
                    timeout: Self.selectionAXFocusTimeout
                )
            }
            let axFocusAvailable = focusedElement != nil
            let axFocusedTextControl = SelectedTextSystemSelectionSupport
                .isBrowserTextControlRole(axRole)
            switch selectedTextFromBrowserAppleScript(
                bundleID: appBundleID,
                axFocusedTextControl: axFocusedTextControl,
                axFocusAvailable: axFocusAvailable
            ) {
            case .selected(let text):
                logSelectionProbeResult(
                    textSource: "appleScript",
                    app: appBundleID,
                    detail: "chars=\(text.count)"
                )
                return .text(text)
            case .confirmedEmpty:
                return .confirmedEmpty
            case .unavailable:
                break
            }
        }

        return .miss
    }

    private func readSelectedTextBySimulatedCopyIfAllowed(
        focusedElement: AXUIElement?,
        rawSelectedRange: CFRange?,
        isBrowser: Bool,
        appBundleID: String,
        axWindowCandidatesAvailable: Bool
    ) -> String? {
        let copiesLineOnEmptySelection = SelectedTextSystemSelectionSupport
            .copiesCurrentLineWhenSelectionEmpty(bundleID: appBundleID)
        let allowCopy = SelectedTextSystemSelectionSupport.shouldAttemptSimulatedCopy(
            focusedElementAvailable: focusedElement != nil,
            selectedTextRange: rawSelectedRange,
            isBrowser: isBrowser,
            copiesLineOnEmptySelection: copiesLineOnEmptySelection
        )
        let allowBlackoutProbe = SelectedTextSystemSelectionSupport.shouldAttemptAXBlackoutClipboardProbe(
            focusedElementAvailable: focusedElement != nil,
            axWindowCandidatesAvailable: axWindowCandidatesAvailable,
            copiesLineOnEmptySelection: copiesLineOnEmptySelection,
            supportsAXBlackoutSelectionRecovery: !SelectedTextSystemSelectionSupport
                .ignoresSelectedTextFlow(bundleID: appBundleID)
        )
        VoxtLog.input(
            "selectionProbe.copy.policy: allow=\(allowCopy) blackout=\(allowBlackoutProbe) focused=\(focusedElement != nil) windows=\(axWindowCandidatesAvailable) copiesLineOnEmpty=\(copiesLineOnEmptySelection)",
            verbose: true
        )

        if allowCopy, let text = selectedTextBySimulatedCopy() {
            logSelectionProbeResult(
                textSource: "copy",
                app: appBundleID,
                detail: "chars=\(text.count)"
            )
            return text
        }

        if allowBlackoutProbe {
            if let capture = captureTextBySimulatedCopy() {
                if SelectedTextSystemSelectionSupport.looksLikeEmptySelectionLineCopy(
                    rawClipboardText: capture.raw
                ) {
                    logSelectionProbeResult(
                        textSource: "nil",
                        app: appBundleID,
                        detail: "ax-blackout-rejected-line-copy"
                    )
                    return nil
                }
                logSelectionProbeResult(
                    textSource: "copyBlackoutFiltered",
                    app: appBundleID,
                    detail: "chars=\(capture.trimmed.count)"
                )
                return capture.trimmed
            }
        }

        let denyDetail = SelectedTextSystemSelectionSupport.denialDetail(
            allowCopy: allowCopy,
            focusedElementAvailable: focusedElement != nil,
            selectedTextRange: rawSelectedRange,
            isBrowser: isBrowser,
            copiesLineOnEmptySelection: copiesLineOnEmptySelection
        )
        logSelectionProbeResult(
            textSource: "nil",
            app: appBundleID,
            detail: denyDetail
        )
        return nil
    }

    // MARK: - Focus resolve

    private struct SystemSelectionFocusResolve {
        let focusedElement: AXUIElement?
        let candidateWindowCount: Int
    }

    private func resolveFocusForSystemSelection() -> SystemSelectionFocusResolve {
        // Actual focused element only — never best-editable / leftover-field fallbacks.
        if let systemFocused = copyFocusedUIElement(
            from: AXUIElementCreateSystemWide(),
            source: "systemWide"
        ) {
            return SystemSelectionFocusResolve(focusedElement: systemFocused, candidateWindowCount: 0)
        }

        guard let processID = NSWorkspace.shared.frontmostApplication?.processIdentifier else {
            return SystemSelectionFocusResolve(focusedElement: nil, candidateWindowCount: 0)
        }

        let appElement = AXUIElementCreateApplication(processID)
        if let appFocused = copyFocusedUIElement(from: appElement, source: "frontmostApp") {
            return SystemSelectionFocusResolve(focusedElement: appFocused, candidateWindowCount: 0)
        }

        let candidateWindows = selectionProbeCandidateWindows(from: appElement)
        guard !candidateWindows.isEmpty else {
            return SystemSelectionFocusResolve(focusedElement: nil, candidateWindowCount: 0)
        }

        for window in candidateWindows {
            if let descendant = findFocusedDescendant(in: window, depthRemaining: 6) {
                return SystemSelectionFocusResolve(
                    focusedElement: descendant,
                    candidateWindowCount: candidateWindows.count
                )
            }
        }

        return SystemSelectionFocusResolve(
            focusedElement: nil,
            candidateWindowCount: candidateWindows.count
        )
    }

    private func selectionProbeCandidateWindows(from appElement: AXUIElement) -> [AXUIElement] {
        var windows: [AXUIElement] = []

        func appendUnique(_ window: AXUIElement) {
            for existing in windows where CFEqual(existing, window) {
                return
            }
            windows.append(window)
        }

        if let focusedWindow = axElementAttribute(
            kAXFocusedWindowAttribute as CFString,
            for: appElement,
            timeout: Self.selectionAXFallbackTimeout
        ) {
            appendUnique(focusedWindow)
        }
        if let mainWindow = axElementAttribute(
            kAXMainWindowAttribute as CFString,
            for: appElement,
            timeout: Self.selectionAXFallbackTimeout
        ) {
            appendUnique(mainWindow)
        }
        let listedWindows = axElementArrayAttribute(
            kAXWindowsAttribute as CFString,
            for: appElement,
            timeout: Self.selectionAXFallbackTimeout
        )
        for window in listedWindows.prefix(3) {
            appendUnique(window)
        }
        return windows
    }

    private func copyFocusedUIElement(from root: AXUIElement, source: String) -> AXUIElement? {
        AXUIElementSetMessagingTimeout(root, Self.selectionAXFocusTimeout)
        var focusedElementRef: CFTypeRef?
        let focusedStatus = AXUIElementCopyAttributeValue(
            root,
            kAXFocusedUIElementAttribute as CFString,
            &focusedElementRef
        )
        guard focusedStatus == .success,
              let focusedElementRef,
              CFGetTypeID(focusedElementRef) == AXUIElementGetTypeID() else {
            VoxtLog.input(
                "selectionProbe.focus.\(source): failed status=\(focusedStatus.rawValue)",
                verbose: true
            )
            return nil
        }
        return unsafeBitCast(focusedElementRef, to: AXUIElement.self)
    }

    // MARK: - AX / AppleScript / clipboard readers

    private func probeAXSelectedText(from element: AXUIElement) -> String? {
        AXUIElementSetMessagingTimeout(element, Self.selectionAXFocusTimeout)
        var selectedTextRef: CFTypeRef?
        let selectedStatus = AXUIElementCopyAttributeValue(
            element,
            kAXSelectedTextAttribute as CFString,
            &selectedTextRef
        )
        guard selectedStatus == .success, let selectedTextRef else { return nil }

        let selectedText: String?
        if let text = selectedTextRef as? String {
            selectedText = text
        } else if let attributed = selectedTextRef as? NSAttributedString {
            selectedText = attributed.string
        } else {
            selectedText = nil
        }

        let trimmed = selectedText?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return trimmed.isEmpty ? nil : selectedText
    }

    private func selectedTextFromAXTextMarker(startingAt element: AXUIElement) -> String? {
        var current: AXUIElement? = element
        for _ in 0..<6 {
            guard let currentElement = current else { break }
            if let text = selectedTextFromAXTextMarkerRange(on: currentElement) {
                return text
            }
            current = axElementAttribute(
                kAXParentAttribute as CFString,
                for: currentElement,
                timeout: Self.selectionAXFocusTimeout
            )
        }
        return nil
    }

    private func selectedTextFromAXTextMarkerRange(on element: AXUIElement) -> String? {
        AXUIElementSetMessagingTimeout(element, Self.selectionAXFocusTimeout)
        var markerRangeRef: CFTypeRef?
        let markerStatus = AXUIElementCopyAttributeValue(
            element,
            "AXSelectedTextMarkerRange" as CFString,
            &markerRangeRef
        )
        guard markerStatus == .success, let markerRangeRef else { return nil }

        var stringRef: CFTypeRef?
        let stringStatus = AXUIElementCopyParameterizedAttributeValue(
            element,
            "AXStringForTextMarkerRange" as CFString,
            markerRangeRef,
            &stringRef
        )
        if stringStatus == .success,
           let text = stringRef as? String,
           !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return text
        }

        var attributedRef: CFTypeRef?
        let attributedStatus = AXUIElementCopyParameterizedAttributeValue(
            element,
            "AXAttributedStringForTextMarkerRange" as CFString,
            markerRangeRef,
            &attributedRef
        )
        guard attributedStatus == .success else { return nil }
        if let attributed = attributedRef as? NSAttributedString,
           !attributed.string.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return attributed.string
        }
        if let text = attributedRef as? String,
           !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return text
        }
        return nil
    }

    private func selectedTextFromBrowserAppleScript(
        bundleID: String,
        axFocusedTextControl: Bool,
        axFocusAvailable: Bool
    ) -> SelectedTextSystemSelectionSupport.BrowserSelectionProbeOutcome {
        if let deniedUntil = browserAutomationDeniedUntilByBundleID[bundleID],
           deniedUntil > Date() {
            return .unavailable
        }

        let displayName = browserScriptProvider(for: bundleID)?.name
        // Always allow tagged page selections. Form-first JS returns `F` empty for
        // caret-only inputs, so residual page ranges do not leak while typing.
        // AX focus is frequently unavailable in Arc/Chromium web content; gating
        // page selection on AX would break article note/translate hotkeys there.
        let scripts = BrowserAutomationScriptBuilder.selectionScripts(
            bundleID: bundleID,
            displayName: displayName,
            allowPageSelection: true
        )
        for source in scripts {
            var executionError: NSDictionary?
            // `runAppleScript` ceilings fractional timeouts to whole seconds; pass 1 explicitly.
            let rawOutput = runAppleScript(
                source,
                error: &executionError,
                logFailure: false,
                timeout: 1
            )

            if let outcome = SelectedTextSystemSelectionSupport.browserSelectionProbeOutcome(
                output: rawOutput,
                hadExecutionError: executionError != nil,
                axFocusedTextControl: axFocusedTextControl,
                axFocusAvailable: axFocusAvailable
            ) {
                return outcome
            }

            if let errorNumber = executionError?["NSAppleScriptErrorNumber"] as? Int {
                if errorNumber == -1743 || errorNumber == -10004 {
                    browserAutomationDeniedUntilByBundleID[bundleID] = Date().addingTimeInterval(300)
                    break
                }
                if errorNumber == -600 {
                    break
                }
            }
        }
        return .unavailable
    }

    private struct SimulatedCopyCapture {
        let raw: String

        var trimmed: String {
            raw.trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }

    private func selectedTextBySimulatedCopy() -> String? {
        captureTextBySimulatedCopy()?.trimmed
    }

    private func captureTextBySimulatedCopy() -> SimulatedCopyCapture? {
        guard AccessibilityPermissionManager.isTrusted() else { return nil }
        guard let source = CGEventSource(stateID: .hidSystemState) else { return nil }

        let pasteboard = NSPasteboard.general
        let previous = readStringFromPasteboard(pasteboard)
        let originalChangeCount = pasteboard.changeCount

        let cKeyCode: CGKeyCode = 0x08
        let cmdDown = CGEvent(keyboardEventSource: source, virtualKey: cKeyCode, keyDown: true)
        cmdDown?.flags = .maskCommand
        let cmdUp = CGEvent(keyboardEventSource: source, virtualKey: cKeyCode, keyDown: false)
        cmdUp?.flags = .maskCommand
        guard cmdDown != nil, cmdUp != nil else { return nil }

        HotkeyEventSupport.markAsVoxtInjected(cmdDown)
        HotkeyEventSupport.markAsVoxtInjected(cmdUp)
        cmdDown?.post(tap: .cgAnnotatedSessionEventTap)
        cmdUp?.post(tap: .cgAnnotatedSessionEventTap)

        let waitSeconds: TimeInterval = 0.15
        let deadline = Date().addingTimeInterval(waitSeconds)
        while pasteboard.changeCount == originalChangeCount, Date() < deadline {
            _ = RunLoop.current.run(mode: .default, before: Date().addingTimeInterval(0.01))
        }

        guard pasteboard.changeCount != originalChangeCount else { return nil }

        let raw = readStringFromPasteboard(pasteboard) ?? ""

        pasteboard.clearContents()
        if let previous, !previous.isEmpty {
            pasteboard.setString(previous, forType: .string)
        }

        let capture = SimulatedCopyCapture(raw: raw)
        guard !capture.trimmed.isEmpty else { return nil }
        return capture
    }

    // MARK: - Helpers

    private func nonEmptySelectedTextRange(_ range: CFRange?) -> CFRange? {
        guard let range,
              SelectedTextSystemSelectionSupport.hasNonEmptySelectedTextRange(length: range.length) else {
            return nil
        }
        return range
    }

    private func logSelectionProbeResult(
        textSource: String,
        app: String,
        detail: String = ""
    ) {
        let suffix = detail.isEmpty ? "" : " detail=\(detail)"
        VoxtLog.input("selectionProbe: source=\(textSource) app=\(app)\(suffix)")
    }
}
