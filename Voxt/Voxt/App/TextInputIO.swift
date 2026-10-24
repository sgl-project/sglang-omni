// TextInputIO.swift
// Provides Text Input IO for app lifecycle and routing.

import Foundation
import AppKit
import ApplicationServices
import Carbon

extension AppDelegate {
    private static let axMessagingTimeout: Float = 0.05
    private static let automaticDictionaryLearningSnippetBeforeCursor = 4_000
    private static let automaticDictionaryLearningSnippetAfterCursor = 1_000
    private static let nativeWritableTextRoles: Set<String> = [
        kAXTextAreaRole as String,
        kAXTextFieldRole as String,
        "AXSearchField",
        kAXComboBoxRole as String
    ]
    private static let genericEditableTextRoles: Set<String> = [
        "AXWebArea",
        "AXGroup",
        "AXLayoutArea",
        "AXScrollArea",
        "AXDocument",
        "AXUnknown"
    ]
    private static let nonEditableFalsePositiveRoles: Set<String> = [
        kAXWindowRole as String,
        kAXButtonRole as String,
        kAXStaticTextRole as String,
        "AXToolbar",
        "AXMenuBar",
        "AXMenuItem",
        "AXMenu",
        "AXSplitter",
        "AXList",
        "AXTable",
        "AXOutline",
        "AXRow"
    ]

    struct FocusedInputTextSnapshot {
        let text: String
        let bundleIdentifier: String?
        let processIdentifier: pid_t?
        let role: String?
        let isEditable: Bool
        let isFocusedTarget: Bool
        let selectedRange: NSRange?
        let failureReason: String?
        let textSource: String?
    }

    func hasWritableFocusedTextInput() -> Bool {
        guard let processID = NSWorkspace.shared.frontmostApplication?.processIdentifier,
              let focusedElement = focusedAXElement(preferredProcessID: processID) else {
            VoxtLog.input("Focused input check: no focused AX element.")
            return false
        }

        if let writableElement = writableTextInputElement(from: focusedElement) {
            let role = axStringAttribute(kAXRoleAttribute as CFString, for: writableElement) ?? "unknown"
            let editable = isWritableTextInputElement(writableElement)
            let valueSettable = isAttributeSettable(kAXValueAttribute as CFString, on: writableElement)
            VoxtLog.input(
                "Focused input check: writable descendant detected. role=\(role), editable=\(editable), valueSettable=\(valueSettable)"
            )
            return true
        }

        let editable = axBoolAttribute("AXEditable" as CFString, for: focusedElement)
        if editable == true {
            let role = axStringAttribute(kAXRoleAttribute as CFString, for: focusedElement) ?? "unknown"
            VoxtLog.input("Focused input check: editable AX element detected. role=\(role)")
            return true
        }

        if isAttributeSettable(kAXValueAttribute as CFString, on: focusedElement) {
            let role = axStringAttribute(kAXRoleAttribute as CFString, for: focusedElement) ?? "unknown"
            VoxtLog.input("Focused input check: settable AX value detected. role=\(role)")
            return true
        }

        guard let role = axStringAttribute(kAXRoleAttribute as CFString, for: focusedElement) else {
            VoxtLog.input(
                "Focused input check: role unavailable. editable=\(editable == true), valueSettable=false"
            )
            return false
        }

        let isWritable = Self.nativeWritableTextRoles.contains(role)
        VoxtLog.input(
            "Focused input check: role=\(role), editable=\(editable == true), valueSettable=false, result=\(isWritable)"
        )
        return isWritable
    }

    func currentFocusedInputTextSnapshot(
        expectedBundleID: String? = nil,
        logDiagnostics: Bool = true
    ) -> FocusedInputTextSnapshot? {
        guard let frontmostApplication = NSWorkspace.shared.frontmostApplication else {
            if logDiagnostics {
                VoxtLog.input("Focused input snapshot unavailable: no frontmost application.")
            }
            return nil
        }
        if let expectedBundleID,
           let bundleIdentifier = frontmostApplication.bundleIdentifier,
           bundleIdentifier != expectedBundleID {
            if logDiagnostics {
                VoxtLog.input(
                    "Focused input snapshot skipped: frontmost app changed. expectedBundleID=\(expectedBundleID), actualBundleID=\(bundleIdentifier)"
                )
            }
            return nil
        }

        let bundleIdentifier = frontmostApplication.bundleIdentifier
        let processIdentifier = frontmostApplication.processIdentifier

        guard let focusedElement = focusedAXElement(
            preferredProcessID: processIdentifier,
            logDiagnostics: logDiagnostics
        ) else {
            if logDiagnostics {
                VoxtLog.input(
                    "Focused input snapshot unavailable: no focused AX element. bundleID=\(bundleIdentifier ?? "unknown")"
                )
            }
            return nil
        }

        let writableElement = writableTextInputElement(from: focusedElement) ?? (
            isWritableTextInputElement(focusedElement) ? focusedElement : nil
        )
        guard let writableElement else {
            let role = axStringAttribute(kAXRoleAttribute as CFString, for: focusedElement) ?? "unknown"
            if logDiagnostics {
                VoxtLog.input(
                    "Focused input snapshot unavailable: no writable text element found. bundleID=\(bundleIdentifier ?? "unknown"), role=\(role)"
                )
            }
            return nil
        }

        let role = axStringAttribute(kAXRoleAttribute as CFString, for: writableElement)
        let isFocusedTarget = axBoolAttribute(kAXFocusedAttribute as CFString, for: writableElement) == true
        let value = axTextValue(for: writableElement)?.trimmingCharacters(in: .whitespacesAndNewlines)

        guard let text = value, !text.isEmpty else {
            let textFailureReason = unreadableTextFailureReason(for: writableElement)
            let role = axStringAttribute(kAXRoleAttribute as CFString, for: writableElement) ?? "unknown"
            if logDiagnostics {
                VoxtLog.input(
                    "Focused input snapshot unavailable: writable element has empty/unreadable value. bundleID=\(bundleIdentifier ?? "unknown"), role=\(role), failureReason=\(textFailureReason)"
                )
            }
            return nil
        }

        let snapshot = FocusedInputTextSnapshot(
            text: text,
            bundleIdentifier: bundleIdentifier,
            processIdentifier: processIdentifier,
            role: role,
            isEditable: isWritableTextInputElement(writableElement),
            isFocusedTarget: isFocusedTarget,
            selectedRange: selectedNSRange(from: axRangeAttribute(kAXSelectedTextRangeAttribute as CFString, for: writableElement)),
            failureReason: nil,
            textSource: "ax-value"
        )
        if logDiagnostics {
            VoxtLog.input("Focused input snapshot ready: \(focusedInputSnapshotSummary(snapshot))")
        }
        return snapshot
    }

    func currentFocusedInputTextSnapshotForAutomaticDictionaryLearning(
        expectedBundleID: String? = nil
    ) async -> FocusedInputTextSnapshot? {
        let startedAt = Date()
        defer {
            let elapsed = Date().timeIntervalSince(startedAt)
            if elapsed >= 0.08 {
                let bundleID = NSWorkspace.shared.frontmostApplication?.bundleIdentifier ?? "unknown"
                VoxtLog.input(
                    "Automatic dictionary focused input snapshot was slow. elapsedMs=\(Int(elapsed * 1000)), bundleID=\(bundleID), expectedBundleID=\(expectedBundleID ?? "nil")"
                )
            }
        }

        guard let frontmostApplication = NSWorkspace.shared.frontmostApplication else {
            return nil
        }
        if let expectedBundleID,
           let bundleIdentifier = frontmostApplication.bundleIdentifier,
           bundleIdentifier != expectedBundleID {
            return nil
        }

        let bundleIdentifier = frontmostApplication.bundleIdentifier
        let processIdentifier = frontmostApplication.processIdentifier
        if let focusedElement = focusedAXElement(
            preferredProcessID: processIdentifier,
            logDiagnostics: false
        ) {
            let writableElement = writableTextInputElement(from: focusedElement) ?? (
                isWritableTextInputElement(focusedElement) ? focusedElement : nil
            )
            if let writableElement,
               let snippetSnapshot = focusedInputSnippetSnapshot(
                    for: writableElement,
                    bundleIdentifier: bundleIdentifier,
                    processIdentifier: processIdentifier
               ) {
                return snippetSnapshot
            }
        }

        if let snapshot = currentFocusedInputTextSnapshot(
            expectedBundleID: expectedBundleID,
            logDiagnostics: false
        ) {
            return snapshot
        }

        if let cdpSnapshot = await electronCDPFocusedInputTextSnapshot(
            bundleIdentifier: frontmostApplication.bundleIdentifier,
            processIdentifier: frontmostApplication.processIdentifier
        ) {
            return cdpSnapshot
        }

        return nil
    }

    private func focusedInputSnippetSnapshot(
        for writableElement: AXUIElement,
        bundleIdentifier: String?,
        processIdentifier: pid_t
    ) -> FocusedInputTextSnapshot? {
        guard axParameterizedAttributeNames(for: writableElement)
            .contains(kAXStringForRangeParameterizedAttribute as String) else {
            return nil
        }

        let role = axStringAttribute(kAXRoleAttribute as CFString, for: writableElement)
        let isFocusedTarget = axBoolAttribute(kAXFocusedAttribute as CFString, for: writableElement) == true
        let selectedRange = axRangeAttribute(kAXSelectedTextRangeAttribute as CFString, for: writableElement)
        let sourceRange: CFRange?
        if let selectedRange,
           selectedRange.location >= 0,
           let numberOfCharacters = axIntAttribute("AXNumberOfCharacters" as CFString, for: writableElement),
           numberOfCharacters > 0 {
            sourceRange = automaticDictionaryLearningSnippetRange(
                selectedRange: selectedRange,
                numberOfCharacters: numberOfCharacters
            )
        } else if let visibleRange = axRangeAttribute(kAXVisibleCharacterRangeAttribute as CFString, for: writableElement),
                  visibleRange.location >= 0,
                  visibleRange.length > 0 {
            sourceRange = visibleRange
        } else {
            sourceRange = nil
        }

        guard let sourceRange,
              sourceRange.length > 0,
              let text = axParameterizedString(
                kAXStringForRangeParameterizedAttribute as CFString,
                range: sourceRange,
                for: writableElement
              ),
              !text.isEmpty else {
            return nil
        }

        return FocusedInputTextSnapshot(
            text: text,
            bundleIdentifier: bundleIdentifier,
            processIdentifier: processIdentifier,
            role: role,
            isEditable: isWritableTextInputElement(writableElement),
            isFocusedTarget: isFocusedTarget,
            selectedRange: selectedNSRange(from: selectedRange),
            failureReason: nil,
            textSource: "ax-focused-range"
        )
    }

    private func automaticDictionaryLearningSnippetRange(
        selectedRange: CFRange,
        numberOfCharacters: Int
    ) -> CFRange {
        let selectedLocation = min(max(selectedRange.location, 0), numberOfCharacters)
        let selectedLength = max(selectedRange.length, 0)
        let before = min(Self.automaticDictionaryLearningSnippetBeforeCursor, selectedLocation)
        let start = max(selectedLocation - before, 0)
        let requestedLength = before + selectedLength + Self.automaticDictionaryLearningSnippetAfterCursor
        let availableLength = max(numberOfCharacters - start, 0)
        return CFRange(location: start, length: min(requestedLength, availableLength))
    }

    private func focusedAXElement(
        preferredProcessID: pid_t,
        logDiagnostics: Bool = true
    ) -> AXUIElement? {
        guard AccessibilityPermissionManager.isTrusted() else {
            if logDiagnostics {
                VoxtLog.input("Focused input check: accessibility not trusted.")
            }
            return nil
        }

        if let appFocusedElement = focusedAXElement(for: preferredProcessID, logDiagnostics: logDiagnostics) {
            return appFocusedElement
        }

        let bundleID = NSWorkspace.shared.frontmostApplication?.bundleIdentifier ?? "unknown"
        if let systemFocusedElement = systemFocusedAXElement(logDiagnostics: logDiagnostics) {
            if logDiagnostics {
                VoxtLog.input("Focused input check: falling back to system-wide focused element. bundleID=\(bundleID)")
            }
            return systemFocusedElement
        }

        if logDiagnostics {
            VoxtLog.input("Focused input check: app/system focus resolution failed. bundleID=\(bundleID)")
        }
        return nil
    }

    private func systemFocusedAXElement(logDiagnostics: Bool = true) -> AXUIElement? {
        let systemWide = AXUIElementCreateSystemWide()
        AXUIElementSetMessagingTimeout(systemWide, Self.axMessagingTimeout)
        var focusedElementRef: CFTypeRef?
        let focusedStatus = AXUIElementCopyAttributeValue(
            systemWide,
            kAXFocusedUIElementAttribute as CFString,
            &focusedElementRef
        )
        guard focusedStatus == .success,
              let focusedElementRef,
              CFGetTypeID(focusedElementRef) == AXUIElementGetTypeID() else {
            let bundleID = NSWorkspace.shared.frontmostApplication?.bundleIdentifier ?? "unknown"
            if logDiagnostics {
                VoxtLog.input(
                    "Focused input check: system-wide focused element unavailable. status=\(focusedStatus.rawValue), bundleID=\(bundleID)"
                )
            }
            return nil
        }
        let focusedElement = unsafeBitCast(focusedElementRef, to: AXUIElement.self)
        return resolveFocusedElement(focusedElement)
    }

    private func focusedAXElement(
        for processID: pid_t,
        logDiagnostics: Bool = true
    ) -> AXUIElement? {
        let bundleID = NSWorkspace.shared.frontmostApplication?.bundleIdentifier ?? "unknown"
        let appElement = AXUIElementCreateApplication(processID)
        AXUIElementSetMessagingTimeout(appElement, Self.axMessagingTimeout)
        if let focusedAppElement = axElementAttribute(kAXFocusedUIElementAttribute as CFString, for: appElement),
           let resolved = resolveFocusedElement(focusedAppElement) {
            if logDiagnostics {
                VoxtLog.input(
                    "Focused input check: using frontmost app focused element. bundleID=\(bundleID), role=\(axStringAttribute(kAXRoleAttribute as CFString, for: resolved) ?? "unknown")"
                )
            }
            return resolved
        }

        guard let focusedWindow = axElementAttribute(kAXFocusedWindowAttribute as CFString, for: appElement) else {
            if logDiagnostics {
                VoxtLog.input("Focused input check: no focused window on frontmost app. bundleID=\(bundleID)")
            }
            return nil
        }

        if let focusedWindowElement = axElementAttribute(kAXFocusedUIElementAttribute as CFString, for: focusedWindow),
           let resolved = resolveFocusedElement(focusedWindowElement) {
            if logDiagnostics {
                VoxtLog.input(
                    "Focused input check: using focused window focused element. bundleID=\(bundleID), role=\(axStringAttribute(kAXRoleAttribute as CFString, for: resolved) ?? "unknown")"
                )
            }
            return resolved
        }

        if let focusedDescendant = findFocusedDescendant(in: focusedWindow, depthRemaining: 8),
           let resolved = resolveFocusedElement(focusedDescendant) {
            if logDiagnostics {
                VoxtLog.input(
                    "Focused input check: resolved focused descendant from window subtree. bundleID=\(bundleID), role=\(axStringAttribute(kAXRoleAttribute as CFString, for: resolved) ?? "unknown")"
                )
            }
            return resolved
        }

        if let bestEditableDescendant = findBestWritableTextDescendant(in: focusedWindow, depthRemaining: 8) {
            if logDiagnostics {
                VoxtLog.input(
                    "Focused input check: using best editable descendant from focused window. bundleID=\(bundleID), role=\(axStringAttribute(kAXRoleAttribute as CFString, for: bestEditableDescendant) ?? "unknown")"
                )
            }
            return bestEditableDescendant
        }

        if logDiagnostics {
            VoxtLog.input("Focused input check: falling back to focused window element. bundleID=\(bundleID)")
        }
        return resolveFocusedElement(focusedWindow)
    }

    private func axBoolAttribute(_ attribute: CFString, for element: AXUIElement) -> Bool? {
        AXUIElementSetMessagingTimeout(element, Self.axMessagingTimeout)
        var valueRef: CFTypeRef?
        let status = AXUIElementCopyAttributeValue(element, attribute, &valueRef)
        guard status == .success, let valueRef else { return nil }
        if let boolValue = valueRef as? Bool {
            return boolValue
        }
        if let numberValue = valueRef as? NSNumber {
            return numberValue.boolValue
        }
        return nil
    }

    func axStringAttribute(_ attribute: CFString, for element: AXUIElement) -> String? {
        axStringAttribute(attribute, for: element, timeout: Self.axMessagingTimeout)
    }

    func axStringAttribute(
        _ attribute: CFString,
        for element: AXUIElement,
        timeout: Float
    ) -> String? {
        AXUIElementSetMessagingTimeout(element, timeout)
        var valueRef: CFTypeRef?
        let status = AXUIElementCopyAttributeValue(element, attribute, &valueRef)
        guard status == .success else { return nil }
        return valueRef as? String
    }

    private func isAttributeSettable(_ attribute: CFString, on element: AXUIElement) -> Bool {
        var settable = DarwinBoolean(false)
        let status = AXUIElementIsAttributeSettable(element, attribute, &settable)
        return status == .success && settable.boolValue
    }

    private func axIntAttribute(_ attribute: CFString, for element: AXUIElement) -> Int? {
        AXUIElementSetMessagingTimeout(element, Self.axMessagingTimeout)
        var valueRef: CFTypeRef?
        let status = AXUIElementCopyAttributeValue(element, attribute, &valueRef)
        guard status == .success, let valueRef else { return nil }
        if let number = valueRef as? NSNumber {
            return number.intValue
        }
        return nil
    }

    func axRangeAttribute(_ attribute: CFString, for element: AXUIElement) -> CFRange? {
        axRangeAttribute(attribute, for: element, timeout: Self.axMessagingTimeout)
    }

    func axRangeAttribute(
        _ attribute: CFString,
        for element: AXUIElement,
        timeout: Float
    ) -> CFRange? {
        AXUIElementSetMessagingTimeout(element, timeout)
        var valueRef: CFTypeRef?
        let status = AXUIElementCopyAttributeValue(element, attribute, &valueRef)
        guard status == .success,
              let valueRef,
              CFGetTypeID(valueRef) == AXValueGetTypeID() else {
            return nil
        }
        let axValue = unsafeBitCast(valueRef, to: AXValue.self)
        guard AXValueGetType(axValue) == .cfRange else { return nil }
        var range = CFRange()
        guard AXValueGetValue(axValue, .cfRange, &range) else { return nil }
        return range
    }

    private func hasSelectedTextRange(_ element: AXUIElement) -> Bool {
        axRangeAttribute(kAXSelectedTextRangeAttribute as CFString, for: element) != nil
    }

    func axParameterizedString(
        _ attribute: CFString,
        range: CFRange,
        for element: AXUIElement
    ) -> String? {
        axParameterizedString(attribute, range: range, for: element, timeout: Self.axMessagingTimeout)
    }

    func axParameterizedString(
        _ attribute: CFString,
        range: CFRange,
        for element: AXUIElement,
        timeout: Float
    ) -> String? {
        var mutableRange = range
        guard let rangeValue = AXValueCreate(.cfRange, &mutableRange) else {
            return nil
        }
        AXUIElementSetMessagingTimeout(element, timeout)
        var valueRef: CFTypeRef?
        let status = AXUIElementCopyParameterizedAttributeValue(
            element,
            attribute,
            rangeValue,
            &valueRef
        )
        guard status == .success, let valueRef else { return nil }
        if let text = valueRef as? String {
            return normalizedAXTextValue(text, for: element)
        }
        if let text = valueRef as? NSAttributedString {
            return normalizedAXTextValue(text.string, for: element)
        }
        return nil
    }

    private func axParameterizedAttributeNames(for element: AXUIElement) -> [String] {
        AXUIElementSetMessagingTimeout(element, Self.axMessagingTimeout)
        var valueRef: CFArray?
        let status = AXUIElementCopyParameterizedAttributeNames(element, &valueRef)
        guard status == .success, let names = valueRef as? [String] else { return [] }
        return names
    }

    private func axTextValue(for element: AXUIElement) -> String? {
        AXUIElementSetMessagingTimeout(element, Self.axMessagingTimeout)
        var valueRef: CFTypeRef?
        _ = AXUIElementCopyAttributeValue(
            element,
            kAXValueAttribute as CFString,
            &valueRef
        )

        if let stringValue = valueRef as? String,
           let normalizedStringValue = normalizedAXTextValue(stringValue, for: element) {
            return normalizedStringValue
        }
        if let attributedValue = valueRef as? NSAttributedString,
           let normalizedAttributedValue = normalizedAXTextValue(attributedValue.string, for: element) {
            return normalizedAttributedValue
        }

        if let selectedText = axStringAttribute(kAXSelectedTextAttribute as CFString, for: element),
           let normalizedSelectedText = normalizedAXTextValue(selectedText, for: element) {
            VoxtLog.input(
                "Focused input snapshot: resolved text via selected text attribute. role=\(axStringAttribute(kAXRoleAttribute as CFString, for: element) ?? "unknown"), length=\(normalizedSelectedText.count)"
            )
            return normalizedSelectedText
        }

        if let visibleRange = axRangeAttribute(kAXVisibleCharacterRangeAttribute as CFString, for: element),
           visibleRange.length > 0,
           let visibleText = axParameterizedString(
               kAXStringForRangeParameterizedAttribute as CFString,
               range: visibleRange,
               for: element
           ) {
            VoxtLog.input(
                "Focused input snapshot: resolved text via visible character range. role=\(axStringAttribute(kAXRoleAttribute as CFString, for: element) ?? "unknown"), length=\(visibleText.count)"
            )
            return visibleText
        }

        if let numberOfCharacters = axIntAttribute("AXNumberOfCharacters" as CFString, for: element),
           numberOfCharacters > 0 {
            let fullRange = CFRange(location: 0, length: numberOfCharacters)
            if let fullText = axParameterizedString(
                kAXStringForRangeParameterizedAttribute as CFString,
                range: fullRange,
                for: element
            ) {
                VoxtLog.input(
                    "Focused input snapshot: resolved text via full character range. role=\(axStringAttribute(kAXRoleAttribute as CFString, for: element) ?? "unknown"), length=\(fullText.count)"
                )
                return fullText
            }
        }
        return nil
    }

    func axElementAttribute(_ attribute: CFString, for element: AXUIElement) -> AXUIElement? {
        axElementAttribute(attribute, for: element, timeout: Self.axMessagingTimeout)
    }

    func axElementAttribute(
        _ attribute: CFString,
        for element: AXUIElement,
        timeout: Float
    ) -> AXUIElement? {
        AXUIElementSetMessagingTimeout(element, timeout)
        var valueRef: CFTypeRef?
        let status = AXUIElementCopyAttributeValue(element, attribute, &valueRef)
        guard status == .success,
              let valueRef,
              CFGetTypeID(valueRef) == AXUIElementGetTypeID()
        else {
            return nil
        }
        return unsafeBitCast(valueRef, to: AXUIElement.self)
    }

    func axElementArrayAttribute(_ attribute: CFString, for element: AXUIElement) -> [AXUIElement] {
        axElementArrayAttribute(attribute, for: element, timeout: Self.axMessagingTimeout)
    }

    func axElementArrayAttribute(
        _ attribute: CFString,
        for element: AXUIElement,
        timeout: Float
    ) -> [AXUIElement] {
        AXUIElementSetMessagingTimeout(element, timeout)
        var valueRef: CFTypeRef?
        let status = AXUIElementCopyAttributeValue(element, attribute, &valueRef)
        guard status == .success,
              let valueRef,
              let array = valueRef as? [Any]
        else {
            return []
        }

        return array.compactMap { item in
            let cfItem = item as AnyObject
            guard CFGetTypeID(cfItem) == AXUIElementGetTypeID() else { return nil }
            return unsafeBitCast(cfItem, to: AXUIElement.self)
        }
    }

    private func writableTextInputElement(from element: AXUIElement, depth: Int = 0) -> AXUIElement? {
        guard depth <= 8 else { return nil }

        if isWritableTextInputElement(element) {
            return element
        }

        if let focusedChild = axElementAttribute(kAXFocusedUIElementAttribute as CFString, for: element),
           let writableFocusedChild = writableTextInputElement(from: focusedChild, depth: depth + 1) {
            return writableFocusedChild
        }

        for child in axElementArrayAttribute(kAXChildrenAttribute as CFString, for: element) {
            if let writableChild = writableTextInputElement(from: child, depth: depth + 1) {
                return writableChild
            }
        }

        return nil
    }

    private func isWritableTextInputElement(_ element: AXUIElement) -> Bool {
        let role = axStringAttribute(kAXRoleAttribute as CFString, for: element)

        if let role, Self.nonEditableFalsePositiveRoles.contains(role) {
            return false
        }
        if axBoolAttribute("AXEditable" as CFString, for: element) == true {
            return true
        }
        if Self.nativeWritableTextRoles.contains(role ?? "") {
            return true
        }
        let hasSettableTextAttributes =
            isAttributeSettable(kAXSelectedTextRangeAttribute as CFString, on: element)
            || isAttributeSettable(kAXSelectedTextAttribute as CFString, on: element)
            || isAttributeSettable(kAXValueAttribute as CFString, on: element)
        let hasSelectedRange = hasSelectedTextRange(element)

        if Self.genericEditableTextRoles.contains(role ?? "") {
            return hasSelectedRange || hasSettableTextAttributes
        }

        return hasSelectedRange && hasSettableTextAttributes
    }

    private func resolveFocusedElement(_ element: AXUIElement, depthRemaining: Int = 8) -> AXUIElement? {
        guard depthRemaining >= 0 else { return nil }

        if isWritableTextInputElement(element) {
            return element
        }

        if let nestedFocused = axElementAttribute(kAXFocusedUIElementAttribute as CFString, for: element),
           let resolvedNestedFocused = resolveFocusedElement(nestedFocused, depthRemaining: depthRemaining - 1) {
            return resolvedNestedFocused
        }

        if let focusedDescendant = findFocusedDescendant(in: element, depthRemaining: depthRemaining - 1),
           let resolvedDescendant = resolveFocusedElement(focusedDescendant, depthRemaining: depthRemaining - 1) {
            return resolvedDescendant
        }

        if let bestEditableDescendant = findBestWritableTextDescendant(in: element, depthRemaining: depthRemaining - 1) {
            return bestEditableDescendant
        }

        return element
    }

    func findFocusedDescendant(in element: AXUIElement, depthRemaining: Int) -> AXUIElement? {
        guard depthRemaining >= 0 else { return nil }

        if axBoolAttribute(kAXFocusedAttribute as CFString, for: element) == true {
            return element
        }

        for child in axElementArrayAttribute(kAXChildrenAttribute as CFString, for: element) {
            if let nested = findFocusedDescendant(in: child, depthRemaining: depthRemaining - 1) {
                return nested
            }
        }

        return nil
    }

    private func findBestWritableTextDescendant(in element: AXUIElement, depthRemaining: Int) -> AXUIElement? {
        guard depthRemaining >= 0 else { return nil }

        let children = axElementArrayAttribute(kAXChildrenAttribute as CFString, for: element)
        var bestElement: AXUIElement?
        var bestScore = 0

        for child in children {
            let score = writableCandidateScore(for: child)
            if score > bestScore {
                bestScore = score
                bestElement = child
            }

            if let nested = findBestWritableTextDescendant(in: child, depthRemaining: depthRemaining - 1) {
                let nestedScore = writableCandidateScore(for: nested)
                if nestedScore > bestScore {
                    bestScore = nestedScore
                    bestElement = nested
                }
            }
        }

        return bestElement
    }

    private func writableCandidateScore(for element: AXUIElement) -> Int {
        var score = 0
        let role = axStringAttribute(kAXRoleAttribute as CFString, for: element)
        if axBoolAttribute("AXEditable" as CFString, for: element) == true {
            score += 6
        }
        if axBoolAttribute(kAXFocusedAttribute as CFString, for: element) == true {
            score += 4
        }
        if hasSelectedTextRange(element) {
            score += 3
        }
        if isAttributeSettable(kAXSelectedTextRangeAttribute as CFString, on: element) {
            score += 3
        }
        if isAttributeSettable(kAXSelectedTextAttribute as CFString, on: element) {
            score += 3
        }
        if isAttributeSettable(kAXValueAttribute as CFString, on: element) {
            score += 2
        }
        if Self.nativeWritableTextRoles.contains(role ?? "") {
            score += 5
        } else if Self.genericEditableTextRoles.contains(role ?? "") {
            score += 2
        }
        if Self.nonEditableFalsePositiveRoles.contains(role ?? "") {
            score = 0
        }
        return score
    }

    private func normalizedAXTextValue(_ rawValue: String, for element: AXUIElement) -> String? {
        let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        if let placeholder = axStringAttribute(kAXPlaceholderValueAttribute as CFString, for: element),
           placeholder.trimmingCharacters(in: .whitespacesAndNewlines) == trimmed {
            return nil
        }
        if let title = axStringAttribute(kAXTitleAttribute as CFString, for: element),
           title.trimmingCharacters(in: .whitespacesAndNewlines) == trimmed {
            return nil
        }
        return trimmed
    }

    private func unreadableTextFailureReason(for element: AXUIElement) -> String {
        let selectedRange = axRangeAttribute(kAXSelectedTextRangeAttribute as CFString, for: element)
        let visibleRange = axRangeAttribute(kAXVisibleCharacterRangeAttribute as CFString, for: element)
        let numberOfCharacters = axIntAttribute("AXNumberOfCharacters" as CFString, for: element) ?? -1
        let hasStringForRange = axParameterizedAttributeNames(for: element)
            .contains(kAXStringForRangeParameterizedAttribute as String)
        let diagnostics =
            "numChars=\(numberOfCharacters), selectedRange=\(selectedRange.map { "{\($0.location),\($0.length)}" } ?? "nil"), visibleRange=\(visibleRange.map { "{\($0.location),\($0.length)}" } ?? "nil"), hasStringForRange=\(hasStringForRange)"
        if let placeholder = axStringAttribute(kAXPlaceholderValueAttribute as CFString, for: element),
           let value = axStringAttribute(kAXValueAttribute as CFString, for: element),
           placeholder.trimmingCharacters(in: .whitespacesAndNewlines)
                == value.trimmingCharacters(in: .whitespacesAndNewlines) {
            return "value-matched-placeholder, \(diagnostics)"
        }
        if let title = axStringAttribute(kAXTitleAttribute as CFString, for: element),
           let value = axStringAttribute(kAXValueAttribute as CFString, for: element),
           title.trimmingCharacters(in: .whitespacesAndNewlines)
                == value.trimmingCharacters(in: .whitespacesAndNewlines) {
            return "value-matched-title, \(diagnostics)"
        }
        if !isAttributeSettable(kAXValueAttribute as CFString, on: element) {
            return "ax-value-not-settable, \(diagnostics)"
        }
        return "missing-ax-value, \(diagnostics)"
    }

    private func selectedNSRange(from range: CFRange?) -> NSRange? {
        guard let range, range.location >= 0, range.length >= 0 else { return nil }
        return NSRange(location: range.location, length: range.length)
    }

    private func focusedInputSnapshotSummary(_ snapshot: FocusedInputTextSnapshot) -> String {
        let preview = String(snapshot.text.prefix(80))
        let selectedRangeDescription = snapshot.selectedRange.map {
            "{\($0.location),\($0.length)}"
        } ?? "nil"
        return
            "bundleID=\(snapshot.bundleIdentifier ?? "nil") pid=\(snapshot.processIdentifier.map(String.init) ?? "nil") "
                + "role=\(snapshot.role ?? "nil") editable=\(snapshot.isEditable) "
                + "focused=\(snapshot.isFocusedTarget) selectedRange=\(selectedRangeDescription) textLength=\(snapshot.text.count) "
                + "textSource=\(snapshot.textSource ?? "nil") preview=\(preview)"
    }
}
