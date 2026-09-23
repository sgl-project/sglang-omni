// SPDX-License-Identifier: Apache-2.0
import AppKit
import ApplicationServices
import Carbon

enum TextInsertionError: LocalizedError {
    case secureField

    var errorDescription: String? {
        L("sys.secureField")
    }
}

struct InsertionTarget {
    let applicationName: String
    let bundleID: String
    let selectedText: String
    let application: NSRunningApplication
    let element: AXUIElement?
    let range: CFRange?
    let value: String?
    let window: AXUIElement?
    let windowTitle: String?
    let document: String?

    func validate(against current: InsertionTarget) throws {
        guard application.processIdentifier == current.application.processIdentifier else {
            throw Failure("sys.destChanged")
        }
        func sameElement(_ lhs: AXUIElement?, _ rhs: AXUIElement?) -> Bool {
            if let lhs, let rhs { return CFEqual(lhs, rhs) }
            return lhs == nil && rhs == nil
        }
        guard sameElement(element, current.element) else { throw Failure("sys.fieldChanged") }
        guard (element != nil || window != nil), sameElement(window, current.window),
              windowTitle == current.windowTitle, document == current.document, value == current.value else {
            throw Failure("sys.contentChanged")
        }
        guard range?.location == current.range?.location, range?.length == current.range?.length,
              selectedText == current.selectedText else {
            throw Failure("sys.cursorChanged")
        }
    }
}

@MainActor
enum TextInsertion {
    static var isTrusted: Bool { AXIsProcessTrusted() }

    private static var activationObserver: NSObjectProtocol?
    private static var enabledProcesses: Set<pid_t> = []

    // Note (Jiaxin Deng): Ask on activation so Chromium can build its accessibility tree before recording.
    static func enableAccessibilityInHostedApps() {
        guard activationObserver == nil else { return }
        let center = NSWorkspace.shared.notificationCenter
        activationObserver = center.addObserver(forName: NSWorkspace.didActivateApplicationNotification,
                                                object: nil, queue: .main) { note in
            guard let app = note.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication else { return }
            MainActor.assumeIsolated { requestManualAccessibility(app.processIdentifier) }
        }
        if let app = NSWorkspace.shared.frontmostApplication { requestManualAccessibility(app.processIdentifier) }
    }

    private static func requestManualAccessibility(_ pid: pid_t) {
        guard isTrusted, pid != ProcessInfo.processInfo.processIdentifier,
              enabledProcesses.insert(pid).inserted else { return }
        let application = AXUIElementCreateApplication(pid)
        AXUIElementSetMessagingTimeout(application, 1)
        AXUIElementSetAttributeValue(application, "AXManualAccessibility" as CFString, kCFBooleanTrue)
    }

    static func requestPermission() {
        AXIsProcessTrustedWithOptions([kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary)
    }

    static func capture() throws -> InsertionTarget {
        guard isTrusted else {
            throw Failure("sys.axPermission")
        }
        guard let app = NSWorkspace.shared.frontmostApplication, !app.isTerminated,
              app.processIdentifier != ProcessInfo.processInfo.processIdentifier else {
            throw Failure("sys.focusField")
        }
        guard !IsSecureEventInputEnabled() else { throw TextInsertionError.secureField }
        requestManualAccessibility(app.processIdentifier)
        let element = focusedElement(of: app.processIdentifier)
        var range: CFRange?
        var selection = ""
        if let element {
            var pid: pid_t = 0
            guard AXUIElementGetPid(element, &pid) == .success, pid == app.processIdentifier else {
                throw Failure("sys.appChanged")
            }
            try rejectSecure(element)
            let role = attribute(element, kAXRoleAttribute) as? String ?? ""
            guard [kAXTextFieldRole, kAXTextAreaRole, kAXComboBoxRole].contains(role)
                    || (attribute(element, "AXEditable") as? NSNumber)?.boolValue == true else {
                throw Failure("sys.focusEditable")
            }
            range = try selectionRange(element)
            if let range { selection = try selectedText(element, range: range) }
        }
        let application = AXUIElementCreateApplication(app.processIdentifier)
        AXUIElementSetMessagingTimeout(application, 1)
        let window = element.flatMap { elementAttribute($0, kAXWindowAttribute) }
            ?? elementAttribute(application, kAXFocusedWindowAttribute)
        guard element != nil || window != nil else { throw Failure("sys.fieldOpaque") }
        // Note (Codex): ponytail: opaque editors permit only window checks; use field checks when AX exposes them.
        return InsertionTarget(applicationName: app.localizedName ?? "Application", bundleID: app.bundleIdentifier ?? "",
                               selectedText: selection, application: app, element: element, range: range,
                               value: element.flatMap { attribute($0, kAXValueAttribute) as? String }, window: window,
                               windowTitle: window.flatMap { attribute($0, kAXTitleAttribute) as? String },
                               document: window.flatMap { attribute($0, kAXDocumentAttribute) as? String })
    }

    static func insert(_ text: String, into target: InsertionTarget) async throws {
        guard !text.isEmpty else { return }
        try Task.checkCancellation()
        try validate(target)
        var settable = DarwinBoolean(false)
        // Note (Jiaxin Deng): Chromium can report successful AXSelectedText writes without applying them.
        if let element = target.element, target.range != nil, !isWebHosted(element),
           AXUIElementIsAttributeSettable(element, kAXSelectedTextAttribute as CFString, &settable) == .success,
           settable.boolValue {
            let result = AXUIElementSetAttributeValue(element, kAXSelectedTextAttribute as CFString, text as CFString)
            guard result == .success else {
                throw Failure("sys.replaceFailed")
            }
            Diagnostics.record("insert.ok", ["destination": target.bundleID, "path": "direct"])
            return
        }

        let clipboard = NSPasteboard.general
        let originalCount = clipboard.changeCount
        let snapshot = try (clipboard.pasteboardItems ?? []).map { item -> NSPasteboardItem in
            let copy = NSPasteboardItem()
            for type in item.types {
                guard let data = item.data(forType: type) else {
                    throw Failure("sys.clipboardKeep")
                }
                copy.setData(data, forType: type)
            }
            return copy
        }
        guard clipboard.changeCount == originalCount else {
            throw Failure("sys.clipboardChanged")
        }
        guard let source = CGEventSource(stateID: .hidSystemState),
              let down = CGEvent(keyboardEventSource: source, virtualKey: 9, keyDown: true),
              let up = CGEvent(keyboardEventSource: source, virtualKey: 9, keyDown: false) else {
            throw Failure("sys.pasteEvent")
        }
        try validate(target)
        try Task.checkCancellation()
        clipboard.clearContents()
        let clearedCount = clipboard.changeCount
        guard clipboard.setString(text, forType: .string) else {
            if clipboard.changeCount == clearedCount {
                clipboard.clearContents()
                if !snapshot.isEmpty { clipboard.writeObjects(snapshot) }
            }
            throw Failure("sys.clipboardWrite")
        }
        let ownedCount = clipboard.changeCount
        defer {
            // Note (Codex): Preserve any clipboard copy made while the destination pastes.
            if clipboard.changeCount == ownedCount {
                clipboard.clearContents()
                if !snapshot.isEmpty { clipboard.writeObjects(snapshot) }
            }
        }
        try validate(target)
        let lengthBeforePaste = target.element.flatMap { characterCount($0) }
        down.flags = .maskCommand
        up.flags = .maskCommand
        down.post(tap: .cgAnnotatedSessionEventTap)
        up.post(tap: .cgAnnotatedSessionEventTap)
        // Note (Codex): Cancellation must not restore the clipboard before the queued paste consumes it.
        await Task.detached { try? await Task.sleep(nanoseconds: 800_000_000) }.value
        // Note (Jiaxin Deng): Report ignored pastes without retrying; a delayed paste could otherwise duplicate text.
        if let element = target.element, let range = target.range,
           let before = lengthBeforePaste, let after = characterCount(element),
           pasteWasIgnored(before: before, after: after,
                           inserted: (text as NSString).length, replaced: range.length) {
            Diagnostics.record("insert.ignored", [
                "destination": target.bundleID,
                "before": String(before), "after": String(after),
                "inserted": String((text as NSString).length),
                "replaced": String(range.length),
            ])
            throw Failure("sys.pasteIgnored")
        }
        Diagnostics.record("insert.ok", ["destination": target.bundleID,
                                         "path": target.element == nil ? "paste-window" : "paste"])
    }

    // Note (Jiaxin Deng): Compare character counts in Accessibility's UTF-16 units.
    // Note (Yifei Leng): Trust only AXNumberOfCharacters. Terminals such as cmux expose an AXTextArea whose
    // AXValue stays empty, so comparing value lengths reported every successful paste as ignored.
    private static func characterCount(_ element: AXUIElement) -> Int? {
        (attribute(element, "AXNumberOfCharacters") as? NSNumber)?.intValue
    }

    // Note (Yifei Leng): An unknown count proves nothing, and neither does replacing a selection of equal length.
    nonisolated static func pasteWasIgnored(before: Int?, after: Int?, inserted: Int, replaced: Int) -> Bool {
        guard let before, let after, inserted != replaced else { return false }
        return after == before
    }

    static func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }

    private static func validate(_ target: InsertionTarget) throws {
        guard isTrusted, !target.application.isTerminated,
              NSWorkspace.shared.frontmostApplication?.processIdentifier == target.application.processIdentifier else {
            throw Failure("sys.destChanged")
        }
        try target.validate(against: capture())
    }

    // Note (Jiaxin Deng): Chromium may expose focus only through the application-level query.
    private static func focusedElement(of pid: pid_t) -> AXUIElement? {
        for source in [AXUIElementCreateApplication(pid), AXUIElementCreateSystemWide()] {
            AXUIElementSetMessagingTimeout(source, 1)
            if let value = attribute(source, kAXFocusedUIElementAttribute),
               CFGetTypeID(value) == AXUIElementGetTypeID() {
                let element = value as! AXUIElement
                AXUIElementSetMessagingTimeout(element, 1)
                return element
            }
        }
        return nil
    }

    private static func rejectSecure(_ element: AXUIElement) throws {
        var current: AXUIElement? = element
        for _ in 0..<8 {
            guard let node = current else { break }
            let role = attribute(node, kAXRoleAttribute) as? String
            let subrole = attribute(node, kAXSubroleAttribute) as? String
            if role == "AXSecureTextField" || subrole == kAXSecureTextFieldSubrole
                || (attribute(node, "AXProtectedContent") as? NSNumber)?.boolValue == true {
                throw TextInsertionError.secureField
            }
            guard let parent = attribute(node, kAXParentAttribute), CFGetTypeID(parent) == AXUIElementGetTypeID() else { break }
            current = (parent as! AXUIElement)
        }
    }

    private static func selectionRange(_ element: AXUIElement) throws -> CFRange? {
        guard let value = attribute(element, kAXSelectedTextRangeAttribute) else { return nil }
        guard CFGetTypeID(value) == AXValueGetTypeID() else { throw Failure("sys.badSelection") }
        let axValue = value as! AXValue
        var range = CFRange()
        guard AXValueGetType(axValue) == .cfRange, AXValueGetValue(axValue, .cfRange, &range),
              range.location >= 0, range.length >= 0 else {
            throw Failure("sys.badSelection")
        }
        return range
    }

    private static func selectedText(_ element: AXUIElement, range: CFRange) throws -> String {
        if range.length == 0 { return "" }
        if let text = attribute(element, kAXSelectedTextAttribute) as? String { return text }
        if let value = attribute(element, kAXValueAttribute) as? String,
           range.location <= (value as NSString).length,
           range.length <= (value as NSString).length - range.location {
            return (value as NSString).substring(with: NSRange(location: range.location, length: range.length))
        }
        throw Failure("sys.selectionRead")
    }

    // Note (Jiaxin Deng): Web-hosted nodes expose DOM or Chromium identifiers.
    private static func isWebHosted(_ element: AXUIElement) -> Bool {
        var names: CFArray?
        guard AXUIElementCopyAttributeNames(element, &names) == .success,
              let list = names as? [String] else { return false }
        return list.contains("ChromeAXNodeId") || list.contains("AXDOMIdentifier")
    }

    private static func attribute(_ element: AXUIElement, _ key: String) -> CFTypeRef? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, key as CFString, &value) == .success else { return nil }
        return value
    }

    private static func elementAttribute(_ element: AXUIElement, _ key: String) -> AXUIElement? {
        guard let value = attribute(element, key), CFGetTypeID(value) == AXUIElementGetTypeID() else { return nil }
        return (value as! AXUIElement)
    }
}
