#if canImport(DictationCore)
import DictationCore
#endif
import AppKit
import ApplicationServices

/// One verified dictation span in a plain AX text field. Never searches another field.
@MainActor
final class EditableTextAnchor: TextRevisionTarget {
    private var location: RevisionLocation
    private let before: TextDraft
    private let read: () throws -> TextDraft
    private let validateIdentity: () throws -> Void
    private let select: (NSRange) throws -> Void
    private let makePaste: () -> ClipboardPaste
    private let deleteSelection: () throws -> Void
    private var prepared: TextDraft?
    private var invalidated = false
    private var generation = UUID()
    private var pendingPaste: ClipboardPaste?
    private var beginObservation: ((EditableTextAnchor) -> Void)?
    private var observer: AXObserver?
    private var notifications: [(AXUIElement, String)] = []
    private var activationObserver: NSObjectProtocol?
    private(set) var confirmationFailure: String?

    static func capture(text: String) throws -> EditableTextAnchor {
        guard !text.isEmpty, AXIsProcessTrusted(), CGPreflightPostEventAccess(),
              let front = NSWorkspace.shared.frontmostApplication,
              front.processIdentifier != ProcessInfo.processInfo.processIdentifier else {
            throw DictationError("请回到原输入框再更正，或复制更正结果。")
        }
        let pid = front.processIdentifier
        let app = AXUIElementCreateApplication(pid)
        AXUIElementSetMessagingTimeout(app, 0.2)
        guard let element = AccessibilityTextTarget.axElement(app, kAXFocusedUIElementAttribute),
              let window = AccessibilityTextTarget.axElement(app, kAXFocusedWindowAttribute) else {
            throw DictationError("当前应用没有提供可定位的输入框。")
        }
        let read = { () throws -> TextDraft in
            guard let draft = try AccessibilityTextTarget.readInput(element).draft(allowCompatibilityPaste: false) else {
                throw DictationError("无法读取原输入框。")
            }
            return draft
        }
        let before = try read()
        let identity = {
            guard !front.isTerminated, NSWorkspace.shared.frontmostApplication?.processIdentifier == pid,
                  let currentWindow = AccessibilityTextTarget.axElement(app, kAXFocusedWindowAttribute),
                  CFEqual(currentWindow, window),
                  let focus = AccessibilityTextTarget.axElement(app, kAXFocusedUIElementAttribute), CFEqual(focus, element),
                  AXIsProcessTrusted(), CGPreflightPostEventAccess() else {
                throw DictationError("原输入框、窗口或权限已变化。")
            }
        }
        let anchor = EditableTextAnchor(text: text, before: before, read: read, validateIdentity: identity,
                                       select: { range in
            var settable = DarwinBoolean(false)
            guard AXUIElementIsAttributeSettable(element, kAXSelectedTextRangeAttribute as CFString, &settable) == .success,
                  settable.boolValue else { throw DictationError("原输入框不支持定位选区。") }
            var range = CFRange(location: range.location, length: range.length)
            guard let value = AXValueCreate(.cfRange, &range),
                  AXUIElementSetAttributeValue(element, kAXSelectedTextRangeAttribute as CFString, value) == .success else {
                throw DictationError("无法选中需要更正的片段。")
            }
        }, makePaste: {
            ClipboardPaste(pasteboard: .general, post: { _, events in
                for event in events { event.postToPid(pid) }
            })
        }, deleteSelection: {
            // An empty paste is not a reliable deletion contract across apps.
            // Delete only the verified selection through an explicitly supported AX setter.
            var settable = DarwinBoolean(false)
            guard AXUIElementIsAttributeSettable(element, kAXSelectedTextAttribute as CFString, &settable) == .success,
                  settable.boolValue,
                  AXUIElementSetAttributeValue(element, kAXSelectedTextAttribute as CFString, "" as CFString) == .success else {
                throw DictationError("原输入框不支持精确删除；请手动应用更正结果。")
            }
        })
        anchor.beginObservation = { $0.observe(app: app, element: element, pid: pid) }
        try identity()
        return anchor
    }

    init(text: String, before: TextDraft, read: @escaping () throws -> TextDraft,
         validateIdentity: @escaping () throws -> Void, select: @escaping (NSRange) throws -> Void,
         makePaste: @escaping () -> ClipboardPaste, deleteSelection: @escaping () throws -> Void) {
        location = RevisionLocation(text: text, start: before.selection.location)
        self.before = before
        self.read = read
        self.validateIdentity = validateIdentity
        self.select = select
        self.makePaste = makePaste
        self.deleteSelection = deleteSelection
    }

    func confirmInsertion() async -> Bool {
        let id = generation
        confirmationFailure = nil
        guard let expected = try? before.inserting(location.text) else { return false }
        for _ in 0..<40 {
            guard id == generation, !Task.isCancelled else { return false }
            do {
                try validateIdentity()
                let current = try read()
                // The captured insertion offset plus exact text establishes the span;
                // a delayed or moved caret does not change that span. Some editors
                // expose an extra final LF, or remove an empty-editor LF placeholder.
                // Only tolerate that final LF outside the exact inserted span.
                let sameText = current.value.utf16.elementsEqual(expected.utf16)
                let terminalLF = current.value.utf16.elementsEqual((expected + "\n").utf16)
                    || expected.utf16.elementsEqual((current.value + "\n").utf16)
                if sameText || terminalLF {
                    try location.validate(in: current.value)
                    return true
                }
                // Some editors expose a placeholder or stale selection before paste.
                // If the changed field consists entirely of this paragraph, its range
                // is unambiguous even when the pre-paste prediction was incorrect.
                // Do not search for a matching paragraph inside unrelated text.
                let wholeField = current.value.utf16.elementsEqual(location.text.utf16)
                    || current.value.utf16.elementsEqual((location.text + "\n").utf16)
                if wholeField, !current.value.utf16.elementsEqual(before.value.utf16) {
                    location = RevisionLocation(text: location.text, start: 0)
                    try location.validate(in: current.value)
                    return true
                }
                guard current.value.utf16.elementsEqual(before.value.utf16) else {
                    confirmationFailure = "输入框读回内容与粘贴预期不一致（预期 \(expected.utf16.count)，实际 \(current.value.utf16.count) 个 UTF-16 单元）。"
                    return false
                }
                try await Task.sleep(nanoseconds: 50_000_000)
            } catch {
                confirmationFailure = error.localizedDescription
                return false
            }
        }
        confirmationFailure = "未在原输入框读回刚才粘贴的文字。"
        return false
    }

    func prepare(original: String) throws {
        cancel()
        try validateIdentity()
        let current = try read()
        try location.validate(in: current.value)
        guard location.text.utf16.elementsEqual(original.utf16) else { throw DictationError("上一段版本不一致。") }
        prepared = current
        invalidated = false
        beginObservation?(self)
        try validatePrepared()
    }

    func apply(_ revision: TextRevision) async throws -> Bool {
        try Task.checkCancellation()
        try validatePrepared()
        guard let prepared else { throw DictationError("更正已经取消。") }
        let id = generation
        let range = try location.selection(for: revision, in: prepared.value)
        if revision.isEmpty { return true }
        let expected = try TextDraft(value: prepared.value, selection: range).inserting(revision.replacement)
        try select(range)
        // An AX setter can succeed before the editor publishes its updated selection.
        // Keep observing focus and text while awaiting that acknowledgement.
        try await waitForSelection(range, generation: id)
        stopObserving()
        let validateSelection = {
            try self.validatePrepared()
            guard try self.read().selection == range else { throw DictationError("修改选区已经变化。") }
        }
        try validateSelection()
        var confirmed = false
        defer {
            pendingPaste?.finishRevision(confirmed: confirmed, text: revision.corrected)
            pendingPaste = nil
        }
        if revision.replacement.isEmpty {
            try deleteSelection()
        } else {
            let paste = makePaste()
            pendingPaste = paste
            try paste.perform(text: revision.replacement, validate: validateSelection)
        }
        for _ in 0..<40 {
            guard id == generation, !Task.isCancelled else { return false }
            try validateIdentity()
            let current = try read()
            if current.value.utf16.elementsEqual(expected.utf16) {
                location.accept(revision)
                self.prepared = nil
                confirmed = true
                return true
            }
            // Never retry an ambiguous write or overwrite user input made after dispatch.
            guard current.value.utf16.elementsEqual(prepared.value.utf16) else { return false }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        return false
    }

    private func waitForSelection(_ range: NSRange, generation id: UUID) async throws {
        var lastSelection: NSRange?
        for _ in 0..<20 {
            try Task.checkCancellation()
            guard id == generation else { throw DictationError("更正已经取消。") }
            try validatePrepared()
            lastSelection = try read().selection
            if lastSelection == range { return }
            try await Task.sleep(nanoseconds: 25_000_000)
        }
        let actual = lastSelection.map { "\($0.location),\($0.length)" } ?? "无法读取"
        throw DictationError("输入框未确认更正选区（预期 \(range.location),\(range.length)，实际 \(actual)；位置和长度按 UTF-16 计算）。尚未粘贴。")
    }

    func cancel() {
        generation = UUID()
        stopObserving()
        prepared = nil
        pendingPaste?.finishRevision(confirmed: false, text: "")
        pendingPaste = nil
    }

    private func validatePrepared() throws {
        guard let prepared, !invalidated else { throw DictationError("更正期间输入框已变化或操作已取消。") }
        try validateIdentity()
        guard try read().value.utf16.elementsEqual(prepared.value.utf16) else {
            throw DictationError("更正期间你修改了输入内容。")
        }
    }

    private func observedChange() {
        do { try validatePrepared() }
        catch { invalidated = true }
    }

    private func observe(app: AXUIElement, element: AXUIElement, pid: pid_t) {
        var observer: AXObserver?
        if AXObserverCreate(pid, { _, _, _, context in
            guard let context else { return }
            MainActor.assumeIsolated {
                Unmanaged<EditableTextAnchor>.fromOpaque(context).takeUnretainedValue().observedChange()
            }
        }, &observer) == .success, let observer {
            self.observer = observer
            for pair in [(app, kAXFocusedUIElementChangedNotification), (app, kAXFocusedWindowChangedNotification),
                         (element, kAXValueChangedNotification)] {
                if AXObserverAddNotification(observer, pair.0, pair.1 as CFString,
                                             Unmanaged.passUnretained(self).toOpaque()) == .success {
                    notifications.append(pair)
                }
            }
            CFRunLoopAddSource(CFRunLoopGetMain(), AXObserverGetRunLoopSource(observer), .commonModes)
        }
        activationObserver = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification, object: nil, queue: .main
        ) { [weak self] note in
            MainActor.assumeIsolated {
                if let app = note.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication,
                   app.processIdentifier != pid { self?.invalidated = true }
            }
        }
    }

    private func stopObserving() {
        if let observer {
            for pair in notifications { AXObserverRemoveNotification(observer, pair.0, pair.1 as CFString) }
            CFRunLoopRemoveSource(CFRunLoopGetMain(), AXObserverGetRunLoopSource(observer), .commonModes)
        }
        notifications = []
        observer = nil
        if let activationObserver { NSWorkspace.shared.notificationCenter.removeObserver(activationObserver) }
        activationObserver = nil
    }
}
