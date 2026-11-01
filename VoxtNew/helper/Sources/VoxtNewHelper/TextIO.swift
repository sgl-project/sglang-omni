import AppKit
import ApplicationServices
import VoxtNewHelperKit
import Carbon
import Foundation

// Text in and out of the app in front: what is focused (can it take
// text?), the selected text (Accessibility first, ⌘C as a fallback that
// puts the clipboard back, when the app allows it), and inserting text
// (the clipboard, ⌘V, keys after it, then the old clipboard back if
// nobody wrote to it since). Go decides the policy (when ⌘C is allowed,
// the delays, the keys); the helper does it, beside the event tap whose
// marker its posted keys carry.

final class TextIO: @unchecked Sendable {
    let front: FrontApps

    init(front: FrontApps) { self.front = front }

    static let textRoles: Set<String> = ["AXTextArea", "AXTextField", "AXComboBox", "AXSearchField"]

    private func requireTrust() throws {
        guard AXIsProcessTrusted() else { throw RPCError.denied(PermissionName.accessibility, "VoxtNew may not use Accessibility") }
    }

    private func focusedElement(pid: pid_t?) -> AXUIElement? {
        let sys = AXUIElementCreateSystemWide()
        AXUIElementSetMessagingTimeout(sys, 0.25)
        var ref: CFTypeRef?
        if AXUIElementCopyAttributeValue(sys, kAXFocusedUIElementAttribute as CFString, &ref) == .success, let ref,
           CFGetTypeID(ref) == AXUIElementGetTypeID() {
            return (ref as! AXUIElement)
        }
        guard let pid, pid > 0 else { return nil }
        let app = AXUIElementCreateApplication(pid)
        AXUIElementSetMessagingTimeout(app, 0.25)
        if AXUIElementCopyAttributeValue(app, kAXFocusedUIElementAttribute as CFString, &ref) == .success, let ref,
           CFGetTypeID(ref) == AXUIElementGetTypeID() {
            return (ref as! AXUIElement)
        }
        return nil
    }

    private func string(_ e: AXUIElement, _ attr: String) -> String? {
        var v: CFTypeRef?
        guard AXUIElementCopyAttributeValue(e, attr as CFString, &v) == .success, let v else { return nil }
        if let s = v as? String { return s }
        if let a = v as? NSAttributedString { return a.string }
        return nil
    }

    private func range(_ e: AXUIElement) -> CFRange? {
        var v: CFTypeRef?
        guard AXUIElementCopyAttributeValue(e, kAXSelectedTextRangeAttribute as CFString, &v) == .success, let v,
              CFGetTypeID(v) == AXValueGetTypeID() else { return nil }
        var r = CFRange()
        return AXValueGetValue(v as! AXValue, .cfRange, &r) ? r : nil
    }

    /// The focused element: its app (the element's own process, which is
    /// not always the app in front), whether it takes text, and whether it
    /// is a password field (its subrole). Go decides where a result goes.
    /// Without Accessibility, permission_denied, as the other textio.*.
    func focus() async throws -> FocusResult {
        try requireTrust()
        let frontApp = await front.current()
        let pid = frontApp.pid
        guard let e = focusedElement(pid: pid_t(pid)) else { return FocusResult(pid: pid, app: pid != 0 ? frontApp : nil, error: "no focused element") }
        var p: pid_t = 0
        AXUIElementGetPid(e, &p)
        var settable = DarwinBoolean(false)
        AXUIElementIsAttributeSettable(e, kAXValueAttribute as CFString, &settable)
        let role = string(e, kAXRoleAttribute)
        let r = range(e)
        let owner = Int(p) == pid ? frontApp : NSRunningApplication(processIdentifier: p).map(FrontApps.describe)
        return FocusResult(pid: Int(p), app: owner, role: role, subrole: string(e, kAXSubroleAttribute),
                           editable: settable.boolValue || Self.textRoles.contains(role ?? "") || inEditableRegion(e),
                           selection: r.map { TextRange(location: $0.location, length: $0.length) },
                           valueLength: string(e, kAXValueAttribute)?.utf16.count ?? 0)
    }

    /// A web page's contenteditable region (WebKit and Chromium name its
    /// root in AXEditableAncestor; a page's plain text has none).
    private func inEditableRegion(_ e: AXUIElement) -> Bool {
        var v: CFTypeRef?
        return AXUIElementCopyAttributeValue(e, "AXEditableAncestor" as CFString, &v) == .success && v != nil
    }

    func selection(_ p: SelectionParams) async throws -> SelectionResult {
        let t0 = monotonicMs()
        try requireTrust()
        var steps: [String] = []
        let pid = await front.current().pid
        func done(_ text: String?, _ via: String) -> SelectionResult {
            SelectionResult(text: text, via: via, ms: round1(monotonicMs() - t0), steps: steps)
        }
        if let e = focusedElement(pid: pid_t(pid)) {
            let role = string(e, kAXRoleAttribute)
            steps.append("focus=\(role ?? "?")")
            if let s = string(e, kAXSelectedTextAttribute), !s.isEmpty {
                return done(s, SelectionVia.ax)
            }
            let r = range(e)
            steps.append("range=\(r.map { "\($0.location),\($0.length)" } ?? "nil")")
            if let r, r.length > 0 {
                var rr = r
                var v: CFTypeRef?
                if let axr = AXValueCreate(.cfRange, &rr),
                   AXUIElementCopyParameterizedAttributeValue(e, kAXStringForRangeParameterizedAttribute as CFString, axr, &v) == .success,
                   let s = v as? String, !s.isEmpty {
                    return done(s, SelectionVia.axRange)
                }
            }
            // WebKit's and Chromium's text markers.
            var marker: CFTypeRef?
            if AXUIElementCopyAttributeValue(e, "AXSelectedTextMarkerRange" as CFString, &marker) == .success, let marker {
                var v: CFTypeRef?
                if AXUIElementCopyParameterizedAttributeValue(e, "AXStringForTextMarkerRange" as CFString, marker, &v) == .success,
                   let s = v as? String, !s.isEmpty {
                    return done(s, SelectionVia.axMarker)
                }
                steps.append("marker=empty")
            }
            if let r, r.length == 0 {
                // A caret and no selection: ⌘C would copy nothing, or a whole line.
                steps.append("caret-only")
                return done(nil, SelectionVia.none)
            }
        } else {
            steps.append("focus=nil")
        }
        guard p.allowCopy ?? false else { return done(nil, SelectionVia.none) }
        let copied = await copySelection(timeoutMs: Double(p.copyTimeoutMs ?? 300))
        steps.append(copied.note)
        return done(copied.text, copied.text != nil ? SelectionVia.copy : SelectionVia.none)
    }

    /// ⌘C, wait for the clipboard to change, read it, put the old one back.
    private func copySelection(timeoutMs: Double) async -> (text: String?, note: String) {
        let (snap, c0) = await onMain { (Clipboard.snapshot(NSPasteboard.general), NSPasteboard.general.changeCount) }
        postKey(UInt16(kVK_ANSI_C), flags: .maskCommand)
        let deadline = monotonicMs() + timeoutMs
        var changed = false
        while monotonicMs() < deadline {
            if await onMain({ NSPasteboard.general.changeCount }) != c0 {
                changed = true
                break
            }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        guard changed else { return (nil, "copy: clipboard unchanged") }
        let text = await onMain { () -> String? in
            let pb = NSPasteboard.general
            let s = pb.string(forType: .string)
            Clipboard.restore(pb, snap)
            return s
        }
        return (text?.isEmpty == false ? text : nil, "copy: changed")
    }

    func paste(_ p: PasteParams) async throws -> PasteResult {
        let t0 = monotonicMs()
        try requireTrust()
        var frontPid = 0
        if let target = p.targetPid, target != 0 {
            let now = await front.current()
            if now.pid != target {
                return PasteResult(posted: false, restored: false, clipboardTaken: false, ms: round1(monotonicMs() - t0), front: now)
            }
            frontPid = now.pid
        }
        let verifyMs = Double(max(0, p.verifyMs ?? 0))
        // The element the text should go into, as it reads before ⌘V.
        var watched: AXUIElement?
        var before = FieldReading(length: nil, selection: nil)
        if verifyMs > 0 {
            if frontPid == 0 { frontPid = await front.current().pid }
            watched = focusedElement(pid: pid_t(frontPid))
            if let e = watched {
                // A hung app must not hold the check up: 0.25 s a read.
                AXUIElementSetMessagingTimeout(e, 0.25)
                before = reading(e)
            }
        }
        let restore = !(p.keepClipboard ?? false)
        let (snap, ours) = await onMain { () -> ([[NSPasteboard.PasteboardType: Data]], Int) in
            let pb = NSPasteboard.general
            let snap = restore ? Clipboard.snapshot(pb) : []
            pb.clearContents()
            let item = NSPasteboardItem()
            item.setString(p.text, forType: .string)
            // Clipboard managers skip these (nspasteboard.org).
            item.setData(Data(), forType: NSPasteboard.PasteboardType("org.nspasteboard.TransientType"))
            item.setData(Data(), forType: NSPasteboard.PasteboardType("org.nspasteboard.AutoGeneratedType"))
            pb.writeObjects([item])
            return (snap, pb.changeCount)
        }
        let secure = secureInputOwner()
        postKey(UInt16(kVK_ANSI_V), flags: .maskCommand)
        let posted = monotonicMs()

        // The check: did the element change?
        var check: String?
        var checkMs: Double?
        var field: PasteField?
        if verifyMs > 0 {
            var last: FieldReading?
            var saw = false
            var moved = false
            if let e = watched, before.readable {
                let deadline = posted + verifyMs
                while true {
                    try? await Task.sleep(nanoseconds: 25_000_000)
                    let r = reading(e)
                    last = r
                    if PasteChecker.compare(before, r) == .changed {
                        saw = true
                        break
                    }
                    if monotonicMs() >= deadline { break }
                }
                if !saw, let now = focusedElement(pid: pid_t(frontPid)) { moved = !CFEqual(now, e) }
            }
            check = PasteChecker.outcome(sawChange: saw, before: before, last: last, focusMoved: moved)
            if check != PasteCheck.unknown { checkMs = round1(monotonicMs() - posted) }
            field = PasteField(role: watched.flatMap { string($0, kAXRoleAttribute) }, before: before.state, after: last?.state)
        }
        let unchanged = check == PasteCheck.unchanged

        // The keys after it, not into a field the text did not reach.
        var pressSkipped = false
        if let keys = p.press, !keys.isEmpty {
            if unchanged {
                pressSkipped = true
            } else {
                let wait = Double(max(0, p.pressDelayMs ?? 120)) - (monotonicMs() - posted)
                if wait > 0 { try? await Task.sleep(nanoseconds: UInt64(wait * 1_000_000)) }
                for k in keys { postKey(UInt16(clamping: k.key), flags: flags(k.modifiers ?? [])) }
            }
        }

        // The old clipboard back: soon once the text showed (the app read
        // the clipboard), after restoreDelayMs when nobody can tell, never
        // when the text did not show (the user can paste it).
        var restored = false
        var taken = false
        var kept = false
        if restore && unchanged {
            kept = await onMain { NSPasteboard.general.changeCount == ours }
        } else if restore {
            let delay = check == PasteCheck.inserted ? 150 : Double(max(0, p.restoreDelayMs ?? 500))
            let wait = check == PasteCheck.inserted ? delay - (monotonicMs() - posted) : delay
            if wait > 0 { try? await Task.sleep(nanoseconds: UInt64(wait * 1_000_000)) }
            (restored, taken) = await onMain { () -> (Bool, Bool) in
                let pb = NSPasteboard.general
                guard pb.changeCount == ours else { return (false, true) }
                Clipboard.restore(pb, snap)
                return (true, false)
            }
        }
        return PasteResult(posted: true, restored: restored, clipboardTaken: taken, ms: round1(monotonicMs() - t0),
                           check: check, checkMs: checkMs, field: field, secureInput: secure,
                           pressSkipped: pressSkipped ? true : nil, kept: kept ? true : nil)
    }

    /// What a paste's check reads of e: the value's length and digest
    /// (only the length past 200 000 characters), the selection, the
    /// selected text marker range's digest.
    private func reading(_ e: AXUIElement) -> FieldReading {
        var count: Int?
        var v: CFTypeRef?
        if AXUIElementCopyAttributeValue(e, kAXNumberOfCharactersAttribute as CFString, &v) == .success, let n = v as? Int {
            count = n
        }
        var length: Int?
        var digest: Int?
        if (count ?? 0) <= 200_000, let value = string(e, kAXValueAttribute) {
            length = value.utf16.count
            digest = PasteChecker.digest(value)
        } else {
            length = count
        }
        var marker: Int?
        var m: CFTypeRef?
        if AXUIElementCopyAttributeValue(e, "AXSelectedTextMarkerRange" as CFString, &m) == .success, let m,
           CFGetTypeID(m) == AXTextMarkerRangeGetTypeID() {
            // Opaque markers: their bytes say where (the node and offset).
            let r = m as! AXTextMarkerRange
            var bytes = Data()
            for t in [AXTextMarkerRangeCopyStartMarker(r), AXTextMarkerRangeCopyEndMarker(r)] {
                bytes.append(AXTextMarkerGetBytePtr(t), count: AXTextMarkerGetLength(t))
            }
            var h = Hasher()
            h.combine(bytes)
            marker = h.finalize()
        }
        return FieldReading(length: length, digest: digest, selection: range(e).map { TextRange(location: $0.location, length: $0.length) },
                            marker: marker)
    }

    /// The app that has secure event input on (a password field focused,
    /// Terminal's Secure Keyboard Entry), nil when it is off. The owner's
    /// pid is the login session's kCGSSessionSecureInputPID (what
    /// `ioreg -l | grep SecureInput` shows); 0 when it is not there.
    private func secureInputOwner() -> App? {
        var pid = 0
        if let d = CGSessionCopyCurrentDictionary() as? [String: Any], let n = d["kCGSSessionSecureInputPID"] as? Int {
            pid = n
        }
        guard IsSecureEventInputEnabled() || pid != 0 else { return nil }
        if pid > 0, let a = NSRunningApplication(processIdentifier: pid_t(pid)) { return FrontApps.describe(a) }
        return App(pid: pid, bundleId: nil, name: nil)
    }

    private func flags(_ modifiers: [String]) -> CGEventFlags {
        var f: CGEventFlags = []
        for m in modifiers {
            switch m {
            case "command": f.insert(.maskCommand)
            case "shift": f.insert(.maskShift)
            case "option": f.insert(.maskAlternate)
            case "control": f.insert(.maskControl)
            default: break
            }
        }
        return f
    }
}

/// Posts one key press (down and up) at the HID level, marked as
/// VoxtNew's own (the event tap lets it through).
func postKey(_ key: UInt16, flags: CGEventFlags) {
    let src = CGEventSource(stateID: .hidSystemState)
    for down in [true, false] {
        guard let e = CGEvent(keyboardEventSource: src, virtualKey: key, keyDown: down) else { continue }
        e.flags = flags
        e.setIntegerValueField(.eventSourceUserData, value: InjectedEvent.marker)
        e.post(tap: .cghidEventTap)
    }
}

/// Every item of the clipboard with every type's data, to put it back.
/// Promised data (file promises) cannot be kept.
enum Clipboard {
    @MainActor static func snapshot(_ pb: NSPasteboard) -> [[NSPasteboard.PasteboardType: Data]] {
        (pb.pasteboardItems ?? []).map { item in
            var d: [NSPasteboard.PasteboardType: Data] = [:]
            for t in item.types {
                if let data = item.data(forType: t) { d[t] = data }
            }
            return d
        }
    }

    @MainActor static func restore(_ pb: NSPasteboard, _ snap: [[NSPasteboard.PasteboardType: Data]]) {
        pb.clearContents()
        guard !snap.isEmpty else { return }
        let items = snap.map { types -> NSPasteboardItem in
            let item = NSPasteboardItem()
            for (t, data) in types { item.setData(data, forType: t) }
            return item
        }
        pb.writeObjects(items)
    }
}
