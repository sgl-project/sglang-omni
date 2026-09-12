#if canImport(DictationCore)
import DictationCore
#endif
import AppKit
import ApplicationServices

@MainActor
enum DictationAccessibility {
    static var isTrusted: Bool { AXIsProcessTrusted() }

    static func requestAccess() {
        let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
        _ = AXIsProcessTrustedWithOptions(options)
    }
}

/// A handle to the original editable element, never to whichever field happens to be focused later.
/// Uses a normal paste: an AX selected-text setter is neither required nor proof of an edit.
@MainActor
final class AccessibilityTextTarget: DictationTextTarget {
    let applicationName: String
    private let application: AXUIElement
    private let element: AXUIElement
    private let pid: pid_t
    private let draft: TextDraft?
    var supportsConfirmation: Bool { draft != nil }
    private let checkCurrentState: () throws -> Void
    private var invalidationReason: String?
    private var observer: AXObserver?
    private var notifications: [(AXUIElement, String)] = []
    private var activationObserver: NSObjectProtocol?
    private var paste: ClipboardPaste?

    static func capture(allowCompatibilityPaste: Bool = false) throws -> AccessibilityTextTarget {
        guard DictationAccessibility.isTrusted else {
            throw DictationError("自动回填需要辅助功能权限。请在 Omni 设置中授权；本轮仍可转写。")
        }
        guard let front = NSWorkspace.shared.frontmostApplication,
              front.processIdentifier != ProcessInfo.processInfo.processIdentifier else {
            throw DictationError("请先点入目标应用的输入框，再按 ⌃⇧Space；本轮结果可手动复制。")
        }
        let applicationName = front.localizedName ?? front.bundleIdentifier ?? "目标应用"
        let application = AXUIElementCreateApplication(front.processIdentifier)
        AXUIElementSetMessagingTimeout(application, 0.3)
        guard let element = axElement(application, kAXFocusedUIElementAttribute),
              let window = axElement(application, kAXFocusedWindowAttribute) else {
            throw DictationError("\(applicationName) 尚未提供可访问的输入框或窗口。请点入输入框后重试。")
        }
        let input = try readInput(element)
        let draft = try input.draft(allowCompatibilityPaste: allowCompatibilityPaste)
        let target = AccessibilityTextTarget(application: application, element: element,
                                             pid: front.processIdentifier, draft: draft,
                                             applicationName: applicationName, checkCurrentState: {
            try validateCurrentState(application: application, element: element, window: window,
                                     pid: front.processIdentifier, input: input)
        })
        target.observeChanges()
        do { try target.validate() }
        catch { target.stopObserving(); throw error }
        return target
    }

    init(application: AXUIElement, element: AXUIElement,
         pid: pid_t, draft: TextDraft?, applicationName: String,
         checkCurrentState: @escaping () throws -> Void) {
        self.application = application
        self.element = element
        self.pid = pid
        self.draft = draft
        self.applicationName = applicationName
        self.checkCurrentState = checkCurrentState
    }

    func validate() throws {
        if let invalidationReason { throw DictationError(invalidationReason) }
        try checkCurrentState()
    }

    private static func validateCurrentState(application: AXUIElement, element: AXUIElement, window: AXUIElement,
                                             pid: pid_t, input: TextInputSnapshot) throws {
        guard NSWorkspace.shared.frontmostApplication?.processIdentifier == pid else {
            throw DictationError("前台应用已变化，本轮不自动回填；请查看结果后复制。")
        }
        guard let currentWindow = axElement(application, kAXFocusedWindowAttribute) else {
            throw DictationError("暂时无法读取原窗口，本轮只提供复制。")
        }
        guard CFEqual(currentWindow, window) else {
            throw DictationError("目标窗口已变化，本轮不自动回填；请查看结果后复制。")
        }
        guard let focus = axElement(application, kAXFocusedUIElementAttribute) else {
            throw DictationError("暂时无法读取输入焦点，本轮只提供复制。")
        }
        guard CFEqual(focus, element) else {
            throw DictationError("焦点已离开原输入框，本轮不自动回填；请查看结果后复制。")
        }
        try input.validateCurrent(readInput(element))
    }

    func observedChange() {
        // AX notifications can repeat without a change to our captured target. Re-read the
        // actual state instead of treating the notification itself as proof of a change.
        guard invalidationReason == nil else { return }
        do { try checkCurrentState() }
        catch { invalidationReason = error.localizedDescription }
    }

    func observedActivation(pid activatedPID: pid_t, applicationName: String? = nil) {
        // Unlike an AX notification, this carries the identity of the activated app.
        // Preserve a real switch even if the original app is already active again.
        guard activatedPID != pid, invalidationReason == nil else { return }
        let destination = applicationName.map { "“\($0)”" } ?? "其他应用"
        invalidationReason = "已切换到\(destination)，本轮不自动回填；请查看结果后复制。"
    }

    func insert(_ text: String) throws {
        try validate()
        guard CGPreflightPostEventAccess() else {
            throw DictationError("macOS 尚未允许模拟粘贴，请重新检查 Omni 听写的辅助功能授权。")
        }
        let paste = ClipboardPaste(pasteboard: .general)
        self.paste = paste
        try paste.perform(text: text, pid: pid, validate: { try self.validate() })
    }

    func finishInsertion(confirmed: Bool) -> String? {
        defer { paste = nil }
        return paste?.finish(confirmed: confirmed)
    }

    func confirms(_ text: String) -> Bool {
        guard let draft, let expected = try? draft.inserting(text),
              let current = try? Self.readInput(element).draft(allowCompatibilityPaste: false) else { return false }
        return current.value == expected && current.selection == NSRange(location: draft.selection.location + text.utf16.count, length: 0)
    }

    func stopObserving() {
        if let observer {
            notifications.forEach { AXObserverRemoveNotification(observer, $0.0, $0.1 as CFString) }
            CFRunLoopRemoveSource(CFRunLoopGetMain(), AXObserverGetRunLoopSource(observer), .commonModes)
        }
        notifications = []
        observer = nil
        if let activationObserver {
            NSWorkspace.shared.notificationCenter.removeObserver(activationObserver)
        }
        activationObserver = nil
    }

    private func observeChanges() {
        var observer: AXObserver?
        let result = AXObserverCreate(pid, { _, _, _, context in
            guard let context else { return }
            // This observer is registered exclusively on the main run loop.
            MainActor.assumeIsolated {
                Unmanaged<AccessibilityTextTarget>.fromOpaque(context).takeUnretainedValue().observedChange()
            }
        }, &observer)
        if result == .success, let observer {
            self.observer = observer
            for pair in [(application, kAXFocusedUIElementChangedNotification),
                         (application, kAXFocusedWindowChangedNotification),
                         (element, kAXValueChangedNotification), (element, kAXSelectedTextChangedNotification)] {
                if AXObserverAddNotification(observer, pair.0, pair.1 as CFString,
                                             Unmanaged.passUnretained(self).toOpaque()) == .success {
                    notifications.append(pair)
                }
            }
            CFRunLoopAddSource(CFRunLoopGetMain(), AXObserverGetRunLoopSource(observer), .commonModes)
        }
        // Latch an application switch even if the user switches back before ASR completes.
        activationObserver = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification, object: nil, queue: .main
        ) { [weak self] notification in
            MainActor.assumeIsolated {
                guard let self,
                      let app = notification.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication else { return }
                self.observedActivation(pid: app.processIdentifier,
                                        applicationName: app.localizedName ?? app.bundleIdentifier)
            }
        }
    }

    private static func readInput(_ element: AXUIElement) throws -> TextInputSnapshot {
        // Reject marked terminal/search/password contexts, including embedded terminal panels.
        var ancestor: AXUIElement? = element
        for _ in 0..<24 {
            guard let node = ancestor else { break }
            if attribute(node, kAXRoleAttribute) as? String == kAXWindowRole { break }
            let labels = [kAXRoleAttribute, kAXSubroleAttribute, kAXDescriptionAttribute, kAXTitleAttribute]
                .compactMap { attribute(node, $0) as? String }.joined(separator: " ").lowercased()
            if ["terminal", "终端", "securetextfield", "searchfield"].contains(where: { labels.contains($0) }) {
                throw DictationError("密码、搜索和标记为终端的区域不自动回填；请使用普通输入框。")
            }
            ancestor = axElement(node, kAXParentAttribute)
        }
        return try TextInputSnapshot(read: { attribute(element, $0) })
    }

    private static func attribute(_ element: AXUIElement, _ name: String) -> CFTypeRef? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, name as CFString, &value) == .success else { return nil }
        return value
    }

    private static func axElement(_ element: AXUIElement, _ name: String) -> AXUIElement? {
        guard let value = attribute(element, name), CFGetTypeID(value) == AXUIElementGetTypeID() else { return nil }
        return (value as! AXUIElement)
    }
}

/// Optional fields stay unknown; compatibility mode never invents an empty draft or cursor.
struct TextInputSnapshot {
    let value: String?
    let selection: NSRange?

    init(read: (String) -> CFTypeRef?) throws {
        let role = read(kAXRoleAttribute) as? String
        let subrole = read(kAXSubroleAttribute) as? String
        guard [kAXTextAreaRole, kAXTextFieldRole].contains(role ?? ""), subrole != kAXSecureTextFieldSubrole,
              read(kAXEnabledAttribute) as? Bool == true,
              read("AXEditable") as? Bool != false,
              read(kAXFocusedAttribute) as? Bool != false else {
            throw DictationError("当前区域不是可用的普通文本输入框，本轮只提供复制。")
        }
        // The containing app's focused-element attribute establishes focus even if the
        // field doesn't expose its own AXFocused. An AXSelectedText setter is not needed.
        value = read(kAXValueAttribute) as? String
        if let selected = read(kAXSelectedTextRangeAttribute) {
            var range = CFRange()
            guard CFGetTypeID(selected) == AXValueGetTypeID(),
                  AXValueGetValue(selected as! AXValue, .cfRange, &range),
                  range.location >= 0, range.location != NSNotFound, range.length >= 0,
                  range.length <= Int.max - range.location else {
                throw DictationError("输入框返回的光标或选区无效，本轮只提供复制。")
            }
            selection = NSRange(location: range.location, length: range.length)
        } else { selection = nil }
    }

    func draft(allowCompatibilityPaste: Bool) throws -> TextDraft? {
        if let value, let selection {
            let draft = TextDraft(value: value, selection: selection)
            _ = try draft.inserting("")
            return draft
        }
        guard allowCompatibilityPaste else {
            throw DictationError("无法读取完整草稿或光标。可在设置开启“兼容粘贴”后重新录音，或手动复制本轮结果。")
        }
        return nil
    }

    func validateCurrent(_ current: TextInputSnapshot) throws {
        if let value, current.value != value {
            throw DictationError("草稿文字已变化或无法读取，本轮不自动回填；请查看结果后复制。")
        }
        if let selection, current.selection != selection {
            throw DictationError("光标或选区已变化或无法读取，本轮不自动回填；请查看结果后复制。")
        }
    }
}
