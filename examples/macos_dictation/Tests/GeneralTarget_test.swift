import AppKit
import ApplicationServices

private func expect(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() { throw DictationError(message) }
}

private func rangeValue(_ range: CFRange) -> AXValue {
    var range = range
    return AXValueCreate(.cfRange, &range)!
}

@MainActor
private func snapshot(value: String? = "前😀后", selection: CFRange? = CFRange(location: 3, length: 0),
                      role: String = kAXTextAreaRole, subrole: String? = nil,
                      enabled: Bool = true, editable: Bool? = nil, focused: Bool? = nil) throws -> TextInputSnapshot {
    var attributes: [String: CFTypeRef] = [kAXRoleAttribute: role as CFString, kAXEnabledAttribute: enabled as NSNumber]
    if let value { attributes[kAXValueAttribute] = value as CFString }
    if let selection { attributes[kAXSelectedTextRangeAttribute] = rangeValue(selection) }
    if let subrole { attributes[kAXSubroleAttribute] = subrole as CFString }
    if let editable { attributes["AXEditable"] = editable as NSNumber }
    if let focused { attributes[kAXFocusedAttribute] = focused as NSNumber }
    return try TextInputSnapshot(read: { attributes[$0] })
}

@MainActor
private final class Target: DictationTextTarget {
    let applicationName: String
    let supportsConfirmation: Bool
    var changed = false
    var inserted: [String] = []
    var confirmations = 0
    var cleanups: [Bool] = []
    init(_ name: String, confirms: Bool) { applicationName = name; supportsConfirmation = confirms }
    func validate() throws { if changed { throw DictationError("输入目标已变化") } }
    func insert(_ text: String) throws { inserted.append(text) }
    func confirms(_ text: String) -> Bool { confirmations += 1; return true }
    func stopObserving() {}
    func finishInsertion(confirmed: Bool) -> String? {
        cleanups.append(confirmed)
        return confirmed ? nil : "剪贴板暂保留本轮文字"
    }
}

@main
private enum GeneralTargetTests {
    @MainActor
    static func reject(_ operation: () throws -> Void) throws {
        var rejected = false
        do { try operation() } catch is DictationError { rejected = true }
        try expect(rejected, "该目标必须拒绝，不能退到兼容粘贴")
    }

    @MainActor
    static func main() async throws {
        // No advertised AX setter is supplied: a normal focused text field can still receive Cmd+V.
        let full = try snapshot()
        let draft = try full.draft(allowCompatibilityPaste: false)
        try expect(draft == TextDraft(value: "前😀后", selection: NSRange(location: 3, length: 0)),
                   "不要求 AXSelectedText setter，且按 UTF-16 读取草稿")
        try full.validateCurrent(snapshot())
        try reject { try full.validateCurrent(snapshot(value: "变了")) }
        try reject { try full.validateCurrent(snapshot(selection: CFRange(location: 0, length: 0))) }
        try reject { try full.validateCurrent(snapshot(value: nil)) }
        try reject { try full.validateCurrent(snapshot(selection: nil)) }

        for role in [kAXTextAreaRole, kAXTextFieldRole] {
            _ = try snapshot(role: role).draft(allowCompatibilityPaste: false)
        }
        for role in [kAXButtonRole, kAXStaticTextRole, kAXGroupRole] {
            try reject { _ = try snapshot(role: role) }
        }
        try reject { _ = try snapshot(enabled: false) }
        try reject { _ = try snapshot(editable: false) }
        try reject { _ = try snapshot(focused: false) }

        var readSecret = false
        try reject {
            _ = try TextInputSnapshot(read: { name in
                if name == kAXRoleAttribute { return kAXTextFieldRole as CFString }
                if name == kAXSubroleAttribute { return kAXSecureTextFieldSubrole as CFString }
                if name == kAXValueAttribute { readSecret = true; return "不应读取" as CFString }
                return nil
            })
        }
        try expect(!readSecret, "密码区域必须在读取内容前拒绝")

        for partial in [try snapshot(value: nil), try snapshot(selection: nil), try snapshot(value: nil, selection: nil)] {
            try reject { _ = try partial.draft(allowCompatibilityPaste: false) }
            let partialDraft = try partial.draft(allowCompatibilityPaste: true)
            try expect(partialDraft == nil, "兼容粘贴不伪造草稿或光标")
        }
        let knownText = try snapshot(selection: nil)
        try knownText.validateCurrent(snapshot(selection: nil))
        try reject { try knownText.validateCurrent(snapshot(value: "改了", selection: nil)) }
        let knownSelection = try snapshot(value: nil)
        try reject { try knownSelection.validateCurrent(snapshot(value: nil, selection: CFRange(location: 0, length: 0))) }
        for badRange in [CFRange(location: -1, length: 0), CFRange(location: 99, length: 0), CFRange(location: 2, length: 0)] {
            try reject { _ = try snapshot(selection: badRange).draft(allowCompatibilityPaste: true) }
        }

        let insertion = DictationInsertion()
        for name in ["Codex", "文本编辑", "飞书", "浏览器"] {
            let target = Target(name, confirms: true)
            insertion.prepare(target)
            insertion.complete("语音")
            let deadline = Date().addingTimeInterval(3)
            while insertion.isDelivering, Date() < deadline { try await Task.sleep(nanoseconds: 1_000_000) }
            try expect(insertion.didInsert && insertion.targetApplicationName == name,
                       "回填状态必须使用实际应用名")
            try expect(target.inserted == ["语音"] && target.cleanups == [true], "完整目标仅粘贴一次并确认")
        }

        let compatibility = Target("测试聊天窗口", confirms: false)
        insertion.prepare(compatibility)
        insertion.complete("兼容文字")
        insertion.complete("迟到的重复文字")
        try expect(insertion.didTriggerPaste && !insertion.didInsert && !insertion.isDelivering && insertion.hasWarning,
                   "兼容路径只能报告已触发，不能报告已确认或触发成功淡出")
        try expect(compatibility.inserted == ["兼容文字"] && compatibility.confirmations == 0 && compatibility.cleanups == [false],
                   "不可验证目标不读回猜测成功、不重试、不能恢复旧剪贴板")
        try expect(insertion.message.contains("已触发粘贴") && insertion.message.contains("无法确认"), "明确显示兼容模式的结果边界")

        let changed = Target("测试聊天窗口", confirms: false)
        insertion.prepare(changed)
        changed.changed = true
        insertion.complete("不应输入")
        try expect(changed.inserted.isEmpty && !insertion.didTriggerPaste, "兼容粘贴仍检查目标变化")
        let cancelled = Target("测试聊天窗口", confirms: false)
        insertion.prepare(cancelled)
        insertion.cancel()
        insertion.complete("迟到结果")
        try expect(cancelled.inserted.isEmpty && insertion.targetApplicationName.isEmpty, "取消清除目标和迟到结果")
        print("PASS: generic text fields, partial snapshots, password/readonly guards, real app names, unverified paste and cancellation")
    }
}
