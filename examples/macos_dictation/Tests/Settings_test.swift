import AppKit
import Carbon

private struct Failure: Error { let message: String }
private func check(_ value: @autoclosure () -> Bool, _ message: String) throws {
    if !value() { throw Failure(message: message) }
}

@MainActor
private final class Registrar: ShortcutRegistering {
    var registered: RecordingShortcut?
    var rejected: Set<RecordingShortcut> = []
    func register(_ shortcut: RecordingShortcut) throws {
        if rejected.contains(shortcut) { throw Failure(message: "冲突") }
        registered = shortcut
    }
    func unregister() { registered = nil }
}

@main
struct SettingsTests {
    @MainActor
    static func main() throws {
        let domain = "local.omni.dictation.tests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: domain)!
        defer { defaults.removePersistentDomain(forName: domain) }
        let prefs = ClientPreferences(defaults: defaults)
        try check(prefs.recordingShortcut == .default && !prefs.polishEnabled && !prefs.personalBackgroundEnabled, "首次设置应保持保守默认")
        let key = RecordingShortcut(keyCode: UInt32(kVK_ANSI_D), modifiers: UInt32(controlKey | shiftKey))
        prefs.recordingShortcut = key
        prefs.polishEnabled = true
        prefs.personalBackground = "测试背景"
        prefs.personalBackgroundEnabled = true
        let reloaded = ClientPreferences(defaults: UserDefaults(suiteName: domain)!)
        try check(reloaded.recordingShortcut == key && reloaded.polishEnabled && reloaded.personalBackground == "测试背景", "设置必须可以重新载入")
        reloaded.personalBackground = ""
        try check(prefs.personalBackground.isEmpty, "清空必须删除保存的背景")

        let registrar = Registrar()
        var saved: [RecordingShortcut] = []
        let control = RecordingShortcutController(shortcut: .default, registrar: registrar) { saved.append($0) }
        control.start()
        try check(registrar.registered == .default, "启动应注册默认键")
        control.beginCapture()
        try check(registrar.registered == nil && control.isCapturing, "录入期间必须注销旧快捷键")
        control.capture(key)
        try check(registrar.registered == nil && saved.isEmpty, "按下期间不能保存注册或触发录音")
        control.release(keyCode: key.keyCode)
        try check(registrar.registered == key && saved == [key] && !control.isCapturing, "松开后才生效并保存")
        let conflict = RecordingShortcut(keyCode: UInt32(kVK_ANSI_F), modifiers: UInt32(controlKey | shiftKey))
        registrar.rejected.insert(conflict)
        control.beginCapture()
        control.capture(conflict)
        control.release(keyCode: conflict.keyCode)
        try check(registrar.registered == key && saved == [key] && control.shortcut == key, "注册失败必须恢复旧键且不覆盖保存值")
        control.beginCapture()
        control.capture(RecordingShortcut(keyCode: UInt32(kVK_ANSI_A), modifiers: 0))
        try check(control.isCapturing && !control.notice.isEmpty, "无修饰键不能录入")
        control.capture(.cancel)
        try check(control.isCapturing && !control.notice.isEmpty, "不能与取消本轮快捷键冲突")
        control.cancelCapture()
        try check(registrar.registered == key, "取消或失焦应恢复原键")
        control.restoreDefault()
        try check(control.shortcut == .default && saved.last == .default, "恢复默认必须注册并保存")
        let gate = ShortcutPressGate()
        try check(gate.press(1) && !gate.press(1), "长按不得重复触发")
        gate.release(1)
        try check(gate.press(1), "松开后可以再次触发")
        gate.reset()
        try check(gate.press(1), "重新注册不得残留按住状态")
        print("PASS: settings persistence/clear, capture on release, rollback, validation, repeat suppression")
    }
}
