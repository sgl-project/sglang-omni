import AppKit
import Carbon

private func expect(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() { throw DictationError(message) }
}

@MainActor
private final class Recorder: AudioRecording {
    func start() async throws {}
    func stop() throws -> Data { Data([1]) }
    func cancel() {}
}

@MainActor
private final class Service: SpeechServing {
    var reply: CheckedContinuation<String, Error>?
    func transcribe(wav: Data) async throws -> String {
        try await withCheckedThrowingContinuation { reply = $0 }
    }
    func polish(text: String) async throws -> String { text }
}

@MainActor
private func wait(_ condition: () -> Bool) async throws {
    let deadline = Date().addingTimeInterval(3)
    while !condition(), Date() < deadline { try await Task.sleep(nanoseconds: 1_000_000) }
    try expect(condition(), "等待状态超时")
}

@main
private enum CurrentCursorTests {
    @MainActor
    static func main() async throws {
        // A named test clipboard and injected event transport: never reads the user's
        // clipboard, inspects another app, requests a microphone or posts an actual key.
        let board = NSPasteboard.withUniqueName()
        defer { board.releaseGlobally() }
        board.clearContents()
        try expect(board.setString("原剪贴板", forType: .string), "测试剪贴板应可写")
        var focus = "录音开始时的应用"
        var destinations: [String] = []
        var events: [CGEvent] = []
        var processIDs: [pid_t?] = []
        var allowed = true
        func target() -> CurrentCursorTarget {
            let paste = ClipboardPaste(pasteboard: board) { pid, sequence in
                processIDs.append(pid)
                destinations.append(focus)
                events.append(contentsOf: sequence)
            }
            return CurrentCursorTarget(paste: paste, canPost: { allowed })
        }

        let insertion = DictationInsertion(observationInterval: 0.002)
        let service = Service()
        let session = DictationSession(recorder: Recorder(), service: service)
        session.onResult = { insertion.complete($0) }
        let current = target()
        insertion.prepare(current)
        try expect(insertion.message.contains("当前光标") && !insertion.message.contains("保持原输入框"),
                   "当前光标模式必须说明可切换输入位置")
        session.submitAudio(Data([1]))
        try await wait { service.reply != nil }
        focus = "另一个应用的输入光标"
        try current.validate()
        // Allow the real preparation observer to run after the switch.
        try await Task.sleep(nanoseconds: 10_000_000)
        service.reply?.resume(returning: "语音😀\n第二行")
        service.reply = nil
        try await wait { insertion.didTriggerPaste }
        insertion.complete("重复回调")
        try expect(destinations == ["另一个应用的输入光标"] && processIDs.count == 1 && processIDs[0] == nil,
                   "识别结束按当前焦点投递，不能指定录音开始时的 PID")
        try expect(events.count == 2 && events[0].type == .keyDown && events[1].type == .keyUp,
                   "仅发送一组按下和松开事件")
        try expect(events.allSatisfy {
            $0.getIntegerValueField(.keyboardEventKeycode) == Int64(kVK_ANSI_V) && $0.flags == .maskCommand
        }, "只能发送 Command+V，不发送 Enter 或继承录音快捷键的修饰键")
        try expect(!insertion.didInsert && insertion.hasWarning && insertion.message.contains("无法确认"),
                   "按键已触发不等于编辑器已确认")
        try expect(board.string(forType: .string) == "语音😀\n第二行", "未确认时保留本轮文字，不能让迟到事件粘入旧内容")

        let revision = board.changeCount
        allowed = false
        insertion.prepare(target())
        insertion.complete("无权限时不写入")
        try expect(!insertion.didTriggerPaste && events.count == 2 && board.changeCount == revision,
                   "权限拒绝不得写剪贴板或投递按键")

        allowed = true
        insertion.prepare(target())
        session.submitAudio(Data([1]))
        try await wait { service.reply != nil }
        insertion.cancel()
        session.cancel()
        service.reply?.resume(returning: "已取消的迟到结果")
        service.reply = nil
        try await Task.sleep(nanoseconds: 20_000_000)
        try expect(events.count == 2 && session.rawText.isEmpty && board.changeCount == revision,
                   "取消后迟到的 ASR 结果不粘贴")

        insertion.prepare(target())
        insertion.complete("")
        try expect(events.count == 2 && board.changeCount == revision, "空结果不粘贴")

        var permissionChecks = 0
        let revoked = CurrentCursorTarget(paste: ClipboardPaste(pasteboard: board) { _, _ in
            throw DictationError("权限失效后不应投递")
        }, canPost: {
            permissionChecks += 1
            return permissionChecks < 4
        })
        insertion.prepare(revoked)
        insertion.complete("准备剪贴板期间权限失效")
        try expect(!insertion.didTriggerPaste && board.string(forType: .string) == "语音😀\n第二行",
                   "投递前权限失效应恢复原剪贴板")

        print("PASS: current-focus routing, one Cmd+V without Enter, permission failures, cancellation, empty input and clipboard cleanup")
    }
}
