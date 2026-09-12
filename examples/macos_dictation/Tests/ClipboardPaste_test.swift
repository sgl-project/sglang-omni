import AppKit
import Carbon

struct PasteCheck: Error { let message: String }
func require(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() { throw PasteCheck(message: message) }
}

@MainActor
final class PasteLifecycleTarget: DictationTextTarget {
    let applicationName = "测试输入框"
    var confirmation = false
    var writeFailure = false
    var finishes: [Bool] = []
    func validate() throws {}
    func insert(_ text: String) throws { if writeFailure { throw DictationError("粘贴准备失败") } }
    func confirms(_ text: String) -> Bool { confirmation }
    func stopObserving() {}
    func finishInsertion(confirmed: Bool) -> String? { finishes.append(confirmed); return nil }
}

@main
struct ClipboardPasteTests {
    @MainActor
    static func main() async throws {
        // Isolated test pasteboard: never reads or writes the user's general clipboard.
        let board = NSPasteboard.withUniqueName()
        defer { board.releaseGlobally() }
        let binaryType = NSPasteboard.PasteboardType("local.omni.test.binary")
        let original = NSPasteboardItem()
        original.setString("原剪贴板", forType: .string)
        original.setData(Data([0, 1, 255]), forType: binaryType)
        let second = NSPasteboardItem()
        second.setString("第二项", forType: .string)
        try require(board.writeObjects([original, second]), "测试剪贴板不可写")

        var events: [CGEvent] = []
        var destination: pid_t?
        let paste = ClipboardPaste(pasteboard: board) { pid, sequence in
            destination = pid
            events.append(contentsOf: sequence)
        }
        try paste.perform(text: "语音😀\n第二行", pid: 12345, validate: {})
        try require(board.string(forType: .string) == "语音😀\n第二行", "发送按键前应准备完整文本")
        try require(destination == 12345 && events.count == 2, "只向原 PID 发送一组按下和松开事件")
        try require(events[0].type == .keyDown && events[1].type == .keyUp, "粘贴必须完整释放按键")
        try require(events.allSatisfy {
            $0.getIntegerValueField(.keyboardEventKeycode) == Int64(kVK_ANSI_V) && $0.flags == .maskCommand
        }, "只能发送 Command+V，不得带 Control/Option 或发送 Enter")
        do {
            try paste.perform(text: "重复", pid: 12345, validate: {})
            throw PasteCheck(message: "不应重复投递")
        } catch is DictationError {}
        try require(events.count == 2, "重复回调不能再粘贴")
        try require(paste.finish(confirmed: true) == nil, "确认后正常恢复应无错误")
        // NSPasteboard.string(forType:) may combine text from multiple items. Check each item.
        try require(board.pasteboardItems?.map { $0.string(forType: .string) } == ["原剪贴板", "第二项"], "逐项恢复原文字和顺序")
        try require(board.pasteboardItems?.first?.data(forType: binaryType) == Data([0, 1, 255]), "恢复原二进制格式")

        let changed = ClipboardPaste(pasteboard: board) { _, _ in }
        try changed.perform(text: "语音", pid: 12345, validate: {})
        board.clearContents()
        board.setString("用户新复制的内容", forType: .string)
        _ = changed.finish(confirmed: true)
        try require(board.string(forType: .string) == "用户新复制的内容", "用户新复制的内容优先，不能覆盖")

        let revision = board.changeCount
        let blocked = ClipboardPaste(pasteboard: board) { _, _ in throw PasteCheck(message: "不应发送") }
        do {
            try blocked.perform(text: "禁止写入", pid: 12345) { throw DictationError("目标变化") }
            throw PasteCheck(message: "目标变化应拒绝")
        } catch is DictationError {}
        try require(board.changeCount == revision, "目标检查失败不能改动剪贴板")

        let concurrentCopy = ClipboardPaste(pasteboard: board) { _, _ in throw PasteCheck(message: "不应发送") }
        do {
            try concurrentCopy.perform(text: "不得写入", pid: 12345) {
                board.clearContents()
                board.setString("用户新复制的内容", forType: .string)
            }
            throw PasteCheck(message: "快照后剪贴板变化应拒绝")
        } catch is DictationError {}
        _ = concurrentCopy.finish(confirmed: false)
        try require(board.string(forType: .string) == "用户新复制的内容", "保存剪贴板期间有新复制时不能覆盖")

        var checks = 0
        var posted = false
        let moved = ClipboardPaste(pasteboard: board) { _, _ in posted = true }
        do {
            try moved.perform(text: "待回填", pid: 12345) {
                checks += 1
                if checks == 2 { throw DictationError("准备剪贴板时切走") }
            }
            throw PasteCheck(message: "最后检查应拒绝")
        } catch is DictationError {}
        _ = moved.finish(confirmed: false)
        try require(!posted && board.string(forType: .string) == "用户新复制的内容", "发送前目标变化应恢复剪贴板且不发送")

        let unconfirmed = ClipboardPaste(pasteboard: board) { _, _ in }
        try unconfirmed.perform(text: "迟到粘贴也只能是本轮文字", pid: 12345, validate: {})
        let warning = unconfirmed.finish(confirmed: false)
        try require(warning?.contains("剪贴板") == true && board.string(forType: .string) == "迟到粘贴也只能是本轮文字",
                    "未确认消费时不可提前恢复旧内容，让迟到事件粘贴错误数据")

        board.clearContents()
        let empty = ClipboardPaste(pasteboard: board) { _, _ in }
        try empty.perform(text: "文本", pid: 12345, validate: {})
        _ = empty.finish(confirmed: true)
        try require(board.pasteboardItems?.isEmpty != false, "恢复原本空的剪贴板")

        // Test the real coordinator's clipboard cleanup lifecycle, not just the clipboard helper.
        let insertion = DictationInsertion(verificationLimit: 0.01)
        let success = PasteLifecycleTarget()
        success.confirmation = true
        insertion.prepare(success)
        insertion.complete("文本")
        try await wait { !insertion.isDelivering }
        insertion.cancel()
        try require(success.finishes == [true], "成功只确认清理一次，之后取消不能再次恢复剪贴板")

        let cancelled = PasteLifecycleTarget()
        insertion.prepare(cancelled)
        insertion.complete("文本")
        insertion.cancel()
        try require(cancelled.finishes == [false], "投递后取消仍必须清理剪贴板资源")

        let timeout = PasteLifecycleTarget()
        insertion.prepare(timeout)
        insertion.complete("文本")
        try await wait { !insertion.isDelivering }
        try require(timeout.finishes == [false], "无法确认时也必须结束剪贴板租用")

        let failure = PasteLifecycleTarget()
        failure.writeFailure = true
        insertion.prepare(failure)
        insertion.complete("文本")
        try require(failure.finishes == [false], "准备或写入抛错时必须回收剪贴板状态")
        print("PASS: 定向 Command+V、单次发送、完整剪贴板恢复、用户复制优先、目标变化、未确认消费")
    }

    @MainActor
    static func wait(_ predicate: () -> Bool) async throws {
        let deadline = Date().addingTimeInterval(3)
        while !predicate(), Date() < deadline { try await Task.sleep(nanoseconds: 1_000_000) }
        try require(predicate(), "等待回填清理超时")
    }
}
