import Foundation

struct InsertionCheck: Error { let message: String }
func expect(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
    if try !condition() { throw InsertionCheck(message: message) }
}

@MainActor
final class FakeTarget: DictationTextTarget {
    let applicationName = "Codex"
    var changed = false
    var writeError = false
    var confirmed = true
    var writes: [String] = []
    var stopped = false
    func validate() throws { if changed { throw DictationError("输入框已变化") } }
    func insert(_ text: String) throws {
        if writeError { throw DictationError("写入失败") }
        writes.append(text)
    }
    func confirms(_ text: String) -> Bool { confirmed }
    func stopObserving() { stopped = true }
}

@MainActor
final class InsertionRecorder: AudioRecording {
    func start() async throws {}
    func stop() throws -> Data { Data([1]) }
    func cancel() {}
}

@MainActor
final class InsertionService: SpeechServing {
    var continuation: CheckedContinuation<String, Error>?
    func transcribe(wav: Data) async throws -> String {
        try await withCheckedThrowingContinuation { continuation = $0 }
    }
    func polish(text: String) async throws -> String { "整理结果" }
}

@main
struct InsertionTests {
    @MainActor
    static func wait(_ predicate: () -> Bool) async throws {
        let deadline = Date().addingTimeInterval(3)
        while !predicate(), Date() < deadline { try await Task.sleep(nanoseconds: 1_000_000) }
        try expect(predicate(), "等待状态超时")
    }

    @MainActor
    static func main() async throws {
        let draft = TextDraft(value: "前😀后", selection: NSRange(location: 3, length: 0))
        try expect(try draft.inserting("语音") == "前😀语音后", "按 UTF-16 光标位置插入，保留两侧原文")
        let selected = TextDraft(value: "前😀后", selection: NSRange(location: 1, length: 2))
        try expect(try selected.inserting("新") == "前新后", "只替换原选区")
        for range in [NSRange(location: 2, length: 0), NSRange(location: 8, length: 1),
                      NSRange(location: NSNotFound, length: 0), NSRange(location: 1, length: Int.max)] {
            do {
                _ = try TextDraft(value: "前😀后", selection: range).inserting("新")
                throw InsertionCheck(message: "无效范围应拒绝")
            } catch is DictationError {}
        }

        let insertion = DictationInsertion(verificationLimit: 0.01, observationInterval: 0.002)
        let normal = FakeTarget()
        insertion.prepare(normal)
        insertion.complete("语音")
        insertion.complete("重复结果")
        try await wait { !insertion.isDelivering }
        try expect(normal.writes == ["语音"] && normal.stopped && insertion.didInsert, "一轮只插入一次并确认")

        let changed = FakeTarget()
        insertion.prepare(changed)
        changed.changed = true
        insertion.complete("不得写入")
        try expect(changed.writes.isEmpty && !insertion.didInsert, "提交前必须重新检查目标")

        let observed = FakeTarget()
        insertion.prepare(observed)
        observed.changed = true
        try await wait { observed.stopped }
        observed.changed = false
        insertion.complete("不得写入")
        try expect(observed.writes.isEmpty, "检测到变化后即使恢复原焦点也不能回填")

        let cancelled = FakeTarget()
        insertion.prepare(cancelled)
        insertion.cancel()
        insertion.complete("迟到结果")
        try expect(cancelled.writes.isEmpty && cancelled.stopped, "取消必须清除回填目标")

        let failed = FakeTarget()
        failed.writeError = true
        insertion.prepare(failed)
        insertion.complete("原文")
        try expect(!insertion.didInsert && insertion.message.contains("写入失败"), "写入失败不能显示成功")

        let uncertain = FakeTarget()
        uncertain.confirmed = false
        insertion.prepare(uncertain)
        insertion.complete("原文")
        try await wait { !insertion.isDelivering }
        try expect(!insertion.didInsert && uncertain.writes.count == 1 && insertion.message.contains("未确认"),
                   "不能确认时不重试或假报成功")

        let empty = FakeTarget()
        insertion.prepare(empty)
        insertion.complete("")
        try expect(empty.writes.isEmpty && empty.stopped, "空转写不写入")

        let old = FakeTarget()
        old.confirmed = false
        insertion.prepare(old)
        insertion.complete("旧结果")
        insertion.cancel()
        let next = FakeTarget()
        insertion.prepare(next)
        insertion.complete("新结果")
        try await wait { !insertion.isDelivering }
        try expect(insertion.didInsert && next.writes == ["新结果"], "旧轮确认任务不能影响新一轮")

        // Use the actual session callback, including cancellation of a delayed ASR.
        let service = InsertionService()
        let session = DictationSession(recorder: InsertionRecorder(), service: service)
        session.onResult = { insertion.complete($0) }
        let delayed = FakeTarget()
        insertion.prepare(delayed)
        session.submitAudio(Data([1]))
        try await wait { service.continuation != nil }
        session.cancel()
        insertion.cancel()
        service.continuation?.resume(returning: "过期文字")
        try await Task.sleep(nanoseconds: 10_000_000)
        try expect(delayed.writes.isEmpty, "真实会话取消后不能触发写入")

        service.continuation = nil
        session.polishEnabled = true
        let polished = FakeTarget()
        insertion.prepare(polished)
        session.submitAudio(Data([1]))
        try await wait { service.continuation != nil }
        service.continuation?.resume(returning: "原文")
        try await wait { session.phase == .ready && !insertion.isDelivering }
        try expect(polished.writes == ["整理结果"] && session.rawText == "原文", "可选整理完成后才回填，原文仍保留")
        print("PASS: 草稿选区、单次回填、变化检测、取消隔离、失败与未确认、真实会话回调")
    }
}
