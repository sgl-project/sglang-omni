import AppKit
import ApplicationServices

struct ObservationCheck: Error { let message: String }

@MainActor
final class ObservedInput {
    var failure: String?
    var checks = 0

    func target() -> AccessibilityTextTarget {
        // Local opaque handles only. No AX reads, app activation, clipboard access or keyboard posting.
        let handle = AXUIElementCreateApplication(ProcessInfo.processInfo.processIdentifier)
        return AccessibilityTextTarget(application: handle, element: handle,
                                       pid: ProcessInfo.processInfo.processIdentifier,
                                       draft: TextDraft(value: "前😀后", selection: NSRange(location: 3, length: 0)),
                                       applicationName: "测试输入框", checkCurrentState: {
            self.checks += 1
            if let failure = self.failure { throw DictationError(failure) }
        })
    }
}

@main
struct TargetObservationTests {
    static func expect(_ condition: @autoclosure () -> Bool, _ message: String) throws {
        if !condition() { throw ObservationCheck(message: message) }
    }

    @MainActor
    static func expectRejected(_ target: AccessibilityTextTarget, reason: String) throws {
        do {
            try target.validate()
            throw ObservationCheck(message: "真正变化后应拒绝回填：\(reason)")
        } catch let error as DictationError {
            try expect(error.localizedDescription == reason, "应保留第一次变化的具体原因")
        }
    }

    @MainActor
    static func main() async throws {
        let unchanged = ObservedInput()
        let target = unchanged.target()
        try target.validate()
        for _ in 0..<4 { target.observedChange() }
        do { try target.validate() }
        catch { throw ObservationCheck(message: "输入状态未变化时，重复 AX 通知不能禁止回填：\(error.localizedDescription)") }
        try expect(unchanged.checks >= 6, "每次通知都应重新核对真实状态")
        target.observedActivation(pid: ProcessInfo.processInfo.processIdentifier)
        try target.validate()

        let switched = ObservedInput().target()
        switched.observedActivation(pid: ProcessInfo.processInfo.processIdentifier + 1)
        switched.observedActivation(pid: ProcessInfo.processInfo.processIdentifier)
        try expectRejected(switched, reason: "已切换到其他应用，本轮不自动回填；请查看结果后复制。")

        for reason in ["已切换到其他应用", "已切换窗口", "已切换输入框", "草稿已修改", "光标已移动", "无法读取输入框"] {
            let input = ObservedInput()
            let changed = input.target()
            input.failure = reason
            changed.observedChange()
            input.failure = nil
            changed.observedChange()
            changed.observedActivation(pid: ProcessInfo.processInfo.processIdentifier + 1)
            try expectRejected(changed, reason: reason)
        }

        let insertion = DictationInsertion(observationInterval: 0.002)
        let input = ObservedInput()
        let observed = input.target()
        insertion.prepare(observed)
        observed.observedChange()
        let deadline = Date().addingTimeInterval(3)
        while input.checks < 3, Date() < deadline { try await Task.sleep(nanoseconds: 1_000_000) }
        try expect(input.checks >= 3 && insertion.message.contains("完成后填入"),
                   "真实回填控制器不能因无变化通知丢弃待回填目标")
        input.failure = "光标已移动"
        observed.observedChange()
        input.failure = nil
        while insertion.message != "光标已移动", Date() < deadline { try await Task.sleep(nanoseconds: 1_000_000) }
        try expect(insertion.message == "光标已移动", "真实变化后应显示原因并丢弃目标，即使光标已恢复")
        insertion.complete("不应发送")
        try expect(!insertion.isDelivering && !insertion.didInsert, "已作废目标不能进入粘贴阶段")

        let cancelled = input.target()
        insertion.prepare(cancelled)
        insertion.cancel()
        cancelled.observedChange()
        insertion.complete("迟到结果")
        try expect(!insertion.isDelivering && !insertion.didInsert, "取消后的通知不能恢复回填")
        print("PASS: 无变化 AX 通知、真实变化保持作废、具体原因、回填控制器观察与取消")
    }
}
