import Foundation

struct TimingFailure: Error { }

@MainActor
final class TimingClock {
    var time = 100.0
    func advance(_ seconds: Double) { time += seconds }
}

@MainActor
final class TimingRecorder: AudioRecording {
    let clock: TimingClock
    init(_ clock: TimingClock) { self.clock = clock }
    func start() async throws { clock.advance(20) }
    func stop() throws -> Data { clock.advance(30); return Data([1]) }
    func cancel() { }
}

@MainActor
final class TimingService: SpeechServing {
    var asr: () async throws -> String = { "原文" }
    var llm: () async throws -> String = { "原文。" }
    var polishCalls = 0
    func transcribe(wav: Data) async throws -> String { try await asr() }
    func polish(text: String) async throws -> String {
        polishCalls += 1
        return try await llm()
    }
}

@main
struct TimingTest {
    @MainActor
    static func expect(_ condition: @autoclosure () -> Bool, _ message: String) {
        precondition(condition(), message)
    }

    @MainActor
    static func near(_ actual: Double?, _ expected: Double) -> Bool {
        actual.map { abs($0 - expected) < 0.000001 } ?? false
    }

    @MainActor
    static func wait(_ predicate: () -> Bool) async throws {
        let deadline = ProcessInfo.processInfo.systemUptime + 3
        while !predicate(), ProcessInfo.processInfo.systemUptime < deadline {
            try await Task.sleep(nanoseconds: 1_000_000)
        }
        expect(predicate(), "Timed out waiting for a session transition")
    }

    @MainActor
    static func main() async throws {
        let clock = TimingClock()
        let service = TimingService()
        let session = DictationSession(recorder: TimingRecorder(clock), service: service,
                                       processingTime: { clock.time })
        var deliveries = 0
        session.onResult = { _ in deliveries += 1; clock.advance(10) }
        service.asr = { clock.advance(1.2); return "原文" }
        service.llm = { clock.advance(0.8); return "原文。" }
        session.polishEnabled = true
        session.toggleRecording()
        try await wait { session.phase == .recording }
        expect(TimingPresentation(session: session) == nil, "Recording must not show previous timings")
        session.toggleRecording()
        session.polishEnabled = false
        try await wait { session.phase == .ready }
        expect(near(session.asrSeconds, 1.2) && near(session.polishSeconds, 0.8), "Stage durations must be independent")
        expect(near(session.totalSeconds, 2), "Total must exclude permission, recording and delivery")
        let completed = TimingPresentation(session: session)!
        expect(completed.summary == "ASR 1.2s · LLM 0.8s" && completed.polishStatus == "完成",
               "Completed round must use its captured polishing switch")
        expect(deliveries == 1, "Metrics must not trigger an extra delivery")

        session.submitAudio(Data([1]))
        expect(session.asrSeconds == nil && session.polishSeconds == nil && session.totalSeconds == nil,
               "A new round must clear all previous timings")
        try await wait { session.phase == .ready }
        let disabled = TimingPresentation(session: session)!
        expect(disabled.summary == "ASR 1.2s" && disabled.polishDuration == "—" && disabled.polishStatus == "已关闭",
               "Disabled LLM must not appear as a zero-duration request")
        expect(near(session.totalSeconds, 1.2), "ASR-only total must exclude delivery")

        session.polishEnabled = true
        service.llm = { clock.advance(4.5); throw TimingFailure() }
        session.submitAudio(Data([1]))
        try await wait { session.phase == .ready }
        let fallback = TimingPresentation(session: session)!
        expect(near(session.polishSeconds, 4.5) && near(session.totalSeconds, 5.7), "Failed LLM work still costs time")
        expect(fallback.polishStatus == "已回退原文" && session.resultText == session.rawText,
               "Rejected or failed polishing must retain raw text and an accurate status")

        service.asr = { clock.advance(3.5); throw TimingFailure() }
        let beforeFailure = deliveries
        session.submitAudio(Data([1]))
        try await wait { session.phase == .failed }
        let failed = TimingPresentation(session: session)!
        expect(near(session.asrSeconds, 3.5) && near(session.totalSeconds, 3.5), "Failed ASR must retain elapsed time")
        expect(failed.asrStatus == "失败" && failed.polishStatus == "未执行" && failed.polishDuration == "—",
               "Failed ASR must distinguish unexecuted LLM from success or disabled")
        expect(deliveries == beforeFailure, "Failed ASR must not deliver a result")

        service.asr = { clock.advance(0.01); return "  " }
        let beforeEmpty = service.polishCalls
        session.submitAudio(Data([1]))
        try await wait { session.phase == .ready }
        let empty = TimingPresentation(session: session)!
        expect(empty.asrDuration == "<0.1s" && empty.asrStatus == "未识别到文字", "Fast requests must not round to zero")
        expect(empty.polishStatus == "未执行" && service.polishCalls == beforeEmpty, "Empty ASR must skip LLM")

        var staleASR: CheckedContinuation<String, Error>?
        service.asr = { try await withCheckedThrowingContinuation { staleASR = $0 } }
        session.submitAudio(Data([1]))
        try await wait { staleASR != nil }
        session.cancel()
        expect(session.asrSeconds == nil && session.totalSeconds == nil && TimingPresentation(session: session) == nil,
               "Cancellation must clear timings and their presentation")
        session.polishEnabled = false
        service.asr = { clock.advance(2.5); return "新原文" }
        session.submitAudio(Data([2]))
        try await wait { session.phase == .ready }
        staleASR?.resume(throwing: TimingFailure())
        try await Task.sleep(nanoseconds: 10_000_000)
        expect(session.phase == .ready && near(session.asrSeconds, 2.5) && near(session.totalSeconds, 2.5),
               "A stale ASR failure must not overwrite a newer round's timings")

        var staleLLM: CheckedContinuation<String, Error>?
        session.polishEnabled = true
        service.llm = { try await withCheckedThrowingContinuation { staleLLM = $0 } }
        session.submitAudio(Data([1]))
        try await wait { staleLLM != nil }
        expect(TimingPresentation(session: session) == nil, "Do not label an unfinished round as complete")
        session.cancel()
        session.polishEnabled = false
        session.submitAudio(Data([2]))
        try await wait { session.phase == .ready }
        staleLLM?.resume(returning: "迟到的整理")
        try await Task.sleep(nanoseconds: 10_000_000)
        expect(session.polishSeconds == nil && near(session.totalSeconds, 2.5) && session.resultText == "新原文",
               "Stale LLM completion must not contaminate the current metrics or text")

        session.reportInputFailure("打开文件失败")
        expect(TimingPresentation(session: session) == nil, "No request means no invented zero timings")
        print("PASS: simulated-clock timing boundaries, failure/fallback/skip labels, reset and stale-result isolation")
    }
}
