import Combine
import Foundation

@MainActor
private final class Recorder: AudioRecording {
    func start() async throws { }
    func stop() throws -> Data { Data([1]) }
    func cancel() { }
}

@MainActor
private final class Service: SpeechServing {
    var reply: CheckedContinuation<String, Error>?
    func transcribe(wav: Data) async throws -> String {
        try await withCheckedThrowingContinuation { reply = $0 }
    }
    func polish(text: String) async throws -> String { text }
    func finish(_ text: String) { reply?.resume(returning: text); reply = nil }
}

@MainActor
private final class Corrector: TextCorrecting {
    var reply: CheckedContinuation<CorrectionResult, Error>?
    func correct(original: String, instruction: String, personalBackground: String) async throws -> CorrectionResult {
        try await withCheckedThrowingContinuation { reply = $0 }
    }
    func finish() {
        reply?.resume(returning: CorrectionResult(status: .ok, correctedText: "新版"))
        reply = nil
    }
}

@MainActor
private func wait(_ predicate: () -> Bool) async throws {
    let deadline = ProcessInfo.processInfo.systemUptime + 3
    while !predicate(), ProcessInfo.processInfo.systemUptime < deadline {
        try await Task.sleep(nanoseconds: 1_000_000)
    }
    precondition(predicate(), "Timed out")
}

@main
private enum CorrectionFeedbackTests {
    @MainActor
    static func main() async throws {
        let service = Service(), correctionService = Service(), corrector = Corrector()
        let ordinary = DictationSession(recorder: Recorder(), service: service)
        let correction = CorrectionSession(recorder: Recorder(), service: correctionService, makeCorrector: { corrector })
        let insertion = DictationInsertion()
        // Match ClientState subscription order, including clearing correction on a new round.
        let reset = ordinary.$phase.removeDuplicates().sink { phase in
            if phase == .authorizing || phase == .recognizing { correction.clear() }
        }
        let feedback = ResultWindowFeedback(session: ordinary, insertion: insertion, correction: correction,
                                             holdDuration: 0.06, fadeDuration: 0.08, attentionHoldDuration: 0.06)
        var values: [Double] = []
        let observation = feedback.$opacity.sink { values.append($0) }
        defer { reset.cancel(); observation.cancel(); ordinary.cancel(); correction.clear() }
        ordinary.submitAudio(Data([1]))
        try await wait { service.reply != nil }
        service.finish("旧版")
        try await wait { ordinary.phase == .ready }
        correction.remember("旧版")
        correction.toggleRecording(personalBackground: "")
        try await wait { correction.phase == .recording }
        correction.toggleRecording(personalBackground: "")
        try await wait { correctionService.reply != nil }
        correctionService.finish("把旧版改成新版")
        try await wait { corrector.reply != nil }
        try await Task.sleep(nanoseconds: 220_000_000)
        precondition(feedback.opacity == 1, "The old dictation fade must not hide an active correction")
        corrector.finish()
        try await wait { correction.phase == .ready }
        values = []
        ordinary.submitAudio(Data([1]))
        try await wait { service.reply != nil }
        precondition(!values.contains(0), "Starting a new ordinary round must not briefly close the result window")
        try await Task.sleep(nanoseconds: 220_000_000)
        precondition(feedback.opacity == 1)
        service.finish("新一轮")
        try await wait { feedback.opacity == 0 }
        feedback.show()
        try await Task.sleep(nanoseconds: 220_000_000)
        precondition(feedback.opacity == 1, "Explicit reopening must remain pinned")
        correction.remember("新一轮")
        correction.toggleRecording(personalBackground: "")
        try await wait { correction.phase == .recording }
        correction.cancel()
        try await wait { feedback.opacity == 0 }
        print("PASS: ordinary/correction fade isolation, no close on new round, manual reopening and cancellation")
    }
}
