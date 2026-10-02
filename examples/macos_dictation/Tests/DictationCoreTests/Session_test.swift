import Foundation
import XCTest
@testable import DictationCore

@MainActor
final class SessionTests: XCTestCase {
    func testRawModeSkipsPolishAndReturnsTranscript() async throws {
        let audio = FakeRecorder()
        let service = FakeService()
        let model = DictationSession(recorder: audio, service: service)
        model.toggleRecording()
        await waitUntil { model.phase == .recording }
        model.toggleRecording()
        await waitUntil { model.phase == .ready }
        XCTAssertEqual(model.rawText, "这个 PR 不要合并。")
        XCTAssertEqual(model.resultText, model.rawText)
        XCTAssertEqual(service.polishCalls, 0)
        XCTAssertEqual(service.receivedAudio, Data([1, 2, 3]))
        XCTAssertNotNil(model.asrSeconds)
        XCTAssertNotNil(model.totalSeconds)
    }

    func testPolishReceivesASRTextAndKeepsBothResults() async throws {
        let service = FakeService()
        let model = DictationSession(recorder: FakeRecorder(), service: service)
        model.polishEnabled = true
        model.toggleRecording()
        await waitUntil { model.phase == .recording }
        model.toggleRecording()
        await waitUntil { model.phase == .ready }
        XCTAssertEqual(service.polishInput, model.rawText)
        XCTAssertEqual(model.resultText, "这个 PR 先不要合并。")
        XCTAssertNotNil(model.polishSeconds)
    }

    func testPolishFailurePreservesASRWithVisibleWarning() async {
        let service = FakeService()
        service.polishHandler = { _ in throw DemoError("润色服务未启动") }
        let model = DictationSession(recorder: FakeRecorder(), service: service)
        model.polishEnabled = true
        model.toggleRecording()
        await waitUntil { model.phase == .recording }
        model.toggleRecording()
        await waitUntil { model.phase == .ready }
        XCTAssertEqual(model.resultText, model.rawText)
        XCTAssertTrue(model.notice.contains("原文"))
        XCTAssertTrue(model.notice.contains("润色服务未启动"))
    }

    func testEmptyASRDoesNotCallPolish() async {
        let service = FakeService()
        service.transcribeHandler = { _ in "   \n" }
        let model = DictationSession(recorder: FakeRecorder(), service: service)
        model.polishEnabled = true
        model.toggleRecording()
        await waitUntil { model.phase == .recording }
        model.toggleRecording()
        await waitUntil { model.phase == .ready }
        XCTAssertTrue(model.resultText.isEmpty)
        XCTAssertEqual(service.polishCalls, 0)
        XCTAssertTrue(model.notice.contains("没有识别"))
    }

    func testPermissionFailureReturnsActionableError() async {
        let audio = FakeRecorder()
        audio.startHandler = { throw DemoError("请在系统设置中允许麦克风权限") }
        let model = DictationSession(recorder: audio, service: FakeService())
        model.toggleRecording()
        await waitUntil { model.phase == .failed }
        XCTAssertTrue(model.notice.contains("麦克风"))
        XCTAssertFalse(model.isBusy)
    }

    func testRecordingStopFailureDoesNotSubmitAudio() async {
        let audio = FakeRecorder()
        audio.stopError = DemoError("录音为空")
        let service = FakeService()
        let model = DictationSession(recorder: audio, service: service)
        model.toggleRecording()
        await waitUntil { model.phase == .recording }
        model.toggleRecording()
        XCTAssertEqual(model.phase, .failed)
        XCTAssertEqual(service.transcribeCalls, 0)
    }

    func testASRFailureDoesNotProduceAResult() async {
        let service = FakeService()
        service.transcribeHandler = { _ in throw DemoError("ASR 连接失败") }
        let model = DictationSession(recorder: FakeRecorder(), service: service)
        model.toggleRecording()
        await waitUntil { model.phase == .recording }
        model.toggleRecording()
        await waitUntil { model.phase == .failed }
        XCTAssertTrue(model.notice.contains("ASR 连接失败"))
        XCTAssertTrue(model.resultText.isEmpty)
    }

    func testCancelDropsLateASRResponseEvenIfTransportIgnoresCancellation() async {
        let service = FakeService()
        var response: CheckedContinuation<String, Error>?
        service.transcribeHandler = { _ in
            try await withCheckedThrowingContinuation { response = $0 }
        }
        let model = DictationSession(recorder: FakeRecorder(), service: service)
        model.polishEnabled = true
        model.toggleRecording()
        await waitUntil { model.phase == .recording }
        model.toggleRecording()
        await waitUntil { response != nil }
        model.cancel()
        response?.resume(returning: "迟到的旧结果")
        await settle()
        XCTAssertEqual(model.phase, .idle)
        XCTAssertTrue(model.rawText.isEmpty)
        XCTAssertTrue(model.resultText.isEmpty)
        XCTAssertEqual(service.polishCalls, 0)
    }

    func testCanceledPolishCannotOverwriteANewerRound() async {
        let service = FakeService()
        var oldResponse: CheckedContinuation<String, Error>?
        service.polishHandler = { _ in
            try await withCheckedThrowingContinuation { oldResponse = $0 }
        }
        let model = DictationSession(recorder: FakeRecorder(), service: service)
        model.polishEnabled = true
        model.toggleRecording()
        await waitUntil { model.phase == .recording }
        model.toggleRecording()
        await waitUntil { oldResponse != nil }
        model.cancel()
        model.polishEnabled = false
        service.transcribeHandler = { _ in "新一轮的原文" }
        model.toggleRecording()
        await waitUntil { model.phase == .recording }
        model.toggleRecording()
        await waitUntil { model.phase == .ready }
        oldResponse?.resume(returning: "旧一轮的润色")
        await settle()
        XCTAssertEqual(model.resultText, "新一轮的原文")
        XCTAssertEqual(model.phase, .ready)
    }

    func testCancelWhileAwaitingMicrophonePermissionNeverStartsSession() async {
        let audio = FakeRecorder()
        var permission: CheckedContinuation<Void, Error>?
        audio.startHandler = {
            try await withCheckedThrowingContinuation { permission = $0 }
        }
        let model = DictationSession(recorder: audio, service: FakeService())
        model.toggleRecording()
        await waitUntil { permission != nil }
        model.cancel()
        permission?.resume()
        await settle()
        XCTAssertEqual(model.phase, .idle)
        XCTAssertNil(model.recordingStartedAt)
    }

    func testBusyHotkeyDoesNotCreateASecondRequest() async {
        let service = FakeService()
        var response: CheckedContinuation<String, Error>?
        service.transcribeHandler = { _ in
            try await withCheckedThrowingContinuation { response = $0 }
        }
        let model = DictationSession(recorder: FakeRecorder(), service: service)
        model.toggleRecording()
        await waitUntil { model.phase == .recording }
        model.toggleRecording()
        await waitUntil { response != nil }
        model.toggleRecording()
        model.toggleRecording()
        XCTAssertEqual(service.transcribeCalls, 1)
        response?.resume(returning: "完成")
        await waitUntil { model.phase == .ready }
    }

    func testRecordingLimitAutomaticallyStopsAndProcesses() async {
        let service = FakeService()
        let model = DictationSession(
            recorder: FakeRecorder(), service: service, recordingLimit: 0.02
        )
        model.toggleRecording()
        await waitUntil { model.phase == .ready }
        XCTAssertEqual(service.transcribeCalls, 1)
    }

    private func waitUntil(
        file: StaticString = #filePath, line: UInt = #line,
        _ predicate: () -> Bool
    ) async {
        let deadline = Date().addingTimeInterval(2)
        while !predicate(), Date() < deadline {
            try? await Task.sleep(nanoseconds: 1_000_000)
        }
        XCTAssertTrue(predicate(), "等待状态超时", file: file, line: line)
    }

    private func settle() async {
        for _ in 0..<10 { await Task.yield() }
        try? await Task.sleep(nanoseconds: 5_000_000)
    }
}

struct DemoError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

@MainActor
final class FakeRecorder: AudioRecording {
    var startHandler: () async throws -> Void = {}
    var stopError: Error?
    var cancelCount = 0
    func start() async throws { try await startHandler() }
    func stop() throws -> Data {
        if let stopError { throw stopError }
        return Data([1, 2, 3])
    }
    func cancel() { cancelCount += 1 }
}

@MainActor
final class FakeService: SpeechServing {
    var transcribeHandler: (Data) async throws -> String = { _ in "这个 PR 不要合并。" }
    var polishHandler: (String) async throws -> String = { _ in "这个 PR 先不要合并。" }
    var transcribeCalls = 0
    var polishCalls = 0
    var receivedAudio: Data?
    var polishInput: String?
    func transcribe(wav: Data) async throws -> String {
        transcribeCalls += 1
        receivedAudio = wav
        return try await transcribeHandler(wav)
    }
    func polish(text: String) async throws -> String {
        polishCalls += 1
        polishInput = text
        return try await polishHandler(text)
    }
}
