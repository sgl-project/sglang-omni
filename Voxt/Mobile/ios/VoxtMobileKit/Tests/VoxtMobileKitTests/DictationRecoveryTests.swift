import Foundation
import Testing
@testable import VoxtMobileKit

/// Captures that grow while recording, like the microphone's, so autosave has something to save.
@MainActor
final class GrowingCapture: AudioCapturing {
    var captured: [Float] = []
    var capturing = false
    func startSession() throws {}
    func endSession() { capturing = false }
    func beginCapture() { captured = []; capturing = true }
    func endCapture() -> [Float] { capturing = false; return captured }
    func capturedSamples(from offset: Int) -> [Float] { offset < captured.count ? Array(captured[offset...]) : [] }
    func speak(_ seconds: Double) { captured += Array(repeating: 0.1, count: Int(seconds * 16_000)) }
}

/// Never lose a recording: audio is on disk while recording, stays there
/// until the Mac's text arrives, and can be sent again from the keyboard or from History.
@MainActor
@Suite struct DictationRecoveryTests {
    @MainActor private struct Rig {
        let controller: DictationController
        let channel: DictationChannel
        let capture: GrowingCapture
        let transcriber: FakeTranscriber
        let store: RecordingStore
        let directory: URL
        let defaults: UserDefaults
        let suite: String

        func relaunch() -> (DictationController, FakeTranscriber) {
            let transcriber = FakeTranscriber()
            let controller = DictationController(channel: channel, capture: GrowingCapture(), transcriber: transcriber,
                                                 recordings: RecordingStore(directory: directory))
            return (controller, transcriber)
        }

        func cleanUp() {
            try? FileManager.default.removeItem(at: directory)
            defaults.removePersistentDomain(forName: suite)
        }
    }

    private func make() -> Rig {
        let suite = "recovery.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        let channel = DictationChannel(defaults: defaults, notifier: .none)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("recovery-\(UUID().uuidString)", isDirectory: true)
        let store = RecordingStore(directory: directory)
        let capture = GrowingCapture()
        let transcriber = FakeTranscriber()
        let controller = DictationController(channel: channel, capture: capture, transcriber: transcriber, recordings: store)
        return Rig(controller: controller, channel: channel, capture: capture, transcriber: transcriber,
                   store: store, directory: directory, defaults: defaults, suite: suite)
    }

    private func savedSeconds(_ store: RecordingStore) -> [TimeInterval] {
        store.all.map(\.duration)
    }

    @Test func audioReachesDiskEveryFewSecondsWhileRecording() throws {
        let rig = make()
        defer { rig.cleanUp() }
        let start = Date()
        try rig.controller.start()
        rig.capture.speak(3)
        rig.controller.tick(now: start.addingTimeInterval(rig.controller.autosaveInterval + 0.1))
        #expect(savedSeconds(rig.store) == [3])
        #expect(rig.store.all.first?.status == .recording)
        rig.capture.speak(2)
        rig.controller.tick(now: start.addingTimeInterval(2 * rig.controller.autosaveInterval + 0.2))
        #expect(savedSeconds(rig.store) == [5])
    }

    @Test func aRecordingIsDeletedOnceItsTextArrives() async throws {
        let rig = make()
        defer { rig.cleanUp() }
        try rig.controller.start()
        rig.capture.speak(2)
        await rig.controller.stop()
        #expect(rig.controller.phase == .done)
        #expect(rig.store.all.isEmpty)
    }

    @Test func aFailedRecordingStaysUntilRetrySendsTheSameAudio() async throws {
        let rig = make()
        defer { rig.cleanUp() }
        rig.transcriber.result = .failure(VoxtClient.Error.transportUnavailable)
        try rig.controller.start()
        rig.capture.speak(2)
        await rig.controller.stop()
        #expect(rig.controller.phase == .failed)
        let saved = try #require(rig.store.unprocessed.first)
        #expect(saved.status == .failed)
        #expect(saved.duration == 2)
        #expect(saved.errorMessage == rig.controller.lastError)

        rig.transcriber.result = .success("Back online.")
        await rig.controller.retry()
        #expect(rig.controller.phase == .done)
        #expect(rig.transcriber.received.count == 2)
        #expect(rig.transcriber.received[0] == rig.transcriber.received[1])
        #expect(rig.store.all.isEmpty)
    }

    @Test func aRunningAppRetriesWithoutComingForward() async throws {
        let rig = make()
        defer { rig.cleanUp() }
        rig.transcriber.result = .failure(URLError(.timedOut))
        try rig.controller.start(requestID: "kb-3")
        rig.capture.speak(2)
        await rig.controller.stop()
        #expect(rig.channel.isSessionAlive())
        #expect(!rig.channel.state.retryOpensApp(sessionAlive: rig.channel.isSessionAlive()))
    }

    @Test func dismissingAFailureOrRecordingAgainKeepsTheFailedAudio() async throws {
        let rig = make()
        defer { rig.cleanUp() }
        rig.transcriber.result = .failure(VoxtClient.Error.transportUnavailable)
        try rig.controller.start()
        rig.capture.speak(2)
        await rig.controller.stop()
        rig.controller.cancel()
        #expect(rig.store.unprocessed.map(\.status) == [.failed])

        try rig.controller.start()
        rig.capture.speak(1)
        rig.transcriber.result = .success("Next one.")
        await rig.controller.stop()
        #expect(rig.controller.phase == .done)
        #expect(rig.store.unprocessed.map(\.duration) == [2])
    }

    @Test func aSessionEndingMidRecordingSavesTheAudio() throws {
        let rig = make()
        defer { rig.cleanUp() }
        try rig.controller.start()
        rig.capture.speak(2)
        rig.controller.endSession()
        let saved = try #require(rig.store.unprocessed.first)
        #expect(saved.status == .interrupted)
        #expect(saved.duration == 2)
    }

    @Test func aSessionEndingMidUploadSavesTheAudio() async throws {
        let rig = make()
        defer { rig.cleanUp() }
        rig.transcriber.holdsResponse = true
        try rig.controller.start()
        rig.capture.speak(2)
        let stopping = Task { await rig.controller.stop() }
        while rig.transcriber.gate == nil { await Task.yield() }
        #expect(rig.store.all.map(\.status) == [.processing])
        rig.controller.endSession()
        #expect(rig.store.unprocessed.map(\.status) == [.interrupted])
        rig.transcriber.gate?.resume()
        await stopping.value
        #expect(rig.store.unprocessed.map(\.status) == [.interrupted])
    }

    @Test func cancelKeepsOnlyRecordingsLongEnoughToMatter() async throws {
        let rig = make()
        defer { rig.cleanUp() }
        try rig.controller.start()
        rig.capture.speak(DictationController.keepCancelledAfter - 1)
        rig.controller.cancel()
        #expect(rig.store.all.isEmpty)

        try rig.controller.start()
        rig.capture.speak(DictationController.keepCancelledAfter + 2)
        rig.controller.cancel()
        let saved = try #require(rig.store.unprocessed.first)
        #expect(saved.status == .cancelled)
        #expect(saved.duration == DictationController.keepCancelledAfter + 2)
        #expect(rig.store.all.count == 1)
    }

    @Test func tapsAndSilenceAreNotKept() async throws {
        let rig = make()
        defer { rig.cleanUp() }
        try rig.controller.start()
        rig.capture.speak(0.1)
        await rig.controller.stop()
        try rig.controller.start()
        rig.capture.captured = Array(repeating: 0, count: 32_000)
        await rig.controller.stop()
        #expect(rig.controller.phase == .failed)
        #expect(rig.store.all.isEmpty)
    }

    @Test func afterARelaunchTheKeyboardCanRetryTheLastRecording() async throws {
        let rig = make()
        defer { rig.cleanUp() }
        rig.transcriber.result = .failure(VoxtClient.Error.transportUnavailable)
        try rig.controller.start(requestID: "kb-7")
        rig.capture.speak(2)
        await rig.controller.stop()

        let (controller, transcriber) = rig.relaunch()
        controller.restoreUnsentRecording()
        #expect(controller.phase == .failed)
        #expect(controller.canRetry)
        #expect(rig.channel.state.canRetry == true)
        #expect(rig.channel.state.requestID == "kb-7")
        #expect(rig.channel.state.errorMessage == DictationController.savedRecordingMessage)
        // The relaunched app has no microphone session and iOS suspends it in the background,
        // so the keyboard's Retry has to bring Voxt forward to send it.
        #expect(!rig.channel.isSessionAlive())
        #expect(rig.channel.state.retryOpensApp(sessionAlive: rig.channel.isSessionAlive()))

        transcriber.result = .success("Sent after all.")
        await controller.retry()
        #expect(controller.phase == .done)
        #expect(rig.channel.state.text == "Sent after all.")
        #expect(rig.store.all.isEmpty)
    }

    @Test func aResendAfterARelaunchKeepsTheKeyboardFromCallingItStalled() async throws {
        // No microphone session runs then, so only the request itself keeps the heartbeat going.
        let rig = make()
        defer { rig.cleanUp() }
        rig.transcriber.result = .failure(VoxtClient.Error.transportUnavailable)
        try rig.controller.start(requestID: "kb-5")
        rig.capture.speak(2)
        await rig.controller.stop()

        let (controller, transcriber) = rig.relaunch()
        controller.restoreUnsentRecording()
        transcriber.holdsResponse = true
        let sending = Task { await controller.retry() }
        while transcriber.gate == nil { await Task.yield() }
        // The keyboard checks as soon as it sees "processing", before the first tick.
        #expect(!rig.channel.isStalled())
        let later = Date().addingTimeInterval(DictationChannel.stalledAfter + 5)
        controller.tick(now: later)
        #expect(!rig.channel.isStalled(now: later))
        transcriber.gate?.resume()
        await sending.value
        #expect(controller.phase == .done)
    }

    @Test func anInAppRecordingWaitsInHistoryInsteadOfTheKeyboard() async throws {
        // No keyboard request owns it, so the keyboard could never insert its text.
        let rig = make()
        defer { rig.cleanUp() }
        rig.transcriber.result = .failure(VoxtClient.Error.transportUnavailable)
        try rig.controller.start()
        rig.capture.speak(2)
        await rig.controller.stop()

        let (controller, _) = rig.relaunch()
        controller.restoreUnsentRecording()
        #expect(controller.phase == .idle)
        #expect(rig.store.unprocessed.count == 1)
    }

    @Test func anOldRecordingWaitsInHistoryInsteadOfTheKeyboard() async throws {
        let rig = make()
        defer { rig.cleanUp() }
        let then = Date()
        rig.controller.now = { then }
        try rig.controller.start(requestID: "kb-old")
        rig.capture.speak(2)
        rig.controller.endSession()

        let (controller, _) = rig.relaunch()
        controller.now = { then.addingTimeInterval(DictationController.keyboardRetryWindow + 60) }
        controller.restoreUnsentRecording()
        #expect(controller.phase == .idle)
        #expect(rig.store.unprocessed.count == 1)
    }

    @Test func historyCanSendASavedRecordingAgainWithoutTheKeyboardInsertingIt() async throws {
        let rig = make()
        defer { rig.cleanUp() }
        try rig.controller.start(requestID: "kb-9")
        rig.capture.speak(2)
        rig.controller.endSession()
        let id = try #require(rig.store.unprocessed.first?.id)

        let (controller, transcriber) = rig.relaunch()
        transcriber.result = .success("From History.")
        await controller.reprocess(id)
        #expect(controller.phase == .done)
        #expect(controller.lastResult == "From History.")
        #expect(rig.channel.state.requestID == nil)
        #expect(transcriber.received.count == 1)
        #expect(rig.store.all.isEmpty)
    }

    @Test func historyResendThatFailsAgainKeepsTheRecording() async throws {
        let rig = make()
        defer { rig.cleanUp() }
        try rig.controller.start()
        rig.capture.speak(2)
        rig.controller.endSession()
        let id = try #require(rig.store.unprocessed.first?.id)

        let (controller, transcriber) = rig.relaunch()
        transcriber.result = .failure(VoxtClient.Error.transportUnavailable)
        await controller.reprocess(id)
        #expect(controller.phase == .failed)
        #expect(rig.store.unprocessed.map(\.status) == [.failed])
    }
}
