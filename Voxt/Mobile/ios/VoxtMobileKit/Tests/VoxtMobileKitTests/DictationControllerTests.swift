import Foundation
import Testing
@testable import VoxtMobileKit

@MainActor
final class FakeCapture: AudioCapturing {
    var sessionRunning = false
    var capturing = false
    var samplesToReturn: [Float] = Array(repeating: 0.1, count: 16_000)
    var startSessionError: Error?

    func startSession() throws {
        if let startSessionError { throw startSessionError }
        sessionRunning = true
    }
    func endSession() { sessionRunning = false; capturing = false }
    func beginCapture() { capturing = true }
    func endCapture() -> [Float] { capturing = false; return samplesToReturn }
}

@MainActor
final class FakeTranscriber: Transcribing {
    var historyID: UUID?
    var received: [Data] = []
    var optionsReceived: [CompositionOptions] = []
    var result: Result<String, Error> = .success("Hello from the Mac.")
    var gate: CheckedContinuation<Void, Never>?
    var holdsResponse = false

    func transcribe(wav: Data) async throws -> (text: String, rawText: String) {
        received.append(wav)
        if holdsResponse { await withCheckedContinuation { gate = $0 } }
        let text = try result.get()
        return (text, text)
    }

    func transcribe(wav: Data, options: CompositionOptions) async throws -> (text: String, rawText: String) {
        optionsReceived.append(options)
        return try await transcribe(wav: wav)
    }

    var timedResult: TranscriptionResult?
    var onTranscribe: (() -> Void)?

    func transcribeResult(wav: Data, options: CompositionOptions) async throws -> TranscriptionResult {
        onTranscribe?()
        if let timedResult { optionsReceived.append(options); received.append(wav); return timedResult }
        let result = try await transcribe(wav: wav, options: options)
        return TranscriptionResult(id: historyID, text: result.text, rawText: result.rawText)
    }
}

@MainActor
@Suite struct DictationControllerTests {
    @Test func notAppliedEditPublishesErrorWithoutKeyboardInsertion() async throws {
        let (controller, channel, _, transcriber) = make()
        try controller.start()
        await controller.stop()
        transcriber.result = .failure(VoxtClient.Error.editNotApplied)
        let selection = "The server has a cash limit."
        try controller.start(options: .init(mode: .edit, selectedText: selection), requestID: "edit-not-applied")
        await controller.stop()
        #expect(controller.phase == .failed)
        #expect(controller.lastError == "Couldn't apply that edit")
        #expect(channel.state.errorMessage == "Couldn't apply that edit")
        #expect(channel.textToInsert(lastInsertedID: nil) == nil)
        #expect(channel.state.text == nil)
        #expect(transcriber.optionsReceived.last?.selectedText == selection)
        #expect(controller.lastResult == nil)
    }

    @Test func debugRecordRetainsRawAndFinalForLanguageDiagnosis() async throws {
        let name = "diagnostic.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let channel = DictationChannel(defaults: defaults, notifier: .none)
        let controller = DictationController(channel: channel, capture: FakeCapture(), transcriber: LanguageDiagnosticTranscriber())
        try controller.start()
        await controller.stop()
        let data = try #require(defaults.data(forKey: "dictation.debugLastResponse"))
        let record = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(record["rawText"] as? String == "我试试这个效果怎么样。")
        #expect(record["text"] as? String == "I will try it.")
        #expect(record["mode"] as? String == "dictate")
        #expect(record["asrMs"] as? Int == 700)
        #expect(record["enhanceMs"] as? Int == 5900)
    }

    @Test func keepsServerHistoryIdentitySeparateFromKeyboardInsertionIdentity() async throws {
        let (controller, channel, _, transcriber) = make()
        let id = UUID()
        transcriber.historyID = id
        try controller.start()
        await controller.stop()
        #expect(controller.lastResultID == id)
        #expect(channel.state.resultID != id.uuidString)
        try controller.start()
        #expect(controller.lastResultID == nil)
        controller.cancel()
    }
    private func make() -> (DictationController, DictationChannel, FakeCapture, FakeTranscriber) {
        let name = "test.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        let channel = DictationChannel(defaults: defaults, notifier: .none)
        let capture = FakeCapture()
        let transcriber = FakeTranscriber()
        let controller = DictationController(channel: channel, capture: capture, transcriber: transcriber)
        return (controller, channel, capture, transcriber)
    }

    @Test func startOpensSessionAndRecords() async throws {
        let (controller, channel, capture, _) = make()
        try controller.start()
        #expect(capture.sessionRunning)
        #expect(capture.capturing)
        #expect(channel.state.phase == .recording)
        #expect(channel.isSessionAlive())
    }

    @Test func stopUploadsWAVAndPublishesText() async throws {
        let (controller, channel, capture, transcriber) = make()
        try controller.start()
        await controller.stop()

        #expect(capture.capturing == false)
        #expect(capture.sessionRunning, "session stays open for the next dictation")
        #expect(transcriber.received.count == 1)
        #expect(String(decoding: transcriber.received[0].prefix(4), as: UTF8.self) == "RIFF")
        #expect(transcriber.received[0].count == 44 + 16_000 * 2)
        #expect(channel.state.phase == .done)
        #expect(channel.state.text == "Hello from the Mac.")
        #expect(controller.lastResult == "Hello from the Mac.")
    }

    @Test func tooShortRecordingIsDroppedWithoutUpload() async throws {
        let (controller, channel, capture, transcriber) = make()
        capture.samplesToReturn = Array(repeating: 0.1, count: 1_000) // 62 ms
        try controller.start()
        await controller.stop()
        #expect(transcriber.received.isEmpty)
        #expect(channel.state.phase == .idle)
    }

    @Test func uploadFailurePublishesError() async throws {
        let (controller, channel, _, transcriber) = make()
        transcriber.result = .failure(VoxtClient.Error.server(status: 503, message: "Voxt is recording on the Mac."))
        try controller.start()
        await controller.stop()
        #expect(channel.state.phase == .failed)
        #expect(channel.state.errorMessage == "Voxt is recording on the Mac.")
    }

    @Test func retryUsesTheSameRecordingIntentAndOrigin() async throws {
        let (controller, channel, _, transcriber) = make()
        transcriber.result = .failure(URLError(.notConnectedToInternet))
        let options = CompositionOptions(mode: .translate, targetLanguage: "ja")
        try controller.start(options: options, requestID: "keyboard-request")
        await controller.stop()
        #expect(controller.canRetry)
        #expect(channel.state.canRetry == true)
        transcriber.result = .success("明日会議をします。")
        await controller.retry()
        #expect(channel.state.text == "明日会議をします。")
        #expect(channel.state.requestID == "keyboard-request")
        #expect(transcriber.received.count == 2)
        #expect(transcriber.received[0] == transcriber.received[1])
        #expect(transcriber.optionsReceived == [options, options])
        #expect(!controller.canRetry)
    }

    /// The tunnel can be down while the app is in the background. The recording never left,
    /// so the keyboard opens Voxt once, where it reconnects and resends.
    @Test func aTunnelFailureAsksTheKeyboardToReconnectThroughTheAppOnce() async throws {
        let (controller, channel, _, transcriber) = make()
        transcriber.result = .failure(VoxtClient.Error.transportUnavailable)
        try controller.start(requestID: "keyboard-request")
        await controller.stop()
        #expect(channel.state.reconnectInApp == true)
        #expect(channel.state.opensAppToReconnect(ownRequestID: "keyboard-request", alreadyOpened: nil))
        #expect(!channel.state.opensAppToReconnect(ownRequestID: "keyboard-request", alreadyOpened: "keyboard-request"), "once per request")
        #expect(!channel.state.opensAppToReconnect(ownRequestID: "another-request", alreadyOpened: nil), "only the keyboard that asked")

        let (other, otherChannel, _, otherTranscriber) = make()
        otherTranscriber.result = .failure(VoxtClient.Error.unauthorized)
        try other.start(requestID: "keyboard-request")
        await other.stop()
        #expect(otherChannel.state.reconnectInApp != true, "reconnecting doesn't fix a rejected pairing")
        #expect(!otherChannel.state.opensAppToReconnect(ownRequestID: "keyboard-request", alreadyOpened: nil))
    }

    @Test func cancelledFailureCannotBeRetried() async throws {
        let (controller, _, _, transcriber) = make()
        transcriber.result = .failure(URLError(.timedOut))
        try controller.start()
        await controller.stop()
        controller.cancel()
        await controller.retry()
        #expect(transcriber.received.count == 1)
        #expect(!controller.canRetry)
    }

    @Test func cancelDiscardsRecording() async throws {
        let (controller, channel, capture, transcriber) = make()
        try controller.start()
        controller.cancel()
        #expect(capture.capturing == false)
        #expect(transcriber.received.isEmpty)
        #expect(channel.state.phase == .idle)
    }

    @Test func cancelProcessingDiscardsEvenAnUncooperativeLateResponse() async throws {
        let (controller, channel, _, transcriber) = make()
        transcriber.holdsResponse = true
        try controller.start()
        let stopping = Task { await controller.stop() }
        while transcriber.gate == nil { await Task.yield() }
        controller.cancel()
        #expect(channel.state.phase == .idle)
        transcriber.gate?.resume()
        await stopping.value
        #expect(channel.state.phase == .idle)
        #expect(channel.state.text == nil)
        #expect(controller.lastResult == nil)
    }

    @Test func startIsIgnoredWhileProcessing() async throws {
        let (controller, channel, _, transcriber) = make()
        transcriber.holdsResponse = true
        try controller.start()
        let stopping = Task { await controller.stop() }
        while transcriber.gate == nil { await Task.yield() }
        #expect(channel.state.phase == .processing)

        try controller.start()
        #expect(channel.state.phase == .processing)

        transcriber.gate?.resume()
        await stopping.value
        #expect(channel.state.phase == .done)
    }

    @Test func commandsFromKeyboardDriveTheController() async throws {
        let (controller, channel, _, transcriber) = make()
        let keyboard = channel
        keyboard.send(.start)
        await controller.handlePendingCommand()
        #expect(channel.state.phase == .recording)

        keyboard.send(.stop)
        await controller.handlePendingCommand()
        #expect(transcriber.received.count == 1)
        #expect(channel.state.phase == .done)

        // Re-reading the same command does nothing.
        await controller.handlePendingCommand()
        #expect(transcriber.received.count == 1)
    }

    @Test func idleTimeoutEndsSession() async throws {
        let (controller, channel, capture, _) = make()
        try controller.start()
        await controller.stop()
        let later = Date().addingTimeInterval(DictationController.idleSessionTimeout + 1)
        controller.tick(now: later)
        #expect(capture.sessionRunning == false)
        #expect(channel.isSessionAlive(now: later) == false)
    }

    @Test func disableAfterSettingControlsWhenTheIdleMicrophoneCloses() async throws {
        for (readiness, stillOpen, closed) in [
            (MicrophoneReadiness.immediately, nil, 1.0),
            (.fiveMinutes, 4 * 60.0, 6 * 60.0),
            (.fifteenMinutes, 14 * 60.0, 16 * 60.0),
            (.oneHour, 59 * 60.0, 61 * 60.0),
            (.never, 24 * 3600.0, nil),
        ] as [(MicrophoneReadiness, TimeInterval?, TimeInterval?)] {
            let (controller, _, capture, _) = make()
            controller.idleTimeout = readiness.timeout
            try controller.start()
            await controller.stop()
            let finished = Date()
            if let stillOpen {
                controller.tick(now: finished.addingTimeInterval(stillOpen))
                #expect(capture.sessionRunning, "\(readiness) keeps the microphone at \(stillOpen) s")
            }
            if let closed {
                controller.tick(now: finished.addingTimeInterval(closed))
                #expect(capture.sessionRunning == false, "\(readiness) closes the microphone by \(closed) s")
            }
        }
    }

    @Test func disableAfterDefaultsToFiveMinutesAndPersists() {
        let name = "test.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        #expect(MicrophoneReadiness.load(from: defaults) == .fiveMinutes)
        #expect(MicrophoneReadiness.fiveMinutes.timeout == DictationController.idleSessionTimeout)
        MicrophoneReadiness.never.save(to: defaults)
        #expect(MicrophoneReadiness.load(from: defaults) == .never)
        #expect(MicrophoneReadiness.never.timeout == nil)
        #expect(MicrophoneReadiness.immediately.timeout == 0)
        #expect(MicrophoneReadiness.allCases.map(\.rawValue) == ["immediately", "fiveMinutes", "fifteenMinutes", "oneHour", "never"])
    }

    /// Per-stage timing for one request: phone stages measured here, host
    /// stages copied from the response, insertion marked by the keyboard.
    @Test func silentRecordingIsNotUploaded() async throws {
        let (controller, channel, capture, transcriber) = make()
        capture.samplesToReturn = [Float](repeating: 0, count: 32_000)
        try controller.start(requestID: "silent")
        await controller.stop()
        #expect(transcriber.received.isEmpty, "no request for a recording without speech")
        #expect(channel.state.phase == .failed)
        #expect(channel.state.errorMessage == "No speech heard. Try again.")
        #expect(channel.state.canRetry != true)
    }

    @Test func dictationRecordsPerStageTimings() async throws {
        let (controller, channel, _, transcriber) = make()
        var clock = Date(timeIntervalSince1970: 1_000)
        controller.now = { clock }
        transcriber.timedResult = TranscriptionResult(text: "Hi", rawText: "Hi", asrMs: 400, enhanceMs: 1_500,
                                                      preflightMs: 120, requestMs: 2_300)
        transcriber.onTranscribe = { clock = clock.addingTimeInterval(2.5) }
        try controller.start(requestID: "timed")
        await controller.stop()
        let timings = try #require(channel.lastTimings)
        #expect(timings.requestID == "timed")
        #expect(timings.audioMs == 1_000)
        #expect(timings.preflightMs == 120)
        #expect(timings.requestMs == 2_300)
        #expect(timings.hostAsrMs == 400)
        #expect(timings.hostEnhanceMs == 1_500)
        #expect(timings.stopToResultMs == 2_500)
        #expect(timings.stopToInsertMs == nil)
        channel.markInserted(requestID: "other", at: clock.addingTimeInterval(1))
        #expect(channel.lastTimings?.stopToInsertMs == nil, "another request's insertion is ignored")
        channel.markInserted(requestID: "timed", at: clock.addingTimeInterval(0.3))
        #expect(channel.lastTimings?.stopToInsertMs == 2_800)
    }

    /// A "No speech detected." result can't be diagnosed without knowing how loud the
    /// recording was. Failed requests keep their audio numbers too.
    @Test func timingsRecordAudioLevelForSuccessFailureAndTheSpeechGate() async throws {
        let (controller, channel, capture, transcriber) = make()
        capture.samplesToReturn = Array(repeating: 0.1, count: 8_000) + Array(repeating: 0, count: 8_000)
        try controller.start(requestID: "loud")
        await controller.stop()
        let loud = try #require(channel.lastTimings)
        #expect(loud.peakDb == -20)
        #expect(loud.speechMs == 500)
        #expect(loud.error == nil)

        transcriber.result = .failure(VoxtClient.Error.server(status: 422, message: "No speech detected."))
        try controller.start(requestID: "empty-asr")
        await controller.stop()
        let failed = try #require(channel.lastTimings)
        #expect(failed.requestID == "empty-asr")
        #expect(failed.audioMs == 1_000)
        #expect(failed.error == "No speech detected.")

        capture.samplesToReturn = [Float](repeating: 0.0001, count: 16_000)
        try controller.start(requestID: "quiet")
        await controller.stop()
        let gated = try #require(channel.lastTimings)
        #expect(gated.requestID == "quiet")
        #expect(gated.peakDb == -80)
        #expect(gated.error == "No speech heard. Try again.")
        #expect(channel.recentTimings.map(\.requestID) == ["loud", "empty-asr", "quiet"])
    }

    /// Device check for a slow request (DEVICE-RECHECK #7): the result reaches the keyboard
    /// only after the delay, the same wait a slow Mac causes.
    @Test func debugResultDelayHoldsTheResultBeforePublishing() async throws {
        let (controller, channel, _, _) = make()
        controller.resultDelay = 0.3
        try controller.start(requestID: "slow")
        let started = Date()
        await controller.stop()
        #expect(Date().timeIntervalSince(started) >= 0.3)
        #expect(channel.state.phase == .done)
    }

    @Test func tickKeepsHeartbeatWhileSessionIsOpen() async throws {
        let (controller, channel, _, _) = make()
        try controller.start()
        let now = Date().addingTimeInterval(10)
        controller.tick(now: now)
        #expect(channel.isSessionAlive(now: now.addingTimeInterval(1)))
    }
    @Test func changingConnectionCancelsOldUploadAndClosesMicrophone() async throws {
        let (controller, channel, capture, transcriber) = make()
        transcriber.holdsResponse = true
        try controller.start(requestID: "old-server")
        let stopping = Task { await controller.stop() }
        while transcriber.gate == nil { await Task.yield() }
        controller.setTranscriber(nil)
        #expect(channel.state.phase == .idle)
        #expect(!capture.sessionRunning)
        transcriber.gate?.resume()
        await stopping.value
        #expect(controller.lastResult == nil)
        #expect(channel.textToInsert(lastInsertedID: nil) == nil)
    }

    // MARK: PiP readiness (armed) — microphone-off idle

    @Test func armingPublishesHeartbeatWithoutMicrophone() async throws {
        let (controller, channel, capture, _) = make()
        controller.setArmed(true)
        #expect(controller.isArmed)
        #expect(channel.isSessionAlive())
        #expect(!capture.sessionRunning)
        #expect(!controller.isSessionOpen)
        #expect(channel.state.phase == .idle)
    }

    @Test func armingAfterLegacyDictationReleasesIdleMicrophoneImmediately() async throws {
        let (controller, channel, capture, transcriber) = make()
        // Legacy path: no PiP, so the session stays open after stop.
        try controller.start()
        await controller.stop()
        #expect(capture.sessionRunning)
        #expect(channel.state.phase == .done)

        controller.setArmed(true)
        #expect(!capture.sessionRunning, "arming with PiP turns the idle microphone off")
        #expect(channel.isSessionAlive())
        #expect(channel.state.phase == .done)
        #expect(transcriber.received.count == 1)
    }

    @Test func armingDuringRecordingPreservesTheMicrophone() async throws {
        let (controller, channel, capture, _) = make()
        try controller.start()
        controller.setArmed(true)
        #expect(capture.sessionRunning, "an active recording is never interrupted")
        #expect(capture.capturing)
        #expect(channel.state.phase == .recording)

        await controller.stop()
        #expect(!capture.sessionRunning, "armed stop closes the mic right after")
        #expect(channel.isSessionAlive())
    }

    @Test func armingDuringProcessingReleasesTheMicrophone() async throws {
        let (controller, _, capture, transcriber) = make()
        transcriber.holdsResponse = true
        try controller.start()
        let stopping = Task { await controller.stop() }
        while transcriber.gate == nil { await Task.yield() }
        #expect(capture.sessionRunning, "legacy stop keeps the session open while processing")
        controller.setArmed(true)
        #expect(!capture.sessionRunning, "recording already ended; the mic is idle")
        transcriber.gate?.resume()
        await stopping.value
    }

    @Test func armedStartOpensMicrophoneOnDemand() async throws {
        let (controller, channel, capture, _) = make()
        controller.setArmed(true)
        try controller.start()
        #expect(capture.sessionRunning)
        #expect(capture.capturing)
        #expect(channel.state.phase == .recording)
    }

    @Test func armedStopClosesMicrophoneImmediatelyButKeepsHeartbeat() async throws {
        let (controller, channel, capture, transcriber) = make()
        controller.setArmed(true)
        try controller.start()
        await controller.stop()
        #expect(!capture.sessionRunning, "mic off as soon as the user stops speaking")
        #expect(channel.isSessionAlive(), "PiP readiness survives the mic teardown")
        #expect(channel.state.phase == .done)
        #expect(transcriber.received.count == 1)
    }

    @Test func armedTooShortRecordingClosesMicrophoneWithoutUpload() async throws {
        let (controller, channel, capture, transcriber) = make()
        capture.samplesToReturn = Array(repeating: 0.1, count: 1_000) // 62 ms
        controller.setArmed(true)
        try controller.start()
        await controller.stop()
        #expect(!capture.sessionRunning)
        #expect(channel.isSessionAlive())
        #expect(transcriber.received.isEmpty)
        #expect(channel.state.phase == .idle)
    }

    @Test func armedCancelClosesMicrophoneImmediatelyButKeepsHeartbeat() async throws {
        let (controller, channel, capture, _) = make()
        controller.setArmed(true)
        try controller.start()
        controller.cancel()
        #expect(!capture.sessionRunning)
        #expect(channel.isSessionAlive())
        #expect(channel.state.phase == .idle)
    }

    @Test func disarmWhileRecordingCancelsAndClearsHeartbeat() async throws {
        let (controller, channel, capture, _) = make()
        controller.setArmed(true)
        try controller.start()
        controller.setArmed(false)
        #expect(!controller.isArmed)
        #expect(!capture.sessionRunning)
        #expect(!channel.isSessionAlive())
        #expect(channel.state.phase == .idle)
    }

    @Test func disarmWhileProcessingDiscardsLateResponseAndClearsHeartbeat() async throws {
        let (controller, channel, _, transcriber) = make()
        transcriber.holdsResponse = true
        controller.setArmed(true)
        try controller.start()
        let stopping = Task { await controller.stop() }
        while transcriber.gate == nil { await Task.yield() }
        controller.setArmed(false)
        #expect(channel.state.phase == .idle)
        #expect(!channel.isSessionAlive())
        transcriber.gate?.resume()
        await stopping.value
        #expect(channel.state.phase == .idle)
        #expect(controller.lastResult == nil)
        #expect(channel.textToInsert(lastInsertedID: nil) == nil)
    }

    @Test func tickWhileArmedKeepsHeartbeatAndNeverOpensMicrophoneOrTimesOut() async throws {
        let (controller, channel, capture, _) = make()
        controller.setArmed(true)
        let later = Date().addingTimeInterval(DictationController.idleSessionTimeout * 3)
        controller.tick(now: later)
        #expect(channel.isSessionAlive(now: later))
        #expect(!capture.sessionRunning)
        #expect(controller.isArmed)
    }

    @Test func microphoneStartFailureWhileArmedStaysArmedForRetry() async throws {
        struct MicError: Error {}
        let (controller, channel, capture, _) = make()
        controller.setArmed(true)
        capture.startSessionError = MicError()
        channel.send(.start)
        await controller.handlePendingCommand()
        #expect(channel.state.phase == .failed)
        #expect(controller.isArmed, "the PiP window is still up; the user can retry")
        #expect(channel.isSessionAlive())
    }

    @Test func endSessionDisarmsReadiness() async throws {
        let (controller, channel, _, _) = make()
        controller.setArmed(true)
        controller.endSession()
        #expect(!controller.isArmed)
        #expect(!channel.isSessionAlive())
    }

    // MARK: Observer final state on teardown

    @Test func disarmWhileRecordingNotifiesObserverAfterMicrophoneIsOff() async throws {
        let (controller, _, capture, _) = make()
        var snapshots: [(isArmed: Bool, micOn: Bool)] = []
        controller.onChange = { snapshots.append((controller.isArmed, controller.isSessionOpen)) }
        controller.setArmed(true)
        try controller.start()
        #expect(capture.sessionRunning)
        snapshots.removeAll()
        controller.setArmed(false)
        #expect(snapshots.last != nil, "disarm must notify observers")
        #expect(snapshots.last?.isArmed == false)
        #expect(snapshots.last?.micOn == false, "the final callback must see the microphone already off")
    }

    @Test func endSessionAtDoneNotifiesObserverWithMicOff() async throws {
        let (controller, _, capture, _) = make()
        var snapshots: [(isArmed: Bool, micOn: Bool)] = []
        try controller.start()
        await controller.stop()
        #expect(controller.phase == .done)
        #expect(capture.sessionRunning, "legacy stop keeps the session open")
        controller.onChange = { snapshots.append((controller.isArmed, controller.isSessionOpen)) }
        controller.endSession()
        #expect(snapshots.last != nil, "idle/done teardown must still notify observers")
        #expect(snapshots.last?.micOn == false)
    }

}

@MainActor private struct LanguageDiagnosticTranscriber: Transcribing {
    func transcribeResult(wav: Data, options: CompositionOptions) async throws -> TranscriptionResult {
        .init(text: "I will try it.", rawText: "我试试这个效果怎么样。", asrMs: 700, enhanceMs: 5900)
    }
    func transcribe(wav: Data) async throws -> (text: String, rawText: String) {
        ("I will try it.", "我试试这个效果怎么样。")
    }
}
