import XCTest
@testable import VoxtMobile
import VoxtMobileKit

@MainActor
private final class ScriptedCapture: AudioCapturing {
    enum StartError: LocalizedError {
        case audioSessionFailed
        var errorDescription: String? { "audio session failed" }
    }

    var startSessionError: Error?
    /// 4801 samples at 16 kHz: just over the controller's 0.3 s minimum, as a tone because
    /// silence is rejected before upload by the no-speech gate.
    var samples: [Float] = (0..<4801).map { 0.2 * Float(sin(Double($0) * 2 * .pi * 220 / 16_000)) }

    func startSession() throws {
        if let startSessionError { throw startSessionError }
    }

    func endSession() {}

    func beginCapture() {}

    func endCapture() -> [Float] { samples }
}

@MainActor
private final class StaticTranscriber: Transcribing {
    var text: String
    var failure: Error?
    var options: CompositionOptions?
    var onTranscribe: (() -> Void)?
    init(text: String) { self.text = text }

    func transcribe(wav: Data) async throws -> (text: String, rawText: String) {
        if let failure { throw failure }
        onTranscribe?()
        return (text, text)
    }

    func transcribe(wav: Data, options: CompositionOptions) async throws -> (text: String, rawText: String) {
        self.options = options
        return try await transcribe(wav: wav)
    }
}

/// The app's own mic button and the keyboard share one DictationController.
/// A dictation only counts toward appResultCount (History-tab selection) when
/// it was started from the app. These tests use an isolated defaults suite and
/// fake capture/transcriber: no credentials, microphone, or live Mac.
@MainActor
final class AppModelResultAttributionTests: XCTestCase {
    private var suiteName: String!

    override func setUpWithError() throws {
        suiteName = "voxt-appmodel-tests.\(UUID().uuidString)"
    }

    override func tearDownWithError() throws {
        UserDefaults().removePersistentDomain(forName: suiteName)
    }

    private func makeModel(capture: ScriptedCapture) -> AppModel {
        AppModel(injectedCapture: capture, injectedDefaults: UserDefaults(suiteName: suiteName))
    }

    func testTimersFollowRecordingAndSessionLifetime() async throws {
        let model = AppModel(injectedCapture: ScriptedCapture(), injectedDefaults: UserDefaults(suiteName: suiteName), injectedTimerScheduling: true)
        XCTAssertFalse(model.isLevelTimerRunning, "idle must not schedule the 30 Hz meter")
        XCTAssertFalse(model.isSessionTimerRunning, "unpaired idle needs no heartbeat")
        try model.controller.start()
        XCTAssertTrue(model.isLevelTimerRunning)
        XCTAssertTrue(model.isSessionTimerRunning)
        model.cancelComposition()
        XCTAssertFalse(model.isLevelTimerRunning)
        model.endSession()
        XCTAssertFalse(model.isSessionTimerRunning)
        model.controller.setArmed(true)
        XCTAssertFalse(model.isLevelTimerRunning)
        XCTAssertTrue(model.isSessionTimerRunning, "PiP readiness needs heartbeat with the mic off")
        model.controller.setArmed(false)
        XCTAssertFalse(model.isSessionTimerRunning)
    }

    func testLevelTimerStopsBeforeTranscriptionAndRestartsForNextRecording() async throws {
        let model = AppModel(injectedCapture: ScriptedCapture(), injectedDefaults: UserDefaults(suiteName: suiteName), injectedTimerScheduling: true)
        let transcriber = StaticTranscriber(text: "fixture")
        transcriber.onTranscribe = {
            XCTAssertEqual(model.phase, .processing)
            XCTAssertFalse(model.isLevelTimerRunning)
        }
        model.controller.setTranscriber(transcriber)
        try model.controller.start()
        await model.controller.stop()
        XCTAssertEqual(model.phase, .done)
        XCTAssertFalse(model.isLevelTimerRunning)
        try model.controller.start()
        XCTAssertTrue(model.isLevelTimerRunning)
        model.endSession()
        XCTAssertFalse(model.isLevelTimerRunning)
        XCTAssertFalse(model.isSessionTimerRunning)
        transcriber.onTranscribe = nil
    }

    /// The keyboard used inside Voxt itself (Voxt already in front) records without
    /// covering the screen; a keyboard that just brought Voxt forward still gets the screen.
    func testHandoffScreenOnlyWhenTheKeyboardBroughtVoxtForward() async throws {
        let inFront = makeModel(capture: ScriptedCapture())
        inFront.sceneBecameActive(at: Date().addingTimeInterval(-10))
        DictationChannel(defaults: UserDefaults(suiteName: suiteName)!, notifier: .none)
            .send(.start, options: .init(), requestID: "already-in-front")
        await inFront.startFromKeyboard()
        XCTAssertEqual(inFront.phase, .recording)
        XCTAssertFalse(inFront.showsKeyboardHandoff, "Voxt was already in front")
        inFront.cancelComposition()

        let broughtForward = makeModel(capture: ScriptedCapture())
        broughtForward.sceneBecameActive(at: Date().addingTimeInterval(-0.5))
        DictationChannel(defaults: UserDefaults(suiteName: suiteName)!, notifier: .none)
            .send(.start, options: .init(), requestID: "brought-forward")
        await broughtForward.startFromKeyboard()
        XCTAssertTrue(broughtForward.showsKeyboardHandoff, "the keyboard just opened Voxt")
        broughtForward.sceneLeftActive()
        broughtForward.cancelComposition()
    }

    func testHandoffCoverRule() {
        let now = Date()
        XCTAssertTrue(KeyboardHandoffCover.shows(activeSince: nil, now: now), "not active yet: opened by the keyboard")
        XCTAssertTrue(KeyboardHandoffCover.shows(activeSince: now.addingTimeInterval(-1.9), now: now))
        XCTAssertFalse(KeyboardHandoffCover.shows(activeSince: now.addingTimeInterval(-2.1), now: now))
    }

    func testKeyboardHandoffClosesOnCancelAndNeverOpensForAppRecording() async throws {
        let model = makeModel(capture: ScriptedCapture())
        let channel = DictationChannel(defaults: UserDefaults(suiteName: suiteName)!, notifier: .none)
        channel.send(.start, options: .init(), requestID: "handoff-test")
        await model.startFromKeyboard()
        XCTAssertEqual(model.phase, .recording)
        XCTAssertTrue(model.showsKeyboardHandoff)
        model.cancelComposition()
        XCTAssertFalse(model.showsKeyboardHandoff)
        await model.toggleRecording()
        XCTAssertTrue(model.showsAppRecording)
        XCTAssertFalse(model.showsKeyboardHandoff)
        model.cancelComposition()
    }

    func testFailedKeyboardStartDoesNotShowHandoff() async throws {
        let capture = ScriptedCapture()
        capture.startSessionError = ScriptedCapture.StartError.audioSessionFailed
        let model = makeModel(capture: capture)
        DictationChannel(defaults: UserDefaults(suiteName: suiteName)!, notifier: .none)
            .send(.start, options: .init(), requestID: "failed-handoff-test")
        await model.startFromKeyboard()
        XCTAssertNotNil(model.errorMessage)
        XCTAssertFalse(model.showsKeyboardHandoff)
    }

    func testThrowingAppStartDoesNotClaimTheNextKeyboardResult() async throws {
        let capture = ScriptedCapture()
        capture.startSessionError = ScriptedCapture.StartError.audioSessionFailed
        let model = makeModel(capture: capture)

        // App mic button: startSession throws before any state change is published.
        await model.toggleRecording()
        XCTAssertNotNil(model.errorMessage, "the throwing start must have been recorded")
        XCTAssertEqual(model.phase, .idle)

        // Keyboard-origin dictation on the same controller succeeds.
        capture.startSessionError = nil
        model.controller.setTranscriber(StaticTranscriber(text: "keyboard text"))
        try model.controller.start()
        XCTAssertEqual(model.phase, .recording)
        await model.controller.stop()

        XCTAssertEqual(model.history.first?.text, "keyboard text")
        XCTAssertEqual(model.appResultCount, 0, "a keyboard result must not count as an app dictation")
    }

    func testSuccessfulAppDictationIncrementsCount() async throws {
        let capture = ScriptedCapture()
        let model = makeModel(capture: capture)
        model.controller.setTranscriber(StaticTranscriber(text: "app text"))

        await model.toggleRecording()
        XCTAssertEqual(model.phase, .recording)
        await model.toggleRecording()

        XCTAssertEqual(model.history.first?.text, "app text")
        XCTAssertEqual(model.appResultCount, 1, "an app-origin result must switch to History")
    }

    func testAppDictationUsesOnlyEnabledPersonalStyle() async throws {
        let model = makeModel(capture: ScriptedCapture())
        let transcriber = StaticTranscriber(text: "style fixture")
        model.controller.setTranscriber(transcriber)
        let styles = PersonalStyleStore(defaults: UserDefaults(suiteName: suiteName)!)
        styles.isEnabled = true
        let example = PersonalStyleExample(before: "Thanks, I'll send it.", after: "Cheers — I'll send it.")
        try styles.keep(example)
        await model.toggleRecording()
        await model.toggleRecording()
        XCTAssertEqual(transcriber.options?.styleExamples, [example])
        styles.isEnabled = false
        await model.toggleRecording()
        await model.toggleRecording()
        XCTAssertEqual(transcriber.options?.styleExamples, [])
    }

    func testEditLatestResultRecordsInstructionAndKeepsOriginalOnCancel() async throws {
        let model = makeModel(capture: ScriptedCapture())
        let transcriber = StaticTranscriber(text: "Alice owes $50 by Friday.")
        model.controller.setTranscriber(transcriber)
        await model.toggleRecording()
        await model.toggleRecording()
        let original = try XCTUnwrap(model.latestText)

        await model.editLatestResult()
        XCTAssertEqual(model.phase, .recording)
        model.cancelComposition()
        XCTAssertEqual(model.latestText, original)
        XCTAssertEqual(model.history.count, 1)

        await model.editLatestResult()
        transcriber.text = "Please send Alice's $50 payment by Friday."
        await model.toggleRecording()
        XCTAssertEqual(transcriber.options?.mode, .edit)
        XCTAssertEqual(transcriber.options?.selectedText, original)
        XCTAssertEqual(model.latestText, transcriber.text)
        XCTAssertEqual(model.history.first?.mode, .edit)
        XCTAssertEqual(model.preferences.mode, .dictate, "Editing once must not change the next dictation mode")
    }

    func testNotAppliedEditKeepsLatestTextAndHistoryAndShowsError() async throws {
        let model = makeModel(capture: ScriptedCapture())
        let transcriber = StaticTranscriber(text: "The server has a cash limit.")
        model.controller.setTranscriber(transcriber)
        await model.toggleRecording()
        await model.toggleRecording()
        let previous = model.latestText
        let previousHistory = model.history.map(\.text)
        transcriber.failure = VoxtClient.Error.editNotApplied
        await model.editLatestResult()
        await model.toggleRecording()
        XCTAssertEqual(model.phase, .failed)
        XCTAssertEqual(model.errorMessage, "Couldn't apply that edit")
        XCTAssertEqual(model.latestText, previous)
        XCTAssertEqual(model.history.map(\.text), previousHistory)
        XCTAssertNil(DictationChannel(defaults: UserDefaults(suiteName: suiteName)!, notifier: .none)
            .textToInsert(lastInsertedID: nil))
    }

    func testEditLatestResultWithoutAResultDoesNotRecord() async {
        let model = makeModel(capture: ScriptedCapture())
        await model.editLatestResult()
        XCTAssertEqual(model.phase, .idle)
    }

    func testArmingAfterDictationDoesNotDuplicateHistory() async throws {
        let model = makeModel(capture: ScriptedCapture())
        model.controller.setTranscriber(StaticTranscriber(text: "one result"))
        await model.toggleRecording()
        await model.toggleRecording()
        XCTAssertEqual(model.history.count, 1)
        model.controller.setArmed(true)
        XCTAssertEqual(model.history.count, 1, "readiness changes must not save an old result again")
        XCTAssertFalse(model.sessionOpen)
        model.controller.setArmed(false)
        XCTAssertEqual(model.history.count, 1)
    }

    func testHistoryMeasuresRecordingThroughResultAndResetsForNextSession() async throws {
        var now = Date(timeIntervalSince1970: 1_700_000_000)
        let model = AppModel(injectedCapture: ScriptedCapture(), injectedDefaults: UserDefaults(suiteName: suiteName), injectedNow: { now })
        let transcriber = StaticTranscriber(text: "timed text")
        transcriber.onTranscribe = { now = now.addingTimeInterval(2) }
        model.controller.setTranscriber(transcriber)
        await model.toggleRecording()
        now = now.addingTimeInterval(7)
        await model.toggleRecording()
        XCTAssertEqual(model.history.first?.elapsedSeconds, 9)
        now = now.addingTimeInterval(50)
        await model.toggleRecording()
        now = now.addingTimeInterval(3)
        await model.toggleRecording()
        XCTAssertEqual(model.history.first?.elapsedSeconds, 5)
    }
}

import XCTest
import AVFoundation
@testable import VoxtMobile

@MainActor
final class FileAudioCaptureLevelTests: XCTestCase {
    func testFixtureNameResolvesInCurrentAppContainer() {
        XCTAssertEqual(FileAudioCapture.fixtureURL("speech.wav"),
                       FileManager.default.temporaryDirectory.appendingPathComponent("speech.wav"))
    }

    func testMissingFixtureThrowsInsteadOfSilentlyRecordingNothing() {
        let capture = FileAudioCapture(url: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString))
        XCTAssertThrowsError(try capture.startSession())
    }

    func testFileMeterIsActiveOnlyDuringRecording() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        defer { try? FileManager.default.removeItem(at: url) }
        let format = AVAudioFormat(standardFormatWithSampleRate: 16000, channels: 1)!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 16000)!
        buffer.frameLength = 16000
        for i in 0..<16000 { buffer.floatChannelData![0][i] = i % 2 == 0 ? 0.5 : -0.5 }
        do {
            let file = try AVAudioFile(forWriting: url, settings: format.settings)
            try file.write(from: buffer)
        }
        let capture = FileAudioCapture(url: url)
        try capture.startSession()
        XCTAssertEqual(capture.level, 0)
        capture.beginCapture()
        XCTAssertGreaterThan(capture.level, 0.4)
        XCTAssertGreaterThan(capture.endCapture().count, 15000)
        XCTAssertEqual(capture.level, 0)
    }
}
