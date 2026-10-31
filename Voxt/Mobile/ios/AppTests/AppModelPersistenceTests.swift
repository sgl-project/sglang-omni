import XCTest
@testable import VoxtMobile
import VoxtMobileKit

private final class TestCredentialStore: TokenStoring {
    var value: String?
    func read() throws -> String? { value }
    func write(_ value: String) throws { self.value = value }
    func delete() throws { value = nil }
}

@MainActor
final class AppModelPersistenceTests: XCTestCase {
    func testKeyboardRecordingDoesNotRevealAppRecorder() async throws {
        let name = "app-recording-source.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let model = AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: defaults)
        try model.controller.start()
        XCTAssertEqual(model.phase, .recording)
        XCTAssertFalse(model.showsAppRecording)
        model.cancelComposition()
        await model.toggleRecording()
        XCTAssertEqual(model.phase, .recording)
        XCTAssertTrue(model.showsAppRecording)
        model.cancelComposition()
    }

    func testDisableAfterSettingPersistsAndDrivesTheController() {
        let name = "app-mic-readiness.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let model = AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: defaults)
        XCTAssertEqual(model.microphoneReadiness, .fiveMinutes)
        XCTAssertEqual(model.controller.idleTimeout, 5 * 60)
        model.setMicrophoneReadiness(.never)
        XCTAssertNil(model.controller.idleTimeout)
        let relaunched = AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: defaults)
        XCTAssertEqual(relaunched.microphoneReadiness, .never)
        XCTAssertNil(relaunched.controller.idleTimeout)
    }

    func testPendingResultCanBeRecoveredOnPairedRelaunchWithHistoryOff() throws {
        let name = "app-recovery.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let tokens = TestCredentialStore()
        try PairingStore(defaults: defaults, tokens: tokens).save(ServerConfig(baseURL: URL(string: "https://fixture.invalid")!, token: "test-only"))
        TranscriptHistory(defaults: defaults).setRetention(.off)
        DictationChannel(defaults: defaults, notifier: .none).publish(.init(phase: .done, text: "Recover this result", resultID: "pending"))
        let model = AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: defaults, injectedTokens: tokens)
        XCTAssertEqual(model.latestText, "Recover this result")
        XCTAssertTrue(model.history.isEmpty)
        XCTAssertNil(DictationChannel(defaults: defaults, notifier: .none).state.text)
        model.unpair()
        XCTAssertNil(model.latestText)
    }
    func testLatestSavedResultRemainsEditableAfterRepeatedRelaunch() async throws {
        let name = "app-history-edit.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let tokens = TestCredentialStore()
        try PairingStore(defaults: defaults, tokens: tokens).save(ServerConfig(baseURL: URL(string: "https://fixture.invalid")!, token: "test-only"))
        TranscriptHistory(defaults: defaults).append(text: "Send Alice fifty dollars by Friday.")
        for _ in 0..<2 {
            let model = AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: defaults, injectedTokens: tokens)
            XCTAssertEqual(model.latestText, "Send Alice fifty dollars by Friday.")
            await model.editLatestResult()
            XCTAssertEqual(model.phase, .recording)
            model.cancelComposition()
        }
    }

    func testHistoryOffPreventsSuccessfulDictationFromBeingPersisted() async throws {
        let name = "app-persistence.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let model = AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: defaults)
        model.setHistoryRetention(.off)
        model.controller.setTranscriber(PersistenceTranscriber())
        try model.controller.start()
        await model.controller.stop()
        XCTAssertTrue(model.history.isEmpty)
        XCTAssertTrue(TranscriptHistory(defaults: defaults).items.isEmpty)
        model.setHistoryRetention(.forever)
        try model.controller.start()
        await model.controller.stop()
        XCTAssertEqual(model.history.map(\.text), ["Saved message"])
    }
}

@MainActor
private struct PersistenceTranscriber: Transcribing {
    func transcribe(wav: Data) async throws -> (text: String, rawText: String) { ("Saved message", "Saved message") }
}
