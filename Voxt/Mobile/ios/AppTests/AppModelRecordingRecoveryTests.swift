import XCTest
@testable import VoxtMobile
import VoxtMobileKit

@MainActor
private final class SpeakingCapture: AudioCapturing {
    /// Two seconds of a 220 Hz tone: speech to the no-speech gate.
    var samples: [Float] = (0..<32_000).map { 0.2 * Float(sin(Double($0) * 2 * .pi * 220 / 16_000)) }
    func startSession() throws {}
    func endSession() {}
    func beginCapture() {}
    func endCapture() -> [Float] { samples }
}

@MainActor
private final class ScriptedTranscriber: Transcribing {
    var failure: Error?
    var text = "Sent from History."
    func transcribe(wav: Data) async throws -> (text: String, rawText: String) {
        if let failure { throw failure }
        return (text, text)
    }
}

/// History lists the recordings the Mac never answered and sends them again, so no recording
/// is lost. Isolated defaults and a temporary recordings folder.
@MainActor
final class AppModelRecordingRecoveryTests: XCTestCase {
    private var suiteName: String!
    private var directory: URL!

    override func setUpWithError() throws {
        suiteName = "voxt-recording-recovery.\(UUID().uuidString)"
        directory = FileManager.default.temporaryDirectory.appendingPathComponent("recordings-\(UUID().uuidString)", isDirectory: true)
    }

    override func tearDownWithError() throws {
        UserDefaults().removePersistentDomain(forName: suiteName)
        try? FileManager.default.removeItem(at: directory)
    }

    private func makeModel() -> AppModel {
        AppModel(injectedCapture: SpeakingCapture(), injectedDefaults: UserDefaults(suiteName: suiteName),
                 injectedRecordings: RecordingStore(directory: directory))
    }

    func testAFailedDictationWaitsInHistoryAndSendsAgainFromThere() async throws {
        let model = makeModel()
        let transcriber = ScriptedTranscriber()
        transcriber.failure = VoxtClient.Error.transportUnavailable
        model.controller.setTranscriber(transcriber)
        try model.controller.start()
        await model.controller.stop()
        XCTAssertEqual(model.unsentRecordings.map(\.status), [.failed])
        XCTAssertEqual(model.unsentRecordings.first?.duration, 2)

        transcriber.failure = nil
        await model.sendAgain(try XCTUnwrap(model.unsentRecordings.first?.id))
        XCTAssertTrue(model.unsentRecordings.isEmpty)
        XCTAssertEqual(model.history.first?.text, "Sent from History.")
        XCTAssertEqual(model.latestText, "Sent from History.")
    }

    func testDeletingAnUnsentRecordingRemovesItsAudio() async throws {
        let model = makeModel()
        try model.controller.start()
        await model.controller.stop() // unpaired: "Pair with your Mac first."
        let id = try XCTUnwrap(model.unsentRecordings.first?.id)
        model.deleteUnsentRecording(id)
        XCTAssertTrue(model.unsentRecordings.isEmpty)
        XCTAssertNil(RecordingStore(directory: directory).recording(id))
    }

    func testARecordingCutOffByAKillIsListedAtTheNextLaunch() throws {
        let store = RecordingStore(directory: directory)
        let id = try store.begin(options: .init(), requestID: nil, now: Date().addingTimeInterval(-3600))
        try store.append(Array(repeating: 0.1, count: 16_000), to: id)
        let model = makeModel()
        XCTAssertEqual(model.unsentRecordings.map(\.id), [id])
        XCTAssertEqual(model.unsentRecordings.first?.status, .interrupted)
    }

    /// After a relaunch the keyboard's Retry opens voxtmobile://reconnect. The resend runs inside
    /// a background task that ends only once it's done, so a person who switches straight back
    /// to the other app doesn't leave it suspended half-way.
    func testRetryAfterARelaunchResendsInsideABackgroundTask() async throws {
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suiteName))
        let tokens = RecoveryTokens()
        try PairingStore(defaults: defaults, tokens: tokens).save(
            ServerConfig(baseURL: URL(string: "http://127.0.0.1:9")!, token: "fixture",
                         tailcat: .init(address: "fixture", clientKey: "fixture", remotePort: 48931)))
        let store = RecordingStore(directory: directory)
        let id = try store.begin(options: .init(), requestID: "kb-1", now: Date())
        try store.append(SpeakingCapture().samples, to: id)
        store.mark(id, .failed, errorMessage: "Couldn't reach your Mac")

        let events = RecoveryEvents()
        let health = try JSONDecoder().decode(VoxtClient.Health.self, from: Data(#"{"ok":true,"app":"Voxt","version":"fixture"}"#.utf8))
        let model = AppModel(injectedCapture: SpeakingCapture(), injectedDefaults: defaults, injectedTokens: tokens,
                             injectedHealthProbe: { _ in health },
                             injectedConnectionFactory: { config in
            PairingConnection(server: config, resolve: { events.add("resend"); return URL(string: "http://127.0.0.1:9")! })
        }, injectedRecordings: store, injectedBackgroundTask: { _ in
            events.add("begin")
            return { events.add("end") }
        })
        XCTAssertEqual(model.phase, .failed)
        XCTAssertTrue(model.canRetry)
        XCTAssertEqual(DictationChannel(defaults: defaults).state.requestID, "kb-1")

        model.handle(url: URL(string: "voxtmobile://reconnect")!)
        for _ in 0..<1500 where !events.all.contains("end") { try await Task.sleep(for: .milliseconds(20)) }
        XCTAssertEqual(events.all.first, "begin")
        XCTAssertTrue(events.all.contains("resend"), "the saved recording is sent again: \(events.all)")
        XCTAssertEqual(events.all.last, "end")
    }
}

private final class RecoveryTokens: TokenStoring {
    var value: String?
    func read() throws -> String? { value }
    func write(_ value: String) throws { self.value = value }
    func delete() throws { value = nil }
}

private final class RecoveryEvents: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [String] = []
    var all: [String] { lock.withLock { events } }
    func add(_ event: String) { lock.withLock { events.append(event) } }
}
