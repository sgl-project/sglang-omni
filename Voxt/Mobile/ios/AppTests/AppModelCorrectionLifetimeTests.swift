import XCTest
import VoxtMobileKit
@testable import VoxtMobile

@MainActor private final class PendingLearning {
    var response: CheckedContinuation<LearnedCorrection?, any Error>?
    var started: CheckedContinuation<Void, Never>?
    func extract() async throws -> LearnedCorrection? {
        try await withCheckedThrowingContinuation { response = $0; started?.resume(); started = nil }
    }
    func wait() async { if response != nil { return }; await withCheckedContinuation { started = $0 } }
    func finish() { response?.resume(returning: .init(original: "Voice", replacement: "Voxt")); response = nil }
}

@MainActor final class AppModelCorrectionLifetimeTests: XCTestCase {
    func testUnpairPreventsPendingLearningFromPublishing() async throws {
        let name = "learning-lifetime.\(UUID())", pending = PendingLearning()
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let model = AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: defaults,
                             injectedCorrectionExchange: { _ in try await pending.extract() })
        model.learning.isEnabled = true
        let channel = CorrectionLearningChannel(defaults: defaults, notifier: .none)
        let id = channel.submit(.init(original: "Use Voice.", edited: "Use Voxt."))
        let task = Task { await model.processPendingCorrectionLearning() }
        await pending.wait()
        model.unpair()
        pending.finish()
        await task.value
        XCTAssertNil(channel.takeResult(for: id))
        XCTAssertTrue(model.learning.isEnabled)
    }
}
