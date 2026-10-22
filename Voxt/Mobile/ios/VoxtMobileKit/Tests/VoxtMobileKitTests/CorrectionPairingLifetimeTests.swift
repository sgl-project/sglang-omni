import Foundation
import Testing
@testable import VoxtMobileKit

@MainActor private final class PendingCorrection {
    var response: CheckedContinuation<LearnedCorrection?, any Error>?
    var started: CheckedContinuation<Void, Never>?
    func extract() async throws -> LearnedCorrection? {
        try await withCheckedThrowingContinuation { response = $0; started?.resume(); started = nil }
    }
    func wait() async { if response != nil { return }; await withCheckedContinuation { started = $0 } }
    func finish(success: Bool) {
        if success { response?.resume(returning: .init(original: "Voice", replacement: "Voxt")) }
        else { response?.resume(throwing: URLError(.cancelled)) }
        response = nil
    }
}

@Suite @MainActor struct CorrectionPairingLifetimeTests {
    @Test func revokedPairingCannotPublishLateCorrectionOrError() async throws {
        for success in [true, false] {
            let name = "correction-pairing.\(UUID())", pending = PendingCorrection()
            let defaults = UserDefaults(suiteName: name)!
            defer { defaults.removePersistentDomain(forName: name) }
            let store = CorrectionLearningStore(defaults: defaults)
            store.isEnabled = true
            let channel = CorrectionLearningChannel(defaults: defaults, notifier: .none)
            let subject = CorrectionLearningProcessor(channel: channel, store: store) { _ in try await pending.extract() }
            let id = channel.submit(.init(original: "Use Voice.", edited: "Use Voxt."))
            let task = Task { await subject.process() }
            await pending.wait()
            subject.invalidate()
            pending.finish(success: success)
            await task.value
            #expect(channel.takeResult(for: id) == nil)
            #expect(store.isEnabled)
        }
    }
}
