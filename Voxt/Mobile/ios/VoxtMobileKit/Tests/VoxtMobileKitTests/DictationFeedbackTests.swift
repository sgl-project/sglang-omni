import Testing
@testable import VoxtMobileKit

struct DictationFeedbackTests {
    @Test func phaseTransitionsEmitOnlyOnce() {
        #expect(DictationFeedback.event(from: .idle, to: .recording) == .started)
        #expect(DictationFeedback.event(from: .recording, to: .processing) == .stopped)
        #expect(DictationFeedback.event(from: .recording, to: .idle) == .stopped)
        #expect(DictationFeedback.event(from: .processing, to: .done) == .success)
        #expect(DictationFeedback.event(from: .processing, to: .failed) == .failure)
        #expect(DictationFeedback.event(from: .failed, to: .failed) == nil)
        #expect(DictationFeedback.event(from: .done, to: .idle) == nil)
    }
    @Test func keyboardWaitsForInsertionSuccess() {
        #expect(DictationFeedback.event(from: .processing, to: .done, includeSuccess: false) == nil)
    }
}
