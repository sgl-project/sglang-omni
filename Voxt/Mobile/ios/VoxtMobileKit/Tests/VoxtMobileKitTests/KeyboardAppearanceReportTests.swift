import Foundation
import Testing
@testable import VoxtMobileKit

/// What the keyboard writes each time it appears, for the app's Try it screen.
@Suite struct KeyboardAppearanceReportTests {
    private func defaults() -> UserDefaults { UserDefaults(suiteName: "keyboard-report-\(UUID().uuidString)")! }

    @Test func withFullAccessTheKeyboardConfirmsItNow() {
        let store = defaults()
        let cameUp = Date()
        KeyboardStatus.reportAppearance(hasFullAccess: true, to: store, now: cameUp.addingTimeInterval(0.1))
        #expect(KeyboardStatus.confirmsFullAccess(KeyboardStatus.load(from: store), since: cameUp))
    }

    @Test func withoutFullAccessNothingIsWrittenAndAnOldConfirmationStaysOld() {
        let store = defaults()
        KeyboardStatus(hasFullAccess: true, updatedAt: Date(timeIntervalSince1970: 1_000)).save(to: store)
        let cameUp = Date()
        KeyboardStatus.reportAppearance(hasFullAccess: false, to: store, now: cameUp)
        #expect(KeyboardStatus.load(from: store)?.updatedAt == Date(timeIntervalSince1970: 1_000))
        #expect(!KeyboardStatus.confirmsFullAccess(KeyboardStatus.load(from: store), since: cameUp))
    }
}
