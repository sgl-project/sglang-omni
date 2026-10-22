import Foundation
import Testing
@testable import VoxtMobileKit

/// Without Full Access the keyboard can't write the App Group, so the app only ever sees
/// `true`. Missing or older than the keyboard's appearance means "not on".
@Suite struct KeyboardStatusTests {
    let keyboardUp = Date(timeIntervalSince1970: 1_000)

    @Test func noStatusIsNotFullAccess() {
        #expect(!KeyboardStatus.confirmsFullAccess(nil, since: keyboardUp))
    }

    @Test func aStatusFromBeforeTheKeyboardCameUpIsStale() {
        let old = KeyboardStatus(hasFullAccess: true, updatedAt: keyboardUp.addingTimeInterval(-5))
        #expect(!KeyboardStatus.confirmsFullAccess(old, since: keyboardUp))
    }

    @Test func aFreshTrueConfirms() {
        let fresh = KeyboardStatus(hasFullAccess: true, updatedAt: keyboardUp.addingTimeInterval(0.4))
        #expect(KeyboardStatus.confirmsFullAccess(fresh, since: keyboardUp))
    }

    @Test func roundTripsThroughTheAppGroup() {
        let name = "keyboard-status.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        #expect(KeyboardStatus.load(from: defaults) == nil)
        let status = KeyboardStatus(hasFullAccess: true, updatedAt: keyboardUp)
        status.save(to: defaults)
        #expect(KeyboardStatus.load(from: defaults) == status)
    }
}
