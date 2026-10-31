@testable import VoxtConnectionKit

/// Keeps the approval flag out of the test runner's UserDefaults, so tests can't affect each other.
final class TestPreferences: ConnectorPreferences, @unchecked Sendable {
    var wasApproved = false
    var showsInDock = true
}
