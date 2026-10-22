import Foundation

/// Commands an on-screen test sends to a Connector launched with VOXT_CONNECTOR_TEST_HOOKS=1.
/// Each one calls the same handler as the real control it stands in for.
public enum ConnectorTestCommand: String, CaseIterable, Sendable {
    case openSettingsFromMenu, showInDockOn, showInDockOff, dockIconClicked
    case simulatePhoneRequest, answerPhoneRequestDeny
}

public enum ConnectorTestHooks {
    public static let environmentKey = "VOXT_CONNECTOR_TEST_HOOKS"

    public static func notificationName(_ command: ConnectorTestCommand) -> Notification.Name {
        Notification.Name("\(ConnectorIdentity.defaultBundleIdentifier).test.\(command.rawValue)")
    }

    /// Returns the observers to keep, or nil when the app wasn't launched for tests.
    @MainActor public static func install(environment: [String: String] = ProcessInfo.processInfo.environment,
                                          center: NotificationCenter = DistributedNotificationCenter.default(),
                                          handle: @escaping @MainActor (ConnectorTestCommand) -> Void) -> [any NSObjectProtocol]? {
        guard environment[environmentKey] == "1" else { return nil }
        return ConnectorTestCommand.allCases.map { command in
            center.addObserver(forName: notificationName(command), object: nil, queue: .main) { _ in
                MainActor.assumeIsolated { handle(command) }
            }
        }
    }
}
