import Foundation
import ServiceManagement

/// Non-secret launch state. No token is ever stored here.
public protocol ConnectorPreferences: AnyObject, Sendable {
    /// Voxt approved this Connector at least once, so connecting at launch shows no new dialog.
    var wasApproved: Bool { get set }
    /// The Connector shows a Dock icon. Off: only the menu bar item opens it.
    var showsInDock: Bool { get set }
}

public final class UserDefaultsConnectorPreferences: ConnectorPreferences, @unchecked Sendable {
    private let defaults: UserDefaults
    private let key = "connectorWasApprovedByVoxt"
    public init(defaults: UserDefaults = .standard) { self.defaults = defaults }
    public var wasApproved: Bool {
        get { defaults.bool(forKey: key) }
        set { defaults.set(newValue, forKey: key) }
    }
    public var showsInDock: Bool {
        get { defaults.object(forKey: "connectorShowsInDock") as? Bool ?? true }
        set { defaults.set(newValue, forKey: "connectorShowsInDock") }
    }
}

public enum ConnectorLoginItemStatus: Equatable, Sendable { case disabled, enabled, requiresApproval }

@MainActor public protocol ConnectorLoginItem: AnyObject {
    var status: ConnectorLoginItemStatus { get }
    /// The status, read without blocking the main thread.
    func currentStatus() async -> ConnectorLoginItemStatus
    func register() throws
    func unregister() throws
}

/// "Open at login" for this app through SMAppService.
@MainActor public final class MainAppLoginItem: ConnectorLoginItem {
    public init() {}
    public var status: ConnectorLoginItemStatus { Self.read() }
    /// SMAppService answers over XPC; reading it off the main actor keeps Settings from waiting for it.
    public func currentStatus() async -> ConnectorLoginItemStatus {
        await Task.detached { Self.read() }.value
    }
    private nonisolated static func read() -> ConnectorLoginItemStatus {
        switch SMAppService.mainApp.status {
        case .enabled: return .enabled
        case .requiresApproval: return .requiresApproval
        default: return .disabled
        }
    }
    public func register() throws { try SMAppService.mainApp.register() }
    public func unregister() throws { try SMAppService.mainApp.unregister() }
}
