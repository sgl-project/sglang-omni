import Foundation

/// Advertises the Connector on the local network while the Add iPhone QR code is shown,
/// and hands each phone to a LocalPairingCoordinator with the same invitation.
@MainActor final class AdvertisedLocalPairing: ConnectorLocalPairing {
    private let coordinator: LocalPairingCoordinator
    private let advertiser = LocalPairingAdvertiser()

    init(invitation: URL, expiresAt: Date, onRequest: @escaping @Sendable (String, String) async -> Bool) throws {
        coordinator = LocalPairingCoordinator(invitation: invitation, expiresAt: expiresAt, onRequest: onRequest)
        try advertiser.start(coordinator: coordinator)
    }

    func stop() {
        advertiser.stop()
        let coordinator = coordinator
        Task { await coordinator.cancel() }
    }
}
