import Foundation

public protocol TokenStoring: AnyObject {
    func read() throws -> String?
    func write(_ value: String) throws
    func delete() throws
}

/// Only the host app makes network requests; the keyboard never needs the token.
public final class PairingStore {
    private let defaults: UserDefaults
    private let tokens: any TokenStoring
    private let addressKey = "server.address"
    private let legacyKey = "server.config"
    private static let envelopePrefix = "voxt-connector-v1:"

    private struct Envelope: Codable {
        let version: Int
        let server: ServerConfig
    }

    public init(defaults: UserDefaults, tokens: any TokenStoring) {
        self.defaults = defaults
        self.tokens = tokens
    }

    public func load() throws -> ServerConfig? {
        let secret = try tokens.read()
        if let secret, secret.hasPrefix(Self.envelopePrefix) {
            let payload = String(secret.dropFirst(Self.envelopePrefix.count))
            let envelope = try JSONDecoder().decode(Envelope.self, from: Data(payload.utf8))
            guard envelope.version == 1 else { throw ServerConfig.Error.invalidAddress }
            // A prior save may have stopped after the atomic Keychain update.
            defaults.removeObject(forKey: addressKey)
            defaults.removeObject(forKey: legacyKey)
            return envelope.server
        }
        return nil
    }

    public func save(_ config: ServerConfig) throws {
        let payload = try JSONEncoder().encode(Envelope(version: 1, server: config))
        // Updating one Keychain item keeps address and credential together even
        // if the app stops before legacy defaults cleanup.
        try tokens.write(Self.envelopePrefix + String(decoding: payload, as: UTF8.self))
        defaults.removeObject(forKey: addressKey)
        defaults.removeObject(forKey: legacyKey)
    }

    public func clear() throws {
        try tokens.delete()
        defaults.removeObject(forKey: addressKey)
        defaults.removeObject(forKey: legacyKey)
    }
}
