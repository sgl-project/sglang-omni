import Foundation
import Testing
@testable import VoxtMobileKit

private final class EnvelopeSecrets: TokenStoring {
    var value: String?
    var failWrites = false
    func read() throws -> String? { value }
    func write(_ value: String) throws {
        if failWrites { throw URLError(.cannotWriteToFile) }
        self.value = value
    }
    func delete() throws { value = nil }
}

@Suite struct PairingEnvelopeTests {
    @Test func addressCannotBeReplacedIndependentlyOfItsCredential() throws {
        let name = "pairing-envelope.\(UUID())", secrets = EnvelopeSecrets()
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let store = PairingStore(defaults: defaults, tokens: secrets)
        let expected = ServerConfig(baseURL: URL(string: "https://first.example.test")!, token: "fixture-first")
        try store.save(expected)
        #expect(defaults.string(forKey: "server.address") == nil)
        defaults.set("https://different.example.test", forKey: "server.address")
        #expect(try store.load() == expected)
    }

    @Test func failedReplacementKeepsCurrentConnectorPairing() throws {
        let name = "pairing-envelope.\(UUID())", secrets = EnvelopeSecrets()
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let expected = ServerConfig(baseURL: URL(string: "https://first.example.test")!, token: "fixture-first")
        let store = PairingStore(defaults: defaults, tokens: secrets)
        try store.save(expected)
        secrets.failWrites = true
        let replacement = ServerConfig(baseURL: URL(string: "https://second.example.test")!, token: "fixture-second")
        #expect(throws: URLError.self) { try store.save(replacement) }
        #expect(try store.load() == expected)
    }

}

@Suite struct TailcatPairingTests {
    @Test func transportSecretsRoundTripOnlyThroughCredentialStore() throws {
        let name = "tailcat-pairing.\(UUID())", secrets = EnvelopeSecrets()
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let transport = ServerConfig.Tailcat(address: "tc-fixture-address", clientKey: "privkey:fixture", remotePort: 48931)
        let expected = ServerConfig(baseURL: URL(string: "http://tailcat.invalid")!, token: "fixture-bearer", tailcat: transport)
        let store = PairingStore(defaults: defaults, tokens: secrets)
        try store.save(expected)
        #expect(try store.load() == expected)
        let stored = defaults.persistentDomain(forName: name) ?? [:]
        #expect(stored.isEmpty)
        try store.clear()
        #expect(secrets.value == nil)
    }

    @Test func tailcatWithoutResolverFailsBeforeHTTP() async {
        let config = ServerConfig(baseURL: URL(string: "http://tailcat.invalid")!, token: "fixture-bearer", tailcat: .init(address: "tc-fixture", clientKey: "privkey:fixture", remotePort: 48931))
        let client = VoxtClient(server: config)
        await #expect(throws: VoxtClient.Error.transportUnavailable) { try await client.health() }
    }
}
