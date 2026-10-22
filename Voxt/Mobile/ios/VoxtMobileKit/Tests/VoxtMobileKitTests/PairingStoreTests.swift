import Foundation
import Testing
@testable import VoxtMobileKit

private final class MemoryTokenStore: TokenStoring {
    var token: String?
    var failWrites = false
    func read() throws -> String? { token }
    func write(_ value: String) throws {
        if failWrites { throw URLError(.cannotWriteToFile) }
        token = value
    }
    func delete() throws { token = nil }
}

/// Items by Keychain service, as the real Keychain keeps them apart.
private final class ServiceKeychain {
    var items: [String: String] = [:]
    final class Item: TokenStoring {
        let keychain: ServiceKeychain
        let service: String
        init(keychain: ServiceKeychain, service: String) { self.keychain = keychain; self.service = service }
        func read() throws -> String? { keychain.items[service] }
        func write(_ value: String) throws { keychain.items[service] = value }
        func delete() throws { keychain.items[service] = nil }
    }
}

/// The real Keychain round trip needs a login Keychain. Under `swift test` with an isolated HOME, or
/// on a CI runner, there is none and macOS waits on a Keychain dialog, so it runs only when asked:
/// VOXT_KEYCHAIN_TESTS=1 swift test --package-path ios/VoxtMobileKit
enum KeychainTests {
    static func enabled(_ environment: [String: String] = ProcessInfo.processInfo.environment) -> Bool {
        environment["VOXT_KEYCHAIN_TESTS"] == "1"
    }
}

@Suite struct PairingStoreTests {
    @Test func oldPairingIsNotMigratedAndConnectorPairingPersists() throws {
        let name = "pairing.test.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let config = ServerConfig(baseURL: URL(string: "https://fixture.example.test")!, token: "fake-fixture")
        defaults.set(try JSONEncoder().encode(config), forKey: "server.config")
        let secrets = MemoryTokenStore()
        let store = PairingStore(defaults: defaults, tokens: secrets)
        #expect(try store.load() == nil)
        let oldEnvelope = try JSONSerialization.data(withJSONObject: ["version": 1, "server": JSONSerialization.jsonObject(with: JSONEncoder().encode(config))])
        secrets.token = "voxt-pairing-v1:" + String(decoding: oldEnvelope, as: UTF8.self)
        #expect(try store.load() == nil)
        try store.save(config)
        #expect(try store.load() == config)
        try store.clear()
        #expect(try store.load() == nil)
    }

    @Test func theKeychainRoundTripRunsOnlyWhenAsked() {
        #expect(!KeychainTests.enabled([:]))
        #expect(!KeychainTests.enabled(["VOXT_KEYCHAIN_TESTS": "0"]))
        #expect(KeychainTests.enabled(["VOXT_KEYCHAIN_TESTS": "1"]))
    }

    @Test func savingAndClearingTouchOnlyTheStoresOwnItem() throws {
        let name = "pairing.items.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let keychain = ServiceKeychain()
        keychain.items["other.service"] = "someone-else"
        let store = PairingStore(defaults: defaults, tokens: ServiceKeychain.Item(keychain: keychain, service: "voxt.pairing"))
        let first = ServerConfig(baseURL: URL(string: "https://one.example.test")!, token: "fake-one")
        let second = ServerConfig(baseURL: URL(string: "https://two.example.test")!, token: "fake-two")
        try store.save(first)
        try store.save(second)
        #expect(try store.load() == second)
        #expect(keychain.items.count == 2 && keychain.items["other.service"] == "someone-else")
        try store.clear()
        #expect(try store.load() == nil)
        #expect(keychain.items == ["other.service": "someone-else"])
    }

    @Test(.enabled(if: KeychainTests.enabled(), "Set VOXT_KEYCHAIN_TESTS=1 to use the login Keychain"))
    func keychainRoundTripUpdatesAndDeletesOnlyItsOwnItem() throws {
        let store = KeychainTokenStore(service: "com.example.voxt.test.\(UUID())")
        defer { try? store.delete() }
        #expect(try store.read() == nil)
        try store.write("fake-one")
        #expect(try store.read() == "fake-one")
        try store.write("fake-two")
        #expect(try store.read() == "fake-two")
        try store.delete()
        #expect(try store.read() == nil)
    }

}
