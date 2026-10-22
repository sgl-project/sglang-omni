import Foundation
import Security

/// App-private Keychain item. Background recording can use it after the first unlock;
/// it is never synchronised to iCloud or copied to the keyboard's shared defaults.
public final class KeychainTokenStore: TokenStoring {
    public struct Failure: LocalizedError {
        public let status: OSStatus
        public var errorDescription: String? {
            VoxtL10n.shared.format("Could not access the pairing credential (%d). Unlock your device and try again.", Int(status))
        }
    }

    /// Items are named after the app's bundle ID, so each signer's build has its own. A build
    /// with earlier installs sets `VOXT_KEYCHAIN_SERVICE_PREFIX` to the names those installs
    /// used; a new name would lose every upgraded phone's pairing.
    public static func service(_ name: String, info: [String: Any]? = Bundle.main.infoDictionary) -> String {
        let prefixKey = "VoxtKeychainServicePrefix"
        if let prefix = info?[prefixKey] as? String, !prefix.isEmpty, !prefix.contains("$(") {
            return prefix + "." + name
        }
        return [info?[kCFBundleIdentifierKey as String] as? String, name].compactMap { $0 }.joined(separator: ".")
    }

    private let service: String
    public init(service: String = KeychainTokenStore.service("pairing")) { self.service = service }

    private var query: [String: Any] {
        [kSecClass as String: kSecClassGenericPassword,
         kSecAttrService as String: service,
         kSecAttrAccount as String: "remote-input-token",
         kSecAttrSynchronizable as String: false]
    }

    public func read() throws -> String? {
        var query = query
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        if status == errSecItemNotFound { return nil }
        guard status == errSecSuccess else { throw Failure(status: status) }
        guard let data = item as? Data, let token = String(data: data, encoding: .utf8) else {
            throw Failure(status: errSecDecode)
        }
        return token
    }

    public func write(_ value: String) throws {
        let attributes: [String: Any] = [
            kSecValueData as String: Data(value.utf8),
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        ]
        var status = SecItemUpdate(query as CFDictionary, attributes as CFDictionary)
        if status == errSecItemNotFound {
            status = SecItemAdd(query.merging(attributes) { _, new in new } as CFDictionary, nil)
        }
        guard status == errSecSuccess else { throw Failure(status: status) }
    }

    public func delete() throws {
        let status = SecItemDelete(query as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else { throw Failure(status: status) }
    }
}
