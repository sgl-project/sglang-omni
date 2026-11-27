import Foundation
import Security

/// The API keys, by account: "llm.<preset>" for a language model service,
/// "asr.remote" for the remote speech service (the macOS app's accounts).
public protocol KeyStore: Sendable {
    func key(_ account: String) -> String?
    func setKey(_ key: String?, for account: String) throws
}

public enum KeyAccount {
    public static func languageModel(_ preset: String) -> String { "llm.\(preset)" }
    public static let speech = "asr.remote"
}

/// The keys in the device's Keychain (service "Voxt"), readable after the
/// first unlock and never synced or backed up off the device.
public struct KeychainStore: KeyStore {
    let service: String

    public init(service: String = "Voxt") {
        self.service = service
    }

    private func query(_ account: String) -> [String: Any] {
        [kSecClass as String: kSecClassGenericPassword,
         kSecAttrService as String: service,
         kSecAttrAccount as String: account]
    }

    public func key(_ account: String) -> String? {
        var q = query(account)
        q[kSecReturnData as String] = true
        q[kSecMatchLimit as String] = kSecMatchLimitOne
        var item: CFTypeRef?
        guard SecItemCopyMatching(q as CFDictionary, &item) == errSecSuccess, let data = item as? Data else {
            return nil
        }
        return String(data: data, encoding: .utf8)
    }

    public func setKey(_ key: String?, for account: String) throws {
        let status = SecItemDelete(query(account) as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw KeychainError(status: status)
        }
        guard let key, !key.isEmpty else {
            return
        }
        var q = query(account)
        q[kSecValueData as String] = Data(key.utf8)
        q[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        let added = SecItemAdd(q as CFDictionary, nil)
        guard added == errSecSuccess else {
            throw KeychainError(status: added)
        }
    }
}

public struct KeychainError: Error, Equatable {
    public var status: OSStatus
}
