// VoxtSecureStorage.swift
// Provides Voxt Secure Storage for permissions and secure storage.

import Foundation
import LocalAuthentication
import Security

enum VoxtSecureStorage {
    enum StorageError: Error, Equatable {
        case invalidUTF8
        case unexpectedStatus(OSStatus)
    }

    enum MigrationLookupResult: Equatable {
        case value(String)
        case missing
        case unavailable
    }

    nonisolated private static let defaultServiceName: String = {
        let bundleID = Bundle.main.bundleIdentifier ?? "com.voxt.Voxt"
        return "\(bundleID).secure-storage"
    }()

    nonisolated private static let stateLock = NSLock()
    nonisolated(unsafe) private static var cachedValues: [String: String] = [:]

    // Unit tests run with code signing disabled in CI. Keep their credentials out
    // of the user's real keychain while exercising the same migration behavior.
    nonisolated(unsafe) private static var usesInMemoryStoreForTesting = isRunningUnderXCTest
    nonisolated(unsafe) private static var protectedValuesForTesting: [String: String] = [:]
    nonisolated(unsafe) private static var legacyValuesForTesting: [String: String] = [:]
    nonisolated(unsafe) private static var protectedWritesFailForTesting = false
    nonisolated(unsafe) private static var deletesFailForTesting = false

    nonisolated private static let isRunningUnderXCTest: Bool = {
        ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil ||
            NSClassFromString("XCTestCase") != nil ||
            NSClassFromString("XCTest.XCTestCase") != nil
    }()

    nonisolated private static var serviceName: String {
        defaultServiceName
    }

    nonisolated static func string(for account: String) -> String? {
        if let cached = cachedValue(for: account) {
            return cached
        }

        if isUsingInMemoryStoreForTesting {
            if let value = protectedValueForTesting(for: account) {
                cache(value, for: account)
                _ = deleteLegacyValueForTesting(for: account)
                return value
            }
            return migrateLegacyValueForTesting(for: account)
        }

        var query = protectedQuery(for: account)
        query[kSecReturnData as String] = kCFBooleanTrue
        query[kSecMatchLimit as String] = kSecMatchLimitOne

        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        switch status {
        case errSecSuccess:
            guard let data = item as? Data,
                  let value = String(data: data, encoding: .utf8)
            else {
                return nil
            }
            cache(value, for: account)
            // A readable Data Protection item is authoritative. Legacy cleanup
            // is best effort and must never make the current value unavailable.
            _ = deleteLegacyValue(for: account, interactionAllowed: false)
            return value
        case errSecItemNotFound:
            return migrateLegacyValue(for: account)
        default:
            VoxtLog.securityWarning("Data Protection Keychain read failed. account=\(account), status=\(status)")
            return nil
        }
    }

    /// Reads only from the authoritative Data Protection Keychain item.
    /// Unlike `string(for:)`, this API never consults the process cache and
    /// never performs legacy migration as a side effect.
    nonisolated static func protectedString(for account: String) throws -> String? {
        if isUsingInMemoryStoreForTesting {
            return protectedValueForTesting(for: account)
        }

        var query = protectedQuery(for: account)
        query[kSecReturnData as String] = kCFBooleanTrue
        query[kSecMatchLimit as String] = kSecMatchLimitOne

        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        switch status {
        case errSecSuccess:
            guard let data = item as? Data,
                  let value = String(data: data, encoding: .utf8)
            else {
                throw StorageError.invalidUTF8
            }
            return value
        case errSecItemNotFound:
            return nil
        default:
            throw StorageError.unexpectedStatus(status)
        }
    }

    /// Upserts an authoritative Data Protection Keychain item. The update
    /// changes only the secret data; accessibility is fixed when adding.
    nonisolated static func setProtectedString(_ value: String, for account: String) throws {
        guard !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            try removeProtectedString(for: account)
            return
        }

        if isUsingInMemoryStoreForTesting {
            guard setProtectedValueForTesting(value, for: account) else {
                throw StorageError.unexpectedStatus(errSecNotAvailable)
            }
            return
        }

        let query = protectedQuery(for: account)
        let attributes: [String: Any] = [
            kSecValueData as String: Data(value.utf8)
        ]
        let updateStatus = SecItemUpdate(query as CFDictionary, attributes as CFDictionary)
        switch updateStatus {
        case errSecSuccess:
            cache(value, for: account)
            return
        case errSecItemNotFound:
            var item = query
            item[kSecValueData as String] = Data(value.utf8)
            item[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlockedThisDeviceOnly
            let addStatus = SecItemAdd(item as CFDictionary, nil)
            if addStatus == errSecDuplicateItem {
                let retryStatus = SecItemUpdate(query as CFDictionary, attributes as CFDictionary)
                guard retryStatus == errSecSuccess else {
                    throw StorageError.unexpectedStatus(retryStatus)
                }
                cache(value, for: account)
                return
            }
            guard addStatus == errSecSuccess else {
                throw StorageError.unexpectedStatus(addStatus)
            }
        default:
            throw StorageError.unexpectedStatus(updateStatus)
        }
        cache(value, for: account)
    }

    nonisolated static func removeProtectedString(for account: String) throws {
        if isUsingInMemoryStoreForTesting {
            stateLock.lock()
            defer { stateLock.unlock() }
            guard !deletesFailForTesting else {
                throw StorageError.unexpectedStatus(errSecNotAvailable)
            }
            protectedValuesForTesting.removeValue(forKey: account)
            cachedValues.removeValue(forKey: account)
            return
        }

        let status = SecItemDelete(protectedQuery(for: account) as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw StorageError.unexpectedStatus(status)
        }
        removeCachedValue(for: account)
    }

    nonisolated static func hasString(for account: String) -> Bool {
        if cachedValue(for: account) != nil {
            return true
        }

        if isUsingInMemoryStoreForTesting {
            return protectedValueForTesting(for: account) != nil || legacyValueForTesting(for: account) != nil
        }

        var protectedLookup = protectedQuery(for: account)
        protectedLookup[kSecMatchLimit as String] = kSecMatchLimitOne
        let protectedStatus = SecItemCopyMatching(protectedLookup as CFDictionary, nil)
        switch protectedStatus {
        case errSecSuccess:
            return true
        case errSecItemNotFound:
            break
        default:
            VoxtLog.securityWarning(
                "Data Protection Keychain presence check failed. account=\(account), status=\(protectedStatus)"
            )
            return false
        }

        var legacyLookup = legacyQuery(for: account, interactionAllowed: false)
        legacyLookup[kSecMatchLimit as String] = kSecMatchLimitOne
        let legacyStatus = SecItemCopyMatching(legacyLookup as CFDictionary, nil)
        switch legacyStatus {
        case errSecSuccess:
            return true
        case errSecItemNotFound, errSecInteractionNotAllowed, errSecAuthFailed:
            return false
        default:
            VoxtLog.securityWarning("Legacy Keychain presence check failed. account=\(account), status=\(legacyStatus)")
            return false
        }
    }

    /// Performs the one-time legacy lookup used by configuration migration.
    /// `missing` is returned only when both stores are definitively empty;
    /// transient Keychain failures remain `unavailable` so callers never erase
    /// configuration metadata based on an inconclusive read.
    nonisolated static func migrationValue(for account: String) -> MigrationLookupResult {
        if isUsingInMemoryStoreForTesting {
            if let value = protectedValueForTesting(for: account) {
                cache(value, for: account)
                _ = deleteLegacyValueForTesting(for: account)
                return .value(value)
            }
            guard let legacyValue = legacyValueForTesting(for: account) else {
                return .missing
            }
            guard !legacyValue.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                return .missing
            }
            guard setProtectedValueForTesting(legacyValue, for: account),
                  protectedValueForTesting(for: account) == legacyValue
            else {
                return .unavailable
            }
            _ = deleteLegacyValueForTesting(for: account)
            return .value(legacyValue)
        }

        var protectedLookup = protectedQuery(for: account)
        protectedLookup[kSecReturnData as String] = kCFBooleanTrue
        protectedLookup[kSecMatchLimit as String] = kSecMatchLimitOne
        var protectedItem: CFTypeRef?
        let protectedStatus = SecItemCopyMatching(protectedLookup as CFDictionary, &protectedItem)
        switch protectedStatus {
        case errSecSuccess:
            guard let data = protectedItem as? Data,
                  let value = String(data: data, encoding: .utf8)
            else {
                return .unavailable
            }
            cache(value, for: account)
            _ = deleteLegacyValue(for: account, interactionAllowed: false)
            return .value(value)
        case errSecItemNotFound:
            break
        default:
            VoxtLog.securityWarning(
                "Data Protection Keychain migration lookup failed. account=\(account), status=\(protectedStatus)"
            )
            return .unavailable
        }

        var legacyLookup = legacyQuery(for: account, interactionAllowed: false)
        legacyLookup[kSecReturnData as String] = kCFBooleanTrue
        legacyLookup[kSecMatchLimit as String] = kSecMatchLimitOne
        var legacyItem: CFTypeRef?
        let legacyStatus = SecItemCopyMatching(legacyLookup as CFDictionary, &legacyItem)
        switch legacyStatus {
        case errSecSuccess:
            guard let legacyData = legacyItem as? Data,
                  let value = String(data: legacyData, encoding: .utf8)
            else {
                return .unavailable
            }
            guard !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                return .missing
            }
            guard set(value, for: account), protectedData(for: account) == legacyData else {
                VoxtLog.securityWarning("Legacy Keychain migration could not be committed. account=\(account)")
                return .unavailable
            }
            if !deleteLegacyValue(for: account, interactionAllowed: false) {
                VoxtLog.securityWarning("Legacy Keychain migration cleanup failed. account=\(account)")
            }
            return .value(value)
        case errSecItemNotFound:
            return .missing
        case errSecInteractionNotAllowed, errSecAuthFailed:
            VoxtLog.securityWarning(
                "Legacy Keychain migration is temporarily unavailable. account=\(account), status=\(legacyStatus)"
            )
            return .unavailable
        default:
            VoxtLog.securityWarning("Legacy Keychain migration lookup failed. account=\(account), status=\(legacyStatus)")
            return .unavailable
        }
    }

    @discardableResult
    nonisolated static func set(_ value: String, for account: String) -> Bool {
        guard !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return removeValue(for: account)
        }

        if isUsingInMemoryStoreForTesting {
            return setProtectedValueForTesting(value, for: account)
        }

        let query = protectedQuery(for: account)
        let status = SecItemCopyMatching(query as CFDictionary, nil)
        switch status {
        case errSecSuccess:
            let attributes: [String: Any] = [
                kSecValueData as String: Data(value.utf8),
                kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
            ]
            let updateStatus = SecItemUpdate(query as CFDictionary, attributes as CFDictionary)
            guard updateStatus == errSecSuccess else {
                VoxtLog.securityWarning(
                    "Data Protection Keychain update failed. account=\(account), status=\(updateStatus)"
                )
                return false
            }
        case errSecItemNotFound:
            var item = query
            item[kSecValueData as String] = Data(value.utf8)
            item[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlockedThisDeviceOnly
            let addStatus = SecItemAdd(item as CFDictionary, nil)
            guard addStatus == errSecSuccess else {
                VoxtLog.securityWarning("Data Protection Keychain add failed. account=\(account), status=\(addStatus)")
                return false
            }
        default:
            VoxtLog.securityWarning(
                "Data Protection Keychain lookup before write failed. account=\(account), status=\(status)"
            )
            return false
        }

        cache(value, for: account)
        return true
    }

    @discardableResult
    nonisolated static func removeValue(for account: String) -> Bool {
        removeCachedValue(for: account)

        if isUsingInMemoryStoreForTesting {
            return removeValuesForTesting(for: account)
        }

        // Delete the legacy copy first. If its ACL prevents noninteractive
        // deletion, keep the protected copy intact so callers can retry without
        // reviving an older credential.
        guard deleteLegacyValue(for: account, interactionAllowed: false) else {
            return false
        }
        return deleteProtectedValue(for: account)
    }

    @discardableResult
    nonisolated static func removeValueWithoutUserInteraction(for account: String) -> Bool {
        removeValue(for: account)
    }

    nonisolated static func clearAllForTesting() {
        stateLock.lock()
        usesInMemoryStoreForTesting = true
        cachedValues.removeAll()
        protectedValuesForTesting.removeAll()
        legacyValuesForTesting.removeAll()
        protectedWritesFailForTesting = false
        deletesFailForTesting = false
        stateLock.unlock()
    }

    nonisolated static func clearCacheForTesting() {
        stateLock.lock()
        cachedValues.removeAll()
        stateLock.unlock()
    }

    nonisolated static func setLegacyValueForTesting(_ value: String, for account: String) {
        stateLock.lock()
        usesInMemoryStoreForTesting = true
        legacyValuesForTesting[account] = value
        cachedValues.removeValue(forKey: account)
        stateLock.unlock()
    }

    nonisolated static func hasLegacyValueForTesting(for account: String) -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return legacyValuesForTesting[account] != nil
    }

    nonisolated static func hasProtectedValueForTesting(for account: String) -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return protectedValuesForTesting[account] != nil
    }

    nonisolated static func removeProtectedValueForTesting(
        for account: String,
        preservingCache: Bool = false
    ) {
        stateLock.lock()
        protectedValuesForTesting.removeValue(forKey: account)
        if !preservingCache {
            cachedValues.removeValue(forKey: account)
        }
        stateLock.unlock()
    }

    nonisolated static func setProtectedWritesFailForTesting(_ shouldFail: Bool) {
        stateLock.lock()
        usesInMemoryStoreForTesting = true
        protectedWritesFailForTesting = shouldFail
        stateLock.unlock()
    }

    nonisolated static func setDeletesFailForTesting(_ shouldFail: Bool) {
        stateLock.lock()
        usesInMemoryStoreForTesting = true
        deletesFailForTesting = shouldFail
        stateLock.unlock()
    }

    nonisolated private static func migrateLegacyValue(for account: String) -> String? {
        var query = legacyQuery(for: account, interactionAllowed: false)
        query[kSecReturnData as String] = kCFBooleanTrue
        query[kSecMatchLimit as String] = kSecMatchLimitOne

        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        switch status {
        case errSecSuccess:
            guard let legacyData = item as? Data,
                  let value = String(data: legacyData, encoding: .utf8)
            else {
                VoxtLog.securityWarning("Legacy Keychain value is not valid UTF-8. account=\(account)")
                return nil
            }
            guard set(value, for: account) else {
                VoxtLog.securityWarning("Legacy Keychain migration write failed. account=\(account)")
                cache(value, for: account)
                return value
            }
            guard protectedData(for: account) == legacyData else {
                removeCachedValue(for: account)
                VoxtLog.securityWarning("Legacy Keychain migration verification failed. account=\(account)")
                cache(value, for: account)
                return value
            }

            if !deleteLegacyValue(for: account, interactionAllowed: false) {
                VoxtLog.securityWarning("Legacy Keychain migration cleanup failed. account=\(account)")
            }
            return value
        case errSecItemNotFound:
            return nil
        case errSecInteractionNotAllowed, errSecAuthFailed:
            VoxtLog.securityWarning(
                "Legacy Keychain migration requires user interaction; migration deferred. account=\(account), status=\(status)"
            )
            return nil
        default:
            VoxtLog.securityWarning("Legacy Keychain read failed. account=\(account), status=\(status)")
            return nil
        }
    }

    nonisolated private static func protectedData(for account: String) -> Data? {
        var query = protectedQuery(for: account)
        query[kSecReturnData as String] = kCFBooleanTrue
        query[kSecMatchLimit as String] = kSecMatchLimitOne

        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess else {
            return nil
        }
        return item as? Data
    }

    nonisolated private static func deleteProtectedValue(for account: String) -> Bool {
        let status = SecItemDelete(protectedQuery(for: account) as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            VoxtLog.securityWarning("Data Protection Keychain delete failed. account=\(account), status=\(status)")
            return false
        }
        return true
    }

    nonisolated private static func deleteLegacyValue(for account: String, interactionAllowed: Bool) -> Bool {
        let status = SecItemDelete(
            legacyQuery(for: account, interactionAllowed: interactionAllowed) as CFDictionary
        )
        guard status == errSecSuccess || status == errSecItemNotFound else {
            VoxtLog.securityWarning("Legacy Keychain delete failed. account=\(account), status=\(status)")
            return false
        }
        return true
    }

    nonisolated private static var isUsingInMemoryStoreForTesting: Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return usesInMemoryStoreForTesting
    }

    nonisolated private static func protectedValueForTesting(for account: String) -> String? {
        stateLock.lock()
        defer { stateLock.unlock() }
        return protectedValuesForTesting[account]
    }

    nonisolated private static func legacyValueForTesting(for account: String) -> String? {
        stateLock.lock()
        defer { stateLock.unlock() }
        return legacyValuesForTesting[account]
    }

    nonisolated private static func setProtectedValueForTesting(_ value: String, for account: String) -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        guard !protectedWritesFailForTesting else { return false }
        protectedValuesForTesting[account] = value
        cachedValues[account] = value
        return true
    }

    nonisolated private static func removeValuesForTesting(for account: String) -> Bool {
        stateLock.lock()
        guard !deletesFailForTesting else {
            stateLock.unlock()
            return false
        }
        protectedValuesForTesting.removeValue(forKey: account)
        legacyValuesForTesting.removeValue(forKey: account)
        stateLock.unlock()
        return true
    }

    nonisolated private static func migrateLegacyValueForTesting(for account: String) -> String? {
        guard let legacyValue = legacyValueForTesting(for: account) else {
            return nil
        }
        guard setProtectedValueForTesting(legacyValue, for: account),
              protectedValueForTesting(for: account) == legacyValue
        else {
            cache(legacyValue, for: account)
            return legacyValue
        }

        _ = deleteLegacyValueForTesting(for: account)
        return legacyValue
    }

    nonisolated private static func deleteLegacyValueForTesting(for account: String) -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        guard !deletesFailForTesting else { return false }
        legacyValuesForTesting.removeValue(forKey: account)
        return true
    }

    nonisolated private static func cachedValue(for account: String) -> String? {
        stateLock.lock()
        defer { stateLock.unlock() }
        return cachedValues[account]
    }

    nonisolated private static func cache(_ value: String, for account: String) {
        stateLock.lock()
        cachedValues[account] = value
        stateLock.unlock()
    }

    nonisolated private static func removeCachedValue(for account: String) {
        stateLock.lock()
        cachedValues.removeValue(forKey: account)
        stateLock.unlock()
    }

    nonisolated private static func protectedQuery(for account: String) -> [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: account,
            kSecUseDataProtectionKeychain as String: kCFBooleanTrue as Any
        ]
    }

    nonisolated static func legacyQuery(
        for account: String,
        interactionAllowed: Bool
    ) -> [String: Any] {
        var query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: account,
            // An unqualified macOS Keychain query also matches Data Protection
            // items. Explicitly select the legacy store so cleanup cannot delete
            // the protected item that was just written and leave only the cache.
            kSecUseDataProtectionKeychain as String: kCFBooleanFalse as Any,
        ]
        if !interactionAllowed {
            let context = LAContext()
            context.interactionNotAllowed = true
            query[kSecUseAuthenticationContext as String] = context
        }
        return query
    }
}
