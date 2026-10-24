import XCTest
@testable import Voxt

final class RemoteModelConfigurationCredentialMigrationTests: RemoteModelConfigurationTestCase {
    func testMetadataOnlyLoadDerivesExactPresenceFromLegacyBundleWithoutMask() throws {
        let bundledValues = """
        {"apiKey":"","appID":"legacy-app","accessToken":""}
        """
        XCTAssertTrue(
            VoxtSecureStorage.set(
                bundledValues,
                for: "remote-provider.doubaoASR.credentials"
            )
        )

        let legacyRaw = """
        [
          {
            "providerID": "doubaoASR",
            "model": "\(DoubaoASRConfiguration.modelV2)",
            "endpoint": "",
            "apiKey": "",
            "appID": "",
            "accessToken": ""
          }
        ]
        """

        let metadata = try XCTUnwrap(
            RemoteModelConfigurationStore.loadConfiguration(
                providerID: RemoteASRProvider.doubaoASR.rawValue,
                from: legacyRaw,
                sensitiveValueLoading: .metadataOnly
            )
        )

        XCTAssertTrue(metadata.appID.isEmpty)
        XCTAssertTrue(metadata.apiKey.isEmpty)
        XCTAssertTrue(metadata.accessToken.isEmpty)
        XCTAssertFalse(metadata.isConfigured)
    }

    func testLegacyBundleMigrationPersistsCredentialPresenceMetadata() {
        let suiteName = "RemoteModelConfigurationTests.credentialPresence.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }

        XCTAssertTrue(
            VoxtSecureStorage.set(
                "{\"apiKey\":\"\",\"appID\":\"legacy-app\",\"accessToken\":\"\"}",
                for: "remote-provider.doubaoASR.credentials"
            )
        )
        defaults.set(
            """
            [{"providerID":"doubaoASR","model":"\(DoubaoASRConfiguration.modelV2)","apiKey":"","appID":"","accessToken":""}]
            """,
            forKey: AppPreferenceKey.remoteASRProviderConfigurations
        )

        RemoteModelConfigurationStore.migrateLegacyStoredSecrets(defaults: defaults)

        let migratedRaw = defaults.string(forKey: AppPreferenceKey.remoteASRProviderConfigurations) ?? ""
        let metadata = RemoteModelConfigurationStore.loadConfiguration(
            providerID: RemoteASRProvider.doubaoASR.rawValue,
            from: migratedRaw,
            sensitiveValueLoading: .metadataOnly
        )
        XCTAssertTrue(migratedRaw.contains("storedCredentialPresence"))
        XCTAssertTrue(metadata?.appID.isEmpty ?? false)
        XCTAssertTrue(metadata?.apiKey.isEmpty ?? false)
        XCTAssertTrue(metadata?.accessToken.isEmpty ?? false)
    }

    func testLegacyBundleProtectedWriteFailureKeepsCredentialAvailableAndRetriesLater() {
        let suiteName = "RemoteModelConfigurationTests.credentialPresenceRetry.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }

        let account = "remote-provider.doubaoASR.credentials"
        let legacyRaw = """
        [{"providerID":"doubaoASR","model":"\(DoubaoASRConfiguration.modelV2)","apiKey":"","appID":"","accessToken":""}]
        """
        VoxtSecureStorage.setLegacyValueForTesting(
            "{\"apiKey\":\"\",\"appID\":\"legacy-app\",\"accessToken\":\"legacy-token\"}",
            for: account
        )
        VoxtSecureStorage.setProtectedWritesFailForTesting(true)
        defaults.set(legacyRaw, forKey: AppPreferenceKey.remoteASRProviderConfigurations)

        RemoteModelConfigurationStore.migrateLegacyStoredSecrets(defaults: defaults)

        let migratedRaw = defaults.string(
            forKey: AppPreferenceKey.remoteASRProviderConfigurations
        ) ?? ""
        XCTAssertEqual(migratedRaw, legacyRaw)
        XCTAssertEqual(defaults.integer(forKey: AppPreferenceKey.remoteCredentialMigrationVersion), 0)
        XCTAssertTrue(VoxtSecureStorage.hasLegacyValueForTesting(for: account))
        XCTAssertFalse(VoxtSecureStorage.hasProtectedValueForTesting(for: account))

        VoxtSecureStorage.setProtectedWritesFailForTesting(false)
        VoxtSecureStorage.clearCacheForTesting()
        RemoteModelConfigurationStore.migrateLegacyStoredSecrets(defaults: defaults)

        let retriedRaw = defaults.string(forKey: AppPreferenceKey.remoteASRProviderConfigurations) ?? ""
        let retried = RemoteModelConfigurationStore.loadConfiguration(
            providerID: RemoteASRProvider.doubaoASR.rawValue,
            from: retriedRaw
        )
        XCTAssertTrue(retriedRaw.contains("storedCredentialPresence"))
        XCTAssertEqual(retried?.appID, "legacy-app")
        XCTAssertEqual(retried?.accessToken, "legacy-token")
        XCTAssertFalse(VoxtSecureStorage.hasLegacyValueForTesting(for: account))
        XCTAssertTrue(VoxtSecureStorage.hasProtectedValueForTesting(for: account))
        XCTAssertEqual(defaults.integer(forKey: AppPreferenceKey.remoteCredentialMigrationVersion), 1)
    }

    func testLegacyBundleCleanupFailureDoesNotHideCredentialOnFirstLoad() throws {
        let account = "remote-provider.doubaoASR.credentials"
        let raw = """
        [{"providerID":"doubaoASR","model":"\(DoubaoASRConfiguration.modelV2)","apiKey":"","appID":"","accessToken":""}]
        """
        VoxtSecureStorage.setLegacyValueForTesting(
            "{\"apiKey\":\"\",\"appID\":\"legacy-app\",\"accessToken\":\"legacy-token\"}",
            for: account
        )
        VoxtSecureStorage.setDeletesFailForTesting(true)

        let firstLoad = try XCTUnwrap(
            RemoteModelConfigurationStore.loadConfiguration(
                providerID: RemoteASRProvider.doubaoASR.rawValue,
                from: raw
            )
        )

        XCTAssertEqual(firstLoad.appID, "legacy-app")
        XCTAssertEqual(firstLoad.accessToken, "legacy-token")
        XCTAssertTrue(VoxtSecureStorage.hasProtectedValueForTesting(for: account))
        XCTAssertTrue(VoxtSecureStorage.hasLegacyValueForTesting(for: account))

        VoxtSecureStorage.setDeletesFailForTesting(false)
        VoxtSecureStorage.clearCacheForTesting()

        let retriedLoad = try XCTUnwrap(
            RemoteModelConfigurationStore.loadConfiguration(
                providerID: RemoteASRProvider.doubaoASR.rawValue,
                from: raw
            )
        )
        XCTAssertEqual(retriedLoad.appID, "legacy-app")
        XCTAssertEqual(retriedLoad.accessToken, "legacy-token")
        XCTAssertFalse(VoxtSecureStorage.hasLegacyValueForTesting(for: account))
    }

    func testLegacyCredentialCleanupFailureKeepsBundledSaveConsistent() throws {
        let stored = TestFactories.makeRemoteConfiguration(
            providerID: RemoteASRProvider.doubaoASR.rawValue,
            model: DoubaoASRConfiguration.modelV2,
            appID: "doubao-app",
            accessToken: "doubao-token"
        )
        let existingRaw = RemoteModelConfigurationStore.saveConfigurations([stored.providerID: stored])
        let cleared = TestFactories.makeRemoteConfiguration(
            providerID: stored.providerID,
            model: stored.model
        )
        VoxtSecureStorage.setLegacyValueForTesting(
            "legacy-app",
            for: "remote-provider.doubaoASR.appID"
        )
        VoxtSecureStorage.setLegacyValueForTesting(
            "legacy-token",
            for: "remote-provider.doubaoASR.accessToken"
        )
        VoxtSecureStorage.setDeletesFailForTesting(true)

        let clearedRaw = try RemoteModelConfigurationStore.saveConfiguration(
            cleared,
            updating: existingRaw
        ).get()

        XCTAssertNotEqual(clearedRaw, existingRaw)
        let resolved = try XCTUnwrap(
            RemoteModelConfigurationStore.loadConfiguration(providerID: stored.providerID, from: clearedRaw)
        )
        XCTAssertTrue(resolved.appID.isEmpty)
        XCTAssertTrue(resolved.accessToken.isEmpty)
        XCTAssertFalse(resolved.isConfigured)
        XCTAssertTrue(
            VoxtSecureStorage.hasLegacyValueForTesting(
                for: "remote-provider.doubaoASR.appID"
            )
        )
        XCTAssertTrue(
            VoxtSecureStorage.hasLegacyValueForTesting(
                for: "remote-provider.doubaoASR.accessToken"
            )
        )

        VoxtSecureStorage.setDeletesFailForTesting(false)
        let cleanedRaw = try RemoteModelConfigurationStore.saveConfiguration(
            cleared,
            updating: clearedRaw
        ).get()
        let cleaned = try XCTUnwrap(
            RemoteModelConfigurationStore.loadConfiguration(providerID: stored.providerID, from: cleanedRaw)
        )

        XCTAssertTrue(cleaned.appID.isEmpty)
        XCTAssertTrue(cleaned.accessToken.isEmpty)
        XCTAssertFalse(
            VoxtSecureStorage.hasString(
                for: "remote-provider.doubaoASR.credentials"
            )
        )
        XCTAssertFalse(
            VoxtSecureStorage.hasLegacyValueForTesting(
                for: "remote-provider.doubaoASR.appID"
            )
        )
        XCTAssertFalse(
            VoxtSecureStorage.hasLegacyValueForTesting(
                for: "remote-provider.doubaoASR.accessToken"
            )
        )
    }

    func testLegacyCleanupFailureDoesNotPairNewCredentialWithOldMetadata() throws {
        let providerID = RemoteASRProvider.aliyunBailianASR.rawValue
        let initial = TestFactories.makeRemoteConfiguration(
            providerID: providerID,
            model: "fun-asr-realtime",
            endpoint: "wss://old.example.com/realtime",
            apiKey: "old-key"
        )
        let existingRaw = RemoteModelConfigurationStore.saveConfigurations([providerID: initial])
        let updated = TestFactories.makeRemoteConfiguration(
            providerID: providerID,
            model: "qwen3-asr-flash-realtime",
            endpoint: "wss://new.example.com/realtime",
            apiKey: "new-key"
        )
        VoxtSecureStorage.setLegacyValueForTesting(
            "old-key",
            for: "remote-provider.\(providerID).apiKey"
        )
        VoxtSecureStorage.setDeletesFailForTesting(true)

        let updatedRaw = try RemoteModelConfigurationStore.saveConfiguration(
            updated,
            updating: existingRaw
        ).get()
        let resolved = try XCTUnwrap(
            RemoteModelConfigurationStore.loadConfiguration(providerID: providerID, from: updatedRaw)
        )

        XCTAssertEqual(resolved.model, "qwen3-asr-flash-realtime")
        XCTAssertEqual(resolved.endpoint, "wss://new.example.com/realtime")
        XCTAssertEqual(resolved.apiKey, "new-key")
    }

    func testLegacyPerFieldCredentialsMigrateToBundledItemOnFirstRead() throws {
        VoxtSecureStorage.set("legacy-app", for: "remote-provider.doubaoASR.appID")
        VoxtSecureStorage.set("legacy-token", for: "remote-provider.doubaoASR.accessToken")
        VoxtSecureStorage.clearCacheForTesting()

        let raw = """
        [
          {
            "providerID": "doubaoASR",
            "model": "\(DoubaoASRConfiguration.modelV2)",
            "endpoint": "",
            "apiKey": "",
            "appID": "",
            "accessToken": ""
          }
        ]
        """

        let migrated = try XCTUnwrap(
            RemoteModelConfigurationStore.loadConfiguration(
                providerID: RemoteASRProvider.doubaoASR.rawValue,
                from: raw
            )
        )
        VoxtSecureStorage.clearCacheForTesting()

        XCTAssertEqual(migrated.appID, "legacy-app")
        XCTAssertEqual(migrated.accessToken, "legacy-token")
        XCTAssertTrue(
            VoxtSecureStorage.hasString(for: "remote-provider.doubaoASR.credentials")
        )
        XCTAssertFalse(
            VoxtSecureStorage.hasString(for: "remote-provider.doubaoASR.appID")
        )
        XCTAssertFalse(
            VoxtSecureStorage.hasString(for: "remote-provider.doubaoASR.accessToken")
        )
    }

    func testMigrateLegacyStoredSecretsSkipsAlreadySanitizedPayloads() {
        let suiteName = "RemoteModelConfigurationTests.migrateLegacyStoredSecrets.skip.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }

        let raw = """
        [
          {
            "providerID": "openAI",
            "model": "gpt-5.2",
            "endpoint": "https://example.com/responses",
            "apiKey": "",
            "appID": "",
            "accessToken": ""
          }
        ]
        """

        defaults.set(raw, forKey: AppPreferenceKey.remoteLLMProviderConfigurations)

        RemoteModelConfigurationStore.migrateLegacyStoredSecrets(defaults: defaults)

        XCTAssertEqual(
            defaults.string(forKey: AppPreferenceKey.remoteLLMProviderConfigurations),
            raw
        )
        XCTAssertNil(
            VoxtSecureStorage.string(
                for: "remote-provider.openAI.apiKey"
            )
        )
        XCTAssertNil(
            VoxtSecureStorage.string(
                for: "remote-provider.openAI.credentials"
            )
        )
        XCTAssertEqual(defaults.integer(forKey: AppPreferenceKey.remoteCredentialMigrationVersion), 1)
    }
}
