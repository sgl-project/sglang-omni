import XCTest
@testable import Voxt

final class RemoteModelConfigurationCredentialWritingTests: RemoteModelConfigurationTestCase {
    func testSavingMetadataOnlyConfigurationPreservesUneditedStoredCredential() throws {
        let provider = RemoteLLMProvider.openAI
        let stored = TestFactories.makeRemoteConfiguration(
            providerID: provider.rawValue,
            model: "gpt-5.2",
            apiKey: "stored-secret"
        )
        let raw = RemoteModelConfigurationStore.saveConfigurations([provider.rawValue: stored])
        var metadata = try XCTUnwrap(
            RemoteModelConfigurationStore.loadConfiguration(
                providerID: provider.rawValue,
                from: raw,
                sensitiveValueLoading: .metadataOnly
            )
        )
        metadata.model = "gpt-5.3"

        let updatedRaw = try RemoteModelConfigurationStore.saveConfiguration(
            metadata,
            updating: raw
        ).get()
        let reloaded = try XCTUnwrap(
            RemoteModelConfigurationStore.loadConfiguration(
                providerID: provider.rawValue,
                from: updatedRaw
            )
        )

        XCTAssertEqual(reloaded.model, "gpt-5.3")
        XCTAssertEqual(reloaded.apiKey, "stored-secret")
        XCTAssertFalse(updatedRaw.contains("stored-secret"))
    }

    func testCredentialEditIntentCanExplicitlyClearStoredCredential() throws {
        let provider = RemoteLLMProvider.openAI
        let stored = TestFactories.makeRemoteConfiguration(
            providerID: provider.rawValue,
            model: "gpt-5.2",
            apiKey: "stored-secret"
        )
        let raw = RemoteModelConfigurationStore.saveConfigurations([provider.rawValue: stored])
        let metadata = try XCTUnwrap(
            RemoteModelConfigurationStore.loadConfiguration(
                providerID: provider.rawValue,
                from: raw,
                sensitiveValueLoading: .metadataOnly
            )
        )
        let cleared = metadata.applyingCredentialEditIntent(
            from: metadata,
            editedFields: [.apiKey]
        )

        let updatedRaw = try RemoteModelConfigurationStore.saveConfiguration(
            cleared,
            updating: raw
        ).get()
        let reloaded = try XCTUnwrap(
            RemoteModelConfigurationStore.loadConfiguration(
                providerID: provider.rawValue,
                from: updatedRaw
            )
        )

        XCTAssertTrue(reloaded.apiKey.isEmpty)
        XCTAssertFalse(reloaded.isConfigured)
        XCTAssertFalse(
            VoxtSecureStorage.hasProtectedValueForTesting(
                for: "remote-provider.\(provider.rawValue).credentials"
            )
        )
    }

    func testReplacingOneCredentialFieldPreservesOtherStoredFields() throws {
        let providerID = RemoteASRProvider.doubaoASR.rawValue
        let stored = TestFactories.makeRemoteConfiguration(
            providerID: providerID,
            model: DoubaoASRConfiguration.modelV2,
            appID: "stored-app-id",
            accessToken: "old-token"
        )
        let raw = RemoteModelConfigurationStore.saveConfigurations([providerID: stored])
        let metadata = try XCTUnwrap(
            RemoteModelConfigurationStore.loadConfiguration(
                providerID: providerID,
                from: raw,
                sensitiveValueLoading: .metadataOnly
            )
        )
        var draft = metadata
        draft.accessToken = "new-token"
        draft = draft.applyingCredentialEditIntent(
            from: metadata,
            editedFields: [.accessToken]
        )

        let runtimeDraft = try RemoteModelConfigurationStore.runtimeConfiguration(for: draft).value
        XCTAssertEqual(runtimeDraft.appID, "stored-app-id")
        XCTAssertEqual(runtimeDraft.accessToken, "new-token")

        let updatedRaw = try RemoteModelConfigurationStore.saveConfiguration(
            draft,
            updating: raw
        ).get()
        let reloaded = try XCTUnwrap(
            RemoteModelConfigurationStore.loadConfiguration(providerID: providerID, from: updatedRaw)
        )

        XCTAssertEqual(reloaded.appID, "stored-app-id")
        XCTAssertEqual(reloaded.accessToken, "new-token")
    }

    func testSavingPreservedCredentialFailsClosedWhenStoredValueIsMissing() throws {
        let provider = RemoteLLMProvider.openAI
        let account = "remote-provider.\(provider.rawValue).credentials"
        let stored = TestFactories.makeRemoteConfiguration(
            providerID: provider.rawValue,
            model: "gpt-5.2",
            apiKey: "stored-secret"
        )
        let raw = RemoteModelConfigurationStore.saveConfigurations([provider.rawValue: stored])
        var metadata = try XCTUnwrap(
            RemoteModelConfigurationStore.loadConfiguration(
                providerID: provider.rawValue,
                from: raw,
                sensitiveValueLoading: .metadataOnly
            )
        )
        metadata.model = "gpt-5.3"
        VoxtSecureStorage.removeProtectedValueForTesting(for: account)

        let result = RemoteModelConfigurationStore.saveConfiguration(metadata, updating: raw)

        XCTAssertEqual(result, .failure(.secureStorageUnavailable))
        XCTAssertFalse(VoxtSecureStorage.hasProtectedValueForTesting(for: account))
    }

    func testStaleCredentialPresenceIsCorrectedOnceAndOneSaveRestoresStableStorage() throws {
        let suiteName = "RemoteModelConfigurationTests.staleCredentialPresence.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }

        let providerID = RemoteASRProvider.doubaoASR.rawValue
        let account = "remote-provider.\(providerID).credentials"
        let configured = TestFactories.makeRemoteConfiguration(
            providerID: providerID,
            model: DoubaoASRConfiguration.modelV2,
            appID: "doubao-app",
            accessToken: "doubao-token"
        )
        let staleRaw = RemoteModelConfigurationStore.saveConfigurations([providerID: configured])
        // Reproduce the old failure mode: the item is gone but a process-local
        // cache still contains the value. Migration must trust Keychain state.
        VoxtSecureStorage.removeProtectedValueForTesting(for: account, preservingCache: true)
        defaults.set(staleRaw, forKey: AppPreferenceKey.remoteASRProviderConfigurations)

        RemoteModelConfigurationStore.migrateLegacyStoredSecrets(defaults: defaults)

        let correctedRaw = defaults.string(forKey: AppPreferenceKey.remoteASRProviderConfigurations) ?? ""
        let corrected = try XCTUnwrap(
            RemoteModelConfigurationStore.loadConfiguration(
                providerID: providerID,
                from: correctedRaw,
                sensitiveValueLoading: .metadataOnly
            )
        )
        XCTAssertFalse(corrected.isConfigured)
        XCTAssertEqual(defaults.integer(forKey: AppPreferenceKey.remoteCredentialMigrationVersion), 1)

        let savedRaw = try RemoteModelConfigurationStore.saveConfiguration(
            configured,
            updating: correctedRaw
        ).get()
        VoxtSecureStorage.clearCacheForTesting()
        RemoteModelConfigurationStore.migrateLegacyStoredSecrets(defaults: defaults)
        let reloaded = try XCTUnwrap(
            RemoteModelConfigurationStore.loadConfiguration(providerID: providerID, from: savedRaw)
        )

        XCTAssertEqual(reloaded.appID, "doubao-app")
        XCTAssertEqual(reloaded.accessToken, "doubao-token")
        XCTAssertTrue(VoxtSecureStorage.hasProtectedValueForTesting(for: account))
    }

    func testSaveConfigurationFailurePreservesPreviousMetadata() {
        let existingRaw = RemoteModelConfigurationStore.saveConfigurations([:])
        let updated = TestFactories.makeRemoteConfiguration(
            providerID: RemoteASRProvider.doubaoASR.rawValue,
            model: DoubaoASRConfiguration.modelV2,
            appID: "doubao-app",
            accessToken: "doubao-token"
        )
        VoxtSecureStorage.setProtectedWritesFailForTesting(true)

        let result = RemoteModelConfigurationStore.saveConfiguration(
            updated,
            updating: existingRaw
        )

        XCTAssertEqual(result, .failure(.secureStorageUnavailable))
        XCTAssertFalse(
            VoxtSecureStorage.hasProtectedValueForTesting(
                for: "remote-provider.doubaoASR.credentials"
            )
        )
    }

    func testMetadataEncodingFailureDoesNotCommitCredential() {
        let existingRaw = RemoteModelConfigurationStore.saveConfigurations([:])
        let updated = TestFactories.makeRemoteConfiguration(
            providerID: RemoteLLMProvider.openAI.rawValue,
            model: "gpt-5.2",
            apiKey: "new-secret",
            generationSettings: LLMGenerationSettings(temperature: .nan)
        )

        let result = RemoteModelConfigurationStore.saveConfiguration(
            updated,
            updating: existingRaw
        )

        XCTAssertEqual(result, .failure(.metadataEncodingFailed))
        XCTAssertFalse(
            VoxtSecureStorage.hasProtectedValueForTesting(
                for: "remote-provider.openAI.credentials"
            )
        )
    }

    func testProviderCredentialsUseSingleBundledKeychainItem() {
        let stored = TestFactories.makeRemoteConfiguration(
            providerID: RemoteASRProvider.doubaoASR.rawValue,
            model: DoubaoASRConfiguration.modelV2,
            appID: "doubao-app",
            accessToken: "doubao-token"
        )

        _ = RemoteModelConfigurationStore.saveConfigurations([stored.providerID: stored])
        VoxtSecureStorage.clearCacheForTesting()

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

    func testSavingOneRemoteASRProviderPreservesOtherProviderSecrets() throws {
        let initial: [String: RemoteProviderConfiguration] = [
            RemoteASRProvider.doubaoASR.rawValue: TestFactories.makeRemoteConfiguration(
                providerID: RemoteASRProvider.doubaoASR.rawValue,
                model: DoubaoASRConfiguration.modelV2,
                appID: "doubao-app",
                accessToken: "doubao-token"
            ),
            RemoteASRProvider.aliyunBailianASR.rawValue: TestFactories.makeRemoteConfiguration(
                providerID: RemoteASRProvider.aliyunBailianASR.rawValue,
                model: "fun-asr-realtime",
                endpoint: "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
                apiKey: "aliyun-key"
            )
        ]

        let raw = RemoteModelConfigurationStore.saveConfigurations(initial)
        let updatedAliyun = TestFactories.makeRemoteConfiguration(
            providerID: RemoteASRProvider.aliyunBailianASR.rawValue,
            model: "qwen3-asr-flash-realtime",
            endpoint: "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
            apiKey: "aliyun-key-updated"
        )

        let mergedRaw = try RemoteModelConfigurationStore.saveConfiguration(
            updatedAliyun,
            updating: raw
        ).get()
        let loaded = RemoteModelConfigurationStore.loadConfigurations(from: mergedRaw)

        XCTAssertEqual(loaded[RemoteASRProvider.aliyunBailianASR.rawValue]?.apiKey, "aliyun-key-updated")
        XCTAssertEqual(loaded[RemoteASRProvider.doubaoASR.rawValue]?.appID, "doubao-app")
        XCTAssertEqual(loaded[RemoteASRProvider.doubaoASR.rawValue]?.accessToken, "doubao-token")
    }
}
