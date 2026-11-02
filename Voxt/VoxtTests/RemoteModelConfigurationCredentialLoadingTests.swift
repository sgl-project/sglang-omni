import XCTest
@testable import Voxt

final class RemoteModelConfigurationCredentialLoadingTests: RemoteModelConfigurationTestCase {
    func testLoadSaveRoundTripPreservesConfigurations() {
        let stored: [String: RemoteProviderConfiguration] = [
            RemoteASRProvider.openAIWhisper.rawValue: TestFactories.makeRemoteConfiguration(
                providerID: RemoteASRProvider.openAIWhisper.rawValue,
                model: "whisper-1",
                endpoint: "https://example.com/asr",
                apiKey: "secret"
            ),
            RemoteASRProvider.doubaoASR.rawValue: TestFactories.makeRemoteConfiguration(
                providerID: RemoteASRProvider.doubaoASR.rawValue,
                model: DoubaoASRConfiguration.modelV2,
                appID: "app-id",
                accessToken: "token",
                doubaoDictionaryMode: DoubaoDictionaryMode.off.rawValue,
                doubaoEnableRequestHotwords: false,
                doubaoEnableRequestCorrections: false
            ),
            RemoteLLMProvider.openAI.rawValue: TestFactories.makeRemoteConfiguration(
                providerID: RemoteLLMProvider.openAI.rawValue,
                model: "gpt-5.2",
                endpoint: "https://example.com/llm",
                apiKey: "secret",
                openAIReasoningEffort: OpenAIReasoningEffort.high.rawValue,
                openAITextVerbosity: OpenAITextVerbosity.low.rawValue,
                openAIMaxOutputTokens: 4096
            )
        ]

        let raw = RemoteModelConfigurationStore.saveConfigurations(stored)
        VoxtSecureStorage.clearCacheForTesting()
        let roundTrip = RemoteModelConfigurationStore.loadConfigurations(from: raw)

        XCTAssertFalse(raw.contains("secret"))
        XCTAssertFalse(raw.contains("app-id"))
        XCTAssertFalse(raw.contains("token"))
        XCTAssertEqual(roundTrip, stored)
    }

    func testMetadataOnlyLoadDoesNotResolveStoredSensitiveValues() {
        let stored: [String: RemoteProviderConfiguration] = [
            RemoteLLMProvider.openAI.rawValue: TestFactories.makeRemoteConfiguration(
                providerID: RemoteLLMProvider.openAI.rawValue,
                model: "gpt-5.2",
                endpoint: "https://example.com/llm",
                apiKey: "secret"
            )
        ]

        let raw = RemoteModelConfigurationStore.saveConfigurations(stored)
        let metadataOnly = RemoteModelConfigurationStore.loadConfigurations(
            from: raw,
            sensitiveValueLoading: .metadataOnly
        )

        XCTAssertEqual(metadataOnly[RemoteLLMProvider.openAI.rawValue]?.model, "gpt-5.2")
        XCTAssertEqual(metadataOnly[RemoteLLMProvider.openAI.rawValue]?.endpoint, "https://example.com/llm")
        XCTAssertTrue(metadataOnly[RemoteLLMProvider.openAI.rawValue]?.apiKey.isEmpty ?? false)
        XCTAssertFalse(raw.contains("__stored__"))
        XCTAssertTrue(metadataOnly[RemoteLLMProvider.openAI.rawValue]?.isConfigured ?? false)
    }

    func testResponsesRequestResolvesMetadataOnlyCredentialAtNetworkBoundary() throws {
        let provider = RemoteLLMProvider.volcengine
        let stored = TestFactories.makeRemoteConfiguration(
            providerID: provider.rawValue,
            model: "doubao-seed-2-0-mini-260215",
            endpoint: "https://ark.cn-beijing.volces.com/api/v3/responses",
            apiKey: "runtime-secret"
        )
        let raw = RemoteModelConfigurationStore.saveConfigurations([provider.rawValue: stored])
        let metadataOnly = try XCTUnwrap(
            RemoteModelConfigurationStore.loadConfiguration(
                providerID: provider.rawValue,
                from: raw,
                sensitiveValueLoading: .metadataOnly
            )
        )
        XCTAssertTrue(metadataOnly.apiKey.isEmpty)

        let request = try RemoteLLMRuntimeClient().makeResponsesRequest(
            provider: provider,
            endpointValue: stored.endpoint,
            model: stored.model,
            systemPrompt: "Answer the user.",
            inputPayload: "Hello",
            configuration: metadataOnly,
            previousResponseID: nil,
            tuning: .init(maxTokens: 128, temperature: 0.1, topP: 0.3),
            textFormat: nil,
            streamingEnabled: false
        )

        XCTAssertEqual(
            request.value(forHTTPHeaderField: "Authorization"),
            "Bearer runtime-secret"
        )
    }

    func testRuntimeRequestRejectsMetadataWhenStoredCredentialIsMissing() throws {
        let provider = RemoteLLMProvider.volcengine
        let stored = TestFactories.makeRemoteConfiguration(
            providerID: provider.rawValue,
            model: "doubao-seed-2-0-mini-260215",
            apiKey: "runtime-secret"
        )
        let raw = RemoteModelConfigurationStore.saveConfigurations([provider.rawValue: stored])
        let metadataOnly = try XCTUnwrap(
            RemoteModelConfigurationStore.loadConfiguration(
                providerID: provider.rawValue,
                from: raw,
                sensitiveValueLoading: .metadataOnly
            )
        )
        VoxtSecureStorage.removeProtectedValueForTesting(
            for: "remote-provider.\(provider.rawValue).credentials"
        )

        XCTAssertThrowsError(
            try RemoteModelConfigurationStore.runtimeConfiguration(for: metadataOnly)
        ) { error in
            XCTAssertEqual(
                error as? RemoteModelConfigurationStore.RuntimeCredentialError,
                .missing
            )
        }
    }

    func testRuntimeRequestRejectsCorruptedBundledCredential() throws {
        let provider = RemoteLLMProvider.volcengine
        let stored = TestFactories.makeRemoteConfiguration(
            providerID: provider.rawValue,
            model: "doubao-seed-2-0-mini-260215",
            apiKey: "runtime-secret"
        )
        let raw = RemoteModelConfigurationStore.saveConfigurations([provider.rawValue: stored])
        let metadataOnly = try XCTUnwrap(
            RemoteModelConfigurationStore.loadConfiguration(
                providerID: provider.rawValue,
                from: raw,
                sensitiveValueLoading: .metadataOnly
            )
        )
        try VoxtSecureStorage.setProtectedString(
            "not-json",
            for: "remote-provider.\(provider.rawValue).credentials"
        )

        XCTAssertThrowsError(
            try RemoteModelConfigurationStore.runtimeConfiguration(for: metadataOnly)
        ) { error in
            XCTAssertEqual(
                error as? RemoteModelConfigurationStore.RuntimeCredentialError,
                .corrupted
            )
        }
    }

    func testTargetedLoadResolvesOnlyRequestedProviderConfiguration() {
        let stored: [String: RemoteProviderConfiguration] = [
            RemoteASRProvider.doubaoASR.rawValue: TestFactories.makeRemoteConfiguration(
                providerID: RemoteASRProvider.doubaoASR.rawValue,
                model: DoubaoASRConfiguration.modelV2,
                appID: "doubao-app",
                accessToken: "doubao-token"
            ),
            RemoteASRProvider.aliyunBailianASR.rawValue: TestFactories.makeRemoteConfiguration(
                providerID: RemoteASRProvider.aliyunBailianASR.rawValue,
                model: "fun-asr-realtime",
                apiKey: "aliyun-key"
            )
        ]

        let raw = RemoteModelConfigurationStore.saveConfigurations(stored)
        let doubao = RemoteModelConfigurationStore.loadConfiguration(
            providerID: RemoteASRProvider.doubaoASR.rawValue,
            from: raw
        )

        XCTAssertEqual(doubao?.providerID, RemoteASRProvider.doubaoASR.rawValue)
        XCTAssertEqual(doubao?.appID, "doubao-app")
        XCTAssertEqual(doubao?.accessToken, "doubao-token")
    }

    func testTargetedMetadataLoadDoesNotReturnStoredCredential() {
        let stored = TestFactories.makeRemoteConfiguration(
            providerID: RemoteASRProvider.doubaoASR.rawValue,
            model: DoubaoASRConfiguration.modelV2,
            appID: "doubao-app",
            accessToken: "doubao-token"
        )

        let raw = RemoteModelConfigurationStore.saveConfigurations([stored.providerID: stored])
        let metadata = RemoteModelConfigurationStore.loadConfiguration(
            providerID: stored.providerID,
            from: raw,
            sensitiveValueLoading: .metadataOnly
        )

        XCTAssertTrue(metadata?.appID.isEmpty ?? false)
        XCTAssertTrue(metadata?.accessToken.isEmpty ?? false)
        XCTAssertTrue(metadata?.isConfigured ?? false)
    }

    func testMetadataOnlyLoadPreservesExactBundledCredentialFieldPresence() {
        let stored = TestFactories.makeRemoteConfiguration(
            providerID: RemoteASRProvider.doubaoASR.rawValue,
            model: DoubaoASRConfiguration.modelV2,
            appID: "doubao-app",
            accessToken: ""
        )

        let raw = RemoteModelConfigurationStore.saveConfigurations([stored.providerID: stored])
        let metadata = RemoteModelConfigurationStore.loadConfiguration(
            providerID: stored.providerID,
            from: raw,
            sensitiveValueLoading: .metadataOnly
        )

        XCTAssertTrue(raw.contains("storedCredentialPresence"))
        XCTAssertTrue(metadata?.appID.isEmpty ?? false)
        XCTAssertTrue(metadata?.apiKey.isEmpty ?? false)
        XCTAssertTrue(metadata?.accessToken.isEmpty ?? false)
        XCTAssertFalse(metadata?.isConfigured ?? true)
    }
}
