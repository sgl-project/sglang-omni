import XCTest
@testable import Voxt

final class RemoteModelConfigurationCodexTests: RemoteModelConfigurationTestCase {
    func testCodexConfigurationUsesLocalLoginAndDoesNotRequireAPIKey() {
        let configuration = TestFactories.makeRemoteConfiguration(
            providerID: RemoteLLMProvider.codex.rawValue,
            model: "gpt-5.4-mini"
        )

        XCTAssertTrue(configuration.isConfigured)
        XCTAssertTrue(RemoteLLMProvider.codex.apiKeyIsOptional)
        XCTAssertTrue(RemoteLLMProvider.codex.usesResponsesAPI)
    }

    func testCodexCredentialProviderReadsLocalAuthFile() async throws {
        let directory = try TemporaryDirectory()
        let token = try makeTestJWT(payload: [
            "exp": Date().addingTimeInterval(3600).timeIntervalSince1970,
            "https://api.openai.com/auth": [
                "chatgpt_account_id": "acct_test"
            ]
        ])
        let authURL = directory.url.appendingPathComponent("auth.json")
        let authData = try JSONSerialization.data(withJSONObject: [
            "auth_mode": "chatgpt",
            "tokens": [
                "access_token": token,
                "refresh_token": "refresh-token"
            ]
        ])
        try authData.write(to: authURL)

        let headers = try await CodexOAuthCredentialProvider(
            environment: ["CODEX_HOME": directory.url.path]
        ).authorizationHeaders()

        XCTAssertEqual(headers["Authorization"], "Bearer \(token)")
        XCTAssertEqual(headers["ChatGPT-Account-ID"], "acct_test")
        XCTAssertEqual(headers["originator"], "codex_cli_rs")
    }

    func testCodexCredentialProviderReadsSelectedAuthFilePath() async throws {
        let directory = try TemporaryDirectory()
        let token = try makeTestJWT(payload: [
            "exp": Date().addingTimeInterval(3600).timeIntervalSince1970
        ])
        let authURL = directory.url.appendingPathComponent("selected-auth.json")
        let authData = try JSONSerialization.data(withJSONObject: [
            "auth_mode": "chatgpt",
            "tokens": [
                "access_token": token,
                "refresh_token": "refresh-token",
                "account_id": "acct_selected"
            ]
        ])
        try authData.write(to: authURL)

        let headers = try await CodexOAuthCredentialProvider(
            environment: ["CODEX_HOME": "/missing-codex-home"],
            authFilePath: authURL.path
        ).authorizationHeaders()

        XCTAssertEqual(headers["Authorization"], "Bearer \(token)")
        XCTAssertEqual(headers["ChatGPT-Account-ID"], "acct_selected")
    }

    func testCodexCredentialProviderReportsAuthFilePermissionDenied() async throws {
        let directory = try TemporaryDirectory()
        let authURL = directory.url.appendingPathComponent("auth.json")
        try Data("{}".utf8).write(to: authURL)
        try FileManager.default.setAttributes([.posixPermissions: 0o000], ofItemAtPath: authURL.path)
        defer {
            try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: authURL.path)
        }

        do {
            _ = try await CodexOAuthCredentialProvider(authFilePath: authURL.path).authorizationHeaders()
            XCTFail("Expected auth file permission error")
        } catch CodexOAuthCredentialProvider.CredentialError.authFilePermissionDenied(let path) {
            XCTAssertEqual(path, authURL.path)
        } catch {
            XCTFail("Expected auth file permission error, got \(error)")
        }
    }

    func testCodexConfigurationRoundTripPreservesAuthFileSelection() {
        let bookmark = Data([1, 2, 3, 4])
        let stored: [String: RemoteProviderConfiguration] = [
            RemoteLLMProvider.codex.rawValue: TestFactories.makeRemoteConfiguration(
                providerID: RemoteLLMProvider.codex.rawValue,
                model: "gpt-5.3-codex-spark",
                codexAuthFilePath: "/Users/test/.config/codex/auth.json",
                codexAuthFileBookmark: bookmark,
                codexFastModeEnabled: true
            )
        ]

        let raw = RemoteModelConfigurationStore.saveConfigurations(stored)
        let roundTrip = RemoteModelConfigurationStore.loadConfigurations(from: raw)
        let restored = roundTrip[RemoteLLMProvider.codex.rawValue]

        XCTAssertEqual(restored?.codexAuthFilePath, "/Users/test/.config/codex/auth.json")
        XCTAssertEqual(restored?.codexAuthFileBookmark, bookmark)
        XCTAssertEqual(restored?.codexFastModeEnabled, true)
    }

    func testDecodeLegacyCodexConfigurationDefaultsFastModeToOff() throws {
        let legacyJSON = """
        [
          {
            "providerID": "\(RemoteLLMProvider.codex.rawValue)",
            "model": "gpt-5.4",
            "endpoint": "",
            "apiKey": ""
          }
        ]
        """

        let loaded = RemoteModelConfigurationStore.loadConfigurations(from: legacyJSON)

        XCTAssertEqual(loaded[RemoteLLMProvider.codex.rawValue]?.codexFastModeEnabled, false)
    }

    func testCodexCredentialProviderUsesUserHomeOutsideAppContainer() {
        let provider = CodexOAuthCredentialProvider(
            environment: [
                "HOME": "/Users/test/Library/Containers/com.voxt.Voxt/Data"
            ],
            userHomeDirectory: "/Users/test"
        )

        XCTAssertEqual(provider.authFilePath(), "/Users/test/.codex/auth.json")
    }

    func testCodexCredentialProviderExpandsCodexHomeWithUserHome() {
        let provider = CodexOAuthCredentialProvider(
            environment: [
                "CODEX_HOME": "~/.config/codex",
                "HOME": "/Users/test/Library/Containers/com.voxt.Voxt/Data"
            ],
            userHomeDirectory: "/Users/test"
        )

        XCTAssertEqual(provider.authFilePath(), "/Users/test/.config/codex/auth.json")
    }

    func testCodexModelCatalogDefaultsToCurrentModel() {
        XCTAssertEqual(RemoteLLMProvider.codex.suggestedModel, "gpt-5.4")

        let ids = RemoteLLMProvider.codex.modelOptions.map(\.id)
        XCTAssertEqual(ids.first, "gpt-5.4")
        XCTAssertTrue(ids.contains("gpt-5.3-codex-spark"))
        XCTAssertTrue(ids.contains("gpt-5.4-mini"))
    }

    func testCodexModelCatalogUsesCurrentFallbackPresets() {
        XCTAssertEqual(RemoteLLMProvider.codex.suggestedModel, "gpt-5.4")

        let latestIDs = RemoteLLMProvider.codex.latestModelOptions.map(\.id)
        XCTAssertEqual(latestIDs.first, "gpt-5.4")
        XCTAssertTrue(latestIDs.contains("gpt-5.5"))
        XCTAssertTrue(latestIDs.contains("gpt-5.3-codex"))
        XCTAssertTrue(latestIDs.contains("gpt-5-codex-mini"))
        XCTAssertTrue(latestIDs.contains("gpt-oss-120b"))
    }

    func testCodexGenerationCapabilitiesMatchCodexBackend() {
        let capabilities = LLMProviderCapabilityRegistry.capabilities(for: .codex)

        XCTAssertFalse(capabilities.supportsThinkingEffort)
        XCTAssertFalse(capabilities.supportsResponseFormat)
        XCTAssertFalse(capabilities.supportsMaxOutputTokens)
        XCTAssertFalse(capabilities.supportsTemperature)
        XCTAssertFalse(capabilities.supportsTopP)
        XCTAssertFalse(capabilities.supportsLogprobs)
        XCTAssertFalse(capabilities.supportsStopSequences)
        XCTAssertFalse(capabilities.supportsExtraBody)
    }

    private func makeTestJWT(payload: [String: Any]) throws -> String {
        let headerData = try JSONSerialization.data(withJSONObject: ["alg": "none", "typ": "JWT"])
        let payloadData = try JSONSerialization.data(withJSONObject: payload)
        return "\(base64URLEncoded(headerData)).\(base64URLEncoded(payloadData)).signature"
    }

    private func base64URLEncoded(_ data: Data) -> String {
        data.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }
}
