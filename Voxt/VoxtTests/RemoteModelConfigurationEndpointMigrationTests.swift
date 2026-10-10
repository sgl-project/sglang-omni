import XCTest
@testable import Voxt

final class RemoteModelConfigurationEndpointMigrationTests: RemoteModelConfigurationTestCase {
    func testDecodeLegacyAliyunEndpointMigratesToResponsesURL() {
        let legacyJSON = """
        [
          {
            "providerID": "aliyunBailian",
            "model": "qwen-plus-latest",
            "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            "apiKey": "",
            "appID": "",
            "accessToken": ""
          }
        ]
        """

        let loaded = RemoteModelConfigurationStore.loadConfigurations(from: legacyJSON)

        XCTAssertEqual(
            loaded["aliyunBailian"]?.endpoint,
            "https://dashscope.aliyuncs.com/compatible-mode/v1/responses"
        )
    }

    func testDecodeLegacyVolcengineEndpointMigratesToResponsesURL() {
        let legacyJSON = """
        [
          {
            "providerID": "volcengine",
            "model": "doubao-1-5-pro",
            "endpoint": "https://ark.cn-beijing.volces.com/api/v3/models",
            "apiKey": "",
            "appID": "",
            "accessToken": ""
          }
        ]
        """

        let loaded = RemoteModelConfigurationStore.loadConfigurations(from: legacyJSON)

        XCTAssertEqual(
            loaded["volcengine"]?.endpoint,
            "https://ark.cn-beijing.volces.com/api/v3/responses"
        )
    }

    func testDecodeLegacyOpenAIEndpointMigratesToResponsesURL() {
        let legacyJSON = """
        [
          {
            "providerID": "openAI",
            "model": "gpt-5.2",
            "endpoint": "https://api.openai.com/v1/chat/completions",
            "apiKey": "",
            "appID": "",
            "accessToken": ""
          }
        ]
        """

        let loaded = RemoteModelConfigurationStore.loadConfigurations(from: legacyJSON)

        XCTAssertEqual(
            loaded["openAI"]?.endpoint,
            "https://api.openai.com/v1/responses"
        )
    }

    func testMigrateLegacyLLMEndpointsRewritesPersistedLegacyURLs() {
        let suiteName = "RemoteModelConfigurationTests.migrateLegacyLLMEndpoints.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }

        defaults.set(
            """
            [
              {
                "providerID": "openAI",
                "model": "gpt-5.2",
                "endpoint": "https://api.openai.com/v1/models",
                "apiKey": "",
                "appID": "",
                "accessToken": ""
              },
              {
                "providerID": "aliyunBailian",
                "model": "qwen-plus-latest",
                "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                "apiKey": "",
                "appID": "",
                "accessToken": ""
              },
              {
                "providerID": "volcengine",
                "model": "doubao-1-5-pro",
                "endpoint": "https://ark.cn-beijing.volces.com/api/v3/models",
                "apiKey": "",
                "appID": "",
                "accessToken": ""
              }
            ]
            """,
            forKey: AppPreferenceKey.remoteLLMProviderConfigurations
        )

        RemoteModelConfigurationStore.migrateLegacyLLMEndpoints(defaults: defaults)

        let migrated = RemoteModelConfigurationStore.loadConfigurations(
            from: defaults.string(forKey: AppPreferenceKey.remoteLLMProviderConfigurations) ?? ""
        )

        XCTAssertEqual(
            migrated["openAI"]?.endpoint,
            "https://api.openai.com/v1/responses"
        )
        XCTAssertEqual(
            migrated["aliyunBailian"]?.endpoint,
            "https://dashscope.aliyuncs.com/compatible-mode/v1/responses"
        )
        XCTAssertEqual(
            migrated["volcengine"]?.endpoint,
            "https://ark.cn-beijing.volces.com/api/v3/responses"
        )
    }
}
