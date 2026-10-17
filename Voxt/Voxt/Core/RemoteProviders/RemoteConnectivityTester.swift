// RemoteConnectivityTester.swift
// Provides Remote Connectivity Tester for remote provider configuration.

import Foundation

enum RemoteProviderTestTarget {
    case asr(RemoteASRProvider)
    case llm(RemoteLLMProvider)
}

struct RemoteProviderConnectivityTester {
    let testTarget: RemoteProviderTestTarget

    func run(configuration: RemoteProviderConfiguration) async throws -> String {
        try await performConnectivityTest(configuration: configuration)
    }

    private func performConnectivityTest(configuration: RemoteProviderConfiguration) async throws -> String {
        let runtimeConfiguration = try RemoteModelConfigurationStore.runtimeConfiguration(
            for: configuration
        )
        let configuration = runtimeConfiguration.value
        let securityContext = switch testTarget {
        case .asr:
            (
                hasCredentials: RemoteEndpointSecurityPolicy.hasExplicitCredentials(configuration),
                allowsWebSocket: true
            )
        case .llm(let provider):
            (
                hasCredentials: RemoteEndpointSecurityPolicy.hasLLMCredentials(
                    provider: provider,
                    configuration: configuration
                ),
                allowsWebSocket: false
            )
        }
        if let message = RemoteEndpointSecurityPolicy.validationMessage(
            endpoint: configuration.endpoint,
            hasCredentials: securityContext.hasCredentials,
            allowsWebSocket: securityContext.allowsWebSocket
        ) {
            throw NSError(
                domain: "Voxt.Settings",
                code: -901,
                userInfo: [NSLocalizedDescriptionKey: message]
            )
        }

        switch testTarget {
        case .asr(let provider):
            return try await testASRProvider(provider, configuration: configuration)
        case .llm(let provider):
            return try await testLLMProvider(provider, configuration: configuration)
        }
    }
}
