import Foundation

extension RemoteModelConfigurationStore {
    static func resolvedASRConfiguration(
        provider: RemoteASRProvider,
        stored: [String: RemoteProviderConfiguration]
    ) -> RemoteProviderConfiguration {
        let allowedModelIDs = Set(provider.modelOptions.map(\.id))
        if let existing = stored[provider.rawValue] {
            var normalized = existing
            if !allowedModelIDs.contains(normalized.model),
               !allowsCustomASRModel(provider: provider, model: normalized.model) {
                normalized.model = provider.suggestedModel
            }
            if provider != .openAIWhisper {
                normalized.openAIChunkPseudoRealtimeEnabled = false
            }
            return normalized
        }
        return RemoteProviderConfiguration(
            providerID: provider.rawValue,
            model: provider.suggestedModel,
            endpoint: "",
            apiKey: ""
        )
    }

    static func resolvedASRConfiguration(
        provider: RemoteASRProvider,
        from raw: String
    ) -> RemoteProviderConfiguration {
        let stored = loadConfiguration(providerID: provider.rawValue, from: raw)
            .map { [provider.rawValue: $0] } ?? [:]
        return resolvedASRConfiguration(provider: provider, stored: stored)
    }

    private static func allowsCustomASRModel(provider: RemoteASRProvider, model: String) -> Bool {
        guard provider == .openAIWhisper else { return false }
        return !model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    static func resolvedLLMConfiguration(
        provider: RemoteLLMProvider,
        stored: [String: RemoteProviderConfiguration]
    ) -> RemoteProviderConfiguration {
        if let existing = stored[provider.rawValue] {
            var normalized = normalizedCompatibilityValues(for: existing)
            if !provider.supportsHostedSearch {
                normalized.searchEnabled = false
            }
            return normalized
        }
        return RemoteProviderConfiguration(
            providerID: provider.rawValue,
            model: provider.suggestedModel,
            endpoint: "",
            apiKey: "",
            searchEnabled: provider.defaultSearchEnabled
        )
    }

    static func resolvedLLMConfiguration(
        provider: RemoteLLMProvider,
        from raw: String
    ) -> RemoteProviderConfiguration {
        let stored = loadConfiguration(providerID: provider.rawValue, from: raw)
            .map { [provider.rawValue: $0] } ?? [:]
        return resolvedLLMConfiguration(provider: provider, stored: stored)
    }

    static func isStoredLLMConfigurationConfigured(
        provider: RemoteLLMProvider,
        stored: [String: RemoteProviderConfiguration]
    ) -> Bool {
        guard stored[provider.rawValue] != nil else { return false }
        let configuration = resolvedLLMConfiguration(provider: provider, stored: stored)
        return configuration.isConfigured && configuration.hasUsableModel
    }


    nonisolated static func normalizedCompatibilityValues(
        for configuration: RemoteProviderConfiguration
    ) -> RemoteProviderConfiguration {
        guard let provider = RemoteLLMProvider(rawValue: configuration.providerID) else {
            return configuration
        }

        var normalized = configuration
        if !provider.supportsHostedSearch {
            normalized.searchEnabled = false
        }

        let trimmedEndpoint = configuration.endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        guard provider.usesResponsesAPI, !trimmedEndpoint.isEmpty else {
            return normalized
        }

        let runtimeClient = RemoteLLMRuntimeClient()
        normalized.endpoint = runtimeClient.resolvedLLMEndpoint(
            provider: provider,
            endpoint: trimmedEndpoint,
            model: configuration.model
        )
        return normalized
    }
}
