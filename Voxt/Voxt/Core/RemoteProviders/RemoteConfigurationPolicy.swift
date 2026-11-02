// RemoteConfigurationPolicy.swift
// Provides Remote Configuration Policy for remote provider configuration.

import Foundation

struct RemoteEndpointPreset: Identifiable, Hashable {
    let id: String
    let title: String
    let url: String
}

enum RemoteASRRealtimeSupport {
    static func isAliyunRealtimeModel(_ model: String) -> Bool {
        let normalized = model.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalized.isEmpty else { return false }
        return normalized.hasPrefix("qwen-audio-3.0-asr-flash-streaming")
            || normalized.hasPrefix("qwen3-asr-flash-realtime")
            || normalized.hasPrefix("fun-asr")
            || normalized.hasPrefix("paraformer-realtime")
    }

    static func isStepFunRealtimeModel(_ model: String) -> Bool {
        StepFunASRModelCapabilities.forModel(model).usesRealtimeWebSocket
    }
}

enum RemoteProviderConfigurationPolicy {
    static let customModelOptionID = "__voxt_custom_model__"

    static func llmProvider(for target: RemoteProviderTestTarget) -> RemoteLLMProvider? {
        if case .llm(let provider) = target {
            return provider
        }
        return nil
    }

    static func isDoubaoASRTest(_ target: RemoteProviderTestTarget) -> Bool {
        if case .asr(let provider) = target {
            return provider == .doubaoASR
        }
        return false
    }

    static func isOpenAIASRTest(_ target: RemoteProviderTestTarget) -> Bool {
        if case .asr(let provider) = target {
            return provider == .openAIWhisper
        }
        return false
    }

    static func testTargetLogName(_ target: RemoteProviderTestTarget) -> String {
        switch target {
        case .asr:
            return "asr"
        case .llm:
            return "llm"
        }
    }

    static func providerModelOptions(target: RemoteProviderTestTarget, configuredModel: String) -> [RemoteModelOption] {
        switch target {
        case .asr(let provider):
            return provider.modelOptions
        case .llm(let provider):
            return provider.modelOptions
        }
    }

    static func supportsCustomModelSelection(target: RemoteProviderTestTarget) -> Bool {
        if llmProvider(for: target) != nil {
            return true
        }
        if case .asr(let provider) = target {
            return provider == .openAIWhisper || provider == .stepFunASR
        }
        return false
    }

    static func pickerModelOptionIDs(target: RemoteProviderTestTarget, configuredModel: String) -> [String] {
        if let llmProvider = llmProvider(for: target) {
            var ids = (llmProvider.latestModelOptions + llmProvider.basicModelOptions + llmProvider.advancedModelOptions).map(\.id)
            if supportsCustomModelSelection(target: target) {
                ids.append(customModelOptionID)
            }
            return ids
        }
        var ids = providerModelOptions(target: target, configuredModel: configuredModel).map(\.id)
        if supportsCustomModelSelection(target: target) {
            ids.append(customModelOptionID)
        }
        return ids
    }

    static func resolvedSelection(
        target: RemoteProviderTestTarget,
        selectedProviderModel: String,
        configuredModel: String
    ) -> String {
        let ids = pickerModelOptionIDs(target: target, configuredModel: configuredModel)
        let trimmedSelected = selectedProviderModel.trimmingCharacters(in: .whitespacesAndNewlines)
        if ids.contains(trimmedSelected) {
            return trimmedSelected
        }
        let trimmedConfigured = configuredModel.trimmingCharacters(in: .whitespacesAndNewlines)
        if ids.contains(trimmedConfigured) {
            return trimmedConfigured
        }
        if supportsCustomModelSelection(target: target) {
            return customModelOptionID
        }
        return ids.first ?? trimmedSelected
    }

    static func initialSelection(target: RemoteProviderTestTarget, configuredModel: String) -> String {
        let ids = pickerModelOptionIDs(target: target, configuredModel: configuredModel)
        if supportsCustomModelSelection(target: target) {
            let trimmedConfigured = configuredModel.trimmingCharacters(in: .whitespacesAndNewlines)
            return ids.contains(trimmedConfigured) ? trimmedConfigured : customModelOptionID
        }
        return ids.contains(configuredModel) ? configuredModel : (ids.first ?? configuredModel)
    }

    static func resolvedModelValue(
        target: RemoteProviderTestTarget,
        resolvedSelection: String,
        customModelID: String
    ) -> String {
        if resolvedSelection == customModelOptionID,
           let fallbackModel = suggestedModelForCustomSelection(target: target) {
            let trimmedCustom = customModelID.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmedCustom.isEmpty ? fallbackModel : trimmedCustom
        }
        return resolvedSelection
    }

    static func nextCustomModelID(
        previousResolvedModel: String,
        newSelection: String,
        currentCustomModelID: String,
        supportsCustomSelection: Bool
    ) -> String {
        let trimmedCurrent = currentCustomModelID.trimmingCharacters(in: .whitespacesAndNewlines)
        if newSelection == customModelOptionID {
            return trimmedCurrent.isEmpty ? previousResolvedModel.trimmingCharacters(in: .whitespacesAndNewlines) : currentCustomModelID
        }
        if supportsCustomSelection && trimmedCurrent.isEmpty {
            return newSelection
        }
        return currentCustomModelID
    }

    static func endpointPresets(target: RemoteProviderTestTarget, resolvedModel: String) -> [RemoteEndpointPreset] {
        switch target {
        case .asr(let provider):
            guard provider == .aliyunBailianASR else { return [] }
            if isAliyunQwenRealtimeModel(resolvedModel) {
                return [
                    RemoteEndpointPreset(id: "aliyun-asr-qwen-cn-beijing", title: AppLocalization.localizedString("Qwen Realtime WS · Beijing"), url: "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"),
                    RemoteEndpointPreset(id: "aliyun-asr-qwen-ap-southeast-1", title: AppLocalization.localizedString("Qwen Realtime WS · Singapore"), url: "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"),
                    RemoteEndpointPreset(id: "aliyun-asr-qwen-us-east-1", title: AppLocalization.localizedString("Qwen Realtime WS · US (Virginia)"), url: "wss://dashscope-us.aliyuncs.com/api-ws/v1/realtime")
                ]
            }
            return [
                RemoteEndpointPreset(id: "aliyun-asr-fun-cn-beijing", title: AppLocalization.localizedString("Realtime WS · Beijing"), url: "wss://dashscope.aliyuncs.com/api-ws/v1/inference"),
                RemoteEndpointPreset(id: "aliyun-asr-fun-ap-southeast-1", title: AppLocalization.localizedString("Realtime WS · Singapore"), url: "wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference"),
                RemoteEndpointPreset(id: "aliyun-asr-fun-us-east-1", title: AppLocalization.localizedString("Realtime WS · US (Virginia)"), url: "wss://dashscope-us.aliyuncs.com/api-ws/v1/inference")
            ]
        case .llm(let provider):
            switch provider {
            case .aliyunBailian:
                return [
                    RemoteEndpointPreset(id: "aliyun-llm-cn-beijing", title: AppLocalization.localizedString("Beijing"), url: "https://dashscope.aliyuncs.com/compatible-mode/v1/responses"),
                    RemoteEndpointPreset(id: "aliyun-llm-ap-southeast-1", title: AppLocalization.localizedString("Singapore"), url: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/responses"),
                    RemoteEndpointPreset(id: "aliyun-llm-us-east-1", title: AppLocalization.localizedString("US (Virginia)"), url: "https://dashscope-us.aliyuncs.com/compatible-mode/v1/responses")
                ]
            case .xiaomiMiMo:
                return [
                    RemoteEndpointPreset(id: "xiaomi-mimo-chat-completions", title: AppLocalization.localizedString("Default"), url: "https://api.xiaomimimo.com/v1/chat/completions")
                ]
            case .volcengine:
                return [
                    RemoteEndpointPreset(id: "volcengine-llm-cn-beijing", title: AppLocalization.localizedString("Beijing"), url: "https://ark.cn-beijing.volces.com/api/v3/responses")
                ]
            case .stepFun:
                if resolvedModel.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "step-router-v1" {
                    return [
                        RemoteEndpointPreset(id: "stepfun-llm-step-plan", title: "Step Plan", url: "https://api.stepfun.com/step_plan/v1/chat/completions")
                    ]
                }
                return [
                    RemoteEndpointPreset(id: "stepfun-llm-chat-completions", title: AppLocalization.localizedString("Default"), url: "https://api.stepfun.com/v1/chat/completions")
                ]
            case .codex:
                return [
                    RemoteEndpointPreset(id: "codex-chatgpt", title: AppLocalization.localizedString("ChatGPT Codex"), url: "https://chatgpt.com/backend-api/codex/responses")
                ]
            default:
                return []
            }
        }
    }

    static func endpointPlaceholder(target: RemoteProviderTestTarget, resolvedModel: String) -> String {
        switch target {
        case .asr(let provider):
            switch provider {
            case .openAIWhisper:
                return "https://api.openai.com/v1/audio/transcriptions"
            case .glmASR:
                return "https://open.bigmodel.cn/api/paas/v4/audio/transcriptions"
            case .aliyunBailianASR:
                return endpointPresets(target: target, resolvedModel: resolvedModel).first?.url ?? "https://..."
            case .doubaoASR:
                return "https://..."
            case .stepFunASR:
                return "https://api.stepfun.com/v1/audio/asr/sse"
            case .xiaomiMiMoASR:
                return "https://api.xiaomimimo.com/v1/chat/completions"
            case .googleGeminiASR:
                return RemoteASREndpointSupport.geminiLiveDefaultEndpoint
            }
        case .llm(let provider):
            return RemoteLLMRuntimeClient().resolvedLLMEndpoint(
                provider: provider,
                endpoint: "",
                model: resolvedModel
            )
        }
    }

    static func remappedEndpointOnModelChange(
        target: RemoteProviderTestTarget,
        previousModel: String,
        newModel: String,
        currentEndpoint: String
    ) -> String {
        let trimmedEndpoint = currentEndpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        let oldPresets = endpointPresets(target: target, resolvedModel: previousModel)
        let newPresets = endpointPresets(target: target, resolvedModel: newModel)

        guard !newPresets.isEmpty else { return currentEndpoint }
        if trimmedEndpoint.isEmpty {
            return newPresets.first?.url ?? currentEndpoint
        }

        let oldPresetURLs = Set(oldPresets.map(\.url))
        let newPresetURLs = Set(newPresets.map(\.url))
        if newPresetURLs.contains(trimmedEndpoint) {
            return trimmedEndpoint
        }
        let allowsHostPresetRemap: Bool
        switch target {
        case .llm(.stepFun):
            allowsHostPresetRemap = false
        default:
            allowsHostPresetRemap = true
        }

        guard oldPresetURLs.contains(trimmedEndpoint) || (allowsHostPresetRemap && hostMatchesAnyPreset(trimmedEndpoint, presets: oldPresets)) else {
            return currentEndpoint
        }

        if let remapped = newPresets.first(where: { preset in
            presetHost(preset.url) == presetHost(trimmedEndpoint)
        }) {
            return remapped.url
        }
        return newPresets.first?.url ?? currentEndpoint
    }

    private static func isAliyunQwenRealtimeModel(_ model: String) -> Bool {
        model.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .hasPrefix("qwen3-asr-flash-realtime")
    }

    private static func suggestedModelForCustomSelection(target: RemoteProviderTestTarget) -> String? {
        if let llmProvider = llmProvider(for: target) {
            return llmProvider.suggestedModel
        }
        if case .asr(let provider) = target, provider == .openAIWhisper {
            return provider.suggestedModel
        }
        return nil
    }

    private static func hostMatchesAnyPreset(_ endpoint: String, presets: [RemoteEndpointPreset]) -> Bool {
        let host = presetHost(endpoint)
        return presets.contains { preset in
            presetHost(preset.url) == host
        }
    }

    private static func presetHost(_ value: String) -> String? {
        URL(string: value)?.host?.lowercased()
    }
}
