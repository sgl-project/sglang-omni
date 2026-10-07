// RemoteProviderSheetState.swift
// Provides Remote Provider Sheet State for settings screens.

import SwiftUI
import UniformTypeIdentifiers

extension RemoteProviderConfigurationSheet {
    var isOllamaLLMProvider: Bool {
        llmProviderForPicker == .ollama
    }

    var isOMLXLLMProvider: Bool {
        llmProviderForPicker == .omlx
    }

    var isOpenAILLMProvider: Bool {
        llmProviderForPicker == .openAI
    }

    var isAliyunASRProvider: Bool {
        asrProviderForSheet == .aliyunBailianASR
    }

    var isStepFunASRProvider: Bool {
        asrProviderForSheet == .stepFunASR
    }

    var aliyunASRModelCapabilities: AliyunASRModelCapabilities {
        AliyunASRModelCapabilities.forModel(resolvedModelValue())
    }

    var stepFunASRModelCapabilities: StepFunASRModelCapabilities {
        StepFunASRModelCapabilities.forModel(resolvedModelValue())
    }

    var isCodexLLMProvider: Bool {
        llmProviderForPicker == .codex
    }

    var isStepFunLLMProvider: Bool {
        llmProviderForPicker == .stepFun
    }

    var supportsStepFunReasoningEffort: Bool {
        isStepFunLLMProvider &&
            resolvedModelValue().trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "step-3.5-flash-2603"
    }

    var usesOpenAIResponsesOptions: Bool {
        isOpenAILLMProvider
    }

    var apiKeyFieldTitle: String {
        if isCodexLLMProvider {
            return AppLocalization.localizedString("Codex Credentials")
        }
        return (llmProviderForPicker?.apiKeyIsOptional == true)
            ? AppLocalization.localizedString("API Key (Optional)")
            : AppLocalization.localizedString("API Key")
    }

    var apiKeyFieldPlaceholder: String {
        if isCodexLLMProvider {
            return AppLocalization.localizedString("Uses ~/.codex/auth.json")
        }
        return (llmProviderForPicker?.apiKeyIsOptional == true)
            ? AppLocalization.localizedString("Paste API key (optional)")
            : AppLocalization.localizedString("Paste API key")
    }

    var apiKeyInput: Binding<String> {
        credentialBinding(for: .apiKey)
    }

    var appIDInput: Binding<String> {
        credentialBinding(for: .appID)
    }

    var accessTokenInput: Binding<String> {
        credentialBinding(for: .accessToken)
    }

    func credentialBinding(
        for field: RemoteProviderConfiguration.CredentialField
    ) -> Binding<String> {
        Binding(
            get: {
                switch field {
                case .apiKey:
                    return apiKey
                case .appID:
                    return appID
                case .accessToken:
                    return accessToken
                }
            },
            set: { value in
                switch field {
                case .apiKey:
                    apiKey = value
                case .appID:
                    appID = value
                case .accessToken:
                    accessToken = value
                }
                editedCredentialFields.insert(field)
            }
        )
    }

    func shouldOfferStoredCredentialClear(
        for field: RemoteProviderConfiguration.CredentialField
    ) -> Bool {
        !editedCredentialFields.contains(field) && configuration.hasStoredCredential(for: field)
    }

    func clearStoredCredential(_ field: RemoteProviderConfiguration.CredentialField) {
        switch field {
        case .apiKey:
            apiKey = ""
        case .appID:
            appID = ""
        case .accessToken:
            accessToken = ""
        }
        editedCredentialFields.insert(field)
    }

    var providerModelMenuOptions: [SettingsMenuOption<String>] {
        var options = providerModelOptions.map { SettingsMenuOption(value: $0.id, title: $0.title) }
        if supportsCustomProviderModelSelection {
            options.append(SettingsMenuOption(value: customModelOptionID, title: AppLocalization.localizedString("Custom...")))
        }
        return options
    }

    var providerModelSelectedTitle: String {
        providerModelMenuOptions.first(where: { $0.value == resolvedSelectionForPicker })?.title
            ?? AppLocalization.localizedString("Custom...")
    }

    var supportsCustomProviderModelSelection: Bool {
        RemoteProviderConfigurationPolicy.supportsCustomModelSelection(target: testTarget)
    }

    var shouldShowCustomProviderModelField: Bool {
        supportsCustomProviderModelSelection && resolvedSelectionForPicker == customModelOptionID
    }

    var customProviderModelPlaceholder: String {
        if isOpenAIASRTest {
            return AppLocalization.localizedString("e.g. gpt-4o-transcribe-xxx")
        }
        return AppLocalization.localizedString("e.g. doubao-seed-2-0-pro-260215")
    }

    var openAITextVerbosityMenuOptions: [SettingsMenuOption<String>] {
        guard OpenAITextVerbosity.supportsModel(resolvedModelValue()) else {
            return [SettingsMenuOption(value: OpenAITextVerbosity.automatic.rawValue, title: OpenAITextVerbosity.automatic.title)]
        }
        return OpenAITextVerbosity.allCases.map { option in
            SettingsMenuOption(value: option.rawValue, title: option.title)
        }
    }

    var openAITextVerbositySelectedTitle: String {
        OpenAITextVerbosity(rawValue: openAITextVerbosity)?.title
            ?? OpenAITextVerbosity.automatic.title
    }

    var generationCapabilities: LLMProviderCapabilities? {
        llmProviderForPicker.map { LLMProviderCapabilityRegistry.capabilities(for: $0) }
    }

    var shouldShowGenerationThinking: Bool {
        if isStepFunLLMProvider {
            return generationThinkingModeMenuOptions.count > 1
        }
        guard let capabilities = generationCapabilities else { return false }
        return capabilities.supportsThinkingToggle ||
            (capabilities.supportsThinkingEffort && (!isStepFunLLMProvider || supportsStepFunReasoningEffort)) ||
            capabilities.supportsThinkingBudget
    }

    var shouldShowGenerationAdvancedControls: Bool {
        guard let capabilities = generationCapabilities else { return false }
        return capabilities.supportsTopK ||
            capabilities.supportsMinP ||
            capabilities.supportsSeed ||
            capabilities.supportsPenalties ||
            capabilities.supportsLogprobs ||
            capabilities.supportsResponseFormat
    }

    var shouldShowGenerationExpertControls: Bool {
        guard let capabilities = generationCapabilities else { return false }
        return capabilities.supportsExtraBody || capabilities.supportsExtraOptions
    }

    var generationThinkingModeMenuOptions: [SettingsMenuOption<String>] {
        if isStepFunLLMProvider {
            var options = [SettingsMenuOption(value: LLMThinkingMode.off.rawValue, title: AppLocalization.localizedString("Off"))]
            if supportsStepFunReasoningEffort {
                options.append(SettingsMenuOption(value: LLMThinkingMode.effort.rawValue, title: AppLocalization.localizedString("Effort")))
            }
            return options
        }
        guard let capabilities = generationCapabilities else { return [] }
        var options = [SettingsMenuOption(value: LLMThinkingMode.providerDefault.rawValue, title: AppLocalization.localizedString("Default"))]
        if capabilities.supportsThinkingToggle {
            options.append(SettingsMenuOption(value: LLMThinkingMode.off.rawValue, title: AppLocalization.localizedString("Off")))
            options.append(SettingsMenuOption(value: LLMThinkingMode.on.rawValue, title: AppLocalization.localizedString("On")))
        }
        if capabilities.supportsThinkingEffort && (!isStepFunLLMProvider || supportsStepFunReasoningEffort) {
            options.append(SettingsMenuOption(value: LLMThinkingMode.effort.rawValue, title: AppLocalization.localizedString("Effort")))
        }
        if capabilities.supportsThinkingBudget {
            options.append(SettingsMenuOption(value: LLMThinkingMode.budget.rawValue, title: AppLocalization.localizedString("Budget")))
        }
        return options
    }

    var generationThinkingModeSelectedTitle: String {
        generationThinkingModeMenuOptions.first(where: { $0.value == generationThinkingMode })?.title
            ?? (isStepFunLLMProvider ? AppLocalization.localizedString("Off") : AppLocalization.localizedString("Default"))
    }

    var sanitizedGenerationThinkingMode: LLMThinkingMode {
        let mode = LLMThinkingMode(rawValue: generationThinkingMode) ?? .providerDefault
        let supportedValues = Set(generationThinkingModeMenuOptions.map(\.value))
        if isStepFunLLMProvider, !supportedValues.contains(mode.rawValue) {
            return .off
        }
        return supportedValues.contains(mode.rawValue) ? mode : .providerDefault
    }

    var generationThinkingEffortMenuOptions: [SettingsMenuOption<String>] {
        let values: [String]
        if usesOpenAIResponsesOptions {
            values = OpenAIReasoningEffort.supportedCases(forModel: resolvedModelValue())
                .filter { $0 != .automatic }
                .map(\.rawValue)
        } else if llmProviderForPicker == .deepseek {
            values = ["none", "low", "high", "max"]
        } else if isStepFunLLMProvider {
            values = supportsStepFunReasoningEffort ? ["low", "high"] : []
        } else if isOllamaLLMProvider {
            values = [
                OllamaThinkMode.low.rawValue,
                OllamaThinkMode.medium.rawValue,
                OllamaThinkMode.high.rawValue
            ]
        } else {
            values = ["none", "minimal", "low", "medium", "high", "xhigh"]
        }
        return values.map { SettingsMenuOption(value: $0, title: generationEffortTitle($0)) }
    }

    var generationThinkingEffortSelectedTitle: String {
        generationThinkingEffortMenuOptions.first(where: { $0.value == generationThinkingEffort })?.title
            ?? AppLocalization.localizedString("Default")
    }

    var generationResponseFormatMenuOptions: [SettingsMenuOption<String>] {
        guard generationCapabilities?.supportsResponseFormat == true else { return [] }
        var formats: [LLMResponseFormat] = [.plain, .json]
        if isOllamaLLMProvider || isOMLXLLMProvider {
            formats.append(.jsonSchema)
        }
        return formats.map { SettingsMenuOption(value: $0.rawValue, title: $0.title) }
    }

    var generationResponseFormatSelectedTitle: String {
        generationResponseFormatMenuOptions.first(where: { $0.value == generationResponseFormat })?.title
            ?? LLMResponseFormat.plain.title
    }

    var shouldShowGenerationJSONSchemaField: Bool {
        LLMResponseFormat(rawValue: generationResponseFormat) == .jsonSchema &&
            (isOllamaLLMProvider || isOMLXLLMProvider)
    }

    var isDoubaoASRTest: Bool {
        RemoteProviderConfigurationPolicy.isDoubaoASRTest(testTarget)
    }

    var isOpenAIASRTest: Bool {
        RemoteProviderConfigurationPolicy.isOpenAIASRTest(testTarget)
    }

    var customModelOptionID: String {
        RemoteProviderConfigurationPolicy.customModelOptionID
    }

    var asrProviderForSheet: RemoteASRProvider? {
        if case .asr(let provider) = testTarget {
            return provider
        }
        return nil
    }

    var activeProviderNotice: String? {
        switch testTarget {
        case .asr(let provider):
            let active = RemoteASRProvider(rawValue: selectedRemoteASRProviderRaw) ?? .openAIWhisper
            guard active != provider else { return nil }
            return AppLocalization.format(
                "Current active Remote ASR provider is %@. Testing %@ here does not switch the active provider.",
                active.title,
                provider.title
            )
        case .llm(let provider):
            let active = RemoteLLMProvider(rawValue: selectedRemoteLLMProviderRaw) ?? .openAI
            guard active != provider else { return nil }
            return AppLocalization.format(
                "Current active Remote LLM provider is %@. Testing %@ here does not switch the active provider.",
                active.title,
                provider.title
            )
        }
    }

    var providerModelOptions: [RemoteModelOption] {
        if isCodexLLMProvider,
           let dynamicCodexModelOptions = operations.modelOptions,
           !dynamicCodexModelOptions.isEmpty {
            return dynamicCodexModelOptions
        }
        return RemoteProviderConfigurationPolicy.providerModelOptions(
            target: testTarget,
            configuredModel: configuration.model
        )
    }

    var resolvedSelectionForPicker: String {
        let ids = pickerModelOptionIDs
        let trimmedSelected = selectedProviderModel.trimmingCharacters(in: .whitespacesAndNewlines)
        if ids.contains(trimmedSelected) {
            return trimmedSelected
        }
        let trimmedConfigured = configuration.model.trimmingCharacters(in: .whitespacesAndNewlines)
        if ids.contains(trimmedConfigured) {
            return trimmedConfigured
        }
        if supportsCustomProviderModelSelection {
            return customModelOptionID
        }
        return ids.first ?? trimmedSelected
    }

    var pickerModelOptionIDs: [String] {
        var ids = providerModelOptions.map(\.id)
        if supportsCustomProviderModelSelection {
            ids.append(customModelOptionID)
        }
        return ids
    }

    var providerModelSelectionBinding: Binding<String> {
        Binding(
            get: { resolvedSelectionForPicker },
            set: {
                handleProviderModelSelectionChange($0)
            }
        )
    }

    var llmProviderForPicker: RemoteLLMProvider? {
        RemoteProviderConfigurationPolicy.llmProvider(for: testTarget)
    }

    var showsSearchSection: Bool {
        llmProviderForPicker?.supportsHostedSearch == true
    }

    func configureModelSelection() {
        let ids = pickerModelOptionIDs
        if supportsCustomProviderModelSelection {
            let trimmedConfigured = configuration.model.trimmingCharacters(in: .whitespacesAndNewlines)
            selectedProviderModel = ids.contains(trimmedConfigured) ? trimmedConfigured : customModelOptionID
            return
        }
        selectedProviderModel = ids.contains(configuration.model) ? configuration.model : (ids.first ?? configuration.model)
    }

    func handleProviderModelSelectionChange(_ newValue: String) {
        let previousModel = resolvedModelValue()
        selectedProviderModel = newValue
        customModelID = RemoteProviderConfigurationPolicy.nextCustomModelID(
            previousResolvedModel: previousModel,
            newSelection: newValue,
            currentCustomModelID: customModelID,
            supportsCustomSelection: supportsCustomProviderModelSelection
        )

        endpoint = RemoteProviderConfigurationPolicy.remappedEndpointOnModelChange(
            target: testTarget,
            previousModel: previousModel,
            newModel: resolvedModelValue(),
            currentEndpoint: endpoint
        )
    }

    func resolvedModelValue() -> String {
        RemoteProviderConfigurationPolicy.resolvedModelValue(
            target: testTarget,
            resolvedSelection: resolvedSelectionForPicker,
            customModelID: customModelID
        )
    }

    var endpointPresets: [RemoteEndpointPreset] {
        RemoteProviderConfigurationPolicy.endpointPresets(
            target: testTarget,
            resolvedModel: resolvedModelValue()
        )
    }

    var endpointPresetHintText: String? {
        guard !endpointPresets.isEmpty else { return nil }
        guard let provider = llmProviderForPicker else { return nil }

        switch provider {
        case .aliyunBailian:
            return AppLocalization.localizedString("Aliyun API keys are region-specific; use the matching endpoint.")
        case .volcengine:
            return AppLocalization.localizedString("Volcengine models should use the Responses endpoint in the same region as the API key.")
        case .codex:
            return nil
        default:
            return nil
        }
    }

    var endpointFieldPlaceholder: String {
        RemoteProviderConfigurationPolicy.endpointPlaceholder(
            target: testTarget,
            resolvedModel: resolvedModelValue()
        )
    }

    func initialEndpointValue() -> String {
        let trimmedEndpoint = configuration.endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmedEndpoint.isEmpty, isCodexLLMProvider else {
            return configuration.endpoint
        }
        return RemoteLLMRuntimeClient().resolvedLLMEndpoint(
            provider: .codex,
            endpoint: "",
            model: resolvedModelValue()
        )
    }

    var codexAuthFileDisplayPath: String {
        let selectedPath = codexAuthFilePath.trimmingCharacters(in: .whitespacesAndNewlines)
        if !selectedPath.isEmpty {
            return selectedPath
        }
        return CodexOAuthCredentialProvider().authFilePath()
    }

    var hasCustomCodexAuthFilePath: Bool {
        !codexAuthFilePath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    func chooseCodexAuthFile() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = false
        panel.allowedContentTypes = [.json]
        panel.directoryURL = URL(fileURLWithPath: codexAuthFileDisplayPath)
            .deletingLastPathComponent()
        panel.message = AppLocalization.localizedString("Choose Codex auth.json")

        guard panel.runModal() == .OK, let selectedURL = panel.url else { return }

        do {
            codexAuthFileBookmark = try SecurityScopedBookmarkSupport.createBookmark(for: selectedURL)
            codexAuthFilePath = selectedURL.path
            codexAuthFileSelectionError = nil
            loadCodexModelOptionsIfNeeded()
        } catch {
            codexAuthFileSelectionError = AppLocalization.format(
                "Failed to update Codex auth path: %@",
                error.localizedDescription
            )
        }
    }

    func clearCodexAuthFileSelection() {
        codexAuthFilePath = ""
        codexAuthFileBookmark = nil
        codexAuthFileSelectionError = nil
        loadCodexModelOptionsIfNeeded()
    }

    func loadCodexModelOptionsIfNeeded() {
        guard isCodexLLMProvider else { return }
        var snapshot = currentConfigurationSnapshot
        snapshot.endpoint = endpoint.trimmingCharacters(in: .whitespacesAndNewlines)

        operations.loadModels {
            await RemoteLLMRuntimeClient().codexModelOptions(configuration: snapshot)
        }
    }

    func testConnection() {
        guard let snapshot = validatedCurrentConfigurationSnapshot() else { return }
        runConnectionTest(for: testTarget, modelForLog: snapshot.model, snapshot: snapshot)
    }

    func saveConfiguration() {
        guard let snapshot = validatedCurrentConfigurationSnapshot() else { return }
        switch onSave(snapshot) {
        case .success:
            close()
        case .failure(let error):
            operations.showFailure(error.localizedDescription)
        }
    }

    func validatedCurrentConfigurationSnapshot() -> RemoteProviderConfiguration? {
        if let message = validationMessage() {
            operations.showFailure(message)
            return nil
        }
        return currentConfigurationSnapshot
    }

    func generationEffortTitle(_ value: String) -> String {
        if let openAIEffort = OpenAIReasoningEffort(rawValue: value) {
            return openAIEffort.title
        }
        switch value {
        case "none":
            return AppLocalization.localizedString("None")
        case "minimal":
            return AppLocalization.localizedString("Minimal")
        case "low":
            return AppLocalization.localizedString("Low")
        case "medium":
            return AppLocalization.localizedString("Medium")
        case "high":
            return AppLocalization.localizedString("High")
        case "xhigh":
            return AppLocalization.localizedString("Extra High")
        default:
            return value
        }
    }

    func runConnectionTest(
        for target: RemoteProviderTestTarget,
        modelForLog: String,
        snapshot: RemoteProviderConfiguration
    ) {
        let providerID = configuration.providerID
        VoxtLog.settings(
            "Remote provider test started. target=\(RemoteProviderConfigurationPolicy.testTargetLogName(target)), provider=\(configuration.providerID), model=\(modelForLog), endpoint=\(sanitizedEndpointForLog(snapshot.endpoint)), proxyMode=\(VoxtNetworkSession.modeDescription), hasAPIKey=\(!snapshot.apiKey.isEmpty), hasAppID=\(!snapshot.appID.isEmpty), hasAccessToken=\(!snapshot.accessToken.isEmpty)"
        )

        operations.testConnection {
            do {
                let tester = RemoteProviderConnectivityTester(testTarget: target)
                let message = try await tester.run(configuration: snapshot)
                try Task.checkCancellation()
                VoxtLog.settings(
                    "Remote provider test succeeded. target=\(RemoteProviderConfigurationPolicy.testTargetLogName(target)), provider=\(providerID), model=\(modelForLog), message=\(message)"
                )
                return message
            } catch {
                try Task.checkCancellation()
                let message = VoxtNetworkSession.directModeConflictMessage(for: error) ?? error.localizedDescription
                VoxtLog.settingsWarning(
                    "Remote provider test failed. target=\(RemoteProviderConfigurationPolicy.testTargetLogName(target)), provider=\(providerID), model=\(modelForLog), error=\(message)"
                )
                throw error
            }
        }
    }

    func sanitizedEndpointForLog(_ endpoint: String) -> String {
        RemoteEndpointSecurityPolicy.sanitizedForLog(endpoint)
    }

    func showOperationToast(_ message: String, duration: Duration = .seconds(4)) {
        operationToastDismissTask?.cancel()
        operationToastMessage = message
        operationToastDismissTask = Task { @MainActor in
            try? await Task.sleep(for: duration)
            guard !Task.isCancelled else { return }
            operationToastMessage = ""
        }
    }

    func dismissOperationToast() {
        operationToastDismissTask?.cancel()
        operationToastMessage = ""
    }
}
