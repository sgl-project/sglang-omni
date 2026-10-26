// LLM debug presets, prompt overrides and request ownership.

import Combine
import Foundation

@MainActor
final class LLMDebugViewModel: ObservableObject {
    @Published private(set) var modelOptions: [LLMDebugModelOption] = []
    @Published private(set) var presetOptions: [LLMDebugPresetOption] = []
    @Published var selectedModelID = ""
    @Published var selectedPresetID = ""
    @Published var variableValues: [String: String] = [:]
    @Published private(set) var results: [LLMDebugResult] = []
    @Published private(set) var isRunning = false
    @Published private(set) var isModelInitializing = false
    @Published private(set) var statusMessage = ""

    private let customLLMManager: CustomLLMModelManager
    private var remoteConfigurations: [String: RemoteProviderConfiguration]
    private var sessionPromptOverrides: [String: String] = [:]

    init(appDelegate: AppDelegate) {
        customLLMManager = appDelegate.customLLMManager
        remoteConfigurations = RemoteModelConfigurationStore.loadConfigurations(
            from: UserDefaults.standard.string(forKey: AppPreferenceKey.remoteLLMProviderConfigurations) ?? ""
        )
        refreshOptions()
        selectedModelID = preferredModelID()
        selectedPresetID = LLMDebugPresetStore.customPresetID
        resetVariableValuesForPreset()
    }

    var selectedPreset: LLMDebugPresetOption? {
        presetOptions.first(where: { $0.id == selectedPresetID })
    }

    var selectedModelTitle: String {
        modelOptions.first(where: { $0.id == selectedModelID })?.title ?? modelDebugLocalized("Select Model")
    }

    var selectedPresetTitle: String {
        selectedPreset?.title ?? modelDebugLocalized("Select Preset")
    }

    func refreshOptions() {
        remoteConfigurations = RemoteModelConfigurationStore.loadConfigurations(
            from: UserDefaults.standard.string(forKey: AppPreferenceKey.remoteLLMProviderConfigurations) ?? ""
        )
        modelOptions = ModelDebugCatalog.availableLLMModels(
            customLLMManager: customLLMManager,
            remoteLLMConfigurations: remoteConfigurations
        )
        presetOptions = ModelDebugCatalog.availableLLMPresets(
            promptOverrides: sessionPromptOverrides
        )
        if !modelOptions.contains(where: { $0.id == selectedModelID }) {
            selectedModelID = modelOptions.first?.id ?? ""
        }
        if !presetOptions.contains(where: { $0.id == selectedPresetID }) {
            selectedPresetID = LLMDebugPresetStore.customPresetID
        }
    }

    func presetDidChange() {
        resetVariableValuesForPreset()
    }

    func clearResults() {
        results.removeAll()
    }

    func removeResult(_ resultID: UUID) {
        results.removeAll { $0.id == resultID }
    }

    func savePromptTemplate(_ prompt: String) {
        guard let preset = selectedPreset else { return }
        let currentPresetID = selectedPresetID
        let preservedVariableValues = variableValues
        switch preset.kind {
        case .custom:
            LLMDebugPresetStore.saveCustomPrompt(prompt)
        case .enhancement, .translation, .rewrite, .transcriptSummary, .appGroup:
            sessionPromptOverrides[preset.id] = prompt
        }
        refreshOptions()
        selectedPresetID = currentPresetID
        variableValues = preservedVariableValues
    }

    func applyPromptTemplate(_ prompt: String) {
        guard let preset = selectedPreset else { return }
        let defaults = UserDefaults.standard
        sessionPromptOverrides.removeValue(forKey: preset.id)
        switch preset.kind {
        case .custom:
            return
        case .enhancement:
            FeatureSettingsStore.saveTranscriptionPrompt(prompt, defaults: defaults)
        case .translation:
            FeatureSettingsStore.saveTranslationPrompt(prompt, defaults: defaults)
        case .rewrite:
            FeatureSettingsStore.saveRewritePrompt(prompt, defaults: defaults)
        case .transcriptSummary:
            AppPreferenceKey.setTranscriptSummaryPromptTemplate(
                AppPromptDefaults.canonicalStoredText(prompt, kind: .transcriptSummary),
                defaults: defaults
            )
        case .appGroup(let groupID):
            applyGroupPrompt(prompt, groupID: groupID, defaults: defaults)
        }
        refreshOptions()
    }

    func handleWindowClose() {
        sessionPromptOverrides.removeAll()
        clearResults()
        statusMessage = ""
        resetVariableValuesForPreset()
    }

    func run() {
        guard let preset = selectedPreset,
              let model = modelOptions.first(where: { $0.id == selectedModelID })
        else { return }

        let needsInitialization = requiresInitialization(for: model)
        isRunning = true
        isModelInitializing = needsInitialization
        statusMessage = AppLocalization.format("Running %@", model.title)

        Task {
            let startedAt = Date()
            let values = self.mergedVariableValues(for: preset)
            let promptResolution = ModelDebugPromptResolver.resolve(
                preset: preset,
                values: values
            )
            do {
                let output = try await run(
                    model: model,
                    preset: preset,
                    values: values,
                    promptResolution: promptResolution
                )
                let result = LLMDebugResult(
                    id: UUID(),
                    modelTitle: model.title,
                    presetTitle: preset.title,
                    inputSummary: promptResolution.inputSummary,
                    requestMetadata: promptResolution.requestMetadata,
                    durationText: String(format: "%.2fs", Date().timeIntervalSince(startedAt)),
                    createdAt: Date(),
                    outputText: output,
                    errorText: nil
                )
                await MainActor.run {
                    self.results.insert(result, at: 0)
                    self.isRunning = false
                    self.isModelInitializing = false
                    self.statusMessage = AppLocalization.format("Completed %@", model.title)
                }
            } catch {
                let result = LLMDebugResult(
                    id: UUID(),
                    modelTitle: model.title,
                    presetTitle: preset.title,
                    inputSummary: promptResolution.inputSummary,
                    requestMetadata: promptResolution.requestMetadata,
                    durationText: String(format: "%.2fs", Date().timeIntervalSince(startedAt)),
                    createdAt: Date(),
                    outputText: "",
                    errorText: error.localizedDescription
                )
                await MainActor.run {
                    self.results.insert(result, at: 0)
                    self.isRunning = false
                    self.isModelInitializing = false
                    self.statusMessage = error.localizedDescription
                }
            }
        }
    }

    private func preferredModelID() -> String {
        let defaults = UserDefaults.standard
        let enhancementMode = EnhancementMode(rawValue: defaults.string(forKey: AppPreferenceKey.enhancementMode) ?? "")
            ?? .off
        switch enhancementMode {
        case .customLLM:
            return "local-llm:\(CustomLLMModelCatalog.canonicalModelRepo(defaults.string(forKey: AppPreferenceKey.customLLMModelRepo) ?? CustomLLMModelManager.defaultModelRepo))"
        case .remoteLLM:
            let provider = RemoteLLMProvider(rawValue: defaults.string(forKey: AppPreferenceKey.remoteLLMSelectedProvider) ?? "")
                ?? .openAI
            return "remote-llm:\(provider.rawValue)"
        case .off, .appleIntelligence:
            return modelOptions.first?.id ?? ""
        }
    }

    private func resetVariableValuesForPreset() {
        guard let preset = selectedPreset else {
            variableValues = [:]
            return
        }
        variableValues = preset.defaultValues
    }

    private func mergedVariableValues(for preset: LLMDebugPresetOption) -> [String: String] {
        preset.defaultValues.merging(variableValues) { _, rhs in rhs }
    }

    private func applyGroupPrompt(_ prompt: String, groupID: UUID, defaults: UserDefaults) {
        guard let data = defaults.data(forKey: AppPreferenceKey.appBranchGroups),
              var groups = try? JSONDecoder().decode([AppBranchGroup].self, from: data),
              let index = groups.firstIndex(where: { $0.id == groupID })
        else { return }
        groups[index].prompt = prompt
        guard let encoded = try? JSONEncoder().encode(groups) else { return }
        defaults.set(encoded, forKey: AppPreferenceKey.appBranchGroups)
    }

    private func requiresInitialization(for model: LLMDebugModelOption) -> Bool {
        switch model.selection {
        case .local(let repo):
            return !customLLMManager.isModelLoaded(repo: repo)
        case .remote:
            return false
        }
    }

    private func run(
        model: LLMDebugModelOption,
        preset: LLMDebugPresetOption,
        values: [String: String],
        promptResolution: LLMDebugResolvedPrompt
    ) async throws -> String {
        switch preset.kind {
        case .custom:
            switch model.selection {
            case .local(let repo):
                return try await customLLMManager.enhance(userPrompt: promptResolution.content, repo: repo)
            case .remote(let provider, let configuration):
                return try await RemoteLLMRuntimeClient().enhance(
                    userPrompt: promptResolution.content,
                    provider: provider,
                    configuration: configuration
                )
            }
        case .enhancement, .appGroup, .translation, .rewrite:
            guard let compiledRequest = promptResolution.compiledRequest else {
                throw NSError(
                    domain: "Voxt.ModelDebug",
                    code: -100,
                    userInfo: [NSLocalizedDescriptionKey: "Debug preset request compilation failed."]
                )
            }
            switch model.selection {
            case .local(let repo):
                return try await customLLMManager.executeCompiledRequest(
                    compiledRequest,
                    repo: repo
                )
            case .remote(let provider, let configuration):
                return try await RemoteLLMRuntimeClient().executeCompiledRequest(
                    compiledRequest,
                    provider: provider,
                    configuration: configuration
                )
            }
        case .transcriptSummary:
            let prompt = promptResolution.content
            switch model.selection {
            case .local(let repo):
                return try await customLLMManager.enhance(userPrompt: prompt, repo: repo)
            case .remote(let provider, let configuration):
                return try await RemoteLLMRuntimeClient().enhance(
                    userPrompt: prompt,
                    provider: provider,
                    configuration: configuration
                )
            }
        }
    }
}
