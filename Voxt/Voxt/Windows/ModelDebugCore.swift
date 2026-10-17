// ModelDebugCore.swift
// Provides Model Debug Core for window and overlay UI.

import AppKit
import SwiftUI
import AVFoundation
import Combine


func modelDebugLocalized(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

func modelDebugClipTimestamp(_ date: Date) -> String {
    date.formatted(.dateTime.month(.defaultDigits).day(.defaultDigits).hour().minute().second())
}

func modelDebugClipTitlePreview(_ text: String) -> String? {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return nil }
    let singleLine = trimmed.replacingOccurrences(of: "\n", with: " ")
    return String(singleLine.prefix(28))
}

struct ASRDebugResult: Identifiable, Equatable {
    enum Source: String {
        case liveRecording
        case reusedClip
    }

    let id: UUID
    let clipID: UUID
    let clipTitle: String
    let modelTitle: String
    let source: Source
    let audioDurationText: String
    let runtimeText: String
    let characterCount: Int
    let createdAt: Date
    let outputText: String
    let errorText: String?

    var isError: Bool { errorText != nil }
}

struct ASRDebugClipItem: Identifiable, Equatable {
    let id: UUID
    let clip: DebugAudioClip
    let defaultTitle: String
    var title: String

    var displayTitle: String { title }
}

extension ASRDebugModelOption {
    var selectorEntry: FeatureModelSelectorEntry {
        let selectionID = FeatureModelSelectionID(rawValue: id)
        let locationTag: String
        switch selection {
        case .remote:
            locationTag = modelDebugLocalized("Remote")
        case .mlx:
            locationTag = modelDebugLocalized("Local")
        }
        return FeatureModelSelectorEntry(
            selectionID: selectionID,
            title: title,
            engine: subtitle,
            sizeText: modelDebugLocalized("Ready"),
            ratingText: "—",
            filterTags: [locationTag, modelDebugLocalized("Installed"), modelDebugLocalized("Configured")],
            displayTags: [locationTag, modelDebugLocalized("Installed")],
            statusText: "",
            usageLocations: [],
            badgeText: nil,
            isSelectable: true,
            disabledReason: nil
        )
    }
}

extension LLMDebugModelOption {
    var selectorEntry: FeatureModelSelectorEntry {
        let selectionID = FeatureModelSelectionID(rawValue: id)
        let locationTag: String
        switch selection {
        case .remote:
            locationTag = modelDebugLocalized("Remote")
        case .local:
            locationTag = modelDebugLocalized("Local")
        }
        return FeatureModelSelectorEntry(
            selectionID: selectionID,
            title: title,
            engine: subtitle,
            sizeText: modelDebugLocalized("Ready"),
            ratingText: "—",
            filterTags: [locationTag, modelDebugLocalized("Installed"), modelDebugLocalized("Configured")],
            displayTags: [locationTag, modelDebugLocalized("Installed")],
            statusText: "",
            usageLocations: [],
            badgeText: nil,
            isSelectable: true,
            disabledReason: nil
        )
    }
}

enum ModelDebugWindowStyle {
    static let width: CGFloat = 750
    static let height: CGFloat = 500
    static let minWidth: CGFloat = 650
    static let minHeight: CGFloat = 560
    static let selectorMinWidth: CGFloat = 150
    static let selectorIdealWidth: CGFloat = 220
    static let resultCardBodyHeight: CGFloat = 208
}

struct LLMDebugResult: Identifiable, Equatable {
    let id: UUID
    let modelTitle: String
    let presetTitle: String
    let inputSummary: String
    let requestMetadata: LLMDebugRequestMetadata?
    let durationText: String
    let createdAt: Date
    let outputText: String
    let errorText: String?

    var isError: Bool { errorText != nil }
}

extension ASRDebugResult.Source {
    var localizedTitle: String {
        switch self {
        case .liveRecording:
            return modelDebugLocalized("Recorded")
        case .reusedClip:
            return modelDebugLocalized("Reused Audio")
        }
    }
}

private final class DebugAudioRecorder: NSObject {
    private var recorder: AVAudioRecorder?
    private(set) var activeURL: URL?

    func start() throws {
        let url = DebugAudioClipIO.temporaryClipURL()
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: 16_000,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false
        ]

        let recorder = try AVAudioRecorder(url: url, settings: settings)
        recorder.isMeteringEnabled = false
        guard recorder.record() else {
            throw NSError(
                domain: "Voxt.ModelDebug",
                code: -10,
                userInfo: [NSLocalizedDescriptionKey: modelDebugLocalized("Failed to start recording.")]
            )
        }
        self.recorder = recorder
        activeURL = url
    }

    func stop() -> URL? {
        recorder?.stop()
        recorder = nil
        defer { activeURL = nil }
        return activeURL
    }

    func cancel() {
        recorder?.stop()
        recorder = nil
        if let activeURL {
            try? FileManager.default.removeItem(at: activeURL)
        }
        activeURL = nil
    }
}

@MainActor
final class ASRDebugViewModel: ObservableObject {
    @Published private(set) var options: [ASRDebugModelOption] = []
    @Published var selectedModelID = ""
    @Published private(set) var clips: [ASRDebugClipItem] = []
    @Published var selectedClipID: UUID?
    @Published private(set) var results: [ASRDebugResult] = []
    @Published private(set) var isRecording = false
    @Published private(set) var isRunning = false
    @Published private(set) var isModelInitializing = false
    @Published private(set) var statusMessage = ""
    @Published private(set) var toastMessage = ""

    private let mlxModelManager: MLXModelManager
    private var remoteConfigurations: [String: RemoteProviderConfiguration]
    private let recorder = DebugAudioRecorder()
    private let mlxTranscriber: MLXTranscriber
    private let remoteTranscriber = RemoteASRTranscriber()
    private var toastDismissTask: Task<Void, Never>?

    init(appDelegate: AppDelegate) {
        let useMirror = UserDefaults.standard.bool(forKey: AppPreferenceKey.useHfMirror)
        let hubURL = useMirror ? MLXModelManager.mirrorHubBaseURL : MLXModelManager.defaultHubBaseURL
        mlxModelManager = MLXModelManager(
            modelRepo: appDelegate.mlxModelManager.currentModelRepo,
            hubBaseURL: hubURL
        )
        mlxTranscriber = MLXTranscriber(modelManager: mlxModelManager)
        mlxTranscriber.dictionaryEntryProvider = {
            appDelegate.dictionaryStore.activeEntriesForRemoteRequest(
                activeGroupID: appDelegate.activeDictionaryGroupID(),
                limit: DictionaryEntryCollection.asrPromptTermLimit
            )
        }
        remoteTranscriber.doubaoDictionaryEntryProvider = {
            appDelegate.dictionaryStore.activeEntriesForRemoteRequest(
                activeGroupID: appDelegate.activeDictionaryGroupID(),
                limit: 5_000
            )
        }
        remoteConfigurations = RemoteModelConfigurationStore.loadConfigurations(
            from: UserDefaults.standard.string(forKey: AppPreferenceKey.remoteASRProviderConfigurations) ?? ""
        )
        refreshOptions()
        selectedModelID = preferredModelID()
    }

    func refreshOptions() {
        remoteConfigurations = RemoteModelConfigurationStore.loadConfigurations(
            from: UserDefaults.standard.string(forKey: AppPreferenceKey.remoteASRProviderConfigurations) ?? ""
        )
        options = ModelDebugCatalog.availableASRModels(
            mlxModelManager: mlxModelManager,
            remoteASRConfigurations: remoteConfigurations
        )
        if !options.contains(where: { $0.id == selectedModelID }) {
            selectedModelID = options.first?.id ?? ""
        }
    }

    var selectedModelTitle: String {
        options.first(where: { $0.id == selectedModelID })?.title ?? modelDebugLocalized("Select Model")
    }

    var selectedClipItem: ASRDebugClipItem? {
        guard let selectedClipID else { return nil }
        return clips.first(where: { $0.id == selectedClipID })
    }

    var selectedClipTitle: String {
        selectedClipItem?.displayTitle ?? modelDebugLocalized("Select Audio")
    }

    var vadSnapshot: VADDebugSnapshot {
        let option = options.first(where: { $0.id == selectedModelID })
        let engine: TranscriptionEngine
        let providerUsesServerVAD: Bool
        switch option?.selection {
        case .remote(let provider, let configuration):
            engine = .remote
            providerUsesServerVAD = RemoteASRRealtimeSupport.usesRealtimeMeetingProfile(
                provider: provider,
                configuration: configuration
            )
        case .mlx:
            engine = .mlxAudio
            providerUsesServerVAD = false
        case .none:
            engine = TranscriptionEngine(rawValue: UserDefaults.standard.string(forKey: AppPreferenceKey.transcriptionEngine) ?? "")
                ?? .mlxAudio
            providerUsesServerVAD = false
        }
        return ModelDebugCatalog.vadSnapshot(
            defaults: .standard,
            transcriptionEngine: engine,
            providerUsesServerVAD: providerUsesServerVAD
        )
    }

    func toggleRecording() {
        if isRecording {
            stopRecordingAndRun()
        } else {
            startRecording()
        }
    }

    func generateSelectedClip() {
        guard !options.isEmpty, options.contains(where: { $0.id == selectedModelID }) else {
            showToast(modelDebugLocalized("No available model."))
            return
        }
        guard let clipItem = selectedClipItem else {
            showToast(modelDebugLocalized("No audio selected."))
            return
        }
        Task {
            await runCurrentSelection(with: clipItem, source: .reusedClip)
        }
    }

    func clearResults() {
        results.removeAll()
    }

    func removeResult(_ resultID: UUID) {
        results.removeAll { $0.id == resultID }
    }

    func deleteClip(_ clipID: UUID) {
        guard let clip = clips.first(where: { $0.id == clipID }) else { return }
        try? FileManager.default.removeItem(at: clip.clip.fileURL)
        clips.removeAll { $0.id == clipID }
        results.removeAll { $0.clipID == clipID }
        if selectedClipID == clipID {
            selectedClipID = clips.first?.id
        }
        statusMessage = ""
    }

    func handleWindowClose() {
        toastDismissTask?.cancel()
        recorder.cancel()
        for clip in clips {
            try? FileManager.default.removeItem(at: clip.clip.fileURL)
        }
        clips.removeAll()
        selectedClipID = nil
        clearResults()
    }

    private func preferredModelID() -> String {
        let defaults = UserDefaults.standard
        let engine = TranscriptionEngine(rawValue: defaults.string(forKey: AppPreferenceKey.transcriptionEngine) ?? "")
            ?? .mlxAudio
        switch engine {
        case .mlxAudio:
            return "mlx:\(MLXModelManager.canonicalModelRepo(defaults.string(forKey: AppPreferenceKey.mlxModelRepo) ?? MLXModelManager.defaultModelRepo))"
        case .remote:
            let provider = RemoteASRProvider(rawValue: defaults.string(forKey: AppPreferenceKey.remoteASRSelectedProvider) ?? "")
                ?? .openAIWhisper
            return "remote-asr:\(provider.rawValue)"
        case .dictation:
            return options.first?.id ?? ""
        }
    }

    private func startRecording() {
        do {
            try recorder.start()
            isRecording = true
            statusMessage = modelDebugLocalized("Recording…")
        } catch {
            statusMessage = error.localizedDescription
        }
    }

    private func stopRecordingAndRun() {
        guard let url = recorder.stop() else {
            isRecording = false
            statusMessage = modelDebugLocalized("No recording was captured.")
            return
        }
        isRecording = false
        do {
            let clip = try DebugAudioClipIO.clip(for: url)
            let timestamp = modelDebugClipTimestamp(clip.createdAt)
            let item = ASRDebugClipItem(
                id: clip.id,
                clip: clip,
                defaultTitle: timestamp,
                title: timestamp
            )
            clips.insert(item, at: 0)
            selectedClipID = item.id
            statusMessage = AppLocalization.format("Recorded clip ready: %@", clip.summaryText)
            Task {
                await runCurrentSelection(with: item, source: .liveRecording)
            }
        } catch {
            try? FileManager.default.removeItem(at: url)
            statusMessage = error.localizedDescription
        }
    }

    private func runCurrentSelection(
        with clipItem: ASRDebugClipItem,
        source: ASRDebugResult.Source
    ) async {
        guard let option = options.first(where: { $0.id == selectedModelID }) else { return }
        let needsInitialization = requiresInitialization(for: option)
        isRunning = true
        isModelInitializing = needsInitialization
        defer {
            isRunning = false
            isModelInitializing = false
        }
        let startedAt = Date()
        statusMessage = AppLocalization.format("Running %@", option.title)

        do {
            let output = try await run(option: option, clip: clipItem.clip)
            let elapsed = Date().timeIntervalSince(startedAt)
            let result = ASRDebugResult(
                id: UUID(),
                clipID: clipItem.id,
                clipTitle: clipItem.displayTitle,
                modelTitle: option.title,
                source: source,
                audioDurationText: String(format: "%.1fs", clipItem.clip.durationSeconds),
                runtimeText: String(format: "%.2fs", elapsed),
                characterCount: output.count,
                createdAt: Date(),
                outputText: output,
                errorText: nil
            )
            results.insert(result, at: 0)
            updateClipTitleIfNeeded(clipID: clipItem.id, transcript: output)
            statusMessage = AppLocalization.format("Completed %@", option.title)
        } catch {
            let elapsed = Date().timeIntervalSince(startedAt)
            let result = ASRDebugResult(
                id: UUID(),
                clipID: clipItem.id,
                clipTitle: clipItem.displayTitle,
                modelTitle: option.title,
                source: source,
                audioDurationText: String(format: "%.1fs", clipItem.clip.durationSeconds),
                runtimeText: String(format: "%.2fs", elapsed),
                characterCount: 0,
                createdAt: Date(),
                outputText: "",
                errorText: error.localizedDescription
            )
            results.insert(result, at: 0)
            statusMessage = error.localizedDescription
        }
    }

    private func requiresInitialization(for option: ASRDebugModelOption) -> Bool {
        switch option.selection {
        case .mlx(let repo):
            let canonicalRepo = MLXModelManager.canonicalModelRepo(repo)
            return mlxModelManager.currentModelRepo != canonicalRepo || !mlxModelManager.isCurrentModelLoaded
        case .remote:
            return false
        }
    }

    private func run(option: ASRDebugModelOption, clip: DebugAudioClip) async throws -> String {
        switch option.selection {
        case .mlx(let repo):
            mlxModelManager.updateModel(repo: repo)
            let transcript = try await mlxTranscriber.transcribeAudioFile(clip.fileURL)
            if MLXModelFamily.family(for: repo) == .senseVoice,
               let metadata = mlxTranscriber.latestSenseVoiceMetadata {
                return metadata.formattedDebugSummary(appendingTranscript: transcript)
            }
            return transcript
        case .remote(let provider, let configuration):
            return try await remoteTranscriber.transcribeDebugAudioFile(
                clip.fileURL,
                provider: provider,
                configuration: configuration
            )
        }
    }

    private func updateClipTitleIfNeeded(clipID: UUID, transcript: String) {
        guard let index = clips.firstIndex(where: { $0.id == clipID }),
              let preview = modelDebugClipTitlePreview(transcript)
        else { return }
        guard clips[index].title == clips[index].defaultTitle else { return }
        clips[index].title = "\(clips[index].defaultTitle) · \(preview)"
    }

    func dismissToast() {
        toastDismissTask?.cancel()
        toastMessage = ""
    }

    private func showToast(_ message: String, duration: TimeInterval = 2.2) {
        toastDismissTask?.cancel()
        toastMessage = message
        toastDismissTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: UInt64(duration * 1_000_000_000))
            guard !Task.isCancelled else { return }
            self.toastMessage = ""
        }
    }
}

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

    var promptPreview: String {
        guard let selectedPreset else { return "" }
        return ModelDebugPromptResolver.resolve(
            preset: selectedPreset,
            values: variableValues
        ).content
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
