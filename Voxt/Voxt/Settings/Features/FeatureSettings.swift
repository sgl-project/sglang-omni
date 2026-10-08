// FeatureSettings.swift
// Provides Feature Settings for feature settings.

import Foundation

struct FeatureModelSelectionID: RawRepresentable, Codable, Hashable, Sendable, Identifiable {
    let rawValue: String

    var id: String { rawValue }

    init(rawValue: String) {
        self.rawValue = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static let dictation = Self(rawValue: "dictation")
    static let appleIntelligence = Self(rawValue: "apple-intelligence")
    private static let legacyWhisperDirectTranslateRawValue = "whisper-direct-translate"

    static func mlx(_ repo: String) -> Self {
        Self(rawValue: "mlx:\(MLXModelManager.canonicalModelRepo(repo))")
    }

    static func whisper(_ modelID: String) -> Self {
        .mlx(MLXWhisperMigrationSupport.repo(forLegacyWhisperModelID: modelID))
    }

    static func remoteASR(_ provider: RemoteASRProvider) -> Self {
        Self(rawValue: "remote-asr:\(provider.rawValue)")
    }

    static func localLLM(_ repo: String) -> Self {
        Self(rawValue: "local-llm:\(CustomLLMModelCatalog.canonicalModelRepo(repo))")
    }

    static func localGGUFTranslation(_ modelID: GGUFTranslationModelID) -> Self {
        Self(rawValue: "local-gguf-translation:\(modelID.rawValue)")
    }

    static func remoteLLM(_ provider: RemoteLLMProvider) -> Self {
        Self(rawValue: "remote-llm:\(provider.rawValue)")
    }

    enum ASRSelection: Hashable, Sendable {
        case dictation
        case mlx(repo: String)
        case remote(provider: RemoteASRProvider)
    }

    enum TextSelection: Hashable, Sendable {
        case appleIntelligence
        case localLLM(repo: String)
        case remoteLLM(provider: RemoteLLMProvider)
    }

    enum TranslationSelection: Hashable, Sendable {
        case localLLM(repo: String)
        case localGGUF(modelID: GGUFTranslationModelID)
        case remoteLLM(provider: RemoteLLMProvider)
    }

    var asrSelection: ASRSelection? {
        if rawValue == Self.dictation.rawValue {
            return .dictation
        }
        if let repo = payload(after: "mlx:") {
            return .mlx(repo: MLXModelManager.canonicalModelRepo(repo))
        }
        if payload(after: "sherpa:") != nil {
            // Migrate removed sherpa-onnx selections to the default local ASR.
            return .mlx(repo: MLXModelManager.defaultModelRepo)
        }
        if let modelID = payload(after: "whisper:") {
            return .mlx(repo: MLXWhisperMigrationSupport.repo(forLegacyWhisperModelID: modelID))
        }
        if let value = payload(after: "remote-asr:"),
           let provider = RemoteASRProvider(rawValue: value) {
            return .remote(provider: provider)
        }
        return nil
    }

    var textSelection: TextSelection? {
        if rawValue == Self.appleIntelligence.rawValue {
            return .appleIntelligence
        }
        if let repo = payload(after: "local-llm:") {
            return .localLLM(repo: repo)
        }
        if let value = payload(after: "remote-llm:"),
           let provider = RemoteLLMProvider(rawValue: value) {
            return .remoteLLM(provider: provider)
        }
        return nil
    }

    var translationSelection: TranslationSelection? {
        if rawValue == Self.legacyWhisperDirectTranslateRawValue {
            return .localLLM(repo: CustomLLMModelManager.defaultModelRepo)
        }
        if let repo = payload(after: "local-llm:") {
            return .localLLM(repo: repo)
        }
        if let value = payload(after: "local-gguf-translation:") {
            // Resolve removed/unknown GGUF variants to the supported default instead of
            // leaving a persisted selection unusable.
            return .localGGUF(modelID: GGUFTranslationModelCatalog.resolvedModelID(value))
        }
        if let value = payload(after: "remote-llm:"),
           let provider = RemoteLLMProvider(rawValue: value) {
            return .remoteLLM(provider: provider)
        }
        return nil
    }

    private func payload(after prefix: String) -> String? {
        guard rawValue.hasPrefix(prefix) else { return nil }
        let value = String(rawValue.dropFirst(prefix.count))
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return value.isEmpty ? nil : value
    }

    static func fromTranscriptSummaryModelSelection(_ rawValue: String?) -> Self? {
        let trimmed = rawValue?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !trimmed.isEmpty else { return nil }
        if trimmed == appleIntelligence.rawValue {
            return .appleIntelligence
        }
        if trimmed.hasPrefix("custom-llm:") {
            return .localLLM(String(trimmed.dropFirst("custom-llm:".count)))
        }
        if trimmed.hasPrefix("remote-llm:") {
            let providerRaw = String(trimmed.dropFirst("remote-llm:".count))
            if let provider = RemoteLLMProvider(rawValue: providerRaw) {
                return .remoteLLM(provider)
            }
        }
        return Self(rawValue: trimmed)
    }
}

enum ObsidianNoteGroupingMode: String, Codable, CaseIterable, Hashable, Sendable {
    case session
    case daily
    case file
}

nonisolated enum VoxtNotePanelCorner: String, Codable, CaseIterable, Identifiable, Hashable, Sendable {
    case topLeft
    case topRight
    case bottomLeft
    case bottomRight

    var id: String { rawValue }

    var title: String {
        switch self {
        case .topLeft:
            return AppLocalization.localizedString("Top left")
        case .topRight:
            return AppLocalization.localizedString("Top right")
        case .bottomLeft:
            return AppLocalization.localizedString("Bottom left")
        case .bottomRight:
            return AppLocalization.localizedString("Bottom right")
        }
    }
}

nonisolated struct VoxtNotePanelSettings: Codable, Hashable, Sendable {
    var corner: VoxtNotePanelCorner
    var revealDelay: Double
    var hideDelay: Double
    var isTranslucent: Bool

    init(
        corner: VoxtNotePanelCorner = .topRight,
        revealDelay: Double = 0.2,
        hideDelay: Double = 0.3,
        isTranslucent: Bool = true
    ) {
        self.corner = corner
        self.revealDelay = revealDelay
        self.hideDelay = hideDelay
        self.isTranslucent = isTranslucent
    }
}

struct RemindersNoteSyncSettings: Codable, Hashable, Sendable {
    var enabled: Bool
    var selectedListIdentifier: String
    var selectedListTitle: String

    init(
        enabled: Bool = true,
        selectedListIdentifier: String = "",
        selectedListTitle: String = ""
    ) {
        self.enabled = enabled
        self.selectedListIdentifier = selectedListIdentifier
        self.selectedListTitle = selectedListTitle
    }
}

struct ObsidianNoteSyncSettings: Codable, Hashable, Sendable {
    var enabled: Bool
    var vaultPath: String
    var vaultBookmarkData: Data?
    var relativeFolder: String
    var groupingMode: ObsidianNoteGroupingMode

    init(
        enabled: Bool = false,
        vaultPath: String = "",
        vaultBookmarkData: Data? = nil,
        relativeFolder: String = "Voxt",
        groupingMode: ObsidianNoteGroupingMode = .file
    ) {
        self.enabled = enabled
        self.vaultPath = vaultPath
        self.vaultBookmarkData = vaultBookmarkData
        self.relativeFolder = relativeFolder
        self.groupingMode = groupingMode
    }
}

struct TranscriptionNoteFeatureSettings: Codable, Hashable, Sendable {
    var enabled: Bool
    var triggerShortcut: TranscriptionNoteTriggerSettings
    var titleModelSelectionID: FeatureModelSelectionID
    var panel: VoxtNotePanelSettings
    var obsidianSync: ObsidianNoteSyncSettings
    var remindersSync: RemindersNoteSyncSettings

    init(
        enabled: Bool = true,
        triggerShortcut: TranscriptionNoteTriggerSettings = .defaultShortcut,
        titleModelSelectionID: FeatureModelSelectionID,
        panel: VoxtNotePanelSettings = .init(),
        obsidianSync: ObsidianNoteSyncSettings = .init(),
        remindersSync: RemindersNoteSyncSettings = .init()
    ) {
        self.enabled = enabled
        self.triggerShortcut = triggerShortcut
        self.titleModelSelectionID = titleModelSelectionID
        self.panel = panel
        self.obsidianSync = obsidianSync
        self.remindersSync = remindersSync
    }

    private enum CodingKeys: String, CodingKey {
        case enabled
        case triggerShortcut
        case titleModelSelectionID
        case panel
        case obsidianSync
        case remindersSync
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        enabled = try container.decodeIfPresent(Bool.self, forKey: .enabled) ?? true
        triggerShortcut = try container.decodeIfPresent(TranscriptionNoteTriggerSettings.self, forKey: .triggerShortcut) ?? .defaultShortcut
        titleModelSelectionID = try container.decode(FeatureModelSelectionID.self, forKey: .titleModelSelectionID)
        panel = try container.decodeIfPresent(VoxtNotePanelSettings.self, forKey: .panel) ?? .init()
        obsidianSync = try container.decodeIfPresent(ObsidianNoteSyncSettings.self, forKey: .obsidianSync) ?? .init()
        remindersSync = try container.decodeIfPresent(RemindersNoteSyncSettings.self, forKey: .remindersSync) ?? .init()
    }
}

struct TranscriptionFeatureSettings: Codable, Hashable, Sendable {
    var asrSelectionID: FeatureModelSelectionID
    var llmEnabled: Bool
    var llmSelectionID: FeatureModelSelectionID
    var prompt: String
    var promptPresetID: String?
    var notes: TranscriptionNoteFeatureSettings

    init(
        asrSelectionID: FeatureModelSelectionID,
        llmEnabled: Bool,
        llmSelectionID: FeatureModelSelectionID,
        prompt: String,
        promptPresetID: String? = nil,
        notes: TranscriptionNoteFeatureSettings? = nil
    ) {
        self.asrSelectionID = asrSelectionID
        self.llmEnabled = llmEnabled
        self.llmSelectionID = llmSelectionID
        self.prompt = prompt
        self.promptPresetID = promptPresetID
        self.notes = notes ?? TranscriptionNoteFeatureSettings(
            enabled: true,
            triggerShortcut: .defaultShortcut,
            titleModelSelectionID: llmSelectionID.textSelection == nil
                ? .localLLM(CustomLLMModelManager.defaultModelRepo)
                : llmSelectionID
        )
    }

    private enum CodingKeys: String, CodingKey {
        case asrSelectionID
        case llmEnabled
        case llmSelectionID
        case prompt
        case promptPresetID
        case notes
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let asrSelectionID = try container.decode(FeatureModelSelectionID.self, forKey: .asrSelectionID)
        let llmEnabled = try container.decode(Bool.self, forKey: .llmEnabled)
        let llmSelectionID = try container.decode(FeatureModelSelectionID.self, forKey: .llmSelectionID)
        let prompt = try container.decode(String.self, forKey: .prompt)
        let decodedNotes = try container.decodeIfPresent(TranscriptionNoteFeatureSettings.self, forKey: .notes)
        self.init(
            asrSelectionID: asrSelectionID,
            llmEnabled: llmEnabled,
            llmSelectionID: llmSelectionID,
            prompt: prompt,
            promptPresetID: try container.decodeIfPresent(String.self, forKey: .promptPresetID),
            notes: decodedNotes
        )
    }
}

struct TranslationFeatureSettings: Codable, Hashable, Sendable {
    var asrSelectionID: FeatureModelSelectionID
    var modelSelectionID: FeatureModelSelectionID
    var targetLanguageRawValue: String
    var prompt: String
    var promptPresetID: String?
    var showResultWindow: Bool

    init(
        asrSelectionID: FeatureModelSelectionID,
        modelSelectionID: FeatureModelSelectionID,
        targetLanguageRawValue: String,
        prompt: String,
        promptPresetID: String? = nil,
        showResultWindow: Bool = true
    ) {
        self.asrSelectionID = asrSelectionID
        self.modelSelectionID = modelSelectionID
        self.targetLanguageRawValue = targetLanguageRawValue
        self.prompt = prompt
        self.promptPresetID = promptPresetID
        self.showResultWindow = showResultWindow
    }

    var targetLanguage: TranslationTargetLanguage {
        TranslationTargetLanguage(rawValue: targetLanguageRawValue) ?? .english
    }

    private enum CodingKeys: String, CodingKey {
        case asrSelectionID
        case modelSelectionID
        case targetLanguageRawValue
        case prompt
        case promptPresetID
        case showResultWindow
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            asrSelectionID: try container.decode(FeatureModelSelectionID.self, forKey: .asrSelectionID),
            modelSelectionID: try container.decode(FeatureModelSelectionID.self, forKey: .modelSelectionID),
            targetLanguageRawValue: try container.decode(String.self, forKey: .targetLanguageRawValue),
            prompt: try container.decode(String.self, forKey: .prompt),
            promptPresetID: try container.decodeIfPresent(String.self, forKey: .promptPresetID),
            showResultWindow: try container.decodeIfPresent(Bool.self, forKey: .showResultWindow) ?? true
        )
    }
}

struct RewriteFeatureSettings: Codable, Hashable, Sendable {
    var asrSelectionID: FeatureModelSelectionID
    var llmSelectionID: FeatureModelSelectionID
    var prompt: String
    var promptPresetID: String?
    var appEnhancementEnabled: Bool
    var continueShortcut: TranscriptionContinueShortcutSettings

    init(
        asrSelectionID: FeatureModelSelectionID,
        llmSelectionID: FeatureModelSelectionID,
        prompt: String,
        promptPresetID: String? = nil,
        appEnhancementEnabled: Bool,
        continueShortcut: TranscriptionContinueShortcutSettings = .defaultShortcut
    ) {
        self.asrSelectionID = asrSelectionID
        self.llmSelectionID = llmSelectionID
        self.prompt = prompt
        self.promptPresetID = promptPresetID
        self.appEnhancementEnabled = appEnhancementEnabled
        self.continueShortcut = continueShortcut
    }

    private enum CodingKeys: String, CodingKey {
        case asrSelectionID
        case llmSelectionID
        case prompt
        case promptPresetID
        case appEnhancementEnabled
        case continueShortcut
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            asrSelectionID: try container.decode(FeatureModelSelectionID.self, forKey: .asrSelectionID),
            llmSelectionID: try container.decode(FeatureModelSelectionID.self, forKey: .llmSelectionID),
            prompt: try container.decode(String.self, forKey: .prompt),
            promptPresetID: try container.decodeIfPresent(String.self, forKey: .promptPresetID),
            appEnhancementEnabled: try container.decode(Bool.self, forKey: .appEnhancementEnabled),
            continueShortcut: try container.decodeIfPresent(TranscriptionContinueShortcutSettings.self, forKey: .continueShortcut) ?? .defaultShortcut
        )
    }
}

struct MeetingFeatureSettings: Codable, Hashable, Sendable {
    static let defaultFinalTranscriptOptimizationEnabled = true

    var asrSelectionID: FeatureModelSelectionID
    var summaryModelSelectionID: FeatureModelSelectionID
    var summaryPrompt: String
    var summaryAutoGenerate: Bool
    var realtimeTranslateEnabled: Bool
    var realtimeTargetLanguageRawValue: String
    var hideOverlayFromScreenSharing: Bool
    var chunkingModeRawValue: String
    var sileroVADSensitivityRawValue: String
    var speakerDiarizationModelRawValue: String
    var finalTranscriptOptimizationEnabled: Bool

    init(
        asrSelectionID: FeatureModelSelectionID,
        summaryModelSelectionID: FeatureModelSelectionID,
        summaryPrompt: String,
        summaryAutoGenerate: Bool,
        realtimeTranslateEnabled: Bool,
        realtimeTargetLanguageRawValue: String,
        hideOverlayFromScreenSharing: Bool,
        chunkingModeRawValue: String = MeetingChunkingMode.quality.rawValue,
        sileroVADSensitivityRawValue: String = MeetingSileroVADSensitivity.balanced.rawValue,
        speakerDiarizationModelRawValue: String = MeetingDiarizationMode.sortformerV2.rawValue,
        finalTranscriptOptimizationEnabled: Bool = MeetingFeatureSettings.defaultFinalTranscriptOptimizationEnabled
    ) {
        self.asrSelectionID = asrSelectionID
        self.summaryModelSelectionID = summaryModelSelectionID
        self.summaryPrompt = summaryPrompt
        self.summaryAutoGenerate = summaryAutoGenerate
        self.realtimeTranslateEnabled = realtimeTranslateEnabled
        self.realtimeTargetLanguageRawValue = realtimeTargetLanguageRawValue
        self.hideOverlayFromScreenSharing = hideOverlayFromScreenSharing
        self.chunkingModeRawValue = chunkingModeRawValue
        self.sileroVADSensitivityRawValue = sileroVADSensitivityRawValue
        self.speakerDiarizationModelRawValue = speakerDiarizationModelRawValue
        self.finalTranscriptOptimizationEnabled = finalTranscriptOptimizationEnabled
    }

    var realtimeTargetLanguage: TranslationTargetLanguage? {
        guard !realtimeTargetLanguageRawValue.isEmpty else { return nil }
        return TranslationTargetLanguage(rawValue: realtimeTargetLanguageRawValue)
    }

    var chunkingMode: MeetingChunkingMode {
        MeetingChunkingMode(rawValue: chunkingModeRawValue) ?? .quality
    }

    var speakerDiarizationModel: MeetingDiarizationMode {
        MeetingDiarizationMode(rawValue: speakerDiarizationModelRawValue) ?? .sortformerV2
    }

    var sileroVADSensitivity: MeetingSileroVADSensitivity {
        MeetingSileroVADSensitivity.resolved(rawValue: sileroVADSensitivityRawValue)
    }

    private enum CodingKeys: String, CodingKey {
        case asrSelectionID
        case summaryModelSelectionID
        case summaryPrompt
        case summaryAutoGenerate
        case realtimeTranslateEnabled
        case realtimeTargetLanguageRawValue
        case hideOverlayFromScreenSharing
        case chunkingModeRawValue
        case sileroVADSensitivityRawValue
        case speakerDiarizationModelRawValue
        case finalTranscriptOptimizationEnabled
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            asrSelectionID: try container.decode(FeatureModelSelectionID.self, forKey: .asrSelectionID),
            summaryModelSelectionID: try container.decode(FeatureModelSelectionID.self, forKey: .summaryModelSelectionID),
            summaryPrompt: try container.decode(String.self, forKey: .summaryPrompt),
            summaryAutoGenerate: try container.decode(Bool.self, forKey: .summaryAutoGenerate),
            realtimeTranslateEnabled: try container.decode(Bool.self, forKey: .realtimeTranslateEnabled),
            realtimeTargetLanguageRawValue: try container.decode(String.self, forKey: .realtimeTargetLanguageRawValue),
            hideOverlayFromScreenSharing: try container.decode(Bool.self, forKey: .hideOverlayFromScreenSharing),
            chunkingModeRawValue: try container.decodeIfPresent(String.self, forKey: .chunkingModeRawValue)
                ?? MeetingChunkingMode.quality.rawValue,
            sileroVADSensitivityRawValue: try container.decodeIfPresent(String.self, forKey: .sileroVADSensitivityRawValue)
                ?? MeetingSileroVADSensitivity.balanced.rawValue,
            speakerDiarizationModelRawValue: try container.decodeIfPresent(String.self, forKey: .speakerDiarizationModelRawValue)
                ?? MeetingDiarizationMode.sortformerV2.rawValue,
            finalTranscriptOptimizationEnabled: try container.decodeIfPresent(Bool.self, forKey: .finalTranscriptOptimizationEnabled)
                ?? MeetingFeatureSettings.defaultFinalTranscriptOptimizationEnabled
        )
    }
}

nonisolated struct FeatureAvailabilitySettings: Codable, Hashable, Sendable {
    var translationEnabled: Bool
    var rewriteEnabled: Bool
    var notesEnabled: Bool
    var appEnhancementEnabled: Bool
    var meetingEnabled: Bool
    var filesEnabled: Bool

    static let allEnabled = FeatureAvailabilitySettings(
        translationEnabled: true,
        rewriteEnabled: true,
        notesEnabled: true,
        appEnhancementEnabled: true,
        meetingEnabled: true,
        filesEnabled: true
    )

    init(
        translationEnabled: Bool = true,
        rewriteEnabled: Bool = true,
        notesEnabled: Bool = true,
        appEnhancementEnabled: Bool = true,
        meetingEnabled: Bool = true,
        filesEnabled: Bool = true
    ) {
        self.translationEnabled = translationEnabled
        self.rewriteEnabled = rewriteEnabled
        self.notesEnabled = notesEnabled
        self.appEnhancementEnabled = appEnhancementEnabled
        self.meetingEnabled = meetingEnabled
        self.filesEnabled = filesEnabled
    }

    private enum CodingKeys: String, CodingKey {
        case translationEnabled
        case rewriteEnabled
        case notesEnabled
        case appEnhancementEnabled
        case meetingEnabled
        case filesEnabled
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            translationEnabled: try container.decodeIfPresent(Bool.self, forKey: .translationEnabled) ?? true,
            rewriteEnabled: try container.decodeIfPresent(Bool.self, forKey: .rewriteEnabled) ?? true,
            notesEnabled: try container.decodeIfPresent(Bool.self, forKey: .notesEnabled) ?? true,
            appEnhancementEnabled: try container.decodeIfPresent(Bool.self, forKey: .appEnhancementEnabled) ?? true,
            meetingEnabled: try container.decodeIfPresent(Bool.self, forKey: .meetingEnabled) ?? true,
            filesEnabled: try container.decodeIfPresent(Bool.self, forKey: .filesEnabled) ?? true
        )
    }

}

struct FeatureSettings: Codable, Hashable, Sendable {
    var transcription: TranscriptionFeatureSettings
    var translation: TranslationFeatureSettings
    var rewrite: RewriteFeatureSettings
    var meeting: MeetingFeatureSettings
    var availability: FeatureAvailabilitySettings

    // SwiftUI may recreate a View value many times while it is rendering.
    // This value is intentionally cheap and side-effect free; persisted
    // settings must be loaded from an explicit lifecycle callback instead of
    // from a @State property initializer.
    static let placeholder = FeatureSettings(
        transcription: TranscriptionFeatureSettings(
            asrSelectionID: .dictation,
            llmEnabled: false,
            llmSelectionID: .localLLM(CustomLLMModelManager.defaultModelRepo),
            prompt: ""
        ),
        translation: TranslationFeatureSettings(
            asrSelectionID: .dictation,
            modelSelectionID: .localLLM(CustomLLMModelManager.defaultModelRepo),
            targetLanguageRawValue: TranslationTargetLanguage.english.rawValue,
            prompt: ""
        ),
        rewrite: RewriteFeatureSettings(
            asrSelectionID: .dictation,
            llmSelectionID: .localLLM(CustomLLMModelManager.defaultModelRepo),
            prompt: "",
            appEnhancementEnabled: true
        ),
        availability: .allEnabled
    )

    init(
        transcription: TranscriptionFeatureSettings,
        translation: TranslationFeatureSettings,
        rewrite: RewriteFeatureSettings,
        meeting: MeetingFeatureSettings? = nil,
        availability: FeatureAvailabilitySettings? = nil
    ) {
        self.transcription = transcription
        self.translation = translation
        self.rewrite = rewrite
        self.meeting = meeting ?? MeetingFeatureSettings(
            asrSelectionID: transcription.asrSelectionID,
            summaryModelSelectionID: transcription.llmSelectionID,
            summaryPrompt: "",
            summaryAutoGenerate: true,
            realtimeTranslateEnabled: false,
            realtimeTargetLanguageRawValue: "",
            hideOverlayFromScreenSharing: false
        )
        self.availability = availability ?? FeatureAvailabilitySettings(
            translationEnabled: true,
            rewriteEnabled: true,
            notesEnabled: transcription.notes.enabled,
            appEnhancementEnabled: rewrite.appEnhancementEnabled,
            meetingEnabled: true
        )
    }

    private enum CodingKeys: String, CodingKey {
        case transcription
        case translation
        case rewrite
        case meeting
        case availability
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let transcription = try container.decode(TranscriptionFeatureSettings.self, forKey: .transcription)
        let translation = try container.decode(TranslationFeatureSettings.self, forKey: .translation)
        let rewrite = try container.decode(RewriteFeatureSettings.self, forKey: .rewrite)
        let meeting = try container.decodeIfPresent(MeetingFeatureSettings.self, forKey: .meeting)
            ?? MeetingFeatureSettings(
                asrSelectionID: transcription.asrSelectionID,
                summaryModelSelectionID: transcription.llmSelectionID,
                summaryPrompt: "",
                summaryAutoGenerate: true,
                realtimeTranslateEnabled: false,
                realtimeTargetLanguageRawValue: "",
                hideOverlayFromScreenSharing: false
            )
        let decodedAvailability = try container.decodeIfPresent(
            FeatureAvailabilitySettings.self,
            forKey: .availability
        )
        self.init(
            transcription: transcription,
            translation: translation,
            rewrite: rewrite,
            meeting: meeting,
            availability: decodedAvailability ?? FeatureAvailabilitySettings(
                translationEnabled: true,
                rewriteEnabled: true,
                notesEnabled: transcription.notes.enabled,
                appEnhancementEnabled: rewrite.appEnhancementEnabled,
                meetingEnabled: true
            )
        )
    }

}
