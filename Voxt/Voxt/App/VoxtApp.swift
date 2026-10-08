// VoxtApp.swift
// Provides Voxt App for app lifecycle and routing.

import AppKit
import ApplicationServices
import CoreAudio
import AVFoundation
import Speech
import Carbon
import Combine
import Darwin

@MainActor
class AppDelegate: NSObject, NSApplicationDelegate {
    static weak var shared: AppDelegate?

    struct StoredBranchURLItem: Codable {
        let id: UUID
        let pattern: String
    }

    struct StoredAppBranchGroup: Codable {
        let id: UUID
        let name: String
        let prompt: String
        let appBundleIDs: [String]
        let urlPatternIDs: [UUID]
        let isExpanded: Bool

        enum CodingKeys: String, CodingKey {
            case id
            case name
            case prompt
            case appBundleIDs
            case urlPatternIDs
            case isExpanded
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            id = try container.decode(UUID.self, forKey: .id)
            name = try container.decode(String.self, forKey: .name)
            prompt = try container.decode(String.self, forKey: .prompt)
            appBundleIDs = try container.decodeIfPresent([String].self, forKey: .appBundleIDs) ?? []
            urlPatternIDs = try container.decodeIfPresent([UUID].self, forKey: .urlPatternIDs) ?? []
            isExpanded = try container.decodeIfPresent(Bool.self, forKey: .isExpanded) ?? true
        }
    }

    struct StoredCustomBrowser: Codable {
        let bundleID: String
        let displayName: String
    }

    struct BrowserScriptProvider {
        let name: String
        let scripts: [String]
    }

    struct EnhancementContextSnapshot {
        let appName: String?
        let bundleID: String?
        let pid: pid_t?
        let capturedAt: Date
    }

    struct OverlayEnhancementIconMatch: Equatable {
        enum Kind: Equatable {
            case app
            case url
        }

        let kind: Kind
        let bundleID: String
        let urlOrigin: String?
    }

    struct EnhancementPromptContext {
        let focusedAppName: String?
        let focusedAppBundleID: String?
        let matchedGroupID: UUID?
        let matchedGroupName: String?
        let matchedAppGroupName: String?
        let matchedURLGroupName: String?
        let overlayIconMatch: OverlayEnhancementIconMatch?
    }

    struct OutputDestinationContext: Equatable {
        let appName: String?
        let bundleID: String?
        let pid: pid_t?
        let browserURLHost: String?
        let browserURLOrigin: String?
    }

    enum MeetingSessionCompletionDisposition {
        case discard
        case save
        case saveAndOpenDetail
    }

    struct SessionLLMExecutionTiming: Equatable {
        let taskLabel: String
        let providerLabel: String
        let startedAt: Date
        let firstChunkAt: Date?
        let completedAt: Date
        let diagnostics: CustomLLMRunDiagnostics?
    }

    let speechTranscriber = SpeechTranscriber()
    var mlxTranscriber: MLXTranscriber?
    let remoteASRTranscriber = RemoteASRTranscriber()
    let mlxModelManager: MLXModelManager
    let customLLMManager: CustomLLMModelManager
    let ggufTranslationModelManager: GGUFTranslationModelManager
    let historyStore = TranscriptionHistoryStore()
    lazy var noteStore = VoxtNoteStore(inMemory: VoxtRuntimeEnvironment.isRunningUnitTests)
    let noteObsidianExportStore = VoxtNoteObsidianExportStore()
    let noteRemindersExportStore = VoxtNoteRemindersExportStore()
    let dictionaryStore = DictionaryStore()
    let dictionarySuggestionStore = DictionarySuggestionStore()
    let appUpdateManager = AppUpdateManager()
    let interactionSoundPlayer = InteractionSoundPlayer()
    let systemAudioMuteController = SystemAudioMuteController()

    let hotkeyManager = HotkeyManager()
    let overlayWindow = RecordingOverlayWindow()
    let toastWindow = FloatingToastWindow()
    let meetingOverlayWindow = MeetingOverlayWindow()
    let meetingDetailWindowManager = MeetingDetailWindowManager.shared
    let overlayState = OverlayState()
    lazy var noteWindowManager = VoxtNoteWindowManager(
        store: noteStore,
        settingsProvider: { [weak self] in
            self?.noteFeatureSettings.panel ?? .init()
        },
        onOpenSettings: { [weak self] in
            self?.openMainWindow(
                target: SettingsNavigationTarget(tab: .feature, featureTab: .note)
            )
        }
    )
    lazy var noteObsidianSyncCoordinator = VoxtObsidianSyncCoordinator(
        noteStore: noteStore,
        settingsProvider: { [weak self] in
            guard let notes = self?.noteFeatureSettings, notes.enabled else {
                return ObsidianNoteSyncSettings(enabled: false)
            }
            return notes.obsidianSync
        },
        exportStore: noteObsidianExportStore
    )
    lazy var noteRemindersSyncCoordinator = VoxtRemindersSyncCoordinator(
        noteStore: noteStore,
        settingsProvider: { [weak self] in
            guard let notes = self?.noteFeatureSettings, notes.enabled else {
                return RemindersNoteSyncSettings(enabled: false)
            }
            return notes.remindersSync
        },
        exportStore: noteRemindersExportStore
    )
    lazy var meetingSessionCoordinator = MeetingSessionCoordinator(
        mlxModelManager: mlxModelManager,
        preferredInputDeviceIDProvider: { [weak self] in
            self?.selectedInputDeviceID
        },
        realtimeTranslationTargetLanguageProvider: { [weak self] in
            self?.meetingRealtimeTranslationTargetLanguage
        },
        realtimeTranslationHandler: { [weak self] text, targetLanguage in
            guard let self else { return .cancelled() }
            return self.makeMeetingTranslationOperation(text, targetLanguage: targetLanguage)
        }
    )
    lazy var meetingFileTaskQueue = MeetingFileTaskQueue(
        analyzer: { @MainActor [weak self] sourceURL, originalFileName, progress in
            guard let self else { throw CancellationError() }
            return try await self.analyzeImportedMeetingFile(
                at: sourceURL, originalFileName: originalFileName, progress: progress
            )
        },
        cancelActiveAnalysis: { @MainActor [weak self] in
            await self?.cancelImportedMeetingFileAnalysis()
        },
        canStart: { @MainActor [weak self] in
            guard let self else { return false }
            return !self.isSessionActive && !self.meetingSessionCoordinator.isActive
        },
        rollbackAnalysis: { @MainActor [weak self] entry in
            _ = self?.historyStore.delete(id: entry.id)
        },
        onAnalysisCompleted: { @MainActor [weak self] taskID, entry in
            guard let self, self.meetingDetailWindowManager.hasFileTranscriptWindow(taskID: taskID) else { return }
            self.showMeetingDetailWindow(for: entry, replacingFileTaskID: taskID, activate: false)
        },
        onTaskRemoved: { @MainActor [weak self] taskID in
            self?.meetingDetailWindowManager.closeFileTranscript(taskID: taskID)
        }
    )
    var statusItem: NSStatusItem?

    var enhancer: (any TextEnhancing)?
    var mainWindowController: NSWindowController?
    var onboardingWindowController: NSWindowController?
    let mainWindowVisibilityState = MainWindowVisibilityState()
    var pendingStatusMenuActions: [() -> Void] = []
    var isStatusMenuOpen = false
    var statusMenuNeedsRebuild = false
    var statusMenuHistoryCancellable: AnyCancellable?
    var statusMenuHistoryGeneration = 0
    var statusMenuRecentEntries: [TranscriptionHistoryListEntry] = []
    private var interfaceLanguageObserver: NSObjectProtocol?
    private var updateAvailabilityObserver: NSObjectProtocol?
    private var selectedInputDeviceObserver: NSObjectProtocol?
    private var featureSettingsObserver: NSObjectProtocol?
    private var fileTaskNotificationObserver: NSObjectProtocol?
    var workspaceWillSleepObserver: NSObjectProtocol?
    var workspaceDidWakeObserver: NSObjectProtocol?
    var workspaceSessionDidBecomeActiveObserver: NSObjectProtocol?
    var workspaceSessionDidResignActiveObserver: NSObjectProtocol?
    var audioInputDevicesObserver: AudioInputDeviceObserver?
    var localEscapeKeyMonitor: Any?
    let overlayShortcutEventGate = OverlayShortcutEventGate()
    var inputDevicesRefreshTask: Task<Void, Never>?
    var inputDevicesSnapshot: [AudioInputDevice] = []
    var microphoneResolvedState = MicrophoneResolvedState.empty

    var isSessionActive = false
    var pendingSessionFinishTask: Task<Void, Never>?
    var silenceMonitorTask: Task<Void, Never>?
    var pauseLLMTask: Task<Void, Never>?
    var pendingDictionaryHistoryScanTask: Task<Void, Never>?
    var pendingAutomaticDictionaryLearningTask: Task<Void, Never>?
    var llmWarmupTasksByRepo: [String: Task<Void, Never>] = [:]
    var remoteLLMWarmupTasksByKey: [String: Task<Void, Never>] = [:]
    var overlayReminderTask: Task<Void, Never>?
    var overlayStatusClearTask: Task<Void, Never>?
    var toastDismissTask: Task<Void, Never>?
    var pendingSelectedTextTranslationRefreshTask: Task<Void, Never>?
    var pendingMeetingStartupTask: Task<Void, Never>?
    var pendingDeepIdleMemoryReclamationTask: Task<Void, Never>?
    var deepIdleMemoryReclamationID = UUID()
    var pendingApplicationTerminationTask: Task<Void, Never>?
    let llmRequests = LLMRequestLifecycle()
    let recordingCaptureStartTasks = TrackedTaskStore()
    let pasteboardTextWriter = PasteboardTextWriter()
    private(set) var isApplicationTerminating = false
    private var didCompleteApplicationTermination = false
    private var requestedApplicationExitStatus: Int32?
    var lastSignificantAudioAt = Date()
    var didTriggerPauseTranscription = false
    var didTriggerPauseLLM = false
    var localVADObservedFramesInCurrentSession = false
    var localVADObservedSpeechInCurrentSession = false
    var recordingVoiceActivityFrameDecider: RecordingVoiceActivityFrameDecider?
    var recordingVoiceActivitySegmenter: ASRVoiceActivitySegmenter?
    var recordingVoiceActivityMode: LocalVADMode?
    var recordingVoiceActivityUseCase: ASRVoiceActivityUseCase?
    var recordingVoiceActivityDebugStats = RecordingVoiceActivityDebugStats()
    var voiceEndCommandState = VoiceEndCommandState()
    let silenceAudioLevelThreshold: Float = 0.06
    let sessionFinishDelay: TimeInterval = 1.2
    var recordingRequestedAt: Date?
    var recordingStartedAt: Date?
    var recordingStoppedAt: Date?
    var transcriptionProcessingStartedAt: Date?
    var transcriptionResultReceivedAt: Date?
    var firstLiveASRPartialReceivedAt: Date?
    var sessionFinalOutputDeliveredAt: Date?
    var sessionLLMExecutionTimings: [SessionLLMExecutionTiming] = []
    var sessionOutputMode: SessionOutputMode = .transcription
    var isSelectedTextTranslationFlow = false
    var answerOverlayInjectionMode: AnswerOverlayInjectionMode = .standard
    var recordingLifecycle = RecordingSessionLifecycle()
    var didCommitSessionOutput: Bool { recordingLifecycle.hasCommittedOutput }
    var activeRecordingSessionID: UUID { recordingLifecycle.id }
    var currentEndingSessionID: UUID? { recordingLifecycle.endingID }
    var isSessionCancellationRequested: Bool { recordingLifecycle.isCancelled }
    var browserAutomationDeniedUntilByBundleID: [String: Date] = [:]
    var pendingCompletedHistoryAudioArchiveURL: URL?
    var latestInjectableOutputText: String?
    var sessionTargetApplicationPID: pid_t?
    var sessionTargetApplicationBundleID: String?
    var pendingTranscriptionStartTask: Task<Void, Never>?
    var pendingTranscriptionHotkeyStartBehavior: HotkeyPreference.TriggerBehavior?
    var isTranscriptionLongPressHotkeyDown = false
    var enhancementContextSnapshot: EnhancementContextSnapshot?
    var lastEnhancementPromptContext: EnhancementPromptContext?
    var sessionOutputDestinationContext: OutputDestinationContext?
    var selectedTextTranslationHadWritableFocusedInput = false
    var rewriteSessionHasSelectedSourceText = false
    var rewriteSessionSelectedSourceText = ""
    var rewriteSessionHadWritableFocusedInput = false
    var rewriteSessionFallbackInjectBundleID: String?
    var transcriptionCaptureSessionMode: TranscriptionCaptureSessionMode = .standard
    var transcriptionCapturePipeline: TranscriptionCapturePipeline = .liveDisplay
    var liveTranscriptSegmentationState = LiveTranscriptSegmentationState()
    var sessionTranslationTargetLanguageOverride: TranslationTargetLanguage?
    var selectedTextTranslationRefreshID = UUID()
    var activeSessionTranslationProviderResolution: TranslationProviderResolution?
    var pendingMeetingSessionCompletionDisposition: MeetingSessionCompletionDisposition = .save
    let tapStopGuardInterval: TimeInterval = 0.35
    let transcriptionStartDebounceInterval: TimeInterval = 0.08
    var mainWindowPresentationState = MainWindowPresentationState()

    override init() {
        let storedRepo = UserDefaults.standard.string(forKey: AppPreferenceKey.mlxModelRepo)
            ?? MLXModelManager.defaultModelRepo
        let repo = MLXModelManager.canonicalModelRepo(storedRepo)
        if repo != storedRepo {
            UserDefaults.standard.set(repo, forKey: AppPreferenceKey.mlxModelRepo)
        }
        mlxModelManager = MLXModelManager(modelRepo: repo, hubBaseURL: MLXModelManager.defaultHubBaseURL)
        let llmRepo = UserDefaults.standard.string(forKey: AppPreferenceKey.customLLMModelRepo)
            ?? CustomLLMModelManager.defaultModelRepo
        customLLMManager = CustomLLMModelManager(modelRepo: llmRepo, hubBaseURL: CustomLLMModelManager.defaultHubBaseURL)
        let ggufModelID = GGUFTranslationModelCatalog.resolvedModelID(
            UserDefaults.standard.string(forKey: AppPreferenceKey.translationGGUFModelID)
        )
        ggufTranslationModelManager = GGUFTranslationModelManager(modelID: ggufModelID)
        UserDefaults.standard.register(defaults: [
            AppPreferenceKey.interactionSoundsEnabled: true,
            AppPreferenceKey.interactionSoundPreset: InteractionSoundPreset.soft.rawValue,
            AppPreferenceKey.muteSystemAudioWhileRecording: false,
            AppPreferenceKey.overlayPosition: OverlayPosition.bottom.rawValue,
            AppPreferenceKey.overlayCardOpacity: 82,
            AppPreferenceKey.overlayCardCornerRadius: 24,
            AppPreferenceKey.overlayScreenEdgeInset: 30,
            AppPreferenceKey.interfaceLanguage: AppInterfaceLanguage.system.rawValue,
            AppPreferenceKey.translationTargetLanguage: TranslationTargetLanguage.english.rawValue,
            AppPreferenceKey.userMainLanguageCodes: UserMainLanguageOption.defaultStoredSelectionValue,
            AppPreferenceKey.translationModelProvider: TranslationModelProvider.customLLM.rawValue,
            AppPreferenceKey.translationGGUFModelID: GGUFTranslationModelCatalog.defaultModelID.rawValue,
            AppPreferenceKey.rewriteModelProvider: RewriteModelProvider.customLLM.rawValue,
            AppPreferenceKey.escapeKeyCancelsOverlaySession: true,
            AppPreferenceKey.translateSelectedTextOnTranslationHotkey: true,
            AppPreferenceKey.showSelectedTextTranslationResultWindow: true,
            AppPreferenceKey.customPasteHotkeyEnabled: false,
            AppPreferenceKey.hideMeetingOverlayFromScreenSharing: false,
            AppPreferenceKey.meetingOverlayCollapsed: false,
            AppPreferenceKey.meetingRealtimeTranslateEnabled: false,
            AppPreferenceKey.meetingRealtimeTranslationTargetLanguage: "",
            AppPreferenceKey.voiceEndCommandEnabled: false,
            AppPreferenceKey.voiceEndCommandPreset: VoiceEndCommandPreset.over.rawValue,
            AppPreferenceKey.voiceEndCommandText: "",
            AppPreferenceKey.autoCopyWhenNoFocusedInput: false,
            AppPreferenceKey.realtimeTextDisplayEnabled: true,
            AppPreferenceKey.alwaysShowRewriteAnswerCard: false,
            AppPreferenceKey.appEnhancementEnabled: true,
            AppPreferenceKey.translationSystemPrompt: "",
            AppPreferenceKey.rewriteSystemPrompt: "",
            AppPreferenceKey.asrHintSettings: ASRHintSettingsStore.defaultStoredValue(),
            AppPreferenceKey.translationFallbackModelProvider: TranslationModelProvider.customLLM.rawValue,
            AppPreferenceKey.rewriteCustomLLMModelRepo: CustomLLMModelManager.defaultModelRepo,
            AppPreferenceKey.remoteASRSelectedProvider: RemoteASRProvider.openAIWhisper.rawValue,
            AppPreferenceKey.remoteASRProviderConfigurations: "",
            AppPreferenceKey.remoteLLMSelectedProvider: RemoteLLMProvider.openAI.rawValue,
            AppPreferenceKey.remoteLLMProviderConfigurations: "",
            AppPreferenceKey.translationRemoteLLMProvider: "",
            AppPreferenceKey.rewriteRemoteLLMProvider: "",
            AppPreferenceKey.launchAtLogin: false,
            AppPreferenceKey.showInDock: true,
            AppPreferenceKey.historyEnabled: true,
            AppPreferenceKey.historyCleanupEnabled: true,
            AppPreferenceKey.historyRetentionPeriod: HistoryRetentionPeriod.ninetyDays.rawValue,
            AppPreferenceKey.historyRetentionCount: HistoryRetentionCount.unlimited.rawValue,
            AppPreferenceKey.historyAudioStorageEnabled: false,
            AppPreferenceKey.dictionaryRecognitionEnabled: true,
            AppPreferenceKey.dictionaryAutoLearningEnabled: true,
            AppPreferenceKey.dictionaryAutoLearningPrompt: "",
            AppPreferenceKey.dictionaryHighConfidenceCorrectionEnabled: true,
            AppPreferenceKey.autoCheckForUpdates: true,
            AppPreferenceKey.betaUpdatesEnabled: false,
            AppPreferenceKey.hotkeyDebugLoggingEnabled: false,
            AppPreferenceKey.llmDebugLoggingEnabled: false,
            AppPreferenceKey.meetingChunkingMode: MeetingChunkingMode.quality.rawValue,
            AppPreferenceKey.meetingSpeakerDiarizationModel: MeetingDiarizationMode.sortformerV2.rawValue,
            AppPreferenceKey.meetingFinalTranscriptOptimizationEnabled: true,
            AppPreferenceKey.networkProxyMode: VoxtNetworkSession.ProxyMode.system.rawValue,
            AppPreferenceKey.customProxyScheme: VoxtNetworkSession.ProxyScheme.http.rawValue,
            AppPreferenceKey.customProxyHost: "",
            AppPreferenceKey.customProxyPort: "",
            AppPreferenceKey.customProxyUsername: "",
            AppPreferenceKey.customProxyPassword: "",
        ])
        FeatureSettingsStore.migrateIfNeeded(defaults: .standard)
        Self.migrateLegacyWhisperSelectionIfNeeded()
        HotkeyPreference.registerDefaults()
        HotkeyPreference.migrateDefaultsIfNeeded()
        Self.migrateLegacyLocalModelMemoryPreferenceIfNeeded()
        Self.migrateLegacyNetworkProxyPreferenceIfNeeded()
        RemoteModelConfigurationStore.migrateLegacyLLMEndpoints()
        VoxtNetworkSession.migrateLegacyProxyCredentials()
        VoxtNetworkSession.clearProcessProxyEnvironmentOverridesIfNeeded(log: true)
        if VoxtNetworkSession.currentProxySettings.mode == .disabled,
           let systemProxy = VoxtNetworkSession.currentSystemProxyStatus.preferredSummary {
            VoxtLog.warning("Voxt direct proxy mode is enabled while macOS system proxy remains active. systemProxy=\(systemProxy)")
        }
        super.init()
        AppDelegate.shared = self
        mlxModelManager.onModelUnloaded = { [weak self] in
            self?.scheduleDeepIdleMemoryReclamation()
        }
        customLLMManager.onModelUnloaded = { [weak self] in
            self?.scheduleDeepIdleMemoryReclamation()
        }
    }

    private static func migrateLegacyLocalModelMemoryPreferenceIfNeeded() {
        let defaults = UserDefaults.standard
        guard defaults.object(forKey: AppPreferenceKey.localModelIdleUnloadDelaySeconds) == nil else { return }
        let resolvedDelay = AppPreferenceKey.resolvedLocalModelIdleUnloadDelaySeconds(defaults: defaults)
        defaults.set(resolvedDelay, forKey: AppPreferenceKey.localModelIdleUnloadDelaySeconds)
    }

    private static func migrateLegacyWhisperSelectionIfNeeded() {
        let defaults = UserDefaults.standard
        guard defaults.string(forKey: AppPreferenceKey.transcriptionEngine) == "whisperKit" else { return }
        let legacyModelID = defaults.string(forKey: AppPreferenceKey.legacyWhisperModelID)
            ?? MLXWhisperMigrationSupport.defaultLegacyModelID
        defaults.set(TranscriptionEngine.mlxAudio.rawValue, forKey: AppPreferenceKey.transcriptionEngine)
        defaults.set(
            MLXWhisperMigrationSupport.repo(forLegacyWhisperModelID: legacyModelID),
            forKey: AppPreferenceKey.mlxModelRepo
        )
    }

    var transcriptionEngine: TranscriptionEngine {
        get {
            let raw = UserDefaults.standard.string(forKey: AppPreferenceKey.transcriptionEngine)
            if raw == "whisperKit" {
                Self.migrateLegacyWhisperSelectionIfNeeded()
            }
            let resolved = TranscriptionEngine.resolved(rawValue: raw)
            return resolved
        }
        set {
            UserDefaults.standard.set(newValue.rawValue, forKey: AppPreferenceKey.transcriptionEngine)
        }
    }

    var enhancementMode: EnhancementMode {
        get {
            EnhancementMode.resolved(
                storedRawValue: UserDefaults.standard.string(forKey: AppPreferenceKey.enhancementMode),
                appleIntelligenceAvailable: appleIntelligenceAvailableForCurrentEnvironment,
                customLLMAvailable: customEnhancementModelAvailable,
                remoteLLMAvailable: remoteEnhancementModelAvailable
            )
        }
        set {
            let previous = enhancementMode
            UserDefaults.standard.set(newValue.rawValue, forKey: AppPreferenceKey.enhancementMode)
            if previous != newValue {
                VoxtLog.info("Enhancement mode changed: \(previous.rawValue) -> \(newValue.rawValue)")
                if newValue == .off {
                    VoxtLog.info("Custom LLM downloaded models are preserved when enhancement is off.")
                }
            }
        }
    }

    var appEnhancementEnabled: Bool {
        UserDefaults.standard.bool(forKey: AppPreferenceKey.appEnhancementEnabled)
    }

    private var appleIntelligenceAvailableForCurrentEnvironment: Bool {
        AppleIntelligenceAvailability.current.isAvailable
    }

    private var customEnhancementModelAvailable: Bool {
        customLLMManager.canAttemptInference(repo: customLLMManager.currentModelRepo)
    }

    private var remoteEnhancementModelAvailable: Bool {
        RemoteModelConfigurationStore.isStoredLLMConfigurationConfigured(
            provider: remoteLLMSelectedProvider,
            stored: remoteLLMConfigurations
        )
    }

    private var isRunningUnitTests: Bool { VoxtRuntimeEnvironment.isRunningUnitTests }

    private var currentSystemVersionLogDescription: String {
        let version = ProcessInfo.processInfo.operatingSystemVersion
        let versionString = "macOS \(version.majorVersion).\(version.minorVersion).\(version.patchVersion)"
        let buildString = ProcessInfo.processInfo.operatingSystemVersionString
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return "\(versionString) (\(buildString))"
    }

    func refreshTextEnhancerAvailability() {
        guard #available(macOS 26.0, *) else {
            enhancer = nil
            return
        }

        if TextEnhancer.isAvailable {
            if enhancer == nil {
                enhancer = TextEnhancer()
                VoxtLog.info("Apple Intelligence text enhancer initialized after availability refresh.")
            }
        } else {
            enhancer = nil
        }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        LoggingBootstrap.bootstrap()
        _ = noteObsidianSyncCoordinator
        _ = noteRemindersSyncCoordinator
        VoxtLog.info("Voxt launching.")
        VoxtLog.info("Runtime system version: \(currentSystemVersionLogDescription)")
        UserDefaults.standard.set(false, forKey: "NSQuitAlwaysKeepsWindows")
        migrateLegacyPreferences()
        remoteASRTranscriber.doubaoDictionaryEntryProvider = { [weak self] in
            guard let self else { return [] }
            return self.dictionaryStore.activeEntriesForRemoteRequest(
                activeGroupID: self.activeDictionaryGroupID(),
                limit: 5_000
            )
        }

        if isRunningUnitTests {
            VoxtLog.info("Voxt launch running under XCTest; skipping app startup services.")
            return
        }

        if maybeRunLLMSmokeAndTerminate() {
            VoxtLog.info("Voxt launch entering LLM smoke mode.")
            return
        }

        SystemNotificationSupport.configure()
        interactionSoundPlayer.prewarm()

        fileTaskNotificationObserver = NotificationCenter.default.addObserver(
            forName: .voxtFileTaskNotificationTapped,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            guard let taskID = notification.userInfo?["fileTaskID"] as? String else { return }
            Task { @MainActor [weak self] in
                self?.handleMeetingFileTaskNotificationTap(taskID: taskID)
            }
        }
        meetingFileTaskQueue.startIfNeeded()

        SileroVADModelProvisioner.prefetchIfNeeded(for: LocalVADMode.stored())
        RemoteModelConfigurationStore.migrateLegacyStoredSecrets()

        synchronizeAppActivationPolicy()

        refreshTextEnhancerAvailability()

        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        if let button = statusItem?.button {
            if let icon = NSImage(named: "voxt") {
                icon.size = NSSize(width: 18, height: 18)
                icon.isTemplate = true
                button.image = icon
            } else {
                button.image = NSImage(systemSymbolName: "waveform.circle", accessibilityDescription: "Voxt")
            }
            button.image?.accessibilityDescription = "Voxt"
        }
        appUpdateManager.syncAutomaticallyChecksForUpdates(autoCheckForUpdates)
        startObservingAudioInputDevices()
        refreshInputDevicesSnapshot(reason: "launch")
        startObservingStatusMenuHistory()
        buildMenu()
        Task { @MainActor [weak self] in
            await self?.recoverInterruptedMeetingFinalizationIfNeeded()
        }
        appUpdateManager.onUpdatePresentationWillBegin = { [weak self] in
            self?.prepareMainWindowForUpdatePresentation()
        }
        appUpdateManager.onUpdatePresentationDidEnd = { [weak self] in
            self?.restoreMainWindowAfterUpdateSessionIfNeeded()
        }
        selectedInputDeviceObserver = NotificationCenter.default.addObserver(
            forName: .voxtSelectedInputDeviceDidChange,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                let previousState = self.microphoneResolvedState
                self.microphoneResolvedState = MicrophonePreferenceManager.syncState(
                    defaults: .standard,
                    availableDevices: self.inputDevicesSnapshot
                )
                self.handleResolvedMicrophoneStateChange(
                    from: previousState,
                    to: self.microphoneResolvedState,
                    reason: "microphone preferences updated"
                )
                self.buildMenu()
            }
        }
        interfaceLanguageObserver = NotificationCenter.default.addObserver(
            forName: .voxtInterfaceLanguageDidChange,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                AppLocalization.refreshLanguageCache()
                self?.buildMenu()
            }
        }
        updateAvailabilityObserver = NotificationCenter.default.addObserver(
            forName: .voxtUpdateAvailabilityDidChange,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.buildMenu()
            }
        }
        featureSettingsObserver = NotificationCenter.default.addObserver(
            forName: .voxtFeatureSettingsDidChange,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.applyFeatureAvailabilityLifecycle()
            }
        }

        setupHotkey()
        noteWindowManager.updateLifecycle(
            isEnabled: FeatureSettingsStore.availability().notesEnabled
        )
        setupLifecycleRecoveryObservers()
        setupEscapeKeyMonitoring()
        overlayWindow.onRequestClose = mainActorCallback { $0.dismissAnswerOverlay() }
        overlayWindow.onRequestInject = mainActorCallback { $0.injectAnswerOverlayContent() }
        overlayWindow.onRequestContinue = mainActorCallback { $0.continueRewriteConversation() }
        overlayWindow.onRequestConversationRecordToggle = mainActorCallback { $0.toggleRewriteConversationRecording() }
        overlayWindow.onRequestDetail = mainActorCallback { $0.showCurrentTranscriptionDetailWindow() }
        overlayWindow.onRequestSessionTranslationTargetPickerToggle = mainActorCallback { $0.toggleSessionTranslationTargetPicker() }
        overlayWindow.onRequestSessionTranslationTargetLanguageSelect = mainActorCallback { appDelegate, language in
            appDelegate.selectSessionTranslationTargetLanguage(language)
        }
        overlayWindow.onRequestSessionTranslationTargetPickerDismiss = mainActorCallback {
            $0.dismissSessionTranslationTargetPicker()
        }
        meetingOverlayWindow.onRequestClose = mainActorCallback { $0.requestMeetingSessionCloseConfirmation() }
        meetingOverlayWindow.onRequestCollapseToggle = mainActorCallback { $0.toggleMeetingOverlayCollapse() }
        meetingOverlayWindow.onRequestPauseToggle = mainActorCallback { $0.toggleMeetingPause() }
        meetingOverlayWindow.onRequestDetail = mainActorCallback { $0.showLiveMeetingDetailWindow() }
        meetingOverlayWindow.onRequestRealtimeTranslateToggle = mainActorCallback { appDelegate, isEnabled in
            appDelegate.handleMeetingRealtimeTranslationToggle(isEnabled)
        }
        meetingOverlayWindow.onRequestCaptureModeChange = mainActorCallback { appDelegate, mode in
            appDelegate.handleMeetingCaptureModeSelection(mode)
        }
        meetingOverlayWindow.onRequestCaptureModePickerToggle = mainActorCallback { $0.toggleMeetingCaptureModePicker() }
        meetingOverlayWindow.onRequestCaptureModePickerDismiss = mainActorCallback { $0.dismissMeetingCaptureModePicker() }
        meetingOverlayWindow.onRequestRealtimeTranslationLanguageConfirm = mainActorCallback {
            $0.confirmMeetingRealtimeTranslationLanguageSelection()
        }
        meetingOverlayWindow.onRequestRealtimeTranslationLanguageCancel = mainActorCallback {
            $0.cancelMeetingRealtimeTranslationLanguageSelection()
        }
        meetingOverlayWindow.onRequestCancelMeeting = mainActorCallback { $0.cancelMeetingSessionWithoutSaving() }
        meetingOverlayWindow.onRequestFinishMeeting = mainActorCallback { $0.finishMeetingSessionAndOpenDetail() }
        meetingOverlayWindow.onRequestDismissCloseConfirmation = mainActorCallback {
            $0.dismissMeetingSessionCloseConfirmation()
        }
        meetingOverlayWindow.onRequestCopySegment = mainActorCallback { appDelegate, segment in
            appDelegate.copyMeetingSegment(segment)
        }
        presentMainWindowOnLaunchIfNeeded()
        scheduleLLMIdleWarmupIfNeeded()
        VoxtLog.info("Voxt launch completed. engine=\(transcriptionEngine.rawValue), enhancement=\(enhancementMode.rawValue)")
    }

    func applicationDidBecomeActive(_ notification: Notification) {
        refreshTextEnhancerAvailability()
    }

    func applyFeatureAvailabilityLifecycle() {
        let availability = FeatureSettingsStore.availability()
        refreshOverlayShortcutEventGate()
        noteWindowManager.updateLifecycle(isEnabled: availability.notesEnabled)
        if !availability.meetingEnabled,
           meetingSessionCoordinator.isActive || meetingSessionCoordinator.isStartingUp {
            cancelMeetingSessionWithoutSaving()
        }
        buildMenu()
        scheduleLLMIdleWarmupIfNeeded()
        scheduleDeepIdleMemoryReclamation()
    }

    private func mainActorCallback(_ action: @escaping @MainActor (AppDelegate) -> Void) -> () -> Void {
        { [weak self] in
            Task { @MainActor [weak self] in
                guard let self else { return }
                action(self)
            }
        }
    }

    private func mainActorCallback<Value>(
        _ action: @escaping @MainActor (AppDelegate, Value) -> Void
    ) -> (Value) -> Void {
        { [weak self] value in
            Task { @MainActor [weak self] in
                guard let self else { return }
                action(self, value)
            }
        }
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        guard !flag else { return true }
        openMainWindow(selectTab: nil)
        return true
    }

    func requestApplicationTermination(exitStatus: Int32? = nil) {
        if let exitStatus {
            requestedApplicationExitStatus = exitStatus
        }
        NSApp.terminate(nil)
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        if didCompleteApplicationTermination {
            return .terminateNow
        }
        guard pendingApplicationTerminationTask == nil else {
            return .terminateLater
        }

        pendingApplicationTerminationTask = Task { @MainActor [weak self] in
            guard let self else {
                sender.reply(toApplicationShouldTerminate: true)
                return
            }
            await self.shutdownForApplicationTermination()
            self.didCompleteApplicationTermination = true
            self.pendingApplicationTerminationTask = nil
            if let exitStatus = self.requestedApplicationExitStatus {
                fflush(stdout)
                fflush(stderr)
                Darwin.exit(exitStatus)
            }
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }

    func applicationWillTerminate(_ notification: Notification) {
        performApplicationTerminationCleanup()
        VoxtLogFileStore.shared.flush()
    }

    private func performApplicationTerminationCleanup() {
        isApplicationTerminating = true
        pendingMeetingStartupTask?.cancel()
        pendingMeetingStartupTask = nil
        statusItem?.button?.isEnabled = false
        mainWindowController?.close()
        onboardingWindowController?.close()
        meetingDetailWindowManager.closeLiveWindow()
        overlayWindow.hide(animated: false)
        toastWindow.hide()
        meetingOverlayWindow.hide()
        noteWindowManager.stop()
        noteObsidianSyncCoordinator.shutdownForApplicationTermination()
        noteRemindersSyncCoordinator.shutdownForApplicationTermination()
        interactionSoundPlayer.reset()
        appUpdateManager.shutdownForApplicationTermination()
        systemAudioMuteController.restoreSystemAudioIfNeeded()
        audioInputDevicesObserver?.stop()
        audioInputDevicesObserver = nil
        removeApplicationObserversAndEventMonitors()
    }

    private func shutdownForApplicationTermination() async {
        guard !isApplicationTerminating else { return }
        isApplicationTerminating = true
        VoxtLog.info("Application termination cleanup started.")

        hotkeyManager.stop()
        let activeLLMTasks = cancelActiveLLMRequest()
        let recordingCaptureStartTasks = cancelRecordingCaptureStartTask()
        let pendingTranscriptionStart = pendingTranscriptionStartTask
        let pendingMeetingStartup = pendingMeetingStartupTask
        pendingTranscriptionStart?.cancel()
        pendingMeetingStartup?.cancel()
        recordingLifecycle.cancel()

        let tasksToCancel = applicationTasksForTermination()
        for task in tasksToCancel {
            task.cancel()
        }
        clearApplicationTaskReferencesForTermination()
        performApplicationTerminationCleanup()

        async let obsidianSyncCompleted = noteObsidianSyncCoordinator
            .waitForPendingApplicationTerminationSync(timeout: 2)
        async let remindersSyncCompleted = noteRemindersSyncCoordinator
            .waitForPendingApplicationTerminationSync(timeout: 2)

        // A meeting startup task can be awaiting MLXModelManager's shared, unstructured
        // loading task. Cancel that inner task before awaiting the outer startup barrier.
        mlxModelManager.cancelPendingModelLoadForApplicationTermination()

        await pendingTranscriptionStart?.value
        await pendingMeetingStartup?.value
        for task in recordingCaptureStartTasks {
            await task.value
        }

        await meetingFileTaskQueue.shutdown()

        let meetingStopTask = meetingSessionCoordinator.stop()
        speechTranscriber.shutdownForApplicationTermination()
        await remoteASRTranscriber.shutdownForApplicationTermination()
        await mlxTranscriber?.shutdownForApplicationTermination()
        await meetingStopTask?.value

        for task in tasksToCancel {
            await task.value
        }
        for task in activeLLMTasks {
            await task.value
        }

        await ggufTranslationModelManager.shutdownForApplicationTermination()
        await customLLMManager.shutdownForApplicationTermination()
        await mlxModelManager.shutdownForApplicationTermination()
        await SileroVADModelProvisioner.shared.shutdownForApplicationTermination()

        let syncCompletion = await (obsidianSyncCompleted, remindersSyncCompleted)
        if !syncCompletion.0 {
            VoxtLog.warning("Obsidian sync did not finish before the application termination deadline.")
        }
        if !syncCompletion.1 {
            VoxtLog.persistenceWarning("Reminders sync did not finish before the application termination deadline.")
        }

        VoxtLog.info("Application termination cleanup completed.")
        VoxtLogFileStore.shared.flush()
    }

    private func applicationTasksForTermination() -> [Task<Void, Never>] {
        var tasks = [
            inputDevicesRefreshTask,
            pendingSessionFinishTask,
            silenceMonitorTask,
            pauseLLMTask,
            pendingDictionaryHistoryScanTask,
            pendingAutomaticDictionaryLearningTask,
            overlayReminderTask,
            overlayStatusClearTask,
            toastDismissTask,
            pendingSelectedTextTranslationRefreshTask,
            pendingDeepIdleMemoryReclamationTask
        ].compactMap { $0 }
        tasks.append(contentsOf: llmWarmupTasksByRepo.values)
        tasks.append(contentsOf: remoteLLMWarmupTasksByKey.values)
        return tasks
    }

    private func clearApplicationTaskReferencesForTermination() {
        inputDevicesRefreshTask = nil
        pendingSessionFinishTask = nil
        silenceMonitorTask = nil
        pauseLLMTask = nil
        pendingDictionaryHistoryScanTask = nil
        pendingAutomaticDictionaryLearningTask = nil
        overlayReminderTask = nil
        overlayStatusClearTask = nil
        toastDismissTask = nil
        pendingSelectedTextTranslationRefreshTask = nil
        pendingDeepIdleMemoryReclamationTask = nil
        pendingMeetingStartupTask = nil
        pendingTranscriptionStartTask = nil
        llmWarmupTasksByRepo.removeAll()
        remoteLLMWarmupTasksByKey.removeAll()
    }

    private func removeApplicationObserversAndEventMonitors() {
        if let interfaceLanguageObserver {
            NotificationCenter.default.removeObserver(interfaceLanguageObserver)
            self.interfaceLanguageObserver = nil
        }
        if let updateAvailabilityObserver {
            NotificationCenter.default.removeObserver(updateAvailabilityObserver)
            self.updateAvailabilityObserver = nil
        }
        if let selectedInputDeviceObserver {
            NotificationCenter.default.removeObserver(selectedInputDeviceObserver)
            self.selectedInputDeviceObserver = nil
        }
        if let featureSettingsObserver {
            NotificationCenter.default.removeObserver(featureSettingsObserver)
            self.featureSettingsObserver = nil
        }
        if let fileTaskNotificationObserver {
            NotificationCenter.default.removeObserver(fileTaskNotificationObserver)
            self.fileTaskNotificationObserver = nil
        }
        let workspaceNotificationCenter = NSWorkspace.shared.notificationCenter
        if let workspaceWillSleepObserver {
            workspaceNotificationCenter.removeObserver(workspaceWillSleepObserver)
            self.workspaceWillSleepObserver = nil
        }
        if let workspaceDidWakeObserver {
            workspaceNotificationCenter.removeObserver(workspaceDidWakeObserver)
            self.workspaceDidWakeObserver = nil
        }
        if let workspaceSessionDidBecomeActiveObserver {
            workspaceNotificationCenter.removeObserver(workspaceSessionDidBecomeActiveObserver)
            self.workspaceSessionDidBecomeActiveObserver = nil
        }
        if let workspaceSessionDidResignActiveObserver {
            workspaceNotificationCenter.removeObserver(workspaceSessionDidResignActiveObserver)
            self.workspaceSessionDidResignActiveObserver = nil
        }
        if let localEscapeKeyMonitor {
            NSEvent.removeMonitor(localEscapeKeyMonitor)
            self.localEscapeKeyMonitor = nil
        }
    }

    deinit {
        if let interfaceLanguageObserver {
            NotificationCenter.default.removeObserver(interfaceLanguageObserver)
        }
        if let updateAvailabilityObserver {
            NotificationCenter.default.removeObserver(updateAvailabilityObserver)
        }
        if let selectedInputDeviceObserver {
            NotificationCenter.default.removeObserver(selectedInputDeviceObserver)
        }
        if let featureSettingsObserver {
            NotificationCenter.default.removeObserver(featureSettingsObserver)
        }
        if let fileTaskNotificationObserver {
            NotificationCenter.default.removeObserver(fileTaskNotificationObserver)
        }
        let workspaceNotificationCenter = NSWorkspace.shared.notificationCenter
        if let workspaceWillSleepObserver {
            workspaceNotificationCenter.removeObserver(workspaceWillSleepObserver)
        }
        if let workspaceDidWakeObserver {
            workspaceNotificationCenter.removeObserver(workspaceDidWakeObserver)
        }
        if let workspaceSessionDidBecomeActiveObserver {
            workspaceNotificationCenter.removeObserver(workspaceSessionDidBecomeActiveObserver)
        }
        if let workspaceSessionDidResignActiveObserver {
            workspaceNotificationCenter.removeObserver(workspaceSessionDidResignActiveObserver)
        }
        if let localEscapeKeyMonitor {
            NSEvent.removeMonitor(localEscapeKeyMonitor)
        }
        inputDevicesRefreshTask?.cancel()
        pendingMeetingStartupTask?.cancel()
        for task in llmWarmupTasksByRepo.values {
            task.cancel()
        }
        llmWarmupTasksByRepo.removeAll()
        for task in remoteLLMWarmupTasksByKey.values {
            task.cancel()
        }
        remoteLLMWarmupTasksByKey.removeAll()
    }

    private func migrateLegacyPreferences() {
        let defaults = UserDefaults.standard
        if defaults.string(forKey: AppPreferenceKey.enhancementMode) == nil,
           defaults.object(forKey: "aiEnhanceEnabled") != nil {
            let oldEnabled = defaults.bool(forKey: "aiEnhanceEnabled")
            enhancementMode = oldEnabled ? .appleIntelligence : .off
        }
    }

    private static func migrateLegacyNetworkProxyPreferenceIfNeeded() {
        let defaults = UserDefaults.standard
        guard defaults.string(forKey: AppPreferenceKey.networkProxyMode) == nil,
              defaults.object(forKey: AppPreferenceKey.useSystemProxy) != nil else {
            return
        }

        let legacyUsesSystemProxy = defaults.bool(forKey: AppPreferenceKey.useSystemProxy)
        let mode: VoxtNetworkSession.ProxyMode = legacyUsesSystemProxy ? .system : .disabled
        defaults.set(mode.rawValue, forKey: AppPreferenceKey.networkProxyMode)
    }

}
