// ASR debug recording, clip lifetime and request ownership.

import AVFoundation
import Combine
import Foundation

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
    private var installationSubscription: AnyCancellable?

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
        installationSubscription = mlxModelManager.$installationRevision
            .dropFirst()
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in
                MainActor.assumeIsolated {
                    self?.refreshOptions()
                }
            }
    }

    func refreshOptions() {
        remoteConfigurations = RemoteModelConfigurationStore.loadConfigurations(
            from: UserDefaults.standard.string(forKey: AppPreferenceKey.remoteASRProviderConfigurations) ?? ""
        )
        options = ModelDebugCatalog.availableASRModels(
            mlxModelManager: mlxModelManager,
            remoteASRConfigurations: remoteConfigurations
        )
        if selectedModelID.isEmpty {
            selectedModelID = preferredModelID()
        }
    }

    var canRunSelectedModel: Bool {
        options.first(where: { $0.id == selectedModelID })?.isSelectable == true
    }

    var selectedModelDisabledReason: String? {
        options.first(where: { $0.id == selectedModelID })?.disabledReason
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
        guard canRunSelectedModel else {
            showToast(selectedModelDisabledReason ?? modelDebugLocalized("No available model."))
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
        guard canRunSelectedModel else {
            showToast(selectedModelDisabledReason ?? modelDebugLocalized("No available model."))
            return
        }
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
        if let reason = option.disabledReason {
            statusMessage = reason
            showToast(reason)
            return
        }
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
