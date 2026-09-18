// SPDX-License-Identifier: Apache-2.0
import AVFoundation
import AppKit
import Combine
import SwiftUI

@MainActor
final class AppModel: ObservableObject {
    enum Phase: String { case idle, starting, recording, processing, preparing }
    let store: AppStore
    let recorder = AudioRecorder()
    let worker = WorkerClient()
    private let shortcut = GlobalShortcut()
    @Published var page: Page = .home
    @Published var phase: Phase = .idle
    @Published var mode: VoiceMode = .dictate
    @Published var resultText = ""
    @Published var rawText = ""
    @Published var liveText = ""
    @Published var liveStatus = ""
    @Published var notice = ""
    @Published var error = ""
    @Published var lastApp = ""
    // ponytail: keys stay in memory for this session; use Keychain if persistence is needed.
    @Published var textAPIKey = ""
    @Published var textModels: [String] = []
    private var sessionAPIKey = ""
    @Published var microphoneAllowed = AVCaptureDevice.authorizationStatus(for: .audio) == .authorized
    @Published var accessibilityAllowed = false
    // Note (Jiaxin Deng): Ad-hoc rebuilds can invalidate a grant still shown as enabled in System Settings.
    @Published private(set) var accessibilityGrantStale = false
    var showMainWindow: (() -> Void)?
    var showVoicePanel: (() -> Void)?
    var hideVoicePanel: (() -> Void)?
    private var target: InsertionTarget?
    var task: Task<Void, Never>?
    @Published var preloadTask: Task<[String: Any], Error>?
    var isPreloading: Bool { preloadTask != nil }
    private var capturingShortcut = false
    private var isShutDown = false
    private var speechStream: ASRStream?
    var generation = UUID()
    private var timer: Timer?
    private var preferencesSubscription: AnyCancellable?
    private struct FailedRecording {
        let url: URL
        let duration: Double
        let mode: VoiceMode
        let target: InsertionTarget?
        let appName: String
    }
    private var retryRecording: FailedRecording?
    private var sessionPreferences = Preferences()

    init(store: AppStore? = nil) {
        let store = store ?? AppStore()
        self.store = store
        if store.preferences.pythonExecutable.isEmpty {
            store.preferences.pythonExecutable = Bundle.main.object(forInfoDictionaryKey: "OmniTyperPython") as? String
                ?? FileManager.default.homeDirectoryForCurrentUser
                    .appendingPathComponent("Library/Application Support/OmniTyper/runtime/bin/python").path
        }
        refreshPermissions()
        preferencesSubscription = store.$preferences.dropFirst().removeDuplicates().sink { [weak self] preferences in
            if preferences.textSettings.baseURL != self?.store.preferences.textSettings.baseURL {
                self?.textAPIKey = ""
                self?.textModels = []
            }
            // Note (Codex): Published emits before storage changes; use its value without queuing work past shutdown.
            self?.configureShortcut(preferences)
        }
        configureShortcut(store.preferences)
        Diagnostics.record("app.start", [
            "version": Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "?",
            "language": store.preferences.uiLanguage ?? "system",
            "style": store.preferences.style,
            // Note (Jiaxin Deng): Do not log the endpoint; it can identify a private host.
            "textAPI": String(!store.preferences.textSettings.model.isEmpty),
        ])
        TextInsertion.enableAccessibilityInHostedApps()
        timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                if self.phase == .recording && self.recorder.elapsed >= 300 { self.finish() }
                self.refreshPermissions()
                self.store.prune()
            }
        }
        if store.preferences.retainsSpeechModel { prepareModels() }
    }

    var isBusy: Bool { phase != .idle }
    var shortcutLabel: String {
        ShortcutCapture.label(keyCode: store.preferences.shortcutKeyCode,
                              modifiers: store.preferences.shortcutModifiers)
    }

    private func configureShortcut(_ preferences: Preferences) {
        guard !capturingShortcut, !isShutDown else { return }
        shortcut.start(keyCode: preferences.shortcutKeyCode, modifiers: preferences.shortcutModifiers,
                       hold: preferences.holdToTalk,
                       onStart: { [weak self] in
                           guard let self, !self.capturingShortcut, !self.isShutDown else { return }
                           self.toggle()
                       },
                       onStop: { [weak self] in
                           guard let self, !self.capturingShortcut, !self.isShutDown else { return }
                           if self.phase == .recording { self.finish() }
                           else if self.phase == .starting { self.cancel() }
                       },
                       onCancel: { [weak self] in if self?.isBusy == true { self?.cancel() } })
    }

    func beginShortcutCapture() {
        capturingShortcut = true
        shortcut.stop()
    }

    func endShortcutCapture() {
        guard capturingShortcut else { return }
        capturingShortcut = false
        configureShortcut(store.preferences)
    }

    func refreshPermissions() {
        let previous = accessibilityAllowed
        let trusted = TextInsertion.isTrusted
        let microphone = AVCaptureDevice.authorizationStatus(for: .audio) == .authorized
        // Note (Jiaxin Deng): Unchanged Published assignments still redraw views on every permission poll.
        if accessibilityAllowed != trusted || microphoneAllowed != microphone {
            Diagnostics.record("permission", ["accessibility": String(trusted), "microphone": String(microphone)])
        }
        if accessibilityAllowed != trusted { accessibilityAllowed = trusted }
        if microphoneAllowed != microphone { microphoneAllowed = microphone }
        if trusted {
            if store.preferences.accessibilityWasTrusted != true { store.preferences.accessibilityWasTrusted = true }
            if accessibilityGrantStale { accessibilityGrantStale = false }
        } else {
            let stale = store.preferences.accessibilityWasTrusted == true
            if accessibilityGrantStale != stale { accessibilityGrantStale = stale }
        }
        if trusted && !previous { configureShortcut(store.preferences) }
    }

    func requestMicrophone() {
        Task { _ = await AVCaptureDevice.requestAccess(for: .audio); refreshPermissions() }
    }

    func requestAccessibility() { TextInsertion.requestPermission(); refreshPermissions() }

    func toggle(_ requestedMode: VoiceMode? = nil) {
        if phase == .recording { finish(); return }
        guard phase == .idle else { return }
        if let requestedMode { mode = requestedMode }
        start()
    }

    private func start() {
        error = ""; notice = ""; target = nil; liveText = ""; liveStatus = L("status.loadingModel")
        sessionPreferences = store.preferences
        sessionAPIKey = textAPIKey
        do {
            let captured = try TextInsertion.capture()
            if captured.bundleID != Bundle.main.bundleIdentifier { target = captured }
        } catch TextInsertionError.secureField {
            error = L("sys.secureField")
            showMainWindow?()
            return
        } catch {
            // Note (Jiaxin Deng): Keep copy-only dictation available and distinguish missing permissions from opaque fields.
            notice = accessibilityAllowed ? L("notice.fieldNotAccessible")
                : accessibilityGrantStale ? L("home.permissions.axStale") : L("sys.axPermission")
            Diagnostics.record("capture.failed", ["reason": Diagnostics.code(of: error),
                                                  "accessibility": String(accessibilityAllowed)])
        }
        if mode == .edit && (target?.selectedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true) {
            error = L("error.editNeedsSelection")
            showMainWindow?()
            return
        }
        lastApp = target?.applicationName ?? "OmniTyper"
        do { _ = try payload(audio: nil) }
        catch { self.error = error.localizedDescription; showMainWindow?(); return }
        phase = .starting
        showVoicePanel?()
        let token = UUID(); generation = token
        task = Task { [self] in
            do {
                let response: [String: Any]
                if let preloadTask { response = try await preloadTask.value }
                else {
                    response = try await worker.request(["op": "prepare", "asr_model": sessionPreferences.asrModel],
                                                        python: sessionPreferences.pythonExecutable)
                }
                guard generation == token, !Task.isCancelled else { return }
                do {
                    let stream = try ASRStream(url: response["realtime_url"] as? String ?? "", onPartial: { [weak self] text in
                        guard let self, self.generation == token else { return }
                        self.liveText = text
                    }, onFailure: { [weak self] in
                        guard let self, self.generation == token else { return }
                        self.liveStatus = L("status.livePreviewSaved")
                    })
                    speechStream = stream
                    try await stream.connect(language: sessionPreferences.language)
                    liveStatus = L("status.listening")
                } catch {
                    guard generation == token, !Task.isCancelled else { return }
                    speechStream?.cancel(); speechStream = nil
                    liveStatus = L("status.livePreviewLater")
                }
                try await recorder.start(deviceUID: sessionPreferences.microphoneUID, onPCM: speechStream?.audioInput)
                guard generation == token, !Task.isCancelled else { recorder.cancel(); return }
                discardRetryRecording()
                phase = .recording; showVoicePanel?()
                if sessionPreferences.sounds { NSSound(named: "Tink")?.play() }
            } catch {
                guard generation == token else { return }
                speechStream?.cancel(); speechStream = nil
                target = nil; phase = .idle; hideVoicePanel?(); self.error = error.localizedDescription; refreshPermissions(); showMainWindow?()
                releaseIdleModel()
            }
        }
    }

    func finish() {
        guard phase == .recording else { return }
        do {
            let duration = recorder.elapsed
            let audio = try recorder.stop()
            if sessionPreferences.sounds { NSSound(named: "Pop")?.play() }
            run(audio: audio, duration: duration, allowInsertion: true)
        } catch {
            speechStream?.cancel(); speechStream = nil
            target = nil; phase = .idle; self.error = error.localizedDescription; hideVoicePanel?(); showMainWindow?()
            releaseIdleModel()
        }
    }

    private func payload(audio: URL?, text: String? = nil) throws -> [String: Any] {
        let preferences = sessionPreferences
        let rule = store.rules.first { $0.bundleID == target?.bundleID }
        let instructions = try Preferences.combinedInstructions(preferences.instructions, rule?.instructions ?? "")
        guard store.dictionary.count <= 200, store.dictionary.allSatisfy(\.isValid) else {
            throw Failure("error.dictionaryInvalid")
        }
        let selectedText = (mode == .edit || mode == .ask) ? (target?.selectedText ?? "") : ""
        guard selectedText.unicodeScalars.count <= 12_000, !selectedText.contains("\0") else {
            throw Failure("error.selectionTooLong")
        }
        var request: [String: Any] = [
            "op": audio == nil ? "process" : "transcribe",
            "asr_model": preferences.asrModel,
            "mode": mode.rawValue, "language": preferences.language,
            "target_language": preferences.targetLanguage, "style": rule?.style ?? preferences.style,
            "instructions": instructions,
            "dictionary": store.dictionary.map { ["spoken": $0.spoken, "written": $0.written] },
            "selected_text": selectedText, "app_name": lastApp
        ]
        if mode != .dictate || (rule?.style ?? preferences.style) != "verbatim" {
            request.merge(try preferences.textSettings.payload(apiKey: sessionAPIKey)) { _, new in new }
        }
        if let audio { request["audio_path"] = audio.path }
        if let text { request["text"] = text }
        return request
    }

    private func run(audio: URL, duration: Double, allowInsertion: Bool) {
        phase = .processing; error = ""; notice = ""
        let token = UUID(); generation = token
        let request = Result { try payload(audio: audio) }
        let requestMode = mode
        let capturedTarget = target
        let preferences = sessionPreferences
        let recording = FailedRecording(url: audio, duration: duration, mode: requestMode,
                                        target: capturedTarget, appName: lastApp)
        task = Task {
            do {
                if let preloadTask { _ = try await preloadTask.value }
                try Task.checkCancellation()
                var payload = try request.get()
                var streamingWarning = ""
                if let stream = speechStream {
                    liveStatus = L("status.finalizing")
                    do {
                        let transcript = try await stream.finish()
                        payload["op"] = "process"
                        payload["audio_path"] = nil
                        payload["text"] = transcript
                    } catch {
                        guard generation == token, !Task.isCancelled else { throw CancellationError() }
                        streamingWarning = L("notice.streamRecovered")
                    }
                    stream.cancel(); speechStream = nil
                }
                try Task.checkCancellation()
                liveStatus = payload["op"] as? String == "process" ? L("status.processingText") : L("status.transcribing")
                let response = try await worker.request(payload, python: preferences.pythonExecutable)
                guard generation == token, !Task.isCancelled else { try? FileManager.default.removeItem(at: audio); return }
                let text = response["text"] as? String ?? ""
                let raw = response["raw_text"] as? String ?? text
                let warning = [streamingWarning, response["warning"] as? String ?? ""].filter { !$0.isEmpty }.joined(separator: " ")
                resultText = text; rawText = raw
                if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    notice = L("notice.noSpeech")
                    try? FileManager.default.removeItem(at: audio)
                } else {
                    let entry = HistoryEntry(mode: requestMode, appName: lastApp, rawText: raw, text: text,
                                             duration: duration, warning: warning.isEmpty ? nil : warning)
                    if let retentionError = store.add(entry, recording: audio) {
                        retryRecording = recording
                        self.error = retentionError
                    }
                    notice = warning
                    if allowInsertion, preferences.autoPaste, requestMode != .ask, let capturedTarget {
                        do {
                            try await TextInsertion.insert(text, into: capturedTarget)
                            guard generation == token else { return }
                            if notice.isEmpty { notice = L("notice.inserted", capturedTarget.applicationName) }
                        } catch {
                            notice = L("notice.readyToCopy", error.localizedDescription)
                            store.note(error.localizedDescription, on: entry.id)
                            Diagnostics.record("insert.failed", ["destination": capturedTarget.bundleID,
                                                                 "reason": Diagnostics.code(of: error)])
                            showMainWindow?()
                        }
                    } else {
                        if notice.isEmpty { notice = requestMode == .ask ? L("notice.answerReady") : L("notice.textReady") }
                        showMainWindow?()
                    }
                }
                guard generation == token else { return }
                target = nil; phase = .idle; hideVoicePanel?()
                releaseIdleModel()
                if !self.error.isEmpty { showMainWindow?() }
            } catch {
                guard generation == token else { try? FileManager.default.removeItem(at: audio); return }
                speechStream?.cancel(); speechStream = nil
                retryRecording = recording
                target = nil; phase = .idle; hideVoicePanel?()
                releaseIdleModel()
                self.error = error.localizedDescription
                Diagnostics.record("dictation.failed", ["reason": Diagnostics.code(of: error)])
                if let raw = (error as? WorkerFailure)?.rawText, !raw.isEmpty {
                    rawText = raw; resultText = raw
                    notice = L("notice.textFailed")
                }
                showMainWindow?()
            }
        }
    }

    var canRetry: Bool { retryRecording != nil && phase == .idle }
    func retryLast() {
        guard phase == .idle, let recording = retryRecording else { return }
        retryRecording = nil
        target = recording.target; mode = recording.mode; lastApp = recording.appName
        sessionPreferences = store.preferences
        sessionAPIKey = textAPIKey
        run(audio: recording.url, duration: recording.duration, allowInsertion: false)
    }

    func retry(_ entry: HistoryEntry) {
        guard phase == .idle, let audio = store.audioURL(for: entry) else { return }
        if entry.mode == .edit || entry.mode == .ask {
            error = L("error.retryNeedsRecording")
            return
        }
        do {
            let copy = FileManager.default.temporaryDirectory.appendingPathComponent("OmniTyper-\(UUID()).wav")
            try FileManager.default.copyItem(at: audio, to: copy)
            discardRetryRecording()
            target = nil; mode = entry.mode; lastApp = entry.appName; sessionPreferences = store.preferences
            sessionAPIKey = textAPIKey
            run(audio: copy, duration: entry.duration, allowInsertion: false)
        } catch { self.error = error.localizedDescription }
    }

    func cancel(releaseModel: Bool = false) {
        let keepModel = !releaseModel && store.preferences.retainsSpeechModel
        let interruptsWorker = phase == .processing || phase == .preparing || (phase == .starting && preloadTask == nil)
        generation = UUID(); task?.cancel(); task = nil
        speechStream?.cancel(); speechStream = nil; liveText = ""; liveStatus = ""
        recorder.cancel(); target = nil; phase = .idle; hideVoicePanel?()
        if !keepModel || interruptsWorker { stopModelWorker() }
        if keepModel && !worker.isRunning && preloadTask == nil { prepareModels() }
        notice = L("notice.cancelled")
    }

    func shutdown() {
        isShutDown = true
        preferencesSubscription?.cancel(); preferencesSubscription = nil
        cancel(releaseModel: true); shortcut.stop(); timer?.invalidate()
        textAPIKey = ""; sessionAPIKey = ""
        discardRetryRecording()
    }

    private func discardRetryRecording() {
        if let retryRecording { try? FileManager.default.removeItem(at: retryRecording.url) }
        retryRecording = nil
    }

    func copyResult() { TextInsertion.copy(resultText); notice = L("notice.copied") }
}
