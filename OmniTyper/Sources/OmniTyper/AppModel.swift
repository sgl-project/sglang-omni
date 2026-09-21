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
    @Published private(set) var isCapturingShortcut = false
    private var isShutDown = false
    @Published var phase: Phase = .idle
    @Published var mode: VoiceMode = .dictate
    @Published var isVoicePanelVisible = false
    @Published var verbatimDictation = false
    @Published var consolePage: Page = .home
    @Published var textAPISettingsRequest: UUID?
    @Published var resultText = ""
    @Published var resultDraft = ""
    @Published var questionText = ""
    @Published var answerText = ""
    var answerHistoryID: UUID?
    var answerTarget: InsertionTarget?
    @Published var draftSelection = NSRange(location: 0, length: 0)
    @Published var unappliedResult = ""
    var draftOperation: DraftSelection?
    var focusDraft: (() -> Void)?
    @Published var isEditingEntireField = false
    @Published var isReviewingResult = false
    @Published var editorFolded = false
    var isReselectingMode = false
    var selectionAllowsModeChange = true
    var selectionPointerMoved = false
    @Published var isSelectingMode = false
    var modeSelectionTask: Task<Void, Never>?
    var selectionInitialMode: VoiceMode = .dictate
    var selectionPointerX: CGFloat = 0
    var selectionTrackX: CGFloat = 0
    var selectionPointerY: CGFloat = 0
    var selectionInitialExpansion: CGFloat = 0
    var selectionExpansionStartY: CGFloat = 0
    var selectionPanelBottom: CGFloat = 0
    var selectionBarOrigin = NSPoint.zero
    var selectionExpansion: CGFloat = 0
    static let presentationSelectionDistance: CGFloat = 100
    static let modeSelectionStep: CGFloat = 64
    @Published var rawText = ""
    @Published var liveText = ""
    @Published var liveStatus = ""
    @Published var notice = ""
    @Published var error = "" { didSet { needsEditSelection = false } }
    @Published private(set) var needsEditSelection = false
    @Published var lastApp = ""
    // ponytail: keys stay in memory for this session; use Keychain if persistence is needed.
    @Published var textAPIKey = ""
    @Published var textModels: [String] = []
    var sessionAPIKey = ""
    @Published var microphoneAllowed = AVCaptureDevice.authorizationStatus(for: .audio) == .authorized
    @Published var accessibilityAllowed = false
    // Note (Codex): A previous grant does not reveal why macOS now reports access unavailable.
    @Published private(set) var accessibilityNeedsRenewal = false
    var showMainWindow: (() -> Void)?
    var showVoicePanel: (() -> Void)?
    var hideVoicePanel: (() -> Void)?
    var target: InsertionTarget?
    var task: Task<Void, Never>?
    var speechStream: ASRStream?
    var generation = UUID()
    private var timer: Timer?
    private var preferencesSubscription: AnyCancellable?
    private let captureTarget: @MainActor () throws -> InsertionTarget
    let insertText: @MainActor (String, InsertionTarget, Bool) async throws -> Void
    var reviewTarget: InsertionTarget?
    var resultHistoryID: UUID?
    @Published var preloadTask: Task<[String: Any], Error>?
    var isPreloading: Bool { preloadTask != nil }
    struct FailedRecording {
        let url: URL
        let duration: Double
        let mode: VoiceMode
        let verbatim: Bool
        let target: InsertionTarget?
        let appName: String
        let draft: DraftSelection?
    }
    var retryRecording: FailedRecording?
    var sessionPreferences = Preferences()

    init(store: AppStore? = nil,
         captureTarget: @escaping @MainActor () throws -> InsertionTarget = { try TextInsertion.capture() },
         insertText: @escaping @MainActor (String, InsertionTarget, Bool) async throws -> Void = {
             try await TextInsertion.insert($0, into: $1, restoringFocus: $2)
         }) {
        let store = store ?? AppStore()
        self.store = store
        self.captureTarget = captureTarget
        self.insertText = insertText
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
            "build": Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "?",
            "bundle": Bundle.main.bundleIdentifier ?? "?",
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

    var isBusy: Bool { phase != .idle || isSelectingMode }
    var shortcutLabel: String {
        ShortcutCapture.label(keyCode: store.preferences.shortcutKeyCode,
                              modifiers: store.preferences.shortcutModifiers)
    }

    private func configureShortcut(_ preferences: Preferences) {
        guard !isCapturingShortcut, !isShutDown else { return }
        shortcut.start(keyCode: preferences.shortcutKeyCode, modifiers: preferences.shortcutModifiers,
                       hold: true,
                       onStart: { [weak self] in
                           guard let self, !self.isCapturingShortcut, !self.isShutDown else { return }
                           self.shortcutPressed()
                       },
                       onStop: { [weak self] in
                           guard let self, !self.isCapturingShortcut, !self.isShutDown else { return }
                           self.shortcutReleased()
                       },
                       onCancel: { [weak self] in
                           if self?.isReviewingResult == true && self?.isBusy == false { self?.finishReview() }
                           else if self?.isBusy == true || self?.isVoicePanelVisible == true { self?.cancel() }
                       })
    }

    func beginShortcutCapture() {
        isCapturingShortcut = true
        shortcut.stop()
        if isSelectingMode { cancel() }
    }

    func endShortcutCapture() {
        guard isCapturingShortcut else { return }
        isCapturingShortcut = false
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
            if accessibilityNeedsRenewal { accessibilityNeedsRenewal = false }
        } else {
            let stale = store.preferences.accessibilityWasTrusted == true
            if accessibilityNeedsRenewal != stale { accessibilityNeedsRenewal = stale }
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
        verbatimDictation = false
        start()
    }

    func selectMode(_ selection: VoiceMode) {
        guard canSelectMode else { return }
        mode = selection
        verbatimDictation = false
        if phase == .idle { sessionPreferences = store.preferences; sessionAPIKey = textAPIKey }
        if isVoicePanelVisible {
            if isReviewingResult { error = "" } else { _ = validateRecording() }
            presentVoicePanel()
        }
    }

    func useVerbatimDictation() {
        guard canSelectMode else { return }
        mode = .dictate
        verbatimDictation = true
        if canRetry { retryLast(mode: .dictate, verbatim: true) }
        else if phase == .idle { start() }
        else { _ = validateRecording(); presentVoicePanel() }
    }

    func presentVoicePanel() {
        isVoicePanelVisible = true
        showVoicePanel?()
        if isReviewingResult && showsEditor && !isBusy {
            DispatchQueue.main.async { [weak self] in
                guard let self, self.isReviewingResult, self.isVoicePanelVisible, self.showsEditor, !self.isBusy else { return }
                self.focusDraft?()
            }
        }
    }

    func dismissVoicePanel() {
        isVoicePanelVisible = false
        hideVoicePanel?()
    }

    func openTextAPISettings() {
        guard phase == .idle else { return }
        dismissVoicePanel()
        consolePage = .settings
        textAPISettingsRequest = UUID()
        showMainWindow?()
    }

    func setError(_ failure: Error) {
        error = failure.localizedDescription
        needsEditSelection = (failure as? Failure)?.code == "error.editNeedsSelection"
    }

    private func validateRecording() -> Bool {
        prepareEditSelection()
        do { _ = try payload(audio: nil); error = ""; return true }
        catch { setError(error); return false }
    }

    func start() {
        guard phase == .idle, !isSelectingMode else { return }
        if mode == .ask && !answerText.isEmpty {
            editorFolded = false
            presentVoicePanel()
            return
        }
        discardRetryRecording()
        error = ""; notice = ""; target = nil; liveText = ""; liveStatus = L("status.loadingModel")
        sessionPreferences = store.preferences
        sessionAPIKey = textAPIKey
        draftOperation = isReviewingResult ? DraftSelection(text: resultDraft, range: draftSelection) : nil
        unappliedResult = ""
        if !isReviewingResult { resultDraft = ""; draftSelection = NSRange(location: 0, length: 0) }
        if let draftOperation {
            do { _ = try draftOperation.selectedText() }
            catch { self.error = error.localizedDescription; presentVoicePanel(); return }
        } else {
            do {
                let captured = try captureTarget()
                if captured.bundleID != Bundle.main.bundleIdentifier { target = captured }
            } catch TextInsertionError.secureField {
                error = L("sys.secureField")
                presentVoicePanel()
                return
            } catch {
                // Note (Jiaxin Deng): Keep copy-only dictation available and distinguish missing permissions from opaque fields.
                notice = accessibilityAllowed ? L("notice.fieldNotAccessible")
                    : accessibilityNeedsRenewal ? L("home.permissions.axStale") : L("sys.axPermission")
                Diagnostics.record("capture.failed", ["reason": Diagnostics.code(of: error),
                                                      "accessibility": String(accessibilityAllowed)])
            }
        }
        lastApp = target?.applicationName ?? "OmniTyper"
        guard validateRecording() else { presentVoicePanel(); return }
        phase = .starting
        presentVoicePanel()
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
                phase = .recording; presentVoicePanel()
                if sessionPreferences.sounds { NSSound(named: "Tink")?.play() }
            } catch {
                guard generation == token else { return }
                speechStream?.cancel(); speechStream = nil
                phase = .idle; self.error = error.localizedDescription; refreshPermissions(); presentVoicePanel()
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
            phase = .idle; self.error = error.localizedDescription; presentVoicePanel()
            releaseIdleModel()
        }
    }

    func cancel(releaseModel: Bool = false) {
        let keepDraft = isReviewingResult && !releaseModel
        if isSelectingMode { mode = selectionInitialMode }
        isSelectingMode = false
        isReselectingMode = false
        modeSelectionTask?.cancel(); modeSelectionTask = nil
        let keepModel = !releaseModel && store.preferences.retainsSpeechModel
        let interruptsWorker = worker.hasPendingRequest && preloadTask == nil
        generation = UUID(); task?.cancel(); task = nil
        speechStream?.cancel(); speechStream = nil; liveText = ""; liveStatus = ""
        recorder.cancel(); target = nil; phase = .idle; draftOperation = nil
        if !keepDraft {
            dismissVoicePanel()
            isReviewingResult = false; resultDraft = ""; reviewTarget = nil; resultHistoryID = nil
            questionText = ""; answerText = ""; answerTarget = nil; answerHistoryID = nil
            unappliedResult = ""
        }
        isEditingEntireField = false
        if !keepModel || interruptsWorker { stopModelWorker() }
        if keepModel && !worker.isRunning && preloadTask == nil { prepareModels() }
        verbatimDictation = false
        notice = L("notice.cancelled")
        if keepDraft { presentVoicePanel() }
    }

    func shutdown() {
        isShutDown = true
        preferencesSubscription?.cancel(); preferencesSubscription = nil
        cancel(releaseModel: true); shortcut.stop(); timer?.invalidate()
        textAPIKey = ""; sessionAPIKey = ""
        discardRetryRecording()
    }

    func copyResult() { TextInsertion.copy(resultText); notice = L("notice.copied") }
}
