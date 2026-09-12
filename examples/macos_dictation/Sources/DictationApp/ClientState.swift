#if canImport(DictationCore)
import DictationCore
#endif
import AppKit
import Combine

@MainActor
final class ClientState: ObservableObject {
    let service: LocalSpeechService
    let preferences: ClientPreferences
    let registrar: CarbonShortcutRegistrar
    let shortcuts: RecordingShortcutController
    let warmup: PolishWarmup
    let session: DictationSession
    let insertion = DictationInsertion()
    let feedback: DictationFeedback
    @Published var asrStatus = "尚未检测"
    @Published var ollamaStatus = "尚未检测"
    @Published var checking = false
    @Published private(set) var configuration: ServiceConfiguration
    @Published var asrBaseURLDraft: String
    @Published var asrModelDraft: String
    @Published var polishBaseURLDraft: String
    @Published var polishModelDraft: String
    @Published private(set) var configurationNotice = ""
    @Published var personalBackgroundDraft = ""
    @Published private(set) var savedBackground = ""
    @Published private(set) var personalBackgroundEnabled = false
    @Published private(set) var backgroundNotice = ""
    @Published var accessibilityGranted = DictationAccessibility.isTrusted
    @Published var pasteToCurrentCursor = true
    @Published var compatibilityPasteEnabled = false
    var openResults: (() -> Void)?
    var openAudio: (() -> Void)?
    private var observers: Set<AnyCancellable> = []
    private var healthTask: Task<Void, Never>?
    private var healthGeneration = UUID()
    private let makeTextTarget: (Bool, Bool) throws -> DictationTextTarget
    var hotkeyNotice: String { shortcuts.notice }
    var shortcutLabel: String { shortcuts.shortcut.display }
    var hasUnsavedBackground: Bool { personalBackgroundDraft != savedBackground }
    var hasUnsavedConfiguration: Bool {
        asrBaseURLDraft != configuration.asr.baseURL.absoluteString || asrModelDraft != configuration.asr.model
            || polishBaseURLDraft != configuration.polish.baseURL.absoluteString || polishModelDraft != configuration.polish.model
    }

    init(preferences: ClientPreferences? = nil, service: LocalSpeechService? = nil, recorder: AudioRecording? = nil,
         makeTextTarget: ((_ pasteToCurrentCursor: Bool, _ allowCompatibilityPaste: Bool) throws -> DictationTextTarget)? = nil) {
        let preferences = preferences ?? ClientPreferences()
        let service = service ?? LocalSpeechService(configuration: preferences.serviceConfiguration)
        configuration = service.configuration
        asrBaseURLDraft = service.configuration.asr.baseURL.absoluteString
        asrModelDraft = service.configuration.asr.model
        polishBaseURLDraft = service.configuration.polish.baseURL.absoluteString
        polishModelDraft = service.configuration.polish.model
        let registrar = CarbonShortcutRegistrar()
        self.service = service
        self.preferences = preferences
        self.registrar = registrar
        self.makeTextTarget = makeTextTarget ?? Self.captureTextTarget
        shortcuts = RecordingShortcutController(shortcut: preferences.recordingShortcut, registrar: registrar) {
            preferences.recordingShortcut = $0
        }
        warmup = PolishWarmup.configured {
            try await service.warmup(personalBackground: $0.personalBackground, model: $0.model)
        }
        session = DictationSession(recorder: recorder ?? MicrophoneRecorder(), service: service)
        feedback = DictationFeedback(session: session, insertion: insertion,
                                     reduceMotion: { NSWorkspace.shared.accessibilityDisplayShouldReduceMotion })
        session.onResult = { [weak self] text in self?.insertion.complete(text) }
        savedBackground = preferences.personalBackground
        personalBackgroundDraft = savedBackground
        personalBackgroundEnabled = preferences.personalBackgroundEnabled
        session.personalBackground = personalBackgroundEnabled ? savedBackground : ""
        session.polishEnabled = preferences.polishEnabled
        shortcuts.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }.store(in: &observers)
        warmup.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }.store(in: &observers)
        session.$polishEnabled.dropFirst().sink { [weak self] enabled in
            guard let self else { return }
            self.preferences.polishEnabled = enabled
            self.updateWarmup(enabled: enabled)
        }.store(in: &observers)
        session.$phase.combineLatest(insertion.$isDelivering).sink { [weak self] phase, delivering in
            if phase == .polishing { self?.warmup.foregroundWillPolish() }
            self?.updateWarmup(busy: [.authorizing, .recording, .recognizing, .polishing].contains(phase) || delivering)
        }.store(in: &observers)
    }

    func beginShortcutCapture() {
        guard !session.isBusy, !insertion.isDelivering else { return }
        shortcuts.beginCapture()
    }

    func restoreShortcut() {
        guard !session.isBusy, !insertion.isDelivering else { return }
        shortcuts.restoreDefault()
    }

    func setPersonalBackgroundEnabled(_ enabled: Bool) {
        personalBackgroundEnabled = enabled
        preferences.personalBackgroundEnabled = enabled
        session.personalBackground = enabled ? savedBackground : ""
        updateWarmup()
    }

    func savePersonalBackground() {
        guard personalBackgroundDraft.count <= ClientPreferences.maximumBackgroundLength else {
            backgroundNotice = "背景最多 \(ClientPreferences.maximumBackgroundLength) 字，请精简后保存。"
            return
        }
        savedBackground = personalBackgroundDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        personalBackgroundDraft = savedBackground
        preferences.personalBackground = savedBackground
        session.personalBackground = personalBackgroundEnabled ? savedBackground : ""
        backgroundNotice = "已保存；修改只影响下一轮。"
        updateWarmup()
    }

    func clearPersonalBackground() {
        personalBackgroundDraft = ""
        savedBackground = ""
        preferences.personalBackground = ""
        setPersonalBackgroundEnabled(false)
        backgroundNotice = "已清空本机保存的个人背景。"
    }

    private func updateWarmup(enabled: Bool? = nil, busy: Bool? = nil) {
        warmup.update(enabled: enabled ?? session.polishEnabled,
                      personalBackground: session.personalBackground,
                      busy: busy ?? (session.isBusy || insertion.isDelivering), model: configuration.polish)
    }

    func saveServiceConfiguration() {
        do {
            let next = ServiceConfiguration(
                asr: try LocalModelConfiguration(baseURL: asrBaseURLDraft, model: asrModelDraft),
                polish: try LocalModelConfiguration(baseURL: polishBaseURLDraft, model: polishModelDraft))
            configuration = next
            preferences.serviceConfiguration = next
            service.configure(next)
            asrBaseURLDraft = next.asr.baseURL.absoluteString
            asrModelDraft = next.asr.model
            polishBaseURLDraft = next.polish.baseURL.absoluteString
            polishModelDraft = next.polish.model
            configurationNotice = "已保存，下一轮生效。"
            asrStatus = "尚未检测"
            ollamaStatus = "尚未检测"
            updateWarmup()
            checkServices(retryWarmup: false)
        } catch { configurationNotice = error.localizedDescription }
    }

    func toggleRecording() {
        guard !shortcuts.isCapturing else { return }
        if session.phase == .recording { session.toggleRecording(); return }
        guard !session.isBusy, !insertion.isDelivering else { return }
        insertion.cancel()
        accessibilityGranted = DictationAccessibility.isTrusted
        do {
            insertion.prepare(try makeTextTarget(pasteToCurrentCursor, compatibilityPasteEnabled))
        }
        catch { insertion.abandon(error.localizedDescription) }
        session.toggleRecording()
    }

    /// Capture once at recording start. Tests replace the factory without touching real editor state.
    private static func captureTextTarget(pasteToCurrentCursor: Bool, allowCompatibilityPaste: Bool) throws -> DictationTextTarget {
        if pasteToCurrentCursor {
            let target = CurrentCursorTarget()
            try target.validate()
            return target
        }
        return try AccessibilityTextTarget.capture(allowCompatibilityPaste: allowCompatibilityPaste)
    }

    func cancel() {
        insertion.cancel()
        session.cancel()
    }

    func closeFeedback() {
        if session.isBusy || insertion.isDelivering { cancel() }
        else { feedback.dismiss() }
    }

    func requestAccessibility() {
        DictationAccessibility.requestAccess()
        accessibilityGranted = DictationAccessibility.isTrusted
    }

    func checkServices(retryWarmup: Bool = true) {
        healthTask?.cancel()
        let id = UUID()
        healthGeneration = id
        checking = true
        healthTask = Task {
            let status = await service.health()
            guard !Task.isCancelled, healthGeneration == id else { return }
            asrStatus = status.asr
            ollamaStatus = status.ollama
            checking = false
            if retryWarmup {
                warmup.retry()
                updateWarmup()
            }
            healthTask = nil
        }
    }
}
