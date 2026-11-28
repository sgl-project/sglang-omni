import Foundation

/// Why a session ended without a result.
public enum SessionProblem: Equatable, Sendable {
    case microphoneDenied
    case speechNotAuthorized
    case noSpeech
    /// The device cannot recognise this language offline (its locale).
    case notOffline(String)
    case speech(String)
    case languageModel(String)
    /// The language model's settings to check (the core's message).
    case config(String)
    /// What the language model's setup lacks (CoreError.field).
    case missing(String)
    /// The speech service's settings to check.
    case speechConfig(String)
    case notTranslated
    case emptyAnswer
    case noSource
    case other(String)
}

/// A session's state, as the voice screen shows it.
public enum SessionState: Equatable, Sendable {
    case idle
    case recording
    case recognizing
    case composing
    case done
    case failed(SessionProblem)
}

/// What a session needs from the app: the core, the microphone, the
/// speech engine the settings choose, the keys and the history.
public struct SessionEnvironment: Sendable {
    public var core: CoreService
    public var recorder: AudioRecorder
    public var keys: KeyStore
    public var history: HistoryStore
    public var speech: @Sendable (AppSettings, KeyStore) -> SpeechEngine
    public var microphoneAllowed: @Sendable () async -> Bool
    /// The shortest recording that is speech.
    public var minimumSeconds: Double = 0.3

    public init(core: CoreService, recorder: AudioRecorder, keys: KeyStore, history: HistoryStore,
                speech: @escaping @Sendable (AppSettings, KeyStore) -> SpeechEngine,
                microphoneAllowed: @escaping @Sendable () async -> Bool) {
        self.core = core
        self.recorder = recorder
        self.keys = keys
        self.history = history
        self.speech = speech
        self.microphoneAllowed = microphoneAllowed
    }

    /// The engine the settings choose: Apple's on the device, or the
    /// remote service with its key.
    public static func speechEngine(core: CoreService, scratch: URL) -> @Sendable (AppSettings, KeyStore) -> SpeechEngine {
        { settings, keys in
            switch settings.speech.engine {
            case .onDevice:
                return AppleSpeechEngine(requireOffline: settings.speech.requireOffline, scratch: scratch)
            case .remote:
                return RemoteSpeechEngine(core: core, endpoint: settings.speech.endpoint, model: settings.speech.model,
                                          key: keys.key(KeyAccount.speech), scratch: scratch)
            }
        }
    }
}

/// One voice session at a time, the iOS counterpart of internal/session:
/// record → recognise → the feature's language-model step (the Go core)
/// → the result, kept in the history. A Rewrite answer can be followed
/// up: the next instruction carries the turns before it.
@MainActor
public final class VoiceSession: ObservableObject {
    @Published public private(set) var state = SessionState.idle
    @Published public private(set) var level: Float = 0
    @Published public private(set) var speech = ""
    @Published public private(set) var result = ""
    @Published public private(set) var title = ""
    @Published public private(set) var warning = ""
    @Published public private(set) var turns: [CoreTurn] = []
    @Published public var feature: Feature
    /// The text Translate or Rewrite works on when it is typed or pasted.
    @Published public var source = ""

    /// The settings a session starts with (read when it starts its steps).
    public var settings: () -> AppSettings
    private let env: SessionEnvironment
    private var work: Task<Void, Never>?
    private var started = Date()
    /// Set from a tap on record until the recording runs: the microphone's
    /// prompt and opening the audio take a while.
    private var starting: UUID?

    public init(environment: SessionEnvironment, feature: Feature = .transcribe, settings: @escaping () -> AppSettings) {
        env = environment
        self.feature = feature
        self.settings = settings
    }

    public var isBusy: Bool {
        switch state {
        case .recording, .recognizing, .composing: return true
        default: return false
        }
    }

    /// Starts recording, after the microphone permission.
    public func start() async {
        guard !isBusy, starting == nil else { return }
        let attempt = UUID()
        starting = attempt
        defer {
            if starting == attempt {
                starting = nil
            }
        }
        reset(keepTurns: feature == .rewrite)
        guard await env.microphoneAllowed() else {
            if starting == attempt {
                state = .failed(.microphoneDenied)
            }
            return
        }
        do {
            try await env.recorder.start { [weak self] value in
                Task { @MainActor in self?.level = value }
            }
            guard starting == attempt else {
                env.recorder.cancel() // cancelled while it opened
                return
            }
            started = Date()
            state = .recording
        } catch {
            guard starting == attempt else { return }
            state = .failed(error as? RecorderError == .microphoneDenied ? .microphoneDenied : .other(error.localizedDescription))
        }
    }

    /// Stops recording and runs the rest of the session.
    public func stop() {
        guard state == .recording else { return }
        let recording = env.recorder.stop()
        level = 0
        guard recording.duration >= env.minimumSeconds else {
            state = .failed(.noSpeech)
            return
        }
        let snapshot = settings()
        state = .recognizing
        work = Task { [weak self] in
            await self?.run(recording: recording, settings: snapshot)
        }
    }

    /// Translates the typed text, without speech.
    public func translateTyped() {
        guard !isBusy else { return }
        let text = source.trimmingCharacters(in: .whitespacesAndNewlines)
        reset(keepTurns: false)
        guard !text.isEmpty else {
            state = .failed(.noSource)
            return
        }
        let snapshot = settings()
        started = Date()
        state = .composing
        work = Task { [weak self] in
            await self?.compose(speech: "", settings: snapshot)
        }
    }

    /// Ends the session where it is; nothing is kept.
    public func cancel() {
        starting = nil
        if state == .recording {
            env.recorder.cancel()
        }
        work?.cancel()
        work = nil
        level = 0
        state = .idle
    }

    /// Forgets a Rewrite conversation.
    public func clearConversation() {
        turns = []
        if !isBusy {
            reset(keepTurns: false)
        }
    }

    private func reset(keepTurns: Bool) {
        speech = ""
        result = ""
        title = ""
        warning = ""
        if !keepTurns {
            turns = []
        }
        state = .idle
    }

    private func run(recording: Recording, settings: AppSettings) async {
        // A Rewrite instruction is in the main language; other speech is
        // too, unless the user speaks other languages (then detected).
        let language = feature == .rewrite || settings.general.otherLanguages.isEmpty ? settings.mainLanguage() : nil
        let engine = env.speech(settings, env.keys)
        let text: String
        do {
            text = try await engine.transcribe(SpeechInput(recording: recording, language: language, terms: settings.activeTerms))
        } catch {
            finish(problem: Self.problem(speech: error))
            return
        }
        guard !Task.isCancelled else { return }
        speech = text
        state = .composing
        await compose(speech: text, settings: settings)
    }

    private func compose(speech text: String, settings: AppSettings) async {
        let request = Self.request(feature: feature, speech: text, source: source, turns: turns, settings: settings, keys: env.keys)
        do {
            let answer = try await env.core.process(request)
            guard !Task.isCancelled else { return }
            result = answer.text
            title = answer.title ?? ""
            warning = answer.degraded == true ? (answer.warning ?? "") : ""
            if feature == .rewrite {
                turns.append(CoreTurn(instruction: text, source: request.text.isEmpty ? nil : request.text, title: answer.title, answer: answer.text))
                source = ""
            }
            state = .done
            keep(speech: text, request: request)
        } catch {
            finish(problem: Self.problem(core: error))
        }
    }

    /// Ends in problem, unless the session was cancelled meanwhile.
    private func finish(problem: SessionProblem) {
        guard !Task.isCancelled else { return }
        state = .failed(problem)
    }

    private func keep(speech text: String, request: ProcessRequest) {
        let entry = HistoryEntry(feature: feature, speech: text, source: feature == .transcribe ? "" : (request.typed == true || feature == .rewrite ? request.text : ""),
                                 result: result, title: title, warning: warning, seconds: Date().timeIntervalSince(started))
        _ = try? env.history.add(entry) // the result is on screen even when the history cannot be written
    }

    /// The core's request for a feature's step.
    nonisolated static func request(feature: Feature, speech: String, source: String, turns: [CoreTurn],
                                    settings: AppSettings, keys: KeyStore) -> ProcessRequest {
        let provider = settings.languageModel.provider(keys: keys)
        var request = ProcessRequest(task: feature.rawValue, text: speech, mainLanguage: settings.mainLanguage(),
                                     otherLanguages: settings.general.otherLanguages, feature: CoreFeature(cleanup: true),
                                     notes: settings.activeNotes, terms: settings.activeTerms, provider: provider)
        let typed = source.trimmingCharacters(in: .whitespacesAndNewlines)
        switch feature {
        case .transcribe:
            let f = settings.transcribe
            request.feature = CoreFeature(cleanup: f.cleanup, structure: f.structure, prompt: f.prompt)
            request.snippets = settings.snippets
        case .translate:
            let f = settings.translate
            request.feature = CoreFeature(cleanup: f.cleanup, prompt: f.prompt)
            request.targetLanguage = f.targetLanguage
            if speech.isEmpty {
                (request.text, request.typed) = (typed, true)
            }
        case .rewrite:
            let f = settings.rewrite
            request.feature = CoreFeature(cleanup: f.cleanup, prompt: f.prompt)
            (request.instruction, request.text) = (speech, typed)
            request.previous = turns.isEmpty ? nil : turns
        }
        return request
    }

    nonisolated static func problem(speech error: Error) -> SessionProblem {
        switch error {
        case SpeechError.notAuthorized: return .speechNotAuthorized
        case SpeechError.noSpeech: return .noSpeech
        case SpeechError.notOffline(let locale): return .notOffline(locale)
        case SpeechError.unavailable(let locale): return .speech(locale)
        case let e as CoreError where e.code == CoreError.config: return .speechConfig(e.message)
        case let e as CoreError: return .speech(e.message)
        default: return .speech(error.localizedDescription)
        }
    }

    nonisolated static func problem(core error: Error) -> SessionProblem {
        guard let e = error as? CoreError else {
            return .other(error.localizedDescription)
        }
        switch e.code {
        case CoreError.config: return e.field.map(SessionProblem.missing) ?? .config(e.message)
        case CoreError.untranslated: return .notTranslated
        case CoreError.empty: return .emptyAnswer
        case CoreError.noText: return .noSpeech
        case CoreError.languageModel: return .languageModel(e.message)
        default: return .other(e.message)
        }
    }
}
