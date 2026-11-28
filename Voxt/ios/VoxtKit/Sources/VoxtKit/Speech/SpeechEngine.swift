import Foundation
import Speech

/// What a speech engine is asked.
public struct SpeechInput: Sendable {
    public var recording: Recording
    /// The language spoken (a lang code), nil to let the engine detect it.
    public var language: String?
    /// The dictionary's terms, as hints.
    public var terms: [Term]

    public init(recording: Recording, language: String?, terms: [Term]) {
        self.recording = recording
        self.language = language
        self.terms = terms
    }
}

/// Recognises a recording.
public protocol SpeechEngine: Sendable {
    func transcribe(_ input: SpeechInput) async throws -> String
}

public enum SpeechError: Error, Equatable {
    case notAuthorized
    case unavailable(String)
    case notOffline(String)
    case noSpeech
}

/// Apple's speech recognition. With requireOffline it runs on the device
/// only, and refuses a language the device cannot recognise offline,
/// rather than sending the audio to Apple.
public struct AppleSpeechEngine: SpeechEngine {
    public var requireOffline: Bool
    public var scratch: URL

    public init(requireOffline: Bool, scratch: URL = FileManager.default.temporaryDirectory) {
        self.requireOffline = requireOffline
        self.scratch = scratch
    }

    public static func requestAuthorization() async -> Bool {
        await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { continuation.resume(returning: $0 == .authorized) }
        }
    }

    public func transcribe(_ input: SpeechInput) async throws -> String {
        // The first session asks; a refusal is the user's to undo in Settings.
        var authorized = SFSpeechRecognizer.authorizationStatus() == .authorized
        if SFSpeechRecognizer.authorizationStatus() == .notDetermined {
            authorized = await Self.requestAuthorization()
        }
        guard authorized else {
            throw SpeechError.notAuthorized
        }
        let locale = Locale(identifier: Self.localeIdentifier(input.language))
        guard let recognizer = SFSpeechRecognizer(locale: locale), recognizer.isAvailable else {
            throw SpeechError.unavailable(locale.identifier)
        }
        if requireOffline && !recognizer.supportsOnDeviceRecognition {
            throw SpeechError.notOffline(locale.identifier)
        }
        let url = try input.recording.write(to: scratch)
        defer { try? FileManager.default.removeItem(at: url) }
        let request = SFSpeechURLRecognitionRequest(url: url)
        request.requiresOnDeviceRecognition = requireOffline
        request.shouldReportPartialResults = false
        request.addsPunctuation = true
        request.contextualStrings = Array(input.terms.map(\.term).prefix(100))
        let box = TaskBox()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let once = Once()
                box.task = recognizer.recognitionTask(with: request) { result, error in
                    if let result, result.isFinal {
                        let text = result.bestTranscription.formattedString
                        once.run { text.isEmpty ? continuation.resume(throwing: SpeechError.noSpeech) : continuation.resume(returning: text) }
                    } else if let error {
                        // Apple reports a recording without speech as an error.
                        let nsError = error as NSError
                        let noSpeech = nsError.domain == "kAFAssistantErrorDomain" && [203, 1110].contains(nsError.code)
                        once.run { continuation.resume(throwing: noSpeech ? SpeechError.noSpeech : error) }
                    }
                }
            }
        } onCancel: {
            box.task?.cancel()
        }
    }

    /// The recognizer's locale for a lang code: the region the device
    /// uses for it when there is one ("en" → "en-US" on a US device).
    static func localeIdentifier(_ code: String?, current: Locale = .current) -> String {
        guard let code, !code.isEmpty else {
            return current.identifier
        }
        switch code {
        case "zh": return "zh-CN"
        case "yue": return "zh-HK"
        default:
            if current.language.languageCode?.identifier == code, let region = current.region?.identifier {
                return "\(code)-\(region)"
            }
            return code
        }
    }
}

/// An OpenAI-compatible speech service, through the Go core
/// (internal/asr/openai, as on macOS).
public struct RemoteSpeechEngine: SpeechEngine {
    public var core: CoreService
    public var endpoint: String
    public var model: String
    public var key: String?
    public var scratch: URL

    public init(core: CoreService, endpoint: String, model: String, key: String?, scratch: URL = FileManager.default.temporaryDirectory) {
        self.core = core
        self.endpoint = endpoint
        self.model = model
        self.key = key
        self.scratch = scratch
    }

    public func transcribe(_ input: SpeechInput) async throws -> String {
        let url = try input.recording.write(to: scratch)
        defer { try? FileManager.default.removeItem(at: url) }
        let result = try await core.transcribe(SpeechRequest(
            path: url.path, endpoint: endpoint.isEmpty ? nil : endpoint, model: model.isEmpty ? nil : model,
            key: key, language: input.language, prompt: nil, terms: input.terms
        ))
        let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else {
            throw SpeechError.noSpeech
        }
        return text
    }
}

private final class TaskBox: @unchecked Sendable {
    var task: SFSpeechRecognitionTask?
}

/// Runs its block once, from any thread.
final class Once: @unchecked Sendable {
    private let lock = NSLock()
    private var done = false

    func run(_ block: () -> Void) {
        lock.lock()
        let first = !done
        done = true
        lock.unlock()
        if first {
            block()
        }
    }
}
