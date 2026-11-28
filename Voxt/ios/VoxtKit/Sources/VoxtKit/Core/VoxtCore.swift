import CVoxtCore
import Foundation

/// A failed core call: a code to tell failures apart by
/// (ios/voxtcore/call.go's Code* constants) and a message for the log.
public struct CoreError: Error, Equatable, Sendable, Decodable, LocalizedError {
    public var code: String
    public var message: String
    /// What a language model's setup lacks, for a config error
    /// (call.go's Field* constants); the message is then for the log.
    public var field: String?

    public init(code: String, message: String, field: String? = nil) {
        self.code = code
        self.message = message
        self.field = field
    }

    public var errorDescription: String? { message }

    public static let cancelled = "cancelled"
    public static let config = "config"
    public static let noText = "no_text"
    public static let empty = "empty"
    public static let untranslated = "untranslated"
    public static let languageModel = "language_model"
    public static let speech = "speech"
}

/// What the app asks of the Go core; VoxtCore is the real one, tests use
/// fakes.
public protocol CoreService: Sendable {
    func process(_ request: ProcessRequest) async throws -> ProcessResult
    func transcribe(_ request: SpeechRequest) async throws -> SpeechResult
    func testProvider(_ provider: CoreProvider) async throws -> TestResult
    func presets() async throws -> [Preset]
    func languages() async throws -> [Language]
}

/// The Go core over its C ABI. A call runs on a thread of its own (the C
/// call blocks) and is cancelled with the task that awaits it.
public struct VoxtCore: CoreService {
    public init() {}

    public func process(_ request: ProcessRequest) async throws -> ProcessResult {
        try await call("process", request)
    }

    public func transcribe(_ request: SpeechRequest) async throws -> SpeechResult {
        try await call("speech.transcribe", request)
    }

    public func testProvider(_ provider: CoreProvider) async throws -> TestResult {
        try await call("provider.test", provider)
    }

    public func presets() async throws -> [Preset] {
        try await call("presets", Empty())
    }

    public func languages() async throws -> [Language] {
        try await call("languages", Empty())
    }

    struct Empty: Encodable {}

    private struct Reply<R: Decodable>: Decodable {
        var result: R?
        var error: CoreError?
    }

    /// Runs method with params; the reply's result, or its error thrown.
    func call<P: Encodable, R: Decodable>(_ method: String, _ params: P) async throws -> R {
        let data = try JSONEncoder().encode(params)
        let id = UUID().uuidString
        let reply: Data = try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                Thread.detachNewThread {
                    continuation.resume(returning: Self.rawCall(id: id, method: method, params: data))
                }
            }
        } onCancel: {
            _ = id.withCString { VoxtCoreCancel($0) }
        }
        let decoded = try JSONDecoder().decode(Reply<R>.self, from: reply)
        if let error = decoded.error {
            throw error
        }
        guard let result = decoded.result else {
            throw CoreError(code: "reply", message: "the core's reply to \(method) has no result")
        }
        return result
    }

    /// The blocking C call.
    static func rawCall(id: String, method: String, params: Data) -> Data {
        let json = String(decoding: params, as: UTF8.self)
        guard let reply = VoxtCoreCall(id, method, json) else {
            return Data(#"{"error":{"code":"reply","message":"no reply"}}"#.utf8)
        }
        defer { VoxtCoreFree(reply) }
        return Data(String(cString: reply).utf8)
    }
}
