import Foundation

/// Composes independent ASR and polishing clients. A session captures a configuration snapshot.
@MainActor
public final class LocalSpeechService: SpeechServing {
    public private(set) var configuration: ServiceConfiguration
    private let transport: LocalHTTPTransport

    public init(configuration: ServiceConfiguration = ServiceConfiguration(), transport: LocalHTTPTransport? = nil) {
        self.configuration = configuration
        self.transport = transport ?? LocalHTTPTransport()
    }

    public func configure(_ configuration: ServiceConfiguration) { self.configuration = configuration }

    public func snapshot() -> SpeechServing {
        LocalSpeechService(configuration: configuration, transport: transport)
    }

    public func transcribe(wav: Data) async throws -> String {
        try await OmniASRClient(configuration: configuration.asr, transport: transport).transcribe(wav: wav)
    }

    public func polish(text: String) async throws -> String {
        try await polish(text: text, personalBackground: "")
    }

    public func polish(text: String, personalBackground: String) async throws -> String {
        try await OllamaPolisher(configuration: configuration.polish, transport: transport)
            .polish(text: text, personalBackground: personalBackground)
    }

    public func warmup(personalBackground: String, model: LocalModelConfiguration? = nil) async throws {
        try await OllamaPolisher(configuration: model ?? configuration.polish, transport: transport)
            .warmup(personalBackground: personalBackground)
    }

    public func health() async -> (asr: String, ollama: String) {
        let config = configuration
        async let asr = OmniASRClient(configuration: config.asr, transport: transport).health()
        async let ollama = OllamaPolisher(configuration: config.polish, transport: transport).health()
        return await (asr, ollama)
    }

    // Retain the original example API; defaults and implementation live in their respective modules.
    public static var asrModel: String { LocalModelConfiguration.omni.model }
    public static var polishModel: String { LocalModelConfiguration.ollama.model }
    public static var asrEndpoint: String { LocalModelConfiguration.omni.endpoint("v1/audio/transcriptions").absoluteString }
    public static var polishEndpoint: String { LocalModelConfiguration.ollama.endpoint("api/chat").absoluteString }
    public static func transcriptionRequest(wav: Data) -> URLRequest { OmniASRClient.request(wav: wav) }
    public static func polishRequest(text: String, personalBackground: String = "", warmup: Bool = false) -> URLRequest {
        OllamaPolisher.request(text: text, personalBackground: personalBackground, warmup: warmup)
    }
    public static func decodePolish(_ data: Data) throws -> String { try OllamaPolisher.decodePolish(data) }
    public static func validatePolish(_ result: String, original: String, personalBackground: String = "") throws {
        try PolishPolicy.validate(result, original: original, personalBackground: personalBackground)
    }
    public typealias Metrics = OllamaPolisher.Metrics
    public static func decodeMetrics(_ data: Data) throws -> Metrics { try OllamaPolisher.decodeMetrics(data) }
}
