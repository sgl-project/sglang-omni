import Foundation

@MainActor
public final class OllamaPolisher: TextPolishing {
    public let configuration: LocalModelConfiguration
    private let transport: LocalHTTPTransport
    public init(configuration: LocalModelConfiguration = .ollama, transport: LocalHTTPTransport) {
        self.configuration = configuration
        self.transport = transport
    }
    public func polish(text: String, personalBackground: String) async throws -> String {
        let bytes = try await transport.send(Self.request(text: text, personalBackground: personalBackground,
                                                         configuration: configuration), stage: "Ollama")
        let result = try Self.decodePolish(bytes)
        try PolishPolicy.validate(result, original: text, personalBackground: personalBackground)
        return result
    }
    public func warmup(personalBackground: String) async throws {
        let data = try await transport.send(Self.request(text: "预热。", personalBackground: personalBackground,
                                                        warmup: true, configuration: configuration), stage: "Ollama")
        struct Completion: Decodable { let done: Bool?; let done_reason: String?; let error: String? }
        let response = try JSONDecoder().decode(Completion.self, from: data)
        guard response.done == true, response.error == nil,
              response.done_reason == "stop" || response.done_reason == "length" else {
            throw DictationError("Ollama 预热未完成。")
        }
    }
    public func health() async -> String {
        var request = URLRequest(url: configuration.endpoint("api/tags"))
        request.timeoutInterval = 5
        do {
            let data = try await transport.send(request, stage: "Ollama")
            let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            let models = object?["models"] as? [[String: Any]] ?? []
            return models.contains { $0["name"] as? String == configuration.model }
                ? "可连接 · 模型已安装" : "可连接 · 缺少指定模型"
        } catch { return "不可用：\(error.localizedDescription)" }
    }
    public static func request(text: String, personalBackground: String = "", warmup: Bool = false,
                               configuration: LocalModelConfiguration = .ollama) -> URLRequest {
        let payload: [String: Any] = [
            "model": configuration.model,
            "messages": PolishPrompt.messages(text: text, personalBackground: personalBackground),
            "think": false, "stream": false, "keep_alive": "10m",
            "options": ["temperature": 0, "top_p": 1, "num_ctx": 8192, "num_predict": warmup ? 1 : 1024],
        ]
        var request = URLRequest(url: configuration.endpoint("api/chat"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        // The payload contains only JSON-compatible constants and strings.
        request.httpBody = try! JSONSerialization.data(withJSONObject: payload)
        return request
    }
    public static func decodePolish(_ data: Data) throws -> String {
        struct Response: Decodable {
            struct Message: Decodable { let content: String }
            let message: Message?
            let done: Bool?
            let done_reason: String?
            let error: String?
        }
        let response: Response
        do { response = try JSONDecoder().decode(Response.self, from: data) }
        catch { throw DictationError("Ollama 返回格式异常。") }
        guard response.error == nil, response.done == true, response.done_reason == "stop",
              let text = response.message?.content, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DictationError("Ollama 未正常完成整理，可能返回空文本或达到输出上限。")
        }
        return text
    }

    /// Optional server metrics, used by the opt-in live regression; no prompt logging.
    public struct Metrics {
        public let loadSeconds: Double?
        public let promptSeconds: Double?
        public let generationSeconds: Double?
        public let promptTokens: Int?
        public let cachedTokens: Int?
    }

    public static func decodeMetrics(_ data: Data) throws -> Metrics {
        struct Response: Decodable {
            let load_duration: Double?
            let prompt_eval_duration: Double?
            let eval_duration: Double?
            let prompt_eval_count: Int?
            let prompt_eval_cached_count: Int?
        }
        let response = try JSONDecoder().decode(Response.self, from: data)
        return Metrics(loadSeconds: response.load_duration.map { $0 / 1e9 },
                       promptSeconds: response.prompt_eval_duration.map { $0 / 1e9 },
                       generationSeconds: response.eval_duration.map { $0 / 1e9 },
                       promptTokens: response.prompt_eval_count, cachedTokens: response.prompt_eval_cached_count)
    }

}
