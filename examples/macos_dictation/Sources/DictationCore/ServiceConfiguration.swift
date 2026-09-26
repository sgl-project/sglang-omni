import Foundation

/// A model already served by a local backend. Selecting it never downloads or loads weights.
public struct LocalModelConfiguration: Codable, Equatable, Sendable {
    public let baseURL: URL
    public let model: String

    public init(baseURL: String, model: String) throws {
        let name = model.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty, name.count <= 256,
              !name.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) }) else {
            throw DictationError("请输入有效的模型名称。")
        }
        guard var parts = URLComponents(string: baseURL.trimmingCharacters(in: .whitespacesAndNewlines)),
              ["http", "https"].contains(parts.scheme?.lowercased() ?? ""),
              ["127.0.0.1", "localhost", "[::1]", "::1"].contains(parts.host?.lowercased() ?? ""),
              parts.user == nil, parts.password == nil, parts.query == nil, parts.fragment == nil,
              parts.path.isEmpty || parts.path == "/",
              parts.port.map({ (1...65535).contains($0) }) ?? true else {
            throw DictationError("服务地址须为本机 HTTP 地址，例如 http://127.0.0.1:8000，不包含接口路径。")
        }
        parts.path = ""
        parts.scheme = parts.scheme?.lowercased()
        parts.host = parts.host?.lowercased()
        guard let url = parts.url else { throw DictationError("服务地址无效。") }
        self.baseURL = url
        self.model = name
    }

    public func endpoint(_ path: String) -> URL { baseURL.appendingPathComponent(path) }

    private enum CodingKeys: String, CodingKey { case baseURL, model }
    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(baseURL: values.decode(String.self, forKey: .baseURL),
                      model: values.decode(String.self, forKey: .model))
    }
    public func encode(to encoder: Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encode(baseURL.absoluteString, forKey: .baseURL)
        try values.encode(model, forKey: .model)
    }

    public static let omni = try! LocalModelConfiguration(baseURL: "http://127.0.0.1:8000", model: "Qwen/Qwen3-ASR-0.6B")
    public static let ollama = try! LocalModelConfiguration(baseURL: "http://127.0.0.1:11434", model: "openbmb/minicpm5-2b:q4_K_M")
}

public struct ServiceConfiguration: Codable, Equatable, Sendable {
    public var asr: LocalModelConfiguration
    public var polish: LocalModelConfiguration
    public init(asr: LocalModelConfiguration = .omni, polish: LocalModelConfiguration = .ollama) {
        self.asr = asr
        self.polish = polish
    }
}
