import Foundation

@MainActor
public final class OmniASRClient: AudioTranscribing {
    public let configuration: LocalModelConfiguration
    private let transport: LocalHTTPTransport
    public init(configuration: LocalModelConfiguration = .omni, transport: LocalHTTPTransport) {
        self.configuration = configuration
        self.transport = transport
    }
    public func transcribe(wav: Data) async throws -> String {
        let bytes = try await transport.send(Self.request(wav: wav, configuration: configuration), stage: "Omni ASR")
        struct Response: Decodable { let text: String }
        do { return try JSONDecoder().decode(Response.self, from: bytes).text }
        catch { throw DictationError("Omni ASR 返回格式异常，缺少 text 字段。") }
    }
    public func health() async -> String {
        var request = URLRequest(url: configuration.endpoint("health"))
        request.timeoutInterval = 5
        do {
            let data = try await transport.send(request, stage: "Omni ASR")
            let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            guard object?["status"] as? String == "healthy" else { return "服务尚未就绪" }
            var modelsRequest = URLRequest(url: configuration.endpoint("v1/models"))
            modelsRequest.timeoutInterval = 5
            let modelsData = try await transport.send(modelsRequest, stage: "Omni ASR")
            let modelsObject = try JSONSerialization.jsonObject(with: modelsData) as? [String: Any]
            guard let models = modelsObject?["data"] as? [[String: Any]] else {
                throw DictationError("Omni ASR 模型列表格式异常。")
            }
            return models.contains { $0["id"] as? String == configuration.model }
                ? "可连接" : "服务可连接，但未提供模型：\(configuration.model)"
        } catch { return "不可用：\(error.localizedDescription)" }
    }
    public static func request(wav: Data, configuration: LocalModelConfiguration = .omni) -> URLRequest {
        let boundary = "OmniDictation-\(UUID().uuidString)"
        var body = Data()
        func append(_ text: String) { body.append(Data(text.utf8)) }
        for (name, value) in [("model", configuration.model), ("response_format", "json")] {
            append("--\(boundary)\r\nContent-Disposition: form-data; name=\"\(name)\"\r\n\r\n\(value)\r\n")
        }
        append("--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"recording.wav\"\r\nContent-Type: audio/wav\r\n\r\n")
        body.append(wav)
        append("\r\n--\(boundary)--\r\n")
        var request = URLRequest(url: configuration.endpoint("v1/audio/transcriptions"))
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.httpBody = body
        return request
    }

}
