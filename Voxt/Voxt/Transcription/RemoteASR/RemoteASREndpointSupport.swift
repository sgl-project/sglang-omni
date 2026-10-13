import Foundation

enum RemoteASREndpointSupport {
    static func audioMIMEType(for fileURL: URL) -> String {
        switch fileURL.pathExtension.lowercased() {
        case "mp3":
            return "audio/mpeg"
        case "m4a":
            return "audio/mp4"
        case "ogg":
            return "audio/ogg"
        default:
            return "audio/wav"
        }
    }

    static func resolvedAliyunFunRealtimeEndpoint(_ endpoint: String) -> String {
        let trimmed = endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
        }
        if var components = URLComponents(string: trimmed) {
            let normalizedPath = components.path.lowercased()
            if normalizedPath.hasSuffix("/api-ws/v1/inference") {
                return trimmed
            }
            if normalizedPath.hasSuffix("/api-ws/v1/realtime") {
                components.path = components.path.replacingOccurrences(of: "/api-ws/v1/realtime", with: "/api-ws/v1/inference")
                components.queryItems = nil
                return components.string ?? trimmed
            }
            if normalizedPath.hasSuffix("/models") {
                return replacingPathSuffix(in: trimmed, oldSuffix: "/models", newSuffix: "/api-ws/v1/inference")
            }
            if normalizedPath.hasSuffix("/chat/completions") {
                return replacingPathSuffix(in: trimmed, oldSuffix: "/chat/completions", newSuffix: "/api-ws/v1/inference")
            }
            if normalizedPath.hasSuffix("/v1") {
                return appendingPath(trimmed, suffix: "/inference")
            }
        }
        return trimmed
    }

    static func isAliyunFunRealtimeModel(_ model: String) -> Bool {
        let normalized = model.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return normalized.hasPrefix("qwen-audio-3.0-asr-flash-streaming")
            || normalized.hasPrefix("fun-asr")
            || normalized.hasPrefix("paraformer-realtime")
    }

    static func isAliyunQwenRealtimeModel(_ model: String) -> Bool {
        let normalized = model.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return normalized.hasPrefix("qwen3-asr-flash-realtime")
    }

    static func isAliyunOmniRealtimeModel(_ model: String) -> Bool {
        let normalized = model.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return normalized.hasPrefix("qwen3.5-omni-flash-realtime")
            || normalized.hasPrefix("qwen3.5-omni-plus-realtime")
            || normalized.hasPrefix("qwen-omni-turbo-realtime")
    }

    static func aliyunQwenRealtimeSessionKind(for model: String) -> AliyunQwenRealtimeSessionKind? {
        if isAliyunQwenRealtimeModel(model) {
            return .qwenASR
        }
        if isAliyunOmniRealtimeModel(model) {
            return .omniASR
        }
        return nil
    }

    static func isAliyunFileTranscriptionModel(_ model: String) -> Bool {
        let normalized = model.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return normalized.hasPrefix("qwen3-asr-flash-filetrans")
            || normalized == "fun-asr"
            || normalized == "paraformer-v2"
    }

    static func resolvedAliyunQwenRealtimeEndpoint(_ endpoint: String, model: String) -> String {
        let trimmed = endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        let encodedModel = model.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? model

        guard !trimmed.isEmpty else {
            return "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=\(encodedModel)"
        }
        guard var components = URLComponents(string: trimmed) else {
            return trimmed
        }
        let normalizedPath = components.path.lowercased()
        if normalizedPath.hasSuffix("/api-ws/v1/realtime") {
            var items = components.queryItems ?? []
            if !items.contains(where: { $0.name == "model" }) {
                items.append(URLQueryItem(name: "model", value: model))
                components.queryItems = items
            }
            return components.string ?? trimmed
        }
        if normalizedPath.hasSuffix("/api-ws/v1/inference") {
            components.path = components.path.replacingOccurrences(of: "/api-ws/v1/inference", with: "/api-ws/v1/realtime")
            var items = components.queryItems ?? []
            if !items.contains(where: { $0.name == "model" }) {
                items.append(URLQueryItem(name: "model", value: model))
            }
            components.queryItems = items
            return components.string ?? trimmed
        }
        if normalizedPath.hasSuffix("/chat/completions") {
            let base = replacingPathSuffix(in: trimmed, oldSuffix: "/chat/completions", newSuffix: "/api-ws/v1/realtime")
            return base.contains("?") ? base : "\(base)?model=\(encodedModel)"
        }
        return trimmed
    }

    static func resolvedStepFunSSEEndpoint(_ endpoint: String) -> String {
        normalizedEndpoint(endpoint, defaultValue: "https://api.stepfun.com/v1/audio/asr/sse")
    }

    static func resolvedStepFunRealtimeEndpoint(_ endpoint: String) -> String {
        let trimmed = endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return "wss://api.stepfun.com/v1/realtime/asr/stream"
        }
        guard var components = URLComponents(string: trimmed) else {
            return trimmed
        }
        let normalizedPath = components.path.lowercased()
        if normalizedPath.hasSuffix("/v1/realtime/asr/stream") {
            components.scheme = "wss"
            return components.string ?? trimmed
        }
        if normalizedPath.hasSuffix("/v1/audio/asr/sse") ||
            normalizedPath.hasSuffix("/step_plan/v1/audio/asr/sse") {
            components.scheme = "wss"
            components.path = "/v1/realtime/asr/stream"
            components.queryItems = nil
            return components.string ?? trimmed
        }
        if normalizedPath.hasSuffix("/v1") {
            components.scheme = "wss"
            components.path = appendingPath(components.path, suffix: "/realtime/asr/stream")
            components.queryItems = nil
            return components.string ?? trimmed
        }
        return trimmed
    }

    static func normalizedEndpoint(_ value: String, defaultValue: String) -> String {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? defaultValue : trimmed
    }

    static func resolvedDoubaoResourceID(from configuration: RemoteProviderConfiguration) -> String {
        DoubaoASRConfiguration.resolvedResourceID(configuration.model)
    }

    static func resolvedDoubaoEndpoint(from configuration: RemoteProviderConfiguration) -> String {
        DoubaoASRConfiguration.resolvedEndpoint(configuration.endpoint, model: configuration.model)
    }

    static func resolvedDoubaoStreamingEndpoint(from configuration: RemoteProviderConfiguration) -> String {
        DoubaoASRConfiguration.resolvedStreamingEndpoint(configuration.endpoint, model: configuration.model)
    }

    static func resolvedXiaomiMiMoASREndpoint(_ endpoint: String) -> String {
        normalizedChatCompletionsEndpoint(
            endpoint,
            defaultValue: "https://api.xiaomimimo.com/v1/chat/completions"
        )
    }

    private static func appendingPath(_ value: String, suffix: String) -> String {
        value.hasSuffix("/") ? value + suffix.dropFirst() : value + suffix
    }

    private static func normalizedChatCompletionsEndpoint(_ value: String, defaultValue: String) -> String {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return defaultValue }
        guard let url = URL(string: trimmed) else { return trimmed }
        let normalizedPath = url.path.lowercased()
        if normalizedPath.hasSuffix("/chat/completions") {
            return trimmed
        }
        if normalizedPath.hasSuffix("/models") {
            return replacingPathSuffix(in: trimmed, oldSuffix: "/models", newSuffix: "/chat/completions")
        }
        if normalizedPath.hasSuffix("/v1") {
            return appendingPath(trimmed, suffix: "/chat/completions")
        }
        if normalizedPath.isEmpty || normalizedPath == "/" {
            return appendingPath(trimmed, suffix: "/v1/chat/completions")
        }
        return trimmed
    }

    private static func replacingPathSuffix(in value: String, oldSuffix: String, newSuffix: String) -> String {
        guard value.lowercased().hasSuffix(oldSuffix) else { return value }
        return String(value.dropLast(oldSuffix.count)) + newSuffix
    }
}

extension RemoteASREndpointSupport {
    static let geminiLiveDefaultEndpoint =
        "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"

    static func resolvedGeminiLiveEndpoint(_ endpoint: String) -> String {
        let trimmed = endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return geminiLiveDefaultEndpoint
        }
        guard var components = URLComponents(string: trimmed) else {
            return trimmed
        }
        switch components.scheme?.lowercased() {
        case "https":
            components.scheme = "wss"
        case "http":
            components.scheme = "ws"
        default:
            break
        }
        if components.path.isEmpty || components.path == "/" {
            components.path = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
        }
        // The API key is attached only at connect time, never stored in settings.
        components.queryItems = components.queryItems?.filter { $0.name != "key" }
        if components.queryItems?.isEmpty == true {
            components.queryItems = nil
        }
        return components.string ?? trimmed
    }

    static func geminiLiveURL(endpoint: String, apiKey: String) -> URL? {
        let resolved = resolvedGeminiLiveEndpoint(endpoint)
        guard var components = URLComponents(string: resolved) else { return nil }
        var items = components.queryItems ?? []
        items.removeAll { $0.name == "key" }
        items.append(URLQueryItem(name: "key", value: apiKey))
        components.queryItems = items
        return components.url
    }
}
