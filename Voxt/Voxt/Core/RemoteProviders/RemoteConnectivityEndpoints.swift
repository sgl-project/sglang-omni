// RemoteConnectivityEndpoints.swift
// Provides Remote Connectivity Endpoints for remote provider configuration.

import Foundation

enum RemoteProviderConnectivityTestEndpoints {
    static func resolvedASRTranscriptionEndpoint(endpoint: String, defaultValue: String) -> String {
        let trimmed = endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return defaultValue }
        guard let url = URL(string: trimmed) else { return trimmed }
        let normalizedPath = url.path.lowercased()
        if normalizedPath.hasSuffix("/audio/transcriptions") {
            return trimmed
        }
        if normalizedPath.hasSuffix("/v1") {
            return trimmed + "/audio/transcriptions"
        }
        if normalizedPath.isEmpty || normalizedPath == "/" {
            return trimmed.hasSuffix("/") ? trimmed + "v1/audio/transcriptions" : trimmed + "/v1/audio/transcriptions"
        }
        return trimmed
    }

    static func resolvedGLMASRTranscriptionEndpoint(endpoint: String, defaultValue: String) -> String {
        let trimmed = endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return defaultValue }
        guard let url = URL(string: trimmed) else { return trimmed }
        let normalizedPath = url.path.lowercased()
        if normalizedPath.hasSuffix("/audio/transcriptions") {
            return trimmed
        }
        if normalizedPath.hasSuffix("/models") {
            return replacingPathSuffix(in: trimmed, oldSuffix: "/models", newSuffix: "/audio/transcriptions")
        }
        if normalizedPath.hasSuffix("/v4") {
            return appendingPath(trimmed, suffix: "/audio/transcriptions")
        }
        return trimmed
    }

    static func resolvedAliyunASRRealtimeEndpoint(endpoint: String, defaultValue: String) -> String {
        let trimmed = endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
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
            return trimmed.hasSuffix("/") ? trimmed + "v1/chat/completions" : trimmed + "/v1/chat/completions"
        }
        return trimmed
    }

    static func resolvedAliyunASRRealtimeWebSocketEndpoint(endpoint: String, defaultValue: String) -> String {
        let trimmed = endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return defaultValue }
        guard var components = URLComponents(string: trimmed) else { return trimmed }
        let normalizedPath = components.path.lowercased()
        if normalizedPath.hasSuffix("/api-ws/v1/inference") {
            return trimmed
        }
        if normalizedPath.hasSuffix("/api-ws/v1/realtime") {
            components.path = components.path.replacingOccurrences(of: "/api-ws/v1/realtime", with: "/api-ws/v1/inference")
            components.queryItems = nil
            return components.string ?? trimmed
        }
        if normalizedPath.hasSuffix("/chat/completions") {
            return replacingPathSuffix(in: trimmed, oldSuffix: "/chat/completions", newSuffix: "/api-ws/v1/inference")
        }
        if normalizedPath.hasSuffix("/models") {
            return replacingPathSuffix(in: trimmed, oldSuffix: "/models", newSuffix: "/api-ws/v1/inference")
        }
        if normalizedPath.hasSuffix("/v1") {
            return appendingPath(trimmed, suffix: "/inference")
        }
        if normalizedPath.isEmpty || normalizedPath == "/" {
            return trimmed.hasSuffix("/") ? trimmed + "api-ws/v1/inference" : trimmed + "/api-ws/v1/inference"
        }
        return trimmed
    }

    static func resolvedAliyunASRQwenRealtimeWebSocketEndpoint(endpoint: String, model: String) -> String {
        let trimmed = endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        let encodedModel = model.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? model
        guard !trimmed.isEmpty else {
            return "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=\(encodedModel)"
        }
        guard var components = URLComponents(string: trimmed) else { return trimmed }
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
                components.queryItems = items
            }
            return components.string ?? trimmed
        }
        if normalizedPath.hasSuffix("/chat/completions") {
            let base = replacingPathSuffix(in: trimmed, oldSuffix: "/chat/completions", newSuffix: "/api-ws/v1/realtime")
            return base.contains("?") ? base : "\(base)?model=\(encodedModel)"
        }
        return trimmed
    }

    static func resolvedDoubaoASREndpoint(_ endpoint: String, model: String) -> String {
        DoubaoASRConfiguration.resolvedStreamingEndpoint(endpoint, model: model)
    }

    static func resolvedStepFunASREndpoint(_ endpoint: String) -> String {
        RemoteASREndpointSupport.normalizedEndpoint(
            endpoint,
            defaultValue: "https://api.stepfun.com/v1/audio/asr/sse"
        )
    }

    static func resolvedXiaomiMiMoASREndpoint(_ endpoint: String) -> String {
        RemoteASREndpointSupport.resolvedXiaomiMiMoASREndpoint(endpoint)
    }

    static func resolvedGeminiLiveASREndpoint(_ endpoint: String) -> String {
        RemoteASREndpointSupport.resolvedGeminiLiveEndpoint(endpoint)
    }

    private static func replacingPathSuffix(in value: String, oldSuffix: String, newSuffix: String) -> String {
        guard value.lowercased().hasSuffix(oldSuffix) else { return value }
        return String(value.dropLast(oldSuffix.count)) + newSuffix
    }

    private static func appendingPath(_ value: String, suffix: String) -> String {
        if value.hasSuffix("/") {
            return value + suffix.dropFirst()
        }
        return value + suffix
    }
}
