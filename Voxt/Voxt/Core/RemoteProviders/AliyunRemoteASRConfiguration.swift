// AliyunRemoteASRConfiguration.swift
// Provides Aliyun Remote ASRConfiguration for remote provider configuration.

import Foundation

enum AliyunRemoteASRConfiguration {
    enum Routing: Equatable {
        case asyncFileTranscription
        case compatibleShortAudio
    }

    enum Region: Equatable {
        case cnBeijing
        case apSoutheast1
        case usEast1
        case unknown
    }

    enum TaskQueryMethod {
        case get
        case post
    }

    static let defaultTranscriptionEndpointCN = "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription"
    static let defaultTranscriptionEndpointSG = "https://dashscope-intl.aliyuncs.com/api/v1/services/audio/asr/transcription"
    static let defaultCompatibleEndpointCN = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    static let defaultCompatibleEndpointSG = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
    static let defaultCompatibleEndpointUS = "https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions"
    static let defaultTranscriptionModel = "qwen3-asr-flash-filetrans"

    static func makeRealtimeTaskID() -> String {
        UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
    }

    static func realtimeSocketEvent(from object: [String: Any]) -> String {
        if let header = object["header"] as? [String: Any],
           let event = header["event"] as? String {
            let normalized = event.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if !normalized.isEmpty {
                return normalized
            }
        }
        return (object["event"] as? String ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
    }

    static func realtimeSocketErrorMessage(from object: [String: Any]) -> String? {
        let payload = object["payload"] as? [String: Any] ?? [:]
        let header = object["header"] as? [String: Any] ?? [:]
        let candidates: [Any?] = [
            payload["message"],
            payload["error_message"],
            header["error_message"],
            header["errorMessage"],
            object["message"],
            object["error_message"]
        ]
        for candidate in candidates {
            if let value = candidate as? String {
                let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines)
                if !normalized.isEmpty {
                    return normalized
                }
            }
        }
        return nil
    }

    static func funRealtimeControlPayload(
        action: String,
        taskID: String,
        model: String? = nil,
        parameters: [String: Any]? = nil,
        context: [[String: Any]] = []
    ) -> [String: Any] {
        var object: [String: Any] = [
            "header": [
                "action": action,
                "task_id": taskID,
                "streaming": "duplex"
            ]
        ]
        if let model {
            var input: [String: Any] = [:]
            if !context.isEmpty {
                input["context"] = context
            }
            object["payload"] = [
                "task_group": "audio",
                "task": "asr",
                "function": "recognition",
                "model": model,
                "parameters": parameters ?? [:],
                "input": input
            ]
        } else {
            object["payload"] = [
                "input": [:]
            ]
        }
        return object
    }

    static func normalizedTranscriptionModel(_ model: String) -> String {
        let trimmed = model.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? defaultTranscriptionModel : trimmed
    }

    static func routing(for model: String) -> Routing? {
        let normalized = normalizedTranscriptionModel(model).lowercased()
        if isAsyncFileTranscriptionModel(normalized) {
            return .asyncFileTranscription
        }
        if isCompatibleShortAudioModel(normalized) {
            return .compatibleShortAudio
        }
        return nil
    }

    static func region(for endpoint: String) -> Region {
        let trimmed = endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let host = URL(string: trimmed)?.host?.lowercased() else {
            return .unknown
        }
        if host.contains("dashscope-us.aliyuncs.com") {
            return .usEast1
        }
        if host.contains("dashscope-intl.aliyuncs.com") {
            return .apSoutheast1
        }
        if host.contains("dashscope.aliyuncs.com") {
            return .cnBeijing
        }
        return .unknown
    }

    static func validationError(model: String, endpoint: String) -> String? {
        let normalized = normalizedTranscriptionModel(model)
        guard let routing = routing(for: normalized) else {
            return AppLocalization.format("Aliyun Remote ASR model %@ is not supported.", normalized)
        }
        let region = region(for: endpoint)
        guard region != .unknown else { return nil }

        switch routing {
        case .asyncFileTranscription:
            if region == .usEast1 {
                return AppLocalization.format(
                    "Aliyun Remote ASR model %@ is not available in US (Virginia). Use the Beijing/Singapore transcription endpoint or switch to a US short-audio model.",
                    normalized
                )
            }
        case .compatibleShortAudio:
            if isUSShortAudioModel(normalized), region != .usEast1 {
                return AppLocalization.format(
                    "Aliyun Remote ASR model %@ requires the US (Virginia) endpoint.",
                    normalized
                )
            }
            if !isUSShortAudioModel(normalized), region == .usEast1 {
                return AppLocalization.format(
                    "Aliyun Remote ASR model %@ is not available in US (Virginia). Use the Beijing/Singapore endpoint or a US-specific model.",
                    normalized
                )
            }
        }

        return nil
    }

    static func endpointPresets(for model: String) -> [RemoteEndpointPreset] {
        guard let routing = routing(for: model) else { return [] }
        switch routing {
        case .asyncFileTranscription:
            return [
                RemoteEndpointPreset(id: "aliyun-filetrans-cn-beijing", title: AppLocalization.localizedString("Transcription HTTP · Beijing"), url: defaultTranscriptionEndpointCN),
                RemoteEndpointPreset(id: "aliyun-filetrans-ap-southeast-1", title: AppLocalization.localizedString("Transcription HTTP · Singapore"), url: defaultTranscriptionEndpointSG)
            ]
        case .compatibleShortAudio:
            if isUSShortAudioModel(model) {
                return [
                    RemoteEndpointPreset(id: "aliyun-compatible-us-east-1", title: AppLocalization.localizedString("Transcription HTTP · US (Virginia)"), url: defaultCompatibleEndpointUS)
                ]
            }
            return [
                RemoteEndpointPreset(id: "aliyun-compatible-cn-beijing", title: AppLocalization.localizedString("Transcription HTTP · Beijing"), url: defaultCompatibleEndpointCN),
                RemoteEndpointPreset(id: "aliyun-compatible-ap-southeast-1", title: AppLocalization.localizedString("Transcription HTTP · Singapore"), url: defaultCompatibleEndpointSG)
            ]
        }
    }

    static func resolvedCompatibleEndpoint(_ endpoint: String, model: String) -> String {
        let trimmed = endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        let defaultValue = isUSShortAudioModel(model) ? defaultCompatibleEndpointUS : defaultCompatibleEndpointCN
        guard !trimmed.isEmpty else { return defaultValue }
        let normalizedValue = normalizedHTTPSValue(trimmed)
        guard let url = URL(string: normalizedValue) else { return normalizedValue }
        let normalizedPath = url.path.lowercased()
        if normalizedPath.hasSuffix("/compatible-mode/v1/chat/completions") || normalizedPath.hasSuffix("/v1/chat/completions") {
            return normalizedValue
        }
        if normalizedPath.hasSuffix("/api/v1/services/audio/asr/transcription") {
            return replacingPathSuffix(in: normalizedValue, oldSuffix: "/api/v1/services/audio/asr/transcription", newSuffix: "/compatible-mode/v1/chat/completions")
        }
        if normalizedPath.hasSuffix("/api-ws/v1/realtime") {
            return replacingPathSuffix(in: normalizedValue, oldSuffix: "/api-ws/v1/realtime", newSuffix: "/compatible-mode/v1/chat/completions")
        }
        if normalizedPath.hasSuffix("/api-ws/v1/inference") {
            return replacingPathSuffix(in: normalizedValue, oldSuffix: "/api-ws/v1/inference", newSuffix: "/compatible-mode/v1/chat/completions")
        }
        return normalizedValue
    }

    static func resolvedTranscriptionEndpoint(_ endpoint: String, model: String) -> String {
        let trimmed = endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        let defaultValue = defaultTranscriptionEndpoint(for: model)
        guard !trimmed.isEmpty else { return defaultValue }
        let normalizedValue = normalizedHTTPSValue(trimmed)
        guard let url = URL(string: normalizedValue) else { return normalizedValue }
        let normalizedPath = url.path.lowercased()
        if normalizedPath.hasSuffix("/api/v1/services/audio/asr/transcription") {
            return normalizedValue
        }
        if normalizedPath.hasSuffix("/api-ws/v1/realtime") {
            return replacingPathSuffix(in: normalizedValue, oldSuffix: "/api-ws/v1/realtime", newSuffix: "/api/v1/services/audio/asr/transcription")
        }
        if normalizedPath.hasSuffix("/api-ws/v1/inference") {
            return replacingPathSuffix(in: normalizedValue, oldSuffix: "/api-ws/v1/inference", newSuffix: "/api/v1/services/audio/asr/transcription")
        }
        if normalizedPath.hasSuffix("/compatible-mode/v1/chat/completions") {
            return replacingPathSuffix(in: normalizedValue, oldSuffix: "/compatible-mode/v1/chat/completions", newSuffix: "/api/v1/services/audio/asr/transcription")
        }
        if normalizedPath.hasSuffix("/v1/chat/completions") {
            return replacingPathSuffix(in: normalizedValue, oldSuffix: "/v1/chat/completions", newSuffix: "/api/v1/services/audio/asr/transcription")
        }
        return normalizedValue
    }

    static func resolvedUploadPolicyEndpoint(_ endpoint: String, model: String) -> String {
        replacingPathSuffix(
            in: resolvedTranscriptionEndpoint(endpoint, model: model),
            oldSuffix: "/api/v1/services/audio/asr/transcription",
            newSuffix: "/api/v1/uploads"
        )
    }

    static func resolvedTaskEndpoint(_ endpoint: String, model: String, taskID: String) -> String {
        replacingPathSuffix(
            in: resolvedTranscriptionEndpoint(endpoint, model: model),
            oldSuffix: "/api/v1/services/audio/asr/transcription",
            newSuffix: "/api/v1/tasks/\(taskID)"
        )
    }

    static func taskQueryMethod(for model: String) -> TaskQueryMethod {
        let normalized = normalizedTranscriptionModel(model).lowercased()
        if normalized.hasPrefix("qwen3-asr-flash-filetrans") {
            return .get
        }
        return .post
    }

    static func submissionBody(model: String, fileURL: String) -> [String: Any] {
        let normalized = normalizedTranscriptionModel(model)
        if normalized.lowercased().hasPrefix("qwen3-asr-flash-filetrans") {
            return [
                "model": normalized,
                "input": [
                    "file_url": fileURL
                ],
                "parameters": [
                    "channel_id": [0]
                ]
            ]
        }
        return [
            "model": normalized,
            "input": [
                "file_urls": [fileURL]
            ],
            "parameters": [
                "channel_id": [0]
            ]
        ]
    }

    private static func defaultTranscriptionEndpoint(for model: String) -> String {
        switch region(for: endpointPresets(for: model).first?.url ?? "") {
        case .apSoutheast1:
            return defaultTranscriptionEndpointSG
        default:
            return defaultTranscriptionEndpointCN
        }
    }

    private static func isAsyncFileTranscriptionModel(_ model: String) -> Bool {
        model.hasPrefix("qwen3-asr-flash-filetrans")
            || model == "fun-asr"
            || model == "paraformer-v2"
    }

    private static func isCompatibleShortAudioModel(_ model: String) -> Bool {
        model.hasPrefix("qwen3-asr-flash")
            && !model.hasPrefix("qwen3-asr-flash-filetrans")
    }

    private static func isUSShortAudioModel(_ model: String) -> Bool {
        model.contains("-us")
    }

    private static func replacingPathSuffix(in value: String, oldSuffix: String, newSuffix: String) -> String {
        guard value.lowercased().hasSuffix(oldSuffix) else { return value }
        return String(value.dropLast(oldSuffix.count)) + newSuffix
    }

    private static func normalizedHTTPSValue(_ value: String) -> String {
        guard var components = URLComponents(string: value) else { return value }
        if components.scheme == "wss" {
            components.scheme = "https"
        } else if components.scheme == "ws" {
            components.scheme = "http"
        }
        return components.string ?? value
    }
}
