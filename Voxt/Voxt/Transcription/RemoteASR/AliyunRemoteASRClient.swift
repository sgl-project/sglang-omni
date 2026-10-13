// AliyunRemoteASRClient.swift
// Provides Aliyun Remote ASRClient for remote ASR adapters.

import Foundation

enum AliyunRemoteASRClient {
    static func transcribe(
        fileURL: URL,
        apiKey: String,
        model: String,
        endpoint: String
    ) async throws -> String {
        let ossURL = try await uploadFile(
            fileURL: fileURL,
            apiKey: apiKey,
            model: model,
            endpoint: endpoint
        )
        let taskID = try await submitTask(
            ossURL: ossURL,
            apiKey: apiKey,
            model: model,
            endpoint: endpoint
        )
        let result = try await pollTaskResult(
            taskID: taskID,
            apiKey: apiKey,
            model: model,
            endpoint: endpoint
        )
        if isNoValidFragment(result) {
            return ""
        }
        if let text = extractText(from: result), !text.isEmpty {
            return text
        }
        if let transcriptionURL = extractTranscriptionURL(from: result) {
            let fetched = try await fetchTranscriptionFile(from: transcriptionURL)
            if isNoValidFragment(fetched) {
                return ""
            }
            if let text = extractResultText(from: fetched), !text.isEmpty {
                return text
            }
        }
        throw NSError(
            domain: "Voxt.RemoteASR",
            code: -37,
            userInfo: [NSLocalizedDescriptionKey: "Aliyun Bailian ASR returned no text content."]
        )
    }

    static func extractText(from object: Any) -> String? {
        if let dict = object as? [String: Any] {
            if let choices = dict["choices"] as? [[String: Any]],
               let first = choices.first,
               let message = first["message"] as? [String: Any] {
                if let content = message["content"] as? String {
                    let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !trimmed.isEmpty {
                        return trimmed
                    }
                }

                if let blocks = message["content"] as? [[String: Any]] {
                    let texts = blocks.compactMap { block -> String? in
                        if let text = block["text"] as? String {
                            return text.trimmingCharacters(in: .whitespacesAndNewlines)
                        }
                        return nil
                    }.filter { !$0.isEmpty }
                    if !texts.isEmpty {
                        return texts.joined(separator: "\n")
                    }
                }
            }

            if let output = dict["output"] as? [String: Any] {
                if let text = output["text"] as? String, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    return text
                }
                if let results = output["results"] as? [[String: Any]] {
                    let texts = results.compactMap { item in
                        (item["text"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines)
                    }.filter { !$0.isEmpty }
                    if !texts.isEmpty {
                        return texts.joined(separator: "\n")
                    }
                }
            }
        }
        return nil
    }

    static func normalizedTranscriptionURL(_ value: String) -> String {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.lowercased().hasPrefix("http://"),
           trimmed.lowercased().contains(".aliyuncs.com/") {
            return "https://" + trimmed.dropFirst("http://".count)
        }
        guard var components = URLComponents(string: trimmed) else { return trimmed }
        if components.scheme?.lowercased() == "http",
           let host = components.host?.lowercased(),
           host.contains("aliyuncs.com") {
            components.scheme = "https"
        }
        return components.string ?? trimmed
    }

    private static func uploadFile(
        fileURL: URL,
        apiKey: String,
        model: String,
        endpoint: String
    ) async throws -> String {
        let policy = try await requestUploadPolicy(
            apiKey: apiKey,
            model: model,
            endpoint: endpoint
        )
        let key = buildOSSKey(uploadDirectory: policy.uploadDirectory, fileURL: fileURL)
        let boundary = "Boundary-\(UUID().uuidString)"
        let fields = [(name: "key", value: key)]
            + policy.formFields.sorted(by: { $0.key < $1.key }).map { (name: $0.key, value: $0.value) }
            + [(name: "success_action_status", value: "200")]
        let body = try MultipartFileBody.create(
            sourceFileURL: fileURL,
            boundary: boundary,
            fields: fields,
            mimeType: audioMIMEType(for: fileURL)
        )
        defer { body.remove() }

        guard let uploadURL = URL(string: policy.uploadHost) else {
            throw NSError(domain: "Voxt.RemoteASR", code: -38, userInfo: [NSLocalizedDescriptionKey: "Invalid Aliyun upload host URL."])
        }
        var request = URLRequest(url: uploadURL)
        request.httpMethod = "POST"
        request.timeoutInterval = 60
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.setValue(String(body.byteCount), forHTTPHeaderField: "Content-Length")

        let (data, response) = try await VoxtNetworkSession.active.upload(for: request, fromFile: body.url)
        guard let http = response as? HTTPURLResponse else {
            throw NSError(domain: "Voxt.RemoteASR", code: -39, userInfo: [NSLocalizedDescriptionKey: "Invalid Aliyun upload response."])
        }
        guard (200...299).contains(http.statusCode) else {
            let payload = String(data: data.prefix(300), encoding: .utf8) ?? ""
            throw NSError(
                domain: "Voxt.RemoteASR",
                code: http.statusCode,
                userInfo: [NSLocalizedDescriptionKey: "Aliyun file upload failed (HTTP \(http.statusCode)): \(payload)"]
            )
        }
        return "oss://\(key)"
    }

    private static func requestUploadPolicy(
        apiKey: String,
        model: String,
        endpoint: String
    ) async throws -> AliyunUploadPolicy {
        guard var components = URLComponents(
            string: AliyunRemoteASRConfiguration.resolvedUploadPolicyEndpoint(endpoint, model: model)
        ) else {
            throw NSError(domain: "Voxt.RemoteASR", code: -40, userInfo: [NSLocalizedDescriptionKey: "Invalid Aliyun upload policy endpoint URL."])
        }
        var queryItems = components.queryItems ?? []
        if !queryItems.contains(where: { $0.name == "action" }) {
            queryItems.append(URLQueryItem(name: "action", value: "getPolicy"))
        }
        if !queryItems.contains(where: { $0.name == "model" }) {
            queryItems.append(URLQueryItem(name: "model", value: model))
        }
        components.queryItems = queryItems
        guard let url = components.url else {
            throw NSError(domain: "Voxt.RemoteASR", code: -41, userInfo: [NSLocalizedDescriptionKey: "Invalid Aliyun upload policy endpoint URL."])
        }

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = 30
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

        let (data, response) = try await VoxtNetworkSession.active.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw NSError(domain: "Voxt.RemoteASR", code: -42, userInfo: [NSLocalizedDescriptionKey: "Invalid Aliyun upload policy response."])
        }
        guard (200...299).contains(http.statusCode) else {
            let payload = String(data: data.prefix(300), encoding: .utf8) ?? ""
            throw NSError(
                domain: "Voxt.RemoteASR",
                code: http.statusCode,
                userInfo: [NSLocalizedDescriptionKey: "Aliyun upload policy request failed (HTTP \(http.statusCode)): \(payload)"]
            )
        }
        let object = try JSONSerialization.jsonObject(with: data)
        guard let policy = AliyunUploadPolicy(object: object) else {
            throw NSError(domain: "Voxt.RemoteASR", code: -43, userInfo: [NSLocalizedDescriptionKey: "Aliyun upload policy payload is incomplete."])
        }
        return policy
    }

    private static func submitTask(
        ossURL: String,
        apiKey: String,
        model: String,
        endpoint: String
    ) async throws -> String {
        guard let url = URL(string: AliyunRemoteASRConfiguration.resolvedTranscriptionEndpoint(endpoint, model: model)) else {
            throw NSError(domain: "Voxt.RemoteASR", code: -44, userInfo: [NSLocalizedDescriptionKey: "Invalid Aliyun transcription endpoint URL."])
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 60
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("enable", forHTTPHeaderField: "X-DashScope-Async")
        request.setValue("enable", forHTTPHeaderField: "X-DashScope-OssResourceResolve")
        request.httpBody = try JSONSerialization.data(
            withJSONObject: AliyunRemoteASRConfiguration.submissionBody(model: model, fileURL: ossURL)
        )

        let (data, response) = try await VoxtNetworkSession.active.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw NSError(domain: "Voxt.RemoteASR", code: -45, userInfo: [NSLocalizedDescriptionKey: "Invalid Aliyun transcription response."])
        }
        guard (200...299).contains(http.statusCode) else {
            let payload = String(data: data.prefix(300), encoding: .utf8) ?? ""
            throw NSError(
                domain: "Voxt.RemoteASR",
                code: http.statusCode,
                userInfo: [NSLocalizedDescriptionKey: "Aliyun Bailian ASR request failed (HTTP \(http.statusCode)): \(payload)"]
            )
        }
        let object = try JSONSerialization.jsonObject(with: data)
        if let taskID = extractTaskID(from: object), !taskID.isEmpty {
            return taskID
        }
        throw NSError(domain: "Voxt.RemoteASR", code: -46, userInfo: [NSLocalizedDescriptionKey: "Aliyun Bailian ASR did not return a task ID."])
    }

    private static func pollTaskResult(
        taskID: String,
        apiKey: String,
        model: String,
        endpoint: String
    ) async throws -> Any {
        let pollEndpoint = AliyunRemoteASRConfiguration.resolvedTaskEndpoint(endpoint, model: model, taskID: taskID)
        guard let url = URL(string: pollEndpoint) else {
            throw NSError(domain: "Voxt.RemoteASR", code: -47, userInfo: [NSLocalizedDescriptionKey: "Invalid Aliyun task query endpoint URL."])
        }

        for _ in 0..<40 {
            var request = URLRequest(url: url)
            request.httpMethod = AliyunRemoteASRConfiguration.taskQueryMethod(for: model) == .post ? "POST" : "GET"
            request.timeoutInterval = 30
            request.setValue("application/json", forHTTPHeaderField: "Accept")
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
            request.setValue("enable", forHTTPHeaderField: "X-DashScope-Async")

            let (data, response) = try await VoxtNetworkSession.active.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                throw NSError(domain: "Voxt.RemoteASR", code: -48, userInfo: [NSLocalizedDescriptionKey: "Invalid Aliyun task query response."])
            }
            guard (200...299).contains(http.statusCode) else {
                let payload = String(data: data.prefix(300), encoding: .utf8) ?? ""
                throw NSError(
                    domain: "Voxt.RemoteASR",
                    code: http.statusCode,
                    userInfo: [NSLocalizedDescriptionKey: "Aliyun task query failed (HTTP \(http.statusCode)): \(payload)"]
                )
            }

            let object = try JSONSerialization.jsonObject(with: data)
            let status = extractTaskStatus(from: object)
            switch status {
            case "SUCCEEDED":
                return object
            case "FAILED", "CANCELED":
                let detail = extractErrorMessage(from: object)
                if isNoValidFragment(object) || detail.uppercased().contains("SUCCESS_WITH_NO_VALID_FRAGMENT") {
                    return object
                }
                throw NSError(
                    domain: "Voxt.RemoteASR",
                    code: -49,
                    userInfo: [NSLocalizedDescriptionKey: detail.isEmpty ? "Aliyun task failed." : detail]
                )
            default:
                try await Task.sleep(nanoseconds: 600_000_000)
            }
        }

        throw NSError(domain: "Voxt.RemoteASR", code: -50, userInfo: [NSLocalizedDescriptionKey: "Aliyun Bailian ASR task timed out."])
    }

    private static func fetchTranscriptionFile(from urlString: String) async throws -> Any {
        let normalizedURLString = normalizedTranscriptionURL(urlString)
        guard let url = URL(string: normalizedURLString) else {
            throw NSError(domain: "Voxt.RemoteASR", code: -51, userInfo: [NSLocalizedDescriptionKey: "Invalid Aliyun transcription result URL."])
        }
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = 30
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        let (data, response) = try await VoxtNetworkSession.active.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw NSError(domain: "Voxt.RemoteASR", code: -52, userInfo: [NSLocalizedDescriptionKey: "Invalid Aliyun transcription file response."])
        }
        guard (200...299).contains(http.statusCode) else {
            let payload = String(data: data.prefix(300), encoding: .utf8) ?? ""
            throw NSError(
                domain: "Voxt.RemoteASR",
                code: http.statusCode,
                userInfo: [NSLocalizedDescriptionKey: "Aliyun transcription result download failed (HTTP \(http.statusCode)): \(payload)"]
            )
        }
        return try JSONSerialization.jsonObject(with: data)
    }

    private static func buildOSSKey(uploadDirectory: String, fileURL: URL) -> String {
        let trimmedDirectory = uploadDirectory.trimmingCharacters(in: .whitespacesAndNewlines)
        let suffix = fileURL.pathExtension.isEmpty ? "" : ".\(fileURL.pathExtension.lowercased())"
        if trimmedDirectory.isEmpty {
            return "voxt/\(UUID().uuidString.lowercased())\(suffix)"
        }
        let normalizedDirectory = trimmedDirectory.hasSuffix("/") ? trimmedDirectory : "\(trimmedDirectory)/"
        return "\(normalizedDirectory)\(UUID().uuidString.lowercased())\(suffix)"
    }

    private static func audioMIMEType(for fileURL: URL) -> String {
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

    private static func extractTaskID(from object: Any) -> String? {
        guard let dict = object as? [String: Any] else { return nil }
        if let output = dict["output"] as? [String: Any],
           let taskID = output["task_id"] as? String,
           !taskID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return taskID
        }
        if let taskID = dict["task_id"] as? String,
           !taskID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return taskID
        }
        return nil
    }

    private static func extractTaskStatus(from object: Any) -> String {
        guard let dict = object as? [String: Any] else { return "" }
        if let output = dict["output"] as? [String: Any] {
            let directStatus = (output["task_status"] as? String ?? "").uppercased()
            if !directStatus.isEmpty {
                return directStatus
            }
            if let results = output["results"] as? [[String: Any]],
               let first = results.first {
                let nestedStatus = (first["task_status"] as? String ?? "").uppercased()
                if !nestedStatus.isEmpty {
                    return nestedStatus
                }
            }
        }
        return (dict["task_status"] as? String ?? "").uppercased()
    }

    private static func extractErrorMessage(from object: Any) -> String {
        guard let dict = object as? [String: Any] else { return "" }
        if let output = dict["output"] as? [String: Any] {
            if let message = output["message"] as? String, !message.isEmpty {
                return message
            }
            if let code = output["code"] as? String, !code.isEmpty {
                return code
            }
        }
        if let message = dict["message"] as? String, !message.isEmpty {
            return message
        }
        if let error = dict["error"] as? [String: Any],
           let message = error["message"] as? String,
           !message.isEmpty {
            return message
        }
        return ""
    }

    static func extractTranscriptionURL(from object: Any) -> String? {
        guard let dict = object as? [String: Any] else { return nil }
        if let output = dict["output"] as? [String: Any] {
            if let result = output["result"] as? [String: Any],
               let url = result["transcription_url"] as? String,
               !url.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return url
            }
            if let results = output["results"] as? [[String: Any]] {
                for item in results {
                    if let url = item["transcription_url"] as? String,
                       !url.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        return url
                    }
                }
            }
            if let url = output["transcription_url"] as? String,
               !url.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return url
            }
        }
        return nil
    }

    static func extractResultText(from object: Any) -> String? {
        if let direct = extractText(from: object),
           !direct.lowercased().hasPrefix("http"),
           !direct.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return direct.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        let texts = collectTexts(from: object)
        if !texts.isEmpty {
            return texts.joined(separator: "\n")
        }
        return nil
    }

    static func isNoValidFragment(_ object: Any) -> Bool {
        guard let detail = statusDetail(from: object) else { return false }
        return detail.uppercased().contains("SUCCESS_WITH_NO_VALID_FRAGMENT")
    }

    private static func statusDetail(from object: Any) -> String? {
        if let dict = object as? [String: Any] {
            if let output = dict["output"] as? [String: Any] {
                if let code = output["code"] as? String, !code.isEmpty {
                    return code
                }
                if let message = output["message"] as? String, !message.isEmpty {
                    return message
                }
                if let results = output["results"] as? [[String: Any]] {
                    for item in results {
                        if let code = item["code"] as? String, !code.isEmpty {
                            return code
                        }
                        if let message = item["message"] as? String, !message.isEmpty {
                            return message
                        }
                    }
                }
            }
            if let code = dict["code"] as? String, !code.isEmpty {
                return code
            }
            if let message = dict["message"] as? String, !message.isEmpty {
                return message
            }
        }
        return nil
    }

    private static func collectTexts(from object: Any) -> [String] {
        var texts: [String] = []

        func visit(_ value: Any) {
            if let dict = value as? [String: Any] {
                for key in ["text", "transcript", "content", "sentence"] {
                    if let text = dict[key] as? String {
                        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                        if !trimmed.isEmpty, !trimmed.lowercased().hasPrefix("http") {
                            texts.append(trimmed)
                        }
                    }
                }
                for key in ["transcripts", "results", "sentences", "paragraphs", "utterances", "segments", "words", "result"] {
                    if let nested = dict[key] {
                        visit(nested)
                    }
                }
            } else if let array = value as? [Any] {
                for item in array {
                    visit(item)
                }
            }
        }

        visit(object)
        var deduped: [String] = []
        for text in texts where deduped.last != text {
            deduped.append(text)
        }
        return deduped
    }
}

private struct AliyunUploadPolicy {
    let uploadHost: String
    let uploadDirectory: String
    let formFields: [String: String]

    init?(object: Any) {
        guard let dict = object as? [String: Any] else { return nil }
        let payload = (dict["data"] as? [String: Any]) ?? dict
        guard let uploadHost = payload["upload_host"] as? String,
              let uploadDirectory = payload["upload_dir"] as? String,
              !uploadHost.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              !uploadDirectory.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return nil
        }

        var fields: [String: String] = [:]
        for key in [
            "OSSAccessKeyId",
            "policy",
            "Signature",
            "signature",
            "x-oss-signature-version",
            "x-oss-credential",
            "x-oss-date",
            "x-oss-security-token",
            "x-oss-object-acl",
            "x-oss-forbid-overwrite",
            "callback"
        ] {
            if let value = payload[key] as? String, !value.isEmpty {
                fields[key] = value
            }
        }
        if fields["OSSAccessKeyId"] == nil,
           let accessKeyID = payload["oss_access_key_id"] as? String,
           !accessKeyID.isEmpty {
            fields["OSSAccessKeyId"] = accessKeyID
        }
        if fields["Signature"] == nil,
           let signature = payload["signature"] as? String,
           !signature.isEmpty {
            fields["Signature"] = signature
        }
        if let value = payload["x_oss_object_acl"] as? String, !value.isEmpty {
            fields["x-oss-object-acl"] = value
        }
        if let value = payload["x_oss_forbid_overwrite"] as? String, !value.isEmpty {
            fields["x-oss-forbid-overwrite"] = value
        }
        if let value = payload["x_oss_security_token"] as? String, !value.isEmpty {
            fields["x-oss-security-token"] = value
        }

        self.uploadHost = uploadHost
        self.uploadDirectory = uploadDirectory
        self.formFields = fields
    }
}
