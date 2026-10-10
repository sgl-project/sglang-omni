import Foundation

enum RemoteASRTextSupport {
    static func xiaomiMiMoASRPayload(
        model: String,
        audioData: Data,
        mimeType: String,
        hintPayload: ResolvedASRHintPayload
    ) -> [String: Any] {
        let effectiveModel = model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? RemoteASRProvider.xiaomiMiMoASR.suggestedModel
            : model
        let language = xiaomiMiMoLanguage(from: hintPayload.language)
        let audioDataURI = "data:\(mimeType);base64,\(audioData.base64EncodedString())"

        return [
            "model": effectiveModel,
            "messages": [
                [
                    "role": "user",
                    "content": [
                        [
                            "type": "input_audio",
                            "input_audio": [
                                "data": audioDataURI
                            ]
                        ]
                    ]
                ]
            ],
            "asr_options": [
                "language": language
            ]
        ]
    }

    static func xiaomiMiMoLanguage(from language: String?) -> String {
        let normalized = language?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() ?? ""
        switch normalized {
        case "zh", "en":
            return normalized
        default:
            return "auto"
        }
    }

    static func openAITranscriptionMultipartFields(
        model: String,
        hintPayload: ResolvedASRHintPayload
    ) -> [String: String] {
        let effectiveModel = model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? RemoteASRProvider.openAIWhisper.suggestedModel
            : model
        var fields: [String: String] = [
            "response_format": "json"
        ]
        if let language = hintPayload.language?.trimmingCharacters(in: .whitespacesAndNewlines), !language.isEmpty {
            fields["language"] = language
        }
        if effectiveModel != "gpt-4o-transcribe-diarize",
           let prompt = hintPayload.prompt?.trimmingCharacters(in: .whitespacesAndNewlines),
           !prompt.isEmpty {
            fields["prompt"] = prompt
        }
        return fields
    }

    static func extractTextFragment(fromLine line: String) -> String? {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        guard let data = line.data(using: .utf8) else {
            return trimmed
        }

        if let object = try? JSONSerialization.jsonObject(with: data) {
            if let value = extractText(in: object), !value.isEmpty {
                return normalizedTextFragment(value)
            }
            return nil
        }

        if let loose = extractLooseTextField(from: trimmed), !loose.isEmpty {
            return normalizedTextFragment(loose)
        }

        if (trimmed.hasPrefix("{") && trimmed.hasSuffix("}")) ||
            (trimmed.hasPrefix("[") && trimmed.hasSuffix("]")) {
            return nil
        }

        return normalizedTextFragment(trimmed)
    }

    static func extractStreamErrorMessage(fromLine line: String) -> String? {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        guard let data = trimmed.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) else {
            return nil
        }
        return extractStreamErrorMessage(in: object)
    }

    static func extractLooseTextField(from line: String) -> String? {
        let patterns = [
            #"(?:["']?text["']?\s*:\s*["'])([^"']+)(?:["'])"#,
            #"(?:["']?transcript["']?\s*:\s*["'])([^"']+)(?:["'])"#,
            #"(?:["']?result_text["']?\s*:\s*["'])([^"']+)(?:["'])"#,
            #"(?:["']?text["']?\s*:\s*)([^,}\]]+)"#,
            #"(?:["']?transcript["']?\s*:\s*)([^,}\]]+)"#,
            #"(?:["']?result_text["']?\s*:\s*)([^,}\]]+)"#
        ]

        for pattern in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else {
                continue
            }
            let range = NSRange(line.startIndex..<line.endIndex, in: line)
            guard let match = regex.firstMatch(in: line, options: [], range: range),
                  match.numberOfRanges > 1,
                  let valueRange = Range(match.range(at: 1), in: line) else {
                continue
            }
            var value = String(line[valueRange]).trimmingCharacters(in: .whitespacesAndNewlines)
            if (value.hasPrefix("\"") && value.hasSuffix("\"")) ||
                (value.hasPrefix("'") && value.hasSuffix("'")) {
                value.removeFirst()
                value.removeLast()
                value = value.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            if !value.isEmpty {
                return value
            }
        }
        return nil
    }

    static func normalizedTextFragment(_ raw: String) -> String? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        if isLikelyJSONObjectString(trimmed) {
            if let data = trimmed.data(using: .utf8),
               let object = try? JSONSerialization.jsonObject(with: data),
               let nested = extractText(in: object),
               !nested.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
               !isLikelyJSONObjectString(nested) {
                return nested.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            if let loose = extractLooseTextField(from: trimmed),
               !isLikelyJSONObjectString(loose) {
                return loose.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            return nil
        }

        return trimmed
    }

    static func isLikelyJSONObjectString(_ value: String) -> Bool {
        (value.hasPrefix("{") && value.hasSuffix("}")) ||
        (value.hasPrefix("[") && value.hasSuffix("]"))
    }

    static func extractDoubaoText(in object: Any) -> String? {
        if let dict = object as? [String: Any],
           let result = dict["result"] as? [String: Any],
           let text = result["text"] as? String {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty, !RemoteASRTextSanitizer.isLikelyIdentifierText(trimmed) {
                return trimmed
            }
        }

        var candidates: [String] = []

        func appendCandidate(_ value: String) {
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, !RemoteASRTextSanitizer.isLikelyIdentifierText(trimmed) else { return }
            candidates.append(trimmed)
        }

        func walk(_ node: Any) {
            if let dict = node as? [String: Any] {
                let directTextKeys = ["text", "transcript", "utterance", "utterance_text", "result_text"]
                for key in directTextKeys {
                    if let value = dict[key] as? String {
                        appendCandidate(value)
                    }
                }

                let containerKeys = ["result", "results", "utterances", "payload_msg", "payload", "data", "nbest", "alternatives"]
                for key in containerKeys {
                    if let value = dict[key] {
                        walk(value)
                    }
                }

                for (_, value) in dict {
                    if value is [String: Any] || value is [Any] {
                        walk(value)
                    }
                }
                return
            }

            if let array = node as? [Any] {
                for item in array {
                    walk(item)
                }
            }
        }

        walk(object)
        return candidates.max(by: { $0.count < $1.count })
    }

    static func extractText(in object: Any) -> String? {
        if let text = object as? String {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return nil }
            if isLikelyJSONObjectString(trimmed) {
                if let data = trimmed.data(using: .utf8),
                   let nestedObject = try? JSONSerialization.jsonObject(with: data),
                   let nestedText = extractText(in: nestedObject),
                   !nestedText.isEmpty {
                    return nestedText
                }
                if let loose = extractLooseTextField(from: trimmed), !loose.isEmpty {
                    return loose
                }
                return nil
            }
            return trimmed
        }
        if let dict = object as? [String: Any] {
            let preferredKeys = ["delta", "text", "transcript", "result_text", "content", "utterance", "data"]
            for key in preferredKeys {
                if let value = dict[key], let text = extractText(in: value), !text.isEmpty {
                    return text
                }
            }
            for value in dict.values {
                if (value is [String: Any] || value is [Any]),
                   let text = extractText(in: value),
                   !text.isEmpty {
                    return text
                }
            }
        }
        if let array = object as? [Any] {
            for item in array {
                if let text = extractText(in: item), !text.isEmpty {
                    return text
                }
            }
        }
        return nil
    }

    private static func extractStreamErrorMessage(in object: Any) -> String? {
        if let dict = object as? [String: Any] {
            if let value = dict["error"],
               let message = extractStreamErrorDescription(from: value) {
                return message
            }

            let markerKeys = ["event", "type", "status"]
            let isErrorPayload = markerKeys.contains { key in
                guard let value = dict[key] as? String else { return false }
                let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                return ["error", "failed", "failure"].contains(normalized)
            }
            if isErrorPayload {
                let messageKeys = ["message", "msg", "error_message", "detail", "code"]
                for key in messageKeys {
                    if let value = dict[key],
                       let message = extractStreamErrorDescription(from: value) {
                        return message
                    }
                }
                return "StepFun ASR stream returned an error event."
            }
        }

        return nil
    }

    private static func extractStreamErrorDescription(from object: Any) -> String? {
        if let text = object as? String {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        }

        if let number = object as? NSNumber {
            return number.stringValue
        }

        if let dict = object as? [String: Any] {
            let preferredKeys = ["message", "msg", "error_message", "detail", "code"]
            for key in preferredKeys {
                if let value = dict[key],
                   let message = extractStreamErrorDescription(from: value) {
                    return message
                }
            }
        }

        if let array = object as? [Any] {
            for item in array {
                if let message = extractStreamErrorDescription(from: item) {
                    return message
                }
            }
        }

        return nil
    }

    static func mergeStreamFragment(current: String, incoming: String) -> String {
        let fragment = incoming.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !fragment.isEmpty else { return current }
        if current.isEmpty { return fragment }
        if fragment == current { return current }
        if fragment.hasPrefix(current) { return fragment }
        if current.hasPrefix(fragment) { return current }
        if fragment.contains(current) { return fragment }
        if current.contains(fragment) { return current }

        // Streaming deltas overlap only at their boundary. Bounding the scan avoids
        // quadratic work as a long transcript grows while preserving ample context.
        let maxOverlap = min(min(current.count, fragment.count), 512)
        if maxOverlap > 0 {
            for length in stride(from: maxOverlap, through: 1, by: -1) {
                let currentSuffix = String(current.suffix(length))
                let incomingPrefix = String(fragment.prefix(length))
                if currentSuffix == incomingPrefix {
                    return current + fragment.dropFirst(length)
                }
            }
        }
        return current + fragment
    }

    static func collectText(from bytes: URLSession.AsyncBytes) async throws -> String {
        var chunks: [String] = []
        for try await line in bytes.lines {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                chunks.append(trimmed)
            }
            if chunks.count >= 6 { break }
        }
        return chunks.joined(separator: " | ")
    }
}
