// RemoteLLMStreamingParser.swift
// Provides Remote LLMStreaming Parser for LLM requests and response handling.

import Foundation

extension RemoteLLMRuntimeClient {
    func extractPrimaryText(from object: Any) -> String? {
        if let dict = object as? [String: Any] {
            if isReasoningContainer(dict) {
                return nil
            }
            if let outputText = extractTextValue(from: dict["output_text"]) {
                return outputText
            }
            if let response = dict["response"] {
                return extractPrimaryText(from: response)
            }
            if let output = dict["output"] as? [Any],
               let outputText = extractTextFromResponsesOutput(output) {
                return outputText
            }
            if let contentText = extractTextFromMessageContent(dict["content"]) {
                return contentText
            }
            if let text = extractTextValue(from: dict["text"]),
               shouldTreatDirectTextFieldAsPrimary(in: dict) {
                return text
            }
            if let result = dict["result"],
               let text = extractPrimaryText(from: result) {
                return text
            }
            if let data = dict["data"],
               let text = extractPrimaryText(from: data) {
                return text
            }
            if let message = dict["message"] as? [String: Any] {
                if let value = extractMessageContent(from: message["content"]) {
                    return value
                }
                if let text = extractPrimaryText(from: message) {
                    return text
                }
            }
            if let choices = dict["choices"] as? [Any],
               let first = choices.first as? [String: Any] {
                if let message = first["message"] as? [String: Any] {
                    if isReasoningContainer(message) {
                        return nil
                    }
                    if let value = extractMessageContent(from: message["content"]) {
                        return value
                    }
                }
                if let delta = first["delta"] as? [String: Any],
                   let value = extractStreamingMessageContent(from: delta["content"]) {
                    return value
                }
                if let text = first["text"] as? String, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    return text
                }
            }
            if let candidates = dict["candidates"] as? [[String: Any]],
               let first = candidates.first,
               let content = first["content"] as? [String: Any],
               let parts = content["parts"] as? [[String: Any]] {
                for part in parts {
                    if let text = extractTextValue(from: part["text"]) {
                        return text
                    }
                }
            }
            if let reply = dict["reply"] as? String, !reply.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return reply
            }
            if let response = dict["response"] as? String, !response.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return response
            }
        }
        if let array = object as? [Any] {
            for item in array {
                if let text = extractPrimaryText(from: item) {
                    return text
                }
            }
        }
        return nil
    }

    func extractStreamingDelta(from object: Any) -> String? {
        guard let dict = object as? [String: Any] else { return nil }

        if let type = dict["type"] as? String {
            if type == "response.output_text.delta",
               let delta = dict["delta"] as? String,
               !delta.isEmpty {
                return delta
            }
            if (type == "response.output_text" || type == "response.output_text.done"),
               let text = dict["text"] as? String,
               !text.isEmpty {
                return text
            }
        }

        if let type = dict["type"] as? String,
           type == "content_block_delta",
           let delta = dict["delta"] as? [String: Any],
           (delta["type"] as? String) == "text_delta",
           let text = delta["text"] as? String,
           !text.isEmpty {
            return text
        }

        if let choices = dict["choices"] as? [[String: Any]],
           let first = choices.first,
           let delta = first["delta"] as? [String: Any] {
            if let content = extractStreamingMessageContent(from: delta["content"]) {
                return content
            }
            if let text = delta["text"] as? String, !text.isEmpty {
                return text
            }
        }

        if let delta = dict["delta"] as? [String: Any] {
            if let text = delta["text"] as? String, !text.isEmpty {
                return text
            }
            if let content = extractStreamingMessageContent(from: delta["content"]) {
                return content
            }
        }

        if let message = dict["message"] as? [String: Any],
           let content = extractStreamingMessageContent(from: message["content"]) {
            return content
        }

        if let reply = dict["reply"] as? String, !reply.isEmpty {
            return reply
        }

        if let response = dict["response"] as? String, !response.isEmpty {
            return response
        }

        return nil
    }

    func decodeResponsesObject(from data: Data, response: HTTPURLResponse) throws -> [String: Any] {
        do {
            guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw responsesDecodeError(
                    data: data,
                    response: response,
                    fallbackDetail: "Responses API returned JSON that is not an object."
                )
            }
            return object
        } catch let error as NSError where error.domain == "Voxt.RemoteLLM" {
            throw error
        } catch {
            throw responsesDecodeError(
                data: data,
                response: response,
                fallbackDetail: error.localizedDescription
            )
        }
    }

    private func responsesDecodeError(
        data: Data,
        response: HTTPURLResponse,
        fallbackDetail: String
    ) -> NSError {
        let contentType = response.value(forHTTPHeaderField: "Content-Type")?
            .lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let preview = String(data: data.prefix(1200), encoding: .utf8) ?? "<non-utf8>"
        let trimmedPreview = preview.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()

        let message: String
        if contentType.contains("text/html") ||
            trimmedPreview.hasPrefix("<!doctype html") ||
            trimmedPreview.hasPrefix("<html") {
            message = "Remote LLM Responses endpoint returned HTML instead of JSON. Check the endpoint URL; it must be an API route compatible with OpenAI Responses, not a gateway console or web page."
        } else if contentType.contains("text/event-stream") ||
                    trimmedPreview.hasPrefix("data:") ||
                    trimmedPreview.hasPrefix("event:") {
            message = "Remote LLM Responses endpoint returned an event stream for a non-streaming request. The API must return a JSON object when stream is false."
        } else {
            message = "Remote LLM Responses endpoint returned invalid JSON: \(fallbackDetail)"
        }

        return NSError(
            domain: "Voxt.RemoteLLM",
            code: -309,
            userInfo: [NSLocalizedDescriptionKey: message]
        )
    }

    func mergedStreamingSnapshot(current: String, next: String) -> String {
        let trimmedNext = next.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedNext.isEmpty else { return current }
        guard !current.isEmpty else { return trimmedNext }

        if trimmedNext.hasPrefix(current) {
            return trimmedNext
        }

        if current.hasPrefix(trimmedNext) {
            return current
        }

        if let overlap = longestSuffixPrefixOverlap(lhs: current, rhs: trimmedNext), overlap > 0 {
            return current + trimmedNext.dropFirst(overlap)
        }

        return current + trimmedNext
    }

    func shouldFlushBufferedEventLines(_ bufferedEventLines: [String]) -> Bool {
        let payload = bufferedEventLines
            .joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !payload.isEmpty else { return false }
        if payload == "[DONE]" { return true }
        guard let first = payload.first, first == "{" || first == "[" else { return false }
        guard let boundary = completeJSONBoundary(in: payload[...]) else { return false }
        return boundary == payload.endIndex
    }

    func drainNonEventStreamPayloads(buffer: inout String) -> [String] {
        var payloads: [String] = []

        while true {
            let trimmedBuffer = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmedBuffer.isEmpty else {
                buffer = ""
                break
            }

            if trimmedBuffer.hasPrefix("[DONE]") {
                payloads.append("[DONE]")
                if let doneRange = buffer.range(of: "[DONE]") {
                    buffer.removeSubrange(doneRange)
                } else {
                    buffer = ""
                }
                continue
            }

            let workingStart = buffer.firstIndex(where: { !$0.isWhitespace && !$0.isNewline })
            guard let workingStart else {
                buffer = ""
                break
            }

            let working = buffer[workingStart...]
            guard let first = working.first else {
                buffer = ""
                break
            }

            if first == "{" || first == "[" {
                guard let end = completeJSONBoundary(in: working) else {
                    break
                }
                let payload = String(working[..<end])
                payloads.append(payload)
                buffer.removeSubrange(workingStart..<end)
                continue
            }

            guard let newline = working.firstIndex(of: "\n") else {
                break
            }
            let payload = String(working[..<newline]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !payload.isEmpty {
                payloads.append(payload)
            }
            buffer.removeSubrange(workingStart...newline)
        }

        return payloads
    }

    func completeJSONBoundary(in text: Substring) -> String.Index? {
        var depth = 0
        var insideString = false
        var escaping = false
        var started = false

        for index in text.indices {
            let character = text[index]

            if insideString {
                if escaping {
                    escaping = false
                    continue
                }
                if character == "\\" {
                    escaping = true
                    continue
                }
                if character == "\"" {
                    insideString = false
                }
                continue
            }

            if character == "\"" {
                insideString = true
                continue
            }

            if character == "{" || character == "[" {
                depth += 1
                started = true
            } else if character == "}" || character == "]" {
                depth -= 1
                if started && depth == 0 {
                    return text.index(after: index)
                }
            }
        }

        return nil
    }

    func longestSuffixPrefixOverlap(lhs: String, rhs: String) -> Int? {
        let maxOverlap = min(lhs.count, rhs.count)
        guard maxOverlap > 0 else { return nil }
        for length in stride(from: maxOverlap, through: 1, by: -1) {
            let lhsSuffix = lhs.suffix(length)
            let rhsPrefix = rhs.prefix(length)
            if lhsSuffix == rhsPrefix {
                return length
            }
        }
        return nil
    }

    func extractMessageContent(from value: Any?) -> String? {
        if let text = extractTextValue(from: value) {
            return text
        }
        if let blocks = value as? [Any],
           let merged = mergeTextSegments(from: blocks.compactMap(extractPrimaryText(from:))) {
            return merged
        }
        return nil
    }

    func extractStreamingMessageContent(from value: Any?) -> String? {
        if let text = value as? String {
            return text.isEmpty ? nil : text
        }
        if let blocks = value as? [[String: Any]] {
            let texts = blocks.compactMap { block -> String? in
                extractTextValue(from: block["text"])
            }
            let merged = texts.joined(separator: "\n")
            return merged.isEmpty ? nil : merged
        }
        return nil
    }

    func recoverStreamingDelta(fromRawPayload text: String) -> String? {
        if let envelopeFragment = extractMalformedStreamingFragment(
            from: text,
            markers: [
                #""content":"#,
                #""text":"#,
                #""response":"#
            ]
        ) {
            let normalized = decodeStreamingJSONStringFragment(envelopeFragment)
            if !normalized.isEmpty {
                return normalized
            }
        }

        let patterns = [
            #""delta"\s*:\s*\{\s*"content"\s*:\s*"((?:\\.|[^"\\])*)""#,
            #""delta"\s*:\s*\{\s*"text"\s*:\s*"((?:\\.|[^"\\])*)""#,
            #""response"\s*:\s*"((?:\\.|[^"\\])*)""#,
            #""message"\s*:\s*\{\s*"[^"]*"\s*:\s*"[^"]*"\s*,\s*"content"\s*:\s*"((?:\\.|[^"\\])*)""#,
            #""message"\s*:\s*\{[\s\S]*?"content"\s*:\s*"((?:\\.|[^"\\])*)""#,
            #""content"\s*:\s*"((?:\\.|[^"\\])*)""#
        ]

        for pattern in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else {
                continue
            }
            let range = NSRange(text.startIndex..<text.endIndex, in: text)
            guard let match = regex.firstMatch(in: text, options: [], range: range),
                  match.numberOfRanges > 1,
                  let valueRange = Range(match.range(at: 1), in: text) else {
                continue
            }

            let rawValue = String(text[valueRange])
            let normalized = decodeStreamingJSONStringFragment(rawValue)
            if !normalized.isEmpty {
                return normalized
            }
        }

        return nil
    }

    func extractMalformedStreamingFragment(from text: String, markers: [String]) -> String? {
        let lowercased = text.lowercased()
        let lowercasedMarkers = markers.map { $0.lowercased() }
        let suffixes = [
            #","role":"#,
            #""},"finish_reason":"#,
            #""},"index":"#,
            #""},"logprobs":"#,
            #""},"object":"#,
            #""},"usage":"#,
            #""},"created":"#,
            #""},"system_fingerprint":"#,
            #""},"model":"#,
            #""},"id":"#,
            #""}],"object":"#
        ]

        for marker in lowercasedMarkers {
            guard let markerRange = lowercased.range(of: marker) else { continue }
            let contentStart = markerRange.upperBound
            let remainder = lowercased[contentStart...]

            let suffixStart = suffixes
                .compactMap { suffix in
                    remainder.range(of: suffix).map { $0.lowerBound }
                }
                .min() ?? lowercased.endIndex

            guard suffixStart > contentStart else { continue }

            var fragment = String(text[contentStart..<suffixStart])
            if hasOddUnescapedQuoteCount(fragment), fragment.hasSuffix("\"") {
                fragment.removeLast()
            }
            if !fragment.isEmpty {
                return fragment
            }
        }

        return nil
    }

    func extractTextValue(from value: Any?) -> String? {
        if let text = value as? String {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        }
        if let dict = value as? [String: Any] {
            if let text = dict["text"] as? String {
                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { return trimmed }
            }
            if let valueText = dict["value"] as? String {
                let trimmed = valueText.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { return trimmed }
            }
        }
        return nil
    }

    func extractTextFromResponsesOutput(_ output: [Any]) -> String? {
        var texts: [String] = []
        for item in output {
            guard let dict = item as? [String: Any] else { continue }
            let type = (dict["type"] as? String)?.lowercased()
            if type == "reasoning" || type == "thinking" {
                continue
            }

            if let contentText = extractTextFromMessageContent(dict["content"]) {
                texts.append(contentText)
                continue
            }
            if shouldTreatDirectTextFieldAsPrimary(in: dict),
               let text = extractTextValue(from: dict["text"]) {
                texts.append(text)
            }
        }
        return mergeTextSegments(from: texts)
    }

    func responsesCompletionIssue(from object: [String: Any]) -> String? {
        guard let rawStatus = object["status"] as? String else { return nil }
        let status = rawStatus.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard status != "completed" else { return nil }

        let reason = (object["incomplete_details"] as? [String: Any])?["reason"] as? String
        let normalizedReason = reason?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return normalizedReason.isEmpty
            ? "Responses API returned status '\(rawStatus)'."
            : "Responses API returned status '\(rawStatus)' (\(normalizedReason))."
    }

    func isValidStructuredJSONObject(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = trimmed.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data),
              object is [String: Any] else {
            return false
        }
        return true
    }

    func extractTextFromMessageContent(_ value: Any?) -> String? {
        if let text = extractTextValue(from: value) {
            return text
        }
        if let blocks = value as? [Any] {
            var texts: [String] = []
            for blockValue in blocks {
                if let block = blockValue as? [String: Any] {
                    if isReasoningContainer(block) {
                        continue
                    }
                    let type = (block["type"] as? String)?.lowercased()
                    if type == nil || type == "text" || type == "output_text" {
                        if let text = extractTextValue(from: block["text"]) {
                            texts.append(text)
                            continue
                        }
                    }
                    if let nestedText = extractPrimaryText(from: block) {
                        texts.append(nestedText)
                    }
                } else if let text = extractTextValue(from: blockValue) {
                    texts.append(text)
                }
            }
            return mergeTextSegments(from: texts)
        }
        return nil
    }

    func mergeTextSegments(from segments: [String]) -> String? {
        let merged = segments
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return merged.isEmpty ? nil : merged
    }

    func shouldTreatDirectTextFieldAsPrimary(in dict: [String: Any]) -> Bool {
        if let type = (dict["type"] as? String)?.lowercased() {
            return type == "text" || type == "output_text" || type == "message"
        }
        return dict["content"] == nil && dict["choices"] == nil && dict["output"] == nil
    }

    private func isReasoningContainer(_ dict: [String: Any]) -> Bool {
        if let type = (dict["type"] as? String)?.lowercased(),
           type == "reasoning" || type == "thinking" || type == "summary_text" {
            return true
        }
        if dict["reasoning_content"] != nil && dict["content"] == nil {
            return true
        }
        return false
    }

    func decodeStreamingJSONStringFragment(_ value: String) -> String {
        let wrapped = "\"\(value)\""
        if let data = wrapped.data(using: .utf8),
           let decoded = try? JSONDecoder().decode(String.self, from: data) {
            return decoded
        }

        return value
            .replacingOccurrences(of: #"\\n"#, with: "\n", options: .regularExpression)
            .replacingOccurrences(of: #"\\r"#, with: "\r", options: .regularExpression)
            .replacingOccurrences(of: #"\\t"#, with: "\t", options: .regularExpression)
            .replacingOccurrences(of: #"\\\""#, with: "\"", options: .regularExpression)
            .replacingOccurrences(of: #"\\\\"#, with: "\\", options: .regularExpression)
    }

    func hasOddUnescapedQuoteCount(_ text: String) -> Bool {
        var escaping = false
        var quoteCount = 0

        for character in text {
            if escaping {
                escaping = false
                continue
            }
            if character == "\\" {
                escaping = true
                continue
            }
            if character == "\"" {
                quoteCount += 1
            }
        }

        return quoteCount % 2 == 1
    }

    func looksLikeStreamingEnvelopeFragment(_ text: String) -> Bool {
        let lowered = text.lowercased()
        if lowered == "[done]" || lowered.hasPrefix("data: [done]") {
            return true
        }

        let markers = [
            "chat.completion.chunk",
            "\"choices\"",
            "\"delta\"",
            "\"finish_reason\"",
            "\"index\"",
            "\"object\"",
            "\"id\"",
            "\"model\"",
            "\"usage\"",
            "\"created\"",
            "\"system_fingerprint\""
        ]

        return markers.contains(where: { lowered.contains($0) })
    }

    func extractStreamingErrorMessage(from object: Any) -> String? {
        guard let dict = object as? [String: Any] else { return nil }

        if let error = dict["error"] as? String,
           !error.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return error
        }

        if let error = dict["error"] as? [String: Any] {
            if let message = error["message"] as? String,
               !message.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return message
            }
            if let detail = error["detail"] as? String,
               !detail.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return detail
            }
            if let code = error["code"] {
                return "Remote LLM stream error: \(code)"
            }
            return "Remote LLM stream error."
        }

        if let message = dict["message"] as? String,
           (dict["status"] as? String)?.lowercased() == "error" ||
           (dict["type"] as? String)?.lowercased().contains("error") == true {
            return message
        }

        return nil
    }

    func responsePayloadPreview(from data: Data, limit: Int = 1200) -> String {
        let decoded = String(data: data, encoding: .utf8)
            ?? String(data: data, encoding: .utf16)
            ?? String(data: data, encoding: .utf16LittleEndian)
            ?? String(data: data, encoding: .utf16BigEndian)
            ?? String(data: data, encoding: .isoLatin1)

        guard let decoded else {
            let hex = data.prefix(48)
                .map { String(format: "%02x", $0) }
                .joined(separator: " ")
            return "<non-text bytes: \(hex)>"
        }

        return String(decoded.prefix(limit))
    }
}
