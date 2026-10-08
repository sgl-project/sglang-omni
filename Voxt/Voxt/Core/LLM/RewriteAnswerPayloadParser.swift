// RewriteAnswerPayloadParser.swift
// Provides Rewrite Answer Payload Parser for LLM requests and response handling.

import Foundation

enum RewriteAnswerPayloadParser {
    nonisolated private static let fallbackTitle = AppLocalization.localizedString("AI Answer")

    nonisolated static func normalize(_ payload: RewriteAnswerPayload) -> RewriteAnswerPayload {
        let normalizedTitle = payload.trimmedTitle.isEmpty ? fallbackTitle : payload.trimmedTitle
        let normalizedContent = sanitizeStructuredCandidate(
            normalizedStreamChunkEnvelope(in: payload.trimmedContent) ?? payload.trimmedContent
        )
        guard !normalizedContent.isEmpty else {
            return RewriteAnswerPayload(title: normalizedTitle, content: normalizedContent)
        }

        if let nestedPayload = extract(from: normalizedContent, fallbackTitle: normalizedTitle) {
            let nestedContent = nestedPayload.trimmedContent
            if !nestedContent.isEmpty {
                return RewriteAnswerPayload(
                    title: nestedPayload.trimmedTitle.isEmpty ? normalizedTitle : nestedPayload.trimmedTitle,
                    content: nestedContent
                )
            }
        }

        return RewriteAnswerPayload(title: normalizedTitle, content: normalizedContent)
    }

    nonisolated static func extract(
        from text: String,
        fallbackTitle: String = AppLocalization.localizedString("AI Answer")
    ) -> RewriteAnswerPayload? {
        let trimmed = sanitizeStructuredCandidate(
            normalizedStreamChunkEnvelope(in: text) ?? text
        ).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        let candidateStrings = [trimmed, strippedCodeFencePayload(from: trimmed)]
            .compactMap { $0?.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        for candidate in candidateStrings {
            if let payload = decodeRewriteAnswerPayload(from: candidate, fallbackTitle: fallbackTitle) {
                return payload
            }

            if let openingBrace = candidate.firstIndex(of: "{") {
                let objectString = String(candidate[openingBrace...])
                if let payload = decodeRewriteAnswerPayload(from: objectString, fallbackTitle: fallbackTitle) {
                    return payload
                }
            }
        }

        return nil
    }

    nonisolated private static func normalizedStreamChunkEnvelope(in text: String) -> String? {
        let lines = text
            .replacingOccurrences(of: "\r\n", with: "\n")
            .components(separatedBy: "\n")

        var aggregated = ""
        var matchedChunkLine = false

        for rawLine in lines {
            var line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !line.isEmpty else { continue }

            if line == "[DONE]" {
                matchedChunkLine = true
                continue
            }

            if line.hasPrefix("data:") {
                matchedChunkLine = true
                line.removeFirst(5)
                if line.hasPrefix(" ") {
                    line.removeFirst()
                }
            }

            let streamingLikeLine = matchedChunkLine || looksLikeStreamingEnvelopeFragment(line)

            guard let data = line.data(using: .utf8),
                  let object = try? JSONSerialization.jsonObject(with: data) else {
                if streamingLikeLine,
                   let recovered = recoveredStreamingChunkContent(fromRawLine: line) {
                    aggregated.append(recovered)
                    matchedChunkLine = true
                    continue
                }
                if streamingLikeLine {
                    matchedChunkLine = true
                    continue
                }
                if matchedChunkLine {
                    continue
                }
                return nil
            }

            if let content = streamingChunkContent(from: object) {
                aggregated.append(content)
                matchedChunkLine = true
                continue
            }

            if isStreamingTerminalChunk(object) {
                matchedChunkLine = true
                continue
            }

            if matchedChunkLine {
                continue
            }
            return nil
        }

        let normalized = aggregated.trimmingCharacters(in: .whitespacesAndNewlines)
        return matchedChunkLine && !normalized.isEmpty ? normalized : nil
    }

    nonisolated private static func recoveredStreamingChunkContent(fromRawLine text: String) -> String? {
        if let envelopeFragment = extractMalformedStreamingFragment(
            from: text,
            markers: [
                #""content":"#,
                #""text":"#
            ]
        ) {
            let decoded = decodeStreamingJSONStringFragment(envelopeFragment)
            if !decoded.isEmpty {
                return decoded
            }
        }

        let patterns = [
            #""delta"\s*:\s*\{\s*"content"\s*:\s*"((?:\\.|[^"\\])*)""#,
            #""delta"\s*:\s*\{\s*"text"\s*:\s*"((?:\\.|[^"\\])*)""#,
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

            let decoded = decodeStreamingJSONStringFragment(String(text[valueRange]))
            if !decoded.isEmpty {
                return decoded
            }
        }

        return nil
    }

    nonisolated private static func extractMalformedStreamingFragment(from text: String, markers: [String]) -> String? {
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

    nonisolated private static func decodeStreamingJSONStringFragment(_ value: String) -> String {
        let repairedValue = repairedMalformedStreamingFragment(value)
        let wrapped = "\"\(repairedValue)\""
        if let data = wrapped.data(using: .utf8),
           let decoded = try? JSONDecoder().decode(String.self, from: data) {
            return decoded
        }

        return repairedValue
            .replacingOccurrences(of: #"\\n"#, with: "\n", options: .regularExpression)
            .replacingOccurrences(of: #"\\r"#, with: "\r", options: .regularExpression)
            .replacingOccurrences(of: #"\\t"#, with: "\t", options: .regularExpression)
            .replacingOccurrences(of: #"\\\""#, with: "\"", options: .regularExpression)
            .replacingOccurrences(of: #"\\\\"#, with: "\\", options: .regularExpression)
    }

    nonisolated private static func repairedMalformedStreamingFragment(_ value: String) -> String {
        var repaired = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !repaired.isEmpty else { return "" }

        let metadataMarkers = [
            "},\"finish_reason\"",
            "},\"index\"",
            "},\"object\"",
            "},\"usage\"",
            "},\"created\"",
            "},\"system_fingerprint\"",
            "},\"model\"",
            "},\"id\"",
            "},\"logprobs\"",
            "}],\"object\""
        ]

        if let markerRange = metadataMarkers
            .compactMap({ repaired.range(of: $0) })
            .min(by: { $0.lowerBound < $1.lowerBound }) {
            repaired = String(repaired[..<markerRange.lowerBound])
        }

        repaired = repaired.trimmingCharacters(in: .whitespacesAndNewlines)
        repaired = repaired.replacingOccurrences(
            of: #"",\s*""$"#,
            with: "",
            options: .regularExpression
        )

        if repaired.hasSuffix("\"}\"") {
            repaired.removeLast(3)
        } else if repaired.hasSuffix("\"}") || repaired.hasSuffix("}\"") {
            repaired.removeLast(2)
        }

        repaired = repaired.replacingOccurrences(
            of: #"(:\s*)""$"#,
            with: #"$1""#,
            options: .regularExpression
        )

        return repaired.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    nonisolated private static func hasOddUnescapedQuoteCount(_ text: String) -> Bool {
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

    nonisolated private static func looksLikeStreamingEnvelopeFragment(_ text: String) -> Bool {
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

    nonisolated private static func sanitizeStructuredCandidate(_ text: String) -> String {
        var cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else { return "" }

        if let regex = try? NSRegularExpression(pattern: "<think>[\\s\\S]*?</think>", options: [.caseInsensitive]) {
            let range = NSRange(cleaned.startIndex..<cleaned.endIndex, in: cleaned)
            cleaned = regex.stringByReplacingMatches(in: cleaned, options: [], range: range, withTemplate: "")
        }

        cleaned = cleaned
            .replacingOccurrences(of: "<think>", with: "", options: .caseInsensitive)
            .replacingOccurrences(of: "</think>", with: "", options: .caseInsensitive)
            .trimmingCharacters(in: .whitespacesAndNewlines)

        if let openingBrace = cleaned.firstIndex(of: "{"),
           openingBrace > cleaned.startIndex {
            let prefix = cleaned[..<openingBrace].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if prefix.contains("reasoning") || prefix.contains("thought") || prefix.contains("思考") {
                cleaned = String(cleaned[openingBrace...])
            }
        }

        if let keyRange = cleaned.range(
            of: #"(?:["']?(?:title|heading|summary|content|answer|body|text)["']?\s*[:：])"#,
            options: .regularExpression
        ) {
            let prefix = cleaned[..<keyRange.lowerBound]
            let shellCharacters = CharacterSet(charactersIn: " \t\r\n{}[]\"',")
            if !prefix.isEmpty,
               prefix.unicodeScalars.allSatisfy({ shellCharacters.contains($0) }) {
                cleaned = String(cleaned[keyRange.lowerBound...])
            }
        }

        return cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    nonisolated private static func streamingChunkContent(from object: Any) -> String? {
        guard let dict = object as? [String: Any] else { return nil }

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
            if let content = delta["content"] as? String, !content.isEmpty {
                return content
            }
            if let text = delta["text"] as? String, !text.isEmpty {
                return text
            }
        }

        if let delta = dict["delta"] as? [String: Any] {
            if let content = delta["content"] as? String, !content.isEmpty {
                return content
            }
            if let text = delta["text"] as? String, !text.isEmpty {
                return text
            }
        }

        if let message = dict["message"] as? [String: Any],
           let content = message["content"] as? String,
           !content.isEmpty {
            return content
        }

        return nil
    }

    nonisolated private static func isStreamingTerminalChunk(_ object: Any) -> Bool {
        guard let dict = object as? [String: Any] else { return false }
        if let choices = dict["choices"] as? [[String: Any]],
           let first = choices.first,
           let finishReason = first["finish_reason"] as? String,
           !finishReason.isEmpty {
            return true
        }
        return false
    }

    nonisolated private static func strippedCodeFencePayload(from text: String) -> String? {
        guard text.hasPrefix("```"), let closingRange = text.range(of: "```", options: .backwards), closingRange.lowerBound > text.startIndex else {
            return nil
        }

        guard let openingNewline = text.firstIndex(of: "\n") else { return nil }

        let bodyStart = text.index(after: openingNewline)
        guard bodyStart < closingRange.lowerBound else { return nil }

        let inner = text[bodyStart..<closingRange.lowerBound]
        guard !inner.isEmpty else { return nil }
        return String(inner)
    }

    nonisolated private static func decodeRewriteAnswerPayload(
        from text: String,
        fallbackTitle: String
    ) -> RewriteAnswerPayload? {
        guard let data = text.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data)
        else {
            return decodeLooseRewriteAnswerPayload(from: text, fallbackTitle: fallbackTitle)
        }

        if let dict = object as? [String: Any] {
            return rewriteAnswerPayload(from: dict, fallbackTitle: fallbackTitle)
        }

        if let string = object as? String {
            let trimmed = string.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, trimmed != text else {
                return decodeLooseRewriteAnswerPayload(from: text, fallbackTitle: fallbackTitle)
            }
            return decodeRewriteAnswerPayload(from: trimmed, fallbackTitle: fallbackTitle) ??
                decodeLooseRewriteAnswerPayload(from: trimmed, fallbackTitle: fallbackTitle)
        }

        return decodeLooseRewriteAnswerPayload(from: text, fallbackTitle: fallbackTitle)
    }

    nonisolated private static func rewriteAnswerPayload(
        from object: [String: Any],
        fallbackTitle: String
    ) -> RewriteAnswerPayload? {
        let titleKeys = ["title", "heading", "summary"]
        let contentKeys = ["content", "answer", "body", "text"]

        let title = titleKeys
            .compactMap { object[$0] }
            .map { String(describing: $0).trimmingCharacters(in: .whitespacesAndNewlines) }
            .first(where: { !$0.isEmpty }) ?? fallbackTitle

        let content = contentKeys
            .compactMap { object[$0] }
            .map { String(describing: $0).trimmingCharacters(in: .whitespacesAndNewlines) }
            .first(where: { !$0.isEmpty }) ?? ""

        guard !content.isEmpty else {
            return decodeLooseRewriteAnswerPayload(
                from: object.map { "\($0.key): \($0.value)" }.joined(separator: "\n"),
                fallbackTitle: fallbackTitle
            )
        }

        return RewriteAnswerPayload(title: title, content: content)
    }

    nonisolated private static func decodeLooseRewriteAnswerPayload(
        from text: String,
        fallbackTitle: String
    ) -> RewriteAnswerPayload? {
        let normalized = text
            .replacingOccurrences(of: "\r\n", with: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return nil }

        let title = firstMatch(
            in: normalized,
            patterns: [
                #"(?is)["']?title["']?\s*[:：]\s*["'](.+?)["']\s*,\s*["']?(?:content|answer|body|text)["']?\s*[:：]"#,
                #"(?is)(?:^|\n|\{)\s*["']?title["']?\s*[:：]\s*["']?(.+?)["']?(?=\s*(?:,\s*["']?(?:content|answer|body|text)["']?\s*[:：]|\n\s*["']?(?:content|answer|body|text)["']?\s*[:：]|\n{2,}|$))"#,
                #"(?is)["']?(?:heading|summary)["']?\s*[:：]\s*["'](.+?)["']\s*,\s*["']?(?:content|answer|body|text)["']?\s*[:：]"#,
                #"(?is)(?:^|\n|\{)\s*["']?(?:heading|summary)["']?\s*[:：]\s*["']?(.+?)["']?(?=\s*(?:,\s*["']?(?:content|answer|body|text)["']?\s*[:：]|\n\s*["']?(?:content|answer|body|text)["']?\s*[:：]|\n{2,}|$))"#
            ]
        )

        let content = firstMatch(
            in: normalized,
            patterns: [
                #"(?is)["']?(?:content|answer|body|text)["']?\s*[:：]\s*["']([\s\S]*?)["']\s*(?:[,}]|$)"#,
                #"(?is)["']?(?:content|answer|body|text)["']?\s*[:：]\s*["']([\s\S]*)$"#,
                #"(?is)(?:^|\n|\{)\s*["']?(?:content|answer|body|text)["']?\s*[:：]\s*["']?([\s\S]+?)["']?\s*$"#
            ]
        )

        guard let content else { return nil }
        let normalizedContent = normalizeJSONFragment(content)
        guard !normalizedContent.isEmpty else { return nil }

        let normalizedTitle = normalizeJSONFragment(title ?? "")
        return RewriteAnswerPayload(
            title: normalizedTitle.isEmpty ? fallbackTitle : normalizedTitle,
            content: normalizedContent
        )
    }

    nonisolated private static func firstMatch(in text: String, patterns: [String]) -> String? {
        let searchRange = NSRange(text.startIndex..<text.endIndex, in: text)
        for pattern in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern) else { continue }
            guard let match = regex.firstMatch(in: text, options: [], range: searchRange),
                  match.numberOfRanges > 1,
                  let range = Range(match.range(at: 1), in: text)
            else {
                continue
            }

            let value = String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !value.isEmpty {
                return value
            }
        }
        return nil
    }

    nonisolated private static func normalizeJSONFragment(_ value: String) -> String {
        var normalized = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return "" }

        if normalized.hasSuffix("\"}") {
            normalized.removeLast(2)
        } else if normalized.hasSuffix("'}") {
            normalized.removeLast(2)
        } else if normalized.hasSuffix("\",") || normalized.hasSuffix("',") {
            normalized.removeLast(2)
        } else if normalized.hasSuffix("\"") || normalized.hasSuffix("'") {
            normalized.removeLast()
        }

        normalized = normalized
            .replacingOccurrences(of: "\\r\\n", with: "\n")
            .replacingOccurrences(of: "\\n", with: "\n")
            .replacingOccurrences(of: "\\t", with: "\t")
            .replacingOccurrences(of: "\\\"", with: "\"")
            .replacingOccurrences(of: "\\\\", with: "\\")

        if hasOddUnescapedQuoteCount(normalized), normalized.hasPrefix("\"") {
            normalized.removeFirst()
        }

        normalized = normalized.replacingOccurrences(
            of: #"(?<=[\p{Han}A-Za-z0-9])"(?=[\p{Han}A-Za-z0-9])"#,
            with: "",
            options: .regularExpression
        )

        return normalized.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
