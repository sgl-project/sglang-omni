// Pure transcription decisions; no capture, task, or model ownership.

import Foundation

extension MLXTranscriptionPlanning {
    nonisolated static func nativeLiveLanguage(from hint: String?) -> String? {
        let normalized = hint?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return normalized.isEmpty ? nil : normalized
    }

    nonisolated static func nativeNemotronLanguage(
        requested: String?,
        availableLanguages: [String],
        defaultLanguage: String
    ) -> String {
        let availableByNormalized = availableLanguages.reduce(into: [String: String]()) { result, language in
            result[language.lowercased()] = language
        }
        let normalizedRequest = requested?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "_", with: "-")
            .lowercased()

        if let normalizedRequest, !normalizedRequest.isEmpty {
            if let exact = availableByNormalized[normalizedRequest] {
                return exact
            }
            let baseLanguage = normalizedRequest.split(separator: "-").first.map(String.init) ?? normalizedRequest
            let preferredLocale = ["zh": "zh-cn", "en": "en-us"][baseLanguage]
            if let preferredLocale, let preferred = availableByNormalized[preferredLocale] {
                return preferred
            }
            if let baseMatch = availableByNormalized
                .filter({ $0.key == baseLanguage || $0.key.hasPrefix("\(baseLanguage)-") })
                .sorted(by: { $0.key < $1.key })
                .first?.value
            {
                return baseMatch
            }
        }

        return availableByNormalized[defaultLanguage.lowercased()]
            ?? availableByNormalized["auto"]
            ?? defaultLanguage
    }

    nonisolated static func resolvedNativeLiveVisiblePreview(
        previousPreview: String,
        previousConfirmedText: String,
        confirmedText: String,
        provisionalText: String
    ) -> String? {
        let normalizedConfirmed = confirmedText.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalizedProvisional = provisionalText.trimmingCharacters(in: .whitespacesAndNewlines)
        // Suppress tiny provisional-only flashes (common junk before the first confirm).
        // Confirmed text still surfaces immediately; this does not change Final.
        if normalizedConfirmed.isEmpty, normalizedProvisional.count < 2 {
            return nil
        }
        let combined = (confirmedText + provisionalText).trimmingCharacters(in: .whitespacesAndNewlines)

        guard !combined.isEmpty else { return nil }
        guard combined != previousPreview else { return nil }

        // Suppress preview collapse/thrash when confirmed is unchanged and the visible
        // string shrinks (empty provisional clear, or unstable provisional rewrite).
        if normalizedConfirmed == previousConfirmedText,
           !previousPreview.isEmpty,
           previousPreview.hasPrefix(normalizedConfirmed),
           previousPreview.count > combined.count {
            return nil
        }

        return combined
    }

    nonisolated static func qwenStreamingVisibleTextParts(
        confirmedText: String,
        provisionalText: String
    ) -> (confirmedText: String, provisionalText: String) {
        let visibleConfirmed = qwenStreamingVisibleText(confirmedText)
        let visibleCombined = qwenStreamingVisibleText(confirmedText + provisionalText)

        guard visibleCombined.hasPrefix(visibleConfirmed) else {
            return (confirmedText: "", provisionalText: visibleCombined)
        }

        return (
            confirmedText: visibleConfirmed,
            provisionalText: String(visibleCombined.dropFirst(visibleConfirmed.count))
        )
    }

    nonisolated static func qwenStreamingVisibleText(
        _ decodedText: String,
        suppressIncompleteWindowHeader: Bool = true
    ) -> String {
        let protocolPrefix = "language "
        let textMarker = "<asr_text>"
        let trimmed = decodedText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }

        var visible = ""
        var cursor = trimmed.startIndex

        while cursor < trimmed.endIndex,
              let prefixRange = trimmed.range(
                  of: protocolPrefix,
                  range: cursor..<trimmed.endIndex
              ) {
            visible.append(contentsOf: trimmed[cursor..<prefixRange.lowerBound])

            if prefixRange.lowerBound != trimmed.startIndex,
               !trimmed[trimmed.index(before: prefixRange.lowerBound)].isWhitespace {
                visible.append(contentsOf: protocolPrefix)
                cursor = prefixRange.upperBound
                continue
            }

            let metadataStart = prefixRange.upperBound
            let metadataTail = trimmed[metadataStart...]

            if let markerRange = metadataTail.range(of: textMarker) {
                let language = metadataTail[..<markerRange.lowerBound]
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                guard isQwenLanguageName(language) else {
                    visible.append(contentsOf: protocolPrefix)
                    cursor = metadataStart
                    continue
                }
                cursor = markerRange.upperBound
                continue
            }

            let isInitialHeader = prefixRange.lowerBound == trimmed.startIndex
            if (isInitialHeader || suppressIncompleteWindowHeader),
               isIncompleteQwenProtocolMetadata(metadataTail, textMarker: textMarker) {
                cursor = trimmed.endIndex
                break
            }

            visible.append(contentsOf: protocolPrefix)
            cursor = metadataStart
        }

        if cursor < trimmed.endIndex {
            visible.append(contentsOf: trimmed[cursor...])
        }

        return removingInitialIncompleteQwenProtocolPrefix(from: visible)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private nonisolated static func isIncompleteQwenProtocolMetadata<S: StringProtocol>(
        _ value: S,
        textMarker: String
    ) -> Bool {
        let tail = String(value)
        guard let markerStart = tail.firstIndex(of: "<") else {
            let language = tail.trimmingCharacters(in: .whitespacesAndNewlines)
            return language.isEmpty || isQwenLanguageNamePrefix(language)
        }

        let language = tail[..<markerStart].trimmingCharacters(in: .whitespacesAndNewlines)
        let partialMarker = String(tail[markerStart...])
        return isQwenLanguageName(language) && textMarker.hasPrefix(partialMarker)
    }

    private nonisolated static func isQwenLanguageName<S: StringProtocol>(_ value: S) -> Bool {
        let candidate = String(value).lowercased()
        return qwenLanguageNames.contains(candidate)
    }

    private nonisolated static func isQwenLanguageNamePrefix<S: StringProtocol>(_ value: S) -> Bool {
        let candidate = String(value).lowercased()
        return qwenLanguageNames.contains { $0.hasPrefix(candidate) }
    }

    private nonisolated static func removingInitialIncompleteQwenProtocolPrefix(from text: String) -> String {
        let protocolPrefix = "language "
        for prefixLength in stride(from: protocolPrefix.count - 1, through: 3, by: -1) {
            let partialPrefix = String(protocolPrefix.prefix(prefixLength))
            if text == partialPrefix { return "" }
        }
        return text
    }

    private nonisolated static let qwenLanguageNames: Set<String> = [
        "arabic", "cantonese", "chinese", "czech", "danish", "dutch", "english",
        "finnish", "french", "german", "greek", "hindi", "hungarian",
        "indonesian", "italian", "japanese", "korean", "macedonian", "malay",
        "persian", "polish", "portuguese", "romanian", "russian", "spanish",
        "swedish", "tagalog", "thai", "turkish", "vietnamese",
    ]

    nonisolated static func removingKnownASRContextLeakage(from text: String) -> String {
        let lines = text
            .components(separatedBy: .newlines)
            .filter { !isKnownASRContextLeakageLine($0) }
        return lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private nonisolated static func isKnownASRContextLeakageLine(_ line: String) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }

        let lowercased = trimmed.lowercased()
        return (
            trimmed.contains("说话者的主要语言") && trimmed.contains("其他常用语言")
        ) || (
            trimmed.contains("请将识别偏向于") && trimmed.contains("不要翻译")
        ) || (
            trimmed.contains("当音频中确实出现这些词") && trimmed.contains("词典词汇")
        ) || (
            lowercased.contains("the speaker's primary language is")
                && lowercased.contains("other commonly used languages")
        ) || (
            lowercased.contains("bias recognition toward correct spelling")
                && lowercased.contains("do not translate")
        ) || (
            lowercased.contains("prefer these dictionary terms")
                && lowercased.contains("match the audio")
        ) || (
            trimmed.contains("話者の主要言語") && trimmed.contains("その他のよく使う言語")
        ) || (
            trimmed.contains("認識を寄せてください") && trimmed.contains("翻訳はしないでください")
        ) || (
            trimmed.contains("音声内で実際に一致する場合") && trimmed.contains("辞書語")
        )
    }
}
