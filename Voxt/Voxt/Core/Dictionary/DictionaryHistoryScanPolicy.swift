import Foundation

enum DictionaryHistoryScanPromptLanguageSupport {
    nonisolated static let noneValue = "None"

    nonisolated static func otherLanguagesPromptValue(from codes: [String]) -> String {
        let options = Array(codes.dropFirst())
            .compactMap(UserMainLanguageOption.option(for:))
        guard !options.isEmpty else { return noneValue }
        return options.map(\.promptName).joined(separator: ", ")
    }
}

struct DictionarySuggestionFilterSettings: Codable, Equatable, Hashable {
    var prompt: String
    var batchSize: Int
    var maxCandidatesPerBatch: Int

    static let defaultBatchSize = 12
    static let defaultMaxCandidatesPerBatch = 12
    static let minimumBatchSize = 1
    static let maximumBatchSize = 50
    static let minimumMaxCandidates = 1
    static let maximumMaxCandidates = 50

    static var defaultPrompt: String {
        defaultPrompt(language: AppLocalization.language)
    }

    static func defaultPrompt(language: AppInterfaceLanguage) -> String {
        AppPromptDefaults.text(for: .dictionaryIngest, language: language)
    }

    static var defaultValue: DictionarySuggestionFilterSettings {
        DictionarySuggestionFilterSettings(
            prompt: defaultPrompt,
            batchSize: defaultBatchSize,
            maxCandidatesPerBatch: defaultMaxCandidatesPerBatch
        )
    }

    func sanitized() -> DictionarySuggestionFilterSettings {
        return DictionarySuggestionFilterSettings(
            prompt: Self.sanitizedPrompt(prompt),
            batchSize: min(max(batchSize, Self.minimumBatchSize), Self.maximumBatchSize),
            maxCandidatesPerBatch: min(
                max(maxCandidatesPerBatch, Self.minimumMaxCandidates),
                Self.maximumMaxCandidates
            )
        )
    }

    static func sanitizedPrompt(_ rawPrompt: String) -> String {
        sanitizedPrompt(rawPrompt, language: AppLocalization.language)
    }

    static func sanitizedPrompt(_ rawPrompt: String, language: AppInterfaceLanguage) -> String {
        let trimmedPrompt = rawPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let localizedDefaultPrompt = defaultPrompt(language: language)
        guard !trimmedPrompt.isEmpty else { return localizedDefaultPrompt }
        if isLegacyDefaultPrompt(trimmedPrompt) {
            return localizedDefaultPrompt
        }
        if AppPromptDefaults.matchesKnownDefault(trimmedPrompt, kind: .dictionaryIngest) {
            return localizedDefaultPrompt
        }
        return trimmedPrompt
    }

    static func canonicalStoredPrompt(_ prompt: String) -> String {
        let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPrompt.isEmpty else { return "" }
        if isLegacyDefaultPrompt(trimmedPrompt) {
            return ""
        }
        return AppPromptDefaults.matchesKnownDefault(trimmedPrompt, kind: .dictionaryIngest) ? "" : prompt
    }

    private static func isLegacyDefaultPrompt(_ prompt: String) -> Bool {
        let legacySentinels = [
            "Output: Structured list of recommended terms",
            "One term per line",
            "Return null if no worthy terms"
        ]
        return legacySentinels.allSatisfy { prompt.localizedCaseInsensitiveContains($0) }
    }
}

enum DictionaryHistoryScanCandidateValidator {
    static func shouldAccept(term: String) -> Bool {
        let trimmed = term.trimmingCharacters(in: .whitespacesAndNewlines)
        return isStructurallyReasonable(trimmed) && !isClearlyGenericVocabulary(trimmed)
    }

    static func shouldAccept(term: String, evidenceSample: String) -> Bool {
        guard shouldAccept(term: term) else { return false }
        let sample = evidenceSample.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !sample.isEmpty else { return true }
        return !isContextSpecificArtifact(term: term, in: sample)
    }

    private static let sentencePunctuation: Set<Character> = [
        ".", ",", ":", ";", "!", "?", "，", "。", "：", "；", "！", "？", "、"
    ]

    private static let genericEnglishTerms: Set<String> = [
        "button",
        "company",
        "email",
        "file",
        "flight",
        "message",
        "model",
        "neither",
        "office",
        "prompt",
        "schedule",
        "setting",
        "station",
        "token",
        "train"
    ]

    private static let genericEnglishReferenceStarters: Set<String> = [
        "my", "our", "your", "their", "this", "that", "these", "those"
    ]

    private static let genericEnglishReferenceEndings: Set<String> = [
        "company",
        "content",
        "data",
        "feature",
        "file",
        "function",
        "issue",
        "message",
        "model",
        "problem",
        "prompt",
        "result",
        "rule",
        "setting",
        "term",
        "text"
    ]

    private static let genericCJKTerms: Set<String> = [
        "会议",
        "公司",
        "地铁",
        "文件",
        "机场",
        "航班",
        "订单",
        "提示词",
        "模型",
        "火车",
        "邮件",
        "设置",
        "车次",
        "酒店",
        "高铁"
    ]

    private static let genericChineseReferencePrefixes: [String] = [
        "这个",
        "那个",
        "这些",
        "那些",
        "这种",
        "那种",
        "我们",
        "你们",
        "他们",
        "她们",
        "它们",
        "我的",
        "你的",
        "他的",
        "她的",
        "它的",
        "我们的",
        "你们的",
        "他们的",
        "她们的",
        "它们的"
    ]

    private static let genericChineseReferenceSuffixes: [String] = [
        "规则",
        "问题",
        "功能",
        "内容",
        "结果",
        "消息",
        "词汇",
        "词语",
        "文本",
        "数据",
        "文件",
        "设置",
        "模型",
        "提示词",
        "方案",
        "接口",
        "公司",
        "事情",
        "情况"
    ]

    private static let travelKeywords: [String] = [
        "flight",
        "flights",
        "train",
        "trains",
        "station",
        "route",
        "航班",
        "车次",
        "列车",
        "火车",
        "高铁",
        "机票",
        "动车"
    ]

    private static func isStructurallyReasonable(_ term: String) -> Bool {
        guard !term.isEmpty else { return false }
        guard term.count <= 48 else { return false }
        guard !term.contains(where: \.isNewline) else { return false }
        guard !term.contains(where: { sentencePunctuation.contains($0) }) else { return false }
        guard term.range(of: #"^\d+$"#, options: .regularExpression) == nil else { return false }

        let latinWordCount = latinWords(in: term).count
        if containsLatinLetters(in: term) {
            guard latinWordCount <= 6 else { return false }
            guard latinLetterCount(in: term) <= 32 else { return false }
        }

        let cjkCount = cjkCharacterCount(in: term)
        if cjkCount > 0, !containsLatinLetters(in: term) {
            guard cjkCount <= 6 else { return false }
        }

        return true
    }

    private static func isClearlyGenericVocabulary(_ term: String) -> Bool {
        let lowercased = term.lowercased()
        if genericEnglishTerms.contains(lowercased) {
            return true
        }
        if genericCJKTerms.contains(term) {
            return true
        }
        if isGenericReferencePhrase(term, lowercased: lowercased) {
            return true
        }
        return false
    }

    private static func isGenericReferencePhrase(_ term: String, lowercased: String) -> Bool {
        if isGenericChineseReferencePhrase(term) {
            return true
        }
        if isGenericEnglishReferencePhrase(lowercased) {
            return true
        }
        return false
    }

    private static func isGenericChineseReferencePhrase(_ term: String) -> Bool {
        guard term.count >= 3, term.count <= 8 else { return false }
        guard let prefix = genericChineseReferencePrefixes.first(where: term.hasPrefix) else {
            return false
        }
        let remainder = String(term.dropFirst(prefix.count))
        guard !remainder.isEmpty else { return false }
        return genericChineseReferenceSuffixes.contains(where: remainder.hasSuffix)
    }

    private static func isGenericEnglishReferencePhrase(_ lowercased: String) -> Bool {
        let words = lowercased.split(whereSeparator: \.isWhitespace)
        guard words.count >= 2, words.count <= 4 else { return false }
        guard let first = words.first, genericEnglishReferenceStarters.contains(String(first)) else {
            return false
        }
        guard let last = words.last else { return false }
        return genericEnglishReferenceEndings.contains(String(last))
    }

    private static func isContextSpecificArtifact(term: String, in sample: String) -> Bool {
        looksLikeTravelRouteEndpoint(term: term, in: sample)
            || looksLikeTransportIdentifier(term: term, in: sample)
    }

    private static func looksLikeTravelRouteEndpoint(term: String, in sample: String) -> Bool {
        let normalizedSample = sample.lowercased()
        guard travelKeywords.contains(where: normalizedSample.contains) else { return false }

        if sample.contains("\(term)到") || sample.contains("到\(term)") {
            return true
        }

        if normalizedSample.contains("from \(term.lowercased())")
            || normalizedSample.contains("to \(term.lowercased())")
            || normalizedSample.contains("\(term.lowercased()) to ")
        {
            return true
        }

        return false
    }

    private static func looksLikeTransportIdentifier(term: String, in sample: String) -> Bool {
        let normalizedSample = sample.lowercased()
        guard travelKeywords.contains(where: normalizedSample.contains) else { return false }
        return term.range(of: #"^[A-Za-z]{1,3}\d{2,4}$"#, options: .regularExpression) != nil
    }

    private static func latinWords(in text: String) -> [Substring] {
        text.split(whereSeparator: \.isWhitespace).filter { token in
            token.contains(where: isLatinLetter)
        }
    }

    private static func containsLatinLetters(in text: String) -> Bool {
        text.contains(where: isLatinLetter)
    }

    private static func latinLetterCount(in text: String) -> Int {
        text.reduce(into: 0) { count, character in
            if isLatinLetter(character) {
                count += 1
            }
        }
    }

    private static func cjkCharacterCount(in text: String) -> Int {
        text.unicodeScalars.reduce(into: 0) { count, scalar in
            if isCJKScalar(scalar) {
                count += 1
            }
        }
    }

    nonisolated private static func isLatinLetter(_ character: Character) -> Bool {
        character.unicodeScalars.contains { scalar in
            (65...90).contains(scalar.value) || (97...122).contains(scalar.value)
        }
    }

    private static func isCJKScalar(_ scalar: UnicodeScalar) -> Bool {
        switch scalar.value {
        case 0x3400...0x4DBF,
             0x4E00...0x9FFF,
             0x3040...0x309F,
             0x30A0...0x30FF,
             0x31F0...0x31FF,
             0xAC00...0xD7AF:
            return true
        default:
            return false
        }
    }
}
