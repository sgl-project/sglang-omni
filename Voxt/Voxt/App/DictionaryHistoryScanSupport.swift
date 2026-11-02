// DictionaryHistoryScanSupport.swift
// Provides Dictionary History Scan Support for app lifecycle and routing.

import Foundation

struct DictionaryHistoryScanModelOption: Identifiable, Hashable {
    enum Source: Hashable {
        case local
        case remote
    }

    let id: String
    let source: Source
    let title: String
    let detail: String
}

struct DictionaryHistoryScanRequest: Hashable {
    let modelOptionID: String
    let filterSettings: DictionarySuggestionFilterSettings
}

private enum DictionaryHistoryScanParseError {
    static let domain = "Voxt.DictionaryHistoryScan"

    static func invalidText(code: Int) -> NSError {
        NSError(
            domain: domain,
            code: code,
            userInfo: [
                NSLocalizedDescriptionKey: AppLocalization.localizedString("Dictionary ingestion returned invalid text.")
            ]
        )
    }
}

enum DictionaryHistoryScanResponseParser {
    static func parseTerms(from rawResponse: String) throws -> [String] {
        let normalizedResponse = CustomLLMOutputSanitizer.normalizeResultText(rawResponse)
        guard !normalizedResponse.isEmpty else {
            throw DictionaryHistoryScanParseError.invalidText(code: -11)
        }

        if let terms = try parseTermPayload(from: normalizedResponse) {
            return normalizeAcceptedTerms(from: terms)
        }

        if let extractedJSONArray = extractJSONArrayString(from: normalizedResponse),
           let terms = try parseTermPayload(from: extractedJSONArray) {
            return normalizeAcceptedTerms(from: terms)
        }

        if let extractedJSONObject = extractJSONObjectString(from: normalizedResponse),
           let terms = try parseTermPayload(from: extractedJSONObject) {
            return normalizeAcceptedTerms(from: terms)
        }

        if let truncatedJSONArrayTerms = try parseTruncatedJSONArrayTerms(from: normalizedResponse) {
            return normalizeAcceptedTerms(from: truncatedJSONArrayTerms)
        }

        if let legacyTerms = parseLegacyTermList(from: normalizedResponse) {
            return normalizeAcceptedTerms(from: legacyTerms)
        }

        VoxtLog.dictionaryWarning(
            "Dictionary history scan returned invalid JSON payload. preview=\(responsePreview(normalizedResponse))"
        )
        throw DictionaryHistoryScanParseError.invalidText(code: -13)
    }

    static func normalizeAcceptedTerms(from terms: [String]) -> [String] {
        var seen = Set<String>()
        var orderedTerms: [String] = []

        for term in terms {
            let normalized = DictionaryStore.normalizeTerm(term)
            guard !normalized.isEmpty else { continue }
            guard DictionaryHistoryScanCandidateValidator.shouldAccept(term: term) else { continue }
            guard seen.insert(normalized).inserted else { continue }
            orderedTerms.append(term)
        }

        return orderedTerms
    }

    static func responsesTextFormatPayload() -> [String: Any] {
        [
            "format": [
                "type": "json_schema",
                "name": "dictionary_history_terms",
                "strict": true,
                "schema": [
                    "type": "object",
                    "additionalProperties": false,
                    "properties": [
                        "terms": [
                            "type": "array",
                            "items": [
                                "type": "object",
                                "additionalProperties": false,
                                "properties": [
                                    "term": [
                                        "type": "string"
                                    ]
                                ],
                                "required": ["term"]
                            ]
                        ]
                    ],
                    "required": ["terms"]
                ]
            ]
        ]
    }

    static func extractJSONArrayString(from text: String) -> String? {
        extractBalancedJSONContainerString(from: text, opening: "[", closing: "]")
    }

    static func extractJSONObjectString(from text: String) -> String? {
        extractBalancedJSONContainerString(from: text, opening: "{", closing: "}")
    }

    static func extractBalancedJSONObjectStrings(from text: String) -> [String] {
        let characters = Array(text)
        var objects: [String] = []
        var startIndex: Int?
        var depth = 0
        var isInsideString = false
        var isEscaping = false

        for (index, character) in characters.enumerated() {
            if isInsideString {
                if isEscaping {
                    isEscaping = false
                    continue
                }
                if character == "\\" {
                    isEscaping = true
                } else if character == "\"" {
                    isInsideString = false
                }
                continue
            }

            if character == "\"" {
                isInsideString = true
                continue
            }

            if character == "{" {
                if depth == 0 {
                    startIndex = index
                }
                depth += 1
                continue
            }

            if character == "}", depth > 0 {
                depth -= 1
                if depth == 0, let startIndex {
                    objects.append(String(characters[startIndex...index]))
                }
            }
        }

        return objects
    }

    private static func extractBalancedJSONContainerString(
        from text: String,
        opening: Character,
        closing: Character
    ) -> String? {
        let characters = Array(text)
        var startIndex: Int?
        var depth = 0
        var isInsideString = false
        var isEscaping = false

        for (index, character) in characters.enumerated() {
            if isInsideString {
                if isEscaping {
                    isEscaping = false
                    continue
                }
                if character == "\\" {
                    isEscaping = true
                } else if character == "\"" {
                    isInsideString = false
                }
                continue
            }

            if character == "\"" {
                isInsideString = true
                continue
            }

            if character == opening {
                if depth == 0 {
                    startIndex = index
                }
                depth += 1
                continue
            }

            if character == closing, depth > 0 {
                depth -= 1
                if depth == 0, let startIndex {
                    return String(characters[startIndex...index])
                }
            }
        }

        return nil
    }

    private static func parseTermPayload(from text: String) throws -> [String]? {
        guard let data = text.data(using: .utf8) else {
            throw DictionaryHistoryScanParseError.invalidText(code: -12)
        }
        do {
            let jsonObject = try JSONSerialization.jsonObject(with: data)
            return try parseTerms(fromJSONObject: jsonObject)
        } catch {
            return nil
        }
    }

    private static func parseTerms(fromJSONObject jsonObject: Any) throws -> [String]? {
        if let terms = try parseTermsArray(jsonObject) {
            return terms
        }

        guard let dictionary = jsonObject as? [String: Any] else {
            return nil
        }

        for key in ["terms", "items", "results", "candidates", "data"] {
            guard let nestedValue = dictionary[key] else { continue }
            if let terms = try parseTermsArray(nestedValue) {
                return terms
            }
        }

        if dictionary.count == 1, let nestedValue = dictionary.values.first {
            return try parseTermsArray(nestedValue)
        }

        return nil
    }

    private static func parseTermsArray(_ rawValue: Any) throws -> [String]? {
        if let rawTerms = rawValue as? [String] {
            return try parseRawTerms(rawTerms)
        }

        guard let rawItems = rawValue as? [[String: Any]] else {
            return nil
        }

        let rawTerms = rawItems.compactMap { $0["term"] as? String }
        guard rawTerms.count == rawItems.count else {
            return nil
        }
        return try parseRawTerms(rawTerms)
    }

    private static func parseTruncatedJSONArrayTerms(from text: String) throws -> [String]? {
        guard text.contains("\"term\"") else { return nil }
        let completeObjects = extractBalancedJSONObjectStrings(from: text)
        guard !completeObjects.isEmpty else { return nil }

        var recoveredTerms: [String] = []
        recoveredTerms.reserveCapacity(completeObjects.count)

        for objectText in completeObjects {
            guard let data = objectText.data(using: .utf8),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let term = object["term"] as? String
            else {
                continue
            }
            recoveredTerms.append(term)
        }

        guard !recoveredTerms.isEmpty else { return nil }
        return recoveredTerms
    }

    private static func parseRawTerms(_ rawTerms: [String]) throws -> [String] {
        var parsedTerms: [String] = []
        parsedTerms.reserveCapacity(rawTerms.count)

        for rawTerm in rawTerms {
            let trimmed = rawTerm.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else {
                throw DictionaryHistoryScanParseError.invalidText(code: -10)
            }
            parsedTerms.append(trimmed)
        }

        return parsedTerms
    }

    private static func parseLegacyTermList(from text: String) -> [String]? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        if trimmed.caseInsensitiveCompare("null") == .orderedSame {
            return []
        }

        let lines = trimmed
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !lines.isEmpty else { return nil }

        var parsedTerms: [String] = []
        var headerCount = 0
        var explicitListMarkerCount = 0
        var rejectedLineCount = 0

        for line in lines {
            let parsedLine = legacyListLine(from: line)
            if parsedLine.isHeader {
                headerCount += 1
                continue
            }
            if parsedLine.hasExplicitListMarker {
                explicitListMarkerCount += 1
            }
            if let term = parsedLine.term {
                parsedTerms.append(term)
            } else {
                rejectedLineCount += 1
            }
        }

        guard !parsedTerms.isEmpty else { return nil }
        guard rejectedLineCount == 0 else { return nil }

        let looksLikeLegacyList =
            explicitListMarkerCount > 0 ||
            headerCount > 0 ||
            parsedTerms.count >= 2 ||
            (parsedTerms.count == 1 && lines.count == 1)

        return looksLikeLegacyList ? parsedTerms : nil
    }

    private static func legacyListLine(from line: String) -> (term: String?, isHeader: Bool, hasExplicitListMarker: Bool) {
        let lowercased = line.lowercased()
        if lowercased == "null" {
            return (nil, false, false)
        }

        // Skip common list headers emitted by legacy prompts before numbered terms.
        if line.hasSuffix(":") {
            return (nil, true, false)
        }

        var cleaned = line
        var hasExplicitListMarker = false
        if let regex = try? NSRegularExpression(pattern: #"^\s*(?:[-*•]\s+|\d+[.)]\s+)"#) {
            let range = NSRange(location: 0, length: (cleaned as NSString).length)
            hasExplicitListMarker = regex.firstMatch(in: cleaned, options: [], range: range) != nil
            cleaned = regex.stringByReplacingMatches(in: cleaned, options: [], range: range, withTemplate: "")
        }

        cleaned = cleaned
            .trimmingCharacters(in: CharacterSet(charactersIn: "\"'`").union(.whitespacesAndNewlines))
            .trimmingCharacters(in: CharacterSet(charactersIn: ",;"))
            .trimmingCharacters(in: .whitespacesAndNewlines)

        guard !cleaned.isEmpty else { return (nil, false, hasExplicitListMarker) }
        guard DictionaryHistoryScanCandidateValidator.shouldAccept(term: cleaned) else {
            return (nil, false, hasExplicitListMarker)
        }
        return (cleaned, false, hasExplicitListMarker)
    }

    private static func responsePreview(_ text: String) -> String {
        let collapsedWhitespace = text
            .replacingOccurrences(of: "\r", with: " ")
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\t", with: " ")
        return String(collapsedWhitespace.prefix(300))
    }
}

struct DictionaryHistoryScanPromptRecord: Encodable {
    let id: String
    let kind: String
    let groupName: String?
    let text: String
    let dictionaryHitTerms: [String]
    let dictionaryCorrectedTerms: [String]
}

enum DictionaryHistoryScanSupport {
    static func buildPrompt(
        for batch: [TranscriptionHistoryEntry],
        filterSettings: DictionarySuggestionFilterSettings,
        groupsByID: [UUID: AppBranchGroup],
        groupsByLowercasedName: [String: AppBranchGroup],
        userMainLanguage: String,
        userOtherLanguages: String
    ) throws -> String {
        let records = batch.map { entry in
            let scope = resolvedHistoryScope(
                for: entry,
                groupsByID: groupsByID,
                groupsByLowercasedName: groupsByLowercasedName
            )
            return DictionaryHistoryScanPromptRecord(
                id: entry.id.uuidString,
                kind: entry.kind.rawValue,
                groupName: scope.groupName,
                text: trimmedHistoryScanText(entry.text),
                dictionaryHitTerms: entry.dictionaryHitTerms,
                dictionaryCorrectedTerms: entry.dictionaryCorrectedTerms
            )
        }

        let recordsXML = historyScanXMLRecords(from: records)
        let settings = filterSettings.sanitized()
        return resolvedPrompt(
            template: settings.prompt,
            userMainLanguage: userMainLanguage,
            userOtherLanguages: userOtherLanguages,
            historyRecordsXML: recordsXML,
            maxCandidatesPerBatch: settings.maxCandidatesPerBatch
        )
    }

    static func parseCandidates(
        terms: [String],
        batch: [TranscriptionHistoryEntry],
        groupsByID: [UUID: AppBranchGroup],
        groupsByLowercasedName: [String: AppBranchGroup]
    ) throws -> [DictionaryHistoryScanCandidate] {
        guard !terms.isEmpty else { return [] }
        var candidatesByKey: [String: DictionaryHistoryScanCandidate] = [:]

        for term in terms {
            let sourceEntries = resolvedSourceEntries(for: term, in: batch)
            let scopedEntries = sourceEntries.isEmpty ? batch : sourceEntries

            let scope = resolvedCandidateScope(
                sourceEntries: scopedEntries,
                groupsByID: groupsByID,
                groupsByLowercasedName: groupsByLowercasedName
            )
            let evidenceSample = resolvedEvidenceSample(
                preferredSample: nil,
                term: term,
                sourceEntries: scopedEntries
            )
            guard DictionaryHistoryScanCandidateValidator.shouldAccept(
                term: term,
                evidenceSample: evidenceSample
            ) else {
                continue
            }
            let key = "\(DictionaryStore.normalizeTerm(term))|\(scope.groupID?.uuidString ?? "global")"
            let historyEntryIDs = scopedEntries.map(\.id)

            if let existing = candidatesByKey[key] {
                let mergedIDs = Array(Set(existing.historyEntryIDs + historyEntryIDs)).sorted {
                    $0.uuidString < $1.uuidString
                }
                candidatesByKey[key] = DictionaryHistoryScanCandidate(
                    term: existing.term.count >= term.count ? existing.term : term,
                    historyEntryIDs: mergedIDs,
                    groupID: existing.groupID,
                    groupNameSnapshot: existing.groupNameSnapshot ?? scope.groupName,
                    evidenceSample: existing.evidenceSample.isEmpty ? evidenceSample : existing.evidenceSample
                )
            } else {
                candidatesByKey[key] = DictionaryHistoryScanCandidate(
                    term: term,
                    historyEntryIDs: historyEntryIDs,
                    groupID: scope.groupID,
                    groupNameSnapshot: scope.groupName,
                    evidenceSample: evidenceSample
                )
            }
        }

        return candidatesByKey.values.sorted {
            $0.term.localizedCaseInsensitiveCompare($1.term) == .orderedAscending
        }
    }

    static func loadGroups(defaults: UserDefaults = .standard) -> [AppBranchGroup] {
        guard let data = defaults.data(forKey: AppPreferenceKey.appBranchGroups),
              let groups = try? JSONDecoder().decode([AppBranchGroup].self, from: data)
        else {
            return []
        }
        return groups
    }

    private static func resolvedPrompt(
        template: String,
        userMainLanguage: String,
        userOtherLanguages: String,
        historyRecordsXML: String,
        maxCandidatesPerBatch: Int
    ) -> String {
        var prompt = template.trimmingCharacters(in: .whitespacesAndNewlines)

        if prompt.contains("{{USER_MAIN_LANGUAGE}}") {
            prompt = prompt.replacingOccurrences(of: "{{USER_MAIN_LANGUAGE}}", with: userMainLanguage)
        } else {
            prompt += "\n\nUser’s main language: \(userMainLanguage)"
        }

        if prompt.contains("{{USER_OTHER_LANGUAGES}}") {
            prompt = prompt.replacingOccurrences(of: "{{USER_OTHER_LANGUAGES}}", with: userOtherLanguages)
        } else {
            prompt += "\n\nOther frequently used languages: \(userOtherLanguages)"
        }

        if prompt.contains("{{HISTORY_RECORDS}}") {
            prompt = prompt.replacingOccurrences(of: "{{HISTORY_RECORDS}}", with: historyRecordsXML)
        } else {
            prompt += "\n\nHistory records:\n\(historyRecordsXML)"
        }

        if prompt.contains("{{MAX_CANDIDATES_PER_BATCH}}") {
            prompt = prompt.replacingOccurrences(
                of: "{{MAX_CANDIDATES_PER_BATCH}}",
                with: String(maxCandidatesPerBatch)
            )
        } else {
            prompt += """

            <outputConstraints>
              <maxCandidateCount>\(maxCandidatesPerBatch)</maxCandidateCount>
              <format>json_array_of_term_objects</format>
            </outputConstraints>

            Return at most \(maxCandidatesPerBatch) accepted terms.
            """
        }

        return prompt
    }

    private static func historyScanXMLRecords(
        from records: [DictionaryHistoryScanPromptRecord]
    ) -> String {
        let body = records.map { record in
            let groupName = record.groupName.map { xmlEscapedText($0) } ?? ""
            let dictionaryHitTerms = record.dictionaryHitTerms
                .map { "<term>\(xmlEscapedText($0))</term>" }
                .joined()
            let dictionaryCorrectedTerms = record.dictionaryCorrectedTerms
                .map { "<term>\(xmlEscapedText($0))</term>" }
                .joined()

            return """
            <historyRecord id="\(xmlEscapedAttribute(record.id))" kind="\(xmlEscapedAttribute(record.kind))">
              <groupName>\(groupName)</groupName>
              <text>\(xmlEscapedText(record.text))</text>
              <dictionaryHitTerms>\(dictionaryHitTerms)</dictionaryHitTerms>
              <dictionaryCorrectedTerms>\(dictionaryCorrectedTerms)</dictionaryCorrectedTerms>
            </historyRecord>
            """
        }.joined(separator: "\n")

        return "<historyRecords>\n\(body)\n</historyRecords>"
    }

    private static func resolvedSourceEntries(
        for term: String,
        in batch: [TranscriptionHistoryEntry]
    ) -> [TranscriptionHistoryEntry] {
        batch.filter {
            $0.text.range(of: term, options: [.caseInsensitive, .diacriticInsensitive]) != nil
        }
    }

    private static func resolvedCandidateScope(
        sourceEntries: [TranscriptionHistoryEntry],
        groupsByID: [UUID: AppBranchGroup],
        groupsByLowercasedName: [String: AppBranchGroup]
    ) -> (groupID: UUID?, groupName: String?) {
        let scopes = sourceEntries.map {
            resolvedHistoryScope(
                for: $0,
                groupsByID: groupsByID,
                groupsByLowercasedName: groupsByLowercasedName
            )
        }
        let uniqueScopedIDs = Array(Set(scopes.compactMap(\.groupID)))
        guard uniqueScopedIDs.count == 1, scopes.allSatisfy({ $0.groupID == uniqueScopedIDs[0] }) else {
            return (nil, nil)
        }
        return (uniqueScopedIDs[0], scopes.first?.groupName)
    }

    private static func resolvedHistoryScope(
        for entry: TranscriptionHistoryEntry,
        groupsByID: [UUID: AppBranchGroup],
        groupsByLowercasedName: [String: AppBranchGroup]
    ) -> (groupID: UUID?, groupName: String?) {
        if let matchedGroupID = entry.matchedGroupID {
            let groupName = groupsByID[matchedGroupID]?.name
                ?? entry.matchedAppGroupName
                ?? entry.matchedURLGroupName
            return (matchedGroupID, groupName)
        }

        if let groupName = (entry.matchedAppGroupName ?? entry.matchedURLGroupName)?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !groupName.isEmpty,
           let group = groupsByLowercasedName[groupName.lowercased()] {
            return (group.id, group.name)
        }

        return (nil, nil)
    }

    private static func resolvedEvidenceSample(
        preferredSample: String?,
        term: String,
        sourceEntries: [TranscriptionHistoryEntry]
    ) -> String {
        let trimmedPreferred = preferredSample?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !trimmedPreferred.isEmpty {
            return String(trimmedPreferred.prefix(80))
        }
        guard let firstEntry = sourceEntries.first else { return "" }
        return historyEvidenceSample(for: term, in: firstEntry.text)
    }

    private static func historyEvidenceSample(for term: String, in text: String) -> String {
        let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedText.isEmpty else { return "" }
        guard let range = trimmedText.range(of: term, options: [.caseInsensitive, .diacriticInsensitive]) else {
            return String(trimmedText.prefix(80))
        }

        let start = trimmedText.distance(from: trimmedText.startIndex, to: range.lowerBound)
        let lowerOffset = max(0, start - 18)
        let upperOffset = min(trimmedText.count, start + term.count + 18)
        let lowerIndex = trimmedText.index(trimmedText.startIndex, offsetBy: lowerOffset)
        let upperIndex = trimmedText.index(trimmedText.startIndex, offsetBy: upperOffset)
        let snippet = trimmedText[lowerIndex..<upperIndex].trimmingCharacters(in: .whitespacesAndNewlines)
        return String(snippet.prefix(80))
    }

    private static func trimmedHistoryScanText(_ text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count > 320 else { return trimmed }
        let index = trimmed.index(trimmed.startIndex, offsetBy: 320)
        return String(trimmed[..<index])
    }

    private static func xmlEscapedText(_ text: String) -> String {
        text
            .replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
    }

    private static func xmlEscapedAttribute(_ text: String) -> String {
        xmlEscapedText(text)
            .replacingOccurrences(of: "\"", with: "&quot;")
            .replacingOccurrences(of: "'", with: "&apos;")
    }
}
