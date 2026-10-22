// MeetingSummaryContextPlanning.swift
// Bounds long meeting summary and follow-up context without repeatedly sending the full transcript.

import Foundation

nonisolated enum MeetingSummaryContextPlanning {
    static let summaryWindowCharacterLimit = 12_000
    static let followUpWindowCharacterLimit = 6_000

    static func windows(
        for transcript: String,
        maximumCharacters: Int = summaryWindowCharacterLimit
    ) -> [String] {
        let limit = max(maximumCharacters, 500)
        let paragraphs = transcript
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !paragraphs.isEmpty else { return [] }

        var output: [String] = []
        var current = ""
        for paragraph in paragraphs {
            if paragraph.count > limit {
                if !current.isEmpty {
                    output.append(current)
                    current = ""
                }
                var cursor = paragraph.startIndex
                while cursor < paragraph.endIndex {
                    let end = paragraph.index(cursor, offsetBy: limit, limitedBy: paragraph.endIndex) ?? paragraph.endIndex
                    output.append(String(paragraph[cursor..<end]))
                    cursor = end
                }
                continue
            }
            let candidate = current.isEmpty ? paragraph : current + "\n" + paragraph
            if candidate.count > limit {
                output.append(current)
                current = paragraph
            } else {
                current = candidate
            }
        }
        if !current.isEmpty {
            output.append(current)
        }
        return output
    }

    static func relevantFollowUpContext(
        transcript: String,
        question: String,
        maximumWindows: Int = 3
    ) -> String {
        let windows = windows(for: transcript, maximumCharacters: followUpWindowCharacterLimit)
        guard windows.count > maximumWindows else { return windows.joined(separator: "\n\n") }

        let queryTerms = searchableTerms(question)
        if queryTerms.isEmpty || isGlobalQuestion(question) {
            return [windows.first, windows.last]
                .compactMap { $0 }
                .joined(separator: "\n\n")
        }

        let ranked = windows.enumerated().map { index, window in
            let normalized = window.lowercased()
            let score = queryTerms.reduce(0) { partial, term in
                partial + normalized.components(separatedBy: term).count - 1
            }
            return (index: index, score: score, window: window)
        }
        let selected = ranked
            .sorted {
                if $0.score == $1.score { return $0.index < $1.index }
                return $0.score > $1.score
            }
            .prefix(max(maximumWindows, 1))
            .sorted { $0.index < $1.index }
        return selected.map(\.window).joined(separator: "\n\n")
    }

    private static func searchableTerms(_ text: String) -> Set<String> {
        let normalized = text.lowercased()
        let words = normalized
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count >= 2 }
        var terms = Set(words)
        let cjk = normalized.unicodeScalars.filter {
            (0x3400...0x9FFF).contains(Int($0.value))
        }
        let characters = cjk.map(String.init)
        if characters.count >= 2 {
            for index in 0..<(characters.count - 1) {
                terms.insert(characters[index] + characters[index + 1])
            }
        }
        return terms
    }

    private static func isGlobalQuestion(_ question: String) -> Bool {
        let normalized = question.lowercased()
        return ["overall", "entire", "whole meeting", "summary", "总结", "整体", "全程", "整个会议"]
            .contains { normalized.contains($0) }
    }
}
