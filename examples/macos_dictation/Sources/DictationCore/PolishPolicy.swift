import Foundation

/// Conservative lexical guard; ambiguous edits fall back to the original transcript.
public enum PolishPolicy {
    public static func validate(_ result: String, original: String, personalBackground: String = "") throws {
        // Catch the observed full/partial translation of Latin text into Chinese.
        // This is a conservative guard, not a general semantic-fidelity classifier.
        if isLatinWithoutHan(original), containsHan(result) {
            throw DictationError("整理改变了原文语言，已拒绝该结果。")
        }
        if containsHan(original), containsHan(result) {
            func letters(_ text: String) -> String {
                String(String.UnicodeScalarView(text.unicodeScalars.filter {
                    !CharacterSet.whitespacesAndNewlines.contains($0) && !CharacterSet.punctuationCharacters.contains($0)
                }))
            }
            let before = letters(original), after = letters(result)
            let transform = StringTransform("Traditional-Simplified")
            if before != after, let canonical = before.applyingTransform(transform, reverse: false),
               canonical == after.applyingTransform(transform, reverse: false) {
                throw DictationError("整理改变了原文简繁用字，已拒绝该结果。")
            }
        }
        // A small model can still paraphrase despite the prompt. Accept formatting,
        // adjacent Chinese stutter removal, and explicitly supplied spelling pairs;
        // reject other lexical edits so the session retains its original transcript.
        func matches(_ pattern: String, in text: String) -> [String] {
            let regex = try! NSRegularExpression(pattern: pattern)
            return regex.matches(in: text, range: NSRange(text.startIndex..., in: text)).compactMap {
                Range($0.range, in: text).map { String(text[$0]) }
            }
        }
        let numberPattern = #"[0-9]+(?:\.[0-9]+)*|[零〇一二三四五六七八九十百千万亿两]+"#
        guard matches(numberPattern, in: original) == matches(numberPattern, in: result) else {
            throw DictationError("整理改变了原文数字，已拒绝该结果。")
        }
        if original.contains("?") || original.contains("？") {
            guard result.contains("?") || result.contains("？") else {
                throw DictationError("整理改变了原文问句，已拒绝该结果。")
            }
        }
        func canonical(_ text: String) -> String {
            var value = contentSignature(text)
            let repeats = try! NSRegularExpression(pattern: #"([\p{Han}]{1,6})\1+"#)
            for _ in 0..<8 {
                let reduced = repeats.stringByReplacingMatches(in: value, range: NSRange(value.startIndex..., in: value), withTemplate: "$1")
                if reduced == value { break }
                value = reduced
            }
            return value
        }
        var before = canonical(original), after = canonical(result)
        for negation in ["不", "没", "未", "无", "别"] {
            guard before.filter({ String($0) == negation }).count == after.filter({ String($0) == negation }).count else {
                throw DictationError("整理改变了原文否定表达，已拒绝该结果。")
            }
        }
        // Free-form background remains quoted context. Only explicit `old → new`
        // or `old -> new` spelling pairs extend the lexical check's equivalence set.
        let pairs = try! NSRegularExpression(pattern: #"(?:^|[：:\n，,；;])\h*([\p{L}\p{N}_+./-]+(?:\h+[\p{L}\p{N}_+./-]+){0,5})\h*(?:→|->)\h*([\p{L}\p{N}_+./-]+)"#)
        for pair in pairs.matches(in: personalBackground, range: NSRange(personalBackground.startIndex..., in: personalBackground)) {
            guard let oldRange = Range(pair.range(at: 1), in: personalBackground),
                  let newRange = Range(pair.range(at: 2), in: personalBackground) else { continue }
            let old = canonical(String(personalBackground[oldRange])), new = canonical(String(personalBackground[newRange]))
            guard !old.isEmpty, !new.isEmpty, old.count <= 40, new.count <= 40 else { continue }
            before = before.replacingOccurrences(of: old, with: new)
            after = after.replacingOccurrences(of: old, with: new)
        }
        guard before == after else { throw DictationError("整理改动了原文用词，已拒绝该结果。") }
    }

    static func isLatinWithoutHan(_ text: String) -> Bool {
        !containsHan(text) && text.unicodeScalars.contains {
            (65...90).contains($0.value) || (97...122).contains($0.value)
        }
    }

    private static func containsHan(_ text: String) -> Bool {
        text.unicodeScalars.contains {
            (0x3400...0x4DBF).contains($0.value) || (0x4E00...0x9FFF).contains($0.value)
                || (0xF900...0xFAFF).contains($0.value) || (0x20000...0x323AF).contains($0.value)
        }
    }

    // Keep ASCII syntax by default. Only ordinary prose separators may disappear;
    // dots inside paths, URL punctuation and punctuation in brackets remain significant.
    private static func contentSignature(_ text: String) -> String {
        let scalars = Array(text.unicodeScalars)
        let prose = CharacterSet(charactersIn: "，。！？、；：…“”‘’")
        let separators = CharacterSet(charactersIn: ".,!?;:")
        let operators = CharacterSet(charactersIn: "=<>!&|+-*/%~^([{\\")
        var result = String.UnicodeScalarView()
        var depth = 0
        func isWord(_ scalar: Unicode.Scalar?) -> Bool {
            guard let scalar else { return false }
            return scalar.isASCII && (CharacterSet.alphanumerics.contains(scalar) || scalar == "_")
        }
        for (index, scalar) in scalars.enumerated() {
            if scalar == "(" || scalar == "[" || scalar == "{" { depth += 1 }
            let next = index + 1 < scalars.count ? scalars[index + 1] : nil
            let followsSyntax = next.map { operators.contains($0) } ?? false
            let significantSeparator = depth > 0 || isWord(next) || followsSyntax
            if !CharacterSet.whitespacesAndNewlines.contains(scalar), !prose.contains(scalar),
               !separators.contains(scalar) || significantSeparator {
                result.append(scalar)
            }
            if scalar == ")" || scalar == "]" || scalar == "}" { depth = max(0, depth - 1) }
        }
        return String(result)
    }
}
