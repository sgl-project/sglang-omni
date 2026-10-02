import Foundation

/// Format an explicitly spelled word after the model has already chosen an edit.
/// Never guesses which original word a spoken name refers to.
enum SpokenSpelling {
    struct Reference {
        let joined: String
        let range: NSRange
    }

    static func references(in instruction: String) -> [Reference] {
        let pattern = #"(?:拼作|拼写为|字母是|改成|改为|写作|是)\s*((?:[A-Za-z][\s.、·]+){1,31}[A-Za-z])\.?(?=\s*(?:[。！？!?，,；;]|$))"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        let source = instruction as NSString
        let found: [Reference] = regex.matches(in: instruction, range: NSRange(location: 0, length: source.length)).compactMap { match in
            let prefix = source.substring(to: match.range.location)
            guard !["不要", "不用", "别", "不"].contains(where: { prefix.hasSuffix($0) }) else { return nil }
            return Reference(joined: source.substring(with: match.range(at: 1)).filter { $0.isASCII && $0.isLetter },
                             range: match.range(at: 1))
        }
        if !found.isEmpty { return found }
        return completeReference(in: instruction).map { [$0] } ?? []
    }

    /// Preserve the recorded instruction; normalize only its explicitly spelled RHS
    /// in the model request. This is formatting, not an inferred source-word mapping.
    static func normalizeInstruction(_ instruction: String, terminology: String) -> String {
        var result = instruction
        for reference in references(in: instruction).reversed() {
            result = (result as NSString).replacingCharacters(in: reference.range,
                with: preferredCase(reference.joined, instruction: instruction, terminology: terminology))
        }
        return result
    }

    private static func preferredCase(_ spelling: String, instruction: String, terminology: String) -> String {
        guard !instruction.contains("大写"), !instruction.contains("小写"),
              let regex = try? NSRegularExpression(pattern: #"(?<![A-Za-z])"# + spelling + #"(?![A-Za-z])"#,
                                                   options: .caseInsensitive) else { return spelling }
        let terms = Set(regex.matches(in: terminology, range: NSRange(location: 0, length: terminology.utf16.count))
            .map { (terminology as NSString).substring(with: $0.range) })
        return terms.count == 1 ? terms.first! : spelling
    }

    /// A narrow fallback for a complete "ASCII name 是 L E T T E R S" command.
    /// Only an exact Latin + Mandarin-pinyin match qualifies, with one source span.
    /// No edit distance, whole-Chinese name guessing, paths or compound commands.
    private static func completeReference(in instruction: String) -> Reference? {
        let command = #"^\s*([A-Za-z][A-Za-z\s.]*?)\s*是\s*((?:[A-Za-z][\s.、·]+){1,31}[A-Za-z]|[A-Za-z]{3,32})[。.!]?\s*$"#
        guard let regex = try? NSRegularExpression(pattern: command),
              let match = regex.firstMatch(in: instruction, range: NSRange(location: 0, length: instruction.utf16.count)) else { return nil }
        let source = instruction as NSString
        let name = source.substring(with: match.range(at: 1)).filter { $0.isASCII && $0.isLetter }.lowercased()
        let spelled = source.substring(with: match.range(at: 2)).filter { $0.isASCII && $0.isLetter }
        guard name.count >= 3, name == spelled.lowercased() else { return nil }
        return Reference(joined: spelled, range: match.range(at: 2))
    }

    static func explicitRevision(original: String, instruction: String, terminology: String) -> TextRevision? {
        guard let joined = completeReference(in: instruction)?.joined ?? scopedName(in: instruction) else { return nil }
        let name = joined.lowercased()
        let tokens = try! NSRegularExpression(pattern: #"(?<![A-Za-z0-9_/\\])(?:[A-Za-z][. ]*){2,32}[\p{Han}]{1,6}"#)
        var candidates: [NSRange] = []
        let originalString = original as NSString
        for token in tokens.matches(in: original, range: NSRange(location: 0, length: originalString.length)) {
            var prefix = ""
            for character in originalString.substring(with: token.range) {
                prefix.append(character)
                guard !character.isASCII,
                      let latin = prefix.applyingTransform(.mandarinToLatin, reverse: false) else { continue }
                let normalized = latin.folding(options: .diacriticInsensitive, locale: Locale(identifier: "en_US_POSIX"))
                    .filter { $0.isASCII && $0.isLetter }.lowercased()
                if normalized == name { candidates.append(NSRange(location: token.range.location, length: prefix.utf16.count)) }
            }
        }
        guard candidates.count == 1, let range = candidates.first else { return nil }
        let spelling = preferredCase(joined, instruction: instruction, terminology: terminology)
        return TextRevision(original: original, corrected: originalString.replacingCharacters(in: range, with: spelling))
    }

    /// "S.G. Lang 的 Lang 是 L.A.N.G." identifies a suffix of the named word.
    /// Require that exact suffix and spelling; unrelated parts and extra commands fail.
    private static func scopedName(in instruction: String) -> String? {
        let command = #"^\s*([A-Za-z][A-Za-z\s.]*?)\s*的\s*([A-Za-z][A-Za-z\s.]*?)\s*是\s*((?:[A-Za-z][\s.、·]+){1,31}[A-Za-z]|[A-Za-z]{2,32})[。.!]?\s*$"#
        guard let regex = try? NSRegularExpression(pattern: command),
              let match = regex.firstMatch(in: instruction, range: NSRange(location: 0, length: instruction.utf16.count)) else { return nil }
        let source = instruction as NSString
        let letters = { (index: Int) in source.substring(with: match.range(at: index)).filter { $0.isASCII && $0.isLetter } }
        let full = letters(1), part = letters(2), spelled = letters(3)
        guard part.count >= 2, full.count > part.count,
              full.lowercased().hasSuffix(part.lowercased()), part.lowercased() == spelled.lowercased() else { return nil }
        return String(full.dropLast(part.count)) + spelled
    }

    static func normalize(_ corrected: String, original: String, instruction: String, terminology: String) -> String {
        var result = corrected
        for reference in references(in: instruction) {
            let revision = TextRevision(original: original, corrected: result)
            guard !revision.isEmpty else { continue }
            let letters = reference.joined.map { NSRegularExpression.escapedPattern(for: String($0)) }
            let pattern = #"(?<![A-Za-z.])"# + letters.joined(separator: #"[\s.、·]*"#) + #"(?![A-Za-z])"#
            guard let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive) else { continue }
            let matches = regex.matches(in: result, range: NSRange(location: 0, length: result.utf16.count))
            guard matches.count == 1, let match = matches.first,
                  NSIntersectionRange(match.range, NSRange(location: revision.range.location,
                                                          length: revision.replacement.utf16.count)).length > 0 else { continue }
            let spelling = (result as NSString).substring(with: match.range)
            // Do not reformat text the model retained unchanged between other edits.
            guard !original.contains(spelling) else { continue }
            let joined = preferredCase(spelling.filter { $0.isASCII && $0.isLetter },
                                       instruction: instruction, terminology: terminology)
            result = (result as NSString).replacingCharacters(in: match.range, with: joined)
        }
        return result
    }
}
