import Foundation

/// LLMs interpret natural editing instructions; exact anchors bound local edits.
struct CorrectionPlan {
    struct ValidationError: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    let original: String
    let instruction: String
    let direct: String?
    let guidance: String
    private let localRange: NSRange?
    private let replacement: String?
    private let singleCharacter: String?

    init(original: String, instruction: String, background: String) throws {
        guard !original.isEmpty, !instruction.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              original.utf8.count + instruction.utf8.count + background.utf8.count <= 5_000 else {
            throw DictationError("上一段或修改意见为空，或输入过长；请精简后重试。")
        }
        self.original = original
        self.instruction = instruction
        let command = Self.command(instruction)
        let normalized = Self.command(SpokenSpelling.normalizeInstruction(instruction, terminology: background))
        var direct: String?
        var localRange: NSRange?
        var replacement: String?
        var character: String?
        var guidance: [String] = []

        if Self.isUnchanged(command) {
            direct = original
        } else if Self.deletesAll(command) {
            direct = ""
        } else if let spelling = SpokenSpelling.explicitRevision(original: original, instruction: instruction, terminology: background) {
            direct = spelling.corrected
        } else if let spelling = Self.groups(#"^([A-Za-z][A-Za-z .]*)是([A-Za-z]+)$"#, normalized),
                  Self.phonetic(spelling[0]) == Self.phonetic(spelling[1]) {
            let ranges = Self.candidateRanges(original, hint: spelling[1]).filter {
                let text = (original as NSString).substring(with: $0)
                return Self.phonetic(text) == Self.phonetic(spelling[1]) && Self.matches(text, in: original).contains($0)
            }
            guard ranges.count == 1, let range = ranges.first else {
                throw DictationError("无法唯一定位这个拼写对应的原文，请说明修改哪一处。")
            }
            direct = (original as NSString).replacingCharacters(in: range, with: spelling[1])
        } else if let parts = Self.groups(#"^(.+?)的(.+?)是([A-Za-z]{2,32})$"#, normalized),
                  let scope = Self.literal(parts[0]), let field = Self.literal(parts[1]) {
            let scopes = Self.matches(scope, in: original)
            if scopes.count > 1 { throw DictationError("原文有多个同名对象，请明确修改哪一处。") }
            if let scopeRange = scopes.first {
                let inside = Self.matches(field, in: scope)
                if inside.count == 1, let part = inside.first {
                    let range = NSRange(location: scopeRange.location + part.location, length: part.length)
                    direct = (original as NSString).replacingCharacters(in: range, with: parts[2])
                }
            }
        } else if let parts = Self.replacementParts(normalized) {
            let addressed = Self.address(parts[0])
            let fields = ["名字", "姓名", "人名", "负责人", "项目名", "项目", "名称", "日期", "时间", "金额", "费用", "备注",
                          "手机号", "电话", "邮箱", "地址", "地点", "城市"]
            var found = Self.matches(addressed.text, in: original)
            if fields.contains(addressed.text), Self.quoted(parts[0]) == nil {
                let pattern = NSRegularExpression.escapedPattern(for: addressed.text) + #"\s*[:：]\s*([^\n，,。；;！？!?]+)"#
                let regex = try NSRegularExpression(pattern: pattern)
                found = regex.matches(in: original, range: NSRange(location: 0, length: original.utf16.count)).compactMap { match in
                    let value = (original as NSString).substring(with: match.range(at: 1)).trimmingCharacters(in: .whitespaces)
                    return value.isEmpty ? nil : (original as NSString).range(of: value, options: .literal, range: match.range(at: 1))
                }
            }
            if found.isEmpty, Self.quoted(parts[0]) != nil {
                throw DictationError("原文没有指定的文字，请确认要修改的内容。")
            }
            if found.isEmpty, ["手机号", "电话"].contains(addressed.text),
               !original.contains(where: { $0.isNumber }),
               !["手机", "电话", "号码"].contains(where: original.contains) {
                throw DictationError("原文没有可定位的电话号码，请确认要修改的内容。")
            }
            if found.isEmpty, addressed.text == "邮箱", !original.contains("@"),
               !["邮箱", "邮件"].contains(where: original.contains) {
                throw DictationError("原文没有可定位的邮箱，请确认要修改的内容。")
            }
            // A short literal source must exist. A unique exact homophone can
            // bound an LLM edit, but is never executed as an automatic mapping.
            // Field names and referring expressions still need model interpretation.
            if found.isEmpty, !fields.contains(addressed.text), Self.literal(parts[1]) != nil,
               addressed.text.range(of: #"^[\p{Han}A-Za-z0-9_-]{1,32}$"#, options: .regularExpression) != nil,
               !["的", "这个", "那个", "前面", "后面", "最后", "整段", "全文", "上一段", "这一段", "开头", "结尾", "不要", "别把"]
                .contains(where: addressed.text.contains) {
                found = Self.candidateRanges(original, hint: addressed.text).filter {
                    let text = (original as NSString).substring(with: $0)
                    return Self.phonetic(text) == Self.phonetic(addressed.text) && Self.matches(text, in: original).contains($0)
                }
                if found.isEmpty { throw DictationError("原文没有指定的文字，请确认要修改的内容。") }
            }
            if let ordinal = addressed.ordinal, !found.isEmpty {
                guard ordinal > 0, ordinal <= found.count else { throw DictationError("原文没有指定的第几处。") }
                found = [found[ordinal - 1]]
            } else if found.count > 1, Self.literal(parts[1]) != nil {
                throw DictationError("原文中有多处相同文字，请说明第几个或补充位置。")
            }
            // These bounds constrain a model result, never preempt a natural-language edit.
            if found.count == 1, let range = found.first {
                localRange = range
                replacement = Self.quoted(parts[1])
                let value = (original as NSString).substring(with: range)
                let prefix = String((original as NSString).substring(to: range.location).suffix(16))
                let suffix = String((original as NSString).substring(from: NSMaxRange(range)).prefix(16))
                guidance.append("局部目标是原文中的“\(value)”；前文“\(prefix)”，后文“\(suffix)”。其他原文必须保留。")
                if Self.quoted(parts[0]) != nil, let replacement {
                    direct = (original as NSString).replacingCharacters(in: range, with: replacement)
                }
            }
        }

        if direct == nil {
            if let filling = Self.groups(#"(?:是|改成|改为|换成)([\p{Han}]{2,8})的([\p{Han}])$"#, command),
               filling[0].contains(filling[1]), command.contains("一个字"),
               !command.contains("不是"), !command.contains("不要") {
                character = filling[1]
                guidance.append("本次只改一个汉字，正确字为“\(filling[1])”。人名的其他字（包括姓氏）必须保留。")
            }
            // Explicit use of a unique background name is a literal, authorized mapping.
            if Self.groups(#"^((?:请)?(?:按|按照|用)(?:个人)?背景(?:里|中)?的?项目名(?:称)?(?:改|修改|更正|替换)?)$"#, command) != nil {
                let regex = try NSRegularExpression(pattern: #"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+"#)
                let terms = Set(regex.matches(in: background, range: NSRange(location: 0, length: background.utf16.count))
                    .map { (background as NSString).substring(with: $0.range) })
                var matches: [(NSRange, String)] = []
                for term in terms {
                    let stem = term.components(separatedBy: CharacterSet(charactersIn: "-_")).first!
                    for range in Self.candidateRanges(original, hint: stem)
                        where Self.phonetic((original as NSString).substring(with: range)) == Self.phonetic(stem) {
                        let existing = Self.matches(term, in: original).first { $0.location == range.location }
                        matches.append((existing ?? range, term))
                    }
                }
                if matches.count == 1, let match = matches.first {
                    direct = (original as NSString).replacingCharacters(in: match.0, with: match.1)
                } else if matches.count > 1 { throw DictationError("背景或原文中有多个项目匹配，请明确项目名和位置。") }
            }
        }
        self.direct = direct
        self.localRange = localRange
        self.replacement = replacement
        self.singleCharacter = character
        self.guidance = guidance.joined(separator: "\n")
    }

    func applying(scope: String, text: String) throws -> String {
        if scope == "clarify" {
            guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, text.count <= 200 else {
                throw DictationError("模型未提供有效的澄清问题，请明确说明编辑要求。")
            }
            throw DictationError("需要确认：" + text)
        }
        guard ["local", "rewrite"].contains(scope) else { throw DictationError("模型返回的编辑范围无效。") }
        guard text.utf16.count <= 8_000 else { throw DictationError("编辑结果过长，上一段保持不变。") }
        let compact = { (value: String) in value.lowercased().filter { !$0.isWhitespace && !$0.isPunctuation } }
        let command = compact(instruction)
        if !command.isEmpty, compact(text).contains(command), !compact(original).contains(command) {
            throw ValidationError(message: "修改意见不能出现在正文中。只返回编辑后的原文，不复述修改意见。")
        }
        if text.hasPrefix("```"), !original.hasPrefix("```") {
            throw ValidationError(message: "不要加Markdown代码围栏，只返回编辑后的正文。")
        }
        if let direct, !text.utf16.elementsEqual(direct.utf16) {
            throw ValidationError(message: "必须使用本次明确指定的文字，不能改变其他内容。")
        }
        if scope == "rewrite" {
            guard localRange == nil, singleCharacter == nil else {
                throw ValidationError(message: "本次明确指定了局部修改，scope应为local，只改指定内容。")
            }
            guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw ValidationError(message: "整段重写结果不能为空；不能把重写当成删除全部。")
            }
            return text
        }
        let revision = TextRevision(original: original, corrected: text)
        if let singleCharacter, revision.isEmpty, !original.contains(singleCharacter) {
            throw ValidationError(message: "原文尚未包含指定的“\(singleCharacter)”，请执行本次单字更正，不能原样返回。")
        }
        if let singleCharacter, !revision.isEmpty {
            let changed = (original as NSString).substring(with: revision.range)
            guard changed.count == 1, revision.replacement == singleCharacter else {
                throw ValidationError(message: "本次只应将一个汉字改成“\(singleCharacter)”，不得删除或改写其他字。")
            }
        }
        if let range = localRange {
            let prefix = (original as NSString).substring(to: range.location)
            let suffix = (original as NSString).substring(from: NSMaxRange(range))
            let length = text.utf16.count - prefix.utf16.count - suffix.utf16.count
            guard length >= 0, text.utf16.starts(with: prefix.utf16), text.utf16.reversed().starts(with: suffix.utf16.reversed()) else {
                throw ValidationError(message: "局部修改越界。" + guidance)
            }
            let middle = (text as NSString).substring(with: NSRange(location: prefix.utf16.count, length: length))
            if let replacement, !middle.utf16.elementsEqual(replacement.utf16) {
                throw ValidationError(message: "指定的新文字是“\(replacement)”，必须原样使用。")
            }
        }
        if text.isEmpty, !Self.deletesAll(Self.command(instruction)) {
            throw ValidationError(message: "未明确要求删除全部，不能返回空正文。")
        }
        return text
    }

    private static func quoted(_ value: String) -> String? {
        let value = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard ["“", "「", "\""].contains(where: value.hasPrefix) else { return nil }
        return literal(value)
    }

    private static func command(_ value: String) -> String {
        var result = value.trimmingCharacters(in: .whitespacesAndNewlines)
        // Only strip sentence punctuation outside a closing quote, never inside a literal.
        if result.hasSuffix("。") || result.hasSuffix(".") { result.removeLast() }
        for suffix in ["，其他不变", ",其他不变", "，其它不变", "，其他保持不变", "，其余不变"] where result.hasSuffix(suffix) {
            result.removeLast(suffix.count)
        }
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func literal(_ value: String) -> String? {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        for (left, right) in [("“", "”"), ("「", "」"), ("\"", "\"")] where trimmed.hasPrefix(left) && trimmed.hasSuffix(right) && trimmed.count >= 2 {
            let inner = String(trimmed.dropFirst().dropLast())
            guard !inner.contains(left), !inner.contains(right), !inner.isEmpty else { return nil }
            return inner
        }
        guard !trimmed.contains(where: { "，,；;。！？!?\n“”「」\"".contains($0) }),
              !["然后", "再把", "并把", "同时", "其他", "其它", "不要", "不需要"].contains(where: trimmed.contains) else { return nil }
        return trimmed
    }

    private static func address(_ value: String) -> (text: String, ordinal: Int?) {
        if ["“", "「", "\""].contains(where: value.trimmingCharacters(in: .whitespaces).hasPrefix), let literal = literal(value) {
            return (literal, nil)
        }
        let value = literal(value) ?? value.trimmingCharacters(in: .whitespacesAndNewlines)
        if let parts = groups(#"^第([一二三四五六七八九十]|[0-9]{1,2})个(.+)$"#, value) {
            let digits = ["一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10]
            return (literal(parts[1]) ?? parts[1], digits[parts[0]] ?? Int(parts[0]))
        }
        return (value, nil)
    }

    private static func groups(_ pattern: String, _ value: String) -> [String]? {
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let match = regex.firstMatch(in: value, range: NSRange(location: 0, length: value.utf16.count)) else { return nil }
        return (1..<match.numberOfRanges).map { (value as NSString).substring(with: match.range(at: $0)) }
    }

    private static func replacementParts(_ value: String) -> [String]? {
        // A quoted source may itself contain an editing verb. Consume the whole
        // quoted source before looking for the separator.
        let source = #"(“[^”]+”|「[^」]+」|"[^"]+"|[^“”「」"]+?)"#
        let prefix = #"^(?:请帮我|请|帮我)?(?:把|将)?"# + source
        return groups(prefix + #"(?:替换成|替换为|改成|改为|换成)(.+)$"#, value)
            ?? groups(prefix + #"改(.+)$"#, value)
    }

    private static func matches(_ needle: String, in text: String) -> [NSRange] {
        guard !needle.isEmpty else { return [] }
        let value = text as NSString
        var search = NSRange(location: 0, length: value.length), ranges: [NSRange] = []
        func identifier(_ character: Character?) -> Bool {
            guard let character else { return false }
            return character.isASCII && (character.isLetter || character.isNumber || "_/\\".contains(character))
        }
        while search.length > 0 {
            let range = value.range(of: needle, options: .literal, range: search)
            guard range.location != NSNotFound else { break }
            if let swiftRange = Range(range, in: text) {
                let previous = text[..<swiftRange.lowerBound].last, next = text[swiftRange.upperBound...].first
                if !(identifier(needle.first) && identifier(previous)) && !(identifier(needle.last) && identifier(next)) {
                    ranges.append(range)
                }
            }
            search = NSRange(location: NSMaxRange(range), length: value.length - NSMaxRange(range))
        }
        return ranges
    }

    private static func candidateRanges(_ text: String, hint: String?) -> [NSRange] {
        var ranges: [NSRange] = []
        func append(_ range: NSRange) {
            if !ranges.contains(range), range.length > 0 { ranges.append(range) }
        }
        // Candidate spans are used only for exact phonetic matching, not model-generated offsets.
        text.enumerateSubstrings(in: text.startIndex..<text.endIndex, options: [.byWords, .localized]) { _, range, _, _ in
            append(NSRange(range, in: text))
        }
        // Tokenizers may split an unfamiliar name. Only add these windows when
        // the instruction names a source term; phonetic matching filters them next.
        if let hint, (2...12).contains(hint.count) {
            let indices = Array(text.indices) + [text.endIndex]
            if hint.count < indices.count {
                for start in 0..<(indices.count - hint.count) {
                    let range = indices[start]..<indices[start + hint.count]
                    if text[range].allSatisfy({ $0.isLetter || $0.isNumber }) { append(NSRange(range, in: text)) }
                }
            }
        }
        if let regex = try? NSRegularExpression(pattern: #"(?<![A-Za-z0-9_/\\])(?:[A-Za-z][. ]*){2,32}[\p{Han}]{1,3}"#) {
            for match in regex.matches(in: text, range: NSRange(location: 0, length: text.utf16.count)) {
                let token = (text as NSString).substring(with: match.range)
                var prefix = ""
                for character in token {
                    prefix.append(character)
                    if !character.isASCII { append(NSRange(location: match.range.location, length: prefix.utf16.count)) }
                }
            }
        }
        return ranges.sorted { $0.location == $1.location ? $0.length < $1.length : $0.location < $1.location }
    }

    private static func phonetic(_ value: String) -> String {
        (value.applyingTransform(.mandarinToLatin, reverse: false) ?? value)
            .folding(options: [.diacriticInsensitive, .caseInsensitive], locale: Locale(identifier: "en_US_POSIX"))
            .filter { $0.isLetter || $0.isNumber }
    }

    private static func isUnchanged(_ command: String) -> Bool {
        let compact = command.filter { !$0.isWhitespace && !$0.isPunctuation }.lowercased()
        return ["不改了", "保持原样", "不改了保持原样", "取消", "不用修改", "不要修改", "不要改了", "keepunchanged", "cancel"].contains(compact)
    }

    private static func deletesAll(_ command: String) -> Bool {
        let compact = command.lowercased().filter { !$0.isWhitespace && !$0.isPunctuation }
        return [
            "^(请|请帮我|帮我)?((删除|删掉|清空)(全部|整段|上一段|这一段)|(全部|整段|上一段|这一段)(删除|删掉|清空)|把(整段|上一段|这一段)(全部|都)?(删除|删掉|清空))$",
            "^(please)?(delete|clear|remove)(all|everything|alltext|allthetext|the(entire|whole)(text|paragraph)|the(last|previous)paragraph)$",
        ].contains { compact.range(of: $0, options: .regularExpression) != nil }
    }
}
