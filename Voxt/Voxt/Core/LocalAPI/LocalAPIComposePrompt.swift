// LocalAPIComposePrompt.swift
// Provides the mode-specific instruction that replaces the global transcription prompt for
// one Local API compose request. The raw speech reaches the model through the same
// template variable the global prompt uses.

import Foundation

nonisolated enum LocalAPIComposePrompt {
    static func hanShare(in text: String) -> Double? {
        let letters = text.unicodeScalars.filter { CharacterSet.letters.contains($0) }
        guard !letters.isEmpty else { return nil }
        let hanCount = letters.filter { scalar in
            switch scalar.value {
            case 0x3400...0x4DBF, 0x4E00...0x9FFF, 0xF900...0xFAFF, 0x20000...0x323AF: return true
            default: return false
            }
        }.count
        return Double(hanCount) / Double(letters.count)
    }

    static func preservingDictationScript(_ cleanedText: String, rawText: String) -> String {
        guard let rawShare = hanShare(in: rawText), let cleanedShare = hanShare(in: cleanedText) else {
            return cleanedText
        }
        let changedScript = (rawShare >= 0.5 && cleanedShare <= 0.1)
            || (rawShare <= 0.1 && cleanedShare >= 0.5)
        return changedScript ? rawText : cleanedText
    }

    /// A mixed transcript needs at least two Han characters and four Latin letters.
    /// Below that, a single initial or symbol is too little evidence for a retry.
    /// Count loss, not language dominance: a short Chinese span inside English matters.
    /// The model follows a line next to the transcript more than one at the top of the prompt.
    static func withSection(_ section: String, beforeTranscriptIn prompt: String) -> String {
        guard let transcript = prompt.range(of: EnhancementPromptResolver.rawTranscriptionTemplateVariable),
              let boundary = prompt.range(of: "\n\n", options: .backwards, range: prompt.startIndex..<transcript.lowerBound)
        else { return prompt + "\n\n" + section }
        return prompt.replacingCharacters(in: boundary, with: "\n\n" + section + "\n\n")
    }

    static func losesMixedDictationContent(_ output: String, rawText: String) -> Bool {
        func counts(_ text: String) -> (han: Int, latinLetters: Int, latinWords: Int) {
            let han = text.unicodeScalars.filter { scalar in
                switch scalar.value {
                case 0x3400...0x4DBF, 0x4E00...0x9FFF, 0xF900...0xFAFF, 0x20000...0x323AF: return true
                default: return false
                }
            }.count
            let regex = try? NSRegularExpression(pattern: #"\p{Latin}+"#)
            let words = (regex?.matches(in: text, range: NSRange(text.startIndex..., in: text)) ?? []).compactMap { match in
                Range(match.range, in: text).map { String(text[$0]) }
            }
            return (han, words.reduce(0) { $0 + $1.unicodeScalars.count }, words.count)
        }
        let source = counts(rawText)
        guard source.han >= 2, source.latinLetters >= 4 else { return false }
        let result = counts(output)
        return Double(result.han) < Double(source.han) * 0.3
            || Double(result.latinWords) < Double(source.latinWords) * 0.3
    }

    /// Which dictation prompt a transcript gets, by script: kana means Japanese; Chinese with English words
    /// (20% Han or more) stays Chinese; anything else is English. At a 50% threshold, mixed zh-en
    /// speech got the English prompt and came back translated.
    enum DictationLanguage {
        case chinese, english, japanese

        var instruction: String {
            switch self {
            case .chinese: chineseDictationInstruction
            case .english: englishDictationInstruction
            case .japanese: japaneseDictationInstruction
            }
        }
        var transcriptPrefix: String {
            switch self {
            case .chinese: "口述转写："
            case .english: "Speech transcript: "
            case .japanese: "書き起こし："
            }
        }
        var outputCue: String {
            switch self {
            case .chinese: "输出："
            case .english: "Output:"
            case .japanese: "出力："
            }
        }
    }

    static func dictationLanguage(of transcript: String) -> DictationLanguage {
        if transcript.unicodeScalars.contains(where: { (0x3040...0x30FF).contains($0.value) }) { return .japanese }
        return (hanShare(in: transcript) ?? 0) >= 0.2 ? .chinese : .english
    }

    /// Qwen3.5-4B writes "4 時" and "10 分" even when the prompt says not to.
    static func withoutDigitSpacing(_ text: String) -> String {
        text.replacingOccurrences(
            of: #"(?<=[0-9]) (?=[\u3040-\u30FF\u3400-\u9FFF])|(?<=[\u3040-\u30FF\u3400-\u9FFF]) (?=[0-9])"#,
            with: "", options: .regularExpression)
    }

    /// The recognizer spells some acronyms letter by letter inside Chinese or Japanese ("放到 C I C D 里面").
    /// Joins a run of two or more spaced capitals with Han or kana (or the text's edge) on both
    /// sides, so English like "Plan A I think" is left alone; a few acronyms get back their slash.
    static func joinedSpelledAcronyms(_ text: String) -> String {
        let slashed = ["CICD": "CI/CD", "IO": "I/O", "AB": "A/B", "TCPIP": "TCP/IP"]
        let cjk = #"[\u3040-\u30FF\u3400-\u9FFF\u3000-\u303F\uFF00-\uFFEF]"#
        guard let regex = try? NSRegularExpression(pattern: "(?<=^|\(cjk) ?)((?:[A-Z] )+[A-Z])(?= ?\(cjk)|$)") else { return text }
        var result = text
        for match in regex.matches(in: text, range: NSRange(text.startIndex..., in: text)).reversed() {
            guard let range = Range(match.range(at: 1), in: result) else { continue }
            let joined = result[range].replacingOccurrences(of: " ", with: "")
            result.replaceSubrange(range, with: slashed[joined] ?? joined)
        }
        return result
    }

    /// The selected text model as compose needs to see it. Every compose mode consults the
    /// model; the cases describe whether the currently selected model can run it at all.
    enum TextModel: Equatable, Sendable {
        case disabled
        case customLLM(ready: Bool)
        case remoteLLM(configured: Bool)
        case appleIntelligence
    }

    /// nil when the model can run compose operations, otherwise the failure reason.
    static func unavailabilityReason(for model: TextModel) -> String? {
        switch model {
        case .disabled, .customLLM(ready: false), .remoteLLM(configured: false):
            "No text model is available. Choose one in Voxt Settings > Transcription."
        case .customLLM(ready: true), .remoteLLM(configured: true), .appleIntelligence:
            nil
        }
    }

    /// The instruction that replaces the global transcription prompt for one compose
    /// request. Every mode — dictate included — runs through the selected text model;
    /// only /v1/dictations keeps the legacy global prompt.
    /// Stored wording pairs never override an explicit tone or a translation target.
    static func styleExamplesForPrompt(_ options: LocalAPIComposeOptions) -> [LocalAPIStyleExample] {
        guard options.style == .automatic, options.mode != .translate else { return [] }
        return options.styleExamples
    }

    static func instruction(
        for options: LocalAPIComposeOptions, rawTranscription: String = "",
        editRetry: EditRetry? = nil,
        retryingContextLeak: Bool = false,
        retryingWritePlaceholder: Bool = false,
        retryingMixedDictation: Bool = false
    ) -> String? {
        let styleExamples = styleExamplesForPrompt(options)
        let custom = options.customPrompt.map { $0.replacingOccurrences(of: targetLanguageTemplateVariable, with: options.targetLanguage ?? "") }
        switch options.mode {
        case .dictate:
            let language = dictationLanguage(of: rawTranscription)
            let usesChinese = language == .chinese
            var task = custom ?? (retryingMixedDictation
                ? (usesChinese ? previousChineseDictationInstruction : previousEnglishDictationInstruction)
                : language.instruction)
            if retryingContextLeak {
                task += usesChinese
                    ? "\n\n第一次输出重复了输入框里已有的文字。只输出这次口述整理后的内容，不要重复光标前后的文字。"
                    : "\n\nThe first attempt repeated text that is already in the field. Output only the cleaned transcript of this speech; never repeat the text before or after the cursor."
            }
            // No cursor context for plain cleanup: with it, Qwen3.5-4B copied the earlier text
            // into 4 of 6 outputs; without it, 6 of 6 were clean in one call.
            let prompt = body(
                task: task,
                selection: nil,
                context: nil,
                style: options.style,
                vocabulary: options.vocabulary,
                languageVariants: options.languageVariants,
                styleExamples: styleExamples,
                transcriptPrefix: language.transcriptPrefix,
                outputCue: retryingMixedDictation ? nil : language.outputCue
            )
            return prompt
        case .translate:
            let target = options.targetLanguage ?? ""
            var task = custom ?? translateTask(target: target)
            // Qwen3.5-4B translated "Thursday... actually, Friday" literally even
            // with examples; writing the corrected sentence first made it drop the old value.
            let corrected = custom == nil ? appliedSelfCorrections(rawTranscription) : rawTranscription
            let twoStep = custom == nil && corrected == rawTranscription && containsCorrectionCue(rawTranscription)
            if twoStep {
                task += """

                First, on a line starting with "Corrected:", write the transcript in its own language with the self-corrections applied.
                Then, on a line starting with "Translation:", write the translation of the corrected line only.
                """
            }
            if custom == nil, let example = translationCorrectionExample(target: target, twoStep: twoStep) {
                task += "\n\nExample:\n" + example
            }
            return body(
                task: task,
                selection: nil,
                context: nil,
                style: options.style,
                vocabulary: options.vocabulary,
                // The explicit translation target is authoritative. Passing other
                // variants made Qwen3.5-4B output Chinese for target en.
                languageVariants: [],
                styleExamples: styleExamples,
                transcriptPrefix: corrected == rawTranscription ? "Speech transcript: " : "Original speech (reference only; do not translate it): ",
                afterTranscript: corrected == rawTranscription ? nil : "Translate this corrected transcript: \(corrected)",
                outputCue: twoStep ? "Corrected:" : "Translation:"
            )
        case .edit:
            let usesChinese = (hanShare(in: options.selectedText) ?? 0) >= 0.5
            var task = custom ?? (usesChinese ? chineseEditInstruction : englishEditInstruction)
            switch editRetry {
            case let .droppedTerms(terms):
                let quoted = terms.map { usesChinese ? "“\($0)”" : "\"\($0)\"" }.joined(separator: usesChinese ? "、" : ", ")
                task += usesChinese
                    ? "\n\n第一次输出删掉了选中文字里的日期或时间。请原样保留\(quoted)。"
                    : "\n\nThe first attempt dropped a date or time from the selected text. Keep \(quoted) exactly as written."
            case .leaked:
                task += usesChinese
                    ? "\n\n第一次输出重复了内容，或抄入了选中范围以外的文字。只输出修改后的选中文字，只输出一遍。"
                    : "\n\nThe first attempt repeated text or copied text from around the selection. Output only the edited selected text, once."
            case .unchanged:
                task += usesChinese
                    ? "\n\n第一次输出未能安全地完成修改：输出与选中文字完全相同。现在执行指令，输出必须与选中文字不同。"
                    : "\n\nThe first attempt did not safely apply the edit: it returned the selected text unchanged. Apply the instruction now; the output must differ from the selected text."
            case .unsafe:
                task += usesChinese
                    ? "\n\n第一次输出未能安全地完成修改。只输出修改后的正文，不得包含修改指令。若两个替换词发音相同，根据原文语境选择正确的写法。"
                    : "\n\nThe first attempt did not safely apply the edit. Output only the revised passage, never the replacement instruction. If both replacement words sound the same, use the spelling that fits the selected text."
            case nil:
                break
            }
            // A whole-field edit sends the field as both selection and context. Repeating it
            // made Qwen3.5-4B copy the text back unchanged for a tone instruction.
            let selected = options.selectedText.trimmingCharacters(in: .whitespacesAndNewlines)
            // A whole-field edit sends the field as both selection and context. Repeating it
            // made Qwen3.5-4B copy the text back unchanged for a tone instruction.
            func unlessSelection(_ context: String) -> String {
                context.trimmingCharacters(in: .whitespacesAndNewlines) == selected ? "" : context
            }
            return body(
                task: task,
                selection: options.selectedText,
                context: (before: unlessSelection(options.contextBefore), after: unlessSelection(options.contextAfter)),
                style: options.style,
                vocabulary: options.vocabulary,
                languageVariants: options.languageVariants,
                styleExamples: styleExamples,
                selectionPrefix: usesChinese ? "选中文字：" : "Selected text: ",
                transcriptPrefix: usesChinese ? "指令：" : "Instruction: ",
                outputCue: usesChinese ? "输出：" : "Output:"
            )
        case .write:
            let usesChinese = (hanShare(in: rawTranscription) ?? 0) >= 0.5
            var task = custom ?? (usesChinese ? chineseWriteInstruction : englishWriteInstruction)
            if retryingWritePlaceholder {
                task += usesChinese
                    ? "\n\n第一次输出包含占位符。不知道的姓名或细节直接省略，输出可以直接发送的完整消息。"
                    : "\n\nThe first attempt contained a placeholder. Leave out any name or detail you do not know and output a complete message that can be sent as is."
            }
            return body(
                task: task,
                selection: nil,
                context: (before: options.contextBefore, after: options.contextAfter),
                style: options.style,
                vocabulary: options.vocabulary,
                languageVariants: options.languageVariants,
                styleExamples: styleExamples,
                transcriptPrefix: usesChinese ? "指令：" : "Instruction: ",
                outputCue: usesChinese ? "消息：" : "Message:"
            )
        }
    }

    static func compose(
        rawTranscription: String,
        options: LocalAPIComposeOptions,
        generate: (String) async throws -> String
    ) async throws -> String {
        // A list item is edited without its marker, which then goes back on: "2." alone was read
        // as a time by the date guard, and the edit was not applied.
        if options.mode == .edit, let marker = listMarker(of: options.selectedText) {
            var item = options
            item.selectedText = String(options.selectedText.dropFirst(marker.count))
            let edited = try await compose(rawTranscription: rawTranscription, options: item, generate: generate)
            return listMarker(of: edited) == nil ? marker + edited : edited
        }
        guard let initialPrompt = instruction(for: options, rawTranscription: rawTranscription) else {
            throw LocalAPIError(message: "This operation has no compose prompt.")
        }
        var first = try await generate(initialPrompt).trimmingCharacters(in: .whitespacesAndNewlines)
        if options.mode == .dictate, EnhancementLanguageGuard.languageFlipped(input: rawTranscription, output: first) {
            let kept = try await EnhancementLanguageGuard.keepingInputLanguage(
                input: rawTranscription,
                output: first,
                retry: {
                    try await generate(withSection(
                        EnhancementLanguageGuard.retryInstruction(for: rawTranscription),
                        beforeTranscriptIn: initialPrompt
                    )).trimmingCharacters(in: .whitespacesAndNewlines)
                },
                log: { VoxtLog.warning("Local API dictate: \($0)") }
            )
            if kept == rawTranscription { return rawTranscription }
            first = kept
        }
        if options.mode == .dictate, leaksContext(first, rawTranscription: rawTranscription, options: options) {
            VoxtLog.warning("Local API dictate output repeated the cursor context; retrying once.")
            guard let retryPrompt = instruction(for: options, rawTranscription: rawTranscription, retryingContextLeak: true) else {
                return rawTranscription
            }
            let retried = try await generate(retryPrompt).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !retried.isEmpty, !leaksContext(retried, rawTranscription: rawTranscription, options: options) else {
                VoxtLog.warning("Local API dictate output repeated the cursor context after retry; returning raw transcript.")
                return rawTranscription
            }
            return joinedSpelledAcronyms(withoutDigitSpacing(laidOutMessage(numberedCountedLines(appliedSelfCorrections(retried)))))
        }
        if options.mode == .dictate, losesMixedDictationContent(first, rawText: rawTranscription) {
            VoxtLog.warning("Local API dictate cleanup lost mixed-language content; retrying previous prompt once.")
            do {
                guard let previousPrompt = instruction(
                    for: options, rawTranscription: rawTranscription, retryingMixedDictation: true
                ) else { return rawTranscription }
                let retried = try await generate(previousPrompt).trimmingCharacters(in: .whitespacesAndNewlines)
                if !losesMixedDictationContent(retried, rawText: rawTranscription) { return joinedSpelledAcronyms(withoutDigitSpacing(retried)) }
            } catch {
                if error is CancellationError { throw error }
            }
            VoxtLog.warning("Local API dictate cleanup discarded after retry: mixed-language content lost; returning raw transcript.")
            return rawTranscription
        }
        // Same condition as the two-step translate prompt in instruction(for:).
        if options.mode == .translate, options.customPrompt == nil, appliedSelfCorrections(rawTranscription) == rawTranscription,
           containsCorrectionCue(rawTranscription) {
            if let translation = translationAfterCorrection(first) { return translation }
            VoxtLog.warning("Local API translate omitted the Translation line; retrying with the one-step prompt.")
            var oneStep = options
            oneStep.mode = .translate
            guard let prompt = instruction(for: oneStep) else { return first }
            return try await generate(prompt).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if options.mode == .write {
            let source = [rawTranscription, options.contextBefore, options.contextAfter].joined(separator: "\n")
            guard containsPlaceholder(first, allowedIn: source) else { return first }
            VoxtLog.warning("Local API write output contained a placeholder; retrying once.")
            guard let retryPrompt = instruction(for: options, rawTranscription: rawTranscription, retryingWritePlaceholder: true) else {
                throw LocalAPIError(message: "This operation has no compose prompt.")
            }
            let retried = try await generate(retryPrompt).trimmingCharacters(in: .whitespacesAndNewlines)
            guard containsPlaceholder(retried, allowedIn: source) else { return retried }
            VoxtLog.warning("Local API write output still contained a placeholder after one retry; nothing inserted.")
            throw LocalAPIError(message: "Couldn't write that message. Try saying who it's for.")
        }
        if options.mode == .dictate { return joinedSpelledAcronyms(withoutDigitSpacing(laidOutMessage(numberedCountedLines(appliedSelfCorrections(first))))) }
        guard options.mode == .edit else { return first }
        let edited = withoutRepeatedClosing(withoutSurroundingField(first, options: options), options: options)
        let selection = options.selectedText.trimmingCharacters(in: .whitespacesAndNewlines)
        let noChangeRequested = ["leave unchanged", "leave this unchanged", "keep it as is", "do not change", "don't change", "no change", "保持原样", "不要改", "不改"]
            .contains { rawTranscription.localizedCaseInsensitiveContains($0) }
        func rejected(_ text: String) -> Bool {
            text.isEmpty || (!noChangeRequested && text == selection)
                || containsEditInstructionLeakage(text, instruction: rawTranscription, selection: selection)
                || leaksContext(text, rawTranscription: rawTranscription, options: options)
        }
        let droppedFirst = rejected(edited) ? [] : missingDateOrTimeTerms(rawTranscription, selection: selection, output: edited)
        guard rejected(edited) || !droppedFirst.isEmpty else { return edited }
        let retry: EditRetry = !droppedFirst.isEmpty ? .droppedTerms(droppedFirst)
            : !edited.isEmpty && edited == selection ? .unchanged
            : leaksContext(edited, rawTranscription: rawTranscription, options: options) ? .leaked : .unsafe
        guard let retryPrompt = instruction(for: options, rawTranscription: rawTranscription, editRetry: retry) else {
            throw EditNotApplied()
        }
        VoxtLog.warning("Local API edit rejected unchanged output or instruction leakage; retrying once.")
        do {
            let retried = withoutRepeatedClosing(withoutSurroundingField(
                try await generate(retryPrompt).trimmingCharacters(in: .whitespacesAndNewlines), options: options), options: options)
            guard !rejected(retried),
                  missingDateOrTimeTerms(rawTranscription, selection: selection, output: retried).isEmpty else { throw EditNotApplied() }
            return retried
        } catch {
            if error is CancellationError { throw error }
            VoxtLog.warning("Local API edit not applied after one retry; preserving selection.")
            throw EditNotApplied()
        }
    }

    // MARK: Message layout

    /// Lays out a dictated message that opens with a greeting and has a list, either from the
    /// model or as three or more points in one sentence ("Hi Karen, X, Y, and Z. Could you…?
    /// Thanks."): greeting line, paragraphs, numbered points, sign-off line. Only line breaks,
    /// numbering, capitals, end punctuation and a leading filler change. Qwen3.5-4B kept such
    /// points in one sentence; asked to number them, it invented intros
    /// ("Two things:") and dropped a sentence, so the layout is done here.
    static func laidOutMessage(_ text: String) -> String {
        let chinese = (hanShare(in: text) ?? 0) >= 0.5
        var blocks = text.components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
        guard !blocks.isEmpty, let greeting = splitGreeting(blocks[0], chinese: chinese) else { return text }
        blocks[0] = greeting.rest
        var signOff: String?
        if let last = blocks.last, let split = splitSignOff(last, chinese: chinese) {
            blocks[blocks.count - 1] = split.rest
            signOff = split.signOff
        }
        blocks = blocks.flatMap { block in numberedPoints(in: block, chinese: chinese) ?? [block] }.filter { !$0.isEmpty }
        let numbered = try! NSRegularExpression(pattern: #"(?m)^\d+[.)] "#)
        guard blocks.contains(where: { numbered.firstMatch(in: $0, range: NSRange($0.startIndex..., in: $0)) != nil }) else {
            return text
        }
        if !chinese, let first = blocks.first { blocks[0] = first.prefix(1).uppercased() + first.dropFirst() }
        return ([greeting.label, greeting.line] + blocks + [signOff]).compactMap { $0 }.joined(separator: "\n\n")
    }

    /// Numbers a run of two or more lines that starts with "First"/"第一" and continues with
    /// counting words ("- Second, …", "- 还有，…"), dropping those words and any bullet.
    static func numberedCountedLines(_ text: String) -> String {
        let counted = #"^(?:[-*•]\s*)?((?i:first|second|third|fourth|fifth|and also|also|finally|lastly)\b[,:]?\s*|(?:第[一二三四五六七八九十]+|还有|另外|最后)[，,:：、]\s*)"#
        var lines = text.components(separatedBy: "\n")
        var index = 0
        while index < lines.count {
            var end = index
            while end < lines.count, lines[end].range(of: counted, options: .regularExpression) != nil { end += 1 }
            let starts = lines[index].range(of: #"^(?:[-*•]\s*)?(?i:first\b|第一)"#, options: .regularExpression) != nil
            guard end - index >= 2, starts else { index = max(end, index + 1); continue }
            for (number, line) in (index..<end).enumerated() {
                var item = lines[line]
                item.removeSubrange(item.range(of: counted, options: .regularExpression)!)
                lines[line] = "\(number + 1). " + item.prefix(1).uppercased() + item.dropFirst()
            }
            index = end
        }
        return lines.map { countedPointsInline($0) ?? $0 }.joined(separator: "\n")
    }

    /// "Intro, first, X, second, Y, and also Z. Rest." → intro, numbered X/Y/Z, rest. The first
    /// marker must be "first"/"第一" at the start of a clause, followed by at least one more marker.
    private static func countedPointsInline(_ line: String) -> String? {
        let chinese = (hanShare(in: line) ?? 0) >= 0.5
        let pattern = chinese
            ? #"(?:^|(?<=[，。；：,;:]))\s*(第[一二三四五]|还有|另外|最后)[，,、:：]?\s*"#
            : #"(?:^|(?<=[.;,:!?]\s))(?:and\s+)?((?i:first|second|third|fourth|fifth|also|finally|lastly))\b,?\s+"#
        let regex = try! NSRegularExpression(pattern: pattern)
        let matches = regex.matches(in: line, range: NSRange(line.startIndex..., in: line)).compactMap { match in
            Range(match.range, in: line).flatMap { whole in Range(match.range(at: 1), in: line).map { (whole, line[$0].lowercased()) } }
        }
        guard let start = matches.firstIndex(where: { $0.1 == "first" || $0.1 == "第一" }) else { return nil }
        var markers = [matches[start]]
        for match in matches[(start + 1)...] where match.1 != "first" && match.1 != "第一" { markers.append(match) }
        guard markers.count >= 2 else { return nil }
        let ends: Set<Character> = chinese ? ["。", "！", "？"] : [".", "!", "?"]
        let lastStart = markers[markers.count - 1].0.upperBound
        var lastEnd = line.endIndex
        var cursor = lastStart
        while cursor < line.endIndex {
            let next = line.index(after: cursor)
            if ends.contains(line[cursor]), chinese || next == line.endIndex || line[next] == " " { lastEnd = next; break }
            cursor = next
        }
        let trim = CharacterSet(charactersIn: chinese ? "，,；;。、 " : ",;. ").union(.whitespaces)
        var items: [String] = []
        for (position, marker) in markers.enumerated() {
            let end = position + 1 < markers.count ? markers[position + 1].0.lowerBound : lastEnd
            var item = String(line[marker.0.upperBound..<end]).trimmingCharacters(in: trim)
            if !chinese, item.lowercased().hasSuffix(" and") { item = String(item.dropLast(4)).trimmingCharacters(in: trim) }
            item = item.trimmingCharacters(in: CharacterSet(charactersIn: chinese ? "。！？" : ".!?"))
            guard !item.isEmpty else { return nil }
            items.append(chinese ? item + "。" : item.prefix(1).uppercased() + item.dropFirst() + ".")
        }
        var intro = String(line[..<markers[0].0.lowerBound]).trimmingCharacters(in: .whitespaces)
        let rest = String(line[lastEnd...]).trimmingCharacters(in: .whitespaces)
        let list = items.enumerated().map { "\($0.offset + 1). \($0.element)" }.joined(separator: "\n")
        var blocks: [String] = []
        if intro.hasSuffix(",") || intro.hasSuffix("，") || intro.hasSuffix(":") || intro.hasSuffix("：") {
            intro = String(intro.dropLast()) + (chinese ? "：" : ":")
            blocks.append(intro + "\n" + list)
        } else {
            if !intro.isEmpty { blocks.append(intro) }
            blocks.append(list)
        }
        if !rest.isEmpty { blocks.append(rest) }
        return blocks.joined(separator: "\n\n")
    }

    private static func firstMatch(_ pattern: String, in text: String) -> [String?]? {
        let regex = try! NSRegularExpression(pattern: pattern)
        guard let match = regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) else { return nil }
        return (0..<match.numberOfRanges).map { Range(match.range(at: $0), in: text).map { String(text[$0]) } }
    }

    /// Greeting at the start, after an optional filler or a short label ("Email to my landlord, Hi Karen,").
    private static func splitGreeting(_ block: String, chinese: Bool) -> (label: String?, line: String, rest: String)? {
        if chinese {
            guard let m = firstMatch(#"^(?:(?:嗯|呃|额|那个)[，,]\s*)*((?:[\p{Han}A-Za-z0-9]{1,8}?(?:您好|你好))|大家好|各位好)([，,！!：:])\s*"#, in: block),
                  let whole = m[0], let name = m[1] else { return nil }
            return (nil, name + "，", String(block.dropFirst(whole.count)))
        }
        let names = #"(?:\s+(?:\p{Lu}[\p{L}'’.-]*|team|all|everyone|everybody|there|guys|folks)){0,3}"#
        if let m = firstMatch(#"^(?:(?i:um|uh|so|okay|ok)[,.]?\s+)*((?i:hi|hey|hello|dear|good morning|good afternoon|good evening)"# + names + #")\s*[,!:.]\s*"#, in: block),
           let whole = m[0], let line = m[1] {
            return (nil, line.prefix(1).uppercased() + line.dropFirst() + ",", String(block.dropFirst(whole.count)))
        }
        if let m = firstMatch(#"^((?:[\p{L}'’]+\s+){0,4}[\p{L}'’]+)[,.]\s+((?:Hi|Hey|Hello|Dear)"# + names + #")\s*[,!:]\s*"#, in: block),
           let whole = m[0], let label = m[1], let line = m[2] {
            return (label.prefix(1).uppercased() + label.dropFirst() + ".", line + ",", String(block.dropFirst(whole.count)))
        }
        return nil
    }

    /// Sign-off at the very end ("Thanks." / "Thanks, Jo." / "谢谢，小林。").
    private static func splitSignOff(_ block: String, chinese: Bool) -> (rest: String, signOff: String)? {
        let pattern = chinese
            ? #"(?:^|(?<=[。！？!?，,]))\s*(谢谢您|谢谢|多谢|辛苦了|麻烦了)(?:[，,]\s*(\p{Han}{1,4}))?([。！!]?)\s*$"#
            : #"(?:^|(?<=[.!?]))\s*((?i:thanks so much|thanks a lot|thank you|many thanks|thanks|cheers|best regards|regards|all the best|best))(?:,?\s+(\p{Lu}[\p{L}'’-]*))?([.!]?)\s*$"#
        guard let m = firstMatch(pattern, in: block), let whole = m[0], let word = m[1] else { return nil }
        var rest = String(block.dropLast(whole.count)).trimmingCharacters(in: .whitespaces)
        if chinese, rest.hasSuffix("，") || rest.hasSuffix(",") { rest = String(rest.dropLast()) + "。" }
        let signOff = m[2].map { word + (chinese ? "，" : ",") + "\n" + $0 } ?? word + (m[3] ?? "")
        return (rest, signOff)
    }

    /// Splits a one-line paragraph around its first sentence of three or more comma-separated
    /// points that ends with "…, and Z" (还有/另外/以及 Z): [before, numbered points, after].
    private static func numberedPoints(in block: String, chinese: Bool) -> [String]? {
        guard !block.contains("\n") else { return nil }
        let sentences = sentences(in: block)
        for (position, sentence) in sentences.enumerated() {
            guard let (lead, points) = points(in: sentence, chinese: chinese) else { continue }
            let items = points.enumerated().map { "\($0.offset + 1). \($0.element)" }.joined(separator: "\n")
            let separator = chinese ? "" : " "
            let before = Array(sentences[..<position]) + [lead].compactMap { $0 }
            return [before.joined(separator: separator), items, sentences[(position + 1)...].joined(separator: separator)]
        }
        return nil
    }

    static let targetLanguageTemplateVariable = "{{TARGET_LANGUAGE}}"

    private static func translateTask(target: String) -> String {
        """
        Translate the transcribed speech into \(target).
        The transcript may contain fillers, repetitions and self-corrections:
        do not carry them into the translation, but preserve meaning, names, and numbers.
        Apply self-corrections before translating. When the speaker says a detail and then replaces it, even after a pause,
        an ellipsis or a full stop (Thursday... actually, Friday; three, sorry, four; 周一，不对，周二),
        keep only the corrected detail: the translation contains only the replacement.
        Never mention, translate or explain the earlier value.
        Translate only what was said: add no places, rooms, times, reasons or other details, and keep every name, number and date.
        Output only the translation, without explanations or the original text.
        """
    }

    /// The built-in instruction of each mode, for the phone's Developer prompt editor. English
    /// variants; Translate keeps the target-language placeholder.
    static func builtInPrompts() -> [String: String] {
        [
            LocalAPIComposeMode.dictate.rawValue: englishDictationInstruction,
            LocalAPIComposeMode.edit.rawValue: englishEditInstruction,
            LocalAPIComposeMode.translate.rawValue: translateTask(target: targetLanguageTemplateVariable),
            LocalAPIComposeMode.write.rawValue: englishWriteInstruction,
        ]
    }

    /// The leading "2. ", "- " or "• " of a one-line selection, nil otherwise.
    static func listMarker(of text: String) -> String? {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).contains("\n"),
              let range = text.range(of: #"^\s*(?:\d+[.)]|[-*•])\s+"#, options: .regularExpression) else { return nil }
        return String(text[range])
    }

    /// Sentences of one line, each with its end punctuation.
    static func sentences(in line: String) -> [String] {
        var sentences: [String] = [], current = ""
        for (index, character) in zip(line.indices, line) {
            current.append(character)
            let next = line.index(after: index)
            if "。！？".contains(character) || (".!?".contains(character) && (next == line.endIndex || line[next] == " ")) {
                sentences.append(current.trimmingCharacters(in: .whitespaces)); current = ""
            }
        }
        if !current.trimmingCharacters(in: .whitespaces).isEmpty { sentences.append(current.trimmingCharacters(in: .whitespaces)) }
        return sentences
    }

    /// The points of one sentence, and the clause before them when the points follow ", so".
    private static func points(in sentence: String, chinese: Bool) -> (lead: String?, points: [String])? {
        var body = sentence
        if chinese {
            guard body.hasSuffix("。") || !"！？".contains(body.last ?? " ") else { return nil }
            body = body.trimmingCharacters(in: CharacterSet(charactersIn: "。"))
            for filler in ["那个，", "嗯，", "呃，", "额，"] where body.hasPrefix(filler) { body = String(body.dropFirst(filler.count)) }
            var parts = body.components(separatedBy: CharacterSet(charactersIn: "，,")).map { $0.trimmingCharacters(in: .whitespaces) }
            guard parts.count >= 3, let joiner = ["还有", "另外", "以及"].first(where: { parts[parts.count - 1].hasPrefix($0) }) else { return nil }
            parts[parts.count - 1] = String(parts[parts.count - 1].dropFirst(joiner.count))
            let subordinate = ["所以", "但是", "因为", "如果", "不过", "可是", "然后"]
            guard parts.allSatisfy({ part in part.unicodeScalars.filter { $0.properties.isIdeographic }.count >= 3
                && !subordinate.contains(where: { part.hasPrefix($0) }) }) else { return nil }
            return (nil, parts.map { $0 + "。" })
        }
        guard body.hasSuffix(".") || !"!?".contains(body.last ?? " ") else { return nil }
        body = body.trimmingCharacters(in: CharacterSet(charactersIn: "."))
        if let filler = body.range(of: #"^(?i:so|ah|uh|um|okay),?\s+"#, options: .regularExpression) { body.removeSubrange(filler) }
        var parts = body.components(separatedBy: ", ")
        var lead: String?
        if let so = parts.indices.dropFirst().last(where: { parts[$0].lowercased().hasPrefix("so ") }), parts.count - so >= 3 {
            lead = parts[..<so].joined(separator: ", ") + "."
            parts = Array(parts[so...])
            parts[0] = String(parts[0].dropFirst(3))
        }
        guard parts.count >= 3, parts[parts.count - 1].hasPrefix("and ") else { return nil }
        parts[parts.count - 1] = String(parts[parts.count - 1].dropFirst(4))
        let subordinate = ["so", "but", "because", "which", "who", "if", "when", "although", "though", "then"]
        guard parts.allSatisfy({ part in
            let words = part.split(separator: " ")
            return words.count >= 3 && !subordinate.contains(words[0].lowercased())
        }) else { return nil }
        return (lead, parts.map { $0.prefix(1).uppercased() + $0.dropFirst() + "." })
    }

    /// The edited selection when the model returned it inside the unchanged text around it
    /// (before + edit + after); otherwise the output unchanged.
    static func withoutSurroundingField(_ output: String, options: LocalAPIComposeOptions) -> String {
        let before = options.contextBefore.trimmingCharacters(in: .whitespacesAndNewlines)
        let after = options.contextAfter.trimmingCharacters(in: .whitespacesAndNewlines)
        let selection = options.selectedText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !before.isEmpty || !after.isEmpty, before != selection, after != selection,
              output.hasPrefix(before), output.hasSuffix(after), output.count > before.count + after.count else { return output }
        let middle = output.dropFirst(before.count).dropLast(after.count).trimmingCharacters(in: .whitespacesAndNewlines)
        return middle.isEmpty ? output : middle
    }

    /// Drops closing sentences ("Thank you.", "谢谢！") the model appended to an edit when the
    /// selection had none and the field already continues with a closing. One edit added
    /// "Thank you." right before the mail's own "Thanks.".
    static func withoutRepeatedClosing(_ output: String, options: LocalAPIComposeOptions) -> String {
        let closing = #"^(?i:thanks|thank you|thanks so much|thanks a lot|many thanks|cheers|best|best regards|regards|谢谢|谢谢您|多谢|辛苦了)[\s,，.。!！]*(?:\p{L}+[.。!！]?)?$"#
        func isClosing(_ sentence: String) -> Bool { sentence.range(of: closing, options: .regularExpression) != nil }
        let next = options.contextAfter.components(separatedBy: "\n").map { $0.trimmingCharacters(in: .whitespaces) }.first { !$0.isEmpty }
        guard let next, isClosing(sentences(in: next).first ?? next),
              !(sentences(in: options.selectedText.trimmingCharacters(in: .whitespacesAndNewlines)).last.map(isClosing) ?? false),
              !output.contains("\n") else { return output }
        var parts = sentences(in: output)
        while parts.count > 1, let last = parts.last, isClosing(last) { parts.removeLast() }
        let chinese = (hanShare(in: output) ?? 0) >= 0.5
        return parts.joined(separator: chinese ? "" : " ")
    }

    struct EditNotApplied: Error, LocalizedError {
        var errorDescription: String? { "Couldn't apply that edit" }
    }

    /// True when the output copies a sentence from the text around the cursor that the speech
    /// and selection do not contain, or repeats a sentence more often than they do. The model has
    /// prefixed the previous insertion in Dictate and returned the note about 2.5 times in
    /// Edit. Sentences under six letters are ignored, as they recur naturally.
    static func leaksContext(_ output: String, rawTranscription: String, options: LocalAPIComposeOptions) -> Bool {
        func normalized(_ text: String) -> String {
            String(text.lowercased().unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) }.map(Character.init))
        }
        func sentences(_ text: String) -> [String] {
            text.components(separatedBy: CharacterSet(charactersIn: "。！？!?.;；\n")).map(normalized).filter { $0.count >= 6 }
        }
        func occurrences(of sentence: String, in text: String) -> Int {
            text.components(separatedBy: sentence).count - 1
        }
        let out = normalized(output), speech = normalized(rawTranscription), selection = normalized(options.selectedText)
        for sentence in sentences(options.contextBefore) + sentences(options.contextAfter)
        where out.contains(sentence) && !speech.contains(sentence) && !selection.contains(sentence) {
            return true
        }
        return Set(sentences(output)).contains { sentence in
            occurrences(of: sentence, in: out) > max(1, occurrences(of: sentence, in: speech), occurrences(of: sentence, in: selection))
        }
    }

    enum EditRetry: Equatable { case unchanged, unsafe, leaked, droppedTerms([String]) }

    /// Applies a spoken correction when one number, time or weekday directly replaces another
    /// ("three, sorry, four", "Thursday... actually, Friday", "九点，不对，十点"). Anything less
    /// certain is left to the model. Qwen3.5-4B kept both values in translations.
    static func appliedSelfCorrections(_ text: String) -> String {
        let number = #"(?:\d{1,4}(?::\d{2})?|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)"#
        let day = #"(?:mon|tues|wednes|thurs|fri|satur|sun)day"#
        // A filler may sit before the cue, and ASR may end the sentence before "sorry":
        // "Monday, uh, sorry, Tuesday" came back as "Monday. Ah, sorry Tuesday".
        let cue = #"\s*(?:(?:,|\.{2,3}|…|—|-)\s*(?:(?:uh|um|ah|er),?\s*)?(?:sorry|actually|i mean|no wait|no|or rather)|\.\s+(?:(?:uh|um|ah|er),?\s+)?sorry)\s*[,.…—-]?\s*"#
        let zhNumber = #"[0-9一二三四五六七八九十两]+点半?"#
        let zhDay = #"(?:周|星期|礼拜)[一二三四五六日天]"#
        let zhCue = #"[，,、]\s*(?:(?:呃|额|嗯|啊)[，,]\s*)?(?:不对|不是|我是说|其实是|哦不)[，,]?\s*"#
        let patterns = [
            "(?i)\\b\(number)\\b\(cue)(\(number))\\b", "(?i)\\b\(day)\\b\(cue)(\(day))\\b",
            "\(zhNumber)\(zhCue)(\(zhNumber))", "\(zhDay)\(zhCue)(\(zhDay))",
        ]
        var result = text
        for pattern in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern) else { continue }
            result = regex.stringByReplacingMatches(in: result, range: NSRange(result.startIndex..., in: result), withTemplate: "$1")
        }
        return result
    }

    /// Words that usually introduce a spoken self-correction. "Actually" or "no" can also be
    /// content; a match only switches Translate to writing the corrected line first.
    static func containsCorrectionCue(_ text: String) -> Bool {
        let patterns = [#"(?i)(?:^|[\s,.…—-])(?:actually|sorry|i mean|no wait|or rather)\b"#, #"(?i)\bno[,.…!—]"#,
                        #"不对|不是，|其实是|我是说|哦不|应该是"#]
        return patterns.contains { text.range(of: $0, options: .regularExpression) != nil }
    }

    /// The text after the last "Translation:" label, or nil when the model did not write one.
    static func translationAfterCorrection(_ output: String) -> String? {
        guard let label = output.range(of: "Translation:", options: [.backwards, .caseInsensitive]) else { return nil }
        let text = output[label.upperBound...].trimmingCharacters(in: .whitespacesAndNewlines)
        return text.isEmpty ? nil : text
    }

    /// Dates and times in the selection that a tone or length edit dropped. Empty when the
    /// instruction itself names a date or time, since then it is a requested replacement.
    /// A Chinese numeral before 点 counts only after a time of day or before 钟/半: 一点 alone means "a bit".
    static func missingDateOrTimeTerms(_ instruction: String, selection: String, output: String) -> [String] {
        let patterns = [
            #"(?i)\b(?:today|tonight|tomorrow|yesterday|noon|midnight|this (?:morning|afternoon|evening|week|weekend)|next week|(?:mon|tues|wednes|thurs|fri|satur|sun)day)\b"#,
            #"(?i)\b\d{1,2}(?::\d{2})?\s?(?:a\.?m\.?|p\.?m\.?)?(?=\W|$)"#,
            #"今天|今晚|明天|后天|昨天|上午|下午|中午|晚上|早上|凌晨|(?:周|星期|礼拜)[一二三四五六日天]|[0-9]+点半?|(?<=[午上晚早晨])[一二三四五六七八九十两]+点半?|[一二三四五六七八九十两]+点(?:钟|半)"#,
        ]
        func terms(in text: String) -> [(Range<String.Index>, String)] {
            patterns.flatMap { pattern -> [(Range<String.Index>, String)] in
                guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
                return regex.matches(in: text, range: NSRange(text.startIndex..., in: text)).compactMap { match in
                    Range(match.range, in: text).map { ($0, String(text[$0])) }
                }
            }.sorted { $0.0.lowerBound < $1.0.lowerBound }
        }
        let replacement = #"(?i)改成|换成|改为|改到|换到|挪到|\b(?:change|replace|make it|move|switch)\b"#
        let spokenTime = #"[0-9一二三四五六七八九十两]+点"#
        let namesATime = instruction.range(of: replacement, options: .regularExpression) != nil
            || instruction.ranges(of: try! Regex(spokenTime)).contains { instruction[$0] != "一点" }
        guard terms(in: instruction).isEmpty, !namesATime else { return [] }
        var missing: [String] = []
        for (_, term) in terms(in: selection) where !output.localizedCaseInsensitiveContains(term) && !missing.contains(term) {
            missing.append(term)
        }
        return missing
    }

    /// Template slots such as [Name], {Your Name}, <Recipient> or 【姓名】 that a Write result
    /// must never insert. Brackets the user dictated or that surround the cursor are content.
    /// A slot needs a letter and no padding spaces, so "[ ]", "[1]" and "2 < 3 and 4 > 1" pass.
    static func containsPlaceholder(_ output: String, allowedIn source: String) -> Bool {
        let patterns = [#"\{\{?[^\s{}][^{}\n]{0,40}?\}?\}"#, #"\[[^\s\[\]][^\[\]\n]{0,40}?\]"#,
                        #"<[^\s<>][^<>\n]{0,38}[^\s<>]>"#, #"【[^】\n]{1,20}】"#, #"［[^］\n]{1,20}］"#]
        for pattern in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern) else { continue }
            for match in regex.matches(in: output, range: NSRange(output.startIndex..., in: output)) {
                guard let range = Range(match.range, in: output) else { continue }
                let slot = String(output[range])
                let letters = slot.unicodeScalars.filter { CharacterSet.letters.contains($0) }.count
                if letters >= 2, !source.contains(slot) { return true }
            }
        }
        return false
    }

    /// One correction example in the target language for the languages the first release
    /// supports; other targets get the rule alone rather than an example in the wrong language.
    static func translationCorrectionExample(target: String, twoStep: Bool = false) -> String? {
        let tag = target.lowercased()
        let pairs: [(speech: String, corrected: String, translation: String)]
        if tag.hasPrefix("zh-hant") || tag == "zh-tw" || tag == "zh-hk" {
            pairs = [("Call Sam on Monday, actually Tuesday, at ten.", "Call Sam on Tuesday at ten.", "週二十點給Sam打電話。"),
                     ("Meet me at three, sorry, four.", "Meet me at four.", "四點見。")]
        } else if tag.hasPrefix("zh") {
            pairs = [("Call Sam on Monday, actually Tuesday, at ten.", "Call Sam on Tuesday at ten.", "周二十点给Sam打电话。"),
                     ("Meet me at three, sorry, four.", "Meet me at four.", "四点见。")]
        } else if tag.hasPrefix("en") {
            pairs = [("周一，不对，周二上午十点给小陈打电话。", "周二上午十点给小陈打电话。", "Call Xiao Chen on Tuesday at 10 a.m.")]
        } else {
            return nil
        }
        return pairs.map { pair in
            twoStep
                ? "Speech transcript: \(pair.speech)\nCorrected: \(pair.corrected)\nTranslation: \(pair.translation)"
                : "Speech transcript: \(pair.speech)\nTranslation: \(pair.translation)"
        }.joined(separator: "\n\n")
    }

    /// Detects copied replacement instructions, including ASR's "cash to cash" and
    /// the model's hyphenated version. Existing selection text is never disallowed.
    static func containsEditInstructionLeakage(_ output: String, instruction: String, selection: String) -> Bool {
        func normalized(_ text: String) -> String {
            text.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted)
                .filter { !$0.isEmpty }.joined(separator: " ")
        }
        let output = " " + normalized(output) + " "
        let selection = " " + normalized(selection) + " "
        let instruction = normalized(instruction)
        var fragments: [String] = []
        if instruction.split(separator: " ").count >= 3 { fragments.append(instruction) }
        // Restrict extraction to explicit replacement operations, not every phrase
        // containing "to" (which could be legitimate edited prose).
        let patterns = [#"\b(?:fix|change|replace|correct)\s+(.+?\s+(?:to|with)\s+.+)$"#,
                        #"(?:把|将)(.+?(?:改成|换成|替换成).+)$"#]
        for pattern in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern),
                  let match = regex.firstMatch(in: instruction, range: NSRange(instruction.startIndex..., in: instruction)),
                  let range = Range(match.range(at: 1), in: instruction) else { continue }
            fragments.append(String(instruction[range]))
        }
        return fragments.contains { fragment in
            let phrase = " " + fragment + " "
            // Han phrases need no word separators.
            if (hanShare(in: fragment) ?? 0) >= 0.5 {
                return output.contains(fragment) && !selection.contains(fragment)
            }
            return output.contains(phrase) && !selection.contains(phrase)
        }
    }

    /// Human locale names next to the bare tags: models follow "English (United Kingdom)"
    /// more reliably than "en-GB" alone.
    static let languageVariantDisplayNames: [String: String] = [
        "en-US": "English (United States)",
        "en-GB": "English (United Kingdom)",
        "en-AU": "English (Australia)",
        "en-CA": "English (Canada)",
        "zh-Hans": "Chinese, Simplified",
        "zh-Hant": "Chinese, Traditional",
        "es-ES": "Spanish (Spain)",
        "es-MX": "Spanish (Mexico)",
        "fr-FR": "French (France)",
        "fr-CA": "French (Canada)",
        "pt-PT": "Portuguese (Portugal)",
        "pt-BR": "Portuguese (Brazil)",
    ]

    // A short statement of intent per language; the model judges structure. A list of rules and format examples made
    // Qwen3.5-4B add unasked bullets and stiffen the register; compared blind on 42 dictations,
    // this read more like the speaker 22 times, the rule list 7 times, with 13 ties.
    private static let chineseDictationInstruction = """
    把这段口述转写成说话人自己会写出来的文字：干净、自然、好读，用他说话时的语言，夹杂的英文照原样保留。
    保留他的意思、口吻和语体，只去掉写字时没人会写的东西：口头禅、说一半重来的话和他自己改掉的说法。
    按他自己打字时会写成的样子来组织。
    只输出整理后的文字。
    """

    private static let englishDictationInstruction = """
    Turn this speech transcript into the text this speaker would have written themselves: clean, natural and easy to read, in the language or mix of languages they spoke.
    Keep their meaning, their voice and their register. Leave out only what nobody types: fillers, false starts and the words they took back.
    Shape it the way this speaker would if they had typed it.
    Output only the text.
    """

    private static let japaneseDictationInstruction = """
    この書き起こしを、話し手が自分で書いたらこうなるという文章にしてください。きれいで自然で読みやすく、話した言語のまま、英語の語はそのまま残します。
    意味、話し方の雰囲気、丁寧さの度合いはそのままにし、書くときには誰も書かないもの（つなぎ言葉、言いかけてやめた部分、言い直す前の言葉）だけを取り除きます。
    形は、話し手が自分で打つときと同じようにしてください。
    整えた文章だけを出力してください。
    """

    private static let previousChineseDictationInstruction = """
    整理转写的口述内容。
    删除口头词、重复和自我纠正，修正标点、列表和段落。不要添加未说出的内容。
    保持口述原本的语言，包括中英混合的内容。不要翻译。

    当说话者明确列举不同事项时，每个事项单独一行。
    直接列举的事项用“- ”项目符号，第一、第二、第三这样的顺序用编号。
    保留引导句并单独一行。普通叙述保持一个段落，不要改成列表。
    保留原词和事实。不要添加事项或解释。

    示例：
    口述：带三样东西：钥匙、外套和水。
    输出：
    带三样东西：
    - 钥匙
    - 外套
    - 水

    自我纠正示例：
    口述：把评审改到周三——不对——周四下午三点。
    输出：把评审改到周四下午三点。
    """

    private static let previousEnglishDictationInstruction = """
    Clean up the transcribed speech.
    Remove fillers, repetitions and self-corrections, and fix punctuation,
    lists and paragraphs. Do not add content that was not said.
    Keep the original spoken language, including when the speech mixes languages.
    Do not translate.

    Keep the transcript's language. Never translate.
    When the speech explicitly enumerates distinct items, put each item on its own line.
    Use '- ' bullets for named items and numbered lines for first, second, third sequences.
    Keep any introductory sentence on its own line. Keep ordinary narration in one paragraph;
    do not turn it into a list. Preserve the words and facts. Never add items or explanations.

    Example:
    Speech: Bring three things: keys, a coat, and water.
    Output:
    Bring three things:
    - keys
    - a coat
    - water

    Self-correction example:
    Speech: Move the review to Wednesday — no wait — Thursday at three.
    Output: Move the review to Thursday at three.
    """

    private static let englishEditInstruction = """
    Apply the instruction to the selected text. Change only what the instruction requests. For a whole-text change such as shortening or changing tone, apply that change to the whole text. Keep all names, numbers, times and facts that the instruction does not ask to change. For a specific replacement, copy the rest verbatim. Keep the selected text's language. The instruction comes from speech recognition and may contain sound-alike errors. If its literal reading would change nothing or make no sense, choose the similar-sounding word that fits the selected text. When both replacement words sound the same, choose the spelling that fits the selected text's context. Never insert the instruction's own phrasing into the result. A tone or length instruction always changes the wording: rewrite the whole text in the requested tone or length and keep its meaning, who says it to whom, and every name, number, date and time (today, tonight, Friday). Output only the edited text.
    Example:
    Selected text: The workshop is on Tuesday at noon.
    Instruction: Actually make it Wednesday.
    Output: The workshop is on Wednesday at noon.

    Selected text: Stop sending me the reports late every Friday.
    Instruction: Make this friendlier.
    Output: Could you send the reports on time on Fridays? Thanks!
    """

    private static let chineseEditInstruction = """
    按指令修改选中的文字。只改指令要求的部分。要求缩短或改变语气等整段修改时，对整段文字执行该要求。保留指令未要求更改的姓名、数字、时间和事实。要求替换具体内容时，其余文字逐字保留。保持选中文字的语言。指令来自语音识别，可能有同音词错误。如果按字面执行不会改变内容，或会造成语义不通，请选用符合选中文字语境的同音或近音词。如果两个替换词发音相同，根据选中文字的语境选择正确的写法。不要把指令本身的措辞插入结果。要求改变语气或长度时，必须改写措辞：按要求的语气或长度重写整段，保留原意、说话人与对象，以及每个姓名、数字、日期和时间。只输出修改后的文字。
    示例：
    选中文字：培训在周二中午开始。
    指令：不对，改成周三。
    输出：培训在周三中午开始。

    选中文字：你们每周五的报告又迟交了。
    指令：语气友好一点。
    输出：每周五的报告这次又晚交了，下次请尽量按时提交，谢谢！
    """

    private static let englishWriteInstruction = """
    Write the actual message requested by the instruction, addressed to its intended recipient. Use the instruction's language. Do not repeat or polish the instruction itself. Preserve all requested facts; add no facts, promises or explanations. If the instruction does not give a name or detail, leave it out. Never write placeholders such as [Name], <Name> or {Name}. Output only the message, ready to send.
    Example:
    Instruction: Tell Pat I cannot attend the lunch and wish them a good time.
    Message: Pat, I cannot attend the lunch. Have a good time!

    Instruction: Let my landlord know the sink is leaking.
    Message: Hi, just letting you know that the sink is leaking.
    """

    private static let chineseWriteInstruction = """
    根据指令写出可以直接发给对方的消息。使用指令的语言。不要复述或润色指令本身。保留要求表达的全部事实，不添加事实、承诺或解释。指令没有给出的姓名或细节就省略，不要写[姓名]这类占位符。只输出消息正文。
    示例：
    指令：告诉小陈我不能参加午餐，祝他们吃得开心。
    消息：小陈，我不能参加午餐了，祝你们吃得开心！

    指令：跟房东说一下水槽漏水了。
    消息：您好，跟您说一下，水槽漏水了。
    """

    private static func body(
        task: String,
        selection: String?,
        context: (before: String, after: String)?,
        style: LocalAPIComposeStyle,
        vocabulary: [String],
        languageVariants: [String] = [],
        styleExamples: [LocalAPIStyleExample] = [],
        selectionPrefix: String = "Selected text:\n",
        transcriptPrefix: String = "Speech transcript: ",
        afterTranscript: String? = nil,
        outputCue: String? = nil
    ) -> String {
        var sections = [task.trimmingCharacters(in: .whitespacesAndNewlines)]

        if let selection, !selection.isEmpty, let context, (context.before + context.after).contains("\n") {
            // Multi-line context goes before the selection, so the selection is the text next to
            // the instruction: after it, Qwen3.5-4B rewrote a whole structured mail for "make this
            // firmer" (3 of 3). One-line context stays after it: placed first, a
            // "make this calmer" came back unchanged.
            if !context.before.isEmpty {
                sections.append("Text before the selected text (already in the field; reference only, do not edit or repeat it):\n\(context.before)")
            }
            if !context.after.isEmpty {
                sections.append("Text after the selected text (already in the field; reference only, do not edit or repeat it):\n\(context.after)")
            }
            sections.append("\(selectionPrefix)\(selection)")
        } else if let selection, !selection.isEmpty {
            sections.append("\(selectionPrefix)\(selection)")
            if let context, !context.before.isEmpty { sections.append("Text before the cursor (already in the field; do not repeat it):\n\(context.before)") }
            if let context, !context.after.isEmpty { sections.append("Text after the cursor (already in the field; do not repeat it):\n\(context.after)") }
        } else if let context {
            if !context.before.isEmpty { sections.append("Text before the cursor (already in the field; do not repeat it):\n\(context.before)") }
            if !context.after.isEmpty { sections.append("Text after the cursor (already in the field; do not repeat it):\n\(context.after)") }
        }

        let toneLine: String?
        switch style {
        case .automatic: toneLine = nil
        case .formal: toneLine = "Tone: formal."
        case .casual: toneLine = "Tone: casual."
        case .concise: toneLine = "Tone: concise."
        }
        if let toneLine { sections.append(toneLine) }

        if !vocabulary.isEmpty {
            sections.append("Prefer these spellings: \(vocabulary.joined(separator: "; ")).")
        }
        if !languageVariants.isEmpty {
            let namedVariants = languageVariants.map { variant in
                "\(variant) — \(languageVariantDisplayNames[variant] ?? variant)"
            }
            sections.append(
                "Preferred language variants: \(namedVariants.joined(separator: ", ")). "
                    + "Apply these preferences only when the output already uses the corresponding language. Use its regional "
                    + "spelling or script consistently in every word of the entire output, including words "
                    + "that already look acceptable; do not translate because of these preferences."
            )
        }
        if !styleExamples.isEmpty {
            var lines = [
                "Personal wording examples. Adjust phrasing only. Do not change facts, names, numbers, or the output language. Do not override the task, the target language, or the preferred spellings.",
            ]
            for example in styleExamples {
                lines.append("Before: \(example.before)")
                lines.append("After: \(example.after)")
            }
            sections.append(lines.joined(separator: "\n"))
        }

        sections.append("\(transcriptPrefix)\(EnhancementPromptResolver.rawTranscriptionTemplateVariable)")
        if let afterTranscript { sections.append(afterTranscript) }
        if let outputCue { sections.append(outputCue) }
        return sections.joined(separator: "\n\n")
    }
}
