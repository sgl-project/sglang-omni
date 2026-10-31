import Foundation
import Testing
@testable import VoxtMobileKit

@Suite struct TextKeyboardTests {
    @Test func numbersAllowEmailAndCurrencyAndShiftAppliesToAccents() {
        for layout in TextKeyboardLayoutID.allCases {
            let keys = TextKeyboardLayout.make(layout).rows(.numbers).flatMap { $0 }
            for symbol in ["@", "$", "&", "'", "\""] {
                #expect(keys.contains { $0.text == symbol || $0.alternates.contains(symbol) })
            }
        }
        var state = TextKeyboardState(shifted: true)
        #expect(state.insertAlternate("é") == .insert("É"))
        #expect(!state.shifted)
    }

    @Test func catalogIncludesJapaneseAndZhuyinAlongsideFourLatinLayouts() {
        #expect(TextKeyboardLayoutID.allCases.map(\.rawValue) == ["en-QWERTY", "fr-AZERTY", "de-QWERTZ", "es-QWERTY", "ja-flick", "zh-Hant-zhuyin", "ko-2beolsik", "zh-Hans-pinyin"])
        #expect(TextKeyboardLayoutID.allCases.count == 8)
        #expect(TextKeyboardLayoutID.pinyin.title == "简体拼音")
        #expect(TextKeyboardLayoutID.pinyin.shortLabel == "拼")
        #expect(TextKeyboardLayoutID.korean.title == "한국어")
        #expect(TextKeyboardLayoutID.korean.shortLabel == "한")
        #expect(TextKeyboardLayoutID(rawValue: "dvorak") == nil)
        #expect(TextKeyboardLayoutID(rawValue: "retired-36") == nil)
    }

    @Test func letterRowsMatchTheFourPublicLayouts() {
        let english = TextKeyboardLayout.make(.englishQWERTY)
        let french = TextKeyboardLayout.make(.frenchAZERTY)
        let german = TextKeyboardLayout.make(.germanQWERTZ)
        let spanish = TextKeyboardLayout.make(.spanishQWERTY)

        #expect(characters(in: english, plane: .letters, row: 0) == ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"])
        #expect(characters(in: english, plane: .letters, row: 2) == ["z", "x", "c", "v", "b", "n", "m"])
        #expect(characters(in: french, plane: .letters, row: 0) == ["a", "z", "e", "r", "t", "y", "u", "i", "o", "p"])
        #expect(characters(in: french, plane: .letters, row: 1) == ["q", "s", "d", "f", "g", "h", "j", "k", "l", "m"])
        #expect(characters(in: german, plane: .letters, row: 0) == ["q", "w", "e", "r", "t", "z", "u", "i", "o", "p", "ü"])
        #expect(characters(in: german, plane: .letters, row: 2) == ["y", "x", "c", "v", "b", "n", "m"])
        #expect(characters(in: spanish, plane: .letters, row: 1).contains("ñ"))
        #expect(characters(in: spanish, plane: .letters, row: 0) == ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"])
    }

    @Test func accentsAreExplicitKeysOrLongPressAlternates() {
        let englishE = character("e", in: .englishQWERTY, plane: .letters)
        #expect(englishE?.alternates.contains("é") == true)

        let french = TextKeyboardLayout.make(.frenchAZERTY)
        #expect(characters(in: french, plane: .letters).contains("é"))
        #expect(character("e", in: .frenchAZERTY, plane: .letters)?.alternates.contains("è") == true)
        #expect(character("c", in: .frenchAZERTY, plane: .letters)?.alternates.contains("ç") == true)

        let german = characters(in: TextKeyboardLayout.make(.germanQWERTZ), plane: .letters)
        #expect(german.contains("ä") && german.contains("ö") && german.contains("ü") && german.contains("ß"))

        #expect(characters(in: TextKeyboardLayout.make(.spanishQWERTY), plane: .letters).contains("ñ"))
        #expect(character("e", in: .spanishQWERTY, plane: .letters)?.alternates.contains("é") == true)
        #expect(character("u", in: .spanishQWERTY, plane: .letters)?.alternates.contains("ü") == true)
    }

    @Test func shiftNumbersDeleteSpaceReturnAndVoiceAreDistinctActions() {
        var state = TextKeyboardState()
        let shift = try! #require(state.visibleKeys().first { $0.kind == .shift })
        #expect(state.press(shift) == nil)
        #expect(state.shifted)
        let upperQ = try! #require(state.visibleKeys().first { $0.text == "Q" })
        #expect(state.press(upperQ) == .insert("Q"))
        #expect(!state.shifted)
        let lowerQ = try! #require(state.visibleKeys().first { $0.text == "q" })
        #expect(state.press(lowerQ) == .insert("q"))

        let numbers = try! #require(state.visibleKeys().first { $0.kind == .numbers })
        #expect(state.press(numbers) == nil)
        #expect(state.plane == .numbers)
        #expect(state.visibleKeys().contains { $0.text == "1" })
        #expect(characters(in: state.layout, plane: .numbers, row: 0) == ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"])
        #expect(state.visibleKeys().contains { $0.text == "¿" } == false)
        let letters = try! #require(state.visibleKeys().first { $0.kind == .letters })
        #expect(state.press(letters) == nil)
        #expect(state.plane == .letters)

        var spanish = TextKeyboardState()
        spanish.active = .spanishQWERTY
        spanish.plane = .numbers
        #expect(spanish.visibleKeys().contains { $0.text == "¿" })
        #expect(spanish.visibleKeys().contains { $0.text == "¡" })

        var actions = TextKeyboardState()
        #expect(actions.press(try! #require(actions.visibleKeys().first { $0.kind == .space })) == .insert(" "))
        #expect(actions.press(try! #require(actions.visibleKeys().first { $0.kind == .return })) == .insert("\n"))
        #expect(actions.press(try! #require(actions.visibleKeys().first { $0.kind == .delete })) == .delete)
        #expect(!actions.visibleKeys().contains { $0.kind == .voice })
        #expect(actions.press(.init(id: "voice", kind: .voice, label: "Voice")) == .voice)
        #expect(actions.insertAlternate("é") == .insert("é"))
    }

    @Test func defaultsUseOnlyEnglishAndPreferredLanguageLayouts() throws {
        let name = "keyboard-defaults.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        #expect(TextKeyboardState.load(from: defaults, preferredLanguages: ["en-US"]).enabled == [.englishQWERTY])
        #expect(TextKeyboardState.load(from: defaults, preferredLanguages: ["zh-Hant-TW", "ja-JP", "en"]).enabled == [.englishQWERTY, .zhuyin, .japaneseFlick])
        #expect(TextKeyboardState.load(from: defaults, preferredLanguages: ["fr-FR", "de-DE", "es-ES", "fr-CA"]).enabled == [.englishQWERTY, .frenchAZERTY, .germanQWERTZ, .spanishQWERTY])
        #expect(TextKeyboardState.load(from: defaults, preferredLanguages: ["zh-Hans-CN", "ko-KR"]).enabled == [.englishQWERTY, .pinyin, .korean])
        #expect(TextKeyboardState.load(from: defaults, preferredLanguages: ["zh-CN"]).enabled == [.englishQWERTY, .pinyin])
        #expect(TextKeyboardState.load(from: defaults, preferredLanguages: ["zh-Hant-TW"]).enabled == [.englishQWERTY, .zhuyin])
        #expect(TextKeyboardState.load(from: defaults, preferredLanguages: ["zh-TW"]).enabled == [.englishQWERTY, .zhuyin],
                "Taiwan without a script tag is Traditional")
    }

    @Test func oldAllLayoutsDefaultMigratesButCustomizedOrderSurvives() throws {
        let name = "keyboard-migration.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        // The unversioned default: exactly these six, never a newer full list.
        let legacy = ["en-QWERTY", "fr-AZERTY", "de-QWERTZ", "es-QWERTY", "ja-flick", "zh-Hant-zhuyin"]
        defaults.set(try JSONSerialization.data(withJSONObject: ["enabled": legacy, "active": TextKeyboardLayoutID.japaneseFlick.rawValue]), forKey: TextKeyboardState.storageKey)
        let migrated = TextKeyboardState.load(from: defaults, preferredLanguages: ["en", "zh-Hant"])
        #expect(migrated.enabled == [.englishQWERTY, .zhuyin])
        #expect(migrated.active == .englishQWERTY)
        let custom = Array(TextKeyboardLayoutID.allCases.reversed())
        TextKeyboardState(enabled: custom, active: .japaneseFlick).save(to: defaults)
        let preserved = TextKeyboardState.load(from: defaults, preferredLanguages: ["en"])
        #expect(preserved.enabled == custom)
        #expect(preserved.active == .japaneseFlick)
        TextKeyboardState(enabled: [.frenchAZERTY], active: .frenchAZERTY).save(to: defaults)
        #expect(TextKeyboardState.load(from: defaults, preferredLanguages: ["ja"]).enabled == [.frenchAZERTY])
    }

    @Test func deliberatelyEnablingEveryLayoutInNewVersionIsNotLegacyDefault() {
        let name = "keyboard-new-selection.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        TextKeyboardState(enabled: TextKeyboardLayoutID.allCases).save(to: defaults)
        #expect(TextKeyboardState.load(from: defaults, preferredLanguages: ["en"]).enabled == TextKeyboardLayoutID.allCases)
    }

    @Test func enabledOrderPersistsAndUnknownOrLastLayoutIsRejected() throws {
        let name = "textkeyboard.test.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }

        var state = TextKeyboardState(enabled: TextKeyboardLayoutID.allCases)
        #expect(state.enabled == [.englishQWERTY, .frenchAZERTY, .germanQWERTZ, .spanishQWERTY, .japaneseFlick, .zhuyin, .korean, .pinyin])
        state.moveEnabled(from: 0, to: 1)
        state.save(to: defaults)
        let restored = TextKeyboardState.load(from: defaults)
        #expect(restored.enabled == [.frenchAZERTY, .englishQWERTY, .germanQWERTZ, .spanishQWERTY, .japaneseFlick, .zhuyin, .korean, .pinyin])

        #expect(TextKeyboardState.resolve(enabledRaw: ["nope", "de-QWERTZ"], activeRaw: "nope").enabled == [.germanQWERTZ])
        #expect(TextKeyboardState.resolve(enabledRaw: ["retired-36"], activeRaw: "retired-36").enabled == [.englishQWERTY])
        #expect(throws: TextKeyboardLayoutError.unknownLayout) { try state.enable(rawValue: "dvorak") }

        var only = TextKeyboardState(enabled: [.spanishQWERTY], active: .spanishQWERTY)
        #expect(throws: TextKeyboardLayoutError.cannotRemoveLastLayout) { try only.disable(.spanishQWERTY) }
        #expect(only.enabled == [.spanishQWERTY])
        #expect(throws: TextKeyboardLayoutError.layoutNotEnabled) { try only.select(.englishQWERTY) }
    }

    @Test func suggestionsArePrefixMatchesInDeterministicOrder() {
        let vocabulary = ["Voice", "Voxt", "voxt mobile", "extra", "vox"]
        #expect(TextKeyboardSuggestions.matches(prefix: "", vocabulary: vocabulary).isEmpty)
        #expect(TextKeyboardSuggestions.matches(prefix: "xt", vocabulary: vocabulary).isEmpty)
        #expect(TextKeyboardSuggestions.matches(prefix: "vo", vocabulary: vocabulary) == ["vox", "Voxt", "Voice", "voxt mobile"])
        #expect(TextKeyboardSuggestions.matches(prefix: "VOX", vocabulary: vocabulary) == ["vox", "Voxt", "voxt mobile"])
    }

    @Test func suggestionReplacesOnlyTheMatchingCurrentWord() {
        #expect(TextKeyboardSuggestions.replacement(currentWord: "Vox", suggestion: "Voxt") == TextKeyboardReplacement(deleteCount: 3, text: "Voxt"))
        #expect(TextKeyboardSuggestions.replacement(currentWord: "vox", suggestion: "Voxt") == TextKeyboardReplacement(deleteCount: 3, text: "Voxt"))
        #expect(TextKeyboardSuggestions.replacement(currentWord: "Hello", suggestion: "Voxt") == nil)
        #expect(TextKeyboardSuggestions.replacement(currentWord: "", suggestion: "Voxt") == nil)
        #expect(TextKeyboardSuggestions.replacement(currentWord: "VoxtExtra", suggestion: "Voxt") == nil)
    }

    @Test func suggestionsStayHiddenWhenTheCaretIsInsideAWord() {
        let vocabulary = ["help", "hello"]
        #expect(TextKeyboardSuggestions.matches(before: "say he", after: "llo", vocabulary: vocabulary).isEmpty)
        #expect(TextKeyboardSuggestions.replacement(before: "say he", after: "llo", suggestion: "help") == nil)
        #expect(TextKeyboardSuggestions.matches(before: "say he", after: " ", vocabulary: vocabulary) == ["help", "hello"])
        #expect(TextKeyboardSuggestions.replacement(before: "say he", after: "", suggestion: "help") == TextKeyboardReplacement(deleteCount: 2, text: "help"))
        #expect(TextKeyboardSuggestions.replacement(before: "say he", after: ",", suggestion: "help") == TextKeyboardReplacement(deleteCount: 2, text: "help"))
    }

    @Test func dictionarySuggestionsKeepDigitsInTheSameWord() {
        #expect(TextKeyboardSuggestions.matches(before: "Qwen3", after: "", vocabulary: ["Qwen3ASR"]) == ["Qwen3ASR"])
        #expect(TextKeyboardSuggestions.replacement(before: "Qwen3", after: "", suggestion: "Qwen3ASR") == TextKeyboardReplacement(deleteCount: 5, text: "Qwen3ASR"))
        #expect(TextKeyboardSuggestions.matches(before: "he", after: "2", vocabulary: ["help"]).isEmpty)
        #expect(TextKeyboardSuggestions.replacement(before: "he", after: "2", suggestion: "help") == nil)
    }

    @Test func returningToTypingReloadsEnabledLayoutsFromStorage() {
        let name = "textkeyboard.test.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }

        var attached = TextKeyboardState(enabled: [.englishQWERTY, .frenchAZERTY], active: .frenchAZERTY, shifted: true, plane: .numbers)
        let saved = TextKeyboardState(enabled: [.spanishQWERTY, .germanQWERTZ], active: .germanQWERTZ)
        saved.save(to: defaults)
        attached.reloadEnabledLayouts(from: defaults)
        #expect(attached.enabled == [.spanishQWERTY, .germanQWERTZ])
        #expect(attached.active == .germanQWERTZ)
        #expect(attached.shifted)
        #expect(attached.plane == .numbers)
    }
}

private func characters(in layout: TextKeyboardLayout, plane: TextKeyboardPlane, row: Int? = nil) -> [String] {
    let rows = layout.rows(plane)
    let chosen = row.map { [rows[$0]] } ?? rows
    return chosen.flatMap { $0.compactMap { $0.kind == .character ? $0.text : nil } }
}

private func character(_ text: String, in id: TextKeyboardLayoutID, plane: TextKeyboardPlane) -> TextKeyboardKey? {
    TextKeyboardLayout.make(id).rows(plane).flatMap { $0 }.first { $0.kind == .character && $0.text == text }
}
