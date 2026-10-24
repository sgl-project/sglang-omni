// LocalAPIComposePromptTests.swift
// Provides Local API compose prompt-building tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class LocalAPIComposePromptTests: XCTestCase {
    func testMixedDictationRetriesPreviousPromptAndPreservesBothLanguages() async throws {
        let raw = "Tomorrow at three 开会，please bring the draft."
        var prompts: [String] = []
        let result = try await LocalAPIComposePrompt.compose(rawTranscription: raw, options: options(.dictate)) { prompt in
            prompts.append(prompt)
            return prompts.count == 1 ? "Tomorrow at three, please bring the draft." : raw
        }
        XCTAssertEqual(result, raw)
        XCTAssertEqual(prompts.count, 2)
        XCTAssertNotEqual(prompts.first, prompts.last)
        XCTAssertTrue(prompts.last?.contains("Remove fillers, repetitions and self-corrections") == true)
    }

    func testMixedDictationFallsBackAfterOneFailedRetry() async throws {
        let raw = "明天下午三点开会，记得带上 laptop 和充电器。"
        var calls = 0
        let result = try await LocalAPIComposePrompt.compose(rawTranscription: raw, options: options(.dictate)) { _ in
            calls += 1
            return "明天下午三点开会，记得带上电脑和充电器。"
        }
        XCTAssertEqual(result, raw)
        XCTAssertEqual(calls, 2)
    }

    func testMixedDictationRetryErrorFallsBackToRawText() async throws {
        let raw = "Please bring 文件 and the laptop."
        var calls = 0
        let result = try await LocalAPIComposePrompt.compose(rawTranscription: raw, options: options(.dictate)) { _ in
            calls += 1
            if calls == 2 { throw LocalAPIError(message: "test failure") }
            return "Please bring documents and the laptop."
        }
        XCTAssertEqual(result, raw)
        XCTAssertEqual(calls, 2)
    }

    func testMixedDictationDoesNotRetryPreservedScriptsOrPureLanguages() async throws {
        for (raw, output) in [
            ("Um, please bring 文件 and the laptop.", "Please bring 文件 and the laptop."),
            ("Um, bring the laptop.", "Bring the laptop."),
            ("嗯，带上电脑。", "带上电脑。"),
            ("Use variable x for 中文内容。", "Use variable x for 中文内容。")
        ] {
            var calls = 0
            let result = try await LocalAPIComposePrompt.compose(rawTranscription: raw, options: options(.dictate)) { _ in
                calls += 1
                return output
            }
            XCTAssertEqual(result, output)
            XCTAssertEqual(calls, 1)
        }
    }

    func testMixedDictationLossThresholdAndSmallScriptControls() {
        let raw = "一二三四五六七八九十 one two three four five six seven eight nine ten"
        XCTAssertTrue(LocalAPIComposePrompt.losesMixedDictationContent("一二 one two three", rawText: raw))
        XCTAssertTrue(LocalAPIComposePrompt.losesMixedDictationContent("一二三 one two", rawText: raw))
        XCTAssertFalse(LocalAPIComposePrompt.losesMixedDictationContent("一二三 one two three", rawText: raw))
        XCTAssertFalse(LocalAPIComposePrompt.losesMixedDictationContent("English only", rawText: "English only"))
        XCTAssertFalse(LocalAPIComposePrompt.losesMixedDictationContent("纯中文", rawText: "纯中文内容"))
        XCTAssertFalse(LocalAPIComposePrompt.losesMixedDictationContent("中文内容", rawText: "中文内容 x"))
        XCTAssertFalse(LocalAPIComposePrompt.losesMixedDictationContent("a name", rawText: "李 a name"))
    }

    func testMixedDictationCancellationDoesNotReturnFallback() async throws {
        var calls = 0
        do {
            _ = try await LocalAPIComposePrompt.compose(rawTranscription: "Please bring 文件 and laptop.", options: options(.dictate)) { _ in
                calls += 1
                if calls == 2 { throw CancellationError() }
                return "Please bring documents and laptop."
            }
            XCTFail("Cancellation must propagate")
        } catch is CancellationError { }
        XCTAssertEqual(calls, 2)
    }

    func testMixedScriptGuardDoesNotChangeExplicitTranslation() async throws {
        var calls = 0
        let result = try await LocalAPIComposePrompt.compose(rawTranscription: "Please bring 文件 and laptop.", options: options(.translate, targetLanguage: "en")) { _ in
            calls += 1
            return "Please bring documents and laptop."
        }
        XCTAssertEqual(result, "Please bring documents and laptop.")
        XCTAssertEqual(calls, 1)
    }

    func testChineseDictationAnsweredInEnglishIsRetriedInChinese() async throws {
        let raw = "嗯明天下午三点我们在三楼会议室开会 讨论下个季度的发布计划 大家提前准备好材料"
        var prompts: [String] = []
        let result = try await LocalAPIComposePrompt.compose(rawTranscription: raw, options: options(.dictate)) { prompt in
            prompts.append(prompt)
            return prompts.count == 1
                ? "We will meet tomorrow at 3 PM in the third floor meeting room to discuss next quarter's release plan."
                : "明天下午三点我们在三楼会议室开会，讨论下个季度的发布计划，大家提前准备好材料。"
        }
        XCTAssertEqual(result, "明天下午三点我们在三楼会议室开会，讨论下个季度的发布计划，大家提前准备好材料。")
        XCTAssertEqual(prompts.count, 2)
        let retry = try XCTUnwrap(prompts.last)
        let note = try XCTUnwrap(retry.range(of: EnhancementLanguageGuard.retryInstruction(for: raw)))
        let transcript = try XCTUnwrap(retry.range(of: EnhancementPromptResolver.rawTranscriptionTemplateVariable))
        XCTAssertLessThan(note.lowerBound, transcript.lowerBound, "the language line sits before the transcript")
    }

    func testChineseDictationStillInEnglishAfterRetryReturnsTheTranscript() async throws {
        let raw = "嗯明天下午三点我们在三楼会议室开会 讨论下个季度的发布计划 大家提前准备好材料"
        var calls = 0
        let result = try await LocalAPIComposePrompt.compose(rawTranscription: raw, options: options(.dictate)) { _ in
            calls += 1
            return "We will meet tomorrow at 3 PM in the third floor meeting room to discuss next quarter's release plan."
        }
        XCTAssertEqual(result, raw)
        XCTAssertEqual(calls, 2)
    }

    func testTranslationIntoAnotherScriptIsNotRetried() async throws {
        let raw = "明天下午三点我们在三楼会议室开会，讨论下个季度的发布计划。"
        var calls = 0
        let result = try await LocalAPIComposePrompt.compose(rawTranscription: raw, options: options(.translate, targetLanguage: "en")) { _ in
            calls += 1
            return "We will meet at 3 PM tomorrow in the third floor meeting room to discuss next quarter's release plan."
        }
        XCTAssertEqual(result, "We will meet at 3 PM tomorrow in the third floor meeting room to discuss next quarter's release plan.")
        XCTAssertEqual(calls, 1)
    }

    private func options(
        _ mode: LocalAPIComposeMode,
        targetLanguage: String? = nil,
        selectedText: String = "",
        contextBefore: String = "",
        contextAfter: String = "",
        style: LocalAPIComposeStyle = .automatic,
        vocabulary: [String] = [],
        languageVariants: [String] = [],
        styleExamples: [LocalAPIStyleExample] = []
    ) -> LocalAPIComposeOptions {
        LocalAPIComposeOptions(
            mode: mode,
            targetLanguage: targetLanguage,
            selectedText: selectedText,
            contextBefore: contextBefore,
            contextAfter: contextAfter,
            style: style,
            vocabulary: vocabulary,
            languageVariants: languageVariants,
            styleExamples: styleExamples
        )
    }

    // MARK: Developer prompts (phone Settings > Developer > Prompts)

    func testCustomPromptReplacesTheBuiltInInstructionForThatRequestOnly() throws {
        var dictate = options(.dictate)
        dictate.customPrompt = "Write the speech as a haiku."
        let prompt = try XCTUnwrap(LocalAPIComposePrompt.instruction(for: dictate, rawTranscription: "Hello there."))
        XCTAssertTrue(prompt.hasPrefix("Write the speech as a haiku."))
        XCTAssertTrue(prompt.contains("{{RAW_TRANSCRIPTION}}"))
        XCTAssertFalse(prompt.contains("Turn this speech transcript into the text this speaker would have written themselves:"))
        let builtIn = try XCTUnwrap(LocalAPIComposePrompt.instruction(for: options(.dictate), rawTranscription: "Hello there."))
        XCTAssertTrue(builtIn.contains("Turn this speech transcript into the text this speaker would have written themselves:"))

        var translate = options(.translate, targetLanguage: "fr")
        translate.customPrompt = "Translate into {{TARGET_LANGUAGE}} like a pirate."
        let pirate = try XCTUnwrap(LocalAPIComposePrompt.instruction(for: translate, rawTranscription: "Meet at three, sorry, four."))
        XCTAssertTrue(pirate.hasPrefix("Translate into fr like a pirate."))
        XCTAssertFalse(pirate.contains("Corrected:"))

        var edit = options(.edit, selectedText: "See you Monday.")
        edit.customPrompt = "Apply the instruction."
        let edited = try XCTUnwrap(LocalAPIComposePrompt.instruction(for: edit, rawTranscription: "Make it Tuesday."))
        XCTAssertTrue(edited.hasPrefix("Apply the instruction."))
        XCTAssertTrue(edited.contains("See you Monday."))
    }

    func testCustomPromptKeepsTheSafetyGuards() async throws {
        var edit = options(.edit, selectedText: "See you Monday.")
        edit.customPrompt = "Apply the instruction."
        var calls = 0
        do {
            _ = try await LocalAPIComposePrompt.compose(rawTranscription: "Make it Tuesday.", options: edit) { _ in
                calls += 1; return "See you Monday."
            }
            XCTFail("an unchanged edit must not be applied")
        } catch { XCTAssertTrue(error is LocalAPIComposePrompt.EditNotApplied) }
        XCTAssertEqual(calls, 2)
    }

    func testCustomPromptIsDecodedTrimmedAndLengthLimited() throws {
        func decode(_ prompt: String) throws -> LocalAPIComposeOptions {
            try JSONDecoder().decode(LocalAPIComposeOptions.self, from: JSONSerialization.data(withJSONObject: ["mode": "dictate", "customPrompt": prompt]))
        }
        XCTAssertEqual(try decode("  Keep it short.  ").customPrompt, "Keep it short.")
        XCTAssertNil(try decode("   ").customPrompt)
        XCTAssertNil(try JSONDecoder().decode(LocalAPIComposeOptions.self, from: Data(#"{"mode":"dictate"}"#.utf8)).customPrompt)
        XCTAssertNil(try decode(String(repeating: "a", count: 4_000)).validationError())
        XCTAssertEqual(try decode(String(repeating: "a", count: 4_001)).validationError(), "Use a custom prompt of at most 4,000 characters.")
    }

    func testBuiltInPromptsListEveryModeForThePhoneEditor() {
        let prompts = LocalAPIComposePrompt.builtInPrompts()
        XCTAssertEqual(Set(prompts.keys), ["dictate", "edit", "translate", "write"])
        XCTAssertTrue(prompts["dictate"]?.contains("Turn this speech transcript into the text this speaker would have written themselves:") == true)
        XCTAssertTrue(prompts["translate"]?.contains("{{TARGET_LANGUAGE}}") == true)
        XCTAssertFalse(prompts.values.contains { $0.contains("{{RAW_TRANSCRIPTION}}") })
    }

    private let wordingExample = LocalAPIStyleExample(
        before: "Thanks, I'll send Alice the file at 3.",
        after: "Cheers — I'll send Alice the file at 3."
    )

    // Cleaned outputs of Qwen3.5-4B for dictated messages with uncounted points.
    func testMessageWithGreetingAndThreePointsIsLaidOutWithANumberedList() {
        XCTAssertEqual(
            LocalAPIComposePrompt.laidOutMessage("Hi Karen, this is Lin's son, from unit 4B. Her heater stopped working Tuesday, the bedroom window won't close, and she still needs the October rent receipt. Could you send someone this week? Thanks."),
            "Hi Karen,\n\nThis is Lin's son, from unit 4B.\n\n1. Her heater stopped working Tuesday.\n2. The bedroom window won't close.\n3. She still needs the October rent receipt.\n\nCould you send someone this week?\n\nThanks."
        )
        XCTAssertEqual(
            LocalAPIComposePrompt.laidOutMessage("Hey Chris, the printer is out of toner, the wifi in the meeting room keeps dropping, and the coffee machine is leaking again. Can you take a look today? Thanks, Jo."),
            "Hey Chris,\n\n1. The printer is out of toner.\n2. The wifi in the meeting room keeps dropping.\n3. The coffee machine is leaking again.\n\nCan you take a look today?\n\nThanks,\nJo"
        )
        XCTAssertEqual(
            LocalAPIComposePrompt.laidOutMessage("王姐您好，\n\n暖气周二坏了，卧室的窗户关不上，还有十月的房租收据我还没拿到。这周能找人来看看吗？\n\n谢谢。"),
            "王姐您好，\n\n1. 暖气周二坏了。\n2. 卧室的窗户关不上。\n3. 十月的房租收据我还没拿到。\n\n这周能找人来看看吗？\n\n谢谢。"
        )
        XCTAssertEqual(
            LocalAPIComposePrompt.laidOutMessage("嗯，王姐您好，我是林阿姨的儿子，住 4B 的。\n\n那个，她家暖气周二坏了，卧室的窗户关不上，还有她还需要十月的房租收据。这周能找人来看看吗？谢谢。"),
            "王姐您好，\n\n我是林阿姨的儿子，住 4B 的。\n\n1. 她家暖气周二坏了。\n2. 卧室的窗户关不上。\n3. 她还需要十月的房租收据。\n\n这周能找人来看看吗？\n\n谢谢。"
        )
    }

    func testMessageWithAnExistingListGetsItsGreetingAndSignOffOnTheirOwnLines() {
        XCTAssertEqual(
            LocalAPIComposePrompt.laidOutMessage("大家好，说一下发布的事。\n\n1. 测试版周五发出去。\n2. 文档快写完了。\n\n有空的请告诉我，谢谢，小林。"),
            "大家好，\n\n说一下发布的事。\n\n1. 测试版周五发出去。\n2. 文档快写完了。\n\n有空的请告诉我。\n\n谢谢，\n小林"
        )
        XCTAssertEqual(
            LocalAPIComposePrompt.laidOutMessage("Email to my landlord, Hi Karen, a few things:\n1. The heater stopped working on Monday.\n2. The bedroom window won't close.\n\nCould you send someone this week? Thanks."),
            "Email to my landlord.\n\nHi Karen,\n\nA few things:\n1. The heater stopped working on Monday.\n2. The bedroom window won't close.\n\nCould you send someone this week?\n\nThanks."
        )
    }

    func testMessageLayoutLeavesTextWithoutAGreetingAndAListUnchanged() {
        for text in [
            "Hi Karen, the heater stopped working on Tuesday, so it's pretty cold in here. Could you send someone this week? Thanks.",
            "Hi Mom, I'm home now. See you tomorrow.",
            "We arrive at ten, check in at noon, and leave at five.",
            "Hi Sam, I bought apples, milk, and bread.",
            "王姐您好，暖气周二坏了，所以屋里挺冷的。这周能找人来看看吗？谢谢。",
            "明天的火车八点出发，九点到达。我会提前到车站等你。",
            "This was my first visit to that bookstore. It was quiet, and I would like to go again next week.",
        ] {
            XCTAssertEqual(LocalAPIComposePrompt.laidOutMessage(text), text)
        }
    }

    // Qwen3-ASR wrote "Monday. Ah, sorry Tuesday" for "Monday, uh, sorry, Tuesday",
    // and Dictate cleanup then kept both days.
    func testSelfCorrectionsAcrossAFillerOrAFullStopBeforeSorry() {
        XCTAssertEqual(
            LocalAPIComposePrompt.appliedSelfCorrections("So her heater stopped working Monday. Ah, sorry Tuesday, the bedroom window won't close."),
            "So her heater stopped working Tuesday, the bedroom window won't close."
        )
        XCTAssertEqual(
            LocalAPIComposePrompt.appliedSelfCorrections("the heater stopped working Monday, uh, sorry, Tuesday, the window"),
            "the heater stopped working Tuesday, the window"
        )
        XCTAssertEqual(LocalAPIComposePrompt.appliedSelfCorrections("暖气周一，呃，不对，周二坏了"), "暖气周二坏了")
        for unchanged in ["We arrive Monday. Actually, Tuesday works better for Sam.", "We arrive at ten. Actually, bring a coat.",
                          "I'm sorry, Tuesday is full."] {
            XCTAssertEqual(LocalAPIComposePrompt.appliedSelfCorrections(unchanged), unchanged)
        }
    }

    func testDictateAppliesAKeptSelfCorrectionAndDropsAStraySoBeforeLayingOut() async throws {
        let cleaned = "Hi Karen. So her heater stopped working Monday. Ah, sorry, Tuesday, the bedroom window won't close, and she still needs the October rent receipt. Thanks."
        let result = try await LocalAPIComposePrompt.compose(rawTranscription: "Um, " + cleaned, options: options(.dictate)) { _ in cleaned }
        XCTAssertEqual(result, "Hi Karen,\n\n1. Her heater stopped working Tuesday.\n2. The bedroom window won't close.\n3. She still needs the October rent receipt.\n\nThanks.")
    }

    // Qwen3.5-4B wrote counted points as "First, …\n- Second, …" and "- 第一，…".
    func testCountedLinesBecomeANumberedList() {
        XCTAssertEqual(
            LocalAPIComposePrompt.numberedCountedLines("First, save the report.\n- Second, send the email.\n- Third, close the window."),
            "1. Save the report.\n2. Send the email.\n3. Close the window."
        )
        XCTAssertEqual(
            LocalAPIComposePrompt.numberedCountedLines("明天出发之前：\n- 第一，带好护照\n- 第二，今晚把手机充满电\n- 还有，早上七点在车站集合"),
            "明天出发之前：\n1. 带好护照\n2. 今晚把手机充满电\n3. 早上七点在车站集合"
        )
        for unchanged in ["First, I want to thank you.\nThe office is open.", "Bring three things:\n- keys\n- a coat", "This was my first visit.",
                          "Second, send it.\nThird, close it."] {
            XCTAssertEqual(LocalAPIComposePrompt.numberedCountedLines(unchanged), unchanged)
        }
    }

    func testCountedPointsInsideAParagraphBecomeANumberedList() {
        XCTAssertEqual(
            LocalAPIComposePrompt.numberedCountedLines("Okay, for tomorrow's trip, first, bring your passport, second, charge your phone tonight, and third, we meet at the station at seven."),
            "Okay, for tomorrow's trip:\n1. Bring your passport.\n2. Charge your phone tonight.\n3. We meet at the station at seven."
        )
        XCTAssertEqual(
            LocalAPIComposePrompt.numberedCountedLines("Hi Mark, I'm writing about a few problems. First, the sink is leaking. Second, the heater doesn't turn on, and also, the lock keeps sticking. Could you send someone? Thanks, Alex."),
            "Hi Mark, I'm writing about a few problems.\n\n1. The sink is leaking.\n2. The heater doesn't turn on.\n3. The lock keeps sticking.\n\nCould you send someone? Thanks, Alex."
        )
        XCTAssertEqual(
            LocalAPIComposePrompt.numberedCountedLines("大家好，说一下发布的事。第一，测试版周五发出去，第二，文档快写完了，还有，安卓版本还需要有人测一下。有空的请告诉我。"),
            "大家好，说一下发布的事。\n\n1. 测试版周五发出去。\n2. 文档快写完了。\n3. 安卓版本还需要有人测一下。\n\n有空的请告诉我。"
        )
        for unchanged in ["This was my first visit to that bookstore. It was quiet.", "First, I want to thank you. The office is open.",
                          "I also need the first draft, and the second one."] {
            XCTAssertEqual(LocalAPIComposePrompt.numberedCountedLines(unchanged), unchanged)
        }
        XCTAssertEqual(
            LocalAPIComposePrompt.laidOutMessage("Hi Karen, so, the heater stopped working Tuesday, the bedroom window won't close, and I still need the receipt. Thanks."),
            "Hi Karen,\n\n1. The heater stopped working Tuesday.\n2. The bedroom window won't close.\n3. I still need the receipt.\n\nThanks."
        )
    }

    // Qwen3-ASR joined the context sentence to the points with ", so".
    func testPointsAfterASoClauseSplitOffTheClauseBeforeThem() {
        XCTAssertEqual(
            LocalAPIComposePrompt.laidOutMessage("Hi Karen, I'm writing for my mom Grace in Unit 4B, so her heater stopped working Tuesday, the bedroom window won't close, and she still needs the October rent receipt. Could you send someone this week? Thanks."),
            "Hi Karen,\n\nI'm writing for my mom Grace in Unit 4B.\n\n1. Her heater stopped working Tuesday.\n2. The bedroom window won't close.\n3. She still needs the October rent receipt.\n\nCould you send someone this week?\n\nThanks."
        )
        let twoParts = "Hi Karen, the heater stopped working on Tuesday, so it's pretty cold in here, and the window is stuck too. Thanks."
        XCTAssertEqual(LocalAPIComposePrompt.laidOutMessage(twoParts), twoParts)
    }

    // Qwen3.5-4B returned the whole email with only the selected sentence changed.
    func testEditThatReturnsTheFieldAroundTheSelectionKeepsOnlyTheEditedSelection() async throws {
        let before = "Hi Karen,\n\n1. Her heater stopped working Tuesday.\n2. The bedroom window won't close.\n\n"
        let after = "\n\nThanks."
        let result = try await LocalAPIComposePrompt.compose(
            rawTranscription: "Make this firmer but polite.",
            options: options(.edit, selectedText: "Could you send someone this week?", contextBefore: before, contextAfter: after)
        ) { _ in "Hi Karen,\n\n1. Her heater stopped working Tuesday.\n2. The bedroom window won't close.\n\nPlease send someone this week.\n\nThanks." }
        XCTAssertEqual(result, "Please send someone this week.")
    }

    // With the whole structured mail after the selection, Qwen3.5-4B rewrote the whole mail for
    // "make this firmer but polite" (3 of 3 tries).
    func testEditPromptPutsTheFieldBeforeTheSelectionNextToTheInstruction() throws {
        let prompt = try XCTUnwrap(LocalAPIComposePrompt.instruction(
            for: options(.edit, selectedText: "Could you send someone this week?",
                         contextBefore: "Hi Karen,\n\n1. Her heater stopped working Tuesday.\n\n", contextAfter: "\n\nThanks."),
            rawTranscription: "Make this firmer but polite."
        ))
        let before = try XCTUnwrap(prompt.range(of: "Text before the selected text"))
        let after = try XCTUnwrap(prompt.range(of: "Text after the selected text"))
        let selection = try XCTUnwrap(prompt.range(of: "Selected text: Could you send someone this week?\n\nInstruction: {{RAW_TRANSCRIPTION}}"))
        XCTAssertLessThan(before.lowerBound, selection.lowerBound)
        XCTAssertLessThan(after.lowerBound, selection.lowerBound)
    }

    // "make this firmer but polite" added "Thank you." before the mail's "Thanks.".
    func testEditDropsAnAddedClosingThatTheFieldAlreadyHasRightAfterTheSelection() async throws {
        let edit = "Please arrange for someone to be sent over this week. Thank you."
        let mail = try await LocalAPIComposePrompt.compose(
            rawTranscription: "Make this firmer but polite.",
            options: options(.edit, selectedText: "Could you send someone this week?",
                             contextBefore: "Hi Karen,\n\n1. Her heater stopped working Tuesday.\n\n", contextAfter: "\n\nThanks.")
        ) { _ in edit }
        XCTAssertEqual(mail, "Please arrange for someone to be sent over this week.")
        let zh = try await LocalAPIComposePrompt.compose(
            rawTranscription: "语气坚定一点。", options: options(.edit, selectedText: "这周能找人来看看吗？", contextAfter: "\n\n谢谢。")
        ) { _ in "请这周务必派人来看一下。谢谢！" }
        XCTAssertEqual(zh, "请这周务必派人来看一下。")
        let noClosingAfter = try await LocalAPIComposePrompt.compose(
            rawTranscription: "Make this more polite.", options: options(.edit, selectedText: "Send me the slides today.")
        ) { _ in "Could you please send me the slides today? Thank you!" }
        XCTAssertEqual(noClosingAfter, "Could you please send me the slides today? Thank you!")
    }

    // "make this shorter" on "2. The bedroom window …" dropped the "2.", the
    // date/time guard read "2" as a time and forced a retry that returned the line unchanged.
    func testEditOfAListItemEditsTheTextAndKeepsTheMarker() async throws {
        var prompts: [String] = []
        let item = try await LocalAPIComposePrompt.compose(
            rawTranscription: "Make this shorter.",
            options: options(.edit, selectedText: "2. The bedroom window in the back room won't close properly anymore.",
                             contextBefore: "Hi Karen,\n\n1. Her heater stopped working Tuesday.\n", contextAfter: "\n3. She still needs the receipt.")
        ) { prompt in prompts.append(prompt); return "The bedroom window won't close." }
        XCTAssertEqual(item, "2. The bedroom window won't close.")
        XCTAssertEqual(prompts.count, 1)
        XCTAssertTrue(prompts[0].contains("Selected text: The bedroom window in the back room won't close properly anymore."))
        let keptMarker = try await LocalAPIComposePrompt.compose(
            rawTranscription: "Make this shorter.", options: options(.edit, selectedText: "- The bedroom window in the back room won't close properly anymore.")
        ) { _ in "- The bedroom window won't close." }
        XCTAssertEqual(keptMarker, "- The bedroom window won't close.")
        let time = try await LocalAPIComposePrompt.compose(
            rawTranscription: "Make this shorter.", options: options(.edit, selectedText: "We meet at 2 tomorrow in the big room downstairs.")
        ) { _ in "We meet at 2 tomorrow downstairs." }
        XCTAssertEqual(time, "We meet at 2 tomorrow downstairs.")
    }

    func testDictateReturnsTheLaidOutMessage() async throws {
        let cleaned = "Hi Karen, the heater stopped working Tuesday, the bedroom window won't close, and I still need the October rent receipt. Thanks."
        let result = try await LocalAPIComposePrompt.compose(rawTranscription: "Um, hi Karen, " + cleaned.dropFirst(10), options: options(.dictate)) { _ in cleaned }
        XCTAssertEqual(result, "Hi Karen,\n\n1. The heater stopped working Tuesday.\n2. The bedroom window won't close.\n3. I still need the October rent receipt.\n\nThanks.")
    }

    func testWritePromptUsesInstructionLanguageAndSameLanguageMessageExample() throws {
        let chinese = try XCTUnwrap(LocalAPIComposePrompt.instruction(
            for: options(.write), rawTranscription: "告诉同事我会晚到十分钟，请他们先开始。"
        ))
        XCTAssertTrue(chinese.contains("根据指令写出可以直接发给对方的消息。"))
        XCTAssertTrue(chinese.contains("消息：小陈，我不能参加午餐了，祝你们吃得开心！"))
        XCTAssertTrue(chinese.contains("指令：{{RAW_TRANSCRIPTION}}"))
        XCTAssertFalse(chinese.contains("Speech transcript:"))
        let english = try XCTUnwrap(LocalAPIComposePrompt.instruction(
            for: options(.write), rawTranscription: "Thank Alex for the notes."
        ))
        XCTAssertTrue(english.contains("Write the actual message requested by the instruction"))
        XCTAssertTrue(english.contains("Message: Pat, I cannot attend the lunch. Have a good time!"))
        XCTAssertTrue(english.contains("Instruction: {{RAW_TRANSCRIPTION}}"))
        XCTAssertFalse(english.contains("Speech transcript:"))
    }

    func testEditPromptMatchesSelectionLanguageAndKeepsUnrequestedDetails() throws {
        let chinese = try XCTUnwrap(LocalAPIComposePrompt.instruction(
            for: options(.edit, selectedText: "我们明天下午三点在会议室开会。"),
            rawTranscription: "简短一点。"
        ))
        XCTAssertTrue(chinese.contains("要求缩短或改变语气等整段修改时"))
        XCTAssertTrue(chinese.contains("同音词错误"))
        XCTAssertTrue(chinese.contains("选中文字：我们明天下午三点在会议室开会。"))
        XCTAssertTrue(chinese.contains("指令：{{RAW_TRANSCRIPTION}}"))
        XCTAssertTrue(chinese.hasSuffix("输出："))
        let english = try XCTUnwrap(LocalAPIComposePrompt.instruction(
            for: options(.edit, selectedText: "Can we meet Thursday at three?"),
            rawTranscription: "Actually make it Friday."
        ))
        XCTAssertTrue(english.contains("For a whole-text change such as shortening or changing tone"))
        XCTAssertTrue(english.contains("sound-alike errors"))
        XCTAssertTrue(english.contains("Output: The workshop is on Wednesday at noon."))
        XCTAssertTrue(english.contains("Selected text: Can we meet Thursday at three?"))
        XCTAssertTrue(english.contains("Instruction: {{RAW_TRANSCRIPTION}}"))
        XCTAssertTrue(english.hasSuffix("Output:"))
    }

    func testUnchangedEditRetriesOnceUnlessNoChangeWasRequested() async throws {
        let selection = "Can we meet Thursday at three?"
        var prompts: [String] = []
        let changed = try await LocalAPIComposePrompt.compose(
            rawTranscription: "actually make it Friday",
            options: options(.edit, selectedText: selection)
        ) { prompt in
            prompts.append(prompt)
            return prompts.count == 1 ? selection : "Can we meet Friday at three?"
        }
        XCTAssertEqual(changed, "Can we meet Friday at three?")
        XCTAssertEqual(prompts.count, 2)
        XCTAssertTrue(prompts[1].contains("The first attempt did not safely apply the edit"))
        prompts.removeAll()
        let unchanged = try await LocalAPIComposePrompt.compose(
            rawTranscription: "leave this unchanged",
            options: options(.edit, selectedText: selection)
        ) { prompt in
            prompts.append(prompt)
            return selection
        }
        XCTAssertEqual(unchanged, selection)
        XCTAssertEqual(prompts.count, 1)
    }

    func testEditRejectsInstructionFragmentAndRetriesOnlyOnce() async throws {
        let selection = "The server has a cash limit."
        for bad in ["The server has a cash to cash limit.", "The server has a cash-to-cash limit."] {
            var calls = 0
            let result = try await LocalAPIComposePrompt.compose(
                rawTranscription: "Fix cash to cash.", options: options(.edit, selectedText: selection)
            ) { _ in calls += 1; return calls == 1 ? bad : "The server has a cache limit." }
            XCTAssertEqual(result, "The server has a cache limit.")
            XCTAssertEqual(calls, 2)
            calls = 0
            do {
                _ = try await LocalAPIComposePrompt.compose(
                    rawTranscription: "Fix cash to cash.", options: options(.edit, selectedText: selection)
                ) { _ in calls += 1; return bad }
                XCTFail("unsafe second attempt must report not_applied")
            } catch { XCTAssertEqual(error.localizedDescription, "Couldn't apply that edit") }
            XCTAssertEqual(calls, 2)
        }
    }

    func testEditKeepsExistingFragmentAndAllowsRequestedReplacement() async throws {
        for (selection, instruction, output) in [
            ("Compare cash to cash accounts.", "Make this polite.", "Please compare cash to cash accounts."),
            ("The meeting is on Tuesday.", "Change Tuesday to Friday.", "The meeting is on Friday."),
            ("请保护消费者的权利。", "把权利改成权利。", "请保护消费者的权力。")
        ] {
            var calls = 0
            let result = try await LocalAPIComposePrompt.compose(rawTranscription: instruction,
                options: options(.edit, selectedText: selection)) { _ in calls += 1; return output }
            XCTAssertEqual(result, output)
            XCTAssertEqual(calls, 1)
        }
    }

    func testPlainDictateAlsoGetsTheCleanupPrompt() {
        let prompt = LocalAPIComposePrompt.instruction(for: options(.dictate))!
        XCTAssertTrue(prompt.contains("Turn this speech transcript"), "even a default compose dictate asks for cleanup")
        XCTAssertTrue(prompt.contains("fillers"))
        XCTAssertTrue(prompt.contains("Keep their meaning"), "cleanup keeps what was said")
        XCTAssertTrue(prompt.contains(EnhancementPromptResolver.rawTranscriptionTemplateVariable))
        XCTAssertFalse(prompt.contains("Tone:"), "automatic style adds no tone line")
        XCTAssertFalse(prompt.contains("Prefer these spellings"))
        XCTAssertFalse(prompt.contains("Text before the cursor"))
    }

    func testDictatePromptPreservesSpokenLanguageWithoutVariants() {
        let prompt = LocalAPIComposePrompt.instruction(for: options(.dictate))!
        XCTAssertTrue(prompt.contains("in the language or mix of languages they spoke"), "dictate keeps the spoken languages, mix included")
    }

    func testDictateWithOptionsBuildsACleanupPrompt() {
        let prompt = LocalAPIComposePrompt.instruction(
            for: options(.dictate, contextBefore: "Dear team,", style: .formal, vocabulary: ["SGLang"])
        )!
        XCTAssertTrue(prompt.contains("Turn this speech transcript"), "option-carrying dictate asks for cleanup")
        XCTAssertTrue(prompt.contains("fillers"), "cleanup names fillers and repetitions")
        XCTAssertFalse(prompt.contains("Dear team,"), "plain cleanup gets no cursor context")
        XCTAssertTrue(prompt.contains("formal"))
        XCTAssertTrue(prompt.contains("SGLang"))
        XCTAssertTrue(prompt.contains("Keep their meaning"), "cleanup keeps what was said")
        XCTAssertTrue(prompt.contains(EnhancementPromptResolver.rawTranscriptionTemplateVariable))
    }

    func testEditPromptUsesSurroundingContext() {
        let prompt = LocalAPIComposePrompt.instruction(
            for: options(.edit, selectedText: "the sentence", contextBefore: "Dear team,", contextAfter: "Best regards")
        )!
        XCTAssertTrue(prompt.contains("Dear team,"), "edit keeps the text before the cursor")
        XCTAssertTrue(prompt.contains("Best regards"), "edit keeps the text after the cursor")
    }

    func testTranslatePromptNamesLanguageAndEmbedsSpeech() {
        let prompt = LocalAPIComposePrompt.instruction(for: options(.translate, targetLanguage: "zh-Hans"))!
        XCTAssertTrue(prompt.contains("zh-Hans"), "translate prompt names the target language")
        XCTAssertTrue(prompt.contains(EnhancementPromptResolver.rawTranscriptionTemplateVariable))
        XCTAssertTrue(prompt.contains("only the translation"))
    }

    func testTranslatePromptCleansFillersWithoutDroppingMeaning() {
        let prompt = LocalAPIComposePrompt.instruction(for: options(.translate, targetLanguage: "en"))!
        XCTAssertTrue(prompt.contains("fillers"), "translate must clean fillers, not translate them")
        XCTAssertTrue(prompt.contains("self-corrections"))
        XCTAssertTrue(prompt.lowercased().contains("preserve meaning"), "cleanup must not drop intended meaning")
    }

    func testEditPromptEmbedsSelectionAndInstruction() {
        let prompt = LocalAPIComposePrompt.instruction(
            for: options(.edit, selectedText: "the original sentence")
        )!
        XCTAssertTrue(prompt.contains("the original sentence"), "edit prompt carries the selected text")
        XCTAssertTrue(prompt.contains(EnhancementPromptResolver.rawTranscriptionTemplateVariable))
        XCTAssertFalse(prompt.contains("Selected text: \n"))
    }

    func testWritePromptCarriesContextAroundTheInstruction() {
        let prompt = LocalAPIComposePrompt.instruction(
            for: options(.write, contextBefore: "Dear team,", contextAfter: "Best regards")
        )!
        XCTAssertTrue(prompt.contains("Dear team,"))
        XCTAssertTrue(prompt.contains("Best regards"))
        XCTAssertTrue(prompt.contains(EnhancementPromptResolver.rawTranscriptionTemplateVariable))
    }

    // MARK: Tone edits, translated self-corrections and Write placeholders

    func testEditPromptShowsAWholeTextToneChangeInBothLanguages() throws {
        let english = try XCTUnwrap(LocalAPIComposePrompt.instruction(
            for: options(.edit, selectedText: "You never listen to me."), rawTranscription: "Make this calmer."
        ))
        XCTAssertTrue(english.contains("Instruction: Make this friendlier."))
        XCTAssertTrue(english.contains("A tone or length instruction always changes the wording"))
        let chinese = try XCTUnwrap(LocalAPIComposePrompt.instruction(
            for: options(.edit, selectedText: "你从来不听我说话。"), rawTranscription: "语气缓和一点。"
        ))
        XCTAssertTrue(chinese.contains("指令：语气友好一点。"))
        XCTAssertTrue(chinese.contains("要求改变语气或长度时，必须改写措辞"))
    }

    func testUnchangedToneEditRetryAsksForADifferentWording() async throws {
        let selection = "You never listen to me. I'm done talking about this tonight."
        var prompts: [String] = []
        let result = try await LocalAPIComposePrompt.compose(
            rawTranscription: "Make this calmer.", options: options(.edit, selectedText: selection, contextBefore: selection)
        ) { prompt in
            prompts.append(prompt)
            return prompts.count == 1 ? selection : "I don't feel heard, and I'd like to stop talking about this tonight."
        }
        XCTAssertEqual(result, "I don't feel heard, and I'd like to stop talking about this tonight.")
        XCTAssertEqual(prompts.count, 2)
        XCTAssertTrue(prompts[1].contains("it returned the selected text unchanged"))
        XCTAssertFalse(prompts[1].contains("never the replacement instruction"), "an unchanged result is not instruction leakage")
    }

    func testEditPromptOmitsContextThatOnlyRepeatsTheSelection() throws {
        let selection = "You never listen to me. I'm done talking about this tonight."
        let prompt = try XCTUnwrap(LocalAPIComposePrompt.instruction(
            for: options(.edit, selectedText: selection, contextBefore: selection)
        ))
        XCTAssertFalse(prompt.contains("Text before the cursor"))
        let longer = try XCTUnwrap(LocalAPIComposePrompt.instruction(
            for: options(.edit, selectedText: "tonight.", contextBefore: "I'm done talking about this tonight.")
        ))
        XCTAssertTrue(longer.contains("Text before the cursor (already in the field; do not repeat it):\nI'm done talking about this tonight."))
    }

    func testTranslatePromptAppliesSelfCorrectionsWithATargetLanguageExample() throws {
        let chinese = try XCTUnwrap(LocalAPIComposePrompt.instruction(for: options(.translate, targetLanguage: "zh-Hans")))
        XCTAssertTrue(chinese.contains("keep only the corrected detail"))
        XCTAssertTrue(chinese.contains("Speech transcript: Call Sam on Monday, actually Tuesday, at ten."))
        XCTAssertTrue(chinese.contains("Translation: 周二十点给Sam打电话。"))
        XCTAssertTrue(chinese.hasSuffix("Translation:"))
        let english = try XCTUnwrap(LocalAPIComposePrompt.instruction(for: options(.translate, targetLanguage: "en")))
        XCTAssertTrue(english.contains("Speech transcript: 周一，不对，周二上午十点给小陈打电话。"))
        XCTAssertTrue(english.contains("Translation: Call Xiao Chen on Tuesday at 10 a.m."))
        let japanese = try XCTUnwrap(LocalAPIComposePrompt.instruction(for: options(.translate, targetLanguage: "ja")))
        XCTAssertTrue(japanese.contains("keep only the corrected detail"))
        XCTAssertFalse(japanese.contains("Call Sam"), "no example in a language other than the target")
        XCTAssertFalse(japanese.contains("Example:"))
    }

    func testTranslateWithACorrectionCueWritesTheCorrectedLineFirstAndReturnsOnlyTheTranslation() async throws {
        var prompts: [String] = []
        let corrected = try await LocalAPIComposePrompt.compose(
            rawTranscription: "Pick me up at the station... no, at the airport.",
            options: options(.translate, targetLanguage: "zh-Hans")
        ) { prompt in
            prompts.append(prompt)
            return "Corrected: Pick me up at the airport.\nTranslation: 到机场接我。"
        }
        XCTAssertEqual(corrected, "到机场接我。")
        XCTAssertTrue(prompts.last?.hasSuffix("Corrected:") == true)
        prompts.removeAll()
        let plain = try await LocalAPIComposePrompt.compose(
            rawTranscription: "I'll be there at five.", options: options(.translate, targetLanguage: "zh-Hans")
        ) { prompt in
            prompts.append(prompt)
            return "我五点到。"
        }
        XCTAssertEqual(plain, "我五点到。")
        XCTAssertFalse(prompts.last?.contains("Corrected:") == true)
        XCTAssertNil(LocalAPIComposePrompt.translationAfterCorrection("我五点到。"))
        XCTAssertEqual(LocalAPIComposePrompt.translationAfterCorrection("Corrected: Meet at four.\n\nTranslation:\n四点见。"), "四点见。")
    }

    /// Qwen3.5-4B kept 三点 for "three, sorry, four" even with the two-step prompt.
    func testNumberAndDayCorrectionsAreAppliedBeforeTranslating() {
        let cases: [(String, String)] = [
            ("Call me at three, sorry, four o'clock.", "Call me at four o'clock."),
            ("Book a table for six, I mean, eight people.", "Book a table for eight people."),
            ("Mom, your checkup is on Thursday... actually, Friday. I'll drive you.", "Mom, your checkup is on Friday. I'll drive you."),
            ("Meet at 3, no, 4 pm.", "Meet at 4 pm."),
            ("明天上午九点，不对，十点开会。", "明天上午十点开会。"),
            ("周一，不对，周二给小陈打电话。", "周二给小陈打电话。"),
        ]
        for (input, expected) in cases {
            XCTAssertEqual(LocalAPIComposePrompt.appliedSelfCorrections(input), expected, input)
        }
        for unchanged in ["Actually, I like the blue one better.", "Call me at three, sorry, I'm running late.",
                          "Pick me up at the station... no, at the airport.", "这个答案不对。"] {
            XCTAssertEqual(LocalAPIComposePrompt.appliedSelfCorrections(unchanged), unchanged, unchanged)
        }
    }

    /// Qwen3.5-4B turned "Book a table for eight people" into 预订一个能容纳八人的包间 (a private room nobody asked for).
    func testTranslatePromptForbidsAddedDetails() throws {
        let prompt = try XCTUnwrap(LocalAPIComposePrompt.instruction(for: options(.translate, targetLanguage: "zh-Hans"), rawTranscription: "Book a table for eight people."))
        XCTAssertTrue(prompt.contains("Translate only what was said: add no places, rooms, times, reasons or other details"))
    }

    func testCorrectedTranslationIsReturnedFromTheFirstCall() async throws {
        var calls = 0
        let result = try await LocalAPIComposePrompt.compose(
            rawTranscription: "Call me at three, sorry, four o'clock.", options: options(.translate, targetLanguage: "zh-Hans")
        ) { _ in calls += 1; return "四点给我打电话。" }
        XCTAssertEqual(result, "四点给我打电话。")
        XCTAssertEqual(calls, 1)
    }

    func testTranslatePromptCarriesTheCorrectedTranscript() throws {
        let prompt = try XCTUnwrap(LocalAPIComposePrompt.instruction(
            for: options(.translate, targetLanguage: "zh-Hans"), rawTranscription: "Call me at three, sorry, four o'clock."))
        XCTAssertTrue(prompt.contains("Original speech (reference only; do not translate it): {{RAW_TRANSCRIPTION}}"))
        XCTAssertTrue(prompt.contains("Translate this corrected transcript: Call me at four o'clock."))
        XCTAssertTrue(prompt.hasSuffix("Translation:"))
        XCTAssertFalse(prompt.hasSuffix("Corrected:"))
    }

    func testCorrectionCues() {
        for text in ["Thursday... actually, Friday.", "three, sorry, four", "six, I mean, eight", "the station... no, the airport", "周一，不对，周二", "九点，其实是十点"] {
            XCTAssertTrue(LocalAPIComposePrompt.containsCorrectionCue(text), text)
        }
        for text in ["I'll be there at five.", "Say no to Friday's call.", "我五点到。", "Nobody knows."] {
            XCTAssertFalse(LocalAPIComposePrompt.containsCorrectionCue(text), text)
        }
    }

    func testToneEditThatDropsADateOrTimeRetriesNamingItThenRefuses() async throws {
        let selection = "You never listen to me. I'm done talking about this tonight."
        var prompts: [String] = []
        let kept = try await LocalAPIComposePrompt.compose(
            rawTranscription: "Make this calmer.", options: options(.edit, selectedText: selection)
        ) { prompt in
            prompts.append(prompt)
            return prompts.count == 1 ? "Let's talk about this tomorrow." : "I'd rather not keep talking about this tonight."
        }
        XCTAssertEqual(kept, "I'd rather not keep talking about this tonight.")
        XCTAssertEqual(prompts.count, 2)
        XCTAssertTrue(prompts[1].contains("Keep \"tonight\""))
        var calls = 0
        do {
            _ = try await LocalAPIComposePrompt.compose(
                rawTranscription: "Make this calmer.", options: options(.edit, selectedText: selection)
            ) { _ in calls += 1; return "Let's talk about this tomorrow." }
            XCTFail("an edit that silently changes a time must not be inserted")
        } catch { XCTAssertTrue(error is LocalAPIComposePrompt.EditNotApplied) }
        XCTAssertEqual(calls, 2)
        calls = 0
        let moved = try await LocalAPIComposePrompt.compose(
            rawTranscription: "Change tonight to tomorrow.", options: options(.edit, selectedText: selection)
        ) { _ in calls += 1; return "You never listen to me. I'm done talking about this tomorrow." }
        XCTAssertEqual(moved, "You never listen to me. I'm done talking about this tomorrow.")
        XCTAssertEqual(calls, 1)
        XCTAssertEqual(LocalAPIComposePrompt.missingDateOrTimeTerms("说得客气一点", selection: "周五之前把合同发给我。", output: "请在周五前把合同发给我。"), [])
        XCTAssertEqual(LocalAPIComposePrompt.missingDateOrTimeTerms("说得客气一点", selection: "周五下午3点之前发给我。", output: "请尽快发给我。"), ["周五", "下午", "3点"])
    }

    // MARK: Cursor context leakage

    private let notesLine = "好，先记一句：明天交报告。"
    private let notesSpeech = "嗯，我想一下下周的安排。这个事情，我觉得是这样的：就是周一先把设计稿定下来，对吧？然后周二周三写代码，周四给测试组看一下。最主要的是周五之前要能发一个可以用的版本，不然下周的评审就来不及了。"

    func testDictateThatRepeatsTextBeforeTheCursorRetriesOnceThenReturnsTheTranscript() async throws {
        // The output began with the previous insertion, which was the text before the cursor.
        let leaked = notesLine + "\n\n" + notesSpeech
        var prompts: [String] = []
        let fixed = try await LocalAPIComposePrompt.compose(
            rawTranscription: notesSpeech, options: options(.dictate, contextBefore: "\n" + notesLine)
        ) { prompt in
            prompts.append(prompt)
            return prompts.count == 1 ? leaked : notesSpeech
        }
        XCTAssertEqual(fixed, notesSpeech)
        XCTAssertEqual(prompts.count, 2)
        XCTAssertTrue(prompts.last?.contains("第一次输出重复了输入框里已有的文字") == true)
        var calls = 0
        let raw = try await LocalAPIComposePrompt.compose(
            rawTranscription: notesSpeech, options: options(.dictate, contextBefore: "\n" + notesLine)
        ) { _ in calls += 1; return leaked }
        XCTAssertEqual(raw, notesSpeech, "a second leak falls back to the transcript instead of duplicating the field")
        XCTAssertEqual(calls, 2)
        // The last sentence of the earlier insertion came back before the new speech.
        calls = 0
        let earlier = "呃，说实话，我还没想好。我觉得这个，先这样吧。"
        let speech = "行，我再补充一点，好，那再补充一点，就就这些可以吗？先这样。"
        _ = try await LocalAPIComposePrompt.compose(rawTranscription: speech, options: options(.dictate, contextBefore: earlier)) { _ in
            calls += 1
            return calls == 1 ? "我觉得这个，先这样吧。\n行，我再补充一点，好，那再补充一点，就就这些可以吗？先这样。" : "行，我再补充一点，就这些可以吗？先这样。"
        }
        XCTAssertEqual(calls, 2)
    }

    func testDictateKeepsASentenceTheSpeakerReallyRepeated() async throws {
        var calls = 0
        let result = try await LocalAPIComposePrompt.compose(
            rawTranscription: "好，先记一句：明天交报告。再说一遍。", options: options(.dictate, contextBefore: notesLine)
        ) { _ in calls += 1; return "好，先记一句：明天交报告。再说一遍。" }
        XCTAssertEqual(result, "好，先记一句：明天交报告。再说一遍。")
        XCTAssertEqual(calls, 1)
    }

    func testEditThatRepeatsTheSelectionOrCopiesSurroundingTextIsNotApplied() async throws {
        // "Edit latest" over that insertion returned the note about 2.5 times.
        let selection = notesLine + "\n\n" + notesSpeech
        let repeated = selection + "\n\n" + selection + "\n\n" + notesLine + "\n\n嗯，我想一下下周的安排。这个事情，我觉得是这样的："
        XCTAssertTrue(LocalAPIComposePrompt.leaksContext(repeated, rawTranscription: "简短一点。", options: options(.edit, selectedText: selection, contextBefore: "\n" + notesLine)))
        var calls = 0
        do {
            _ = try await LocalAPIComposePrompt.compose(
                rawTranscription: "简短一点。", options: options(.edit, selectedText: selection, contextBefore: "\n" + notesLine)
            ) { _ in calls += 1; return repeated }
            XCTFail("a repeated note must not replace the selection")
        } catch { XCTAssertTrue(error is LocalAPIComposePrompt.EditNotApplied) }
        XCTAssertEqual(calls, 2)
        calls = 0
        var prompts: [String] = []
        let english = try await LocalAPIComposePrompt.compose(
            rawTranscription: "Make this more polite.",
            options: options(.edit, selectedText: "Send me the slides today.", contextBefore: "Hi Sam. I hope your trip went well. ")
        ) { prompt in
            prompts.append(prompt)
            // Not the exact text before the selection, so it is not unwrapped: the leak retry runs.
            return prompts.count == 1 ? "Hi Sam! I hope your trip went well. Could you please send me the slides today?" : "Could you please send me the slides today?"
        }
        XCTAssertEqual(english, "Could you please send me the slides today?")
        XCTAssertTrue(prompts.last?.contains("Output only the edited selected text, once") == true)
    }

    func testCursorContextIsMarkedAsAlreadyInTheField() throws {
        let prompt = try XCTUnwrap(LocalAPIComposePrompt.instruction(
            for: options(.edit, selectedText: "明天交报告。", contextBefore: notesLine), rawTranscription: "简短一点。"
        ))
        XCTAssertTrue(prompt.contains("Text before the cursor (already in the field; do not repeat it):\n" + notesLine))
    }

    /// With cursor context, Qwen3.5-4B copied it into 4 of 6 dictations (retried, up to 15 s);
    /// without it, 6 of 6 passed in one call.
    func testPlainDictateSendsNoCursorContextToTheModel() throws {
        for raw in [notesSpeech, "Um, so I'll send the deck by Friday."] {
            let prompt = try XCTUnwrap(LocalAPIComposePrompt.instruction(
                for: options(.dictate, contextBefore: "\n" + notesLine, contextAfter: "Best regards"), rawTranscription: raw
            ))
            XCTAssertFalse(prompt.contains(notesLine))
            XCTAssertFalse(prompt.contains("Best regards"))
            XCTAssertFalse(prompt.contains("Text before the cursor"))
        }
    }

    func testRequestedTimeReplacementIsNotTreatedAsADroppedTime() {
        XCTAssertEqual(LocalAPIComposePrompt.missingDateOrTimeTerms("把三点改成四点。", selection: "我们明天下午三点在会议室开会。", output: "我们明天下午四点在会议室开会。"), [])
        XCTAssertEqual(LocalAPIComposePrompt.missingDateOrTimeTerms("简短一点。", selection: "明天下午三点开会。", output: "开会。"), ["明天", "下午", "三点"])
        XCTAssertEqual(LocalAPIComposePrompt.missingDateOrTimeTerms("Change the time to four.", selection: "See you at 3 pm.", output: "See you at 4 pm."), [])
    }

    func testWritePromptForbidsPlaceholders() throws {
        let english = try XCTUnwrap(LocalAPIComposePrompt.instruction(
            for: options(.write), rawTranscription: "Write to my manager that I'm leaving."
        ))
        XCTAssertTrue(english.contains("Never write placeholders such as [Name]"))
        let chinese = try XCTUnwrap(LocalAPIComposePrompt.instruction(
            for: options(.write), rawTranscription: "告诉经理我要离职。"
        ))
        XCTAssertTrue(chinese.contains("不要写[姓名]这类占位符"))
    }

    func testPlaceholderDetection() {
        for text in ["Hi [Manager's Name], I'm leaving.", "Best, {Your Name}", "Dear <Recipient>,", "您好【姓名】，", "您好［经理姓名］，", "Hi {{name}},"] {
            XCTAssertTrue(LocalAPIComposePrompt.containsPlaceholder(text, allowedIn: ""), text)
        }
        for text in ["Hi Dana, I'm leaving.", "Use the [x] box.", "Check [ ] items", "2 < 3 and 4 > 1", "See [1]."] {
            XCTAssertFalse(LocalAPIComposePrompt.containsPlaceholder(text, allowedIn: ""), text)
        }
        XCTAssertFalse(
            LocalAPIComposePrompt.containsPlaceholder("Reply with [Project Name] in the subject.", allowedIn: "Tell them to put [Project Name] in the subject."),
            "brackets the user asked for are content"
        )
    }

    func testWriteRetriesOncePlaceholderOutputThenRefuses() async throws {
        var prompts: [String] = []
        let fixed = try await LocalAPIComposePrompt.compose(
            rawTranscription: "Write to my manager that I'm leaving.", options: options(.write)
        ) { prompt in
            prompts.append(prompt)
            return prompts.count == 1 ? "Hi [Manager's Name], I'm leaving." : "Hi, I wanted to let you know I'm leaving."
        }
        XCTAssertEqual(fixed, "Hi, I wanted to let you know I'm leaving.")
        XCTAssertEqual(prompts.count, 2)
        XCTAssertTrue(prompts.last?.contains("The first attempt contained a placeholder") == true)
        var calls = 0
        do {
            _ = try await LocalAPIComposePrompt.compose(
                rawTranscription: "Write to my manager that I'm leaving.", options: options(.write)
            ) { _ in calls += 1; return "Hi [Manager's Name], I'm leaving." }
            XCTFail("a placeholder must never reach the text field")
        } catch {
            XCTAssertFalse(error is LocalAPIComposePrompt.EditNotApplied)
        }
        XCTAssertEqual(calls, 2)
        calls = 0
        let named = try await LocalAPIComposePrompt.compose(
            rawTranscription: "Write to Dana, my manager, that I'm leaving.", options: options(.write)
        ) { _ in calls += 1; return "Hi Dana, I'm leaving." }
        XCTAssertEqual(named, "Hi Dana, I'm leaving.")
        XCTAssertEqual(calls, 1)
    }

    func testStyleAdjustsToneLine() {
        for style in [LocalAPIComposeStyle.formal, .casual, .concise] {
            let prompt = LocalAPIComposePrompt.instruction(for: options(.write, style: style))!
            XCTAssertTrue(prompt.contains(style.rawValue), "style \(style.rawValue) appears in the prompt")
        }
        let automatic = LocalAPIComposePrompt.instruction(for: options(.write))!
        XCTAssertFalse(automatic.contains("formal"))
        XCTAssertFalse(automatic.contains("casual"))
        XCTAssertFalse(automatic.contains("concise"))
    }

    func testVocabularyIsListedWhenPresent() {
        let withTerms = LocalAPIComposePrompt.instruction(
            for: options(.write, vocabulary: ["Voxt", "Tailscale"])
        )!
        XCTAssertTrue(withTerms.contains("Voxt"))
        XCTAssertTrue(withTerms.contains("Tailscale"))

        let withoutTerms = LocalAPIComposePrompt.instruction(for: options(.write))!
        XCTAssertFalse(withoutTerms.contains("Prefer these spellings"))
    }

    func testEditPromptLeavesSelectionVerbatim() {
        let selection = "line1\n\nline2"
        let prompt = LocalAPIComposePrompt.instruction(for: options(.edit, selectedText: selection))!
        XCTAssertTrue(prompt.contains(selection))
    }

    // MARK: Language variants

    func testVariantsAddRegionalPreferenceWithoutTranslating() {
        let prompt = LocalAPIComposePrompt.instruction(for: options(.dictate, languageVariants: ["en-US", "zh-Hans"]))!
        XCTAssertTrue(prompt.contains("en-US"))
        XCTAssertTrue(prompt.contains("zh-Hans"))
        XCTAssertTrue(prompt.contains("do not translate"), "variants must never change the output language")
        XCTAssertTrue(prompt.contains("only when the output"), "spelling/script applies conditionally")
    }

    func testVariantsAreOmittedWhenEmpty() {
        let prompt = LocalAPIComposePrompt.instruction(for: options(.dictate))!
        XCTAssertFalse(prompt.contains("Preferred language variants"))
    }

    func testVariantsDemandConsistentRegionalSpelling() {
        let prompt = LocalAPIComposePrompt.instruction(for: options(.dictate, languageVariants: ["en-GB"]))!
        XCTAssertTrue(prompt.contains("English (United Kingdom)"), "variants carry human locale names")
        XCTAssertTrue(prompt.contains("consistently"), "spelling must be applied consistently")
        XCTAssertTrue(prompt.lowercased().contains("every word"), "consistency spans the whole output")
    }

    func testAutomaticModesIncludeStyleExamplesAfterTaskAndVocabulary() {
        let examples = [wordingExample]
        for mode in [LocalAPIComposeMode.dictate, .edit, .write] {
            let prompt = LocalAPIComposePrompt.instruction(for: options(
                mode,
                selectedText: mode == .edit ? "the sentence" : "",
                style: .automatic,
                vocabulary: ["grey"],
                styleExamples: examples
            ))!
            XCTAssertTrue(prompt.contains("Personal wording examples"), "\(mode) keeps a distinct example section")
            XCTAssertTrue(prompt.contains(wordingExample.before))
            XCTAssertTrue(prompt.contains(wordingExample.after))
            XCTAssertTrue(prompt.contains("Do not change facts, names, numbers, or the output language."))
            XCTAssertTrue(prompt.contains("Prefer these spellings: grey."))
            let task = prompt.range(of: mode == .dictate ? "Keep their meaning" : mode == .edit ? "Keep the selected text's language" : "Write the actual message requested")!
            let spellings = prompt.range(of: "Prefer these spellings")!
            let section = prompt.range(of: "Personal wording examples")!
            XCTAssertLessThan(task.lowerBound, section.lowerBound)
            XCTAssertLessThan(spellings.lowerBound, section.lowerBound)
        }
    }

    func testTranslateAndExplicitToneOmitStyleExamples() {
        let examples = [wordingExample]
        let translate = LocalAPIComposePrompt.instruction(for: options(
            .translate,
            targetLanguage: "en",
            style: .automatic,
            vocabulary: ["grey"],
            styleExamples: examples
        ))!
        XCTAssertFalse(translate.contains("Personal wording examples"))
        XCTAssertFalse(translate.contains("Cheers"))
        XCTAssertTrue(translate.contains("into en"))
        XCTAssertTrue(translate.contains("Prefer these spellings: grey."))

        for style in [LocalAPIComposeStyle.formal, .casual, .concise] {
            let prompt = LocalAPIComposePrompt.instruction(for: options(
                .dictate,
                style: style,
                vocabulary: ["grey"],
                styleExamples: examples
            ))!
            XCTAssertFalse(prompt.contains("Personal wording examples"), "\(style) omits stored examples")
            XCTAssertFalse(prompt.contains("Cheers"))
            XCTAssertTrue(prompt.contains("Tone: \(style.rawValue)."))
            XCTAssertTrue(prompt.contains("Prefer these spellings: grey."))
            XCTAssertTrue(prompt.contains("in the language or mix of languages they spoke"))
        }
    }

    func testEmptyStyleExamplesAddNoSection() {
        let prompt = LocalAPIComposePrompt.instruction(for: options(.dictate))!
        XCTAssertFalse(prompt.contains("Personal wording examples"))
    }

    func testExplicitTranslateTargetWinsOverVariants() {
        let prompt = LocalAPIComposePrompt.instruction(
            for: options(.translate, targetLanguage: "zh-Hans", languageVariants: ["en-US"])
        )!
        XCTAssertTrue(prompt.contains("into zh-Hans"), "the translate target stays authoritative")
        XCTAssertFalse(prompt.contains("into en-US"), "a variant must not hijack the target")
        XCTAssertFalse(prompt.contains("Preferred language variants"), "translation must not receive competing output languages")
    }

    // MARK: Model availability

    func testDisabledEnhancementMakesComposeUnavailable() {
        XCTAssertNotNil(
            LocalAPIComposePrompt.unavailabilityReason(for: .disabled),
            "compose must fail closed even when enhancement is switched off"
        )
    }

    func testUnreadyModelsAreUnavailable() {
        XCTAssertNotNil(LocalAPIComposePrompt.unavailabilityReason(for: .customLLM(ready: false)))
        XCTAssertNotNil(LocalAPIComposePrompt.unavailabilityReason(for: .remoteLLM(configured: false)))
    }

    func testReadyModelsAreAvailable() {
        XCTAssertNil(LocalAPIComposePrompt.unavailabilityReason(for: .customLLM(ready: true)))
        XCTAssertNil(LocalAPIComposePrompt.unavailabilityReason(for: .remoteLLM(configured: true)))
        XCTAssertNil(LocalAPIComposePrompt.unavailabilityReason(for: .appleIntelligence))
    }

    // MARK: No Mac-side prompt leakage

    private func resolverInput(
        globalPrompt: String,
        appendsLanguagePreservationRules: Bool
    ) -> EnhancementPromptResolver.Input {
        EnhancementPromptResolver.Input(
            globalPrompt: globalPrompt,
            rawTranscription: "speech",
            userMainLanguagePromptValue: "English",
            userOtherLanguagesPromptValue: "",
            dictionaryGlossary: nil,
            appEnhancementEnabled: false,
            appendsLanguagePreservationRules: appendsLanguagePreservationRules,
            groups: [],
            urlsByID: [:],
            frontmostBundleID: nil,
            focusedAppName: nil,
            normalizedActiveURL: nil,
            supportedBrowserBundleIDs: []
        )
    }

    func testComposeOverrideSuppressesMacLanguageRules() {
        let compose = EnhancementPromptResolver.resolve(
            resolverInput(globalPrompt: "Translate the speech.", appendsLanguagePreservationRules: false)
        )
        XCTAssertFalse(compose.content.contains("Language rule"), "Mac language preferences must not leak into compose prompts")

        let dictation = EnhancementPromptResolver.resolve(
            resolverInput(globalPrompt: "Clean this up.", appendsLanguagePreservationRules: true)
        )
        XCTAssertTrue(dictation.content.contains("Language rule"), "the standard dictation prompt keeps its rules")
    }

    // MARK: Dictate principles

    // On Qwen3.5-4B, the rule-based prompt passed 14/24 and 16/24 of two test
    // sets (fillers, self-corrections, casual, rambling, lists, terms in zh, en, ja, mixed); these principles 19/24 and 21/24.
    func testDictatePromptFollowsTheTranscriptsScript() throws {
        let japanese = try XCTUnwrap(LocalAPIComposePrompt.instruction(for: options(.dictate), rawTranscription: "えーと、明日の会議は三時からです。"))
        XCTAssertTrue(japanese.contains("話し手が自分で書いたらこうなる"), "kana gets the Japanese prompt")
        XCTAssertTrue(japanese.contains("書き起こし：{{RAW_TRANSCRIPTION}}"))
        XCTAssertTrue(japanese.hasSuffix("出力："))
        // 9 Han and 15 Latin letters: under the old 50% rule this went to the English prompt and came back translated.
        let mixed = try XCTUnwrap(LocalAPIComposePrompt.instruction(for: options(.dictate), rawTranscription: "我觉得这个 project 的 timeline 有点紧"))
        XCTAssertTrue(mixed.contains("说话人自己会写出来的文字"), "Chinese with English words gets the Chinese prompt")
        XCTAssertTrue(mixed.contains("口述转写：{{RAW_TRANSCRIPTION}}"))
        XCTAssertTrue(mixed.hasSuffix("输出："))
        let english = try XCTUnwrap(LocalAPIComposePrompt.instruction(for: options(.dictate), rawTranscription: "Move the review to Thursday."))
        XCTAssertTrue(english.contains("the text this speaker would have written themselves"))
        XCTAssertTrue(english.hasSuffix("Output:"))
    }

    // Qwen3.5-4B wrote "会議は 4 時から" and "10 分" even when asked not to.
    func testDictateDropsSpacesBetweenDigitsAndHanOrKana() async throws {
        let japanese = try await LocalAPIComposePrompt.compose(rawTranscription: "会議は四時から", options: options(.dictate)) { _ in
            "会議は 4 時から、10 分くらいで着く。"
        }
        XCTAssertEqual(japanese, "会議は4時から、10分くらいで着く。")
        let english = try await LocalAPIComposePrompt.compose(rawTranscription: "Meet at 4 pm", options: options(.dictate)) { _ in
            "Meet at 4 pm in room 2."
        }
        XCTAssertEqual(english, "Meet at 4 pm in room 2.")
    }

    // The recognizer spells some acronyms letter by letter inside Chinese ("放到 C I C D 里面").
    func testSpelledOutAcronymsNextToChineseOrJapaneseAreJoined() {
        XCTAssertEqual(LocalAPIComposePrompt.joinedSpelledAcronyms("整个东西应该放到 C I C D 里面，每次都要跑。"), "整个东西应该放到 CI/CD 里面，每次都要跑。")
        XCTAssertEqual(LocalAPIComposePrompt.joinedSpelledAcronyms("这个 A P I 太慢了"), "这个 API 太慢了")
        XCTAssertEqual(LocalAPIComposePrompt.joinedSpelledAcronyms("用 A I 写的"), "用 AI 写的")
        XCTAssertEqual(LocalAPIComposePrompt.joinedSpelledAcronyms("APIの I O が遅い"), "APIの I/O が遅い")
        for unchanged in ["Plan A I think works.", "I went to A B C Mart.", "这个 API 太慢了", "用 Q 键"] {
            XCTAssertEqual(LocalAPIComposePrompt.joinedSpelledAcronyms(unchanged), unchanged)
        }
    }

    func testDictateJoinsSpelledOutAcronyms() async throws {
        let result = try await LocalAPIComposePrompt.compose(rawTranscription: "放到 C I C D 里面", options: options(.dictate)) { _ in
            "整个放进 C I C D 里，每次强制跑。"
        }
        XCTAssertEqual(result, "整个放进 CI/CD 里，每次强制跑。")
    }

    // The dictate prompt states the intent and lets the model judge structure: a rule list made the 4B model add
    // unasked bullets and stiffen the register. Compared blind on 42 dictations, this
    // prompt read more like the speaker 22 times, the rule list 7 times, with 13 ties.
    func testDictatePromptIsAShortStatementOfIntent() throws {
        for (speech, intent) in [("Um, see you later.", "Keep their meaning, their voice and their register"),
                                 ("嗯，明天见。", "保留他的意思、口吻和语体"),
                                 ("えーと、また明日。", "意味、話し方の雰囲気、丁寧さの度合いはそのままにし")] {
            let prompt = try XCTUnwrap(LocalAPIComposePrompt.instruction(for: options(.dictate), rawTranscription: speech))
            XCTAssertTrue(prompt.contains(intent), intent)
            let task = prompt.components(separatedBy: "\n\n")[0]
            XCTAssertLessThanOrEqual(task.components(separatedBy: "\n").count, 5, "a few lines, not a rule list")
            for rule in ["Format example", "格式示例", "書式の例", "1. ", "- "] { XCTAssertFalse(task.contains(rule), rule) }
        }
    }
}
