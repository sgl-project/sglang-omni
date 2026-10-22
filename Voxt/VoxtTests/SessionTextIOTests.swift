// SessionTextIOTests.swift
// Provides Session Text IOTests for Voxt test coverage.

import XCTest
@testable import Voxt

@MainActor
final class SessionTextIOTests: XCTestCase {
    func testObsidianBypassesSelectedTextFlowWithoutAffectingOtherApps() {
        XCTAssertTrue(
            SelectedTextSystemSelectionSupport.ignoresSelectedTextFlow(bundleID: "md.obsidian")
        )
        XCTAssertFalse(
            SelectedTextSystemSelectionSupport.ignoresSelectedTextFlow(bundleID: "com.todesktop.230313mzl4w4u92")
        )
        XCTAssertFalse(
            SelectedTextSystemSelectionSupport.ignoresSelectedTextFlow(bundleID: "com.apple.TextEdit")
        )
    }

    func testSelectedTextDictionaryHotkeyAcceptsUpToFiveWords() {
        let candidate = "one two three four five"
        XCTAssertEqual(SelectedTextDictionaryHotkeySupport.candidateTerm(from: candidate), candidate)
    }

    func testSelectedTextDictionaryHotkeyRejectsMoreThanFiveWords() {
        XCTAssertNil(SelectedTextDictionaryHotkeySupport.candidateTerm(from: "one two three four five six"))
    }

    func testSelectedTextDictionaryHotkeyAcceptsUpToTenCJKCharacters() {
        let chinese = String(repeating: "词", count: 10)
        let japanese = String(repeating: "あ", count: 10)
        let korean = String(repeating: "한", count: 10)
        XCTAssertEqual(SelectedTextDictionaryHotkeySupport.candidateTerm(from: chinese), chinese)
        XCTAssertEqual(SelectedTextDictionaryHotkeySupport.candidateTerm(from: japanese), japanese)
        XCTAssertEqual(SelectedTextDictionaryHotkeySupport.candidateTerm(from: korean), korean)
    }

    func testSelectedTextDictionaryHotkeyRejectsMoreThanTenCJKCharacters() {
        XCTAssertNil(SelectedTextDictionaryHotkeySupport.candidateTerm(from: String(repeating: "词", count: 11)))
        XCTAssertNil(SelectedTextDictionaryHotkeySupport.candidateTerm(from: String(repeating: "あ", count: 11)))
        XCTAssertNil(SelectedTextDictionaryHotkeySupport.candidateTerm(from: String(repeating: "한", count: 11)))
    }

    func testSelectedTextDictionaryHotkeyTrimsWhitespace() {
        XCTAssertEqual(
            SelectedTextDictionaryHotkeySupport.candidateTerm(from: "  OpenAI\n"),
            "OpenAI"
        )
    }

    func testSelectedTextDictionaryHotkeyRejectsEmptySelection() {
        XCTAssertNil(SelectedTextDictionaryHotkeySupport.candidateTerm(from: " \n "))
        XCTAssertNil(SelectedTextDictionaryHotkeySupport.candidateTerm(from: nil))
    }

    func testSelectedTextDictionaryHotkeyRejectsPunctuationOnlySelection() {
        XCTAssertNil(SelectedTextDictionaryHotkeySupport.candidateTerm(from: "\"\""))
        XCTAssertNil(SelectedTextDictionaryHotkeySupport.candidateTerm(from: "——"))
        XCTAssertNil(SelectedTextDictionaryHotkeySupport.candidateTerm(from: "「」"))
        XCTAssertNil(SelectedTextDictionaryHotkeySupport.candidateTerm(from: ".."))
    }

    func testSharedContentSelectionRejectsPunctuationOnlyLikeDictionary() {
        // Same false-positive class that dictionary already rejected, but notes used to accept.
        for payload in ["\"\"", "——", "「」", "..", "!!"] {
            XCTAssertNil(
                SystemSelectionTextSupport.contentSelection(from: payload),
                "Expected content selection to reject punctuation-only payload \(payload)"
            )
            XCTAssertNil(SystemSelectionTextSupport.dictionaryCandidateTerm(from: payload))
        }

        XCTAssertEqual(SystemSelectionTextSupport.contentSelection(from: "  OpenAI\n"), "OpenAI")
        XCTAssertEqual(
            SystemSelectionTextSupport.contentSelection(from: String(repeating: "词", count: 20)),
            String(repeating: "词", count: 20)
        )
        XCTAssertNil(
            SystemSelectionTextSupport.dictionaryCandidateTerm(
                from: String(repeating: "词", count: 20)
            )
        )
    }

    func testSelectedTextSystemSelectionRejectsCaretOnlyRange() {
        XCTAssertFalse(SelectedTextSystemSelectionSupport.hasNonEmptySelectedTextRange(length: 0))
    }

    func testSelectedTextSystemSelectionAcceptsNonEmptyRange() {
        XCTAssertTrue(SelectedTextSystemSelectionSupport.hasNonEmptySelectedTextRange(length: 1))
        XCTAssertTrue(SelectedTextSystemSelectionSupport.hasNonEmptySelectedTextRange(length: 12))
    }

    func testSimulatedCopyPolicyByAXEvidenceClass() {
        // Caret-only range: deny.
        XCTAssertFalse(
            SelectedTextSystemSelectionSupport.shouldAttemptSimulatedCopy(
                focusedElementAvailable: true,
                selectedTextRange: CFRange(location: 12, length: 0),
                isBrowser: false,
                copiesLineOnEmptySelection: false
            )
        )

        // Non-empty range: allow recovery copy.
        XCTAssertTrue(
            SelectedTextSystemSelectionSupport.shouldAttemptSimulatedCopy(
                focusedElementAvailable: true,
                selectedTextRange: CFRange(location: 0, length: 4),
                isBrowser: false,
                copiesLineOnEmptySelection: false
            )
        )

        // AX focus dead, but app does not invent a line copy (e.g. WeChat).
        XCTAssertTrue(
            SelectedTextSystemSelectionSupport.shouldAttemptSimulatedCopy(
                focusedElementAvailable: false,
                selectedTextRange: nil,
                isBrowser: false,
                copiesLineOnEmptySelection: false
            )
        )

        // AX focus dead + line-copy editor (Cursor / VS Code): deny.
        XCTAssertFalse(
            SelectedTextSystemSelectionSupport.shouldAttemptSimulatedCopy(
                focusedElementAvailable: false,
                selectedTextRange: nil,
                isBrowser: false,
                copiesLineOnEmptySelection: true
            )
        )

        // Focused editor without range attribute: deny.
        XCTAssertFalse(
            SelectedTextSystemSelectionSupport.shouldAttemptSimulatedCopy(
                focusedElementAvailable: true,
                selectedTextRange: nil,
                isBrowser: false,
                copiesLineOnEmptySelection: false
            )
        )

        // Browser focused without range attribute: deny (avoid residual page copy).
        XCTAssertFalse(
            SelectedTextSystemSelectionSupport.shouldAttemptSimulatedCopy(
                focusedElementAvailable: true,
                selectedTextRange: nil,
                isBrowser: true,
                copiesLineOnEmptySelection: false
            )
        )
    }

    func testAXBlackoutClipboardProbeOnlyWhenLineCopyEditorHasNoAXSurface() {
        XCTAssertTrue(
            SelectedTextSystemSelectionSupport.shouldAttemptAXBlackoutClipboardProbe(
                focusedElementAvailable: false,
                axWindowCandidatesAvailable: false,
                copiesLineOnEmptySelection: true
            )
        )
        // Still have window candidates: keep denying unfiltered Cmd+C.
        XCTAssertFalse(
            SelectedTextSystemSelectionSupport.shouldAttemptAXBlackoutClipboardProbe(
                focusedElementAvailable: false,
                axWindowCandidatesAvailable: true,
                copiesLineOnEmptySelection: true
            )
        )
        XCTAssertFalse(
            SelectedTextSystemSelectionSupport.shouldAttemptAXBlackoutClipboardProbe(
                focusedElementAvailable: false,
                axWindowCandidatesAvailable: false,
                copiesLineOnEmptySelection: false
            )
        )
        XCTAssertFalse(
            SelectedTextSystemSelectionSupport.shouldAttemptAXBlackoutClipboardProbe(
                focusedElementAvailable: true,
                axWindowCandidatesAvailable: false,
                copiesLineOnEmptySelection: true
            )
        )
        // Obsidian copies the current line when AX is completely unavailable;
        // without a selection signal, clipboard content is not distinguishable
        // from a real single-line selection.
        XCTAssertFalse(
            SelectedTextSystemSelectionSupport.shouldAttemptAXBlackoutClipboardProbe(
                focusedElementAvailable: false,
                axWindowCandidatesAvailable: false,
                copiesLineOnEmptySelection: true,
                supportsAXBlackoutSelectionRecovery: false
            )
        )
    }

    func testLooksLikeEmptySelectionLineCopyRecognizesClipboardCapabilityShape() {
        XCTAssertTrue(
            SelectedTextSystemSelectionSupport.looksLikeEmptySelectionLineCopy(
                rawClipboardText: "    const foo = 1;\n"
            )
        )
        XCTAssertTrue(
            SelectedTextSystemSelectionSupport.looksLikeEmptySelectionLineCopy(
                rawClipboardText: "single line\r\n"
            )
        )
        XCTAssertFalse(
            SelectedTextSystemSelectionSupport.looksLikeEmptySelectionLineCopy(
                rawClipboardText: "selectedWord"
            )
        )
        XCTAssertFalse(
            SelectedTextSystemSelectionSupport.looksLikeEmptySelectionLineCopy(
                rawClipboardText: "line one\nline two\n"
            )
        )
    }

    func testCopiesCurrentLineWhenSelectionEmptyRecognizesEditorFamilies() {
        let lineCopyEditors = [
            "com.microsoft.VSCode",
            "com.microsoft.VSCodeInsiders",
            "com.vscodium",
            "com.todesktop.230313mzl4w4u92", // Cursor
            "com.exafunction.windsurf",
            "com.google.antigravity",
            "com.jetbrains.intellij",
            "com.jetbrains.WebStorm",
            "com.sublimetext.4",
            "dev.zed.Zed",
            "com.trae.app",
            "cn.trae.app",
            "com.qoder.app",
            "md.obsidian", // Obsidian
            "org.example.MyVSCodeFork"
        ]
        for bundleID in lineCopyEditors {
            XCTAssertTrue(
                SelectedTextSystemSelectionSupport.copiesCurrentLineWhenSelectionEmpty(bundleID: bundleID),
                "Expected line-copy editor classification for \(bundleID)"
            )
        }

        let safeApps = [
            "com.tencent.xinWeChat",
            "com.apple.Safari",
            "com.google.Chrome",
            "com.apple.TextEdit",
            "com.apple.Notes",
            "com.tinyspeck.slackmacgap"
        ]
        for bundleID in safeApps {
            XCTAssertFalse(
                SelectedTextSystemSelectionSupport.copiesCurrentLineWhenSelectionEmpty(bundleID: bundleID),
                "Expected non-line-copy classification for \(bundleID)"
            )
        }
    }

    func testBrowserSelectionScriptsIncludeJavaScriptSelection() {
        let js = BrowserAutomationScriptBuilder.selectionJavaScript
        XCTAssertTrue(js.contains("selectionStart"))
        XCTAssertTrue(js.contains("selectionEnd"))
        XCTAssertTrue(js.contains("INPUT"))
        XCTAssertTrue(js.contains("TEXTAREA"))
        XCTAssertTrue(js.contains("isCollapsed"))
        XCTAssertTrue(js.contains("getSelection()"))
        XCTAssertTrue(js.contains(BrowserAutomationScriptBuilder.selectionSourceSeparator))
        // Must stay AppleScript-string safe (no raw double quotes inside the JS payload).
        XCTAssertFalse(js.contains("\""))

        let formOnly = BrowserAutomationScriptBuilder.formOrEditorSelectionJavaScript
        XCTAssertTrue(formOnly.contains("return 'P'+sep;"))
        XCTAssertFalse(formOnly.contains("return 'P'+sep+String(sel.toString()||'');"))

        let safariScripts = BrowserAutomationScriptBuilder.selectionScripts(
            bundleID: "com.apple.Safari",
            displayName: "Safari"
        )
        XCTAssertFalse(safariScripts.isEmpty)
        XCTAssertTrue(safariScripts.contains(where: { $0.contains("selectionStart") }))
        XCTAssertTrue(safariScripts.contains(where: { $0.contains("do JavaScript") }))

        let chromeScripts = BrowserAutomationScriptBuilder.selectionScripts(
            bundleID: "com.google.Chrome",
            displayName: "Google Chrome"
        )
        XCTAssertFalse(chromeScripts.isEmpty)
        XCTAssertTrue(chromeScripts.contains(where: { $0.contains("execute javascript") }))
        XCTAssertTrue(chromeScripts.contains(where: { $0.contains("isCollapsed") }))

        let axBlindScripts = BrowserAutomationScriptBuilder.selectionScripts(
            bundleID: "company.thebrowser.Browser",
            displayName: "Arc",
            allowPageSelection: false
        )
        XCTAssertTrue(axBlindScripts.contains(where: { $0.contains("return 'P'+sep;") }))
        XCTAssertFalse(axBlindScripts.contains(where: {
            $0.contains("return 'P'+sep+String(sel.toString()||'');")
        }))
    }

    func testBrowserSelectionScriptOutputParsingAndTextControlDistrust() {
        let separator = BrowserAutomationScriptBuilder.selectionSourceSeparator
        XCTAssertEqual(
            SelectedTextSystemSelectionSupport.parseBrowserSelectionScriptOutput("F\(separator)"),
            .init(source: .form, text: "")
        )
        XCTAssertEqual(
            SelectedTextSystemSelectionSupport.parseBrowserSelectionScriptOutput("P\(separator)ab"),
            .init(source: .page, text: "ab")
        )
        XCTAssertEqual(
            SelectedTextSystemSelectionSupport.parseBrowserSelectionScriptOutput("E\(separator)hello"),
            .init(source: .editor, text: "hello")
        )

        XCTAssertEqual(
            SelectedTextSystemSelectionSupport.browserSelectionProbeOutcome(
                output: "F\(separator)",
                hadExecutionError: false,
                axFocusedTextControl: true
            ),
            .confirmedEmpty
        )
        XCTAssertEqual(
            SelectedTextSystemSelectionSupport.browserSelectionProbeOutcome(
                output: "P\(separator)ab",
                hadExecutionError: false,
                axFocusedTextControl: true
            ),
            .confirmedEmpty
        )
        XCTAssertEqual(
            SelectedTextSystemSelectionSupport.browserSelectionProbeOutcome(
                output: "P\(separator)ab",
                hadExecutionError: false,
                axFocusedTextControl: false
            ),
            .selected("ab")
        )
        XCTAssertEqual(
            SelectedTextSystemSelectionSupport.browserSelectionProbeOutcome(
                output: "F\(separator)ab",
                hadExecutionError: false,
                axFocusedTextControl: true
            ),
            .selected("ab")
        )
        XCTAssertTrue(SelectedTextSystemSelectionSupport.isBrowserTextControlRole("AXTextField"))
        XCTAssertTrue(SelectedTextSystemSelectionSupport.isBrowserTextControlRole("AXTextArea"))
        XCTAssertFalse(SelectedTextSystemSelectionSupport.isBrowserTextControlRole("AXWebArea"))

        // Tagged page selections must work even when AX focus is unavailable (Arc).
        XCTAssertEqual(
            SelectedTextSystemSelectionSupport.browserSelectionProbeOutcome(
                output: "P\(separator)article",
                hadExecutionError: false,
                axFocusedTextControl: false,
                axFocusAvailable: false
            ),
            .selected("article")
        )
        // Untagged payloads stay rejected while AX is blind — ambiguous residual risk.
        XCTAssertEqual(
            SelectedTextSystemSelectionSupport.browserSelectionProbeOutcome(
                output: "some word",
                hadExecutionError: false,
                axFocusedTextControl: false,
                axFocusAvailable: false
            ),
            .confirmedEmpty
        )
        // Form/editor hits remain trusted even when AX focus is blind.
        XCTAssertEqual(
            SelectedTextSystemSelectionSupport.browserSelectionProbeOutcome(
                output: "F\(separator)term",
                hadExecutionError: false,
                axFocusedTextControl: false,
                axFocusAvailable: false
            ),
            .selected("term")
        )
        XCTAssertEqual(
            SelectedTextSystemSelectionSupport.browserSelectionProbeOutcome(
                output: "P\(separator)article",
                hadExecutionError: false,
                axFocusedTextControl: false,
                axFocusAvailable: true
            ),
            .selected("article")
        )

        let rejectedUntagged = SelectedTextSystemSelectionSupport.browserSelectionProbeDecision(
            output: "residual",
            hadExecutionError: false,
            axFocusedTextControl: false,
            axFocusAvailable: false
        )
        XCTAssertEqual(rejectedUntagged?.outcome, .confirmedEmpty)
        XCTAssertTrue(rejectedUntagged?.reason.contains("reject-untagged-while-ax-focus-unavailable") == true)

        let acceptedPage = SelectedTextSystemSelectionSupport.browserSelectionProbeDecision(
            output: "P\(separator)residual",
            hadExecutionError: false,
            axFocusedTextControl: false,
            axFocusAvailable: false
        )
        XCTAssertEqual(acceptedPage?.outcome, .selected("residual"))
        XCTAssertTrue(acceptedPage?.reason.contains("accept source=P") == true)

        let accepted = SelectedTextSystemSelectionSupport.browserSelectionProbeDecision(
            output: "F\(separator)selected",
            hadExecutionError: false,
            axFocusedTextControl: false,
            axFocusAvailable: false
        )
        XCTAssertEqual(accepted?.outcome, .selected("selected"))
        XCTAssertTrue(accepted?.reason.contains("accept source=F") == true)

        let summary = SelectedTextSystemSelectionSupport.browserSelectionRawDebugSummary(
            "F\(separator)hello"
        )
        XCTAssertTrue(summary.contains("hasSep=true"))
        XCTAssertTrue(summary.contains("source=F"))
        XCTAssertTrue(summary.contains("textChars=5"))

        // Production Arc/Chromium AppleScript wraps the JS string in literal quotes.
        // Exact shapes from user logs:
        //   rawChars=8  prefix="F::V::"
        //   rawChars=11 prefix="F::V::789"
        XCTAssertEqual(
            SelectedTextSystemSelectionSupport.parseBrowserSelectionScriptOutput("\"F\(separator)\""),
            .init(source: .form, text: "")
        )
        XCTAssertEqual(
            SelectedTextSystemSelectionSupport.parseBrowserSelectionScriptOutput("\"F\(separator)789\""),
            .init(source: .form, text: "789")
        )
        XCTAssertEqual(
            SelectedTextSystemSelectionSupport.browserSelectionProbeOutcome(
                output: "\"F\(separator)789\"",
                hadExecutionError: false,
                axFocusedTextControl: false,
                axFocusAvailable: false
            ),
            .selected("789")
        )
        XCTAssertEqual(
            SelectedTextSystemSelectionSupport.browserSelectionProbeOutcome(
                output: "\"F\(separator)\"",
                hadExecutionError: false,
                axFocusedTextControl: false,
                axFocusAvailable: false
            ),
            .confirmedEmpty
        )
        XCTAssertEqual(
            SelectedTextSystemSelectionSupport.browserSelectionProbeOutcome(
                output: "\"P\(separator)\"",
                hadExecutionError: false,
                axFocusedTextControl: false,
                axFocusAvailable: false
            ),
            .confirmedEmpty
        )
    }

    func testConfirmedCaretOnlyShortCircuitsFurtherSelectionProbes() {
        XCTAssertTrue(
            SelectedTextSystemSelectionSupport.isConfirmedCaretOnly(
                selectedTextRange: CFRange(location: 8, length: 0)
            )
        )
        XCTAssertFalse(
            SelectedTextSystemSelectionSupport.isConfirmedCaretOnly(
                selectedTextRange: CFRange(location: 0, length: 3)
            )
        )
        XCTAssertFalse(
            SelectedTextSystemSelectionSupport.isConfirmedCaretOnly(selectedTextRange: nil)
        )
    }

    func testDefinitiveEmptyBrowserSelectionStopsDialectRetries() {
        XCTAssertTrue(
            SelectedTextSystemSelectionSupport.isDefinitiveEmptyBrowserSelection(
                output: "",
                hadExecutionError: false
            )
        )
        XCTAssertTrue(
            SelectedTextSystemSelectionSupport.isDefinitiveEmptyBrowserSelection(
                output: "  \n\t",
                hadExecutionError: false
            )
        )
        XCTAssertFalse(
            SelectedTextSystemSelectionSupport.isDefinitiveEmptyBrowserSelection(
                output: "hello",
                hadExecutionError: false
            )
        )
        // Script-form failures should keep trying other dialects.
        XCTAssertFalse(
            SelectedTextSystemSelectionSupport.isDefinitiveEmptyBrowserSelection(
                output: nil,
                hadExecutionError: true
            )
        )
        XCTAssertFalse(
            SelectedTextSystemSelectionSupport.isDefinitiveEmptyBrowserSelection(
                output: "",
                hadExecutionError: true
            )
        )

        XCTAssertEqual(
            SelectedTextSystemSelectionSupport.browserSelectionProbeOutcome(
                output: "",
                hadExecutionError: false
            ),
            .confirmedEmpty
        )
        XCTAssertEqual(
            SelectedTextSystemSelectionSupport.browserSelectionProbeOutcome(
                output: "  \n",
                hadExecutionError: false
            ),
            .confirmedEmpty
        )
        XCTAssertEqual(
            SelectedTextSystemSelectionSupport.browserSelectionProbeOutcome(
                output: "hello",
                hadExecutionError: false
            ),
            .selected("hello")
        )
        XCTAssertNil(
            SelectedTextSystemSelectionSupport.browserSelectionProbeOutcome(
                output: nil,
                hadExecutionError: true
            )
        )
        XCTAssertNil(
            SelectedTextSystemSelectionSupport.browserSelectionProbeOutcome(
                output: "",
                hadExecutionError: true
            )
        )
        // Untagged page-like hits are rejected when AX already shows a text control focus.
        XCTAssertEqual(
            SelectedTextSystemSelectionSupport.browserSelectionProbeOutcome(
                output: "ab",
                hadExecutionError: false,
                axFocusedTextControl: true
            ),
            .confirmedEmpty
        )
    }

    func testSelectionProbeDenialDetailMatchesCapabilityClasses() {
        XCTAssertEqual(
            SelectedTextSystemSelectionSupport.denialDetail(
                allowCopy: true,
                focusedElementAvailable: false,
                selectedTextRange: nil,
                isBrowser: false,
                copiesLineOnEmptySelection: false
            ),
            "copy-missed"
        )
        XCTAssertEqual(
            SelectedTextSystemSelectionSupport.denialDetail(
                allowCopy: false,
                focusedElementAvailable: false,
                selectedTextRange: nil,
                isBrowser: false,
                copiesLineOnEmptySelection: true
            ),
            "ax-focus-unavailable-line-copy-editor"
        )
        XCTAssertEqual(
            SelectedTextSystemSelectionSupport.denialDetail(
                allowCopy: false,
                focusedElementAvailable: true,
                selectedTextRange: CFRange(location: 3, length: 0),
                isBrowser: false,
                copiesLineOnEmptySelection: false
            ),
            "confirmed-caret-only"
        )
        XCTAssertEqual(
            SelectedTextSystemSelectionSupport.denialDetail(
                allowCopy: false,
                focusedElementAvailable: true,
                selectedTextRange: nil,
                isBrowser: false,
                copiesLineOnEmptySelection: false
            ),
            "focused-without-range-non-browser"
        )
    }

    func testRewriteAlwaysPresentsAnswerOverlay() {
        XCTAssertTrue(
            AppDelegate.shouldPresentRewriteAnswerOverlay(
                sessionOutputMode: .rewrite,
                hasSelectedSourceText: false
            )
        )
        XCTAssertTrue(
            AppDelegate.shouldPresentRewriteAnswerOverlay(
                sessionOutputMode: .rewrite,
                hasSelectedSourceText: true
            )
        )
    }

    func testOnlyDirectAnswerRewriteUsesStructuredOutput() {
        XCTAssertTrue(
            AppDelegate.shouldUseStructuredRewriteAnswerOutput(
                sessionOutputMode: .rewrite,
                hasSelectedSourceText: false
            )
        )
        XCTAssertFalse(
            AppDelegate.shouldUseStructuredRewriteAnswerOutput(
                sessionOutputMode: .rewrite,
                hasSelectedSourceText: true
            )
        )
    }

    func testNonRewriteSessionsDoNotPresentRewriteAnswerOverlay() {
        XCTAssertFalse(
            AppDelegate.shouldPresentRewriteAnswerOverlay(
                sessionOutputMode: .transcription,
                hasSelectedSourceText: false
            )
        )
        XCTAssertFalse(
            AppDelegate.shouldUseStructuredRewriteAnswerOutput(
                sessionOutputMode: .transcription,
                hasSelectedSourceText: false
            )
        )

        XCTAssertFalse(
            AppDelegate.shouldPresentRewriteAnswerOverlay(
                sessionOutputMode: .translation,
                hasSelectedSourceText: false
            )
        )
        XCTAssertFalse(
            AppDelegate.shouldUseStructuredRewriteAnswerOutput(
                sessionOutputMode: .translation,
                hasSelectedSourceText: false
            )
        )
    }

    func testSelectedTextTranslationShowsAnswerOverlayOnlyWhenConfigured() {
        XCTAssertTrue(
            AppDelegate.shouldPresentSelectedTextTranslationAnswerOverlay(
                sessionOutputMode: .translation,
                isSelectedTextTranslationFlow: true,
                showResultWindow: true
            )
        )
        XCTAssertFalse(
            AppDelegate.shouldPresentSelectedTextTranslationAnswerOverlay(
                sessionOutputMode: .translation,
                isSelectedTextTranslationFlow: true,
                showResultWindow: false
            )
        )
        XCTAssertFalse(
            AppDelegate.shouldPresentSelectedTextTranslationAnswerOverlay(
                sessionOutputMode: .translation,
                isSelectedTextTranslationFlow: false,
                showResultWindow: true
            )
        )
        XCTAssertFalse(
            AppDelegate.shouldPresentSelectedTextTranslationAnswerOverlay(
                sessionOutputMode: .transcription,
                isSelectedTextTranslationFlow: true,
                showResultWindow: true
            )
        )
    }

    func testSelectedTextTranslationAutoInjectFollowsResultWindowToggle() {
        XCTAssertTrue(
            AppDelegate.shouldAutoInjectSelectedTextTranslationResult(
                sessionOutputMode: .translation,
                isSelectedTextTranslationFlow: true,
                showResultWindow: false
            )
        )
        XCTAssertFalse(
            AppDelegate.shouldAutoInjectSelectedTextTranslationResult(
                sessionOutputMode: .translation,
                isSelectedTextTranslationFlow: true,
                showResultWindow: true
            )
        )
        XCTAssertFalse(
            AppDelegate.shouldAutoInjectSelectedTextTranslationResult(
                sessionOutputMode: .translation,
                isSelectedTextTranslationFlow: false,
                showResultWindow: false
            )
        )
    }

    func testPreparedDeliveryContextAppliesDictionaryCorrectionsBeforeDelivery() {
        let matcher = DictionaryMatcher(
            entries: [TestFactories.makeEntry(term: "Anthropic", observedVariants: ["anthropic ai"])],
            blockedGlobalMatchKeys: []
        )

        let context = AppDelegate.preparedDeliveryContext(
            originalText: """
            {"title":"AI Answer","content":"anthropic ai"}
            """,
            llmDurationSeconds: 0.5,
            sessionOutputMode: .rewrite,
            userMainLanguage: .fallbackOption(),
            matcher: matcher,
            usesConservativeEvidence: false,
            automaticReplacementEnabled: true
        )

        XCTAssertEqual(context.outputText, "Anthropic")
        XCTAssertEqual(context.dictionaryCorrectedTerms, ["Anthropic"])
        XCTAssertEqual(context.rewriteAnswerPayload?.title, "AI Answer")
        XCTAssertEqual(context.rewriteAnswerPayload?.content, "Anthropic")
    }

    func testPreparedDeliveryContextKeepsOriginalTextForConservativeDictionaryEvidence() {
        let matcher = DictionaryMatcher(
            entries: [TestFactories.makeEntry(term: "Anthropic", observedVariants: ["anthropic ai"])],
            blockedGlobalMatchKeys: []
        )

        let context = AppDelegate.preparedDeliveryContext(
            originalText: "anthropic ai",
            llmDurationSeconds: nil,
            sessionOutputMode: .transcription,
            userMainLanguage: .fallbackOption(),
            matcher: matcher,
            usesConservativeEvidence: true,
            automaticReplacementEnabled: true
        )

        XCTAssertEqual(context.outputText, "anthropic ai")
        XCTAssertEqual(context.dictionaryCorrectedTerms, [])
        XCTAssertEqual(context.dictionaryMatches.map(\.term), ["Anthropic"])
    }

    func testPreparedDeliveryContextPreservesTextWhenAutomaticReplacementIsDisabled() {
        let matcher = DictionaryMatcher(
            entries: [TestFactories.makeEntry(term: "Anthropic", observedVariants: ["anthropic ai"])],
            blockedGlobalMatchKeys: []
        )

        let context = AppDelegate.preparedDeliveryContext(
            originalText: """
            {"title":"AI Answer","content":"anthropic ai"}
            """,
            llmDurationSeconds: nil,
            sessionOutputMode: .rewrite,
            userMainLanguage: .fallbackOption(),
            matcher: matcher,
            usesConservativeEvidence: false,
            automaticReplacementEnabled: false
        )

        XCTAssertEqual(context.outputText, "anthropic ai")
        XCTAssertEqual(context.dictionaryCorrectedTerms, [])
        XCTAssertEqual(context.dictionaryMatches.map(\.term), ["Anthropic"])
        XCTAssertEqual(context.rewriteAnswerPayload?.content, "anthropic ai")
    }
}
