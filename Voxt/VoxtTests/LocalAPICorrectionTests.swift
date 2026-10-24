// LocalAPICorrectionTests.swift
// Prompt, strict parser, and passage checks for POST /v1/corrections.

import XCTest
@testable import Voxt

final class LocalAPICorrectionTests: XCTestCase {
    func testPromptAsksForOneFullWordAndIncludesThePassages() {
        let prompt = LocalAPICorrectionPrompt.instruction(original: "请用星浪处理。", edited: "请用星朗处理。")
        XCTAssertTrue(prompt.contains("请用星浪处理。"))
        XCTAssertTrue(prompt.contains("请用星朗处理。"))
        XCTAssertTrue(prompt.contains("full"))
        XCTAssertTrue(prompt.contains("{\"correction\":null}"))
        XCTAssertTrue(prompt.lowercased().contains("append"))
        XCTAssertTrue(prompt.lowercased().contains("delet"))
        XCTAssertTrue(prompt.lowercased().contains("number"))
        XCTAssertTrue(prompt.lowercased().contains("punctuation"))
        XCTAssertTrue(prompt.lowercased().contains("rewrite"))
        XCTAssertTrue(prompt.contains("only"))
    }

    func testPromptDoesNotFewShotTheLiveChinesePair() {
        let prompt = LocalAPICorrectionPrompt.instruction(original: "Use Voice today.", edited: "Use Voxt today.")
        XCTAssertFalse(prompt.contains("星浪"))
        XCTAssertFalse(prompt.contains("星朗"))
        XCTAssertTrue(prompt.contains("Use Voice today."))
        XCTAssertTrue(prompt.contains("Use Voxt today."))
    }

    func testParserAcceptsTheFullChineseName() throws {
        let parsed = try LocalAPICorrection.interpret(
            modelOutput: #"{"correction":{"original":"星浪","replacement":"星朗"}}"#,
            original: "请用星浪处理。",
            edited: "请用星朗处理。"
        )
        XCTAssertEqual(parsed, LocalAPICorrectionPair(original: "星浪", replacement: "星朗"))
    }

    func testParserAcceptsAnAlphanumericModelNameAndRejectsItsDigit() throws {
        let parsed = try LocalAPICorrection.interpret(
            modelOutput: #"{"correction":{"original":"Qwen2","replacement":"Qwen3"}}"#,
            original: "Use Qwen2 today.",
            edited: "Use Qwen3 today."
        )
        XCTAssertEqual(parsed, LocalAPICorrectionPair(original: "Qwen2", replacement: "Qwen3"))
        XCTAssertThrowsError(try LocalAPICorrection.interpret(
            modelOutput: #"{"correction":{"original":"2","replacement":"3"}}"#,
            original: "Use Qwen2 today.",
            edited: "Use Qwen3 today."
        ))
    }

    func testParserAcceptsAFullLatinWord() throws {
        let parsed = try LocalAPICorrection.interpret(
            modelOutput: #"{"correction":{"original":"Voice","replacement":"Voxt"}}"#,
            original: "Use Voice today.",
            edited: "Use Voxt today."
        )
        XCTAssertEqual(parsed, LocalAPICorrectionPair(original: "Voice", replacement: "Voxt"))
    }

    func testParserReturnsNilWhenTheModelReportsNoCorrection() throws {
        let parsed = try LocalAPICorrection.interpret(
            modelOutput: #"{"correction":null}"#,
            original: "Use Voice today.",
            edited: "Use Voxt today."
        )
        XCTAssertNil(parsed)
    }

    func testParserRejectsInvalidModelJSON() {
        XCTAssertThrowsError(try LocalAPICorrection.interpret(
            modelOutput: "星浪 -> 星朗",
            original: "请用星浪处理。",
            edited: "请用星朗处理。"
        ))
        XCTAssertThrowsError(try LocalAPICorrection.interpret(
            modelOutput: #"{"correction":{"original":"星浪"}}"#,
            original: "请用星浪处理。",
            edited: "请用星朗处理。"
        ))
        XCTAssertThrowsError(try LocalAPICorrection.interpret(
            modelOutput: #"{"original":"星浪","replacement":"星朗"}"#,
            original: "请用星浪处理。",
            edited: "请用星朗处理。"
        ))
    }

    func testParserRejectsASingleHanFragmentAndAHallucinatedTerm() {
        XCTAssertThrowsError(try LocalAPICorrection.interpret(
            modelOutput: #"{"correction":{"original":"浪","replacement":"朗"}}"#,
            original: "请用星浪处理。",
            edited: "请用星朗处理。"
        ))
        XCTAssertThrowsError(try LocalAPICorrection.interpret(
            modelOutput: #"{"correction":{"original":"ice","replacement":"oxt"}}"#,
            original: "Use Voice today.",
            edited: "Use Voxt today."
        ))
        XCTAssertThrowsError(try LocalAPICorrection.interpret(
            modelOutput: #"{"correction":{"original":"Keyboard","replacement":"Voxt"}}"#,
            original: "Use Voice today.",
            edited: "Use Voxt today."
        ))
    }

    func testIgnorableEditsAreNotWordCorrections() {
        XCTAssertTrue(LocalAPICorrection.isIgnorable(original: "Use Voxt.", edited: "Use Voxt. Thanks!"))
        XCTAssertTrue(LocalAPICorrection.isIgnorable(original: "Use Voxt today.", edited: "Use today."))
        XCTAssertTrue(LocalAPICorrection.isIgnorable(original: "Use Voxt.", edited: "Use Voxt!"))
        XCTAssertTrue(LocalAPICorrection.isIgnorable(original: "Pay $50.", edited: "Pay $60."))
        XCTAssertTrue(LocalAPICorrection.isIgnorable(original: "Use 50 today.", edited: "Use 60 today."))
        XCTAssertFalse(LocalAPICorrection.isIgnorable(original: "Use Qwen2 today.", edited: "Use Qwen3 today."))
        XCTAssertTrue(LocalAPICorrection.isIgnorable(
            original: "See Alice about the invoice",
            edited: "See Bob about the bill"
        ))
        XCTAssertFalse(LocalAPICorrection.isIgnorable(original: "请用星浪处理。", edited: "请用星朗处理。"))
        XCTAssertFalse(LocalAPICorrection.isIgnorable(original: "Use Voice today.", edited: "Use Voxt today."))
        XCTAssertFalse(LocalAPICorrection.isIgnorable(original: "Use oxt.", edited: "Use Voxt."))
        XCTAssertFalse(LocalAPICorrection.isIgnorable(original: "Use Vooxt.", edited: "Use Voxt."))
        XCTAssertFalse(LocalAPICorrection.isIgnorable(original: "oxt today.", edited: "Voxt today."))
    }

    func testParserAcceptsALetterInsertedOrDeletedInsideAWord() throws {
        let inserted = try LocalAPICorrection.interpret(
            modelOutput: #"{"correction":{"original":"oxt","replacement":"Voxt"}}"#,
            original: "Use oxt.",
            edited: "Use Voxt."
        )
        XCTAssertEqual(inserted, LocalAPICorrectionPair(original: "oxt", replacement: "Voxt"))
        let deleted = try LocalAPICorrection.interpret(
            modelOutput: #"{"correction":{"original":"Vooxt","replacement":"Voxt"}}"#,
            original: "Use Vooxt.",
            edited: "Use Voxt."
        )
        XCTAssertEqual(deleted, LocalAPICorrectionPair(original: "Vooxt", replacement: "Voxt"))
    }
}
