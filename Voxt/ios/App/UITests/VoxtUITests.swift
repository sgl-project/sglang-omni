import XCTest

/// Drives the app in the simulator against a fake OpenAI-compatible
/// server on this Mac (no real service, no key). Start one before the run:
///
///   python3 App/UITests/fake_server.py
///
/// The server answers every chat with a fixed English text (a title and
/// content when Rewrite writes from scratch) and every transcription with
/// a fixed sentence, only when it carries the key typed here (so a key the
/// Keychain did not keep fails the test). The recording tests record this
/// Mac's microphone and allow the system's prompt on their first run.
/// Set VOXT_UI_LANGUAGE (TEST_RUNNER_VOXT_UI_LANGUAGE through xcodebuild)
/// to run the app in "en" or "zh-Hans"; VOXT_FAKE_SERVER to move the
/// server. Each test leaves screenshots in the result bundle.
///
/// The tests run in name order and share the app's settings: test1
/// points the language model at the server for the ones after it.
final class VoxtUITests: XCTestCase {
    static let answer = "Hello, this is a test of Voxt on iOS."
    static let speech = "你好，这是 Voxt 在 iOS 上的测试。"

    var server: String { ProcessInfo.processInfo.environment["VOXT_FAKE_SERVER"] ?? "http://127.0.0.1:18080/v1" }
    var language: String { ProcessInfo.processInfo.environment["VOXT_UI_LANGUAGE"] ?? "en" }

    var app: XCUIApplication!

    override func setUp() {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launchArguments = ["-AppleLanguages", "(\(language))", "-AppleLocale", language == "en" ? "en_US" : "zh_CN"]
        app.launch()
    }

    func test0TabsShowTheThreeScreens() {
        snap("voice")
        tab(1)
        snap("history")
        tab(2)
        XCTAssertTrue(app.buttons["settings.languageModels"].waitForExistence(timeout: 5))
        snap("settings")
        app.swipeUp()
        snap("settings-bottom")
    }

    func test1LanguageModelConnectsToTheServer() {
        tab(2)
        app.buttons["settings.languageModels"].tap()
        let service = app.buttons["lm.service"]
        XCTAssertTrue(service.waitForExistence(timeout: 5))
        service.tap()
        let custom = app.buttons["Custom (OpenAI-compatible)"]
        XCTAssertTrue(custom.waitForExistence(timeout: 5), "the presets come from the Go core")
        custom.tap()
        replace(app.textFields["lm.baseURL"], with: server)
        replace(app.textFields["lm.model"], with: "fake")
        snap("language-model")
        app.buttons["lm.test"].tap()
        let result = app.staticTexts["lm.result"]
        XCTAssertTrue(result.waitForExistence(timeout: 20))
        snap("language-model-tested")
        XCTAssertTrue(result.label.contains(language == "en" ? "Connected in" : "已连接"), result.label)
    }

    func test2TranslateTypedText() {
        app.segmentedControls["voice.feature"].buttons.element(boundBy: 1).tap()
        let source = app.textViews["voice.source"]
        XCTAssertTrue(source.waitForExistence(timeout: 5))
        source.tap()
        source.typeText("你好，世界。这是一段要翻译的文字。")
        snap("translate-typed")
        app.buttons["voice.translate"].tap()
        let result = app.staticTexts["voice.result"]
        XCTAssertTrue(result.waitForExistence(timeout: 30))
        XCTAssertEqual(result.label, Self.answer)
        snap("translate-result")

        tab(1)
        let entry = app.cells.firstMatch
        XCTAssertTrue(entry.waitForExistence(timeout: 5))
        XCTAssertTrue(app.staticTexts[Self.answer].exists, "the session is kept in the history")
        snap("history-entry")
        entry.tap()
        XCTAssertTrue(app.staticTexts["你好，世界。这是一段要翻译的文字。"].waitForExistence(timeout: 5))
        snap("history-detail")
    }

    /// Record → WAV → the Go core's speech upload → cleanup. The simulator
    /// records this Mac's microphone; the first run allows it in the
    /// system's prompt.
    func test3TranscribeWithTheRemoteSpeechService() {
        tab(2)
        app.buttons["settings.speechModels"].tap()
        let remote = app.buttons["speech.remote"]
        XCTAssertTrue(remote.waitForExistence(timeout: 5))
        remote.tap()
        replace(app.textFields["speech.endpoint"], with: server)
        replace(app.secureTextFields["speech.key"], with: "test")
        snap("speech-remote")

        tab(0)
        app.segmentedControls["voice.feature"].buttons.element(boundBy: 0).tap()
        let record = app.buttons["voice.record"]
        startRecording(record)
        sleep(2)
        snap("recording")
        record.tap()
        let result = app.staticTexts["voice.result"]
        if !result.waitForExistence(timeout: 30) {
            snap("transcribe-failed")
            XCTFail("no result; status: \(app.staticTexts["voice.status"].firstMatch.label)")
            return
        }
        XCTAssertEqual(result.label, Self.answer)
        snap("transcribe-result")
    }

    /// A spoken instruction over a typed text: the answer replaces the
    /// text, so it has no title.
    func test4RewriteTypedText() {
        rewrite(source: "Voxt is a voice app.")
        XCTAssertFalse(app.staticTexts["Test title"].exists)
        snap("rewrite-result")
    }

    /// Only an instruction: Voxt writes from scratch, a JSON answer with a
    /// title.
    func test5RewriteFromScratch() {
        rewrite(source: nil)
        XCTAssertTrue(app.staticTexts["Test title"].exists)
        snap("write-result")
    }

    private func rewrite(source text: String?) {
        app.segmentedControls["voice.feature"].buttons.element(boundBy: 2).tap()
        let source = app.textViews["voice.source"]
        XCTAssertTrue(source.waitForExistence(timeout: 5))
        if let text {
            source.tap()
            source.typeText(text)
        }
        let record = app.buttons["voice.record"]
        startRecording(record)
        sleep(2)
        record.tap()
        let result = app.staticTexts["voice.result"]
        if !result.waitForExistence(timeout: 30) {
            snap("rewrite-failed")
            XCTFail("no result; status: \(app.staticTexts["voice.status"].firstMatch.label)")
            return
        }
        XCTAssertEqual(result.label, Self.answer)
    }

    /// Taps record, allowing the system's prompts (microphone, speech
    /// recognition) on the first run, then makes sure it records.
    private func startRecording(_ record: XCUIElement) {
        record.tap()
        let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        var prompted = false
        while springboard.alerts.firstMatch.waitForExistence(timeout: 3) {
            snap("system-prompt")
            springboard.alerts.firstMatch.buttons.allElementsBoundByIndex.last?.tap() // Allow / OK
            prompted = true
        }
        if prompted && !app.staticTexts["voice.status"].waitForExistence(timeout: 2) {
            record.tap()
        }
        XCTAssertTrue(app.staticTexts["voice.status"].waitForExistence(timeout: 5), "recording")
    }

    private func tab(_ index: Int) {
        let button = app.tabBars.buttons.element(boundBy: index)
        XCTAssertTrue(button.waitForExistence(timeout: 5))
        button.tap()
    }

    private func replace(_ field: XCUIElement, with text: String) {
        XCTAssertTrue(field.waitForExistence(timeout: 5))
        // At the right end, so that the cursor follows the old text.
        field.coordinate(withNormalizedOffset: CGVector(dx: 0.98, dy: 0.5)).tap()
        let old = (field.value as? String) ?? ""
        if !old.isEmpty && old != field.placeholderValue {
            field.typeText(String(repeating: XCUIKeyboardKey.delete.rawValue, count: old.count))
        }
        field.typeText(text + "\n") // return: the keyboard goes, uncovering the tab bar
    }

    private func snap(_ name: String) {
        let shot = XCTAttachment(screenshot: app.screenshot())
        shot.name = "\(language)-\(name)"
        shot.lifetime = .keepAlways
        add(shot)
    }
}
