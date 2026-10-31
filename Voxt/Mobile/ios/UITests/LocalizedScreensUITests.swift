import XCTest

/// Screenshots of every app screen state in one interface language, for checking truncation and
/// clipping. Runs only with TEST_RUNNER_VOXT_SHOT_LANGUAGE=<en|zh-Hans|zh-Hant|ja|es|ko>.
/// Finds everything by identifier, never by English text. Fake data only.
@MainActor
final class LocalizedScreensUITests: XCTestCase {
    private let fakeServer = #"{"baseURL":"http://127.0.0.1:9","token":"ui-fixture-only"}"#
    private var language = ""

    override func setUpWithError() throws {
        continueAfterFailure = true
        language = ProcessInfo.processInfo.environment["VOXT_SHOT_LANGUAGE"] ?? ""
        try XCTSkipIf(language.isEmpty, "Set TEST_RUNNER_VOXT_SHOT_LANGUAGE")
    }

    override func tearDown() {
        // Leave the simulator in English: app-hosted unit tests read the shared language.
        let app = XCUIApplication()
        app.terminate()
        app.launchEnvironment = ["VOXT_TEST_RESET": "1"]
        app.launch()
        app.terminate()
        super.tearDown()
    }

    func testEveryScreenInOneLanguage() {
        // First open
        var app = launch(["VOXT_TEST_ONBOARDING": "welcome"], paired: false)
        wait(app.buttons["onboardingNext"]); shot("welcome", app)
        app = launch(["VOXT_TEST_ONBOARDING": "connect"], paired: false)
        wait(app.buttons["findMacButton"]); shot("connect", app)
        tap(app.buttons["noConnector"]); shot("connect-no-connector", app)
        app = launch(["VOXT_TEST_ONBOARDING": "addKeyboard"])
        wait(app.staticTexts["addKeyboardPath"]); shot("add-keyboard", app)
        app = launch(["VOXT_TEST_ONBOARDING": "fullAccess"])
        wait(app.buttons["whatsFullAccess"]); shot("full-access", app)
        tap(app.buttons["whatsFullAccess"]); shot("full-access-explained", app)
        app = launch(["VOXT_TEST_ONBOARDING": "tryIt", "VOXT_TEST_TRY_KEYBOARD": "noFullAccess"])
        wait(app.buttons["tryItOpenSettings"], timeout: 8); shot("try-full-access-off", app)

        // Home
        app = launch()
        wait(app.switches["keepVoxtReady"]); shot("home-day-one", app)
        app = launch(["VOXT_TEST_HISTORY": "Running ten minutes late, save me a seat.|2\n我会晚十分钟到。|180\nHi Karen, the heater in Unit 4B stopped working on Tuesday. Could you send someone this week?|30"])
        wait(app.staticTexts["weekSummary"]); shot("home-with-dictations", app)

        // Settings, top to bottom, and Advanced
        app = launch(["VOXT_TEST_PREFERRED_LANGUAGES": "en-US,es-ES,ja-JP,de-DE"])
        tap(app.buttons["openSettings"])
        wait(app.staticTexts["settingsSection-yourMac"]); shot("settings-1", app)
        for page in 2...4 { app.swipeUp(); shot("settings-\(page)", app) }
        tap(app.buttons["settingsAdvanced"]); wait(app.switches["developerModeToggle"]); shot("settings-advanced", app)

        // Translation: three languages, the add sheet, and the empty state
        app = launch(["VOXT_TEST_PREFERRED_LANGUAGES": "en-US,es-ES,ja-JP,de-DE"])
        tap(app.buttons["openSettings"])
        tap(app.buttons["translationLanguages"]); wait(app.staticTexts["pill-0"]); shot("translation-languages", app)
        app = launch(["VOXT_TEST_PREFERRED_LANGUAGES": "en-US,es-ES"])
        tap(app.buttons["openSettings"])
        tap(app.buttons["translationLanguages"]); tap(app.buttons["addLanguage"]); wait(app.buttons["languageOption-ja"])
        tap(app.buttons["languageOption-ja"]); tap(app.buttons["languageOption-de"]); shot("translation-add-language", app)
        app = launch(["VOXT_TEST_PREFERRED_LANGUAGES": "en-US"])
        tap(app.buttons["openSettings"])
        tap(app.buttons["translationLanguages"]); wait(app.buttons["addLanguage"]); shot("translation-languages-empty", app)
    }

    // MARK: Helpers

    private func launch(_ environment: [String: String] = [:], paired: Bool = true) -> XCUIApplication {
        let app = XCUIApplication()
        app.terminate()
        let seeded = environment.filter { $0.key == "VOXT_TEST_PREFERRED_LANGUAGES" }
        app.launchEnvironment = seeded.merging(["VOXT_TEST_RESET": "1", "VOXT_TEST_LANGUAGE": language]) { $1 }
        app.launch()
        app.terminate()
        app.launchEnvironment = environment.merging(["VOXT_TEST_LANGUAGE": language]) { $1 }
        if paired { app.launchEnvironment["VOXT_TEST_SERVER"] = fakeServer }
        if let appearance = ProcessInfo.processInfo.environment["VOXT_SHOT_APPEARANCE"] {
            app.launchEnvironment["VOXT_TEST_APPEARANCE"] = appearance
        }
        app.launch()
        let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        if springboard.buttons["Open"].waitForExistence(timeout: 1) { springboard.buttons["Open"].tap() }
        return app
    }

    private func wait(_ element: XCUIElement, timeout: TimeInterval = 10) {
        XCTAssertTrue(element.waitForExistence(timeout: timeout), element.debugDescription)
    }

    private func tap(_ element: XCUIElement) {
        wait(element)
        if element.exists { element.tap() }
        Thread.sleep(forTimeInterval: 0.8) // let the change animate before the next shot
    }

    private func shot(_ name: String, _ app: XCUIApplication) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = "\(language)-\(name)"
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
