import XCTest

/// Try it with the real Voxt keyboard (Full Access on) and a Mac: tools/design-tour-fixture.py, or
/// Voxt's Local API. Runs only when the test runner gets VOXT_E2E_URL, VOXT_E2E_TOKEN and
/// VOXT_E2E_WAV (prefix TEST_RUNNER_ for xcodebuild); VOXT_E2E_EXPECT is text the result must
/// contain, the fixture's reply by default.
@MainActor
final class TryItEndToEndUITests: XCTestCase {
    private var env: [String: String] { ProcessInfo.processInfo.environment }

    override func setUpWithError() throws {
        continueAfterFailure = false
        try XCTSkipIf(env["VOXT_E2E_URL"] == nil, "Set TEST_RUNNER_VOXT_E2E_URL to run end-to-end UI tests")
    }

    func testTryItPointsAtTheMicThenFinishesAfterARealDictation() throws {
        // On a fresh simulator Settings lists the Voxt keyboard only after Voxt has been opened once.
        let app = XCUIApplication()
        app.launchEnvironment = ["VOXT_TEST_RESET": "1"]
        app.launch()
        app.terminate()
        enableVoxtKeyboardWithFullAccess()
        app.launchEnvironment = [
            "VOXT_TEST_SERVER": String(data: try JSONSerialization.data(withJSONObject: [
                "baseURL": try XCTUnwrap(env["VOXT_E2E_URL"]), "token": try XCTUnwrap(env["VOXT_E2E_TOKEN"]),
            ]), encoding: .utf8)!,
            "VOXT_TEST_AUDIO_FILE": try XCTUnwrap(env["VOXT_E2E_WAV"]),
            "VOXT_TEST_ONBOARDING": "tryIt",
        ]
        // README images: TEST_RUNNER_VOXT_SHOT_APPEARANCE=dark.
        app.launchEnvironment["VOXT_TEST_APPEARANCE"] = env["VOXT_SHOT_APPEARANCE"]
        app.launch()
        XCTAssertTrue(app.staticTexts["onboardingTitle"].waitForExistence(timeout: 10))
        XCTAssertEqual(app.staticTexts["onboardingTitle"].label, "Try it")

        let mic = app.buttons["Start dictation"]
        switchToVoxt(app, mic: mic)
        // The keyboard reports Full Access as it appears; the app points at the mic.
        XCTAssertTrue(app.staticTexts["Tap and say anything"].waitForExistence(timeout: 5))
        Thread.sleep(forTimeInterval: 2.5)
        XCTAssertFalse(app.staticTexts["Allow Full Access is off"].exists, "the 2 s wait passed with Full Access on")
        shot("try-it-mic-real-keyboard", app)

        mic.tap()
        let stop = app.buttons["Stop dictation"]
        XCTAssertTrue(stop.waitForExistence(timeout: 10), "Keyboard never saw recording start")
        stop.tap()

        XCTAssertTrue(app.staticTexts["That's it. Use it in any app."].waitForExistence(timeout: 60))
        let field = app.textFields["tryItField"].exists ? app.textFields["tryItField"] : app.textViews["tryItField"]
        let expected = env["VOXT_E2E_EXPECT"] ?? "design review dictation"
        XCTAssertTrue((field.value as? String ?? "").contains(expected), "Field holds: \(field.value ?? "nil")")
        shot("try-it-done-real-keyboard", app)

        app.buttons["onboardingNext"].tap()
        XCTAssertTrue(app.descendants(matching: .any)["connectedLabel"].waitForExistence(timeout: 10)
                      || app.staticTexts["Recent"].waitForExistence(timeout: 5), "Done leads to Home")
        shot("home-after-try-it", app)
    }

    // MARK: Helpers (same steps as DictationFlowUITests)

    /// A fresh simulator shows two one-time iOS tips over the keyboard: "Quickly Change Keyboards"
    /// after the first switch, and the slide-to-type tip over the first system keyboard. Continue closes either.
    private func dismissKeyboardTip(_ app: XCUIApplication) {
        let tip = app.staticTexts.matching(NSPredicate(format: "label == 'Quickly Change Keyboards' OR label BEGINSWITH 'Speed up your typing by sliding'")).firstMatch
        let close = app.buttons["Continue"].firstMatch
        if tip.waitForExistence(timeout: 1), close.exists { close.tap() }
    }

    private func switchToVoxt(_ app: XCUIApplication, mic: XCUIElement) {
        let next = app.buttons.matching(NSPredicate(format: "label == 'Next keyboard' OR identifier == 'Next keyboard'")).firstMatch
        var switches = 0
        while !mic.waitForExistence(timeout: 2), switches < 6, next.exists {
            next.tap()
            dismissKeyboardTip(app)
            switches += 1
        }
        if !mic.exists, next.exists {
            next.press(forDuration: 1.2)
            let voxt = app.descendants(matching: .any).matching(NSPredicate(format: "label == 'Voxt'")).firstMatch
            if voxt.waitForExistence(timeout: 3) { voxt.tap() }
            _ = mic.waitForExistence(timeout: 5)
        }
        XCTAssertTrue(mic.exists, "Voxt keyboard did not appear")
    }

    private func enableVoxtKeyboardWithFullAccess() {
        let settings = XCUIApplication(bundleIdentifier: "com.apple.Preferences")
        settings.terminate()
        settings.launch()
        tapCell(settings, settings.staticTexts["General"].firstMatch)
        tapCell(settings, settings.staticTexts["Keyboard"].firstMatch)
        tapCell(settings, settings.cells["KEYBOARDS"].firstMatch)
        XCTAssertTrue(settings.navigationBars["Keyboards"].waitForExistence(timeout: 5))
        if !labeled(settings, "Voxt").waitForExistence(timeout: 3) {
            tapCell(settings, settings.cells["AddNewKeyboard"].firstMatch)
            tapCell(settings, labeled(settings, "Voxt"))
            XCTAssertTrue(settings.navigationBars["Keyboards"].waitForExistence(timeout: 5))
        }
        tapCell(settings, labeled(settings, "Voxt"))
        let fullAccess = settings.switches.firstMatch
        XCTAssertTrue(fullAccess.waitForExistence(timeout: 5))
        if fullAccess.value as? String != "1" {
            fullAccess.coordinate(withNormalizedOffset: CGVector(dx: 0.9, dy: 0.5)).tap()
            let allow = settings.alerts.buttons["Allow"]
            if allow.waitForExistence(timeout: 5) { allow.tap() }
        }
        wait(for: [expectation(for: NSPredicate(format: "value == '1'"), evaluatedWith: settings.switches.firstMatch)], timeout: 5)
        settings.terminate()
    }

    private func labeled(_ app: XCUIApplication, _ label: String) -> XCUIElement {
        app.descendants(matching: .any).matching(NSPredicate(format: "label == %@", label)).firstMatch
    }

    private func tapCell(_ app: XCUIApplication, _ element: XCUIElement) {
        if !element.waitForExistence(timeout: 5) { app.swipeUp() }
        XCTAssertTrue(element.waitForExistence(timeout: 3), element.debugDescription)
        element.tap()
    }

    private func shot(_ name: String, _ app: XCUIApplication) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
