import XCTest

/// The app screens, against a fake pairing; no Mac is needed.
@MainActor
final class AppScreensUITests: XCTestCase {
    private let fakeServer = #"{"baseURL":"http://127.0.0.1:9","token":"ui-fixture-only"}"#

    // MARK: First open

    func testFirstOpenWalksOneStepPerScreen() {
        let app = launch(paired: false)
        XCTAssertTrue(app.staticTexts["Say anything. Only your Mac hears it."].waitForExistence(timeout: 10))
        shot("welcome", app)
        app.buttons["onboardingNext"].tap()

        XCTAssertTrue(title(app, is: "Connect your Mac"))
        XCTAssertTrue(app.descendants(matching: .any)["qrScanner"].exists)
        XCTAssertTrue(app.buttons["findMacButton"].exists)
        app.buttons["noConnector"].tap()
        XCTAssertTrue(app.staticTexts["connectorHint"].waitForExistence(timeout: 5))
        shot("connect", app)

        // Pairing is faked by relaunching paired; the walkthrough resumes after Connect.
        app.terminate()
        app.launchEnvironment = ["VOXT_TEST_SERVER": fakeServer]
        app.launchEnvironment["VOXT_TEST_APPEARANCE"] = ProcessInfo.processInfo.environment["VOXT_SHOT_APPEARANCE"]
        app.launch()
        XCTAssertTrue(title(app, is: "Add the Voxt keyboard"))
        // The button opens Settings › Voxt; the text names the rows found there.
        XCTAssertEqual(app.staticTexts["addKeyboardPath"].label, "Settings opens on Voxt. Tap Keyboards, then turn on Voxt.")
        shot("add-keyboard", app)
        app.buttons["onboardingNext"].tap()

        XCTAssertTrue(title(app, is: "Then allow Full Access"))
        XCTAssertTrue(staticText(app, "The keyboard needs it to talk to the Voxt app on this phone. Voxt doesn't keep what you type. When you ask it to edit text, that text goes only to your Mac.").exists)
        app.buttons["whatsFullAccess"].tap()
        XCTAssertTrue(app.staticTexts["fullAccessExplained"].waitForExistence(timeout: 5))
        shot("full-access", app)
        app.buttons["onboardingNext"].tap()
        let settings = XCUIApplication(bundleIdentifier: "com.apple.Preferences")
        XCTAssertTrue(settings.wait(for: .runningForeground, timeout: 10), "Open Settings opens the Settings app")
        app.activate()

        XCTAssertTrue(title(app, is: "Try it"))
        // Which card shows depends on the simulator's keyboard: the switch hint, or, with the
        // Voxt keyboard up and no Full Access, the Full Access card after its 2 s wait. Wait out
        // those 2 s so Skip doesn't move under the tap.
        Thread.sleep(forTimeInterval: 2.5)
        // With the Voxt keyboard up and Full Access on (after an end-to-end run), it points at the mic.
        XCTAssertTrue(app.staticTexts["tryItHint"].exists || app.staticTexts["Allow Full Access is off"].exists
                      || app.staticTexts["Tap and say anything"].exists)
        app.buttons["onboardingSkip"].tap()
        XCTAssertTrue(app.staticTexts["Your dictations show up here"].waitForExistence(timeout: 5), "Done leads to Home")
        app.terminate()
        app.launchEnvironment = [:]
        app.launch()
        XCTAssertTrue(app.staticTexts["Your dictations show up here"].waitForExistence(timeout: 10), "the walkthrough stays done")
    }

    func testTryItWithoutFullAccessOffersSettings() {
        let app = launch(["VOXT_TEST_ONBOARDING": "tryIt", "VOXT_TEST_TRY_KEYBOARD": "noFullAccess"])
        XCTAssertTrue(title(app, is: "Try it"))
        XCTAssertTrue(app.staticTexts["Allow Full Access is off"].waitForExistence(timeout: 6))
        XCTAssertTrue(app.buttons["tryItOpenSettings"].exists)
        shot("try-full-access-off", app)
    }

    func testTryItWithVoxtUpPointsAtTheMic() {
        let app = launch(["VOXT_TEST_ONBOARDING": "tryIt", "VOXT_TEST_TRY_KEYBOARD": "voxt"])
        XCTAssertTrue(title(app, is: "Try it"))
        XCTAssertTrue(app.staticTexts["Tap and say anything"].waitForExistence(timeout: 6))
        XCTAssertFalse(app.staticTexts["Allow Full Access is off"].exists)
        shot("try-mic", app)
    }

    // MARK: Home

    func testHomeOnDayOneHasNoNumbersAndNoSetupCards() {
        let app = launch()
        XCTAssertTrue(app.staticTexts["Your dictations show up here"].waitForExistence(timeout: 10))
        XCTAssertTrue(app.switches["keepVoxtReady"].exists)
        XCTAssertFalse(app.staticTexts["weekSummary"].exists)
        for gone in ["keyboardSetup", "showRecorder", "recordButton", "Writing guide"] {
            XCTAssertFalse(app.buttons[gone].exists, gone)
        }
        XCTAssertFalse(app.staticTexts["usageWords"].exists)
        XCTAssertFalse(app.textFields["tryField"].exists)
        shot("home-day-one", app)
    }

    func testHomeShowsThisWeekAndRecentDictations() {
        let app = launch(["VOXT_TEST_HISTORY": "Running ten minutes late, save me a seat.|2\nSee you at the station at six.|180"])
        let summary = app.staticTexts["weekSummary"]
        XCTAssertTrue(summary.waitForExistence(timeout: 10))
        XCTAssertTrue(summary.label.hasPrefix("This week: 15 words on "), summary.label)
        XCTAssertTrue(app.staticTexts["Running ten minutes late, save me a seat."].exists)
        XCTAssertTrue(app.buttons["copyRecent-0"].exists)
        shot("home-with-dictations", app)
        app.buttons["seeAllHistory"].tap()
        XCTAssertTrue(app.navigationBars["History"].waitForExistence(timeout: 5))
    }

    func testPracticeFieldOnlyForTests() {
        let app = launch(["VOXT_TEST_PRACTICE_FIELD": "1"])
        // A vertical TextField is a text view on iOS 17.0 and a text field on iOS 26.3.
        XCTAssertTrue(app.textFields["tryField"].waitForExistence(timeout: 10) || app.textViews["tryField"].exists)
        let field = app.textFields["tryField"].exists ? app.textFields["tryField"] : app.textViews["tryField"]
        XCTAssertFalse(app.buttons["Hide keyboard"].exists)
        field.tap()
        // Keyboard UI tests close the keyboard with this bar (DictationFlowUITests).
        let hide = app.buttons["Hide keyboard"]
        XCTAssertTrue(hide.waitForExistence(timeout: 5))
        hide.tap()
        XCTAssertFalse(hide.waitForExistence(timeout: 2))
    }

    func testKeepVoxtReadyStartsTheFloatingWindow() {
        let app = launch()
        let toggle = app.switches["keepVoxtReady"]
        XCTAssertTrue(toggle.waitForExistence(timeout: 10))
        if !toggle.isEnabled {
            XCTAssertTrue(app.staticTexts["floatingWindowUnavailable"].exists)
            print("PiP is unavailable in this runtime; the floating window needs a device run.")
            return
        }
        XCTAssertEqual(toggle.value as? String, "0")
        tapSwitch(toggle)
        // On only once Picture in Picture has really started.
        XCTAssertTrue(waitForValue("1", of: toggle, timeout: 15))
        tapSwitch(toggle)
        XCTAssertTrue(waitForValue("0", of: toggle, timeout: 5))
    }

    // MARK: Settings

    func testSettingsHasEverydaySectionsInOrder() {
        let app = launch()
        openSettings(app)
        var previous = -CGFloat.infinity
        for section in ["yourMac", "keyboard", "translation", "writing", "microphone", "privacy", "app", "about"] {
            let title = app.staticTexts["settingsSection-\(section)"]
            XCTAssertTrue(title.exists, section)
            XCTAssertGreaterThan(title.frame.minY, previous, section)
            previous = title.frame.minY
        }
        XCTAssertFalse(app.steppers["typingSpeed"].exists)
        XCTAssertFalse(app.switches["developerModeToggle"].exists, "Developer mode lives in About › Advanced")
        XCTAssertFalse(app.buttons["floatingWindowToggle"].exists, "Keep Voxt ready lives on Home")
        XCTAssertEqual(app.switches["cleanUpWords"].value as? String, "1")
        XCTAssertEqual(app.switches["hapticFeedback"].value as? String, "1")
        shot("settings", app)
    }

    /// About › Source code: a simulator build carries no VoxtSourceRevision, so the row says
    /// "Development build" and links nowhere (a release links its commit; SourceRevisionTests).
    func testAboutSaysDevelopmentBuildWithoutASourceRevision() {
        let app = launch()
        openSettings(app)
        let row = reveal(app.descendants(matching: .any)["sourceCode"], in: app)
        XCTAssertTrue(row.label.contains("Source code"), row.label)
        XCTAssertTrue(row.label.contains("Development build"), row.label)
        XCTAssertFalse(app.links["sourceCode"].exists, "no link without a commit")
    }

    func testInterfaceLanguageOffersSpanishAndKorean() {
        let app = launch()
        openSettings(app)
        let language = reveal(app.buttons["interfaceLanguage"], in: app)
        language.tap()
        XCTAssertTrue(app.buttons["한국어"].waitForExistence(timeout: 3))
        app.buttons["Español"].tap()
        XCTAssertTrue(app.navigationBars["Ajustes"].waitForExistence(timeout: 5))
        XCTAssertTrue(language.label.contains("Español"), language.label)
        language.tap()
        app.buttons["한국어"].tap()
        XCTAssertTrue(app.navigationBars["설정"].waitForExistence(timeout: 5))
        language.tap()
        app.buttons["English"].tap()
    }

    func testPinyinRowsShowOnlyWithThePinyinLayout() {
        let app = launch()
        openSettings(app)
        setLayout("zh-Hans-pinyin", on: false, in: app)
        XCTAssertFalse(app.switches["pinyinFuzzy"].exists)
        XCTAssertFalse(app.buttons["clearPinyinLearnedWords"].exists)
        setLayout("zh-Hans-pinyin", on: true, in: app)
        let fuzzy = reveal(app.switches["pinyinFuzzy"], in: app)
        XCTAssertEqual(fuzzy.value as? String, "0", "off by default")
        XCTAssertFalse(app.buttons["clearPinyinLearnedWords"].isEnabled, "nothing learned yet")
        tapSwitch(fuzzy)
        XCTAssertTrue(waitForValue("1", of: fuzzy, timeout: 3))
        app.terminate()
        app.launchEnvironment = [:]
        app.launch()
        openSettings(app)
        XCTAssertEqual(reveal(app.switches["pinyinFuzzy"], in: app).value as? String, "1")
    }

    private func setLayout(_ id: String, on: Bool, in app: XCUIApplication) {
        reveal(app.buttons["Text keyboards"], in: app).tap()
        let add = app.buttons["addLayout-" + id], remove = app.buttons["removeLayout-" + id]
        XCTAssertTrue(add.waitForExistence(timeout: 5) || remove.exists)
        if on, add.exists { add.tap() }
        if !on, remove.exists { remove.tap() }
        XCTAssertTrue((on ? remove : add).waitForExistence(timeout: 3))
        app.navigationBars.buttons["Done"].firstMatch.tap()
        XCTAssertTrue(app.staticTexts["settingsSection-yourMac"].waitForExistence(timeout: 5))
    }

    func testDeveloperModeIsUnderAdvancedAndStillAsks() {
        let app = launch()
        openSettings(app)
        reveal(app.buttons["settingsAdvanced"], in: app).tap()
        let toggle = app.switches["developerModeToggle"]
        XCTAssertTrue(toggle.waitForExistence(timeout: 5))
        XCTAssertTrue(app.buttons["variants"].exists)
        shot("settings-advanced", app)
        tapSwitch(toggle)
        XCTAssertTrue(app.alerts["Turn on Developer mode?"].waitForExistence(timeout: 5))
        app.alerts.buttons["Cancel"].tap()
        XCTAssertEqual(toggle.value as? String, "0")
    }

    func testCleanUpAndHapticsPersist() {
        let app = launch()
        openSettings(app)
        for id in ["hapticFeedback", "cleanUpWords"] {
            let toggle = reveal(app.switches[id], in: app)
            tapSwitch(toggle)
            XCTAssertTrue(waitForValue("0", of: toggle, timeout: 3), id)
        }
        app.terminate()
        app.launchEnvironment = [:]
        app.launch()
        openSettings(app)
        XCTAssertEqual(app.switches["cleanUpWords"].value as? String, "0")
        XCTAssertEqual(app.switches["hapticFeedback"].value as? String, "0")
    }

    func testSetUpAgainOpensAddKeyboardAndKeepsThePairing() {
        let app = launch()
        openSettings(app)
        reveal(app.buttons["setUpAgain"], in: app).tap()
        XCTAssertTrue(title(app, is: "Add the Voxt keyboard"))
        app.buttons["onboardingNext"].tap()
        XCTAssertTrue(title(app, is: "Then allow Full Access"))
    }

    /// Device tests set their own first-run state: "done" opens Home even when a test left the steps open.
    func testOnboardingLaunchSettingDoneOpensHome() {
        let app = launch()
        openSettings(app)
        reveal(app.buttons["setUpAgain"], in: app).tap()
        XCTAssertTrue(title(app, is: "Add the Voxt keyboard"))
        app.terminate()
        app.launchEnvironment["VOXT_TEST_ONBOARDING"] = "done"
        app.launch()
        XCTAssertTrue(app.switches["keepVoxtReady"].waitForExistence(timeout: 10), "Home shows")
    }

    // MARK: Translation languages

    func testOnePhoneLanguageStartsEmpty() {
        let app = launch(["VOXT_TEST_PREFERRED_LANGUAGES": "en-US"])
        openTranslation(app)
        XCTAssertTrue(app.staticTexts["Choose translation languages"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.buttons["addLanguage"].exists)
        shot("translation-languages-empty", app)
    }

    func testThePhonesOtherLanguagesAreTheStartingTargets() {
        let app = launch(["VOXT_TEST_PREFERRED_LANGUAGES": "en-US,ja-JP,es-ES"])
        openTranslation(app)
        XCTAssertEqual(app.staticTexts["pill-0"].label, "日本語")
        XCTAssertEqual(app.staticTexts["pill-1"].label, "Español")
        XCTAssertFalse(app.staticTexts["pill-2"].exists)
    }

    func testAddThreeReorderAndRemove() {
        let app = launch(["VOXT_TEST_PREFERRED_LANGUAGES": "en-US"])
        openTranslation(app)
        app.buttons["addLanguage"].tap()
        for code in ["es", "ja", "de"] {
            let option = app.buttons["languageOption-\(code)"]
            XCTAssertTrue(option.waitForExistence(timeout: 5), code)
            option.tap()
        }
        XCTAssertFalse(app.buttons["languageOption-fr"].isEnabled, "a 4th is not offered")
        shot("translation-add-language", app)
        app.searchFields.firstMatch.tap()
        app.searchFields.firstMatch.typeText("Kor")
        XCTAssertTrue(app.buttons["languageOption-ko"].waitForExistence(timeout: 5))
        XCTAssertFalse(app.buttons["languageOption-fr"].exists)
        // iOS 26 hides the bar buttons while search is active; close search first.
        // The search bar's own button: "Close" on iOS 26.3, "Cancel" on iOS 17.0.
        if app.buttons["Close"].exists { app.buttons["Close"].tap() } else { app.buttons["Cancel"].firstMatch.tap() }
        XCTAssertTrue(app.buttons["addLanguageDone"].waitForExistence(timeout: 3))
        app.buttons["addLanguageDone"].tap()

        XCTAssertEqual(pills(app), ["Español", "日本語", "Deutsch"])
        shot("translation-languages", app)

        let deutsch = app.cells.containing(.staticText, identifier: "targetName-de").firstMatch
        let espanol = app.cells.containing(.staticText, identifier: "targetName-es").firstMatch
        // Drop on the upper part of the first row so it lands above it.
        deutsch.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
            .press(forDuration: 1.0, thenDragTo: espanol.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.1)))
        XCTAssertEqual(pills(app), ["Deutsch", "Español", "日本語"])

        app.cells.containing(.staticText, identifier: "targetName-ja").firstMatch.swipeLeft()
        shot("translation-languages-swipe-remove", app)
        app.buttons["Remove"].tap()
        XCTAssertEqual(pills(app), ["Deutsch", "Español"])

        app.terminate()
        app.launchEnvironment = [:]
        app.launch()
        openTranslation(app)
        XCTAssertEqual(pills(app), ["Deutsch", "Español"])
    }

    func testTheKeyboardsTranslationLinkOpensTranslationRunningAndCold() {
        let app = launch(["VOXT_TEST_PREFERRED_LANGUAGES": "en-US"])
        XCTAssertTrue(app.switches["keepVoxtReady"].waitForExistence(timeout: 10))
        openTranslationLink(app)
        XCTAssertTrue(app.navigationBars["Translation"].waitForExistence(timeout: 10), "running app")
        XCTAssertTrue(app.staticTexts["Choose translation languages"].exists)
        app.buttons["translationDone"].tap()
        XCTAssertFalse(app.navigationBars["Translation"].waitForExistence(timeout: 2))
        app.terminate()
        openTranslationLink(app)
        XCTAssertTrue(app.navigationBars["Translation"].waitForExistence(timeout: 15), "cold launch")
    }

    // MARK: Helpers

    /// As the keyboard does it: open the link from outside the app.
    private func openTranslationLink(_ app: XCUIApplication) {
        app.open(URL(string: "voxtmobile://translation")!)
        let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        if springboard.buttons["Open"].waitForExistence(timeout: 3) { springboard.buttons["Open"].tap() }
    }

    private func launch(_ environment: [String: String] = [:], paired: Bool = true) -> XCUIApplication {
        continueAfterFailure = false
        let app = XCUIApplication()
        // Keyboard settings are seeded on the reset launch, so it sees the test languages too.
        app.launchEnvironment = environment.filter { $0.key == "VOXT_TEST_PREFERRED_LANGUAGES" }
        app.launchEnvironment["VOXT_TEST_RESET"] = "1"
        app.launch()
        app.terminate()
        app.launchEnvironment = environment
        if paired { app.launchEnvironment["VOXT_TEST_SERVER"] = fakeServer }
        // Screenshot runs: TEST_RUNNER_VOXT_SHOT_APPEARANCE=dark|light picks the theme.
        if let appearance = ProcessInfo.processInfo.environment["VOXT_SHOT_APPEARANCE"] {
            app.launchEnvironment["VOXT_TEST_APPEARANCE"] = appearance
        }
        app.launch()
        let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        if springboard.buttons["Open"].waitForExistence(timeout: 2) { springboard.buttons["Open"].tap() }
        return app
    }

    private func openSettings(_ app: XCUIApplication) {
        XCTAssertTrue(app.buttons["Settings"].waitForExistence(timeout: 10))
        app.buttons["Settings"].tap()
        XCTAssertTrue(app.staticTexts["settingsSection-yourMac"].waitForExistence(timeout: 5))
    }

    private func openTranslation(_ app: XCUIApplication) {
        openSettings(app)
        reveal(app.buttons["translationLanguages"], in: app).tap()
        XCTAssertTrue(app.navigationBars["Translation"].waitForExistence(timeout: 5))
    }

    private func pills(_ app: XCUIApplication) -> [String] {
        (0..<3).map { app.staticTexts["pill-\($0)"] }.filter(\.exists).map(\.label)
    }

    private func title(_ app: XCUIApplication, is text: String) -> Bool {
        let title = app.staticTexts["onboardingTitle"]
        guard title.waitForExistence(timeout: 10) else { return false }
        let deadline = Date().addingTimeInterval(5)
        while title.label != text, Date() < deadline { Thread.sleep(forTimeInterval: 0.2) }
        XCTAssertEqual(title.label, text)
        return title.label == text
    }

    @discardableResult
    private func reveal(_ element: XCUIElement, in app: XCUIApplication) -> XCUIElement {
        for _ in 0..<8 where !(element.exists && element.isHittable) { app.swipeUp() }
        XCTAssertTrue(element.exists, element.debugDescription)
        return element
    }

    private func waitForValue(_ value: String, of element: XCUIElement, timeout: TimeInterval) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while element.value as? String != value, Date() < deadline { Thread.sleep(forTimeInterval: 0.25) }
        return element.value as? String == value
    }

    /// Labels with an apostrophe can't go in a subscript query (iOS 26.3: "Invalid query").
    private func staticText(_ app: XCUIApplication, _ label: String) -> XCUIElement {
        app.staticTexts.matching(NSPredicate(format: "label == %@", label)).firstMatch
    }

    private func tapSwitch(_ toggle: XCUIElement) {
        toggle.coordinate(withNormalizedOffset: CGVector(dx: 0.9, dy: 0.5)).tap()
    }

    private func shot(_ name: String, _ app: XCUIApplication) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
