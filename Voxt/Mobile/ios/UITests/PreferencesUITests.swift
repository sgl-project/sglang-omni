import XCTest

/// Exercises local settings with an explicitly fake pairing; no live Mac or token is needed.
@MainActor
final class PreferencesUITests: XCTestCase {
    func testCloudConsentDefaultsOffAndPersists() {
        let app = launchSettings()
        let toggle = app.switches["allowCloudModels"]
        XCTAssertTrue(toggle.waitForExistence(timeout: 5))
        XCTAssertEqual(toggle.value as? String, "0")
        toggle.tap()
        app.buttons["Done"].tap()
        app.buttons["Settings"].tap()
        XCTAssertEqual(toggle.value as? String, "1")
        snap("cloud-consent", app)
    }

    func testPairingOffersOnlyConnectorQR() {
        continueAfterFailure = false
        let app = XCUIApplication()
        app.launchEnvironment = ["VOXT_TEST_RESET": "1", "VOXT_TEST_ONBOARDING": "connect"]
        app.launch()
        XCTAssertTrue(app.staticTexts["Connect your Mac"].waitForExistence(timeout: 5))
        XCTAssertFalse(app.buttons["copyPhonePairingKey"].exists)
        XCTAssertFalse(app.buttons["manualPairing"].exists)
        XCTAssertFalse(app.textFields["tokenField"].exists)
        // Offered, but not tapped here: a real Bonjour browse would ask for Local Network access.
        XCTAssertTrue(app.buttons["findMacButton"].exists)
        snap("connector-pairing", app)
    }

    func testConnectorQRThenDictationThroughNativeTailcat() async throws {
        continueAfterFailure = false
        let env = ProcessInfo.processInfo.environment
        let base = try XCTUnwrap(env["VOXT_CONNECTOR_FIXTURE_URL"])
        let (data, response) = try await URLSession.shared.data(from: XCTUnwrap(URL(string: base + "/fixture/qr")))
        XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 200)
        let fields = try JSONDecoder().decode([String: String].self, from: data)
        let qr = try XCTUnwrap(URL(string: XCTUnwrap(fields["url"])))
        let app = XCUIApplication()
        app.launchEnvironment = ["VOXT_TEST_RESET": "1", "VOXT_TEST_ONBOARDING": "connect"]
        app.launch()
        XCTAssertTrue(app.staticTexts["Connect your Mac"].waitForExistence(timeout: 5))
        app.open(qr)
        let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        if springboard.buttons["Open"].waitForExistence(timeout: 1) { springboard.buttons["Open"].tap() }
        // Pairing moves the first-open walkthrough on to the keyboard.
        XCTAssertTrue(app.staticTexts["Add the Voxt keyboard"].waitForExistence(timeout: 40))
        // URL launch can replace the process without the fixture environment.
        // Relaunch without reset also verifies that the exchanged credentials persist.
        app.terminate()
        app.launchEnvironment = [:]
        app.launch()
        XCTAssertTrue(app.staticTexts["Add the Voxt keyboard"].waitForExistence(timeout: 10))
        app.buttons["onboardingNext"].tap()
        XCTAssertTrue(app.buttons["onboardingNext"].waitForExistence(timeout: 5))
    }

    private func launchSettings(practiceField: Bool = false) -> XCUIApplication {
        continueAfterFailure = false
        let app = XCUIApplication()
        app.launchEnvironment = ["VOXT_TEST_RESET": "1"]
        app.launch()
        app.terminate()
        app.launchEnvironment["VOXT_TEST_SERVER"] = #"{"baseURL":"http://127.0.0.1:9","token":"ui-fixture-only"}"#
        if practiceField { app.launchEnvironment["VOXT_TEST_PRACTICE_FIELD"] = "1" }
        app.launch()
        let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        if springboard.buttons["Open"].waitForExistence(timeout: 2) { springboard.buttons["Open"].tap() }
        XCTAssertTrue(app.buttons["Settings"].waitForExistence(timeout: 10))
        app.buttons["Settings"].tap()
        return app
    }

    func testDictionaryAddSearchAndDeletePersists() {
        let app = launchSettings()
        let dictionary = app.buttons["Personal dictionary"]
        XCTAssertTrue(dictionary.waitForExistence(timeout: 5))
        dictionary.tap()
        app.buttons["Add word"].tap()
        let word = app.alerts.textFields.firstMatch
        XCTAssertTrue(word.waitForExistence(timeout: 3))
        word.tap()
        word.typeText("SGLangFixture")
        app.alerts.buttons["Save"].tap()
        let screenshot = XCTAttachment(screenshot: app.screenshot())
        screenshot.name = "dictionary-after-save"
        screenshot.lifetime = .keepAlways
        add(screenshot)
        XCTAssertTrue(app.buttons["SGLangFixture"].waitForExistence(timeout: 3))
        let search = app.searchFields.firstMatch
        search.tap()
        search.typeText("SGLangFixture")
        XCTAssertTrue(app.buttons["SGLangFixture"].exists)
        app.buttons["SGLangFixture"].swipeLeft()
        app.buttons["Delete"].firstMatch.tap()
        XCTAssertFalse(app.buttons["SGLangFixture"].exists)
        if app.buttons["Cancel"].exists { app.buttons["Cancel"].tap() }
        // iOS 26 search replaces navigation actions with a close button
        // after the last row is deleted.
        else if app.buttons["close"].exists { app.buttons["close"].tap() }
        XCTAssertTrue(app.buttons["dictionaryDone"].waitForExistence(timeout: 3))
        app.buttons["dictionaryDone"].tap()
        XCTAssertTrue(app.buttons["Personal dictionary"].waitForExistence(timeout: 3))
        app.buttons["Personal dictionary"].tap()
        XCTAssertFalse(app.buttons["SGLangFixture"].exists)
        snap("dictionary-empty", app)
    }
    func testCorrectionLearningIsOptInAndPersists() {
        let app = launchSettings()
        app.buttons["Personal dictionary"].tap()
        let toggle = app.switches["learnCorrectedWords"]
        XCTAssertTrue(toggle.waitForExistence(timeout: 3))
        XCTAssertEqual(toggle.value as? String, "0")
        toggle.coordinate(withNormalizedOffset: CGVector(dx: 0.92, dy: 0.5)).tap()
        XCTAssertEqual(toggle.value as? String, "1")
        snap("learning-enabled-light", app)
        app.terminate()
        app.launchEnvironment = [:]
        app.launch()
        app.buttons["Settings"].tap()
        app.buttons["appearancePicker"].tap()
        app.buttons["Dark"].tap()
        app.buttons["Personal dictionary"].tap()
        XCTAssertTrue(toggle.waitForExistence(timeout: 3))
        XCTAssertEqual(toggle.value as? String, "1")
        snap("learning-enabled-dark", app)
        toggle.coordinate(withNormalizedOffset: CGVector(dx: 0.92, dy: 0.5)).tap()
        XCTAssertEqual(toggle.value as? String, "0")
    }

    func testPersonalStyleIsOptInAndPersists() {
        let app = launchSettings()
        let entry = app.buttons["Personal style"]
        XCTAssertTrue(entry.waitForExistence(timeout: 3))
        entry.tap()
        let toggle = app.switches["learnPersonalStyle"]
        XCTAssertTrue(toggle.waitForExistence(timeout: 3))
        XCTAssertEqual(toggle.value as? String, "0")
        toggle.coordinate(withNormalizedOffset: CGVector(dx: 0.92, dy: 0.5)).tap()
        XCTAssertEqual(toggle.value as? String, "1")
        XCTAssertTrue(app.staticTexts["No saved examples yet."].exists)
        snap("personal-style-opt-in", app)
        app.terminate()
        app.launchEnvironment = [:]
        app.launch()
        app.buttons["Settings"].tap()
        app.buttons["Personal style"].tap()
        XCTAssertEqual(toggle.value as? String, "1")
        toggle.coordinate(withNormalizedOffset: CGVector(dx: 0.92, dy: 0.5)).tap()
        XCTAssertEqual(toggle.value as? String, "0")
    }

    func testHistorySyncIsOptInAndUnpairStopsIt() {
        let app = launchSettings()
        let toggle = app.switches["historySyncToggle"]
        for _ in 0..<5 {
            if toggle.exists && toggle.isHittable { break }
            app.swipeUp()
        }
        XCTAssertTrue(toggle.exists)
        XCTAssertEqual(toggle.value as? String, "0")
        toggle.coordinate(withNormalizedOffset: CGVector(dx: 0.92, dy: 0.5)).tap()
        XCTAssertEqual(toggle.value as? String, "1")
        XCTAssertTrue(app.buttons["Clear synced history"].exists)
        snap("history-sync-enabled", app)
        app.buttons["Done"].firstMatch.tap()
        app.buttons["History"].tap()
        XCTAssertTrue(app.staticTexts["Sync is enabled. Deletions reach your Mac when both devices can sync."].exists)
        XCTAssertFalse(app.staticTexts["Synced with your Mac. Deleting a record removes it from both devices."].exists)
        app.terminate()
        app.launchEnvironment = [:]
        app.launch()
        app.buttons["Settings"].tap()
        for _ in 0..<5 {
            if toggle.exists && toggle.isHittable { break }
            app.swipeUp()
        }
        XCTAssertEqual(toggle.value as? String, "1")
        for _ in 0..<5 {
            if app.buttons["Unpair"].isHittable { break }
            app.swipeUp()
        }
        app.buttons["Unpair"].tap()
        XCTAssertTrue(app.buttons["confirmUnpair"].firstMatch.waitForExistence(timeout: 5))
        app.buttons["confirmUnpair"].firstMatch.tap()
        // Unpair returns to the walkthrough's Connect step.
        XCTAssertTrue(app.staticTexts["Connect your Mac"].waitForExistence(timeout: 5))
        app.terminate()
        app.launchEnvironment["VOXT_TEST_SERVER"] = #"{"baseURL":"http://127.0.0.1:9","token":"ui-fixture-only"}"#
        app.launch()
        XCTAssertTrue(app.buttons["Settings"].waitForExistence(timeout: 5))
        app.buttons["Settings"].tap()
        for _ in 0..<5 {
            if toggle.exists && toggle.isHittable { break }
            app.swipeUp()
        }
        XCTAssertEqual(toggle.value as? String, "0")
    }

    func testInterfaceLanguageChangesScreensAndPersists() {
        let app = launchSettings()
        let language = app.buttons["interfaceLanguage"]
        XCTAssertTrue(language.waitForExistence(timeout: 3))
        language.tap()
        app.buttons["简体中文"].tap()
        XCTAssertTrue(app.navigationBars["设置"].waitForExistence(timeout: 3))
        XCTAssertTrue(app.buttons["个人词典"].exists)
        snap("interface-language-zh-hans", app)
        app.buttons["完成"].firstMatch.tap()
        snap("interface-language-home-after-change", app)
        XCTAssertTrue(app.buttons["主页"].waitForExistence(timeout: 3), app.debugDescription)
        app.buttons["历史"].tap()
        XCTAssertTrue(app.navigationBars["历史"].exists)
        app.terminate()
        app.launchEnvironment = [:]
        app.launch()
        XCTAssertTrue(app.buttons["主页"].waitForExistence(timeout: 5))
        app.buttons["设置"].tap()
        app.buttons["interfaceLanguage"].tap()
        app.buttons["日本語"].tap()
        XCTAssertTrue(app.navigationBars["設定"].waitForExistence(timeout: 3))
        snap("interface-language-ja", app)
        app.buttons["interfaceLanguage"].tap()
        app.buttons["English"].tap()
        XCTAssertTrue(app.navigationBars["Settings"].waitForExistence(timeout: 3))
    }

    func testInterfaceLanguageReachesKeyboardAndPreservesTypedText() {
        // The simulator's Voxt keyboard is enabled by the keyboard E2E setup.
        let app = launchSettings(practiceField: true)
        app.buttons["interfaceLanguage"].tap()
        app.buttons["繁體中文"].tap()
        app.buttons["完成"].firstMatch.tap()
        let field = app.textFields["tryField"]
        field.tap()
        let mic = app.buttons["開始聽寫"]
        for _ in 0..<6 {
            if mic.waitForExistence(timeout: 2) { break }
            let next = app.buttons["Next keyboard"].firstMatch
            if next.exists { next.tap() }
        }
        XCTAssertTrue(mic.exists, app.debugDescription)
        app.buttons["顯示文字鍵盤"].tap()
        for key in ["h", "e", "l", "l", "o"] { app.buttons["textKey-\(key)"].tap() }
        XCTAssertEqual(field.value as? String, "hello")
        snap("interface-keyboard-zh-hant", app)
        app.buttons["設定"].tap()
        app.buttons["interfaceLanguage"].tap()
        app.buttons["日本語"].tap()
        app.buttons["完了"].firstMatch.tap()
        field.tap()
        let voice = app.buttons["textLayout-voice"]
        if voice.waitForExistence(timeout: 3) { voice.tap() }
        XCTAssertTrue(app.buttons["ディクテーションを開始"].waitForExistence(timeout: 5), app.debugDescription)
        XCTAssertEqual(field.value as? String, "hello")
        snap("interface-keyboard-ja", app)
    }

    func testHistoryRetentionAndAppearancePersist() {
        let app = launchSettings()
        let appearance = app.buttons["appearancePicker"]
        XCTAssertTrue(appearance.waitForExistence(timeout: 5))
        appearance.tap()
        app.buttons["Dark"].tap()
        XCTAssertTrue(appearance.label.contains("Dark"), appearance.label)
        app.buttons["historyRetention"].tap()
        app.buttons["7 days"].tap()
        app.buttons["Done"].firstMatch.tap()
        app.buttons["Settings"].tap()
        XCTAssertTrue(app.buttons["appearancePicker"].label.contains("Dark"))
        XCTAssertTrue(app.buttons["historyRetention"].label.contains("7 days"))
        snap("settings-dark", app)
    }

    func testMicrophoneTurnOffAfterPersistsWithExplanation() {
        let app = launchSettings()
        let readiness = app.buttons["microphoneReadiness"]
        for _ in 0..<6 where !readiness.isHittable { app.swipeUp() }
        XCTAssertTrue(readiness.waitForExistence(timeout: 5))
        XCTAssertTrue(readiness.label.contains("5 minutes"))
        XCTAssertTrue(app.staticTexts["While the microphone is ready, iOS shows the recording indicator. After it turns off, the next dictation opens Voxt first."].exists)
        snap("settings-microphone-default", app)
        readiness.tap()
        app.buttons["Never"].tap()
        XCTAssertTrue(readiness.label.contains("Never"))
        app.terminate()
        app.launchEnvironment["VOXT_TEST_RESET"] = nil
        app.launch()
        XCTAssertTrue(app.buttons["Settings"].waitForExistence(timeout: 10))
        app.buttons["Settings"].tap()
        for _ in 0..<6 where !app.buttons["microphoneReadiness"].isHittable { app.swipeUp() }
        XCTAssertTrue(app.buttons["microphoneReadiness"].label.contains("Never"))
        snap("settings-microphone-never", app)
    }

    func testRegionalLanguagePreferencePersistsAfterRelaunch() {
        let app = launchSettings()
        app.buttons["settingsAdvanced"].tap()
        let variants = app.buttons["Language variants"]
        XCTAssertTrue(variants.waitForExistence(timeout: 5))
        variants.tap()
        app.buttons["variant-en"].tap()
        app.buttons["English (United Kingdom)"].tap()
        XCTAssertTrue(app.buttons["variant-en"].label.contains("United Kingdom"))
        snap("language-variants", app)
        app.terminate()
        app.launchEnvironment = [:]
        app.launch()
        app.buttons["Settings"].tap()
        app.buttons["settingsAdvanced"].tap()
        app.buttons["Language variants"].tap()
        XCTAssertTrue(app.buttons["variant-en"].label.contains("United Kingdom"))
    }

    func testEnabledTextKeyboardsPersistAfterRelaunch() {
        let app = launchSettings()
        let layouts = app.buttons["Text keyboards"]
        XCTAssertTrue(layouts.waitForExistence(timeout: 3))
        layouts.tap()
        let addGerman = app.buttons["addLayout-de-QWERTZ"]
        XCTAssertTrue(addGerman.waitForExistence(timeout: 5))
        addGerman.tap()
        app.buttons["removeLayout-de-QWERTZ"].tap()
        XCTAssertTrue(app.buttons["addLayout-de-QWERTZ"].waitForExistence(timeout: 5))
        app.terminate()
        app.launchEnvironment = [:]
        app.launch()
        app.buttons["Settings"].tap()
        app.buttons["Text keyboards"].tap()
        XCTAssertTrue(app.buttons["addLayout-de-QWERTZ"].waitForExistence(timeout: 5))
        app.buttons["addLayout-de-QWERTZ"].tap()
        XCTAssertTrue(app.buttons["removeLayout-de-QWERTZ"].waitForExistence(timeout: 5))
    }

    private func snap(_ name: String, _ app: XCUIApplication) {
        let image = XCTAttachment(screenshot: app.screenshot())
        image.name = name
        image.lifetime = .keepAlways
        add(image)
    }

}

/// The pairing screen's in-app QR scanner, with the camera replaced by VOXT_TEST_SCAN
/// (one decoded string per line, 1.5 s apart). The simulator has no camera.
@MainActor
final class PairingScannerUITests: XCTestCase {
    private func shot(_ name: String, _ app: XCUIApplication) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    func testScannedCodesGoThroughThePairingLinkParser() {
        continueAfterFailure = false
        let expires = Int(Date().timeIntervalSince1970) + 110
        let link = "voxtmobile://connect?v=1&tailcat=tc-unreachable&port=48932&key=privkey:" + String(repeating: "a", count: 64)
            + "&proof=synthetic-proof&expires=\(expires)"
        let app = XCUIApplication()
        app.launchEnvironment = ["VOXT_TEST_RESET": "1", "VOXT_TEST_ONBOARDING": "connect", "VOXT_TEST_SCAN": "https://example.com/menu\n" + link,
                                 "VOXT_TEST_SCAN_INTERVAL": "6"]
        app.launch()
        XCTAssertTrue(app.otherElements["qrScanner"].waitForExistence(timeout: 5) || app.images["qrScanner"].exists
                      || app.descendants(matching: .any)["qrScanner"].exists)
        XCTAssertTrue(app.buttons["findMacButton"].exists, "same-Wi-Fi pairing stays on the same screen")
        let hint = app.staticTexts["scanHint"]
        XCTAssertTrue(hint.waitForExistence(timeout: 5))
        XCTAssertEqual(hint.label, "This isn't a Voxt pairing code.")
        shot("scanner-wrong-code", app)
        // The Voxt link is opened like a Camera scan; this fixture Mac can't be reached, so
        // pairing fails and the scanner comes back with the message.
        let deadline = Date().addingTimeInterval(60)
        while hint.label != "Couldn't pair. Scan a new QR in Voxt Connector.", Date() < deadline {
            Thread.sleep(forTimeInterval: 0.5)
        }
        XCTAssertEqual(hint.label, "Couldn't pair. Scan a new QR in Voxt Connector.")
        XCTAssertTrue(app.descendants(matching: .any)["qrScanner"].exists)
        shot("scanner-pairing-failed", app)
    }

    func testDeniedCameraOffersSettingsAndKeepsWiFiPairing() {
        continueAfterFailure = false
        let app = XCUIApplication()
        app.launchEnvironment = ["VOXT_TEST_RESET": "1", "VOXT_TEST_ONBOARDING": "connect", "VOXT_TEST_CAMERA": "denied"]
        app.launch()
        XCTAssertTrue(app.staticTexts["cameraDenied"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.buttons["cameraSettingsButton"].exists)
        XCTAssertTrue(app.buttons["findMacButton"].exists)
        XCTAssertFalse(app.descendants(matching: .any)["qrScanner"].exists)
        shot("scanner-camera-denied", app)
    }
}

/// Design review: the Mac status card in every state, light and dark (simulator only).
@MainActor
final class StatusCardGalleryUITests: XCTestCase {
    func testStatusCardGalleryLightAndDark() {
        for appearance in ["light", "dark"] {
            let app = XCUIApplication()
            app.launchEnvironment = ["VOXT_TEST_STATUS_GALLERY": "1", "VOXT_TEST_APPEARANCE": appearance]
            app.launch()
            XCTAssertTrue(app.descendants(matching: .any)["connectedLabel"].firstMatch.waitForExistence(timeout: 10))
            for page in 0..<2 {
                let shot = XCTAttachment(screenshot: app.screenshot())
                shot.name = "status-card-\(appearance)-\(page + 1)"
                shot.lifetime = .keepAlways
                add(shot)
                app.swipeUp()
            }
            app.terminate()
        }
    }
}
