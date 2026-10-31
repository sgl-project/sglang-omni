import XCTest

/// End-to-end tests against a real Voxt Local API server. They run only when the test
/// runner gets these variables (prefix them with TEST_RUNNER_ for xcodebuild):
///   VOXT_E2E_URL    e.g. https://your-mac.tailnet.ts.net:47911
///   VOXT_E2E_TOKEN  pairing token
///   VOXT_E2E_WAV    absolute path to a WAV with speech (stands in for the microphone)
///   VOXT_E2E_EXPECT substring the transcribed text must contain
final class DictationFlowUITests: XCTestCase {
    private var env: [String: String] { ProcessInfo.processInfo.environment }

    override func setUpWithError() throws {
        continueAfterFailure = false
        try XCTSkipIf(env["VOXT_E2E_URL"] == nil, "Set TEST_RUNNER_VOXT_E2E_URL to run end-to-end UI tests")
    }

    /// Saves a full-screen screenshot as an always-kept attachment so the restyle can be
    /// reviewed in light and dark without watching the simulator. Names are stable
    /// ("01-pairing" …) and unique per result bundle.
    private func snap(_ name: String, _ app: XCUIApplication) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    /// A leftover system prompt (for example "Open in Voxt?") blocks every tap.
    private func dismissSystemAlerts() {
        let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        for title in ["Cancel", "Don’t Allow", "OK"] where springboard.alerts.buttons[title].exists {
            springboard.alerts.buttons[title].tap()
        }
    }

    private func launchPairedApp() throws -> XCUIApplication {
        try launchAppPairedByLink()
    }

    /// A dictation into the app's own text field goes through the Mac, lands in the field and in History.
    func testInAppDictationGoesThroughMac() throws {
        try enableVoxtKeyboard()
        let app = try launchPairedApp()
        snap("02-home-idle", app)
        let value = try dictateIntoPracticeField(app)
        if let expected = env["VOXT_E2E_EXPECT"] {
            XCTAssertTrue(value.contains(expected), "Field holds: \(value)")
        }
        print("UI E2E in-app text: \(value)")

        // The keyboard covers the tab bar; Home's bar closes it.
        app.buttons["Hide keyboard"].tap()
        app.tabBars.buttons["History"].tap()
        let result = app.staticTexts["historyText"].firstMatch
        XCTAssertTrue(result.waitForExistence(timeout: 10), "History has no entry for the dictation")
        XCTAssertEqual(result.label, value.trimmingCharacters(in: .whitespaces))
        snap("04-history", app)
    }

    /// Uses connector QR for transport tests, injected localhost config for UI fixtures. Avoids typeText, whose
    /// hardware-key events make iOS hide the software keyboard for the rest of the run.
    private func launchAppPairedByLink(audioFile: String? = nil, typingLayouts: [String] = [], opensRecorder: Bool = true) throws -> XCUIApplication {
        dismissSystemAlerts()
        let app = XCUIApplication()
        app.launchEnvironment = [
            "VOXT_TEST_AUDIO_FILE": try audioFile ?? XCTUnwrap(env["VOXT_E2E_WAV"]),
            // Home has no practice field; this test-only one stands in for a text field.
            "VOXT_TEST_PRACTICE_FIELD": "1",
        ]
        if env["VOXT_E2E_USE_SAVED_PAIRING"] != "1" { app.launchEnvironment["VOXT_TEST_RESET"] = "1" }
        app.launch()
        if env["VOXT_E2E_USE_SAVED_PAIRING"] == "1" {
            XCTAssertTrue(app.descendants(matching: .any)["connectedLabel"].waitForExistence(timeout: 45))
            if opensRecorder, app.buttons["showRecorder"].exists { app.buttons["showRecorder"].tap() }
            return app
        }

        if let raw = env["VOXT_E2E_PAIRING_LINK"] {
            app.open(try XCTUnwrap(URL(string: raw)))
        } else {
            app.terminate()
            app.launchEnvironment["VOXT_TEST_SERVER"] = String(data: try JSONSerialization.data(withJSONObject: ["baseURL": try XCTUnwrap(env["VOXT_E2E_URL"]), "token": try XCTUnwrap(env["VOXT_E2E_TOKEN"])]), encoding: .utf8)!
            app.launch()
        }
        let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        let connected = app.descendants(matching: .any)["connectedLabel"]
        let deadline = Date().addingTimeInterval(30)
        while Date() < deadline, !connected.exists {
            if springboard.buttons["Open"].exists { springboard.buttons["Open"].tap() }
            Thread.sleep(forTimeInterval: 0.5)
        }
        if !connected.exists { print("After pairing link:\n\(app.debugDescription)") }
        XCTAssertTrue(connected.exists, "Pairing link did not connect to Voxt")
        if !typingLayouts.isEmpty {
            app.buttons["Settings"].tap()
            let languages = app.buttons["Text keyboards"]
            for _ in 0..<6 where !languages.isHittable { app.swipeUp() }
            languages.tap()
            XCTAssertTrue(app.navigationBars["Keyboard languages"].waitForExistence(timeout: 5))
            for id in typingLayouts {
                let add = app.buttons["addLayout-" + id]
                if add.exists { add.tap() }
                XCTAssertTrue(app.buttons["removeLayout-" + id].exists)
            }
            app.navigationBars["Keyboard languages"].buttons["Done"].tap()
            app.navigationBars["Settings"].buttons["Done"].tap()
        }
        // Home has no in-app recorder; tests that still ask for it skip that step.
        if opensRecorder, app.buttons["showRecorder"].exists { app.buttons["showRecorder"].tap() }
        return app
    }

    private func switchEditingAudio(_ name: String) throws {
        let control = try XCTUnwrap(env["VOXT_EDIT_AUDIO_CONTROL_URL"])
        let url = try XCTUnwrap(URL(string: control.replacingOccurrences(of: "/edit-audio", with: "/" + name + "-audio")))
        let finished = expectation(description: "Switch spoken fixture")
        URLSession.shared.dataTask(with: url) { _, response, error in
            XCTAssertNil(error)
            XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 200)
            finished.fulfill()
        }.resume()
        wait(for: [finished], timeout: 10)
    }

    func testKeyboardEditsLatestResultWithoutDeletingOnCancel() throws {
        try switchEditingAudio("dictate")
        try enableVoxtKeyboard()
        let app = try launchAppPairedByLink()
        let field = app.textFields["tryField"]
        field.tap()
        let mic = app.buttons["Start dictation"]
        let next = app.buttons["Next keyboard"].firstMatch
        for _ in 0..<6 {
            if mic.waitForExistence(timeout: 2) { break }
            if next.exists { next.tap() }
        }
        XCTAssertTrue(mic.exists)
        snap("keyboard-voice-idle", app)
        mic.tap()
        XCTAssertTrue(app.buttons["Stop dictation"].waitForExistence(timeout: 10))
        app.buttons["Stop dictation"].tap()
        let inserted = XCTNSPredicateExpectation(predicate: NSPredicate(format: "value CONTAINS %@", "Friday"), object: field)
        XCTAssertEqual(XCTWaiter.wait(for: [inserted], timeout: 180), .completed)
        let original = try XCTUnwrap(field.value as? String)
        try switchEditingAudio("edit")
        // The pen arms "Editing what you just said", then the mic records the instruction.
        let pen = app.buttons["keyboardPen"]
        pen.tap()
        XCTAssertTrue(app.staticTexts["Editing what you just said"].waitForExistence(timeout: 5))
        mic.tap()
        XCTAssertTrue(app.buttons["Stop dictation"].waitForExistence(timeout: 10))
        XCTAssertEqual(field.value as? String, original)
        app.buttons["keyboardCancelComposition"].tap()
        XCTAssertTrue(app.buttons["Start dictation"].waitForExistence(timeout: 5))
        XCTAssertEqual(field.value as? String, original)
        XCTAssertTrue(app.staticTexts["Tap to speak"].exists, "the pen applies to one recording only")
        pen.tap()
        XCTAssertTrue(app.staticTexts["Editing what you just said"].waitForExistence(timeout: 5))
        mic.tap()
        XCTAssertTrue(app.buttons["Stop dictation"].waitForExistence(timeout: 10))
        app.buttons["Stop dictation"].tap()
        let changed = XCTNSPredicateExpectation(predicate: NSPredicate(format: "value CONTAINS %@", "Thursday"), object: field)
        XCTAssertEqual(XCTWaiter.wait(for: [changed], timeout: 180), .completed)
        let rewritten = try XCTUnwrap(field.value as? String)
        XCTAssertFalse(rewritten.contains("Friday"), rewritten)
        XCTAssertEqual(rewritten.components(separatedBy: "Alice").count - 1, 1, rewritten)
        XCTAssertTrue(rewritten.contains("50") || rewritten.lowercased().contains("fifty"), rewritten)
        snap("keyboard-voice-edited", app)
    }

    func testLatestResultCanBeEditedWithOneButton() throws {
        try switchEditingAudio("dictate")
        let control = try XCTUnwrap(env["VOXT_EDIT_AUDIO_CONTROL_URL"])
        let app = try launchAppPairedByLink()
        let original = recordAndReadResult(app)
        XCTAssertTrue(original.contains("Alice"), original)
        XCTAssertTrue(original.contains("50") || original.lowercased().contains("fifty"), original)
        XCTAssertTrue(original.contains("Friday"), original)
        snap("edit-latest-before", app)

        let switched = expectation(description: "Use editing instruction for the next recording")
        URLSession.shared.dataTask(with: try XCTUnwrap(URL(string: control))) { _, response, error in
            XCTAssertNil(error)
            XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 200)
            switched.fulfill()
        }.resume()
        wait(for: [switched], timeout: 10)

        let edit = app.buttons["editLatestResult"]
        XCTAssertTrue(edit.waitForExistence(timeout: 3), "The result needs a one-button voice edit")
        edit.tap()
        let stop = app.buttons["recordButton"]
        let recording = XCTNSPredicateExpectation(predicate: NSPredicate(format: "label == %@", "Stop recording"), object: stop)
        XCTAssertEqual(XCTWaiter.wait(for: [recording], timeout: 5), .completed)
        stop.tap()
        let result = app.staticTexts["historyText"].firstMatch
        let changed = XCTNSPredicateExpectation(predicate: NSPredicate(format: "label CONTAINS %@", "Thursday"), object: result)
        XCTAssertEqual(XCTWaiter.wait(for: [changed], timeout: 180), .completed)
        XCTAssertTrue(result.label.contains("Alice"), result.label)
        XCTAssertTrue(result.label.contains("50") || result.label.lowercased().contains("fifty"), result.label)
        XCTAssertFalse(result.label.contains("Friday"), result.label)
        app.tabBars.buttons["History"].tap()
        snap("edit-latest-after", app)
    }

    func testTranslationUsesEnglishAndRemovesFillers() throws {
        let app = try launchAppPairedByLink()
        app.buttons["compositionMode-translate"].tap()
        let result = recordAndReadResult(app)
        XCTAssertTrue(result.lowercased().contains("tomorrow"), result)
        XCTAssertTrue(result.contains("3"), result)
        XCTAssertTrue(result.lowercased().contains("release"), result)
        XCTAssertFalse(result.lowercased().hasPrefix("um"), result)
        XCTAssertFalse(result.lowercased().hasPrefix("well"), result)
        snap("08-translated-result", app)
    }

    func testHistorySyncImportsAndDeletesRemoteRecord() throws {
        try XCTSkipIf(env["VOXT_E2E_HISTORY_TEXT"] == nil, "Run with the history-sync launcher")
        let fixture = try XCTUnwrap(env["VOXT_E2E_HISTORY_TEXT"], "The live launcher supplies its own disposable history record")
        let app = try launchAppPairedByLink()
        app.buttons["Settings"].tap()
        let toggle = app.switches["historySyncToggle"]
        for _ in 0..<6 {
            if toggle.exists && toggle.isHittable { break }
            app.swipeUp()
        }
        XCTAssertEqual(toggle.value as? String, "0")
        toggle.coordinate(withNormalizedOffset: CGVector(dx: 0.92, dy: 0.5)).tap()
        let status = app.staticTexts["historySyncStatus"]
        let synced = XCTNSPredicateExpectation(predicate: NSPredicate(format: "label == %@", "Up to date"), object: status)
        XCTAssertEqual(XCTWaiter.wait(for: [synced], timeout: 30), .completed)
        app.buttons["Done"].firstMatch.tap()
        app.tabBars.buttons["History"].tap()
        let search = app.searchFields.firstMatch
        search.tap()
        search.typeText(fixture + "\n")
        let record = app.staticTexts.matching(identifier: "historyText").matching(NSPredicate(format: "label == %@", fixture)).firstMatch
        XCTAssertTrue(record.waitForExistence(timeout: 5))
        snap("history-sync-imported", app)
        record.coordinate(withNormalizedOffset: CGVector(dx: 0.9, dy: 0.5))
            .press(forDuration: 0.05, thenDragTo: record.coordinate(withNormalizedOffset: CGVector(dx: 0.4, dy: 0.5)))
        let delete = app.buttons["Delete"].firstMatch
        XCTAssertTrue(delete.waitForExistence(timeout: 3))
        delete.tap()
        XCTAssertFalse(record.exists)
        snap("history-sync-deleted", app)
        // The launcher independently checks this exact UUID is a server tombstone.
    }

    func testHelpMeWriteProducesTheRequestedEmail() throws {
        try XCTSkipIf(env["VOXT_E2E_FIXTURES"] == nil, "Set TEST_RUNNER_VOXT_E2E_FIXTURES for the spoken writing instructions")
        let fixtures = try XCTUnwrap(env["VOXT_E2E_FIXTURES"])
        let app = try launchAppPairedByLink(audioFile: fixtures + "/write.wav")
        let mode = app.buttons["compositionMode-write"]
        if !mode.isHittable { app.buttons["compositionMode-edit"].swipeLeft() }
        mode.tap()
        let result = recordAndReadResult(app)
        XCTAssertTrue(result.contains("Emma"), result)
        XCTAssertTrue(result.contains("Friday"), result)
        XCTAssertTrue(result.contains("Will"), result)
        XCTAssertFalse(result.lowercased().hasPrefix("write a short email"), result)
        snap("09-written-email", app)
    }

    func testVoiceEditPreservesNameAmountAndDeadline() throws {
        try XCTSkipIf(env["VOXT_E2E_FIXTURES"] == nil, "Set TEST_RUNNER_VOXT_E2E_FIXTURES for the spoken editing instructions")
        let fixtures = try XCTUnwrap(env["VOXT_E2E_FIXTURES"])
        let app = try launchAppPairedByLink(audioFile: fixtures + "/edit.wav")
        let field = app.textFields["tryField"]
        field.tap()
        field.typeText("hey Alice, pay me $50 by Friday. thanks!")
        app.buttons["Hide keyboard"].tap()
        // Dismissing the keyboard can leave this row at y=52 under the
        // navigation bar, where XCTest's tap succeeds without changing the mode.
        let editMode = app.buttons["compositionMode-edit"]
        for _ in 0..<3 where editMode.frame.minY < app.navigationBars.firstMatch.frame.maxY {
            app.scrollViews.firstMatch.swipeDown()
        }
        XCTAssertGreaterThanOrEqual(editMode.frame.minY, app.navigationBars.firstMatch.frame.maxY)
        editMode.tap()
        let editHint = app.staticTexts["Enter text in the practice field, then say how to rewrite it"]
        if !editHint.waitForExistence(timeout: 3) {
            print("After selecting Edit:\n\(app.debugDescription)")
            snap("edit-mode-selection-failed", app)
        }
        XCTAssertTrue(editHint.exists, "The main app must enter Edit before recording")
        let result = recordAndReadResult(app)
        XCTAssertTrue(result.contains("Alice"), result)
        XCTAssertTrue(result.contains("50") || result.lowercased().contains("fifty"), result)
        XCTAssertTrue(result.contains("Friday"), result)
        XCTAssertFalse(result.lowercased().hasPrefix("make the selected"), result)
        snap("10-edited-result", app)
    }

    private func recordAndReadResult(_ app: XCUIApplication) -> String {
        let record = app.buttons["recordButton"]
        record.tap()
        let recording = XCTNSPredicateExpectation(predicate: NSPredicate(format: "label == %@", "Stop recording"), object: record)
        XCTAssertEqual(XCTWaiter.wait(for: [recording], timeout: 5), .completed)
        record.tap()
        let result = app.staticTexts["historyText"].firstMatch
        XCTAssertTrue(result.waitForExistence(timeout: 180), "No composed result returned")
        let historyTab = app.tabBars.buttons["History"]
        let selected = XCTNSPredicateExpectation(predicate: NSPredicate(format: "isSelected == true"), object: historyTab)
        XCTAssertEqual(XCTWaiter.wait(for: [selected], timeout: 5), .completed)
        // A result exists before the tab's crossfade finishes.
        // An action on the selected tab waits for app idleness before later screenshots.
        historyTab.tap()
        return result.label
    }

    func testKeyboardDictatesIntoTextField() throws {
        try enableVoxtKeyboard()
        let app = try launchAppPairedByLink()
        let value = try dictateIntoPracticeField(app)
        if let expected = env["VOXT_E2E_EXPECT"] {
            XCTAssertTrue(value.contains(expected), "Field holds: \(value)")
        } else {
            XCTAssertFalse(value.isEmpty)
        }
        print("UI E2E keyboard text: \(value)")
    }

    /// A fresh simulator shows two one-time iOS tips over the keyboard: "Quickly Change Keyboards"
    /// after the first switch, and the slide-to-type tip over the first system keyboard. Continue closes either.
    private func dismissKeyboardTip(_ app: XCUIApplication) {
        let tip = app.staticTexts.matching(NSPredicate(format: "label == 'Quickly Change Keyboards' OR label BEGINSWITH 'Speed up your typing by sliding'")).firstMatch
        let close = app.buttons["Continue"].firstMatch
        if tip.waitForExistence(timeout: 1), close.exists { close.tap() }
    }

    /// Switches the practice field to the Voxt keyboard, records once and returns the text the Mac sent.
    private func dictateIntoPracticeField(_ app: XCUIApplication) throws -> String {
        let field = app.textFields["tryField"]
        XCTAssertTrue(field.waitForExistence(timeout: 5))
        field.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
        let mic = app.buttons["Start dictation"]
        let keyboard = app.keyboards.firstMatch
        let shown = Date().addingTimeInterval(8)
        // A third-party keyboard isn't reported as a Keyboard element, so stop at either one.
        while Date() < shown, !mic.exists, !(keyboard.exists && keyboard.frame.minY < app.windows.firstMatch.frame.height) {
            Thread.sleep(forTimeInterval: 0.3)
        }
        let nextKeyboard = app.buttons.matching(NSPredicate(format: "label == 'Next keyboard' OR identifier == 'Next keyboard'")).firstMatch
        var switches = 0
        while !mic.waitForExistence(timeout: 2), switches < 6, nextKeyboard.exists {
            print("Keyboard before switch \(switches): \(nextKeyboard.value ?? "nil")")
            nextKeyboard.tap()
            dismissKeyboardTip(app)
            switches += 1
        }
        if !mic.exists, nextKeyboard.exists {
            // The globe's long-press menu lists every enabled keyboard.
            nextKeyboard.press(forDuration: 1.2)
            let voxtChoice = app.descendants(matching: .any).matching(NSPredicate(format: "label == 'Voxt'")).firstMatch
            if voxtChoice.waitForExistence(timeout: 3) {
                voxtChoice.tap()
            } else {
                print("Keyboard picker:\n\(app.debugDescription)")
            }
            _ = mic.waitForExistence(timeout: 5)
        }
        if !mic.exists {
            print("Keyboard state:\n\(app.debugDescription)")
        }
        XCTAssertTrue(mic.exists, "Voxt keyboard did not appear")
        snap("06-keyboard-idle", app)

        mic.tap()
        XCTAssertTrue(app.buttons["Stop dictation"].waitForExistence(timeout: 10), "Keyboard never saw recording start")
        snap("07-keyboard-recording", app)
        app.buttons["Stop dictation"].tap()

        let deadline = Date().addingTimeInterval(180)
        var value = ""
        // An empty field reports its placeholder as its value.
        let placeholders = ["Switch to the Voxt keyboard and tap the mic", field.placeholderValue ?? ""]
        while Date() < deadline {
            value = field.value as? String ?? ""
            if !value.isEmpty, !placeholders.contains(value) { break }
            Thread.sleep(forTimeInterval: 1)
        }
        return value
    }

    func testKeyboardLearnsCorrectedSpellingAndCanUndoIt() throws {
        let fixtures = try XCTUnwrap(env["VOXT_E2E_FIXTURES"])
        try enableVoxtKeyboard()
        let app = try launchAppPairedByLink(audioFile: fixtures + "/variants.wav")
        app.buttons["Settings"].tap()
        app.buttons["Personal dictionary"].tap()
        let learn = app.switches["learnCorrectedWords"]
        XCTAssertTrue(learn.waitForExistence(timeout: 3))
        XCTAssertEqual(learn.value as? String, "0")
        learn.coordinate(withNormalizedOffset: CGVector(dx: 0.92, dy: 0.5)).tap()
        app.buttons["dictionaryDone"].tap()
        app.buttons["Done"].firstMatch.tap()
        let field = app.textFields["tryField"]
        field.tap()
        let mic = app.buttons["Start dictation"]
        let next = app.buttons["Next keyboard"].firstMatch
        for _ in 0..<6 {
            if mic.waitForExistence(timeout: 2) { break }
            if next.exists { next.tap() }
        }
        XCTAssertTrue(mic.exists)
        mic.tap()
        XCTAssertTrue(app.buttons["Stop dictation"].waitForExistence(timeout: 10))
        app.buttons["Stop dictation"].tap()
        let inserted = XCTNSPredicateExpectation(predicate: NSPredicate(format: "value ENDSWITH %@", "gray."), object: field)
        XCTAssertEqual(XCTWaiter.wait(for: [inserted], timeout: 180), .completed)
        app.buttons["Show text keyboard"].tap()
        for _ in 0..<5 { app.buttons["textKey-delete"].tap() }
        for letter in ["g", "r", "e", "y"] { app.buttons["textKey-\(letter)"].tap() }
        app.buttons["textKey-numbers"].tap()
        app.buttons["textKey-."].tap()
        let learned = app.staticTexts["learnedWordNotice"]
        XCTAssertTrue(learned.waitForExistence(timeout: 90))
        XCTAssertTrue(learned.label.contains("grey"), learned.label)
        snap("12-learned-word", app)
        app.buttons["Hide keyboard"].tap()
        app.buttons["Settings"].tap()
        app.buttons["Personal dictionary"].tap()
        XCTAssertTrue(app.buttons["grey"].waitForExistence(timeout: 3))
        XCTAssertTrue(app.buttons["undoLearned-grey"].exists)
        app.buttons["undoLearned-grey"].tap()
        XCTAssertFalse(app.buttons["grey"].exists)
        snap("13-learning-undone", app)
        app.buttons["dictionaryDone"].tap()
        app.buttons["Done"].firstMatch.tap()
        let estimate = app.staticTexts["usageTime"]
        for _ in 0..<5 {
            if estimate.exists && estimate.isHittable { break }
            app.swipeUp()
        }
        XCTAssertTrue(estimate.exists)
        XCTAssertTrue(estimate.label.contains("saved"), estimate.label)
        snap("14-measured-time-estimate", app)
    }

    private func dismissPersonalStyleSettings(_ app: XCUIApplication) {
        app.buttons["personalStyleDone"].tap()
        let styleClosed = XCTNSPredicateExpectation(predicate: NSPredicate(format: "exists == false"),
                                                   object: app.buttons["personalStyleDone"])
        XCTAssertEqual(XCTWaiter.wait(for: [styleClosed], timeout: 5), .completed)
        app.buttons["Done"].firstMatch.tap()
        let settingsClosed = XCTNSPredicateExpectation(predicate: NSPredicate(format: "exists == false"),
                                                      object: app.buttons["Personal style"])
        XCTAssertEqual(XCTWaiter.wait(for: [settingsClosed], timeout: 5), .completed)
    }

    func testNewRecordingDoesNotReviewPreviousDictationEdits() throws {
        let fixtures = try XCTUnwrap(env["VOXT_E2E_FIXTURES"])
        try enableVoxtKeyboard()
        let app = try launchAppPairedByLink(audioFile: fixtures + "/style.wav")
        app.buttons["Settings"].tap()
        app.buttons["Personal style"].tap()
        let learn = app.switches["learnPersonalStyle"]
        XCTAssertTrue(learn.waitForExistence(timeout: 3))
        learn.coordinate(withNormalizedOffset: CGVector(dx: 0.92, dy: 0.5)).tap()
        dismissPersonalStyleSettings(app)
        let field = app.textFields["tryField"]
        field.tap()
        let mic = app.buttons["Start dictation"]
        let next = app.buttons["Next keyboard"].firstMatch
        for _ in 0..<6 {
            if mic.waitForExistence(timeout: 2) { break }
            if next.exists { next.tap() }
        }
        XCTAssertTrue(mic.exists)
        mic.tap()
        XCTAssertTrue(app.buttons["Stop dictation"].waitForExistence(timeout: 10))
        app.buttons["Stop dictation"].tap()
        let inserted = XCTNSPredicateExpectation(predicate: NSPredicate(format: "value ENDSWITH[c] %@", "Thanks."), object: field)
        XCTAssertEqual(XCTWaiter.wait(for: [inserted], timeout: 180), .completed)
        mic.tap()
        XCTAssertTrue(app.buttons["Stop dictation"].waitForExistence(timeout: 10))
        app.buttons["Show text keyboard"].tap()
        for _ in 0..<7 { app.buttons["textKey-delete"].tap() }
        for letter in ["m", "a", "n", "y", "space", "t", "h", "a", "n", "k", "s"] {
            app.buttons["textKey-\(letter)"].tap()
        }
        app.buttons["textKey-numbers"].tap()
        app.buttons["textKey-."].tap()
        // Let the normal 1-second observation timer and 1.5-second stability
        // interval run while a new recording is active.
        Thread.sleep(forTimeInterval: 3)
        XCTAssertFalse(app.staticTexts["styleReviewAfter"].exists,
                       "A review from the prior insertion must not cover the active recording controls")
        app.buttons["textLayout-voice"].tap()
        XCTAssertTrue(app.buttons["Stop dictation"].exists)
        // The host app also has a Cancel button; use the keyboard's lower one.
        let cancel = app.buttons["keyboardCancelComposition"]
        XCTAssertTrue(cancel.isHittable)
        snap("personal-style-active-recording-controls", app)
        cancel.tap()
        XCTAssertTrue(app.buttons["Start dictation"].waitForExistence(timeout: 5))
    }

    func testKeyboardOffersPersonalStyleForReviewAndCanRemoveIt() throws {
        let fixtures = try XCTUnwrap(env["VOXT_E2E_FIXTURES"])
        try enableVoxtKeyboard()
        let app = try launchAppPairedByLink(audioFile: fixtures + "/style.wav")
        app.buttons["Settings"].tap()
        app.buttons["Personal style"].tap()
        let learn = app.switches["learnPersonalStyle"]
        XCTAssertTrue(learn.waitForExistence(timeout: 3))
        learn.coordinate(withNormalizedOffset: CGVector(dx: 0.92, dy: 0.5)).tap()
        dismissPersonalStyleSettings(app)
        let field = app.textFields["tryField"]
        field.tap()
        let mic = app.buttons["Start dictation"]
        let next = app.buttons["Next keyboard"].firstMatch
        for _ in 0..<6 {
            if mic.waitForExistence(timeout: 2) { break }
            if next.exists { next.tap() }
        }
        XCTAssertTrue(mic.exists)
        mic.tap()
        XCTAssertTrue(app.buttons["Stop dictation"].waitForExistence(timeout: 10))
        app.buttons["Stop dictation"].tap()
        let inserted = XCTNSPredicateExpectation(predicate: NSPredicate(format: "value ENDSWITH[c] %@", "Thanks."), object: field)
        XCTAssertEqual(XCTWaiter.wait(for: [inserted], timeout: 180), .completed)
        let original = try XCTUnwrap(field.value as? String)
        app.buttons["Show text keyboard"].tap()
        for _ in 0..<7 { app.buttons["textKey-delete"].tap() }
        for letter in ["m", "a", "n", "y", "space", "t", "h", "a", "n", "k", "s"] {
            app.buttons["textKey-\(letter)"].tap()
        }
        app.buttons["textKey-numbers"].tap()
        app.buttons["textKey-."].tap()
        let review = app.staticTexts["styleReviewAfter"]
        XCTAssertTrue(review.waitForExistence(timeout: 8))
        XCTAssertEqual(app.staticTexts["styleReviewBefore"].label, original)
        XCTAssertTrue(review.label.hasSuffix("many thanks."))
        let title = app.staticTexts["Keep this wording?"]
        XCTAssertTrue(title.exists)
        XCTAssertGreaterThanOrEqual(title.frame.minY, app.buttons["Hide keyboard"].frame.maxY - 2,
                                    "The review heading must be below the host's keyboard-dismiss bar")
        snap("personal-style-review", app)
        app.buttons["keepPersonalStyle"].tap()
        XCTAssertFalse(review.exists)
        app.buttons["Hide keyboard"].tap()
        app.buttons["Settings"].tap()
        app.buttons["Personal style"].tap()
        let saved = app.staticTexts["savedStyleExample"]
        XCTAssertTrue(saved.waitForExistence(timeout: 3))
        XCTAssertTrue(saved.label.hasSuffix("many thanks."))
        snap("personal-style-saved", app)
        dismissPersonalStyleSettings(app)
        // Same recorded speech before and after Keep: the changed wording must reach
        // the real model through the phone's stored examples, not just a test API body.
        let personalized = recordAndReadResult(app)
        XCTAssertTrue(personalized.lowercased().contains("many thanks"), personalized)
        XCTAssertTrue(personalized.contains("Alice"), personalized)
        XCTAssertTrue(personalized.contains("three") || personalized.contains("3"), personalized)
        snap("personal-style-next-dictation", app)
        app.tabBars.buttons["Home"].tap()
        app.buttons["Settings"].tap()
        app.buttons["Personal style"].tap()
        learn.coordinate(withNormalizedOffset: CGVector(dx: 0.92, dy: 0.5)).tap()
        XCTAssertEqual(learn.value as? String, "0")
        XCTAssertTrue(saved.exists, "Opting out must retain the saved example locally")
        dismissPersonalStyleSettings(app)
        let disabled = recordAndReadResult(app)
        XCTAssertFalse(disabled.lowercased().contains("many thanks"), disabled)
        XCTAssertTrue(disabled.lowercased().contains("thanks"), disabled)
        app.tabBars.buttons["Home"].tap()
        app.buttons["Settings"].tap()
        app.buttons["Personal style"].tap()
        saved.coordinate(withNormalizedOffset: CGVector(dx: 0.9, dy: 0.5))
            .press(forDuration: 0.05, thenDragTo: saved.coordinate(withNormalizedOffset: CGVector(dx: 0.3, dy: 0.5)))
        app.buttons["Delete"].firstMatch.tap()
        XCTAssertFalse(saved.exists)
        XCTAssertTrue(app.staticTexts["No saved examples yet."].exists)
    }

    func testZhuyinOfflineCandidatesAndSentencePreserveCommittedWords() throws {
        try enableVoxtKeyboard()
        let app = try launchAppPairedByLink(typingLayouts: ["ja-flick", "zh-Hant-zhuyin"], opensRecorder: false)
        let field = app.textFields["tryField"]
        field.tap()
        let mic = app.buttons["Start dictation"]
        for _ in 0..<6 {
            if mic.waitForExistence(timeout: 2) { break }
            let next = app.buttons["Next keyboard"].firstMatch
            if next.exists { next.tap() }
        }
        app.buttons["Show text keyboard"].tap()
        let layout = app.buttons["textLayout-zh-Hant-zhuyin"]
        // Every enabled layout is visible in the fixed toolbar.
        XCTAssertTrue(layout.waitForExistence(timeout: 3))
        layout.tap()
        XCTAssertLessThanOrEqual(layout.frame.maxX, app.frame.maxX - 8, "The active layout must remain fully visible after switching")
        for key in ["ㄋ", "ㄧ", "ˇ"] { app.buttons["zhuyinKey-\(key)"].tap() }
        let alternative = app.buttons["zhuyinCandidate-妳"]
        XCTAssertTrue(alternative.waitForExistence(timeout: 10))
        snap("zhuyin-alternatives", app)
        alternative.tap()
        XCTAssertEqual(field.value as? String, "妳")
        for key in ["ㄋ", "ㄧ", "ˇ", "ㄏ", "ㄠ", "ˇ"] { app.buttons["zhuyinKey-\(key)"].tap() }
        let sentence = app.buttons["zhuyinCandidate-你好"]
        XCTAssertTrue(sentence.waitForExistence(timeout: 10))
        snap("zhuyin-sentence-candidates", app)
        sentence.tap()
        XCTAssertEqual(field.value as? String, "妳你好")
        app.buttons["zhuyinKey-ㄋ"].tap()
        app.buttons["zhuyinDelete"].tap()
        XCTAssertEqual(field.value as? String, "妳你好")
        app.buttons["zhuyinKey-ㄊ"].tap()
        app.buttons["zhuyinKey-ㄚ"].tap()
        app.buttons["zhuyinSpace"].tap()
        XCTAssertTrue(app.buttons["zhuyinCandidate-他"].waitForExistence(timeout: 10))
        app.buttons["zhuyinReturn"].tap()
        XCTAssertEqual(field.value as? String, "妳你好他")
        app.buttons["textLayout-voice"].tap()
        XCTAssertTrue(app.buttons["Start dictation"].waitForExistence(timeout: 3))
        snap("zhuyin-committed", app)
        app.buttons["voiceLayout-zh-Hant-zhuyin"].tap()
        XCTAssertTrue(app.buttons["zhuyinKey-ㄋ"].waitForExistence(timeout: 3))
        app.otherElements["zhuyinLayoutToolbar"].swipeLeft()
        XCTAssertTrue(app.buttons["Start dictation"].waitForExistence(timeout: 3), "Swiping past the final text layout returns to voice")
    }

    func testJapaneseDictionaryCandidatesReplaceOnlyTheCurrentReading() throws {
        try enableVoxtKeyboard()
        let app = try launchAppPairedByLink(typingLayouts: ["ja-flick", "zh-Hant-zhuyin"], opensRecorder: false)
        let field = app.textFields["tryField"]
        field.tap()
        let mic = app.buttons["Start dictation"]
        for _ in 0..<6 {
            if mic.waitForExistence(timeout: 2) { break }
            let next = app.buttons["Next keyboard"].firstMatch
            if next.exists { next.tap() }
        }
        app.buttons["Show text keyboard"].tap()
        app.buttons["textLayout-ja-flick"].tap()
        func flickLeft(_ id: String) {
            let key = app.buttons["kanaKey-\(id)"]
            key.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
                .press(forDuration: 0.05, thenDragTo: key.coordinate(withNormalizedOffset: CGVector(dx: 0.05, dy: 0.5)))
        }
        app.buttons["kanaKey-ka"].tap()
        flickLeft("a")
        XCTAssertEqual(field.value as? String, "かい")
        flickLeft("ka")
        XCTAssertEqual(field.value as? String, "かいき")
        snap("japanese-before-modifier", app)
        app.buttons["japaneseModifier"].tap()
        XCTAssertEqual(field.value as? String, "かいぎ")
        let meeting = app.buttons["japaneseCandidate-会議"]
        XCTAssertTrue(meeting.waitForExistence(timeout: 10), "The bundled dictionary must produce real kanji")
        snap("japanese-kanji-candidates", app)
        meeting.tap()
        XCTAssertEqual(field.value as? String, "会議")
        XCTAssertFalse(meeting.exists)
        app.buttons["kanaKey-ka"].tap()
        flickLeft("a")
        app.buttons["kanaKey-ta"].tap()
        app.buttons["japaneseModifier"].tap()
        let wa = app.buttons["kanaKey-wa"]
        wa.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
            .press(forDuration: 0.05, thenDragTo: wa.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.05)))
        XCTAssertEqual(field.value as? String, "会議かいだん")
        XCTAssertTrue(app.buttons["japaneseCandidate-階段"].waitForExistence(timeout: 10))
        let talks = app.buttons["japaneseCandidate-会談"]
        XCTAssertTrue(talks.exists, "Homophones must be available as separate choices")
        talks.tap()
        XCTAssertEqual(field.value as? String, "会議会談", "Selection must preserve the previously committed word")
        app.buttons["kanaKey-sa"].tap()
        XCTAssertEqual(field.value as? String, "会議会談さ")
        app.buttons["japaneseDelete"].tap()
        XCTAssertEqual(field.value as? String, "会議会談")
        XCTAssertFalse(app.buttons["japaneseCandidate-さ"].exists)
        snap("japanese-kanji-committed", app)
    }

    func testIncomingDictationDoesNotReplaceJapaneseMarkedText() throws {
        try enableVoxtKeyboard()
        let app = try launchAppPairedByLink(typingLayouts: ["ja-flick", "zh-Hant-zhuyin"])
        let field = app.textFields["tryField"]
        // Opening the recorder scrolls Home down to the mic controls (since 4013291),
        // which takes the practice field off screen; bring it back before tapping.
        for _ in 0..<3 where !field.isHittable { app.scrollViews.firstMatch.swipeDown() }
        XCTAssertTrue(field.isHittable)
        field.tap()
        let mic = app.buttons["Start dictation"]
        for _ in 0..<6 {
            if mic.waitForExistence(timeout: 2) { break }
            let next = app.buttons["Next keyboard"].firstMatch
            if next.exists { next.tap() }
        }
        mic.tap()
        XCTAssertTrue(app.buttons["Stop dictation"].waitForExistence(timeout: 10))
        app.buttons["Show text keyboard"].tap()
        app.buttons["textLayout-ja-flick"].tap()
        app.buttons["kanaKey-ka"].tap()
        XCTAssertEqual(field.value as? String, "か")
        // End this keyboard-origin recording from the host while kana is marked.
        // Switching to the voice keyboard would commit it and miss this boundary.
        let stop = app.buttons["recordButton"]
        for _ in 0..<3 where !stop.isHittable { app.scrollViews.firstMatch.swipeDown() }
        XCTAssertTrue(stop.isHittable)
        stop.tap()
        XCTAssertTrue(app.staticTexts["latestResultText"].waitForExistence(timeout: 180))
        XCTAssertEqual(field.value as? String, "か", "A changed field must retain typed kana and keep the result in Home")
        // The changed field rejects the dictation. Its current kana composition
        // belongs to the user and must remain editable after the result arrives.
        XCTAssertTrue(app.buttons["japaneseCandidate-か"].exists)
        app.buttons["japaneseModifier"].tap()
        XCTAssertEqual(field.value as? String, "が")
        snap("japanese-incoming-dictation", app)
    }

    func testJapaneseMarkedTextEditsAndCommitsWithoutDuplicatingKana() throws {
        try enableVoxtKeyboard()
        let app = try launchAppPairedByLink(typingLayouts: ["ja-flick", "zh-Hant-zhuyin"], opensRecorder: false)
        let field = app.textFields["tryField"]
        field.tap()
        let mic = app.buttons["Start dictation"]
        for _ in 0..<6 {
            if mic.waitForExistence(timeout: 2) { break }
            let next = app.buttons["Next keyboard"].firstMatch
            if next.exists { next.tap() }
        }
        XCTAssertTrue(mic.exists)
        app.buttons["voiceLayout-ja-flick"].tap()
        let layout = app.buttons["textLayout-ja-flick"]
        XCTAssertTrue(layout.waitForExistence(timeout: 3))
        layout.tap()
        app.buttons["kanaKey-ka"].tap()
        XCTAssertEqual(field.value as? String, "か")
        let a = app.buttons["kanaKey-a"]
        a.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
            .press(forDuration: 0.05, thenDragTo: a.coordinate(withNormalizedOffset: CGVector(dx: 0.05, dy: 0.5)))
        XCTAssertEqual(field.value as? String, "かい")
        XCTAssertTrue(app.buttons["kanaKey-ka"].exists, "A key flick must stay in the text keyboard")
        app.buttons["japaneseDelete"].tap()
        XCTAssertEqual(field.value as? String, "か")
        app.buttons["japaneseModifier"].tap()
        XCTAssertEqual(field.value as? String, "が")
        app.buttons["japaneseReturn"].tap()
        XCTAssertEqual(field.value as? String, "が", "Return commits the reading without adding a newline")
        app.buttons["kanaKey-ka"].tap()
        XCTAssertEqual(field.value as? String, "がか", "A new reading must not replace the committed text")
        snap("japanese-marked-text", app)
        app.buttons["textLayout-voice"].tap()
        XCTAssertTrue(mic.waitForExistence(timeout: 3))
        XCTAssertEqual(field.value as? String, "がか")
        app.buttons["voiceLayout-ja-flick"].tap()
        app.buttons["kanaKey-ka"].tap()
        XCTAssertEqual(field.value as? String, "がかか")
        app.buttons["Hide keyboard"].tap()
        // Reopening by the field's center can put the caret before this short
        // string. Tap its trailing empty area to test appending, not caret motion.
        field.coordinate(withNormalizedOffset: CGVector(dx: 0.95, dy: 0.5)).tap()
        XCTAssertTrue(mic.waitForExistence(timeout: 3))
        app.buttons["voiceLayout-ja-flick"].tap()
        XCTAssertFalse(app.buttons["japaneseCandidate-か"].exists,
                       "Dismissing the keyboard must discard the previous candidate")
        // The host restores focus before the extension finishes reattaching.
        // Set the desired insertion point after the keyboard is visible.
        field.coordinate(withNormalizedOffset: CGVector(dx: 0.95, dy: 0.5)).tap()
        snap("japanese-reopened", app)
        app.buttons["kanaKey-sa"].tap()
        XCTAssertEqual(field.value as? String, "がかかさ")
    }

    private func showTextLayout(_ id: String, in app: XCUIApplication) -> XCUIElement {
        let field = app.textFields["tryField"]
        field.tap()
        let mic = app.buttons["Start dictation"]
        let next = app.buttons["Next keyboard"].firstMatch
        for _ in 0..<6 {
            if mic.waitForExistence(timeout: 2) { break }
            if next.exists { next.tap() }
        }
        XCTAssertTrue(mic.exists)
        app.buttons["Show text keyboard"].tap()
        let layout = app.buttons["textLayout-" + id]
        XCTAssertTrue(layout.waitForExistence(timeout: 3))
        layout.tap()
        return field
    }

    /// 두벌식 through the KoreanInput package: syllables compose in place, a final
    /// consonant joins the next syllable, backspace removes one jamo.
    func testKoreanComposesSyllablesAndDeletesOneJamoAtATime() throws {
        try enableVoxtKeyboard()
        let app = try launchAppPairedByLink(typingLayouts: ["ko-2beolsik"], opensRecorder: false)
        let field = showTextLayout("ko-2beolsik", in: app)
        for jamo in ["ㅎ", "ㅏ", "ㄴ", "ㄱ", "ㅡ", "ㄹ"] { app.buttons["textKey-\(jamo)"].tap() }
        XCTAssertEqual(field.value as? String, "한글")
        for expected in ["한그", "한ㄱ", "한"] {
            app.buttons["textKey-delete"].tap()
            XCTAssertEqual(field.value as? String, expected)
        }
        for jamo in ["ㄷ", "ㅏ", "ㄹ", "ㄱ"] { app.buttons["textKey-\(jamo)"].tap() }
        XCTAssertEqual(field.value as? String, "한닭")
        app.buttons["textKey-shift"].tap()
        XCTAssertTrue(app.buttons["textKey-ㄲ"].exists, "shift shows the doubled consonants")
        snap("12-korean-keyboard", app)
        app.buttons["textLayout-voice"].tap()
        XCTAssertTrue(app.buttons["Start dictation"].waitForExistence(timeout: 3))
        XCTAssertEqual(field.value as? String, "한닭")
    }

    /// Spanish through SpanishInput: ñ key, accents on long press, ¿ on the numbers
    /// plane, and a capital after ¿ and after "? ".
    func testSpanishCapitalizesAfterInvertedMarksAndOffersAccents() throws {
        try enableVoxtKeyboard()
        let app = try launchAppPairedByLink(typingLayouts: ["es-QWERTY"], opensRecorder: false)
        let field = showTextLayout("es-QWERTY", in: app)
        XCTAssertTrue(app.buttons["textKey-Q"].waitForExistence(timeout: 3), "an empty field starts with a capital")
        app.buttons["textKey-numbers"].tap()
        app.buttons["textKey-¿"].tap()
        app.buttons["textKey-letters"].firstMatch.tap()
        XCTAssertTrue(app.buttons["textKey-Q"].waitForExistence(timeout: 3), "shift stays on after ¿")
        app.buttons["textKey-Q"].tap()
        app.buttons["textKey-u"].tap()
        app.buttons["textKey-e"].press(forDuration: 1.0)
        let accent = app.buttons["é"]
        XCTAssertTrue(accent.waitForExistence(timeout: 3))
        accent.tap()
        app.buttons["textKey-numbers"].tap()
        app.buttons["textKey-?"].tap()
        app.buttons["textKey-space"].tap()
        app.buttons["textKey-letters"].firstMatch.tap()
        XCTAssertTrue(app.buttons["textKey-Ñ"].waitForExistence(timeout: 3), "a capital after \"? \"")
        app.buttons["textKey-Ñ"].tap()
        app.buttons["textKey-a"].tap()
        XCTAssertEqual(field.value as? String, "¿Qué? Ña")
        snap("13-spanish-keyboard", app)
    }

    /// Pinyin through PinyinInput: letters stay marked, a candidate commits, a word from
    /// the start leaves the rest marked, Space takes the first candidate, "，" commits first.
    func testPinyinConvertsPicksPartWordsAndCommitsWithSpaceAndPunctuation() throws {
        try enableVoxtKeyboard()
        let app = try launchAppPairedByLink(typingLayouts: ["zh-Hans-pinyin"], opensRecorder: false)
        let field = showTextLayout("zh-Hans-pinyin", in: app)
        func type(_ letters: String) { for letter in letters { app.buttons["textKey-\(letter)"].tap() } }
        func pick(_ text: String) {
            let candidate = app.buttons["candidate-\(text)"]
            XCTAssertTrue(candidate.waitForExistence(timeout: 10), "no candidate \(text)")
            candidate.tap()
        }
        type("nihao")
        XCTAssertEqual(field.value as? String, "ni hao", "letters stay marked in the field, one space per syllable")
        pick("你好")
        XCTAssertEqual(field.value as? String, "你好")
        type("nihao")
        pick("你")
        XCTAssertEqual(field.value as? String, "你好你hao", "the rest stays marked")
        pick("好")
        XCTAssertEqual(field.value as? String, "你好你好")
        type("wo")
        app.buttons["textKey-delete"].tap()
        XCTAssertEqual(field.value as? String, "你好你好w")
        app.buttons["textKey-delete"].tap()
        XCTAssertEqual(field.value as? String, "你好你好")
        type("women")
        XCTAssertTrue(app.buttons["candidate-我们"].waitForExistence(timeout: 10))
        app.buttons["textKey-space"].tap()
        XCTAssertEqual(field.value as? String, "你好你好我们")
        type("xiexie")
        XCTAssertTrue(app.buttons["candidate-谢谢"].waitForExistence(timeout: 10))
        app.buttons["textKey-numbers"].tap()
        app.buttons["textKey-，"].tap()
        XCTAssertEqual(field.value as? String, "你好你好我们谢谢，")
        snap("14-pinyin-keyboard", app)
    }

    /// Pinyin v1.1: initials stand for syllables (nh → 你好), and a tapped word comes
    /// first next time (Space takes it). Starts and ends by picking 你好, so earlier
    /// runs' picks don't change the result.
    func testPinyinAbbreviationsAndRememberedPicks() throws {
        try enableVoxtKeyboard()
        let app = try launchAppPairedByLink(typingLayouts: ["zh-Hans-pinyin"], opensRecorder: false)
        let field = showTextLayout("zh-Hans-pinyin", in: app)
        func type(_ letters: String) { for letter in letters { app.buttons["textKey-\(letter)"].tap() } }
        func pick(_ text: String) {
            let candidate = app.buttons["candidate-\(text)"]
            XCTAssertTrue(candidate.waitForExistence(timeout: 10), "no candidate \(text)")
            candidate.tap()
        }
        func space(after letters: String, expecting text: String) {
            type(letters)
            XCTAssertTrue(app.buttons["candidate-\(text)"].waitForExistence(timeout: 10), "no candidate \(text)")
            app.buttons["textKey-space"].tap()
        }
        type("nh")
        XCTAssertEqual(field.value as? String, "n h")
        pick("你好")
        XCTAssertEqual(field.value as? String, "你好")
        space(after: "nihao", expecting: "你好")
        XCTAssertEqual(field.value as? String, "你好你好")
        type("nihao")
        pick("拟好")
        XCTAssertEqual(field.value as? String, "你好你好拟好")
        space(after: "nihao", expecting: "拟好")
        XCTAssertEqual(field.value as? String, "你好你好拟好拟好", "the picked word now comes first")
        snap("15-pinyin-remembered", app)
        type("nh")
        pick("你好")
        XCTAssertEqual(field.value as? String, "你好你好拟好拟好你好")
    }

    /// Settings › Keyboard reaches the pinyin keyboard without a relaunch: Clear learned words
    /// forgets a pick, and the fuzzy switch makes zongguo give 中国. Turns fuzzy back off
    /// (VOXT_TEST_RESET also turns it off at every launch, so a failure here leaves nothing behind).
    func testPinyinSettingsClearPicksAndTurnOnFuzzy() throws {
        try enableVoxtKeyboard()
        let app = try launchAppPairedByLink(typingLayouts: ["zh-Hans-pinyin"], opensRecorder: false)
        var field = showTextLayout("zh-Hans-pinyin", in: app)
        func type(_ letters: String) { for letter in letters { app.buttons["textKey-\(letter)"].tap() } }
        func space(after letters: String, expecting text: String) {
            type(letters)
            XCTAssertTrue(app.buttons["candidate-\(text)"].waitForExistence(timeout: 10), "no candidate \(text)")
            app.buttons["textKey-space"].tap()
        }
        func settings(_ change: () -> Void) {
            app.buttons["Hide keyboard"].firstMatch.tap()
            app.buttons["Settings"].tap()
            change()
            app.navigationBars["Settings"].buttons["Done"].tap()
            field = showTextLayout("zh-Hans-pinyin", in: app)
        }
        func reveal(_ element: XCUIElement) -> XCUIElement {
            for _ in 0..<8 where !element.isHittable { app.swipeUp() }
            return element
        }
        func setFuzzy(_ on: Bool) {
            let fuzzy = reveal(app.switches["pinyinFuzzy"])
            if (fuzzy.value as? String == "1") != on { fuzzy.coordinate(withNormalizedOffset: CGVector(dx: 0.9, dy: 0.5)).tap() }
            // The switch's value follows the tap a moment later (as in AppScreensUITests.waitForValue).
            let deadline = Date().addingTimeInterval(3)
            while fuzzy.value as? String != (on ? "1" : "0"), Date() < deadline { Thread.sleep(forTimeInterval: 0.25) }
            XCTAssertEqual(fuzzy.value as? String, on ? "1" : "0")
        }

        type("nihao")
        let picked = app.buttons["candidate-拟好"]
        XCTAssertTrue(picked.waitForExistence(timeout: 10))
        picked.tap()
        space(after: "nihao", expecting: "拟好")
        XCTAssertEqual(field.value as? String, "拟好拟好", "the pick comes first")

        settings {
            let clear = reveal(app.buttons["clearPinyinLearnedWords"])
            XCTAssertTrue(clear.isEnabled, "a word was learned")
            clear.tap()
            XCTAssertFalse(clear.isEnabled)
            setFuzzy(true)
        }
        space(after: "nihao", expecting: "你好")
        // Tapping the field again after Settings can put the cursor first: check the text, not its end.
        XCTAssertTrue((field.value as? String ?? "").contains("你好"), "cleared: \(field.value ?? "")")
        space(after: "zongguo", expecting: "中国")
        XCTAssertTrue((field.value as? String ?? "").contains("中国"), "fuzzy on: \(field.value ?? "")")
        settings { setFuzzy(false) }
    }

    /// Pinyin while composing "wojiushi…":
    /// syllables spaced in the field, no toolbar while composing, plain candidates without the
    /// letters themselves, ⌄ opens them all, Return enters the letters, iOS key proportions,
    /// and no system dictation key next to ours.
    func testPinyinSpacesSyllablesShowsCandidatesAndUsesStandardKeys() throws {
        try enableVoxtKeyboard()
        let app = try launchAppPairedByLink(typingLayouts: ["zh-Hans-pinyin"], opensRecorder: false)
        let field = showTextLayout("zh-Hans-pinyin", in: app)
        let key = { (name: String) in app.buttons["textKey-\(name)"] }
        let toolbar = app.descendants(matching: .any)["textLayoutToolbar"]
        XCTAssertTrue(toolbar.waitForExistence(timeout: 3))
        snap("pinyin-idle", app)
        for letter in "wojiushishi" { key(String(letter)).tap() }
        let first = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH 'candidate-'")).firstMatch
        XCTAssertTrue(first.waitForExistence(timeout: 10))
        snap("pinyin-composing", app)

        XCTAssertEqual(field.value as? String, "wo jiu shi shi")
        XCTAssertFalse(toolbar.exists, "no toolbar while composing")
        XCTAssertFalse(app.buttons["candidate-wo jiu shi shi"].exists)
        XCTAssertFalse(app.buttons["candidate-wojiushishi"].exists, "the letters aren't a candidate")
        XCTAssertTrue(first.isSelected, "the first candidate is the default")
        // iOS QWERTY: a–l letter-width and centered, z–m letter-width, shift and delete wider.
        let q = key("q").frame, p = key("p").frame, a = key("a").frame, l = key("l").frame
        XCTAssertEqual(a.width, q.width, accuracy: 1)
        XCTAssertEqual(key("z").frame.width, q.width, accuracy: 1)
        XCTAssertGreaterThan(a.minX - q.minX, q.width * 0.4, "a starts half a key in")
        XCTAssertEqual(a.minX - q.minX, p.maxX - l.maxX, accuracy: 1, "a–l centered")
        XCTAssertGreaterThan(key("shift").frame.width, q.width * 1.3)
        XCTAssertGreaterThan(key("delete").frame.width, q.width * 1.3)

        app.buttons["candidateExpand"].tap()
        let grid = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH 'candidateGrid-'")).firstMatch
        XCTAssertTrue(grid.waitForExistence(timeout: 3))
        XCTAssertFalse(key("q").exists, "the grid takes the keys' place")
        snap("pinyin-expanded", app)
        app.buttons["candidateCollapse"].tap()
        XCTAssertTrue(key("q").waitForExistence(timeout: 3))

        key("return").tap()
        XCTAssertEqual(field.value as? String, "wojiushishi", "Return enters the letters as typed")
        XCTAssertTrue(toolbar.waitForExistence(timeout: 3), "the toolbar is back when idle")
        // UIInputViewController.hasDictationKey: Apple says the system key is then disabled.
        let dictate = app.buttons["Dictate"]
        print("VOXT-DICTATE-KEY exists=\(dictate.exists) enabled=\(dictate.exists && dictate.isEnabled)")
        XCTAssertTrue(!dictate.exists || !dictate.isEnabled, "the system dictation key is still on")
    }

    /// The Voxt name at the voice keyboard's top left opens Voxt on Settings. The host here is Voxt's own practice field, so Settings opens over it.
    func testKeyboardNameOpensVoxtSettings() throws {
        try enableVoxtKeyboard()
        let app = try launchAppPairedByLink(opensRecorder: false)
        app.textFields["tryField"].tap()
        let mic = app.buttons["Start dictation"]
        let next = app.buttons["Next keyboard"].firstMatch
        for _ in 0..<6 {
            if mic.waitForExistence(timeout: 2) { break }
            if next.exists { next.tap() }
        }
        let name = app.buttons["keyboardOpenVoxt"]
        XCTAssertTrue(name.waitForExistence(timeout: 3))
        XCTAssertEqual(name.label, "Open Voxt settings")
        name.tap()
        XCTAssertTrue(app.staticTexts["settingsSection-yourMac"].waitForExistence(timeout: 5), "Settings opened")
    }

    /// Long 简拼 input must not stall the keys; prints VOXT-PINYIN-TIMING lines (simulator baseline).
    func testPinyinLongAbbreviationsStayResponsive() throws {
        try enableVoxtKeyboard()
        let app = try launchAppPairedByLink(typingLayouts: ["zh-Hans-pinyin"], opensRecorder: false)
        _ = showTextLayout("zh-Hans-pinyin", in: app)
        measurePinyinTiming(in: app, place: "simulator")
    }

    func testTextKeyboardTypesDeletesAndReturnsToVoice() throws {
        try enableVoxtKeyboard()
        let app = try launchAppPairedByLink(opensRecorder: false)
        let field = app.textFields["tryField"]
        field.tap()
        let mic = app.buttons["Start dictation"]
        let next = app.buttons["Next keyboard"].firstMatch
        for _ in 0..<6 {
            if mic.waitForExistence(timeout: 2) { break }
            if next.exists { next.tap() }
        }
        XCTAssertTrue(mic.exists)
        let type = app.buttons["Show text keyboard"]
        XCTAssertTrue(type.waitForExistence(timeout: 3))
        type.tap()
        for letter in ["h", "e", "l", "l", "o"] { app.buttons["textKey-\(letter)"].tap() }
        XCTAssertEqual(field.value as? String, "hello")
        app.buttons["textKey-delete"].tap()
        XCTAssertEqual(field.value as? String, "hell")
        app.buttons["textKey-o"].tap()
        snap("11-text-keyboard", app)
        app.buttons["textLayout-voice"].tap()
        XCTAssertTrue(mic.waitForExistence(timeout: 3))
        XCTAssertEqual(field.value as? String, "hello")
    }

    /// Settings > General > Keyboard > Keyboards > Add New Keyboard… > Voxt, then Allow Full Access.
    private func enableVoxtKeyboard() throws {
        // On a fresh simulator Settings lists the Voxt keyboard only after Voxt has been opened once.
        let voxt = XCUIApplication()
        voxt.launch()
        voxt.terminate()
        let settings = XCUIApplication(bundleIdentifier: "com.apple.Preferences")
        settings.terminate()
        settings.launch()

        tapCell(settings, settings.staticTexts["General"].firstMatch, "General")
        tapCell(settings, settings.staticTexts["Keyboard"].firstMatch, "Keyboard")
        tapCell(settings, settings.cells["KEYBOARDS"].firstMatch, "Keyboards")
        XCTAssertTrue(settings.navigationBars["Keyboards"].waitForExistence(timeout: 5))

        let voxtRow = labeled(settings, "Voxt")
        if !voxtRow.waitForExistence(timeout: 3) {
            let add = settings.cells["AddNewKeyboard"].firstMatch
            tapCell(settings, add, "Add New Keyboard")
            let voxt = labeled(settings, "Voxt")
            tapCell(settings, voxt, "Voxt (in Add New Keyboard)")
            XCTAssertTrue(settings.navigationBars["Keyboards"].waitForExistence(timeout: 5))
        }

        tapCell(settings, labeled(settings, "Voxt"), "Voxt")
        let fullAccess = settings.switches.firstMatch
        XCTAssertTrue(fullAccess.waitForExistence(timeout: 5))
        if (fullAccess.value as? String) != "1" {
            fullAccess.coordinate(withNormalizedOffset: CGVector(dx: 0.9, dy: 0.5)).tap()
            let allow = settings.alerts.buttons["Allow"]
            if allow.waitForExistence(timeout: 5) { allow.tap() }
        }
        let fullAccessOn = expectation(for: NSPredicate(format: "value == '1'"), evaluatedWith: settings.switches.firstMatch)
        wait(for: [fullAccessOn], timeout: 5)
        settings.terminate()
    }

    private func labeled(_ app: XCUIApplication, _ label: String) -> XCUIElement {
        app.descendants(matching: .any).matching(NSPredicate(format: "label == %@", label)).firstMatch
    }

    private func tapCell(_ app: XCUIApplication, _ element: XCUIElement, _ name: String) {
        if !element.waitForExistence(timeout: 5) {
            app.swipeUp()
        }
        if !element.waitForExistence(timeout: 3) {
            print("Settings screen without \(name):\n\(app.debugDescription)")
        }
        XCTAssertTrue(element.exists, "Settings row \(name) not found")
        element.tap()
    }
}

final class PhysicalMicrophoneUITests: XCTestCase {
    func testEditButtonStartsRealMicrophone() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["VOXT_PHYSICAL_MIC_TEST"] == "1")
        continueAfterFailure = false
        let app = XCUIApplication()
        app.terminate()
        app.launch()
        XCTAssertTrue(app.buttons["keyboardSetup"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.buttons["keyboardSetup"].isHittable)
        XCTAssertFalse(app.buttons["recordButton"].exists)
        let home = XCTAttachment(screenshot: app.screenshot())
        home.name = "physical-keyboard-home"
        home.lifetime = .keepAlways
        add(home)
        defer {
            let screenshot = XCTAttachment(screenshot: app.screenshot())
            screenshot.name = "physical-edit-state"
            screenshot.lifetime = .keepAlways
            add(screenshot)
            let tree = XCTAttachment(string: app.debugDescription)
            tree.name = "physical-edit-accessibility"
            tree.lifetime = .keepAlways
            add(tree)
        }
        app.tabBars.buttons["History"].tap()
        let edit = app.buttons["editLatestResult"]
        XCTAssertTrue(edit.waitForExistence(timeout: 5), "Edit latest result must be present")
        XCTAssertTrue(edit.isEnabled, "Edit latest result must be enabled")
        edit.tap()
        let stop = app.buttons["recordButton"]
        let recording = XCTNSPredicateExpectation(predicate: NSPredicate(format: "label == %@", "Stop recording"), object: stop)
        XCTAssertEqual(XCTWaiter.wait(for: [recording], timeout: 8), .completed)
        XCTAssertLessThan(stop.frame.maxY, app.tabBars.firstMatch.frame.minY,
                          "Recording controls must be fully above the tab bar on a small iPhone")
        // Stop after verifying capture starts; no test audio is injected.
        stop.tap()
        let stopped = XCTNSPredicateExpectation(predicate: NSPredicate(format: "label != %@", "Stop recording"), object: stop)
        XCTAssertEqual(XCTWaiter.wait(for: [stopped], timeout: 8), .completed)
    }
}

/// Types fixed inputs on the Voxt pinyin keyboard in `app` and prints how long the slowest
/// key tap and the sentence after the last key took. The sentences are the bundled data's
/// first candidates (PinyinInput); Space commits each one.
func measurePinyinTiming(in app: XCUIApplication, place: String, file: StaticString = #filePath, line: UInt = #line) {
    let inputs = [("plain", "nihaonihaonihao", "你好你好你好"),
                  ("initials10", "nhwmzgrdsj", "你好我们在个人第四届"),
                  ("initials16", "bpmfdtnlbpmfdtnl", "被迫免费的太难了被迫免费的太难了")]
    for (name, letters, sentence) in inputs {
        var slowest = 0.0
        for letter in letters {
            let start = Date()
            app.buttons["textKey-\(letter)"].tap()
            slowest = max(slowest, Date().timeIntervalSince(start))
        }
        let start = Date()
        let shown = app.buttons["candidate-\(sentence)"].waitForExistence(timeout: 15)
        let candidate = Date().timeIntervalSince(start)
        XCTAssertTrue(shown, "no sentence for \(name)", file: file, line: line)
        app.buttons["textKey-space"].tap()
        print("VOXT-PINYIN-TIMING place=\(place) input=\(name) letters=\(letters.count) slowestKeyMs=\(Int(slowest * 1000)) sentenceAfterLastKeyMs=\(Int(candidate * 1000))")
        XCTAssertLessThan(slowest, 3, "a key tap stalled (\(name))", file: file, line: line)
    }
}

/// Real-iPhone check of the floating-window (PiP) path with the real microphone.
/// Runs only with VOXT_DEVICE_PIP=1. The Mac plays the speech: each "VOXT-SPEAK" line in the
/// xcodebuild log makes the watcher run afplay, so recordings hold real audio.
@MainActor
final class DevicePiPUITests: XCTestCase {
    private struct Step: Codable {
        var name: String
        var voxtInFront: Bool?
        var recordingStarted: Bool?
        var insertedText: String?
        var stopToTextSeconds: Double?
        var indicatorPixels: Int
        var note: String?
    }

    private let voxt = XCUIApplication()
    private let notes = XCUIApplication(bundleIdentifier: "com.apple.mobilenotes")
    private var steps: [Step] = []
    private let expected = "Alice"

    func testFloatingWindowKeepsMicrophoneOffWhenIdle() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["VOXT_DEVICE_PIP"] == "1")
        continueAfterFailure = true
        defer { report() }

        // Instrument check: a frame taken before anything runs, for comparison with recording frames.
        steps.append(Step(name: "00-start", indicatorPixels: shot("00-start")))

        // 1. Floating window on, then leave Voxt so the window floats.
        if let variant = ProcessInfo.processInfo.environment["VOXT_PIP_VARIANT"] {
            voxt.launchEnvironment["VOXT_PIP_VARIANT"] = variant
        }
        voxt.launch()
        setFloatingWindow(on: true)
        steps.append(Step(name: "01-floating-on", indicatorPixels: shot("01-floating-on")))
        XCUIDevice.shared.press(.home)
        sleep(2)
        steps.append(Step(name: "01-floating-home", indicatorPixels: shot("01-floating-home")))

        // 2. Notes, idle for 10 s: the indicator should be off.
        openNewNote()
        sleep(10)
        steps.append(Step(name: "02-notes-idle-10s", indicatorPixels: shot("02-notes-idle-10s")))

        // 3. Dictate through the Voxt keyboard.
        dictate("03-dictate")
        // 4. Five seconds after insertion.
        sleep(5)
        steps.append(Step(name: "04-after-5s", indicatorPixels: shot("04-after-5s")))

        // 5a. Home, then back to Notes.
        XCUIDevice.shared.press(.home)
        sleep(2)
        notes.activate()
        dictate("05a-after-home")
        // 5b. Another app, then back to Notes.
        XCUIApplication(bundleIdentifier: "com.apple.Preferences").activate()
        sleep(2)
        notes.activate()
        dictate("05b-after-app-switch")
        sleep(5)
        steps.append(Step(name: "05-after-5s", indicatorPixels: shot("05-after-5s")))

        // 6. Floating window off: the next dictation is expected to hand off to Voxt.
        voxt.activate()
        setFloatingWindow(on: false)
        XCUIDevice.shared.press(.home)
        sleep(2)
        notes.activate()
        dictate("06-floating-off")
        sleep(5)
        steps.append(Step(name: "06-after-5s", indicatorPixels: shot("06-after-5s")))

        // 7. The same line 15 dB quieter (DEVICE-RECHECK #6): still text, no "No speech".
        dictate("07-low-volume", clip: "dictation-quiet")

        // 8. A slow result (#7): the app holds each result 12 s, as a slow Mac would.
        //    Once into an empty note, once into a note that already has text.
        voxt.terminate()
        voxt.launchEnvironment["VOXT_TEST_RESULT_DELAY_MS"] = "12000"
        voxt.launch()
        XCUIDevice.shared.press(.home)
        sleep(2)
        openNewNote()
        dictate("08a-slow-empty-note", timeout: 75)
        dictate("08b-slow-note-with-text", timeout: 75)
    }

    /// Re-pair on the same Wi-Fi so the Connector's pair link (m12+) brings the Mac's name.
    /// Needs the user once: Voxt Connector > Add iPhone, compare the code, click Allow.
    /// The test unpairs, taps Find my Mac, prints VOXT-USER when the code shows, and checks
    /// that Settings then names the Mac (not "Your Mac" and never "tailcat.invalid").
    func testSameWiFiRepairShowsTheMacName() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["VOXT_DEVICE_REPAIR"] == "1")
        continueAfterFailure = false
        voxt.launch()
        let settings = voxt.buttons["Settings"]
        XCTAssertTrue(settings.waitForExistence(timeout: 10))
        settings.tap()
        let unpair = voxt.buttons["Unpair"]
        for _ in 0..<8 where !unpair.isHittable { voxt.swipeUp() }
        unpair.tap()
        let find = voxt.buttons["findMacButton"]
        XCTAssertTrue(find.waitForExistence(timeout: 10))
        find.tap()
        let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        if springboard.alerts.buttons["Allow"].waitForExistence(timeout: 3) { springboard.alerts.buttons["Allow"].tap() }
        let code = voxt.staticTexts["localPairingCode"]
        XCTAssertTrue(code.waitForExistence(timeout: 30), "no code: is Add iPhone open on the Mac?")
        print("VOXT-USER click Allow on the Mac: \(code.label)")
        XCTAssertTrue(voxt.descendants(matching: .any)["connectedLabel"].waitForExistence(timeout: 120), "not paired within 2 minutes")
        XCTAssertTrue(settings.waitForExistence(timeout: 10))
        settings.tap()
        let name = voxt.descendants(matching: .any)["pairedMacName"]
        for _ in 0..<8 where !name.isHittable { voxt.swipeUp() }
        let shot = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
        shot.name = "repair-mac-name"
        shot.lifetime = .keepAlways
        add(shot)
        print("VOXT-MAC-NAME \(name.label)")
        XCTAssertFalse(name.label.contains("tailcat.invalid"))
        XCTAssertFalse(name.label.hasPrefix("Your Mac"), "the m12 pair link carries the name")
    }

    /// The window must stay open when Settings is dismissed. No speech needed: open it,
    /// close Settings, go Home for 5 s, come back, and check it is still open.
    func testFloatingWindowSurvivesClosingSettings() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["VOXT_DEVICE_PIP"] == "1")
        continueAfterFailure = true
        defer { report() }
        voxt.launch()
        setFloatingWindow(on: true)
        // The window must outlive the Settings sheet it was opened from.
        let settings = voxt.buttons["Settings"]
        XCTAssertTrue(settings.waitForExistence(timeout: 5))
        settings.tap()
        voxt.buttons["Done"].firstMatch.tap()
        XCUIDevice.shared.press(.home)
        sleep(5)
        steps.append(Step(name: "home-5s", indicatorPixels: shot("pip-home-5s"), note: "mic indicator while idle with the window open"))
        voxt.activate()
        setFloatingWindow(on: false)
        let survived = steps.last { $0.name == "06-window-still-open" }?.note == "wasOpen=true"
        XCTAssertTrue(survived, "the floating window closed after Settings was dismissed")
    }

    /// After pairing through a Connector that sends its name: Settings names the Mac.
    /// Pinyin v1.1 on the phone: long 简拼 input keeps the keys responsive (VOXT-PINYIN-TIMING).
    func testPinyinTypingStaysResponsive() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["VOXT_DEVICE_PIP"] == "1")
        voxt.launch()
        let settings = voxt.buttons["Settings"]
        XCTAssertTrue(settings.waitForExistence(timeout: 10))
        settings.tap()
        let languages = voxt.buttons["Text keyboards"]
        for _ in 0..<6 where !languages.isHittable { voxt.swipeUp() }
        languages.tap()
        XCTAssertTrue(voxt.navigationBars["Keyboard languages"].waitForExistence(timeout: 5))
        let add = voxt.buttons["addLayout-zh-Hans-pinyin"]
        if add.exists { add.tap() }
        XCTAssertTrue(voxt.buttons["removeLayout-zh-Hans-pinyin"].exists)
        voxt.navigationBars["Keyboard languages"].buttons["Done"].tap()
        voxt.navigationBars["Settings"].buttons["Done"].tap()
        XCUIDevice.shared.press(.home)
        openNewNote()
        _ = showVoxtKeyboard()
        notes.buttons["Show text keyboard"].tap()
        let layout = notes.buttons["textLayout-zh-Hans-pinyin"]
        XCTAssertTrue(layout.waitForExistence(timeout: 3))
        layout.tap()
        measurePinyinTiming(in: notes, place: "iphone")
    }

    func testSettingsShowsTheMacName() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["VOXT_DEVICE_PIP"] == "1")
        voxt.launch()
        let settings = voxt.buttons["Settings"]
        XCTAssertTrue(settings.waitForExistence(timeout: 10))
        settings.tap()
        let name = voxt.descendants(matching: .any)["pairedMacName"]
        for _ in 0..<8 where !name.isHittable { voxt.swipeUp() }
        let shot = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
        shot.name = "settings-mac-name"
        shot.lifetime = .keepAlways
        add(shot)
        print("VOXT-MAC-NAME \(name.label)")
        XCTAssertFalse(name.label.contains("tailcat.invalid"))
        XCTAssertFalse(name.label.hasPrefix("Your Mac"), "a Connector pair link with name= was used")
        XCTAssertTrue(name.label.contains(", connected, Voxt "), name.label)
        voxt.buttons["Done"].firstMatch.tap()
        let home = voxt.descendants(matching: .any)["connectedLabel"]
        XCTAssertTrue(home.waitForExistence(timeout: 10))
        XCTAssertEqual(home.label, name.label, "home page and Settings use the same words")
    }

    /// DEVICE-RECHECK #8: dictate, then Edit with no selection rewrites that result; Undo
    /// restores it. (A Voxt relaunch clears the latest result, so this dictates first.)
    func testEditLatestResultThenUndo() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["VOXT_DEVICE_PIP"] == "1")
        continueAfterFailure = true
        defer { report() }
        voxt.launch()
        XCUIDevice.shared.press(.home)
        sleep(2)
        openNewNote()
        dictate("09a-dictate")
        editLatestThenUndo("09b-edit-undo")
    }

    /// Home's Keep Voxt ready switch (it lives on Home, not in Settings); its
    /// value is "1" while the floating window is open.
    private func setFloatingWindow(on: Bool) {
        let toggle = voxt.descendants(matching: .any)["keepVoxtReady"]
        XCTAssertTrue(toggle.waitForExistence(timeout: 10), "Keep Voxt ready is on Home")
        for _ in 0..<6 where !toggle.isHittable { voxt.swipeUp() }
        let isOn = { (toggle.value as? String) == "1" }
        // At step 6 this says whether the window survived the dictations.
        print("VOXT-TOGGLE enabled=\(toggle.isEnabled) wasOpen=\(isOn())")
        if !on { steps.append(Step(name: "06-window-still-open", indicatorPixels: 0, note: "wasOpen=\(isOn())")) }
        _ = shot(on ? "01-card-before" : "06-card-before")
        if isOn() != on { toggle.tap() }
        let changed = XCTNSPredicateExpectation(predicate: NSPredicate(format: "value == %@", on ? "1" : "0"), object: toggle)
        let result = XCTWaiter.wait(for: [changed], timeout: 8)
        _ = shot(on ? "01-card-after" : "06-card-after")
        XCTAssertEqual(result, .completed, "Floating window did not turn \(on ? "on" : "off")")
    }

    private func openNewNote() {
        notes.launch()
        let new = notes.buttons.matching(NSPredicate(format: "label CONTAINS[c] 'new note' OR identifier CONTAINS[c] 'NewNote' OR label CONTAINS[c] 'compose'")).firstMatch
        if new.waitForExistence(timeout: 8) { new.tap() } else { print("Notes UI:\n\(notes.debugDescription)") }
        let text = notes.textViews.firstMatch
        XCTAssertTrue(text.waitForExistence(timeout: 8), "No note text view")
        text.tap()
    }

    private func noteText() -> String { notes.textViews.firstMatch.value as? String ?? "" }

    private func showVoxtKeyboard() -> XCUIElement {
        let mic = notes.buttons["Start dictation"]
        if !mic.waitForExistence(timeout: 3) { notes.textViews.firstMatch.tap() }
        let next = notes.buttons.matching(NSPredicate(format: "label == 'Next keyboard' OR identifier == 'Next keyboard'")).firstMatch
        var switches = 0
        while !mic.waitForExistence(timeout: 2), switches < 6 {
            if notes.buttons["textLayout-voice"].exists { notes.buttons["textLayout-voice"].tap() }
            else if next.exists { next.tap() }
            switches += 1
        }
        if !mic.exists { print("Keyboard state:\n\(notes.debugDescription)") }
        return mic
    }

    /// The pen arms one edit for the next recording; a plain mic tap always dictates.
    private func armPen() {
        let pen = notes.buttons["keyboardPen"]
        if pen.waitForExistence(timeout: 3), !pen.isSelected { pen.tap() }
    }

    /// Edit mode with no selection rewrites the latest result; Undo puts it back.
    private func editLatestThenUndo(_ name: String) {
        let mic = showVoxtKeyboard()
        armPen()
        let before = noteText()
        _ = shot(name + "-a-keyboard")
        mic.tap()
        if voxt.wait(for: .runningForeground, timeout: 5) { notes.activate() }
        let stop = notes.buttons["Stop dictation"]
        guard stop.waitForExistence(timeout: 10) else {
            steps.append(Step(name: name, recordingStarted: false, indicatorPixels: shot(name + "-c-no-recording")))
            XCTFail("\(name): recording never started")
            return
        }
        print("VOXT-SPEAK \(name) edit-shorter")
        NSLog("VOXT-SPEAK %@ edit-shorter", name)
        sleep(4)
        stop.tap()
        let stopped = Date()
        var edited = ""
        while Date().timeIntervalSince(stopped) < 60 {
            let now = noteText()
            if now != before, now.contains(expected) { edited = now; break }
            Thread.sleep(forTimeInterval: 0.25)
        }
        _ = shot(name + "-b-edited")
        XCTAssertFalse(edited.isEmpty, "\(name): the latest result was not replaced")
        let undo = notes.buttons["keyboardUndo"]
        var restored = false
        if undo.waitForExistence(timeout: 5) {
            undo.tap()
            let tapped = Date()
            while Date().timeIntervalSince(tapped) < 5, !restored {
                restored = noteText() == before
                if !restored { Thread.sleep(forTimeInterval: 0.25) }
            }
        }
        _ = shot(name + "-c-undone")
        steps.append(Step(name: name, recordingStarted: true, insertedText: edited.isEmpty ? noteText() : edited,
                          stopToTextSeconds: edited.isEmpty ? nil : Date().timeIntervalSince(stopped),
                          indicatorPixels: 0, note: "undoShown=\(undo.exists || restored) restored=\(restored) before=\(before)"))
        XCTAssertTrue(restored, "\(name): Undo did not restore the text")
    }

    private func dictate(_ name: String, clip: String = "dictation", timeout: TimeInterval = 60) {
        let mic = showVoxtKeyboard()
        // The keyboard keeps the user's last mode; this check needs plain dictation.
        guard mic.exists else {
            steps.append(Step(name: name, indicatorPixels: shot(name + "-no-keyboard"), note: "Voxt keyboard not found"))
            XCTFail("\(name): Voxt keyboard not found")
            return
        }
        let before = noteText()
        _ = shot(name + "-a-keyboard")
        mic.tap()
        // A closed microphone hands off to Voxt first; that can take more than 2 s.
        let voxtInFront = voxt.wait(for: .runningForeground, timeout: 5)
        if voxtInFront {
            _ = shot(name + "-b-voxt-in-front")
            notes.activate()
        }
        let stop = notes.buttons["Stop dictation"]
        let started = stop.waitForExistence(timeout: 10)
        guard started else {
            steps.append(Step(name: name, voxtInFront: voxtInFront, recordingStarted: false, indicatorPixels: shot(name + "-c-no-recording")))
            XCTFail("\(name): recording never started")
            return
        }
        print("VOXT-SPEAK \(name) \(clip)")
        NSLog("VOXT-SPEAK %@ %@", name, clip)
        sleep(2)
        let recordingPixels = shot(name + "-c-recording")
        sleep(5)
        stop.tap()
        let stopped = Date()
        var inserted = ""
        while Date().timeIntervalSince(stopped) < timeout {
            let now = noteText()
            if now != before, now.contains(expected) { inserted = now; break }
            Thread.sleep(forTimeInterval: 0.25)
        }
        let seconds = inserted.isEmpty ? nil : Date().timeIntervalSince(stopped)
        let afterPixels = shot(name + "-d-after")
        let keyboardMessage = notes.staticTexts.matching(NSPredicate(
            format: "label CONTAINS 'No speech' OR label CONTAINS 'Text is ready'")).firstMatch
        let shown = keyboardMessage.exists ? keyboardMessage.label : ""
        steps.append(Step(name: name, voxtInFront: voxtInFront, recordingStarted: true,
                          insertedText: inserted.isEmpty ? noteText() : String(inserted.dropFirst(before.count)),
                          stopToTextSeconds: seconds, indicatorPixels: afterPixels,
                          note: "recording frame indicatorPixels=\(recordingPixels)" + (shown.isEmpty ? "" : " keyboard=\(shown)")))
        XCTAssertFalse(inserted.isEmpty, "\(name): no text containing \(expected) within \(Int(timeout)) s \(shown)")
    }

    /// Saves a full-screen screenshot and returns the count of indicator-orange pixels in the
    /// status bar. Interpreted only after checking a recording frame against an idle frame.
    @discardableResult
    private func shot(_ name: String) -> Int {
        let screenshot = XCUIScreen.main.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
        guard let image = screenshot.image.cgImage else { return -1 }
        // Status bar only (iPhone SE: 20 pt); it turns orange while the microphone session runs.
        let width = image.width, height = max(1, image.height * 3 / 100)
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        guard let context = CGContext(data: &pixels, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                                      space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return -1 }
        context.draw(image, in: CGRect(x: 0, y: height - image.height, width: width, height: image.height))
        var count = 0
        for index in stride(from: 0, to: pixels.count, by: 4) {
            let r = Int(pixels[index]), g = Int(pixels[index + 1]), b = Int(pixels[index + 2])
            if r > 220, g > 110, g < 190, b < 80 { count += 1 }
        }
        print("VOXT-SHOT \(name) indicatorPixels=\(count)")
        return count
    }

    private func report() {
        let data = (try? JSONEncoder().encode(steps)) ?? Data()
        let json = String(decoding: data, as: UTF8.self)
        print("VOXT-RESULT \(json)")
        let attachment = XCTAttachment(string: json)
        attachment.name = "device-pip-steps.json"
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
