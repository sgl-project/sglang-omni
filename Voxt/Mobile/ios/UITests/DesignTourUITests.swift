import XCTest

/// Synthetic visual baseline, simulator only. The loopback fixture never calls the Mac.
@MainActor
final class DesignTourUITests: XCTestCase {
    private let env = ProcessInfo.processInfo.environment
    private var base: String { env["VOXT_DESIGN_FIXTURE_URL"]! }
    private var appearance: String { env["VOXT_TOUR_APPEARANCE"] ?? "light" }
    private let result = "This is a design review dictation."

    override func setUpWithError() throws {
        continueAfterFailure = false
        #if !targetEnvironment(simulator)
        throw XCTSkip("Design tour is simulator-only")
        #endif
        try XCTSkipIf(env["VOXT_DESIGN_FIXTURE_URL"] == nil, "Start tools/design-tour-fixture.py")
        XCTAssertTrue(base.hasPrefix("http://127.0.0.1:"))
    }

    func testUnpairedScreenAccessibilityQueryReturns() {
        let app = XCUIApplication()
        app.launchEnvironment = ["VOXT_TEST_RESET": "1"]
        if let wav = env["VOXT_E2E_WAV"] { app.launchEnvironment["VOXT_TEST_AUDIO_FILE"] = wav }
        app.launch()
        let started = Date()
        XCTAssertTrue(app.staticTexts["onboardingTitle"].waitForExistence(timeout: 5))
        XCTAssertFalse(app.buttons["manualPairing"].exists)
        XCTAssertFalse(app.debugDescription.isEmpty)
        XCTAssertLessThan(Date().timeIntervalSince(started), 10)
        snap("unpaired-accessibility", app)
    }

    private func snap(_ state: String, _ app: XCUIApplication) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = "tour-\(appearance)-\(state)"
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    private func control(failure: Bool = false, delay: Double = 5) async throws {
        var request = URLRequest(url: URL(string: base + "/control")!)
        request.httpMethod = "POST"
        request.setValue("Bearer design-review", forHTTPHeaderField: "Authorization")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["failure": failure, "delay": delay])
        let (_, response) = try await URLSession.shared.data(for: request)
        XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 200)
    }

    private func pair(_ app: XCUIApplication, address: String) {
        app.terminate()
        app.launchEnvironment["VOXT_TEST_SERVER"] = String(data: try! JSONSerialization.data(withJSONObject: ["baseURL": address, "token": "design-review"]), encoding: .utf8)!
        app.launch()
        XCTAssertTrue(app.buttons["Settings"].waitForExistence(timeout: 8))
    }

    private func reveal(_ element: XCUIElement, in app: XCUIApplication) {
        for _ in 0..<7 {
            if element.exists && element.isHittable { return }
            app.swipeUp()
        }
        for _ in 0..<9 {
            if element.exists && element.isHittable { return }
            app.swipeDown()
        }
        XCTAssertTrue(element.isHittable)
    }

    private func label(_ element: XCUIElement, equals value: String) {
        let wait = XCTNSPredicateExpectation(predicate: NSPredicate(format: "label == %@", value), object: element)
        XCTAssertEqual(XCTWaiter.wait(for: [wait], timeout: 12), .completed)
    }

    private func waitForVoiceLevel(_ mic: XCUIElement) {
        let audible = XCTNSPredicateExpectation(predicate: NSPredicate { _, _ in
            let value = (mic.value as? String)?.split(separator: " ").last.flatMap { Float($0) } ?? 0
            return value > 0.10
        }, object: mic)
        XCTAssertEqual(XCTWaiter.wait(for: [audible], timeout: 10), .completed)
    }

    /// A fresh simulator shows two one-time iOS tips over the keyboard: "Quickly Change Keyboards"
    /// after the first switch, and the slide-to-type tip over the first system keyboard. Continue closes either.
    private func dismissKeyboardTip(_ app: XCUIApplication) {
        let tip = app.staticTexts.matching(NSPredicate(format: "label == 'Quickly Change Keyboards' OR label BEGINSWITH 'Speed up your typing by sliding'")).firstMatch
        let close = app.buttons["Continue"].firstMatch
        if tip.waitForExistence(timeout: 1), close.exists { close.tap() }
    }

    /// Opens the fixture's text page in Safari. Safari's first cold launch on a fresh simulator can
    /// leave the page blank or still loading over its Start Page; a reload or opening it again loads it.
    private func openTestPage(_ safari: XCUIApplication, base: String) {
        let field = safari.textViews["Session test"]
        for _ in 0..<3 {
            safari.open(URL(string: base + "/keyboard-test")!)
            if field.waitForExistence(timeout: 15) { break }
            let reload = safari.buttons["ReloadButton"]
            if reload.exists { reload.tap() }
            if field.waitForExistence(timeout: 10) { break }
        }
        // Safari's one-time toolbar tip takes the next tap, which would leave the field unfocused.
        let tip = safari.staticTexts["View Bookmarks, Share Menu, and Open Tabs"]
        if tip.waitForExistence(timeout: 2) {
            safari.buttons["xmark.circle.fill"].firstMatch.tap()
            XCTAssertTrue(tip.waitForNonExistence(timeout: 3))
        }
    }

    private func keyboardSettings(enabled: Bool) {
        let settings = XCUIApplication(bundleIdentifier: "com.apple.Preferences")
        settings.terminate()
        settings.launch()
        func tap(_ element: XCUIElement) {
            if !element.waitForExistence(timeout: 3) { settings.swipeUp() }
            XCTAssertTrue(element.waitForExistence(timeout: 3))
            element.tap()
        }
        tap(settings.staticTexts["General"].firstMatch)
        tap(settings.staticTexts["Keyboard"].firstMatch)
        tap(settings.cells["KEYBOARDS"].firstMatch)
        XCTAssertTrue(settings.navigationBars["Keyboards"].waitForExistence(timeout: 5))
        let row = settings.cells.containing(.staticText, identifier: "Voxt").firstMatch
        if !enabled {
            if row.exists {
                row.swipeLeft()
                tap(settings.buttons["Delete"].firstMatch)
                XCTAssertFalse(row.exists)
            }
        } else {
            if !row.exists {
                tap(settings.cells["AddNewKeyboard"].firstMatch)
                tap(settings.staticTexts["Voxt"].firstMatch)
            }
            tap(settings.staticTexts["Voxt"].firstMatch)
            let fullAccess = settings.switches.firstMatch
            XCTAssertTrue(fullAccess.waitForExistence(timeout: 5))
            if fullAccess.value as? String != "1" {
                fullAccess.coordinate(withNormalizedOffset: CGVector(dx: 0.9, dy: 0.5)).tap()
                let allow = settings.alerts.buttons["Allow"]
                if allow.waitForExistence(timeout: 3) { allow.tap() }
            }
            XCTAssertEqual(fullAccess.value as? String, "1")
        }
        settings.terminate()
    }

    /// Regression: warming the external keyboard and cancelling must leave the session
    /// reachable for the next recording, including after the app handoff is dismissed.
    func testExternalKeyboardWarmupCancelThenRecord() async throws {
        try await externalKeyboardFlow(warmup: true)
    }

    func testFirstHandoffInsertsIntoSameField() async throws {
        try await externalKeyboardFlow(warmup: false)
    }

    func testChangedFieldAfterHandoffOffersCopy() async throws {
        try await externalKeyboardFlow(warmup: false, changeField: true)
    }

    private func externalKeyboardFlow(warmup: Bool, changeField: Bool = false) async throws {
        try await control(delay: 0.2)
        let app = XCUIApplication()
        app.launchEnvironment = ["VOXT_TEST_RESET": "1", "VOXT_TEST_AUDIO_FILE": try XCTUnwrap(env["VOXT_E2E_WAV"])]
        app.launch()
        keyboardSettings(enabled: true)
        app.activate()
        pair(app, address: base)
        XCTAssertTrue(app.descendants(matching: .any)["connectedLabel"].waitForExistence(timeout: 10))
        let safari = XCUIApplication(bundleIdentifier: "com.apple.mobilesafari")
        safari.terminate()
        openTestPage(safari, base: base)
        let field = safari.textViews["Session test"]
        XCTAssertTrue(field.waitForExistence(timeout: 10))
        field.tap()
        let start = safari.buttons["Start dictation"]
        if !start.waitForExistence(timeout: 2), safari.buttons["Next keyboard"].firstMatch.exists {
            safari.buttons["Next keyboard"].firstMatch.press(forDuration: 1.2)
            let choice = safari.descendants(matching: .any).matching(NSPredicate(format: "label == 'Voxt'")).firstMatch
            if choice.waitForExistence(timeout: 3) { choice.tap() }
        }
        for _ in 0..<8 {
            if start.waitForExistence(timeout: 1) { break }
            dismissKeyboardTip(safari)
            if safari.buttons["textLayout-voice"].exists { safari.buttons["textLayout-voice"].tap() }
            else if safari.buttons["Next keyboard"].firstMatch.exists { safari.buttons["Next keyboard"].firstMatch.tap(); dismissKeyboardTip(safari) }
        }
        if !start.exists { print(safari.debugDescription) }
        XCTAssertTrue(start.exists)
        if changeField {
            // Populate through the keyboard so the request captures known context.
            field.typeText("Original text")
            XCTAssertEqual(field.value as? String, "Original text")
        }
        start.tap()
        XCTAssertTrue(app.wait(for: .runningForeground, timeout: 5))
        XCTAssertTrue(app.buttons["keyboardHandoffMic"].waitForExistence(timeout: 10))
        safari.activate()
        if warmup {
        let cancel = safari.buttons["keyboardCancelComposition"]
        XCTAssertTrue(cancel.waitForExistence(timeout: 5))
        cancel.tap()
        XCTAssertTrue(start.waitForExistence(timeout: 5))
        // Allow iOS to suspend an app without a live background audio session.
        try await Task.sleep(for: .seconds(7))
        snap("external-keyboard-warmed", safari)
        start.tap()
        }
        XCTAssertTrue(safari.buttons["Stop dictation"].waitForExistence(timeout: 5),
                      "Warm keyboard handed off again instead of recording in the original app")
        XCTAssertEqual(safari.state, .runningForeground)
        try await Task.sleep(for: .seconds(1))
        if changeField {
            let other = safari.textViews["Other field"]
            XCTAssertTrue(other.waitForExistence(timeout: 5))
            other.tap()
            XCTAssertEqual(other.value as? String, "Different text")
        }
        safari.buttons["Stop dictation"].tap()
        if changeField {
            let copy = safari.buttons["keyboardCopyResult"]
            if !copy.waitForExistence(timeout: 15) { print(safari.debugDescription) }
            XCTAssertTrue(copy.exists)
            XCTAssertFalse((field.value as? String ?? "").contains(result))
            XCTAssertFalse(safari.staticTexts["Writing it up…"].exists)
            snap("handoff-copy", safari)
            return
        }
        let inserted = XCTNSPredicateExpectation(predicate: NSPredicate(format: "value CONTAINS %@", result), object: field)
        XCTAssertEqual(XCTWaiter.wait(for: [inserted], timeout: 15), .completed)
        var stats = URLRequest(url: URL(string: base + "/stats")!)
        stats.setValue("Bearer design-review", forHTTPHeaderField: "Authorization")
        let (data, _) = try await URLSession.shared.data(for: stats)
        let counts = try JSONDecoder().decode([String: Int].self, from: data)
        XCTAssertGreaterThan(counts["composeCount"] ?? 0, 0)
        snap("external-keyboard-inserted", safari)
    }

    /// Voxt paired with the fixture, Safari on the test page with the Voxt keyboard up.
    private func safariWithVoxtKeyboard() async throws -> (app: XCUIApplication, safari: XCUIApplication, field: XCUIElement, start: XCUIElement) {
        try await control(delay: 0.5)
        let app = XCUIApplication()
        app.launchEnvironment = ["VOXT_TEST_RESET": "1", "VOXT_TEST_AUDIO_FILE": try XCTUnwrap(env["VOXT_E2E_WAV"])]
        app.launch()
        keyboardSettings(enabled: true)
        app.activate()
        pair(app, address: base)
        XCTAssertTrue(app.descendants(matching: .any)["connectedLabel"].waitForExistence(timeout: 10))
        let safari = XCUIApplication(bundleIdentifier: "com.apple.mobilesafari")
        safari.terminate()
        openTestPage(safari, base: base)
        let field = safari.textViews["Session test"]
        XCTAssertTrue(field.waitForExistence(timeout: 10))
        field.tap()
        let start = safari.buttons["Start dictation"]
        for _ in 0..<8 {
            if start.waitForExistence(timeout: 1) { break }
            dismissKeyboardTip(safari)
            if safari.buttons["textLayout-voice"].exists { safari.buttons["textLayout-voice"].tap() }
            else if safari.buttons["Next keyboard"].firstMatch.exists { safari.buttons["Next keyboard"].firstMatch.tap(); dismissKeyboardTip(safari) }
        }
        XCTAssertTrue(start.exists)
        return (app, safari, field, start)
    }

    /// While recording, the keyboard shows only the mic, its caption and an X.
    func testRecordingShowsOnlyTheMicTheCaptionAndCancel() async throws {
        let (app, safari, _, start) = try await safariWithVoxtKeyboard()
        XCTAssertTrue(safari.buttons["keyboardPen"].exists)
        start.tap()
        if app.wait(for: .runningForeground, timeout: 5) {
            XCTAssertTrue(app.buttons["keyboardHandoffMic"].waitForExistence(timeout: 10))
            safari.activate()
        }
        let stop = safari.buttons["Stop dictation"]
        XCTAssertTrue(stop.waitForExistence(timeout: 10))
        XCTAssertTrue(safari.staticTexts["Tap again to finish"].exists)
        XCTAssertTrue(safari.buttons["keyboardCancelComposition"].isHittable, "the X cancels")
        for hidden in ["keyboardPen", "voiceLayoutToolbar", "Return", "keyboardUndo"] {
            XCTAssertFalse(safari.descendants(matching: .any)[hidden].exists, "\(hidden) is hidden while recording")
        }
        snap("recording-view", safari)
        safari.buttons["keyboardCancelComposition"].tap()
        XCTAssertTrue(start.waitForExistence(timeout: 5))
        XCTAssertTrue(safari.buttons["keyboardPen"].waitForExistence(timeout: 3), "the keys come back when recording ends")
    }

    /// After a dictation, ↶ at the top left takes it out again, once.
    func testUndoTakesBackAJustDictatedResult() async throws {
        let (app, safari, field, start) = try await safariWithVoxtKeyboard()
        start.tap()
        if app.wait(for: .runningForeground, timeout: 5) {
            XCTAssertTrue(app.buttons["keyboardHandoffMic"].waitForExistence(timeout: 10))
            safari.activate()
        }
        let stop = safari.buttons["Stop dictation"]
        XCTAssertTrue(stop.waitForExistence(timeout: 10))
        try await Task.sleep(for: .seconds(1))
        stop.tap()
        let inserted = XCTNSPredicateExpectation(predicate: NSPredicate(format: "value CONTAINS %@", result), object: field)
        let outcome = await XCTWaiter().fulfillment(of: [inserted], timeout: 20)
        XCTAssertEqual(outcome, .completed)
        let undo = safari.buttons["keyboardUndo"]
        XCTAssertTrue(undo.waitForExistence(timeout: 5), "↶ after an insert")
        snap("undo-offered", safari)
        undo.tap()
        let empty = XCTNSPredicateExpectation(predicate: NSPredicate(format: "NOT (value CONTAINS %@)", result), object: field)
        let restored = await XCTWaiter().fulfillment(of: [empty], timeout: 5)
        XCTAssertEqual(restored, .completed, "field: \(field.value as? String ?? "")")
        XCTAssertFalse(undo.waitForExistence(timeout: 2), "one undo only")
    }

    /// Hold the mic, slide onto a language pill, let go: that one recording translates
    /// into the pill's language; the next tap is plain dictation again.
    func testHoldingTheMicAndSlidingToAPillTranslatesOneRecording() async throws {
        try await control(delay: 0.5)
        let app = XCUIApplication()
        app.launchEnvironment = ["VOXT_TEST_RESET": "1", "VOXT_TEST_AUDIO_FILE": try XCTUnwrap(env["VOXT_E2E_WAV"])]
        app.launch()
        keyboardSettings(enabled: true)
        app.activate()
        pair(app, address: base)
        XCTAssertTrue(app.descendants(matching: .any)["connectedLabel"].waitForExistence(timeout: 10))
        app.buttons["Settings"].tap()
        let languages = app.buttons["translationLanguages"]
        reveal(languages, in: app)
        languages.tap()
        app.buttons["addLanguage"].tap()
        let spanish = app.buttons["languageOption-es"]
        XCTAssertTrue(spanish.waitForExistence(timeout: 5))
        if !spanish.isSelected { spanish.tap() }
        app.buttons["addLanguageDone"].tap()
        app.navigationBars.buttons.element(boundBy: 0).tap()
        // A reset seeds targets from the simulator's other languages (zh-Hans here), so find
        // where Español sits: the Settings row lists the targets in pill order.
        let order = languages.label.components(separatedBy: ", ")
        let firstTarget = order.firstIndex { $0.hasSuffix("Languages") }.map { $0 + 1 } ?? 0
        let pill = try XCTUnwrap(order.firstIndex { $0.contains("Español") }, "targets: \(languages.label)") - firstTarget
        XCTAssertTrue((0..<3).contains(pill), "targets: \(languages.label)")
        app.buttons["Done"].firstMatch.tap()

        let safari = XCUIApplication(bundleIdentifier: "com.apple.mobilesafari")
        safari.terminate()
        openTestPage(safari, base: base)
        let field = safari.textViews["Session test"]
        XCTAssertTrue(field.waitForExistence(timeout: 10))
        field.tap()
        let start = safari.buttons["Start dictation"]
        for _ in 0..<8 {
            if start.waitForExistence(timeout: 1) { break }
            dismissKeyboardTip(safari)
            if safari.buttons["textLayout-voice"].exists { safari.buttons["textLayout-voice"].tap() }
            else if safari.buttons["Next keyboard"].firstMatch.exists { safari.buttons["Next keyboard"].firstMatch.tap(); dismissKeyboardTip(safari) }
        }
        XCTAssertTrue(start.exists)
        // Hold past the delay, slide onto Español's pill (1st above, 2nd up-left, 3rd up-right), let go.
        let mic = start.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
        let pillOffsets = [CGVector(dx: 0, dy: -82), CGVector(dx: -58, dy: -58), CGVector(dx: 58, dy: -58)]
        mic.press(forDuration: 0.7, thenDragTo: mic.withOffset(pillOffsets[pill]))
        if app.wait(for: .runningForeground, timeout: 5) {
            XCTAssertTrue(app.buttons["keyboardHandoffMic"].waitForExistence(timeout: 10))
            safari.activate()
        }
        let stop = safari.buttons["Stop dictation"]
        XCTAssertTrue(stop.waitForExistence(timeout: 10), "letting go on the pill starts recording")
        XCTAssertTrue(safari.staticTexts["Translating to Español"].waitForExistence(timeout: 3))
        snap("translate-pill-recording", safari)
        try await Task.sleep(for: .seconds(1))
        stop.tap()
        let inserted = XCTNSPredicateExpectation(predicate: NSPredicate(format: "value CONTAINS %@", result), object: field)
        let outcome = await XCTWaiter().fulfillment(of: [inserted], timeout: 20)
        XCTAssertEqual(outcome, .completed)
        func lastOptions() async throws -> [String: Any] {
            var request = URLRequest(url: URL(string: base + "/last-options")!)
            request.setValue("Bearer design-review", forHTTPHeaderField: "Authorization")
            let (data, _) = try await URLSession.shared.data(for: request)
            return try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        }
        let translated = try await lastOptions()
        XCTAssertEqual(translated["mode"] as? String, "translate")
        XCTAssertEqual(translated["targetLanguage"] as? String, "es")

        // The next tap dictates.
        XCTAssertTrue(start.waitForExistence(timeout: 10))
        start.tap()
        XCTAssertTrue(stop.waitForExistence(timeout: 10))
        try await Task.sleep(for: .seconds(1))
        stop.tap()
        try await Task.sleep(for: .seconds(3))
        let plain = try await lastOptions()
        XCTAssertEqual(plain["mode"] as? String, "dictate", "a pill applies to one recording")
    }

    /// The handoff screen shows only what to do — swipe back along the bottom — and how
    /// to skip it next time; Cancel and Keep Voxt ready are the only buttons.
    func testHandoffScreenShowsHowToGoBackAndHowToSkipIt() async throws {
        try await control(delay: 1.0)
        let app = XCUIApplication()
        app.launchEnvironment = ["VOXT_TEST_RESET": "1", "VOXT_TEST_AUDIO_FILE": try XCTUnwrap(env["VOXT_E2E_WAV"])]
        app.launch()
        keyboardSettings(enabled: true)
        app.activate()
        pair(app, address: base)
        XCTAssertTrue(app.descendants(matching: .any)["connectedLabel"].waitForExistence(timeout: 10))
        let safari = XCUIApplication(bundleIdentifier: "com.apple.mobilesafari")
        safari.terminate()
        openTestPage(safari, base: base)
        let field = safari.textViews["Session test"]
        XCTAssertTrue(field.waitForExistence(timeout: 10))
        field.tap()
        let start = safari.buttons["Start dictation"]
        for _ in 0..<8 {
            if start.waitForExistence(timeout: 1) { break }
            dismissKeyboardTip(safari)
            if safari.buttons["textLayout-voice"].exists { safari.buttons["textLayout-voice"].tap() }
            else if safari.buttons["Next keyboard"].firstMatch.exists { safari.buttons["Next keyboard"].firstMatch.tap(); dismissKeyboardTip(safari) }
        }
        start.tap()
        XCTAssertTrue(app.wait(for: .runningForeground, timeout: 6))
        XCTAssertTrue(app.staticTexts["Voxt is listening"].waitForExistence(timeout: 10))
        XCTAssertTrue(app.descendants(matching: .any)["keyboardHandoffSwipeHint"].exists)
        XCTAssertTrue(app.staticTexts["Turn on Keep Voxt ready to skip this"].exists)
        // The listening mic is a status picture here (hit testing off), not a control.
        let hittable = app.buttons.allElementsBoundByIndex.filter(\.isHittable).map(\.identifier).filter { $0 != "keyboardHandoffMic" }
        XCTAssertEqual(Set(hittable), ["keyboardHandoffKeepReady", "keyboardHandoffCancel"], "\(hittable)")
        snap("handoff-screen", app)
        app.buttons["keyboardHandoffCancel"].tap()
        XCTAssertFalse(app.staticTexts["Voxt is listening"].waitForExistence(timeout: 2))
    }

    /// Going back to the other app about one second after Voxt comes forward, without
    /// waiting for the handoff screen's mic, then tapping Stop must still send the request;
    /// otherwise the keyboard ends with "Voxt stopped…" and the Mac receives nothing.
    func testFirstHandoffWithAQuickReturnStillInserts() async throws {
        try await quickReturnHandoff(afterRelaunchWithSavedState: false)
    }

    /// The same with saved state: Voxt has dictated in an earlier session, then XCUITest
    /// terminates and relaunches it (same pairing, no VOXT_TEST_RESET) before the first tap.
    func testQuickReturnAfterARelaunchWithSavedStateStillInserts() async throws {
        try await quickReturnHandoff(afterRelaunchWithSavedState: true)
    }

    private func quickReturnHandoff(afterRelaunchWithSavedState: Bool) async throws {
        try await control(delay: 1.0)
        let app = XCUIApplication()
        app.launchEnvironment = ["VOXT_TEST_RESET": "1", "VOXT_TEST_AUDIO_FILE": try XCTUnwrap(env["VOXT_E2E_WAV"])]
        app.launch()
        keyboardSettings(enabled: true)
        app.activate()
        pair(app, address: base)
        XCTAssertTrue(app.descendants(matching: .any)["connectedLabel"].waitForExistence(timeout: 10))
        if afterRelaunchWithSavedState {
            try await dictateOnceInSafari(app)
            app.terminate()
            app.launchEnvironment.removeValue(forKey: "VOXT_TEST_RESET")
            app.launch()
            XCTAssertTrue(app.descendants(matching: .any)["connectedLabel"].waitForExistence(timeout: 10))
        }
        let safari = XCUIApplication(bundleIdentifier: "com.apple.mobilesafari")
        safari.terminate()
        openTestPage(safari, base: base)
        let field = safari.textViews["Session test"]
        XCTAssertTrue(field.waitForExistence(timeout: 10))
        field.tap()
        let start = safari.buttons["Start dictation"]
        if !start.waitForExistence(timeout: 2), safari.buttons["Next keyboard"].firstMatch.exists {
            safari.buttons["Next keyboard"].firstMatch.press(forDuration: 1.2)
            let choice = safari.descendants(matching: .any).matching(NSPredicate(format: "label == 'Voxt'")).firstMatch
            if choice.waitForExistence(timeout: 3) { choice.tap() }
        }
        for _ in 0..<8 {
            if start.waitForExistence(timeout: 1) { break }
            dismissKeyboardTip(safari)
            if safari.buttons["textLayout-voice"].exists { safari.buttons["textLayout-voice"].tap() }
            else if safari.buttons["Next keyboard"].firstMatch.exists { safari.buttons["Next keyboard"].firstMatch.tap(); dismissKeyboardTip(safari) }
        }
        XCTAssertTrue(start.exists)
        start.tap()
        XCTAssertTrue(app.wait(for: .runningForeground, timeout: 6))
        try await Task.sleep(for: .seconds(1))
        safari.activate()
        let stop = safari.buttons["Stop dictation"]
        XCTAssertTrue(stop.waitForExistence(timeout: 6), "back in the other app while recording")
        try await Task.sleep(for: .seconds(1))
        stop.tap()
        let inserted = XCTNSPredicateExpectation(predicate: NSPredicate(format: "value CONTAINS %@", result), object: field)
        let outcome = await XCTWaiter().fulfillment(of: [inserted], timeout: 20)
        snap("quick-return-after-stop", safari)
        XCTAssertEqual(outcome, .completed,
                       "keyboard shows: \(safari.staticTexts.allElementsBoundByIndex.map(\.label).prefix(6))")
    }

    /// One complete keyboard dictation from Safari, handoff included.
    private func dictateOnceInSafari(_ app: XCUIApplication) async throws {
        let safari = XCUIApplication(bundleIdentifier: "com.apple.mobilesafari")
        safari.terminate()
        openTestPage(safari, base: base)
        let field = safari.textViews["Session test"]
        XCTAssertTrue(field.waitForExistence(timeout: 10))
        field.tap()
        let start = safari.buttons["Start dictation"]
        for _ in 0..<8 {
            if start.waitForExistence(timeout: 1) { break }
            dismissKeyboardTip(safari)
            if safari.buttons["textLayout-voice"].exists { safari.buttons["textLayout-voice"].tap() }
            else if safari.buttons["Next keyboard"].firstMatch.exists { safari.buttons["Next keyboard"].firstMatch.tap(); dismissKeyboardTip(safari) }
        }
        start.tap()
        if app.wait(for: .runningForeground, timeout: 5) {
            XCTAssertTrue(app.buttons["keyboardHandoffMic"].waitForExistence(timeout: 10))
            safari.activate()
        }
        let stop = safari.buttons["Stop dictation"]
        XCTAssertTrue(stop.waitForExistence(timeout: 6))
        stop.tap()
        let inserted = XCTNSPredicateExpectation(predicate: NSPredicate(format: "value CONTAINS %@", result), object: field)
        let outcome = await XCTWaiter().fulfillment(of: [inserted], timeout: 20)
        XCTAssertEqual(outcome, .completed, "the warm-up dictation must insert")
    }

    /// A backgrounded app without an audio session is suspended after the handoff, so Stop
    /// never arrives and the keyboard shows "Tap again to finish" forever. File capture runs
    /// no audio session of its own, which makes this the case to test.
    func testRecordingSurvivesFifteenSecondsInTheBackgroundAfterHandoff() async throws {
        try await control(delay: 0.2)
        let app = XCUIApplication()
        app.launchEnvironment = ["VOXT_TEST_RESET": "1", "VOXT_TEST_AUDIO_FILE": try XCTUnwrap(env["VOXT_E2E_WAV"])]
        app.launch()
        keyboardSettings(enabled: true)
        app.activate()
        pair(app, address: base)
        XCTAssertTrue(app.descendants(matching: .any)["connectedLabel"].waitForExistence(timeout: 10))
        let safari = XCUIApplication(bundleIdentifier: "com.apple.mobilesafari")
        safari.terminate()
        openTestPage(safari, base: base)
        let field = safari.textViews["Session test"]
        XCTAssertTrue(field.waitForExistence(timeout: 10))
        field.tap()
        let start = safari.buttons["Start dictation"]
        if !start.waitForExistence(timeout: 2), safari.buttons["Next keyboard"].firstMatch.exists {
            safari.buttons["Next keyboard"].firstMatch.press(forDuration: 1.2)
            let choice = safari.descendants(matching: .any).matching(NSPredicate(format: "label == 'Voxt'")).firstMatch
            if choice.waitForExistence(timeout: 3) { choice.tap() }
        }
        for _ in 0..<8 {
            if start.waitForExistence(timeout: 1) { break }
            dismissKeyboardTip(safari)
            if safari.buttons["textLayout-voice"].exists { safari.buttons["textLayout-voice"].tap() }
            else if safari.buttons["Next keyboard"].firstMatch.exists { safari.buttons["Next keyboard"].firstMatch.tap(); dismissKeyboardTip(safari) }
        }
        XCTAssertTrue(start.exists)
        // The tour's sequence: warm the session (handoff, back, Cancel), go Home, stay idle, come back.
        start.tap()
        if app.wait(for: .runningForeground, timeout: 3) {
            XCTAssertTrue(app.buttons["keyboardHandoffMic"].waitForExistence(timeout: 10))
            safari.activate()
        }
        let cancel = safari.buttons["keyboardCancelComposition"]
        XCTAssertTrue(cancel.waitForExistence(timeout: 5))
        cancel.tap()
        XCUIDevice.shared.press(.home)
        try await Task.sleep(for: .seconds(30))
        safari.activate()
        if !start.waitForExistence(timeout: 3) { field.tap() }
        XCTAssertTrue(start.waitForExistence(timeout: 5))
        start.tap()
        XCTAssertTrue(safari.buttons["Stop dictation"].waitForExistence(timeout: 5),
                      "a warm session records without another handoff")
        XCTAssertNotEqual(app.state, .runningForeground, "Voxt records from the background")
        try await Task.sleep(for: .seconds(15))
        snap("background-15s-recording", safari)
        safari.buttons["Stop dictation"].tap()
        let inserted = XCTNSPredicateExpectation(predicate: NSPredicate(format: "value CONTAINS %@", result), object: field)
        XCTAssertEqual(XCTWaiter.wait(for: [inserted], timeout: 15), .completed,
                       "Stop must reach the backgrounded app; keyboard shows: \(safari.staticTexts.allElementsBoundByIndex.map(\.label).prefix(6))")
    }

    /// Consecutive dictations must not be glued together ("Friday.Please send").
    func testConsecutiveDictationsAreSeparatedByOneSpace() async throws {
        try await control(delay: 0.2)
        let app = XCUIApplication()
        app.launchEnvironment = ["VOXT_TEST_RESET": "1", "VOXT_TEST_AUDIO_FILE": try XCTUnwrap(env["VOXT_E2E_WAV"])]
        app.launch()
        keyboardSettings(enabled: true)
        app.activate()
        pair(app, address: base)
        XCTAssertTrue(app.descendants(matching: .any)["connectedLabel"].waitForExistence(timeout: 10))
        let safari = XCUIApplication(bundleIdentifier: "com.apple.mobilesafari")
        safari.terminate()
        openTestPage(safari, base: base)
        let field = safari.textViews["Session test"]
        XCTAssertTrue(field.waitForExistence(timeout: 10))
        field.tap()
        let start = safari.buttons["Start dictation"]
        if !start.waitForExistence(timeout: 2), safari.buttons["Next keyboard"].firstMatch.exists {
            safari.buttons["Next keyboard"].firstMatch.press(forDuration: 1.2)
            let choice = safari.descendants(matching: .any).matching(NSPredicate(format: "label == 'Voxt'")).firstMatch
            if choice.waitForExistence(timeout: 3) { choice.tap() }
        }
        for _ in 0..<8 {
            if start.waitForExistence(timeout: 1) { break }
            dismissKeyboardTip(safari)
            if safari.buttons["textLayout-voice"].exists { safari.buttons["textLayout-voice"].tap() }
            else if safari.buttons["Next keyboard"].firstMatch.exists { safari.buttons["Next keyboard"].firstMatch.tap(); dismissKeyboardTip(safari) }
        }
        XCTAssertTrue(start.exists)
        for expected in [result, result + " " + result] {
            start.tap()
            if app.wait(for: .runningForeground, timeout: 3) {
                XCTAssertTrue(app.buttons["keyboardHandoffMic"].waitForExistence(timeout: 10))
                safari.activate()
            }
            XCTAssertTrue(safari.buttons["Stop dictation"].waitForExistence(timeout: 10))
            try await Task.sleep(for: .seconds(1))
            safari.buttons["Stop dictation"].tap()
            let done = XCTNSPredicateExpectation(predicate: NSPredicate(format: "value == %@", expected), object: field)
            XCTAssertEqual(XCTWaiter.wait(for: [done], timeout: 15), .completed, "field: \(field.value as? String ?? "")")
        }
        snap("consecutive-dictations", safari)
    }

    /// With no selection, Edit changes only our own
    /// last result while it is verified at the cursor; one Undo restores it.
    func testPenWritesEditsTheLastResultOrTheSelectionWithOneUndo() async throws {
        try await control(delay: 0.2)
        let app = XCUIApplication()
        app.launchEnvironment = ["VOXT_TEST_RESET": "1", "VOXT_TEST_AUDIO_FILE": try XCTUnwrap(env["VOXT_E2E_WAV"])]
        app.launch()
        keyboardSettings(enabled: true)
        app.activate()
        pair(app, address: base)
        XCTAssertTrue(app.descendants(matching: .any)["connectedLabel"].waitForExistence(timeout: 10))
        let safari = XCUIApplication(bundleIdentifier: "com.apple.mobilesafari")
        safari.terminate()
        openTestPage(safari, base: base)
        let field = safari.textViews["Session test"]
        XCTAssertTrue(field.waitForExistence(timeout: 10))
        field.tap()
        let start = safari.buttons["Start dictation"]
        if !start.waitForExistence(timeout: 2), safari.buttons["Next keyboard"].firstMatch.exists {
            safari.buttons["Next keyboard"].firstMatch.press(forDuration: 1.2)
            let choice = safari.descendants(matching: .any).matching(NSPredicate(format: "label == 'Voxt'")).firstMatch
            if choice.waitForExistence(timeout: 3) { choice.tap() }
        }
        for _ in 0..<8 {
            if start.waitForExistence(timeout: 1) { break }
            dismissKeyboardTip(safari)
            if safari.buttons["textLayout-voice"].exists { safari.buttons["textLayout-voice"].tap() }
            else if safari.buttons["Next keyboard"].firstMatch.exists { safari.buttons["Next keyboard"].firstMatch.tap(); dismissKeyboardTip(safari) }
        }
        XCTAssertTrue(start.exists)
        let pen = safari.buttons["keyboardPen"]
        XCTAssertFalse(safari.buttons["Writing options"].exists, "the keyboard has no mode menu: a plain mic tap dictates")
        func composeCount() async throws -> Int {
            var request = URLRequest(url: URL(string: base + "/stats")!)
            request.setValue("Bearer design-review", forHTTPHeaderField: "Authorization")
            return try JSONDecoder().decode([String: Int].self, from: try await URLSession.shared.data(for: request).0)["composeCount"] ?? -1
        }

        // A. Nothing selected and nothing of ours at the cursor: the pen offers to write.
        //    Tapping it again disarms it; nothing is sent and the field is untouched.
        let before = try await composeCount()
        pen.tap()
        XCTAssertTrue(safari.staticTexts["Describe what to write"].waitForExistence(timeout: 5))
        snap("pen-write", safari)
        pen.tap()
        XCTAssertTrue(safari.staticTexts["Tap to speak"].waitForExistence(timeout: 3))
        let afterAttempt = try await composeCount()
        XCTAssertEqual(afterAttempt, before)
        XCTAssertEqual(field.value as? String ?? "", "")

        // B. Dictate a result into the field (first tap hands off to start the microphone).
        start.tap()
        if app.wait(for: .runningForeground, timeout: 5) {
            XCTAssertTrue(app.buttons["keyboardHandoffMic"].waitForExistence(timeout: 10))
            safari.activate()
        }
        XCTAssertTrue(safari.buttons["Stop dictation"].waitForExistence(timeout: 10))
        try await Task.sleep(for: .seconds(1))
        safari.buttons["Stop dictation"].tap()
        let inserted = XCTNSPredicateExpectation(predicate: NSPredicate(format: "value CONTAINS %@", result), object: field)
        XCTAssertEqual(XCTWaiter.wait(for: [inserted], timeout: 15), .completed)

        // C. The pen now targets that result and says so before and while listening.
        pen.tap()
        XCTAssertTrue(safari.staticTexts["Editing what you just said"].waitForExistence(timeout: 3))
        snap("pen-last-result", safari)
        start.tap()
        XCTAssertTrue(safari.buttons["Stop dictation"].waitForExistence(timeout: 10))
        XCTAssertTrue(safari.staticTexts["Editing your last result. Tap again to finish."].waitForExistence(timeout: 3))
        snap("edit-last-result-listening", safari)
        try await Task.sleep(for: .seconds(1))
        safari.buttons["Stop dictation"].tap()
        let edited = "This is an edited design review dictation."
        let replaced = XCTNSPredicateExpectation(predicate: NSPredicate(format: "value == %@", edited), object: field)
        XCTAssertEqual(XCTWaiter.wait(for: [replaced], timeout: 15), .completed, "field: \(field.value as? String ?? "")")
        var lastOptions = URLRequest(url: URL(string: base + "/last-options")!)
        lastOptions.setValue("Bearer design-review", forHTTPHeaderField: "Authorization")
        // Values can be null (customPrompt when Developer mode is off), so read loosely.
        let (optionsData, _) = try await URLSession.shared.data(for: lastOptions)
        let options = try XCTUnwrap(try JSONSerialization.jsonObject(with: optionsData) as? [String: Any])
        XCTAssertEqual(options["mode"] as? String, "edit")
        XCTAssertEqual(options["selectedText"] as? String, result, "only our last result was sent as the edit target")

        // D. One Undo restores the original.
        let undo = safari.buttons["keyboardUndo"]
        XCTAssertTrue(undo.waitForExistence(timeout: 5))
        snap("edit-undo-offered", safari)
        undo.tap()
        let restored = XCTNSPredicateExpectation(predicate: NSPredicate(format: "value == %@", result), object: field)
        XCTAssertEqual(XCTWaiter.wait(for: [restored], timeout: 5), .completed, "field: \(field.value as? String ?? "")")
        XCTAssertFalse(undo.waitForExistence(timeout: 2), "one undo only")
        snap("edit-undone", safari)

        // E. A selection wins: the pen edits the selected words.
        field.doubleTap()
        pen.tap()
        XCTAssertTrue(safari.staticTexts["Editing your selection"].waitForExistence(timeout: 5))
        snap("pen-selection", safari)
        pen.tap()
    }

    func testKeyboardShowsOnlyEnabledLanguages() async throws {
        let app = XCUIApplication()
        app.launchEnvironment = ["VOXT_TEST_RESET": "1", "VOXT_TEST_PRACTICE_FIELD": "1"]
        app.launch()
        keyboardSettings(enabled: true)
        app.activate()
        pair(app, address: base)
        app.buttons["Settings"].tap()
        let layouts = app.buttons["Text keyboards"]
        reveal(layouts, in: app)
        layouts.tap()
        let french = app.buttons["addLayout-fr-AZERTY"]
        XCTAssertTrue(french.waitForExistence(timeout: 5))
        french.tap()
        XCTAssertTrue(app.buttons["removeLayout-fr-AZERTY"].waitForExistence(timeout: 3))
        for id in ["de-QWERTZ", "es-QWERTY", "ja-flick", "zh-Hant-zhuyin", "ko-2beolsik", "zh-Hans-pinyin"] {
            let remove = app.buttons["removeLayout-" + id]
            if remove.exists { remove.tap() }
        }
        snap("language-settings", app)
        app.navigationBars["Keyboard languages"].buttons["Done"].tap()
        app.navigationBars["Settings"].buttons["Done"].tap()
        let field = app.textFields["tryField"]
        reveal(field, in: app)
        field.tap()
        dismissKeyboardTip(app)
        let start = app.buttons["Start dictation"]
        if !start.waitForExistence(timeout: 2), app.buttons["Next keyboard"].firstMatch.exists {
            app.buttons["Next keyboard"].firstMatch.press(forDuration: 1.2)
            let voxt = app.descendants(matching: .any).matching(NSPredicate(format: "label == 'Voxt'")).firstMatch
            if voxt.waitForExistence(timeout: 3) { voxt.tap() }
        }
        XCTAssertTrue(start.waitForExistence(timeout: 5))
        app.buttons["Show text keyboard"].tap()
        let toolbar = app.otherElements["textLayoutToolbar"]
        XCTAssertTrue(toolbar.waitForExistence(timeout: 5))
        XCTAssertEqual(toolbar.buttons.count, 3, "Exactly voice plus the two enabled layouts")
        XCTAssertEqual(toolbar.scrollViews.count, 0)
        for id in ["en-QWERTY", "fr-AZERTY"] {
            XCTAssertTrue(toolbar.buttons["textLayout-" + id].isHittable)
        }
        XCTAssertFalse(app.buttons["textKey-voice"].exists, "Only the toolbar mic returns to voice")
        XCTAssertGreaterThan(app.buttons["textKey-space"].frame.width, app.buttons["textKey-numbers"].frame.width * 1.8)
        snap("typing-languages", app)
        toolbar.buttons["textLayout-voice"].tap()
        let voiceLayouts = app.otherElements["voiceLayoutToolbar"]
        XCTAssertEqual(voiceLayouts.buttons.count, 2)
        XCTAssertEqual(voiceLayouts.buttons["voiceLayout-en-QWERTY"].label, "Show text keyboard")
        XCTAssertTrue(voiceLayouts.buttons["voiceLayout-fr-AZERTY"].isHittable)
        snap("voice-languages", app)
    }

    /// Settings > Developer: confirmation, editing the Dictate prompt, the prompt reaching the
    /// Mac only while Developer mode is on, and reset.
    /// Developer mode lives in Settings › Advanced.
    private func openAdvancedSettings(_ app: XCUIApplication) {
        let advanced = app.buttons["settingsAdvanced"]
        reveal(advanced, in: app)
        advanced.tap()
        XCTAssertTrue(app.navigationBars["Advanced"].waitForExistence(timeout: 5))
    }

    private func closeAdvancedSettings(_ app: XCUIApplication) {
        app.navigationBars["Advanced"].buttons.firstMatch.tap()
        app.navigationBars["Settings"].buttons["Done"].tap()
    }

    func testDeveloperModePromptReachesTheMacOnlyWhileOnAndResets() async throws {
        try await control(delay: 0.2)
        let app = XCUIApplication()
        app.launchEnvironment = ["VOXT_TEST_RESET": "1", "VOXT_TEST_AUDIO_FILE": try XCTUnwrap(env["VOXT_E2E_WAV"])]
        app.launch()
        keyboardSettings(enabled: true)
        app.activate()
        pair(app, address: base)
        app.buttons["Settings"].tap()
        openAdvancedSettings(app)
        let toggle = app.switches["developerModeToggle"]
        XCTAssertTrue(toggle.waitForExistence(timeout: 5))
        XCTAssertTrue(app.staticTexts["Lets you edit the prompts Voxt sends to your Mac. Asks to confirm before it turns on."].exists)
        tapSwitch(toggle)
        let confirm = app.alerts["Turn on Developer mode?"]
        XCTAssertTrue(confirm.waitForExistence(timeout: 3))
        XCTAssertTrue(confirm.staticTexts["Developer mode lets you edit the prompts Voxt uses. Custom prompts can lower quality."].exists)
        snap("developer-confirm", app)
        confirm.buttons["Turn On"].tap()
        let prompts = app.buttons["developerPrompts"]
        XCTAssertTrue(prompts.waitForExistence(timeout: 3))
        snap("developer-on", app)
        prompts.tap()
        let dictate = app.textViews["prompt-dictate"]
        XCTAssertTrue(dictate.waitForExistence(timeout: 5))
        let loaded = XCTNSPredicateExpectation(predicate: NSPredicate(format: "value CONTAINS %@", "Clean up the transcribed speech."), object: dictate)
        XCTAssertEqual(XCTWaiter.wait(for: [loaded], timeout: 8), .completed, "editor starts from the Mac's built-in prompt")
        XCTAssertFalse(app.staticTexts["customBadge-dictate"].exists)
        snap("developer-prompts", app)
        dictate.tap()
        dictate.typeText(" Be brief.")
        XCTAssertTrue(app.staticTexts["customBadge-dictate"].waitForExistence(timeout: 3))
        snap("developer-prompt-edited", app)
        app.navigationBars["Prompts"].buttons["Done"].tap()
        closeAdvancedSettings(app)

        let safari = XCUIApplication(bundleIdentifier: "com.apple.mobilesafari")
        func dictateOnce() async throws -> String? {
            safari.terminate()
            openTestPage(safari, base: base)
            let field = safari.textViews["Session test"]
            XCTAssertTrue(field.waitForExistence(timeout: 10))
            field.tap()
            let start = safari.buttons["Start dictation"]
            if !start.waitForExistence(timeout: 2), safari.buttons["Next keyboard"].firstMatch.exists {
                safari.buttons["Next keyboard"].firstMatch.press(forDuration: 1.2)
                let voxt = safari.descendants(matching: .any).matching(NSPredicate(format: "label == 'Voxt'")).firstMatch
                if voxt.waitForExistence(timeout: 3) { voxt.tap() }
            }
            XCTAssertTrue(start.waitForExistence(timeout: 5))
            let before = try await composeCount()
            start.tap()
            // With no live session the first tap opens Voxt to start the microphone; go back.
            if app.wait(for: .runningForeground, timeout: 3) {
                XCTAssertTrue(app.buttons["keyboardHandoffMic"].waitForExistence(timeout: 10))
                safari.activate()
            }
            let stop = safari.buttons["Stop dictation"]
            XCTAssertTrue(stop.waitForExistence(timeout: 5), "recording started")
            try await Task.sleep(for: .seconds(1))
            stop.tap()
            let deadline = Date().addingTimeInterval(15)
            while try await composeCount() == before, Date() < deadline { try await Task.sleep(for: .milliseconds(300)) }
            let arrived = try await composeCount() > before
            if !arrived { print("No compose request; screen:\n\(app.debugDescription)") }
            XCTAssertTrue(arrived, "a compose request reached the Mac")
            var request = URLRequest(url: URL(string: base + "/last-options")!)
            request.setValue("Bearer design-review", forHTTPHeaderField: "Authorization")
            let options = try JSONSerialization.jsonObject(with: try await URLSession.shared.data(for: request).0) as? [String: Any]
            return options?["customPrompt"] as? String
        }
        let sent = try await dictateOnce()
        XCTAssertTrue(sent?.contains("Be brief.") == true, "custom prompt sent: \(sent ?? "nil")")
        XCTAssertTrue(sent?.contains("Clean up the transcribed speech.") == true)

        app.activate()
        XCTAssertTrue(app.buttons["Settings"].waitForExistence(timeout: 5))
        app.buttons["Settings"].tap()
        XCTAssertTrue(app.navigationBars["Settings"].waitForExistence(timeout: 5))
        openAdvancedSettings(app)
        tapSwitch(toggle)
        XCTAssertFalse(app.alerts.firstMatch.waitForExistence(timeout: 1), "turning off needs no confirmation")
        XCTAssertFalse(prompts.exists)
        snap("developer-off", app)
        closeAdvancedSettings(app)
        let sentWhileOff = try await dictateOnce()
        XCTAssertNil(sentWhileOff, "no custom prompt while Developer mode is off")

        app.activate()
        XCTAssertTrue(app.buttons["Settings"].waitForExistence(timeout: 5))
        app.buttons["Settings"].tap()
        XCTAssertTrue(app.navigationBars["Settings"].waitForExistence(timeout: 5))
        openAdvancedSettings(app)
        tapSwitch(toggle)
        XCTAssertTrue(confirm.waitForExistence(timeout: 3))
        confirm.buttons["Turn On"].tap()
        XCTAssertTrue(prompts.waitForExistence(timeout: 3))
        prompts.tap()
        XCTAssertTrue(app.staticTexts["customBadge-dictate"].waitForExistence(timeout: 5), "the prompt was kept while off")
        let reset = app.buttons["resetPrompt-dictate"]
        reveal(reset, in: app)
        reset.tap()
        XCTAssertFalse(app.staticTexts["customBadge-dictate"].waitForExistence(timeout: 2))
        let builtIn = XCTNSPredicateExpectation(predicate: NSPredicate(format: "NOT (value CONTAINS %@)", "Be brief."), object: dictate)
        XCTAssertEqual(XCTWaiter.wait(for: [builtIn], timeout: 3), .completed)
        snap("developer-prompt-reset", app)
    }

    /// A SwiftUI Toggle reports a switch inside the labelled element on iOS 26; tap that one.
    private func tapSwitch(_ toggle: XCUIElement) {
        let inner = toggle.switches.firstMatch
        (inner.exists ? inner : toggle).tap()
    }

    private func composeCount() async throws -> Int {
        var request = URLRequest(url: URL(string: base + "/stats")!)
        request.setValue("Bearer design-review", forHTTPHeaderField: "Authorization")
        let stats = try JSONSerialization.jsonObject(with: try await URLSession.shared.data(for: request).0) as? [String: Any]
        return stats?["composeCount"] as? Int ?? 0
    }

    func testSettingsDictationRowsAlign() {
        let app = XCUIApplication()
        app.launchEnvironment = ["VOXT_TEST_RESET": "1"]
        app.launch()
        pair(app, address: base)
        app.buttons["Settings"].tap()
        let titles = ["Writing style", "Personal dictionary", "Learn my style"]
        let rows = titles.map { app.staticTexts[$0] }
        for row in rows { XCTAssertTrue(row.waitForExistence(timeout: 5)) }
        for row in rows.dropFirst() { XCTAssertEqual(row.frame.minX, rows[0].frame.minX, accuracy: 1) }
        snap("settings-dictation", app)
    }

    /// Never lose a recording: a keyboard dictation the Mac didn't answer survives a relaunch.
    /// History lists it, and the keyboard's Retry brings Voxt forward to send it into the field.
    func testAnUnsentRecordingSurvivesARelaunchAndTheKeyboardRetrySendsIt() async throws {
        try await control(failure: true, delay: 0)
        let app = XCUIApplication()
        app.launchEnvironment = ["VOXT_TEST_RESET": "1", "VOXT_TEST_AUDIO_FILE": try XCTUnwrap(env["VOXT_E2E_WAV"])]
        app.launch()
        keyboardSettings(enabled: true)
        app.activate()
        pair(app, address: base)
        XCTAssertTrue(app.descendants(matching: .any)["connectedLabel"].waitForExistence(timeout: 10))
        let safari = XCUIApplication(bundleIdentifier: "com.apple.mobilesafari")
        safari.terminate()
        openTestPage(safari, base: base)
        let field = safari.textViews["Session test"]
        XCTAssertTrue(field.waitForExistence(timeout: 10))
        field.tap()
        let start = safari.buttons["Start dictation"]
        for _ in 0..<8 {
            if start.waitForExistence(timeout: 1) { break }
            dismissKeyboardTip(safari)
            if safari.buttons["textLayout-voice"].exists { safari.buttons["textLayout-voice"].tap() }
            else if safari.buttons["Next keyboard"].firstMatch.exists { safari.buttons["Next keyboard"].firstMatch.tap(); dismissKeyboardTip(safari) }
        }
        start.tap()
        if app.wait(for: .runningForeground, timeout: 5) {
            XCTAssertTrue(app.buttons["keyboardHandoffMic"].waitForExistence(timeout: 10))
            safari.activate()
        }
        let stop = safari.buttons["Stop dictation"]
        XCTAssertTrue(stop.waitForExistence(timeout: 6))
        try await Task.sleep(for: .seconds(1))
        stop.tap()
        XCTAssertTrue(safari.buttons["keyboardRetryComposition"].waitForExistence(timeout: 15))

        app.terminate()
        app.launchEnvironment.removeValue(forKey: "VOXT_TEST_RESET")
        app.launch()
        app.tabBars.buttons["History"].tap()
        XCTAssertTrue(app.staticTexts["Not sent yet"].waitForExistence(timeout: 5))
        XCTAssertEqual(app.buttons.matching(identifier: "sendRecordingAgain").count, 1)
        snap("history-unsent", app)

        try await control(delay: 0)
        func composeCount() async throws -> Int {
            var request = URLRequest(url: URL(string: base + "/stats")!)
            request.setValue("Bearer design-review", forHTTPHeaderField: "Authorization")
            return try JSONDecoder().decode([String: Int].self, from: try await URLSession.shared.data(for: request).0)["composeCount"] ?? -1
        }
        let sentBefore = try await composeCount()
        safari.activate()
        field.tap()
        let retry = safari.buttons["keyboardRetryComposition"]
        XCTAssertTrue(retry.waitForExistence(timeout: 10))
        snap("keyboard-saved-recording", safari)
        retry.tap()
        // Retry opens Voxt to resend; a person sees it come forward before tapping back.
        let frontmost = XCTNSPredicateExpectation(predicate: NSPredicate { _, _ in app.state == .runningForeground }, object: nil)
        XCTAssertEqual(XCTWaiter.wait(for: [frontmost], timeout: 10), .completed, "Retry must open Voxt")
        try await Task.sleep(for: .seconds(1))
        safari.activate()
        let inserted = XCTNSPredicateExpectation(predicate: NSPredicate(format: "value CONTAINS %@", result), object: field)
        let landed = await XCTWaiter().fulfillment(of: [inserted], timeout: 20)
        snap("keyboard-after-retry", safari)
        let sentAfter = try await composeCount()
        XCTAssertGreaterThan(sentAfter, sentBefore, "Retry must reach the Mac")
        XCTAssertEqual(landed, .completed, "field: \(field.value as? String ?? "")")

        app.activate()
        app.tabBars.buttons["History"].tap()
        XCTAssertTrue(app.staticTexts["historyText"].firstMatch.waitForExistence(timeout: 5))
        XCTAssertFalse(app.staticTexts["Not sent yet"].exists)
    }

    func testDesignTour() async throws {
        try await control()
        let app = XCUIApplication()
        app.launchEnvironment = ["VOXT_TEST_RESET": "1", "VOXT_TEST_AUDIO_FILE": try XCTUnwrap(env["VOXT_E2E_WAV"]),
                                 "VOXT_TEST_PRACTICE_FIELD": "1"]
        app.launch()
        XCTAssertTrue(app.staticTexts["onboardingTitle"].waitForExistence(timeout: 5))
        snap("onboarding", app)
        keyboardSettings(enabled: false)
        app.activate()
        pair(app, address: "http://127.0.0.1:9")
        snap("home-keyboard-disabled", app)
        keyboardSettings(enabled: true)
        app.activate()
        pair(app, address: base)
        XCTAssertTrue(app.descendants(matching: .any)["connectedLabel"].waitForExistence(timeout: 12))
        XCTAssertTrue(app.navigationBars["Voxt"].staticTexts["Voxt"].exists)
        snap("home-keyboard-enabled", app)
        app.buttons["Settings"].tap()
        XCTAssertTrue(app.buttons["Done"].waitForExistence(timeout: 5))
        snap("settings", app)
        // "Clean up my words" replaced the literal-dictation switch; off means literal.
        let cleanUp = app.switches["cleanUpWords"]
        reveal(cleanUp, in: app)
        XCTAssertEqual(cleanUp.value as? String, "1")
        cleanUp.coordinate(withNormalizedOffset: CGVector(dx: 0.9, dy: 0.5)).tap()
        XCTAssertEqual(cleanUp.value as? String, "0")
        snap("settings-literal-on", app)
        cleanUp.coordinate(withNormalizedOffset: CGVector(dx: 0.9, dy: 0.5)).tap()
        XCTAssertEqual(cleanUp.value as? String, "1")
        app.buttons["Done"].tap()

        let field = app.textFields["tryField"]
        XCTAssertTrue(field.waitForExistence(timeout: 5))
        field.tap()
        dismissKeyboardTip(app)
        let start = app.buttons["Start dictation"]
        if !start.waitForExistence(timeout: 2) {
            if app.buttons["textLayout-voice"].exists {
                app.buttons["textLayout-voice"].tap()
            } else {
                let globe = app.buttons["Next keyboard"].firstMatch
                XCTAssertTrue(globe.waitForExistence(timeout: 3))
                globe.press(forDuration: 1.2)
                let voxt = app.descendants(matching: .any).matching(NSPredicate(format: "label == 'Voxt'")).firstMatch
                XCTAssertTrue(voxt.waitForExistence(timeout: 3))
                voxt.tap()
            }
        }
        if !start.waitForExistence(timeout: 5) {
            snap("keyboard-selection-failed", app)
            print(app.debugDescription)
        }
        XCTAssertTrue(start.exists)
        XCTAssertFalse(app.buttons["Writing options"].exists, "the keyboard has no mode menu: a plain mic tap dictates")
        app.buttons["keyboardPen"].tap()
        XCTAssertTrue(app.staticTexts["Describe what to write"].waitForExistence(timeout: 3))
        snap("keyboard-pen", app)
        app.buttons["keyboardPen"].tap()
        XCTAssertTrue(app.staticTexts["Tap to speak"].waitForExistence(timeout: 3))
        let idleMicY = start.frame.midY
        snap("keyboard-idle", app)
        start.tap()
        // Exercise the same URL delivered when a keyboard first opens the microphone.
        app.open(URL(string: "voxtmobile://session")!)
        let handoff = app.buttons["keyboardHandoffMic"]
        XCTAssertTrue(handoff.waitForExistence(timeout: 10))
        XCTAssertFalse(app.buttons["compositionMode-dictate"].isHittable)
        waitForVoiceLevel(handoff)
        snap("keyboard-bounce", app)
        app.buttons["keyboardHandoffCancel"].tap()
        XCTAssertFalse(handoff.exists)
        field.tap()
        XCTAssertTrue(start.waitForExistence(timeout: 5))
        start.tap()
        let stop = app.buttons["Stop dictation"]
        XCTAssertTrue(stop.waitForExistence(timeout: 10))
        waitForVoiceLevel(stop)
        snap("keyboard-recording", app)
        stop.tap()
        XCTAssertTrue(app.staticTexts["Writing it up…"].waitForExistence(timeout: 3))
        snap("keyboard-processing", app)
        let inserted = XCTNSPredicateExpectation(predicate: NSPredicate(format: "value CONTAINS %@", result), object: field)
        XCTAssertEqual(XCTWaiter.wait(for: [inserted], timeout: 15), .completed)
        snap("keyboard-completed", app)
        // The keyboard covers the tab bar; Home's bar closes it.
        app.buttons["Hide keyboard"].tap()
        app.tabBars.buttons["History"].tap()
        let latest = app.staticTexts["historyText"].firstMatch
        XCTAssertTrue(latest.waitForExistence(timeout: 5))
        XCTAssertEqual(latest.label, result)
        XCTAssertGreaterThan(app.buttons["editLatestResult"].frame.minX, latest.frame.midX)
        snap("history", app)
        app.tabBars.buttons["Home"].tap()
        // Home scrolls far enough that its last item clears the tab bar.
        let lastHomeItem = app.buttons["copyRecent-0"]
        reveal(lastHomeItem, in: app)
        XCTAssertLessThan(lastHomeItem.frame.maxY, app.tabBars.firstMatch.frame.minY)
        snap("home-with-history", app)
        reveal(field, in: app)
        field.tap()
        XCTAssertTrue(start.waitForExistence(timeout: 5))

        try await control(failure: true)
        start.tap()
        XCTAssertTrue(stop.waitForExistence(timeout: 5))
        stop.tap()
        XCTAssertTrue(app.buttons["keyboardRetryComposition"].waitForExistence(timeout: 15))
        XCTAssertEqual(start.frame.midY, idleMicY, accuracy: 2)
        snap("keyboard-failed", app)
        try await control()
    }
}
