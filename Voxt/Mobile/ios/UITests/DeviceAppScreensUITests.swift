import XCTest

/// The app screens on a real iPhone. Runs only with
/// TEST_RUNNER_VOXT_DEVICE_A1A4=1. Uses the phone's saved pairing and never resets it. Each
/// test sets its own first-open state at launch, and teardown finishes the first-open steps
/// even after a failure, so the next test and the phone's owner find Home.
/// "Keep Voxt ready" is left as it was found.
/// Preconditions: Voxt keyboard added with Full Access; the paired Mac reachable.
/// Haptics can't be checked here; the script lists it as the one human check.
@MainActor
final class DeviceAppScreensUITests: XCTestCase {
    private let voxt = XCUIApplication()
    private let settings = XCUIApplication(bundleIdentifier: "com.apple.Preferences")
    private var onDevice: Bool { ProcessInfo.processInfo.environment["VOXT_DEVICE_A1A4"] == "1" }

    override func setUpWithError() throws {
        try XCTSkipUnless(onDevice)
        continueAfterFailure = false
    }

    override func tearDownWithError() throws {
        guard onDevice else { return }
        voxt.terminate()
        launch(onboarding: "tryIt")
        finishOnboarding()
        voxt.terminate()
    }

    /// "Open Settings" lands in the Settings app on Voxt's page.
    func testFullAccessOpenSettingsLandsOnVoxtInSettings() {
        var keyboardsRow = openSettingsFromFullAccess(freshSettings: false)
        // Run 172914: seconds after xcodebuild reinstalled Voxt, Settings (already running) showed
        // Apps › Voxt without Keyboards; the row was there when checked later. Open it once more
        // from a fresh Settings. A cold Settings landing elsewhere fails on the nav bar, not here.
        let reopened = !keyboardsRow
        if reopened { keyboardsRow = openSettingsFromFullAccess(freshSettings: true) }
        shot("device-a1-4-settings-voxt")
        print("VOXT-DEVICE open-settings keyboardsRow=\(keyboardsRow) reopenNeeded=\(reopened)")
        // Keyboards is the row the Add-keyboard and Full Access text point at.
        XCTAssertTrue(keyboardsRow, "Settings › Apps › Voxt has no Keyboards row")
        voxt.activate()
        XCTAssertTrue(title(is: "Try it"), "back in Voxt, the next step shows")
    }

    /// The Home switch starts Picture in Picture; it stays on after going Home and back.
    func testKeepVoxtReadyStartsPictureInPicture() {
        launch(onboarding: "done")
        let toggle = voxt.switches["keepVoxtReady"]
        XCTAssertTrue(toggle.waitForExistence(timeout: 10))
        XCTAssertTrue(toggle.isEnabled, "Picture in Picture reported unavailable")
        let wasOn = toggle.value as? String == "1"
        if wasOn { set(toggle, on: false) }
        set(toggle, on: true)
        // The switch reads the window's isActive, which turns on only once PiP has started.
        XCTAssertTrue(waitForValue("1", of: toggle, timeout: 15), "PiP did not start")
        XCUIDevice.shared.press(.home)
        sleep(3)
        shot("device-a2-pip-on-home-screen")
        voxt.activate()
        XCTAssertTrue(waitForValue("1", of: toggle, timeout: 5), "PiP closed after going Home")
        print("VOXT-DEVICE keep-ready started=true afterHome=\(toggle.value as? String ?? "-") wasOn=\(wasOn)")
        if !wasOn { set(toggle, on: false) }
    }

    /// With the real keyboard, Try it points at the mic, a dictation lands, "That's it" shows.
    func testTryItWithTheRealKeyboard() {
        launch(onboarding: "tryIt")
        XCTAssertTrue(title(is: "Try it"))
        let mic = voxt.buttons["Start dictation"]
        let next = voxt.buttons.matching(NSPredicate(format: "label == 'Next keyboard' OR identifier == 'Next keyboard'")).firstMatch
        var switches = 0
        while !mic.waitForExistence(timeout: 2), switches < 6 {
            if voxt.buttons["textLayout-voice"].exists { voxt.buttons["textLayout-voice"].tap() } else if next.exists { next.tap() }
            switches += 1
        }
        XCTAssertTrue(mic.exists, "Voxt keyboard not found: is it added with Full Access?")
        XCTAssertTrue(voxt.staticTexts["Tap and say anything"].waitForExistence(timeout: 5), "no Full Access report from the keyboard")
        sleep(3)
        XCTAssertFalse(voxt.staticTexts["Allow Full Access is off"].exists)
        shot("device-a1-5-try-mic")
        mic.tap()
        let stop = voxt.buttons["Stop dictation"]
        XCTAssertTrue(stop.waitForExistence(timeout: 10), "recording never started")
        print("VOXT-SPEAK tryit dictation")
        NSLog("VOXT-SPEAK %@ dictation", "tryit")
        sleep(5)
        stop.tap()
        let done = voxt.staticTexts["That's it. Use it in any app."]
        XCTAssertTrue(done.waitForExistence(timeout: 60), "no result came back")
        let field = voxt.textFields["tryItField"].exists ? voxt.textFields["tryItField"] : voxt.textViews["tryItField"]
        print("VOXT-DEVICE try-it text=\(field.value as? String ?? "")")
        shot("device-a1-5-try-done")
        voxt.buttons["onboardingNext"].tap()
        XCTAssertTrue(voxt.switches["keepVoxtReady"].waitForExistence(timeout: 10), "Done leads to Home")
    }

    // MARK: Helpers

    /// The phone's saved pairing; a simulator check passes a fake Mac in VOXT_A1A4_TEST_SERVER.
    private func launch(onboarding: String) {
        voxt.launchEnvironment = ["VOXT_TEST_ONBOARDING": onboarding]
        voxt.launchEnvironment["VOXT_TEST_SERVER"] = ProcessInfo.processInfo.environment["VOXT_A1A4_TEST_SERVER"]
        voxt.launch()
    }

    /// Full Access step → Open Settings. Returns whether Voxt's page shows the Keyboards row within 15 s.
    /// The iOS 26.3 simulator always opens the Settings top page here, so only a phone runs this through.
    private func openSettingsFromFullAccess(freshSettings: Bool) -> Bool {
        voxt.terminate()
        if freshSettings { settings.terminate() }
        launch(onboarding: "fullAccess")
        XCTAssertTrue(title(is: "Then allow Full Access"))
        voxt.buttons["onboardingNext"].tap()
        XCTAssertTrue(settings.wait(for: .runningForeground, timeout: 10), "Settings is not in front")
        XCTAssertTrue(settings.navigationBars["Voxt"].waitForExistence(timeout: 10), "Settings did not open on Voxt")
        return settings.descendants(matching: .any).matching(NSPredicate(format: "label == 'Keyboards'")).firstMatch
            .waitForExistence(timeout: 15)
    }

    /// Skip on Try it saves the steps as done.
    private func finishOnboarding() {
        let skip = voxt.buttons["onboardingSkip"]
        sleep(3) // the Try it card settles after its 2 s wait; Skip doesn't move under the tap
        if skip.waitForExistence(timeout: 5) { skip.tap() }
        XCTAssertTrue(voxt.switches["keepVoxtReady"].waitForExistence(timeout: 10), "the first-open steps are done")
    }

    private func title(is text: String) -> Bool {
        let title = voxt.staticTexts["onboardingTitle"]
        guard title.waitForExistence(timeout: 10) else { return false }
        let deadline = Date().addingTimeInterval(5)
        while title.label != text, Date() < deadline { Thread.sleep(forTimeInterval: 0.2) }
        return title.label == text
    }

    private func set(_ toggle: XCUIElement, on: Bool) {
        guard (toggle.value as? String == "1") != on else { return }
        toggle.coordinate(withNormalizedOffset: CGVector(dx: 0.9, dy: 0.5)).tap()
        _ = waitForValue(on ? "1" : "0", of: toggle, timeout: 15)
    }

    private func waitForValue(_ value: String, of element: XCUIElement, timeout: TimeInterval) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while element.value as? String != value, Date() < deadline { Thread.sleep(forTimeInterval: 0.25) }
        return element.value as? String == value
    }

    private func shot(_ name: String) {
        let attachment = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
