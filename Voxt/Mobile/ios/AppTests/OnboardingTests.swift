import XCTest
@testable import VoxtMobile
import VoxtMobileKit

/// First open, one step per screen.
@MainActor
final class OnboardingTests: XCTestCase {
    func testAFreshInstallStartsAtWelcomeAndWalksEveryStep() {
        var progress = OnboardingProgress()
        XCTAssertEqual(progress.screen(paired: false), .welcome)
        progress.advance(paired: false)
        XCTAssertEqual(progress.screen(paired: false), .connect)
        // Pairing moves past Connect without a tap.
        XCTAssertEqual(progress.screen(paired: true), .addKeyboard)
        progress.advance(paired: true)
        XCTAssertEqual(progress.screen(paired: true), .fullAccess)
        progress.advance(paired: true)
        XCTAssertEqual(progress.screen(paired: true), .tryIt)
        progress.advance(paired: true)
        XCTAssertNil(progress.screen(paired: true))
        XCTAssertTrue(progress.isDone)
    }

    func testConnectCannotBeSkippedWithoutAMac() {
        var progress = OnboardingProgress()
        progress.advance(paired: false)
        progress.advance(paired: false)
        XCTAssertEqual(progress.screen(paired: false), .connect)
    }

    func testUnpairingAfterSetupAsksForAMacAgain() {
        var progress = OnboardingProgress()
        progress.finish()
        XCTAssertNil(progress.screen(paired: true))
        XCTAssertEqual(progress.screen(paired: false), .connect)
    }

    func testSetUpAgainStartsAtAddKeyboard() {
        var progress = OnboardingProgress()
        progress.finish()
        progress.setUpAgain()
        XCTAssertEqual(progress.screen(paired: true), .addKeyboard)
    }

    func testAPairedUserFromBeforeTheRedesignSkipsIt() {
        let (defaults, name) = makeDefaults()
        defer { defaults.removePersistentDomain(forName: name) }
        XCTAssertNil(OnboardingProgress.load(from: defaults, paired: true).screen(paired: true))
        XCTAssertEqual(OnboardingProgress.load(from: defaults, paired: false).screen(paired: false), .welcome)
    }

    func testTheTestLaunchSettingPicksAStepOrDone() {
        XCTAssertEqual(OnboardingProgress(testLaunchSetting: "fullAccess"), OnboardingProgress(step: .fullAccess, isDone: false))
        XCTAssertEqual(OnboardingProgress(testLaunchSetting: "done"), OnboardingProgress(step: .tryIt, isDone: true))
        XCTAssertNil(OnboardingProgress(testLaunchSetting: "later"))
    }

    func testProgressSurvivesRelaunch() throws {
        let (defaults, name) = makeDefaults()
        defer { defaults.removePersistentDomain(forName: name) }
        let tokens = MemoryTokens()
        let model = AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: defaults, injectedTokens: tokens)
        XCTAssertEqual(model.onboardingScreen, .welcome)
        model.advanceOnboarding()
        XCTAssertEqual(model.onboardingScreen, .connect)
        try PairingStore(defaults: defaults, tokens: tokens).save(ServerConfig(baseURL: URL(string: "https://fixture.invalid")!, token: "test-only"))
        let relaunched = AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: defaults, injectedTokens: tokens)
        XCTAssertEqual(relaunched.onboardingScreen, .addKeyboard)
    }

    func testSetUpAgainKeepsThePairing() throws {
        let (defaults, name) = makeDefaults()
        defer { defaults.removePersistentDomain(forName: name) }
        let tokens = MemoryTokens()
        try PairingStore(defaults: defaults, tokens: tokens).save(ServerConfig(baseURL: URL(string: "https://fixture.invalid")!, token: "test-only"))
        let model = AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: defaults, injectedTokens: tokens)
        XCTAssertNil(model.onboardingScreen)
        model.setUpAgain()
        XCTAssertEqual(model.onboardingScreen, .addKeyboard)
        XCTAssertNotNil(model.server)
        model.finishOnboarding()
        XCTAssertNil(AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: defaults, injectedTokens: tokens).onboardingScreen)
    }

    // MARK: Try it

    func testTryItShowsTheSwitchHintUntilTheKeyboardChanges() {
        let start = Date(timeIntervalSince1970: 1_000)
        let tryIt = TryItProgress(startHistoryCount: 0)
        XCTAssertEqual(tryIt.card(keyboardEnabled: true, status: nil, historyCount: 0, now: start), .switchKeyboard)
    }

    func testTryItWaitsTwoSecondsForTheKeyboardBeforeSayingFullAccessIsOff() {
        let changed = Date(timeIntervalSince1970: 1_000)
        var tryIt = TryItProgress(startHistoryCount: 0)
        tryIt.keyboardChanged(at: changed)
        XCTAssertEqual(tryIt.card(keyboardEnabled: true, status: nil, historyCount: 0, now: changed.addingTimeInterval(1.9)), .switchKeyboard)
        XCTAssertEqual(tryIt.card(keyboardEnabled: true, status: nil, historyCount: 0, now: changed.addingTimeInterval(2)), .fullAccessOff)
        let stale = KeyboardStatus(hasFullAccess: true, updatedAt: changed.addingTimeInterval(-60))
        XCTAssertEqual(tryIt.card(keyboardEnabled: true, status: stale, historyCount: 0, now: changed.addingTimeInterval(3)), .fullAccessOff)
    }

    func testTryItNeverBlamesFullAccessWhileVoxtIsNotAdded() {
        let changed = Date(timeIntervalSince1970: 1_000)
        var tryIt = TryItProgress(startHistoryCount: 0)
        tryIt.keyboardChanged(at: changed)
        XCTAssertEqual(tryIt.card(keyboardEnabled: false, status: nil, historyCount: 0, now: changed.addingTimeInterval(5)), .switchKeyboard)
    }

    func testAFreshStatusPointsAtTheMicAndADictationFinishes() {
        let changed = Date(timeIntervalSince1970: 1_000)
        var tryIt = TryItProgress(startHistoryCount: 4)
        tryIt.keyboardChanged(at: changed)
        let fresh = KeyboardStatus(hasFullAccess: true, updatedAt: changed.addingTimeInterval(0.3))
        XCTAssertEqual(tryIt.card(keyboardEnabled: true, status: fresh, historyCount: 4, now: changed.addingTimeInterval(0.5)), .ready)
        XCTAssertEqual(tryIt.card(keyboardEnabled: true, status: fresh, historyCount: 5, now: changed.addingTimeInterval(9)), .done)
    }

    private func makeDefaults() -> (UserDefaults, String) {
        let name = "onboarding.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        return (defaults, name)
    }
}

final class MemoryTokens: TokenStoring {
    var value: String?
    func read() throws -> String? { value }
    func write(_ value: String) throws { self.value = value }
    func delete() throws { value = nil }
}
