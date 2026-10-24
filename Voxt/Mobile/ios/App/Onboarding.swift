import Foundation
import VoxtMobileKit

/// Where the first-open walkthrough is, saved so a relaunch resumes it.
struct OnboardingProgress: Codable, Equatable {
    enum Step: Int, Codable, CaseIterable, Comparable {
        case welcome, connect, addKeyboard, fullAccess, tryIt
        static func < (a: Step, b: Step) -> Bool { a.rawValue < b.rawValue }
        /// "welcome", "connect", "addKeyboard", "fullAccess" or "tryIt" (UI test launch setting).
        init?(name: String) {
            guard let step = Self.allCases.first(where: { "\($0)" == name }) else { return nil }
            self = step
        }
    }

    static let storageKey = "onboarding.progress"

    var step: Step = .welcome
    var isDone = false

    init(step: Step = .welcome, isDone: Bool = false) {
        self.step = step
        self.isDone = isDone
    }

    /// The VOXT_TEST_ONBOARDING launch setting: a step name opens that step, "done" opens Home.
    init?(testLaunchSetting name: String) {
        if name == "done" {
            self.init(step: .tryIt, isDone: true)
        } else if let step = Step(name: name) {
            self.init(step: step)
        } else {
            return nil
        }
    }

    /// Nothing saved and already paired: set up before the walkthrough existed, so skip it.
    static func load(from defaults: UserDefaults, paired: Bool) -> OnboardingProgress {
        if let data = defaults.data(forKey: storageKey),
           let saved = try? JSONDecoder().decode(OnboardingProgress.self, from: data) {
            return saved
        }
        return paired ? OnboardingProgress(step: .tryIt, isDone: true) : OnboardingProgress()
    }

    func save(to defaults: UserDefaults) {
        defaults.set(try? JSONEncoder().encode(self), forKey: Self.storageKey)
    }

    /// The screen to show, or nil for the app. Without a Mac nothing past Connect shows;
    /// once paired, Connect is behind.
    func screen(paired: Bool) -> Step? {
        if !paired { return !isDone && step == .welcome ? .welcome : .connect }
        if isDone { return nil }
        return max(step, .addKeyboard)
    }

    /// On from the screen that shows, which depends on whether a Mac is paired.
    mutating func advance(paired: Bool) {
        guard let shown = screen(paired: paired) else { return }
        if let next = Step(rawValue: shown.rawValue + 1) { step = next } else { finish() }
    }

    mutating func finish() {
        step = .tryIt
        isDone = true
    }

    /// Settings › About › Set up again: the keyboard steps, keeping the pairing.
    mutating func setUpAgain() {
        step = .addKeyboard
        isDone = false
    }
}

/// What the Try-it screen shows. The app can't see which keyboard is up, only that
/// the keyboard changed; the Voxt keyboard reports itself through `KeyboardStatus`, and only
/// with Full Access.
struct TryItProgress: Equatable {
    enum Card: Equatable { case switchKeyboard, fullAccessOff, ready, done }
    static let statusWait: TimeInterval = 2

    var startHistoryCount: Int
    private(set) var keyboardChangedAt: Date?

    init(startHistoryCount: Int) { self.startHistoryCount = startHistoryCount }

    mutating func keyboardChanged(at date: Date) { keyboardChangedAt = date }

    func card(keyboardEnabled: Bool, status: KeyboardStatus?, historyCount: Int, now: Date) -> Card {
        if historyCount > startHistoryCount { return .done }
        guard let changed = keyboardChangedAt else { return .switchKeyboard }
        // The keyboard writes its status as it appears, which can be just before the app
        // hears that the keyboard changed.
        if KeyboardStatus.confirmsFullAccess(status, since: changed.addingTimeInterval(-1)) { return .ready }
        if keyboardEnabled, now.timeIntervalSince(changed) >= Self.statusWait { return .fullAccessOff }
        return .switchKeyboard
    }
}
