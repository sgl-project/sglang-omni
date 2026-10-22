import Foundation
import Testing
@testable import VoxtMobileKit

/// The Mac status card on the home page and in Settings: which Mac, its state, and one
/// VoiceOver sentence.
@Suite struct MacStatusCardTests {
    private let l10n = VoxtL10n(defaults: UserDefaults(suiteName: "status-\(UUID().uuidString)")!)

    @Test func connectedShowsTheNameAGreenDotAndTheShortVersion() {
        let card = MacStatusCard(macName: "Studio Mac mini", state: .connected(version: "1.16.0"), language: "en", l10n: l10n)
        #expect(card.title == "Studio Mac mini")
        #expect(card.detail == "Connected · Voxt 1.16")
        #expect(card.dot == .green)
        #expect(!card.showsRetry)
        #expect(card.accessibilityLabel == "Studio Mac mini, connected, Voxt 1.16")
        #expect(MacStatusCard(macName: nil, state: .connected(version: "1.16.2"), language: "en", l10n: l10n).detail
                == "Connected · Voxt 1.16.2")
    }

    @Test func withoutALanguageTheCardFollowsTheLanguagePickedInVoxt() {
        // Home builds the card without a language: Voxt set to Japanese on an English phone
        // must not show "Can't reach your Mac".
        let name = "status-ja-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        defaults.set("ja", forKey: VoxtL10n.preferenceKey)
        let japanese = VoxtL10n(defaults: defaults)
        let card = MacStatusCard(macName: nil, state: .unreachable(reason: nil), l10n: japanese)
        #expect(card.title == japanese.string("Your Mac", language: "ja"))
        #expect(card.detail == japanese.string("Can't reach your Mac", language: "ja"))
        #expect(card.detail != "Can't reach your Mac")
    }

    @Test func connectingShowsProgressInsteadOfADot() {
        let card = MacStatusCard(macName: "Studio Mac mini", state: .connecting, language: "en", l10n: l10n)
        #expect(card.detail == "Connecting…")
        #expect(card.dot == nil)
        #expect(!card.showsRetry)
        #expect(card.accessibilityLabel == "Studio Mac mini, connecting")
    }

    @Test func unreachableShowsAnOrangeDotRetryAndTheReason() {
        let card = MacStatusCard(macName: nil, state: .unreachable(reason: "Pair this iPhone again in Voxt Connector."),
                                 language: "en", l10n: l10n)
        #expect(card.title == "Your Mac")
        #expect(card.detail == "Can't reach your Mac")
        #expect(card.dot == .orange)
        #expect(card.showsRetry)
        #expect(card.reason == "Pair this iPhone again in Voxt Connector.")
        #expect(card.accessibilityLabel == "Your Mac, can't reach your Mac")
    }

    /// A cloud model is not "can't reach": the Mac answers, but dictation stays off.
    /// A reason that only repeats "can't reach" says what to do instead; a specific reason is kept.
    @Test func unreachableSaysWhatToDoUnlessTheReasonAddsInformation() {
        let advice = "Check that Voxt and Voxt Connector are open on your Mac."
        for generic in [nil, "The connection to your Mac is not ready. Reconnect and try again.",
                        "Can't reach your Mac. Check your internet connection and make sure Voxt is running on the Mac.",
                        "Can't reach your Mac. Check that Tailscale is connected on this phone and Voxt is running on the Mac."] {
            #expect(MacStatusCard(macName: nil, state: .unreachable(reason: generic), language: "en", l10n: l10n).reason == advice,
                    "\(generic ?? "nil")")
        }
        let specific = "Pair this iPhone again in Voxt Connector."
        #expect(MacStatusCard(macName: nil, state: .unreachable(reason: specific), language: "en", l10n: l10n).reason == specific)
    }

    @Test func cloudModelIsItsOwnStateWithoutRetry() {
        let card = MacStatusCard(macName: "Studio Mac mini", state: .cloudModel(version: "1.16.0"), language: "en", l10n: l10n)
        #expect(card.detail == "Cloud model on your Mac")
        #expect(card.dot == .orange)
        #expect(!card.showsRetry, "retrying the connection doesn't change the Mac's model")
        #expect(card.reason == "Dictation stays off until Voxt uses a local model.")
        #expect(card.accessibilityLabel == "Studio Mac mini, cloud model on your Mac")
    }

    @Test func glyphFollowsTheKindOfMacInTheName() {
        let cases: [(String?, String)] = [
            ("Someone's MacBook Pro", "laptopcomputer"), ("Studio Mac mini", "macmini"),
            ("Office Mac Studio", "macstudio"), ("Family iMac", "desktopcomputer"),
            ("Workshop", "desktopcomputer"), (nil, "desktopcomputer"), ("MACBOOK AIR", "laptopcomputer"),
        ]
        for (name, symbol) in cases {
            #expect(MacStatusCard.symbol(forMacName: name) == symbol, "\(name ?? "nil")")
        }
    }

    @Test func localizedDetailAndVoiceOver() {
        let ja = MacStatusCard(macName: "Studio Mac mini", state: .connected(version: "1.16.0"), language: "ja", l10n: l10n)
        #expect(ja.detail == "接続済み · Voxt 1.16")
        #expect(ja.accessibilityLabel == "Studio Mac mini、接続済み、Voxt 1.16")
        let hans = MacStatusCard(macName: nil, state: .unreachable(reason: nil), language: "zh-Hans", l10n: l10n)
        #expect(hans.title == "你的 Mac")
        #expect(hans.detail == "无法连接你的 Mac")
        let hant = MacStatusCard(macName: nil, state: .connecting, language: "zh-Hant", l10n: l10n)
        #expect(hant.detail == "正在連線…")
    }

    @Test func longNamesAreShortenedInTheMiddleForVoiceOverFreeLayouts() {
        let name = "Someone's Extremely Long Office Mac Studio Ultra"
        let short = ConnectionStatusText.shortened(name)
        #expect(short.count == ConnectionStatusText.maximumNameLength)
        #expect(short.hasPrefix("Someone's E"))
        #expect(short.hasSuffix("Studio Ultra"))
        #expect(ConnectionStatusText.shortened("Studio Mac mini") == "Studio Mac mini")
    }
}
