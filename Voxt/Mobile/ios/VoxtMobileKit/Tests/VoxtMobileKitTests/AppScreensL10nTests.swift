import Foundation
import Testing
@testable import VoxtMobileKit

/// Every string on the app screens has Chinese and Japanese text.
@Suite(.serialized)
struct AppScreensL10nTests {
    static let keys = [
        "Step %d of 5",
        "You say",
        "“hi karen um the heater in 4B stopped working on tuesday, uh, can you send someone this week? thanks”",
        "Voxt types",
        "Hi Karen,\n\nThe heater in Unit 4B stopped working on Tuesday. Could you send someone this week?\n\nThanks",
        "Settings opens on Voxt. Tap Keyboards, then turn on Voxt.",
        "In Settings, tap Keyboards, then turn on Voxt.",
        "The picture plays these taps in a loop.",
        "The keyboard needs it to talk to the Voxt app on this phone. Voxt doesn't keep what you type. When you ask it to edit text, that text goes only to your Mac.",
        "iOS asks this of every keyboard that works with its own app. Voxt's keyboard uses it only to pass your recording to the Voxt app and to get the text back.",
        "What's Full Access?",
        "Allow Full Access for “Voxt” Keyboards?",
        "iOS shows this warning for every keyboard with Full Access.",
        "Don't Allow",
        "Allow",
        "In Keyboards, turn on Allow Full Access, then tap Allow.",
        "Say anything",
        "Skip for now",
        "Tap and say anything",
        "Hold the globe key and choose Voxt.",
        "Allow Full Access is off",
        "Voxt is added, but the keyboard can't reach the Voxt app without Full Access.",
        "That's it. Use it in any app.",
        "Say anything. Only your Mac hears it.",
        "Get started",
        "Add the Voxt keyboard",
        "Continue",
        "Apps",
        "ALLOW VOXT TO ACCESS",
        "Camera",
        "Keyboards",
        "Allow Full Access",
        "Then allow Full Access",
        "Unpair this Mac?",
        "Pair again",
        "Layouts",
        "Haptic feedback",
        "Languages",
        "Clean up my words",
        "Writing style",
        "Learn my style",
        "Language",
        "Version",
        "Set up again",
        "Advanced",
        "Keyboard",
        "Translation",
        "Writing",
        "Off: inserts your words as spoken, without cleanup. Faster.",
        "Privacy",
        "About",
        "On: your Mac may send audio and text to its selected cloud providers.",
        "Off: your voice and text stay on this iPhone and your Mac.",
        "Lets you edit the prompts Voxt sends to your Mac. Asks to confirm before it turns on.",
        "Recent",
        "See all",
        "This week: %@ on %@",
        "Your dictations show up here",
        "Keep Voxt ready",
        "The keyboard won't switch apps to start the mic. Voxt waits in a small floating window; the mic turns on only when you tap it.",
        "Remove",
        "Drag to change the order. The first one sits straight above the mic.",
        "Choose translation languages",
        "3 of 3 chosen. Remove one to add another.",
        "Search languages",
        "Add a language",
        "Choose up to 3 languages. On the keyboard, hold the mic and slide up to one.",
        "On the keyboard, hold the mic and slide up to a language. Your words come out in that language.",
        "Up to 3 languages",
        "Confirm on your Mac: code",
        "Don't have Voxt Connector yet?",
        "Voxt Connector runs on your Mac next to Voxt and shows this QR. Set up both from the link below.",
        "Camera paused",
        "Connect your Mac",
        "On your Mac, open Voxt Connector and choose Add iPhone. Then point this iPhone at the QR.",
    ]

    @Test func appScreensAreTranslated() {
        let name = "l10n-screens.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let l10n = VoxtL10n(defaults: defaults)
        for key in Self.keys {
            for language in ["zh-Hans", "zh-Hant", "ja"] {
                #expect(l10n.hasTranslation(key, language: language), "\(language): \(key)")
            }
        }
    }
}
