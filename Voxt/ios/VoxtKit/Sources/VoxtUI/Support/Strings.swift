import Foundation
import VoxtKit
#if os(iOS)
import UIKit
#else
import AppKit
#endif

/// A user-visible string: the literal English message is the key, its
/// Simplified Chinese in Resources/zh-Hans.lproj (as internal/i18n's T on
/// macOS). LocalizationTests checks that every L literal has both.
func L(_ key: String) -> String {
    Bundle.module.localizedString(forKey: key, value: key, table: nil)
}

/// L with format arguments ("%@", "%d").
func L(_ key: String, _ args: CVarArg...) -> String {
    String(format: L(key), locale: Locale.current, arguments: args)
}

/// Whether the app shows Chinese (the language names follow it).
var showsChinese: Bool {
    Bundle.module.preferredLocalizations.first?.hasPrefix("zh") == true
}

extension Feature {
    var label: String {
        switch self {
        case .transcribe: return L("Transcribe")
        case .translate: return L("Translate")
        case .rewrite: return L("Rewrite")
        }
    }

    var symbol: String {
        switch self {
        case .transcribe: return "text.bubble"
        case .translate: return "globe"
        case .rewrite: return "wand.and.stars"
        }
    }
}

extension SessionProblem {
    var message: String {
        switch self {
        case .microphoneDenied: return L("Voxt cannot use the microphone. Allow it in Settings → Privacy → Microphone.")
        case .speechNotAuthorized: return L("Speech recognition is not allowed. Allow it in Settings → Privacy → Speech Recognition.")
        case .noSpeech: return L("No speech was heard.")
        case .notOffline(let locale): return L("This iPhone cannot recognise %@ offline. Turn off \"Only on this iPhone\" or choose a remote speech model.", locale)
        case .speech(let detail): return L("Speech recognition failed: %@", detail)
        case .languageModel(let detail): return L("The language model failed: %@", detail)
        case .config(let detail): return L("Check Settings → Language models: %@", detail)
        case .speechConfig(let detail): return L("Check Settings → Speech models: %@", detail)
        case .missing("key"): return L("Add the API key in Settings → Language models.")
        case .missing("model"): return L("Choose a model in Settings → Language models.")
        case .missing("address"): return L("Enter the API address in Settings → Language models.")
        case .missing("sign_in"): return L("This service needs a sign-in, which the iOS app does not offer yet. Choose another in Settings → Language models.")
        case .missing: return L("Choose a service in Settings → Language models.")
        case .notTranslated: return L("The language model did not translate the text.")
        case .emptyAnswer: return L("The language model returned an empty answer.")
        case .noSource: return L("Type or paste the text first.")
        case .other(let detail): return detail
        }
    }
}

enum Clipboard {
    static func copy(_ text: String) {
        #if os(iOS)
        UIPasteboard.general.string = text
        #else
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        #endif
    }

    static var text: String? {
        #if os(iOS)
        UIPasteboard.general.string
        #else
        NSPasteboard.general.string(forType: .string)
        #endif
    }
}
