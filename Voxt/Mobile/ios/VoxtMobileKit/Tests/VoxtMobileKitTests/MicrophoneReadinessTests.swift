import Foundation
import Testing
@testable import VoxtMobileKit

/// Settings › Microphone says what the choice does: how long the mic stays on after a
/// dictation ("Immediately" under "Turn off after" reads as the opposite).
@Suite(.serialized)
struct MicrophoneReadinessTests {
    private func resolver() -> (VoxtL10n, UserDefaults, String) {
        let name = "mic-readiness.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        return (VoxtL10n(defaults: defaults), defaults, name)
    }

    @Test func choicesReadAsHowLongTheMicStaysOn() {
        let (l10n, defaults, name) = resolver()
        defer { defaults.removePersistentDomain(forName: name) }
        let english = MicrophoneReadiness.allCases.map { l10n.string($0.titleKey, language: "en") }
        #expect(english == ["Off", "5 minutes", "15 minutes", "1 hour", "Always"])
        let chinese = MicrophoneReadiness.allCases.map { l10n.string($0.titleKey, language: "zh-Hans") }
        #expect(chinese == ["不保持", "5 分钟", "15 分钟", "1 小时", "一直保持"])
        #expect(l10n.string(MicrophoneReadiness.settingTitleKey, language: "zh-Hans") == "说完后保持麦克风")
        #expect(l10n.string(MicrophoneReadiness.settingCaptionKey, language: "zh-Hans")
                == "保持期间，下次说话不用跳回 Voxt，iOS 会在右上角显示橙色麦克风。选「不保持」，每次说话都会先打开 Voxt。")
    }

    @Test func savedChoicesKeepTheirMeaning() {
        // Raw values are stored in the App Group; the wording change must not move anyone's choice.
        #expect(MicrophoneReadiness.allCases.map(\.rawValue) == ["immediately", "fiveMinutes", "fifteenMinutes", "oneHour", "never"])
        #expect(MicrophoneReadiness.immediately.timeout == 0)
        #expect(MicrophoneReadiness.never.timeout == nil)
    }
}
