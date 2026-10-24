import Foundation

/// How long the microphone stays ready after a dictation. While it is ready iOS shows the
/// recording indicator; once it closes, the next keyboard dictation opens Voxt first.
public enum MicrophoneReadiness: String, CaseIterable, Sendable {
    case immediately, fiveMinutes, fifteenMinutes, oneHour, never

    private static let key = "microphone.disableAfter"

    /// Seconds of idle time before the session closes; nil keeps it open.
    public var timeout: TimeInterval? {
        switch self {
        case .immediately: 0
        case .fiveMinutes: DictationController.idleSessionTimeout
        case .fifteenMinutes: 15 * 60
        case .oneHour: 60 * 60
        case .never: nil
        }
    }

    /// Settings › Microphone: how long the mic stays on after a dictation.
    public static let settingTitleKey = "Keep the mic on after dictating"
    public static let settingCaptionKey = "While it's on, your next dictation starts without opening Voxt, and iOS shows the orange mic dot. With Off, every dictation opens Voxt first."

    /// Catalog key of the choice; "Off" and "Always" have keys of their own, so their
    /// translations ("不保持", "一直保持") don't leak into other uses of the words.
    public var titleKey: String {
        switch self {
        case .immediately: "Keep the mic on: Off"
        case .fiveMinutes: "5 minutes"
        case .fifteenMinutes: "15 minutes"
        case .oneHour: "1 hour"
        case .never: "Keep the mic on: Always"
        }
    }

    public var title: String { VoxtL10n.shared.string(titleKey) }

    public static func load(from defaults: UserDefaults) -> MicrophoneReadiness {
        defaults.string(forKey: key).flatMap(MicrophoneReadiness.init(rawValue:)) ?? .fiveMinutes
    }

    public func save(to defaults: UserDefaults) {
        defaults.set(rawValue, forKey: Self.key)
    }
}
