import Foundation

/// Mac names in compact places are shortened in the middle.
public enum ConnectionStatusText {
    public static let maximumNameLength = 24

    public static func shortened(_ name: String) -> String {
        guard name.count > maximumNameLength else { return name }
        let head = (maximumNameLength - 1) / 2
        let tail = maximumNameLength - 1 - head
        return String(name.prefix(head)) + "…" + String(name.suffix(tail))
    }
}

/// What the Mac status card on the home page and in Settings shows:
/// a glyph and the Mac's name, then a state dot and one line of state.
public struct MacStatusCard: Equatable, Sendable {
    public enum State: Equatable, Sendable {
        case connected(version: String)
        case connecting
        case unreachable(reason: String?)
        /// The Mac answers, but Voxt there uses a cloud model, so dictation stays off.
        case cloudModel(version: String)
    }

    public enum Dot: Equatable, Sendable { case green, orange }

    public let title: String
    public let symbol: String
    public let detail: String
    /// nil while connecting: the card shows progress instead.
    public let dot: Dot?
    public let showsRetry: Bool
    /// A smaller third line, when there is something specific to say.
    public let reason: String?
    /// The whole card is one VoiceOver element.
    public let accessibilityLabel: String

    /// `language` nil: the language picked in Voxt's Settings, else the system's.
    public init(macName: String?, state: State, language: String? = nil, l10n: VoxtL10n = .shared) {
        let language = language ?? l10n.languageCode
        let name = macName?.trimmingCharacters(in: .whitespacesAndNewlines)
        title = (name?.isEmpty == false ? name : nil) ?? l10n.string("Your Mac", language: language)
        symbol = Self.symbol(forMacName: name)
        switch state {
        case let .connected(version):
            let short = Self.shortVersion(version)
            detail = l10n.format("Connected · Voxt %@", language: language, arguments: [short])
            dot = .green
            showsRetry = false
            reason = nil
            accessibilityLabel = l10n.format("%1$@, connected, Voxt %2$@", language: language, arguments: [title, short])
        case .connecting:
            detail = l10n.string("Connecting…", language: language)
            dot = nil
            showsRetry = false
            reason = nil
            accessibilityLabel = l10n.format("%@, connecting", language: language, arguments: [title])
        case let .unreachable(why):
            detail = l10n.string("Can't reach your Mac", language: language)
            dot = .orange
            showsRetry = true
            reason = l10n.string(why.flatMap { Self.repeatsUnreachable.contains($0) ? nil : $0 }
                                 ?? "Check that Voxt and Voxt Connector are open on your Mac.", language: language)
            accessibilityLabel = l10n.format("%@, can't reach your Mac", language: language, arguments: [title])
        case .cloudModel:
            detail = l10n.string("Cloud model on your Mac", language: language)
            dot = .orange
            showsRetry = false
            reason = l10n.string("Dictation stays off until Voxt uses a local model.", language: language)
            accessibilityLabel = l10n.format("%@, cloud model on your Mac", language: language, arguments: [title])
        }
    }

    /// Failure texts that only say the Mac can't be reached; the card says what to do instead.
    static let repeatsUnreachable: Set<String> = [
        "The connection to your Mac is not ready. Reconnect and try again.",
        "Can't reach your Mac. Check your internet connection and make sure Voxt is running on the Mac.",
        "Can't reach your Mac. Check that Tailscale is connected on this phone and Voxt is running on the Mac.",
    ]

    /// The SF Symbol for the kind of Mac the name suggests.
    public static func symbol(forMacName name: String?) -> String {
        let lower = name?.lowercased() ?? ""
        if lower.contains("macbook") { return "laptopcomputer" }
        if lower.contains("mini") { return "macmini" }
        if lower.contains("studio") { return "macstudio" }
        return "desktopcomputer"
    }

    /// "1.16.0" reads as "1.16"; other versions are shown as they are.
    static func shortVersion(_ version: String) -> String {
        version.hasSuffix(".0") && version.split(separator: ".").count == 3 ? String(version.dropLast(2)) : version
    }
}
