import Foundation

/// The Connector's bundle ID and the Voxt builds it trusts. project.yml sets the Connector's ID;
/// a fork that renames Voxt lists its bundle IDs, separated by spaces, under VoxtBundleIdentifiers
/// in the Connector's Info.plist. Voxt reads the Connector's ID the same way, from
/// VoxtConnectorBundleIdentifier in its own Info.plist.
public enum ConnectorIdentity {
    public static let defaultBundleIdentifier = "com.voxt.VoxtConnector"
    public static let defaultVoxtBundleIdentifiers = ["com.voxt.Voxt", "com.voxt.Voxt.omni.dev"]
    public static let voxtBundleIdentifiersKey = "VoxtBundleIdentifiers"

    /// Subsystem for this app's logs.
    public static var bundleIdentifier: String { Bundle.main.bundleIdentifier ?? defaultBundleIdentifier }

    /// The Info.plist override when every entry is a bundle ID, otherwise the defaults: the list
    /// goes into a code requirement, where a quote would end the identifier string.
    public static func voxtBundleIdentifiers(info: [String: Any]? = Bundle.main.infoDictionary) -> [String] {
        let value = info?[voxtBundleIdentifiersKey]
        let listed = (value as? [String]) ?? (value as? String).map { $0.split(separator: " ").map(String.init) } ?? []
        return !listed.isEmpty && listed.allSatisfy(isBundleIdentifier) ? listed : defaultVoxtBundleIdentifiers
    }

    static func isBundleIdentifier(_ text: String) -> Bool {
        !text.isEmpty && text.allSatisfy { $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "." || $0 == "-") }
    }
}
