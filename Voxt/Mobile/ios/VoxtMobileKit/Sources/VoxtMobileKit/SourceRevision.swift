import Foundation

/// The public commit a release was built from: `VoxtSourceRevision` in Info.plist, which the release
/// build fills in. Nil for a development build (empty, or the unexpanded setting).
public struct SourceRevision: Equatable, Sendable {
    public let full: String
    public let repository: URL

    public init?(info: [String: Any]?) {
        guard let value = info?["VoxtSourceRevision"] as? String,
              value.count == 40, value.allSatisfy(\.isHexDigit) else { return nil }
        full = value.lowercased()
        repository = Self.repository(info: info)
    }

    /// The repository the build names in `VoxtSourceURL` (build setting `VOXT_SOURCE_URL`),
    /// upstream when it names none. xcconfig reads `//` as a comment, so the setting has no scheme.
    public static func repository(info: [String: Any]?) -> URL {
        let upstream = URL(string: "https://github.com/sgl-project/sglang-omni")!
        let urlKey = "VoxtSourceURL"
        guard var text = info?[urlKey] as? String, !text.isEmpty, !text.contains("$(") else { return upstream }
        if !text.contains("://") { text = "https://" + text }
        while text.hasSuffix("/") { text.removeLast() }
        return URL(string: text) ?? upstream
    }

    public var short: String { String(full.prefix(7)) }
    public var url: URL { repository.appendingPathComponent("commit").appendingPathComponent(full) }
}
