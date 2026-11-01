import Foundation

public struct ConnectorPairingCode: Equatable, Sendable {
    public enum Failure: Error { case invalidPayload }
    public let url: URL
    public let expiresAt: Date
    public init(url: URL, now: Date = Date()) throws {
        guard let c = URLComponents(url: url, resolvingAgainstBaseURL: false),
              c.scheme == "voxtmobile", c.host == "connect", c.path.isEmpty,
              c.user == nil, c.password == nil, c.port == nil, c.fragment == nil,
              let items = c.queryItems else { throw Failure.invalidPayload }
        // Fields added later are ignored. The known ones must each appear once, and the access
        // token never belongs in a QR code.
        let known: Set = ["v", "tailcat", "port", "key", "proof", "expires"]
        let names = items.map(\.name)
        guard known.allSatisfy({ name in names.filter { $0 == name }.count == 1 }), !names.contains("token")
        else { throw Failure.invalidPayload }
        let values = Dictionary(items.filter { known.contains($0.name) }.map { ($0.name, $0.value ?? "") },
                                uniquingKeysWith: { first, _ in first })
        guard values["v"] == "1", let address = values["tailcat"], address.hasPrefix("tc"),
              !address.contains(where: \.isWhitespace), values["port"] == "48932",
              let key = values["key"], key.hasPrefix("privkey:"), key.utf8.count == 72,
              key.dropFirst(8).utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
              let proof = values["proof"], !proof.isEmpty, proof.utf8.count <= 256,
              proof.utf8.allSatisfy({ (48...57).contains($0) || (65...90).contains($0) || (97...122).contains($0) || $0 == 45 || $0 == 95 }),
              let rawExpiry = values["expires"], let expiry = TimeInterval(rawExpiry), expiry.isFinite,
              expiry > now.timeIntervalSince1970, expiry <= now.timeIntervalSince1970 + 125 else { throw Failure.invalidPayload }
        self.url = url; self.expiresAt = Date(timeIntervalSince1970: expiry)
    }
}
