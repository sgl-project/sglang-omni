import Foundation

/// A short-lived QR capability for the isolated pairing endpoint.
public struct PairingBootstrap: Sendable {
    public enum Error: Swift.Error { case invalidLink, expired }
    public let address: String
    public let port: UInt16
    public let temporaryKey: String
    public let proof: String
    public let expiresAt: Date

    public init(url: URL, now: Date = Date()) throws {
        guard let c = URLComponents(url: url, resolvingAgainstBaseURL: false),
              c.scheme == "voxtmobile", c.host == "connect", c.path.isEmpty,
              c.fragment == nil, c.user == nil, c.password == nil, c.port == nil,
              let items = c.queryItems else { throw Error.invalidLink }
        // Fields the Mac adds later are ignored. The known ones must each appear once, and the
        // access token never belongs in a QR code.
        let known: Set = ["v", "tailcat", "port", "key", "proof", "expires"]
        let names = items.map(\.name)
        guard known.allSatisfy({ name in names.filter { $0 == name }.count == 1 }), !names.contains("token")
        else { throw Error.invalidLink }
        var values: [String: String] = [:]
        for item in items where known.contains(item.name) {
            guard let value = item.value, !value.isEmpty,
                  !value.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) }) else {
                throw Error.invalidLink
            }
            values[item.name] = value
        }
        guard values["v"] == "1", let address = values["tailcat"], address.hasPrefix("tc"), !address.contains(where: \.isWhitespace),
              let rawPort = values["port"], rawPort.utf8.allSatisfy({ (48...57).contains($0) }),
              let port = UInt16(rawPort), port == 48932,
              let key = values["key"], key.hasPrefix("privkey:"), key.utf8.count == 72,
              key.dropFirst(8).utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
              let proof = values["proof"], proof.count <= 256, !proof.contains(where: \.isWhitespace) else { throw Error.invalidLink }
        guard let rawExpiry = values["expires"], let expiry = TimeInterval(rawExpiry), expiry.isFinite,
              expiry <= now.timeIntervalSince1970 + 125 else { throw Error.invalidLink }
        guard expiry > now.timeIntervalSince1970 else { throw Error.expired }
        expiresAt = Date(timeIntervalSince1970: expiry)
        self.address = address
        self.port = port
        self.temporaryKey = key
        self.proof = proof
    }
}
