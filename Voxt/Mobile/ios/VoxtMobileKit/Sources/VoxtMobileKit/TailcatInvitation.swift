import Foundation

/// A Mac invitation names an already-authorized phone public key; it never supplies its private key.
public struct TailcatInvitation: Sendable {
    public enum Error: Swift.Error, Equatable { case invalidLink, differentDevice }
    private let address: String
    private let port: UInt16
    private let token: String
    private let client: String
    private let name: String?

    public init(pairingLink: URL) throws {
        guard let components = URLComponents(url: pairingLink, resolvingAgainstBaseURL: false),
              components.scheme == "voxtmobile", components.host == "pair",
              components.path.isEmpty, components.fragment == nil,
              components.user == nil, components.password == nil, components.port == nil,
              let items = components.queryItems else { throw Error.invalidLink }
        // The four known fields appear once each; `name` (the Mac's name, for display only) at most
        // once; fields the Mac adds later are ignored.
        let required: Set = ["tailcat", "port", "token", "client"]
        let names = items.map(\.name)
        guard required.allSatisfy({ name in names.filter { $0 == name }.count == 1 }),
              names.filter({ $0 == "name" }).count <= 1 else { throw Error.invalidLink }
        var values: [String: String] = [:]
        for item in items where required.contains(item.name) || item.name == "name" {
            guard let value = item.value, !value.isEmpty,
                  !value.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) }) else {
                throw Error.invalidLink
            }
            values[item.name] = value
        }
        guard let address = values["tailcat"], !address.contains(where: \.isWhitespace),
              let portString = values["port"], portString.utf8.allSatisfy({ (48...57).contains($0) }),
              let port = UInt16(portString), port > 0,
              let token = values["token"], !token.trimmingCharacters(in: .whitespaces).isEmpty,
              let client = values["client"], Self.validPublicKey(client) else { throw Error.invalidLink }
        self.address = address
        self.port = port
        self.token = token
        self.client = client
        let name = values["name"]?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.name = name.map { String($0.prefix(64)) }.flatMap { $0.isEmpty ? nil : $0 }
    }

    public func server(clientKey: String, publicKey: String) throws -> ServerConfig {
        guard publicKey == client else { throw Error.differentDevice }
        return ServerConfig(baseURL: URL(string: "http://tailcat.invalid")!, token: token,
                            tailcat: .init(address: address, clientKey: clientKey, remotePort: port), name: name)
    }

    private static func validPublicKey(_ value: String) -> Bool {
        let hex = value.dropFirst("nodekey:".count)
        return value.hasPrefix("nodekey:") && hex.utf8.count == 64 && hex.utf8.allSatisfy {
            (48...57).contains($0) || (65...70).contains($0) || (97...102).contains($0)
        }
    }
}
