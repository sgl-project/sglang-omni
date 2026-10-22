import Foundation

nonisolated enum TailcatError: Error, Equatable {
    case invalidConfiguration, invalidReply, helperRejected
}

nonisolated struct TailcatConfiguration: Sendable {
    let state: String?
    let allowedClients: [String]
    let localPort: Int

    func encoded() throws -> Data {
        guard (1024...65534).contains(localPort), !allowedClients.isEmpty,
              Set(allowedClients).count == allowedClients.count,
              allowedClients.allSatisfy(Self.isPublicKey) else {
            throw TailcatError.invalidConfiguration
        }
        return try JSONSerialization.data(withJSONObject: [
            "version": 1, "state": state.map { $0 as Any } ?? NSNull(),
            "allowedClients": allowedClients, "localPort": localPort
        ], options: [.sortedKeys])
    }

    static func isPublicKey(_ value: String) -> Bool {
        value.hasPrefix("nodekey:") && value.utf8.count == 72 &&
            value.dropFirst(8).utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }
}

nonisolated struct TailcatReady: Sendable {
    let address: String
    let port: Int
    let state: String

    init(data: Data) throws {
        struct Reply: Decodable {
            let version: Int
            let address: String
            let port: Int
            let state: String
            let error: String?
        }
        guard let reply = try? JSONDecoder().decode(Reply.self, from: data),
              reply.error == nil, reply.version == 1, reply.port == 48931,
              reply.address.hasPrefix("tc"), !reply.address.contains(where: { $0.isWhitespace }),
              !reply.state.isEmpty else { throw TailcatError.invalidReply }
        address = reply.address
        port = reply.port
        state = reply.state
    }
}
