import Foundation

public struct TailcatIdentity: Equatable, Codable, Sendable {
    public let privateKey: String
    public let publicKey: String
}

enum NativeResponse {
    private struct Reply: Decodable {
        let error: String?
        let handle: Int64?
        let port: Int?
        let privateKey: String?
        let publicKey: String?
    }

    private static func reply(from data: Data) throws -> Reply {
        let reply: Reply
        do { reply = try JSONDecoder().decode(Reply.self, from: data) }
        catch { throw TailcatError.invalidResponse }
        // Native error payloads never become user-visible text or log messages.
        guard reply.error == nil else { throw TailcatError.startFailed }
        return reply
    }

    static func endpoint(from data: Data) throws -> NativeEndpoint {
        let reply = try reply(from: data)
        guard let handle = reply.handle, handle > 0,
              let port = reply.port, (1...65535).contains(port) else {
            throw TailcatError.invalidResponse
        }
        return NativeEndpoint(handle: handle, port: port)
    }

    static func identity(from data: Data) throws -> TailcatIdentity {
        let reply = try reply(from: data)
        guard let privateKey = reply.privateKey, validKey(privateKey, prefix: "privkey:"),
              let publicKey = reply.publicKey, validKey(publicKey, prefix: "nodekey:") else {
            throw TailcatError.invalidResponse
        }
        return TailcatIdentity(privateKey: privateKey, publicKey: publicKey)
    }

    private static func validKey(_ value: String, prefix: String) -> Bool {
        guard value.hasPrefix(prefix) else { return false }
        let bytes = value.dropFirst(prefix.count).utf8
        return bytes.count == 64 && bytes.allSatisfy {
            (48...57).contains($0) || (65...70).contains($0) || (97...102).contains($0)
        }
    }
}
