import CryptoKit
import Foundation

/// Same-Wi-Fi pairing without the QR: an
/// ephemeral X25519 exchange, a 6-digit code shown on the phone and on the Mac, and the QR's
/// invitation sent sealed after the user clicks Allow on the Mac.
public struct LocalPairingCrypto: Sendable {
    public let code: String
    private let key: SymmetricKey

    public init(privateKey: Curve25519.KeyAgreement.PrivateKey, peerPublicKey: Data, isPhone: Bool) throws {
        let peer = try Curve25519.KeyAgreement.PublicKey(rawRepresentation: peerPublicKey)
        let shared = try privateKey.sharedSecretFromKeyAgreement(with: peer)
        let own = privateKey.publicKey.rawRepresentation
        let phoneKey = isPhone ? own : peerPublicKey
        let macKey = isPhone ? peerPublicKey : own
        var digest = SHA256()
        digest.update(data: Data("voxt-local-pairing-v1".utf8))
        digest.update(data: phoneKey)
        digest.update(data: macKey)
        digest.update(data: shared.withUnsafeBytes { Data($0) })
        let value = Data(digest.finalize()).prefix(8).reduce(UInt64(0)) { $0 << 8 | UInt64($1) }
        code = String(format: "%06llu", value % 1_000_000)
        key = shared.hkdfDerivedSymmetricKey(using: SHA256.self, salt: phoneKey + macKey,
                                             sharedInfo: Data("voxt-local-pairing-v1 invitation".utf8), outputByteCount: 32)
    }

    public func seal(_ invitation: URL) throws -> Data {
        try ChaChaPoly.seal(Data(invitation.absoluteString.utf8), using: key).combined
    }

    public func open(_ sealed: Data) throws -> URL {
        let plain = try ChaChaPoly.open(ChaChaPoly.SealedBox(combined: sealed), using: key)
        guard let url = URL(string: String(decoding: plain, as: UTF8.self)) else { throw LocalPairingClient.Failure.invalidReply }
        return url
    }
}

/// Newline-delimited JSON messages over the local TCP connection.
public enum LocalPairingWire {
    public static func readLine(_ fd: Int32) throws -> Data {
        var line = Data()
        var byte: UInt8 = 0
        while line.count < 4096 {
            let count = read(fd, &byte, 1)
            if count == 1 {
                if byte == 0x0A { return line }
                line.append(byte)
            } else if count == 0 || errno == EAGAIN || errno == EWOULDBLOCK {
                throw LocalPairingClient.Failure.timedOut
            } else {
                throw LocalPairingClient.Failure.invalidReply
            }
        }
        throw LocalPairingClient.Failure.invalidReply
    }

    public static func phoneHello(publicKey: Data, deviceName: String) -> Data {
        line(["v": 1, "publicKey": publicKey.base64EncodedString(), "device": String(deviceName.prefix(64))])
    }

    public static func macHello(publicKey: Data) -> Data {
        line(["v": 1, "publicKey": publicKey.base64EncodedString()])
    }

    public static func approved(sealed: Data) -> Data {
        line(["v": 1, "status": "approved", "sealed": sealed.base64EncodedString()])
    }

    /// Why the Mac said no. `expired`: nobody answered before the Add iPhone window closed.
    public enum DenialReason: String, Sendable { case expired }

    public static func denied(reason: DenialReason? = nil) -> Data {
        var fields: [String: Any] = ["v": 1, "status": "denied"]
        if let reason { fields["reason"] = reason.rawValue }
        return line(fields)
    }

    /// The 32-byte X25519 key in a hello from either side.
    public static func decodeHello(_ line: Data) throws -> Data {
        guard let object = try? JSONSerialization.jsonObject(with: line) as? [String: Any],
              object["v"] as? Int == 1, let text = object["publicKey"] as? String,
              let key = Data(base64Encoded: text), key.count == 32 else { throw LocalPairingClient.Failure.invalidReply }
        return key
    }

    private static func line(_ object: [String: Any]) -> Data {
        var data = (try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])) ?? Data()
        data.append(0x0A)
        return data
    }
}

/// One newline-delimited message stream; a socket in tests, an NWConnection in the app.
public protocol LocalPairingChannel: Sendable {
    func send(_ data: Data) async throws
    func readLine() async throws -> Data
}

/// A connected socket (tests, and any BSD socket).
public struct SocketPairingChannel: LocalPairingChannel {
    let fd: Int32
    public init(fd: Int32, timeoutSeconds: Int) {
        self.fd = fd
        var timeout = timeval(tv_sec: timeoutSeconds, tv_usec: 0)
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
    }
    public func send(_ data: Data) async throws {
        let fd = fd
        try await Task.detached {
            guard data.withUnsafeBytes({ write(fd, $0.baseAddress, $0.count) }) == data.count else {
                throw LocalPairingClient.Failure.invalidReply
            }
        }.value
    }
    public func readLine() async throws -> Data {
        let fd = fd
        return try await Task.detached { try LocalPairingWire.readLine(fd) }.value
    }
}

/// The phone's side: hello, show the code, then wait for the Mac's Allow.
public struct LocalPairingClient: Sendable {
    public enum Failure: Error, Equatable { case denied, timedOut, invalidReply }

    /// Don't Allow is a refusal; a window that closed without an answer is no answer.
    static func failure(forDenial reply: Data) -> Failure {
        let reason = (try? JSONSerialization.jsonObject(with: reply) as? [String: Any])?["reason"] as? String
        return reason == LocalPairingWire.DenialReason.expired.rawValue ? .timedOut : .denied
    }
    private let privateKey: Curve25519.KeyAgreement.PrivateKey
    private let replyTimeout: Int

    public init(privateKey: Curve25519.KeyAgreement.PrivateKey = .init(), replyTimeout: Int = 130) {
        self.privateKey = privateKey
        self.replyTimeout = replyTimeout
    }

    public func pair(fd: Int32, deviceName: String, showCode: @escaping @Sendable (String) -> Void) async throws -> URL {
        try await pair(channel: SocketPairingChannel(fd: fd, timeoutSeconds: replyTimeout), deviceName: deviceName, showCode: showCode)
    }

    /// Returns the same invitation a QR scan would give; the caller runs the normal pairing exchange.
    public func pair(channel: some LocalPairingChannel, deviceName: String,
                     showCode: @escaping @Sendable (String) -> Void) async throws -> URL {
        let privateKey = privateKey
        do {
            try await channel.send(LocalPairingWire.phoneHello(publicKey: privateKey.publicKey.rawRepresentation, deviceName: deviceName))
            let first = try await channel.readLine()
            // A busy Mac, or a closed Add iPhone window, refuses before the key exchange.
            if (try? JSONSerialization.jsonObject(with: first) as? [String: Any])?["status"] as? String == "denied" {
                throw Self.failure(forDenial: first)
            }
            let macKey = try LocalPairingWire.decodeHello(first)
            let crypto = try LocalPairingCrypto(privateKey: privateKey, peerPublicKey: macKey, isPhone: true)
            showCode(crypto.code)
            let decision = try await channel.readLine()
            guard let object = try? JSONSerialization.jsonObject(with: decision) as? [String: Any] else { throw Failure.invalidReply }
            switch object["status"] as? String {
            case "approved":
                guard let text = object["sealed"] as? String, let sealed = Data(base64Encoded: text) else { throw Failure.invalidReply }
                guard let url = try? crypto.open(sealed) else { throw Failure.invalidReply }
                return url
            case "denied": throw Self.failure(forDenial: decision)
            default: throw Failure.invalidReply
            }
        } catch let failure as Failure {
            throw failure
        } catch {
            throw Failure.invalidReply
        }
    }
}
