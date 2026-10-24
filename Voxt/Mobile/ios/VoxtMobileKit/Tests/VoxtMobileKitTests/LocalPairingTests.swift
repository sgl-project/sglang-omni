import CryptoKit
import Foundation
import Testing
@testable import VoxtMobileKit

/// Same-Wi-Fi pairing: key exchange, a 6-digit
/// code shown on both screens, and the QR invitation sent sealed after Allow on the Mac.
@Suite struct LocalPairingTests {
    private let phonePrivate = try! Curve25519.KeyAgreement.PrivateKey(rawRepresentation: Data(repeating: 1, count: 32))
    private let macPrivate = try! Curve25519.KeyAgreement.PrivateKey(rawRepresentation: Data(repeating: 2, count: 32))

    @Test func bothSidesDeriveTheSameCodeAndItMatchesTheFormula() throws {
        let phone = try LocalPairingCrypto(privateKey: phonePrivate, peerPublicKey: macPrivate.publicKey.rawRepresentation, isPhone: true)
        let mac = try LocalPairingCrypto(privateKey: macPrivate, peerPublicKey: phonePrivate.publicKey.rawRepresentation, isPhone: false)
        #expect(phone.code == mac.code)
        #expect(phone.code.count == 6 && phone.code.allSatisfy(\.isNumber))
        // Formula, checked independently: first 8 bytes of SHA-256(label ‖ phoneKey ‖ macKey ‖ shared), big-endian, mod 10^6.
        let shared = try phonePrivate.sharedSecretFromKeyAgreement(with: macPrivate.publicKey).withUnsafeBytes { Data($0) }
        var digest = SHA256()
        digest.update(data: Data("voxt-local-pairing-v1".utf8))
        digest.update(data: phonePrivate.publicKey.rawRepresentation)
        digest.update(data: macPrivate.publicKey.rawRepresentation)
        digest.update(data: shared)
        let value = Data(digest.finalize()).prefix(8).reduce(UInt64(0)) { $0 << 8 | UInt64($1) }
        #expect(phone.code == String(format: "%06llu", value % 1_000_000))
    }

    @Test func invitationIsSealedForThisPhoneOnly() throws {
        let invitation = URL(string: "voxtmobile://connect?v=1&tailcat=tc-fixture&port=48932&key=privkey:00&proof=fixture&expires=1")!
        let mac = try LocalPairingCrypto(privateKey: macPrivate, peerPublicKey: phonePrivate.publicKey.rawRepresentation, isPhone: false)
        let sealed = try mac.seal(invitation)
        #expect(!String(decoding: sealed, as: UTF8.self).contains("tc-fixture"))
        let phone = try LocalPairingCrypto(privateKey: phonePrivate, peerPublicKey: macPrivate.publicKey.rawRepresentation, isPhone: true)
        #expect(try phone.open(sealed) == invitation)
        let stranger = Curve25519.KeyAgreement.PrivateKey()
        let other = try LocalPairingCrypto(privateKey: stranger, peerPublicKey: macPrivate.publicKey.rawRepresentation, isPhone: true)
        #expect(throws: (any Error).self) { try other.open(sealed) }
    }

    @Test func clientShowsTheCodeThenReturnsTheInvitationAfterAllow() async throws {
        var fds: [Int32] = [0, 0]
        #expect(socketpair(AF_UNIX, SOCK_STREAM, 0, &fds) == 0)
        let invitation = URL(string: "voxtmobile://connect?v=1&tailcat=tc-fixture&port=48932&key=privkey:00&proof=fixture&expires=1")!
        let macFD = fds[1], phoneFD = fds[0], macKey = macPrivate
        let macSide = Task.detached { () throws -> String in
            let macPrivate = macKey
            let handle = FileHandle(fileDescriptor: macFD, closeOnDealloc: true)
            let hello = try LocalPairingWire.readLine(macFD)
            let phoneKey = try LocalPairingWire.decodeHello(hello)
            handle.write(LocalPairingWire.macHello(publicKey: macPrivate.publicKey.rawRepresentation))
            let crypto = try LocalPairingCrypto(privateKey: macPrivate, peerPublicKey: phoneKey, isPhone: false)
            handle.write(try LocalPairingWire.approved(sealed: crypto.seal(invitation)))
            return crypto.code
        }
        final class Shown: @unchecked Sendable { var code: String? }
        let shown = Shown()
        let client = LocalPairingClient(privateKey: phonePrivate)
        let result = try await client.pair(fd: phoneFD, deviceName: "iPhone") { shown.code = $0 }
        #expect(result == invitation)
        #expect(shown.code == (try await macSide.value))
    }

    @Test func busyMacRefusingBeforeTheKeyExchangeIsReportedAsDenied() async throws {
        var fds: [Int32] = [0, 0]
        #expect(socketpair(AF_UNIX, SOCK_STREAM, 0, &fds) == 0)
        let macFD = fds[1], phoneFD = fds[0]
        let macSide = Task.detached { () throws -> Void in
            let handle = FileHandle(fileDescriptor: macFD, closeOnDealloc: true)
            _ = try LocalPairingWire.readLine(macFD)
            handle.write(LocalPairingWire.denied())
        }
        await #expect(throws: LocalPairingClient.Failure.denied) {
            _ = try await LocalPairingClient(privateKey: phonePrivate, replyTimeout: 2).pair(fd: phoneFD, deviceName: "iPhone") { _ in }
        }
        _ = try? await macSide.value
    }

    @Test func deniedOrSilentMacFallsBackToTheQR() async throws {
        for reply in [#"{"v":1,"status":"denied"}"#, nil] as [String?] {
            var fds: [Int32] = [0, 0]
            #expect(socketpair(AF_UNIX, SOCK_STREAM, 0, &fds) == 0)
            let macFD = fds[1], phoneFD = fds[0], macKey = macPrivate
            let macSide = Task.detached { () throws -> Void in
                let macPrivate = macKey
                let handle = FileHandle(fileDescriptor: macFD, closeOnDealloc: true)
                _ = try LocalPairingWire.readLine(macFD)
                handle.write(LocalPairingWire.macHello(publicKey: macPrivate.publicKey.rawRepresentation))
                if let reply { handle.write(Data((reply + "\n").utf8)) } else { sleep(2) }
            }
            let client = LocalPairingClient(privateKey: phonePrivate, replyTimeout: 1)
            await #expect(throws: LocalPairingClient.Failure.self) {
                _ = try await client.pair(fd: phoneFD, deviceName: "iPhone") { _ in }
            }
            _ = try? await macSide.value
        }
    }
}
