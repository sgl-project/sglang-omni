import CryptoKit
import Foundation
import Testing
@testable import VoxtConnectionKit

/// The Connector's half of same-Wi-Fi pairing.
@Suite struct LocalPairingSessionTests {
    private let invitation = URL(string: "voxtmobile://connect?v=1&tailcat=tc-fixture&port=48932&key=privkey:00&proof=fixture&expires=1")!

    private func pair() -> (phone: Int32, mac: Int32) {
        var fds: [Int32] = [0, 0]
        #expect(socketpair(AF_UNIX, SOCK_STREAM, 0, &fds) == 0)
        return (fds[0], fds[1])
    }

    /// A minimal phone: hello, then the Mac's hello and decision.
    private func phone(_ fd: Int32) -> Task<(code: String, decision: [String: String]), Error> {
        Task.detached {
            let key = Curve25519.KeyAgreement.PrivateKey()
            let channel = SocketPairingChannel(fd: fd, timeoutSeconds: 5)
            try await channel.send(LocalPairingWire.phoneHello(publicKey: key.publicKey.rawRepresentation, deviceName: "Test iPhone"))
            let first = try await channel.readLine()
            if let early = (try? JSONSerialization.jsonObject(with: first) as? [String: Any])?.compactMapValues({ $0 as? String }),
               early["status"] != nil {
                close(fd)
                return ("", early)   // refused before the key exchange (busy or closed window)
            }
            let macKey = try LocalPairingWire.decodeHello(first)
            let crypto = try LocalPairingCrypto(privateKey: key, peerPublicKey: macKey, isPhone: true)
            let raw = (try JSONSerialization.jsonObject(with: try await channel.readLine()) as? [String: Any]) ?? [:]
            var decision = raw.compactMapValues { $0 as? String }
            if let sealed = decision["sealed"].flatMap({ Data(base64Encoded: $0) }) {
                decision["opened"] = try crypto.open(sealed).absoluteString
            }
            close(fd)
            return (crypto.code, decision)
        }
    }

    @Test func codeMatchesTheIPhoneImplementation() throws {
        let phoneKey = try Curve25519.KeyAgreement.PrivateKey(rawRepresentation: Data(repeating: 1, count: 32))
        let macKey = try Curve25519.KeyAgreement.PrivateKey(rawRepresentation: Data(repeating: 2, count: 32))
        let mac = try LocalPairingCrypto(privateKey: macKey, peerPublicKey: phoneKey.publicKey.rawRepresentation, isPhone: false)
        #expect(mac.code == "078122", "frozen vector from ios/VoxtMobileKit LocalPairingCrypto")
    }

    @Test func allowSendsTheSealedInvitationAndBothSidesShowTheSameCode() async throws {
        let (phoneFD, macFD) = pair()
        let phoneSide = phone(phoneFD)
        final class Seen: @unchecked Sendable { var code = ""; var device = "" }
        let seen = Seen()
        let coordinator = LocalPairingCoordinator(invitation: invitation, expiresAt: Date().addingTimeInterval(120)) { code, device in
            seen.code = code; seen.device = device; return true
        }
        await coordinator.serve(SocketPairingChannel(fd: macFD, timeoutSeconds: 5))
        let result = try await phoneSide.value
        #expect(result.decision["status"] == "approved")
        #expect(result.decision["opened"] == invitation.absoluteString)
        #expect(seen.code == result.code)
        #expect(seen.device == "Test iPhone")
        close(macFD)
    }

    @Test func denyOrNoAnswerBeforeExpiryRefuses() async throws {
        // Don't Allow carries no reason; a window that closed unanswered says "expired",
        // so the phone shows "No answer from your Mac" instead of "Your Mac said no".
        for (expiry, answer, reason) in [(120.0, false, nil), (0.5, true, "expired")] as [(Double, Bool, String?)] {
            let (phoneFD, macFD) = pair()
            let phoneSide = phone(phoneFD)
            let coordinator = LocalPairingCoordinator(invitation: invitation, expiresAt: Date().addingTimeInterval(expiry)) { _, _ in
                if answer { try? await Task.sleep(for: .seconds(3)) }   // too late: the window closed
                return answer
            }
            await coordinator.serve(SocketPairingChannel(fd: macFD, timeoutSeconds: 5))
            let result = try await phoneSide.value
            #expect(result.decision["status"] == "denied")
            #expect(result.decision["sealed"] == nil)
            #expect(result.decision["reason"] == reason)
            close(macFD)
        }
    }

    @Test func oneRequestAtATimeAndCancelDropsThePendingOne() async throws {
        let first = pair(), second = pair()
        let firstPhone = phone(first.phone), secondPhone = phone(second.phone)
        let coordinator = LocalPairingCoordinator(invitation: invitation, expiresAt: Date().addingTimeInterval(120)) { _, _ in
            try? await Task.sleep(for: .seconds(30))
            return true
        }
        let serving = Task { await coordinator.serve(SocketPairingChannel(fd: first.mac, timeoutSeconds: 5)) }
        try await Task.sleep(for: .milliseconds(300))
        await coordinator.serve(SocketPairingChannel(fd: second.mac, timeoutSeconds: 5))
        #expect(try await secondPhone.value.decision["status"] == "denied", "a second phone during a pending request")
        await coordinator.cancel()
        await serving.value
        #expect(try await firstPhone.value.decision["status"] == "denied", "Add iPhone cancelled")
        close(first.mac); close(second.mac)
    }
}
