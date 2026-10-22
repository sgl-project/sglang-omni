import Foundation
import Testing
@testable import VoxtMobileKit

/// In-app QR scan on the pairing screen: the decoded text goes through the same link parser
/// as a Camera-app scan, and the screen moves scanning → pairing → paired.
@Suite struct PairingScanTests {
    private let now = Date(timeIntervalSince1970: 1_800_000_000)
    private let link = "voxtmobile://connect?v=1&tailcat=tc-fixture&port=48932&key=privkey:" + String(repeating: "a", count: 64)
        + "&proof=synthetic-proof&expires=1800000120"

    @Test func validCodePairsAndIgnoresFurtherScans() {
        var scan = PairingScan()
        scan.camera(.authorized)
        #expect(scan.state == .scanning)
        #expect(scan.decoded(link, now: now) == URL(string: link))
        #expect(scan.state == .pairing)
        #expect(scan.decoded(link, now: now) == nil, "one pairing at a time")
        scan.pairingFinished(error: nil)
        #expect(scan.state == .paired)
    }

    @Test func otherCodesShowAHintAndKeepScanning() {
        var scan = PairingScan()
        scan.camera(.authorized)
        #expect(scan.decoded("https://example.com/menu", now: now) == nil)
        #expect(scan.state == .scanning)
        #expect(scan.hint == "This isn't a Voxt pairing code.")
        #expect(scan.decoded("voxtmobile://connect?v=1", now: now) == nil)
        #expect(scan.hint == "This isn't a Voxt pairing code.")
        #expect(scan.decoded(link, now: now.addingTimeInterval(600)) == nil)
        #expect(scan.hint == "This code has expired. Show a new one in Voxt Connector.")
        #expect(scan.decoded(link, now: now) != nil)
        #expect(scan.hint == nil)
    }

    @Test func failedPairingReturnsToScanningWithTheMessage() {
        var scan = PairingScan()
        scan.camera(.authorized)
        _ = scan.decoded(link, now: now)
        scan.pairingFinished(error: "Couldn't pair. Scan a new QR in Voxt Connector.")
        #expect(scan.state == .scanning)
        #expect(scan.hint == "Couldn't pair. Scan a new QR in Voxt Connector.")
    }

    @Test func deniedCameraStopsScanning() {
        var scan = PairingScan()
        scan.camera(.notDetermined)
        #expect(scan.state == .starting)
        scan.camera(.denied)
        #expect(scan.state == .cameraDenied)
        #expect(scan.decoded(link, now: now) == nil)
        scan.camera(.authorized)
        #expect(scan.state == .scanning, "access granted later in Settings")
    }

    @Test func expiredMacWindowIsNoAnswerNotARefusal() async throws {
        let reply = try JSONSerialization.jsonObject(with: LocalPairingWire.denied(reason: .expired)) as? [String: Any]
        #expect(reply?["status"] as? String == "denied")
        #expect(reply?["reason"] as? String == "expired")
        #expect(LocalPairingClient.failure(forDenial: LocalPairingWire.denied(reason: .expired)) == .timedOut)
        #expect(LocalPairingClient.failure(forDenial: LocalPairingWire.denied()) == .denied)
    }
}
