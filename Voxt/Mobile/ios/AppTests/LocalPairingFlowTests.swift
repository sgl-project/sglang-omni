import XCTest
@testable import VoxtMobile
import VoxtMobileKit

/// Find my Mac on this Wi-Fi: show the code the Mac also shows, then pair with the
/// invitation exactly as after a QR scan.
@MainActor
final class LocalPairingFlowTests: XCTestCase {
    private func model() -> (AppModel, UserDefaults, String) {
        let name = "local-pairing.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        return (AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: defaults), defaults, name)
    }

    func testShowsTheCodeThenOpensTheInvitationLikeAQRScan() async {
        let (model, defaults, name) = model()
        defer { defaults.removePersistentDomain(forName: name) }
        let invitation = URL(string: "voxtmobile://connect?v=1&tailcat=tc-fixture")!
        var opened: URL?
        var codeWhileWaiting: String?
        model.openPairingLink = { opened = $0 }
        model.localPairingSearch = { showCode in
            showCode("123456")
            codeWhileWaiting = await MainActor.run { model.localPairingCode }
            return invitation
        }
        await model.findMacOnWiFi()
        XCTAssertEqual(codeWhileWaiting, "123456")
        XCTAssertEqual(opened, invitation)
        XCTAssertNil(model.localPairingCode, "the code is cleared once the Mac answered")
        XCTAssertNil(model.localPairingError)
    }

    func testFailuresPointToTheQR() async {
        let cases: [(Error, String)] = [
            (LocalPairingBrowserFailure.notFound, "No Mac found on this Wi-Fi. Use the QR instead."),
            (LocalPairingClient.Failure.denied, "Your Mac said no. Scan the QR code in Voxt Connector instead."),
            (LocalPairingClient.Failure.timedOut, "No answer from your Mac. Scan the QR code in Voxt Connector instead."),
            (LocalPairingClient.Failure.invalidReply, "Couldn't reach your Mac. Use the QR instead."),
        ]
        for (error, message) in cases {
            let (model, defaults, name) = model()
            var opened = false
            model.openPairingLink = { _ in opened = true }
            model.cameraAccessChanged(.authorized)
            model.localPairingSearch = { _ in throw error }
            await model.findMacOnWiFi()
            XCTAssertEqual(model.localPairingError, message)
            XCTAssertNil(model.localPairingCode)
            XCTAssertFalse(opened)
            XCTAssertEqual(model.pairingScan.state, .scanning, "the scanner on the same screen keeps scanning")
            defaults.removePersistentDomain(forName: name)
        }
    }

    /// The in-app scanner opens the scanned link exactly like a Camera-app scan, and a code
    /// that isn't Voxt's only shows a hint.
    func testInAppScanOpensTheLinkLikeACameraScan() {
        let (model, defaults, name) = model()
        defer { defaults.removePersistentDomain(forName: name) }
        var opened: [URL] = []
        model.openPairingLink = { opened.append($0) }
        model.cameraAccessChanged(.authorized)
        model.scanned("https://example.com/menu")
        XCTAssertEqual(model.pairingScan.hint, "This isn't a Voxt pairing code.")
        XCTAssertTrue(opened.isEmpty)
        let expires = Int(Date().timeIntervalSince1970) + 100
        let link = "voxtmobile://connect?v=1&tailcat=tc-fixture&port=48932&key=privkey:" + String(repeating: "a", count: 64)
            + "&proof=synthetic-proof&expires=\(expires)"
        model.scanned(link)
        XCTAssertEqual(opened, [URL(string: link)!])
        XCTAssertEqual(model.pairingScan.state, .pairing)
        model.scanned(link)
        XCTAssertEqual(opened.count, 1, "one pairing at a time")
    }
}
