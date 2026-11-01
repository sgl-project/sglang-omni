import Foundation
import Testing
@testable import VoxtMobileKit

/// Keychain item names follow the app's bundle ID, so each signer's build has its own. A build
/// that already has installs can keep its earlier names with `VoxtKeychainServicePrefix`.
@Suite struct KeychainServiceTests {
    @Test func servicesAreNamedAfterTheBundleID() {
        #expect(KeychainTokenStore.service("pairing", info: ["CFBundleIdentifier": "com.example.voxt"]) == "com.example.voxt.pairing")
    }

    @Test func aConfiguredPrefixKeepsEarlierNames() {
        let info: [String: Any] = ["CFBundleIdentifier": "com.example.voxt", "VoxtKeychainServicePrefix": "com.example.legacy"]
        #expect(KeychainTokenStore.service("tailcat-identity", info: info) == "com.example.legacy.tailcat-identity")
    }

    @Test func anUnsetOrUnexpandedPrefixFallsBackToTheBundleID() {
        for prefix in ["", "$(VOXT_KEYCHAIN_SERVICE_PREFIX)"] {
            let info: [String: Any] = ["CFBundleIdentifier": "com.example.voxt", "VoxtKeychainServicePrefix": prefix]
            #expect(KeychainTokenStore.service("pairing", info: info) == "com.example.voxt.pairing")
        }
    }
}
