import Foundation
import Testing
@testable import VoxtConnectionKit

@Suite struct ConnectorIdentityTests {
    @Test func defaultsAreTheNeutralBundleIdentifiers() {
        #expect(ConnectorIdentity.defaultBundleIdentifier == "com.voxt.VoxtConnector")
        #expect(ConnectorIdentity.voxtBundleIdentifiers(info: [:]) == ["com.voxt.Voxt", "com.voxt.Voxt.omni.dev"])
        #expect(ConnectorIdentity.voxtBundleIdentifiers(info: nil) == ["com.voxt.Voxt", "com.voxt.Voxt.omni.dev"])
    }

    @Test func infoPlistNamesTheVoxtBuildsToTrust() {
        let info: [String: Any] = ["VoxtBundleIdentifiers": "org.example.Voxt org.example.Voxt.dev"]
        #expect(ConnectorIdentity.voxtBundleIdentifiers(info: info) == ["org.example.Voxt", "org.example.Voxt.dev"])
        #expect(ConnectorIdentity.voxtBundleIdentifiers(info: ["VoxtBundleIdentifiers": ["org.example.Voxt"]]) == ["org.example.Voxt"])
    }

    @Test func anOverrideThatIsNotABundleIdentifierIsIgnored() {
        // A quote would end the identifier string inside the code requirement.
        for bad: Any in [#"org.example" or identifier "evil"#, "", "   ", 7, ["ok.id", "bad id\""]] {
            #expect(ConnectorIdentity.voxtBundleIdentifiers(info: ["VoxtBundleIdentifiers": bad]) == ["com.voxt.Voxt", "com.voxt.Voxt.omni.dev"])
        }
    }

    @Test func requirementNamesEachTrustedVoxtBuild() {
        #expect(ConnectorLink.voxtRequirement(team: "ABCDE12345", identifiers: ["org.example.Voxt"])
                == #"anchor apple generic and certificate leaf[subject.OU] = "ABCDE12345" and (identifier "org.example.Voxt")"#)
    }

    @Test func logsAndTestHooksUseTheConnectorIdentifier() {
        #expect(ConnectorTestHooks.notificationName(.showInDockOff).rawValue == "com.voxt.VoxtConnector.test.showInDockOff")
        #expect(!ConnectorIdentity.bundleIdentifier.isEmpty)
    }
}
