import Foundation
import Testing
@testable import VoxtMobileKit

/// App Group IDs are unique across Apple teams, so a build signed by anyone else uses its
/// own group (VOXT_MOBILE_ID_PREFIX). The app and the keyboard read it from Info.plist.
@Suite struct AppGroupTests {
    @Test func readsTheGroupTheBuildWroteIntoInfoPlist() {
        #expect(DictationChannel.appGroup(from: ["VoxtAppGroup": "group.com.example.voxt.shared"]) == "group.com.example.voxt.shared")
    }

    @Test func fallsBackToAltStoresRenamedGroupThenTheNeutralDefault() {
        #expect(DictationChannel.appGroup(from: ["ALTAppGroups": ["group.com.voxt.VoxtMobile.shared.ABCDE12345"]])
                == "group.com.voxt.VoxtMobile.shared.ABCDE12345")
        #expect(DictationChannel.appGroup(from: nil) == "group.com.voxt.VoxtMobile.shared")
        #expect(DictationChannel.appGroup(from: ["VoxtAppGroup": ""]) == "group.com.voxt.VoxtMobile.shared")
        #expect(DictationChannel.appGroup(from: ["VoxtAppGroup": "$(VOXT_APP_GROUP)"]) == "group.com.voxt.VoxtMobile.shared",
                "an unexpanded build setting is not a group")
    }

    /// Darwin notification names are system-wide. Naming them after the App Group keeps two
    /// builds signed by different teams on one phone from waking each other.
    @Test func notificationNamesComeFromTheAppGroup() {
        #expect(DictationChannel.notification("command", appGroup: "group.com.example.voxt.shared")
                == "group.com.example.voxt.shared.command")
        #expect(DictationChannel.commandNotification == DictationChannel.appGroup + ".command")
        #expect(DictationChannel.stateNotification == DictationChannel.appGroup + ".state")
    }
}
