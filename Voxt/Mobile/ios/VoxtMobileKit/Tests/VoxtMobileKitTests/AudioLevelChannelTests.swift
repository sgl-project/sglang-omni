import Foundation
import Testing
@testable import VoxtMobileKit

struct AudioLevelChannelTests {
    @Test func levelTravelsBetweenAppAndKeyboardWithoutReplacingResult() {
        let name = "level-test.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let app = DictationChannel(defaults: defaults, notifier: .none)
        let keyboard = DictationChannel(defaults: UserDefaults(suiteName: name)!, notifier: .none)
        app.publish(.init(phase: .recording, requestID: "synthetic"))
        app.publishAudioLevel(0.25)
        #expect(keyboard.audioLevel == 0.25)
        #expect(keyboard.state.requestID == "synthetic")
        app.publishAudioLevel(0)
        #expect(keyboard.audioLevel == 0)
    }
}
