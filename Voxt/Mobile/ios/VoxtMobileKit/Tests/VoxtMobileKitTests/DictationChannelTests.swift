import Foundation
import Testing
@testable import VoxtMobileKit

@Suite struct DictationChannelTests {
    private func makeChannel() -> (DictationChannel, UserDefaults) {
        let name = "test.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        return (DictationChannel(defaults: defaults, notifier: .none), defaults)
    }

    @Test func freshChannelIsIdleAndSessionDead() {
        let (channel, _) = makeChannel()
        #expect(channel.state.phase == .idle)
        #expect(channel.isSessionAlive(now: Date()) == false)
    }

    @Test func heartbeatKeepsSessionAliveForThreeSeconds() {
        let (channel, _) = makeChannel()
        let t0 = Date(timeIntervalSince1970: 1_000)
        channel.heartbeat(now: t0)
        #expect(channel.isSessionAlive(now: t0.addingTimeInterval(2.9)))
        #expect(channel.isSessionAlive(now: t0.addingTimeInterval(3.1)) == false)
    }

    /// When the app stops running after a handoff, a keyboard that trusts the stored recording
    /// phase shows "Tap again to finish" forever.
    @Test func recordingOrProcessingWithoutAHeartbeatIsStalledAfterTheGracePeriod() {
        let (channel, _) = makeChannel()
        let t0 = Date(timeIntervalSince1970: 1_000)
        for phase in [DictationChannel.Phase.recording, .processing] {
            channel.publish(.init(phase: phase, requestID: "r"))
            channel.heartbeat(now: t0)
            #expect(channel.isStalled(now: t0.addingTimeInterval(9)) == false)
            #expect(channel.isStalled(now: t0.addingTimeInterval(DictationChannel.stalledAfter + 0.1)))
            channel.endSession()
            #expect(channel.isStalled(now: t0), "a removed heartbeat with an active phase is stalled at once")
        }
        for phase in [DictationChannel.Phase.idle, .done, .failed] {
            channel.publish(.init(phase: phase))
            #expect(channel.isStalled(now: t0.addingTimeInterval(60)) == false)
        }
    }

    @Test func endSessionMarksDeadImmediately() {
        let (channel, _) = makeChannel()
        let t0 = Date(timeIntervalSince1970: 1_000)
        channel.heartbeat(now: t0)
        channel.endSession()
        #expect(channel.isSessionAlive(now: t0) == false)
    }

    @Test func commandsAreSequencedAndReadByOtherSide() {
        let (keyboard, defaults) = makeChannel()
        let app = DictationChannel(defaults: defaults, notifier: .none)

        keyboard.send(.start)
        let first = app.pendingCommand(after: 0)
        #expect(first?.kind == .start)

        keyboard.send(.stop)
        let second = app.pendingCommand(after: first!.sequence)
        #expect(second?.kind == .stop)
        #expect(second!.sequence > first!.sequence)
        #expect(app.pendingCommand(after: second!.sequence) == nil)
    }

    @Test func commandCarriesCompositionAndOriginAcrossInstances() {
        let (keyboard, defaults) = makeChannel()
        let app = DictationChannel(defaults: defaults, notifier: .none)
        let options = CompositionOptions(mode: .edit, selectedText: "Original")
        keyboard.send(.start, options: options, requestID: "origin")
        #expect(app.pendingCommand(after: 0)?.options == options)
        #expect(app.pendingCommand(after: 0)?.requestID == "origin")
    }

    @Test func oldPersistedStateStillDecodes() {
        let (channel, defaults) = makeChannel()
        defaults.set(Data(#"{"phase":"done","text":"old text","resultID":"old-id"}"#.utf8), forKey: "dictation.state")
        #expect(channel.state.text == "old text")
        #expect(channel.state.requestID == nil)
    }

    @Test func finishedResultIsInsertedExactlyOnce() {
        let (app, defaults) = makeChannel()
        let keyboard = DictationChannel(defaults: defaults, notifier: .none)

        app.publish(.init(phase: .done, text: "Hello world.", resultID: "r1"))

        let insertion = keyboard.textToInsert(lastInsertedID: nil)
        #expect(insertion?.text == "Hello world.")
        #expect(insertion?.resultID == "r1")
        #expect(keyboard.textToInsert(lastInsertedID: "r1") == nil)
    }

    @Test func nonDonePhasesInsertNothing() {
        let (app, defaults) = makeChannel()
        let keyboard = DictationChannel(defaults: defaults, notifier: .none)
        app.publish(.init(phase: .processing))
        #expect(keyboard.textToInsert(lastInsertedID: nil) == nil)
        app.publish(.init(phase: .failed, errorMessage: "Mac unreachable"))
        #expect(keyboard.textToInsert(lastInsertedID: nil) == nil)
        #expect(keyboard.state.errorMessage == "Mac unreachable")
    }
}
