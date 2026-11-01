import AppKit
import Foundation
import Testing
@testable import VoxtConnectionKit

/// Records what the dock asks AppKit to do, in order, and holds delayed work until the test runs it.
@MainActor private final class Recorder {
    var policies: [NSApplication.ActivationPolicy] = []
    var events: [String] = []
    var delayed: [(Duration, @MainActor () -> Void)] = []
    var resignWaiters: [@MainActor () -> Void] = []
    func runDelayed() { let work = delayed; delayed = []; work.forEach { $0.1() } }
    func resign() { let work = resignWaiters; resignWaiters = []; work.forEach { $0() } }
}

@MainActor private func dock(_ preferences: TestPreferences = TestPreferences(),
                             _ recorder: Recorder = Recorder()) -> ConnectorDock {
    let dock = ConnectorDock(preferences: preferences, setPolicy: { recorder.policies.append($0) },
                             activate: { recorder.events.append("activate") },
                             after: { delay, work in recorder.delayed.append((delay, work)) },
                             onNextResign: { recorder.resignWaiters.append($0) })
    dock.openSettingsWindow = { recorder.events.append("open") }
    return dock
}

@Test func showInDockDefaultsToOnAndIsRemembered() throws {
    let suite = "connector-dock-\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    #expect(UserDefaultsConnectorPreferences(defaults: defaults).showsInDock)
    UserDefaultsConnectorPreferences(defaults: defaults).showsInDock = false
    #expect(!UserDefaultsConnectorPreferences(defaults: defaults).showsInDock)
}

@Test func dockIconMapsToRegularAndHiddenToMenuBarOnly() {
    #expect(ConnectorDock.activationPolicy(showsInDock: true) == .regular)
    #expect(ConnectorDock.activationPolicy(showsInDock: false) == .accessory)
}

@MainActor @Test func launchAppliesTheSavedChoice() {
    let shown = Recorder()
    dock(TestPreferences(), shown).applyAtLaunch()
    #expect(shown.policies == [.regular])

    let hiddenPreferences = TestPreferences(); hiddenPreferences.showsInDock = false
    let hidden = Recorder()
    dock(hiddenPreferences, hidden).applyAtLaunch()
    #expect(hidden.policies == [.accessory]); #expect(hidden.events.isEmpty); #expect(hidden.delayed.isEmpty)
}

@MainActor @Test func toggleSwitchesThePolicyAtOnceAndSavesIt() {
    let preferences = TestPreferences(), recorder = Recorder()
    let dock = dock(preferences, recorder)
    #expect(dock.showsInDock)
    dock.showsInDock = false
    #expect(recorder.policies == [.accessory]); #expect(!preferences.showsInDock)
    dock.showsInDock = true
    #expect(recorder.policies == [.accessory, .regular]); #expect(preferences.showsInDock)
}

/// A menu click or Dock click must make the Connector the active app before the window opens;
/// opened first, the window stays behind the active app.
@MainActor @Test func showingSettingsActivatesBeforeOpeningTheWindow() {
    let recorder = Recorder()
    let dock = dock(TestPreferences(), recorder)
    dock.showSettings()
    #expect(recorder.events == ["activate", "open"])
}

@MainActor @Test func reopeningShowsSettingsInsteadOfTheDefault() {
    let recorder = Recorder()
    let dock = dock(TestPreferences(), recorder)
    #expect(dock.reopen(hasVisibleWindows: false) == false)
    #expect(dock.reopen(hasVisibleWindows: true) == false)
    #expect(recorder.events == ["activate", "open", "activate", "open"])
}

/// Leaving the Dock hands activation to the previous app, whose windows cover Settings within
/// 50 ms. Re-activating on the next main-loop turn loses to that app, and a fixed 150 ms timer
/// brings Settings back only at about 640 ms. So Settings comes back as soon as the Connector has actually lost activation, with a
/// timer only in case that never happens.
@MainActor @Test func hidingTheIconShowsSettingsAgainOnceActivationIsLost() {
    let recorder = Recorder()
    let dock = dock(TestPreferences(), recorder)
    dock.showsInDock = false
    #expect(recorder.events.isEmpty)
    #expect(recorder.resignWaiters.count == 1)
    recorder.resign()
    #expect(recorder.events == ["activate", "open"])
    recorder.runDelayed() // the fallback timer finds the work done
    #expect(recorder.events == ["activate", "open"])

    recorder.events = []
    dock.showsInDock = true
    #expect(recorder.delayed.isEmpty); #expect(recorder.resignWaiters.isEmpty); #expect(recorder.events.isEmpty)
}

@MainActor @Test func ifActivationIsNeverLostTheTimerShowsSettings() {
    let recorder = Recorder()
    let dock = dock(TestPreferences(), recorder)
    dock.showsInDock = false
    #expect(recorder.delayed.map(\.0) == [ConnectorDock.reactivationFallback])
    recorder.runDelayed()
    #expect(recorder.events == ["activate", "open"])
    recorder.resign() // a late resign doesn't show it twice
    #expect(recorder.events == ["activate", "open"])
}
