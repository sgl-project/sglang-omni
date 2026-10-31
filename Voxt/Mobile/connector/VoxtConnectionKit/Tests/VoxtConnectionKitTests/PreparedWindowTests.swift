import AppKit
import Testing
@testable import VoxtConnectionKit

/// Creating the Settings window inside SwiftUI's openWindow takes about 1 s on the click path,
/// so it is made once ahead of time; showing only orders it front.
@MainActor @Test func aPreparedWindowIsShownWithoutBeingMadeAgain() {
    var made = 0, shown: [NSWindow] = []
    let prepared = PreparedWindow(make: {
        made += 1
        return NSWindow(contentRect: .init(x: 0, y: 0, width: 420, height: 300), styleMask: [.titled], backing: .buffered, defer: true)
    }, present: { shown.append($0) })
    prepared.prepare()
    #expect(made == 1); #expect(shown.isEmpty)
    prepared.show(); prepared.show()
    #expect(made == 1); #expect(shown.count == 2); #expect(shown.first === prepared.window)
}

@MainActor @Test func showingBeforePreparingMakesTheWindowOnce() {
    var made = 0, shown = 0
    let prepared = PreparedWindow(make: {
        made += 1
        return NSWindow(contentRect: .init(x: 0, y: 0, width: 420, height: 300), styleMask: [.titled], backing: .buffered, defer: true)
    }, present: { _ in shown += 1 })
    prepared.show(); prepared.prepare()
    #expect(made == 1); #expect(shown == 1)
}

/// SwiftUI draws a hosted view only once its window has been on screen; the first show then takes
/// about 440 ms. Preparing puts it on screen once, invisibly.
@MainActor @Test func preparingWarmsTheWindowUpOnceAndShowingDoesNot() {
    var warmed = 0
    let prepared = PreparedWindow(make: {
        NSWindow(contentRect: .init(x: 0, y: 0, width: 420, height: 300), styleMask: [.titled], backing: .buffered, defer: true)
    }, present: { _ in }, warmUp: { _ in warmed += 1 })
    prepared.prepare(); prepared.prepare()
    #expect(warmed == 1)
    prepared.show()
    #expect(warmed == 1)
}

/// The default warm-up leaves the window hidden and fully opaque for the real show.
@MainActor @Test func theDefaultWarmUpLeavesTheWindowHiddenAndOpaque() {
    let window = NSWindow(contentRect: .init(x: -10_000, y: -10_000, width: 420, height: 300), styleMask: [.titled], backing: .buffered, defer: false)
    PreparedWindow.warmUp(window)
    #expect(!window.isVisible)
    #expect(window.alphaValue == 1)
}
