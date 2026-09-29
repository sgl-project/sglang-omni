// SPDX-License-Identifier: Apache-2.0
import CoreGraphics
import Foundation
import Testing
@testable import OmniTyper

struct GlobalShortcutTests {
    @Test @MainActor func consumedShortcutStaysHeldUntilPhysicalRelease() async throws {
        for keyCode: UInt16 in [49, 63] {
            for releaseByPolling in [false, true] {
                let shortcut = GlobalShortcut()
                defer { shortcut.stop() }
                var callbacks: [String] = []
                let flags: CGEventFlags = keyCode == 63 ? .maskSecondaryFn : [.maskControl, .maskAlternate]
                shortcut.start(keyCode: keyCode, modifiers: keyCode == 63 ? 0 : flags.rawValue, hold: true,
                               onStart: { callbacks.append("start") }, onStop: { callbacks.append("stop") },
                               onCancel: { callbacks.append("cancel") })
                let down = try #require(CGEvent(keyboardEventSource: nil, virtualKey: keyCode, keyDown: true))
                down.flags = flags
                #expect(shortcut.receive(keyCode == 63 ? .flagsChanged : .keyDown, event: down))

                // Note (Codex): Preference updates must not release an unchanged held shortcut.
                shortcut.start(keyCode: keyCode, modifiers: keyCode == 63 ? 0 : flags.rawValue, hold: true,
                               onStart: { callbacks.append("start") }, onStop: { callbacks.append("stop") },
                               onCancel: { callbacks.append("cancel") })
                #expect(shortcut.pressed)

                // Note (Codex): The consumed event is absent from session state while the physical key remains held.
                for _ in 0..<20 {
                    shortcut.pollForRelease(keyState: { state, key in state == .hidSystemState && key == keyCode },
                                            flagsState: { state in state == .hidSystemState ? flags : [] })
                }
                #expect(shortcut.pressed)
                down.setIntegerValueField(.keyboardEventAutorepeat, value: 1)
                #expect(shortcut.receive(keyCode == 63 ? .flagsChanged : .keyDown, event: down))

                if releaseByPolling {
                    shortcut.pollForRelease(keyState: { _, _ in false }, flagsState: { _ in [] })
                    #expect(!shortcut.pressed)
                }
                let up = try #require(CGEvent(keyboardEventSource: nil, virtualKey: keyCode, keyDown: false))
                up.flags = keyCode == 63 ? [] : flags
                #expect(shortcut.receive(keyCode == 63 ? .flagsChanged : .keyUp, event: up))
                #expect(!shortcut.pressed)
                await withCheckedContinuation { continuation in
                    DispatchQueue.main.async { continuation.resume() }
                }
                #expect(callbacks == ["start", "stop"])
            }
        }
    }
}
