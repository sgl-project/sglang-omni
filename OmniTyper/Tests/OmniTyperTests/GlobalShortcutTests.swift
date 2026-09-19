// SPDX-License-Identifier: Apache-2.0
import CoreGraphics
import Carbon
import Foundation
import Testing
@testable import OmniTyper

struct GlobalShortcutTests {
    @Test @MainActor func conflictsAreRejectedAndTheProbeReleasesItsRegistration() throws {
        #expect(throws: (any Error).self) { try GlobalShortcut.validate(keyCode: 0, modifiers: 0) }
        var shortcuts: Unmanaged<CFArray>?
        try #require(CopySymbolicHotKeys(&shortcuts) == noErr)
        let enabled = try #require((shortcuts?.takeRetainedValue() as? [[String: Any]])?.first {
            ($0[kHISymbolicHotKeyEnabled] as? Bool) == true
                && ($0[kHISymbolicHotKeyCode] as? Int) == 49
        })
        let modifiers = try #require(enabled[kHISymbolicHotKeyModifiers] as? UInt32)
        var flags: CGEventFlags = []
        if modifiers & UInt32(cmdKey) != 0 { flags.insert(.maskCommand) }
        if modifiers & UInt32(optionKey) != 0 { flags.insert(.maskAlternate) }
        if modifiers & UInt32(controlKey) != 0 { flags.insert(.maskControl) }
        if modifiers & UInt32(shiftKey) != 0 { flags.insert(.maskShift) }
        #expect(throws: (any Error).self) { try GlobalShortcut.validate(keyCode: 49, modifiers: flags.rawValue) }

        let key: UInt16 = 20 // Control + Option + Shift + Command + 3
        let combo: CGEventFlags = [.maskControl, .maskAlternate, .maskShift, .maskCommand]
        let carbon = UInt32(controlKey | optionKey | shiftKey | cmdKey)
        var occupied: EventHotKeyRef?
        defer { if let occupied { UnregisterEventHotKey(occupied) } }
        try #require(RegisterEventHotKey(UInt32(key), carbon, EventHotKeyID(signature: 0x4F4D5453, id: 1),
                                        GetApplicationEventTarget(), OptionBits(kEventHotKeyExclusive), &occupied) == noErr)
        #expect(throws: (any Error).self) { try GlobalShortcut.validate(keyCode: key, modifiers: combo.rawValue) }
        UnregisterEventHotKey(occupied); occupied = nil
        try GlobalShortcut.validate(keyCode: key, modifiers: combo.rawValue)
        try GlobalShortcut.validate(keyCode: key, modifiers: combo.rawValue)
    }

    @Test @MainActor func shortcutCaptureSuppressesQueuedActivationAndPreferenceChanges() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = AppStore(directory: directory)
        let model = AppModel(store: store)
        defer { model.shutdown() }
        let down = try #require(CGEvent(keyboardEventSource: nil, virtualKey: 49, keyDown: true))
        down.flags = [.maskControl, .maskAlternate]
        _ = model.shortcut.receive(.keyDown, event: down)
        model.beginShortcutCapture()
        store.preferences.sounds.toggle()
        _ = model.shortcut.receive(.keyDown, event: down)
        await withCheckedContinuation { continuation in DispatchQueue.main.async { continuation.resume() } }
        #expect(!model.isBusy)
        #expect(model.error.isEmpty)
        model.endShortcutCapture()
        model.beginShortcutCapture()
        model.shutdown()
        model.endShortcutCapture()
        _ = model.shortcut.receive(.keyDown, event: down)
        await withCheckedContinuation { continuation in DispatchQueue.main.async { continuation.resume() } }
        #expect(!model.isBusy)
        #expect(model.error.isEmpty)
    }

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
