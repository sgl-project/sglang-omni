// SPDX-License-Identifier: Apache-2.0
import AppKit
import Testing
@testable import OmniTyper

@MainActor
struct WindowDraggingTests {
    @Test func popupHeaderMovesTheWindowWithoutCoveringControlsOrTakingFocus() throws {
        _ = NSApplication.shared
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let model = AppModel(store: AppStore(directory: directory))
        let panel = RecordingPanel(contentRect: NSRect(x: 100, y: 100, width: 460, height: 190),
                                   styleMask: [.nonactivatingPanel, .borderless], backing: .buffered, defer: false)
        panel.isReleasedWhenClosed = false
        defer { model.shutdown(); panel.close(); try? FileManager.default.removeItem(at: directory) }
        model.phase = .recording
        panel.contentView = VoicePanelHostingView(rootView: VoicePanel(model: model, recorder: model.recorder, worker: model.worker))
        panel.contentView?.layoutSubtreeIfNeeded()
        #expect(!panel.canBecomeKey && !panel.canBecomeMain)
        #expect(panel.contentView?.acceptsFirstMouse(for: nil) == true)
        func event(_ type: NSEvent.EventType, at point: NSPoint) throws -> NSEvent {
            try #require(NSEvent.mouseEvent(with: type, location: point, modifierFlags: [], timestamp: 0,
                                            windowNumber: panel.windowNumber, context: nil, eventNumber: 0,
                                            clickCount: 1, pressure: 1))
        }
        for point in [NSPoint(x: 380, y: 153), NSPoint(x: 426, y: 153), NSPoint(x: 230, y: 80)] {
            let intercepted = panel.windowDrag.handle(try event(.leftMouseDown, at: point), in: panel)
            #expect(!intercepted, "Stop, Cancel, and transcript content must not drag the window")
        }
        let origin = panel.frame.origin
        let started = panel.windowDrag.handle(try event(.leftMouseDown, at: NSPoint(x: 100, y: 153)), in: panel)
        let moved = panel.windowDrag.handle(try event(.leftMouseDragged, at: NSPoint(x: 140, y: 173)), in: panel)
        let ended = panel.windowDrag.handle(try event(.leftMouseUp, at: NSPoint(x: 100, y: 153)), in: panel)
        #expect(started && moved && ended)
        #expect(panel.frame.origin == NSPoint(x: origin.x + 40, y: origin.y + 20))
    }
}
