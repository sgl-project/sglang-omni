// SPDX-License-Identifier: Apache-2.0
import AppKit
import SwiftUI

final class WindowDrag {
    let regions = NSHashTable<NSView>.weakObjects()
    private var start: (pointer: NSPoint, origin: NSPoint)?

    func handle(_ event: NSEvent, in window: NSWindow) -> Bool {
        if event.type == .leftMouseDown {
            start = nil
            guard regions.allObjects.contains(where: { view in
                view.window === window && !view.isHiddenOrHasHiddenAncestor
                    && view.bounds.intersection(view.visibleRect).contains(view.convert(event.locationInWindow, from: nil))
            }) else { return false }
            start = (window.convertPoint(toScreen: event.locationInWindow), window.frame.origin)
            return true
        }
        guard let start else { return false }
        if event.type == .leftMouseDragged {
            let pointer = window.convertPoint(toScreen: event.locationInWindow)
            window.setFrameOrigin(NSPoint(x: start.origin.x + pointer.x - start.pointer.x,
                                          y: start.origin.y + pointer.y - start.pointer.y))
            return true
        }
        if event.type == .leftMouseUp { self.start = nil; return true }
        return false
    }
}

struct WindowDragArea: NSViewRepresentable {
    func makeNSView(context: Context) -> DragView { DragView() }
    func updateNSView(_ view: DragView, context: Context) {}

    final class DragView: NSView {
        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            (window as? RecordingPanel)?.windowDrag.regions.add(self)
        }
        override func hitTest(_ point: NSPoint) -> NSView? { nil }
    }
}
