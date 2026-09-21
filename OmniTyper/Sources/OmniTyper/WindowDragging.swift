// SPDX-License-Identifier: Apache-2.0
import AppKit
import SwiftUI

final class WindowDrag {
    var enabled = true
    let regions = NSHashTable<NSView>.weakObjects()
    private var start: (pointer: NSPoint, origin: NSPoint)?

    func handle(_ event: NSEvent, in window: NSWindow) -> Bool {
        if event.type == .leftMouseDown {
            start = nil
            guard enabled, regions.allObjects.contains(where: { view in
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
    var enabled = true
    func makeNSView(context: Context) -> DragView { DragView() }
    func updateNSView(_ view: DragView, context: Context) { view.isHidden = !enabled }

    final class DragView: NSView {
        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            (window as? RecordingPanel)?.windowDrag.regions.add(self)
        }
        override func hitTest(_ point: NSPoint) -> NSView? { nil }
    }
}

@MainActor
final class PanelSelectionDrag {
    weak var model: AppModel?
    weak var region: NSView?
    private var selecting = false

    func handle(_ event: NSEvent, in window: NSWindow) -> Bool {
        let pointer: NSPoint
        if let location = event.cgEvent?.location, let primaryScreen = NSScreen.screens.first {
            // Note (Codex): Queued mouse events retain coordinates from before the selector resized its window.
            pointer = NSPoint(x: location.x, y: primaryScreen.frame.maxY - location.y)
        } else { pointer = window.convertPoint(toScreen: event.locationInWindow) }
        if event.type == .leftMouseDown {
            selecting = false
            guard let model, !model.isSelectingMode, model.phase != .processing,
                  let region, region.window === window, !region.isHiddenOrHasHiddenAncestor,
                  region.bounds.intersection(region.visibleRect).contains(region.convert(window.convertPoint(fromScreen: pointer), from: nil)) else { return false }
            selecting = true
            model.beginPanelSelection(at: pointer)
            return true
        }
        guard selecting, let model else { return false }
        if event.type == .leftMouseDragged {
            model.updateModeSelection(pointerX: pointer.x, pointerY: pointer.y)
            return true
        }
        if event.type == .leftMouseUp {
            selecting = false
            model.endPanelSelection(at: pointer)
            return true
        }
        return false
    }
}

struct PanelSelectionArea: NSViewRepresentable {
    let model: AppModel
    func makeNSView(context: Context) -> SelectionView { SelectionView(model: model) }
    func updateNSView(_ view: SelectionView, context: Context) {}

    final class SelectionView: NSView {
        let model: AppModel
        init(model: AppModel) { self.model = model; super.init(frame: .zero) }
        required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }
        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            guard let panel = window as? RecordingPanel else { return }
            panel.selectionDrag.model = model
            panel.selectionDrag.region = self
        }
        override func hitTest(_ point: NSPoint) -> NSView? { nil }
    }
}
