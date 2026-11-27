package main

import (
	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/macx"
)

// Tests fake both. With server, overlayWindowBounds asks the window server:
// AppKit's frame trails it while the user drags the window.
var (
	overlayPointerDown  = macx.MouseButtonDown
	overlayWindowBounds = func(w *mygo.Window, server bool) mygo.Rectangle {
		if server {
			if frame, ok := macx.WindowServerFrame(w.NativeHandle()); ok {
				return mygo.Rectangle(frame)
			}
		}
		return w.ContentBounds()
	}
)

// overlayDragged takes where the user moved the overlay window to. OnMove posts
// it and each sync calls it first: a frame set from a stale place moves it back.
func (a *app) overlayDragged() {
	if a.ov.win == nil || !a.ov.shown {
		return
	}
	bounds, moved, dragging := draggedBounds(a.ov.win, a.ov.frame, a.ov.dragging)
	a.ov.dragging = dragging
	if moved {
		a.overlayDraggedTo(bounds)
	}
}

// draggedBounds is where window is, whether frame did not put it there, and
// whether the user is dragging it. The window server checks AppKit's late reports.
func draggedBounds(window *mygo.Window, frame mygo.Rectangle, dragging bool) (mygo.Rectangle, bool, bool) {
	down := overlayPointerDown()
	bounds := overlayWindowBounds(window, false)
	if down || dragging || bounds.X != frame.X || bounds.Y != frame.Y {
		bounds = overlayWindowBounds(window, true)
	}
	moved := bounds.X != frame.X || bounds.Y != frame.Y
	return bounds, moved, down && (dragging || moved)
}
