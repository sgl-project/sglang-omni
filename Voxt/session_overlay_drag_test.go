package main

import (
	"strings"
	"testing"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

// TestPreviewDraggedWhileItGrows (#4701): the live preview, dragged while new
// words come in, grows where the user has it. A frame worked out from AppKit's
// frame, which trails a dragged window, sent it back toward where the drag began.
func TestPreviewDraggedWhileItGrows(t *testing.T) {
	a, _, _ := newTestApp(t)
	snap(a, session.Snapshot{State: session.Recording, Partial: "hello"})
	start := a.ov.frame

	appKit, server, down := start, start, true // AppKit still has the window where the drag started
	previousBounds, previousDown := overlayWindowBounds, overlayPointerDown
	overlayWindowBounds = func(_ *mygo.Window, fromServer bool) mygo.Rectangle {
		if fromServer {
			return server
		}
		return appKit
	}
	overlayPointerDown = func() bool { return down }
	t.Cleanup(func() { overlayWindowBounds, overlayPointerDown = previousBounds, previousDown })
	a.ov.win = new(mygo.Window)          // stands in for the window, which a test does not draw
	t.Cleanup(func() { a.ov.win = nil }) // no main loop to destroy it on

	words := "more words coming in "
	speak := func(lines int) mygo.Rectangle {
		snap(a, session.Snapshot{State: session.Recording, Partial: strings.Repeat(words, 4*lines)})
		return a.ov.frame
	}

	server.X, server.Y = start.X-240, start.Y-160
	a.measureHook = nil // the new words are not measured yet, as on the Mac
	t.Cleanup(func() { a.measureHook = guessText })
	if guessed := speak(3); guessed != server {
		t.Errorf("unmeasured words moved the dragged preview to %+v from %+v", guessed, server)
	}
	a.measureHook = guessText // the view measured them
	a.overlaySync()
	grown := a.ov.frame
	if grown.Height <= start.Height {
		t.Fatalf("the preview did not grow: %+v, started at %+v", grown, start)
	}
	if centre, want := grown.X+grown.Width/2, server.X+server.Width/2; centre < want-1 || centre > want+1 {
		t.Errorf("grown with its centre at %d, want %d where it was dragged; started at %+v", centre, want, start)
	}
	if bottom, want := grown.Y+grown.Height, server.Y+server.Height; bottom != want {
		t.Errorf("grown with its bottom at %d, want %d where it was dragged; started at %+v", bottom, want, start)
	}

	server = grown
	server.X = ovTestArea.X - 200 // dragged over the screen's left edge, the button still down
	if held := speak(5); held.X != server.X {
		t.Errorf("pulled back to x %d while dragged over the edge at %d", held.X, server.X)
	}

	server, appKit, down = a.ov.frame, a.ov.frame, false // let go there
	if dropped := speak(7); dropped.X != ovTestArea.X-layout.Pad {
		t.Errorf("let go over the edge: x %d, want %d back on the screen", dropped.X, ovTestArea.X-layout.Pad)
	}
}
