package main

import (
	"bytes"
	"log"
	"os"
	"strings"
	"testing"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/macx"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func TestOverlayDisplayPrefersFocusedWindow(t *testing.T) {
	displayA := mygo.Display{ID: 1, Label: "A", WorkArea: mygo.Rectangle{X: 0, Y: 0, Width: 1440, Height: 900}}
	displayB := mygo.Display{ID: 2, Label: "B", WorkArea: mygo.Rectangle{X: 1440, Y: 0, Width: 1440, Height: 900}}
	oldFrame, oldMatching, oldPointer := overlayFocusedWindowFrame, overlayDisplayMatching, overlayPointerDisplay
	t.Cleanup(func() {
		overlayFocusedWindowFrame, overlayDisplayMatching, overlayPointerDisplay = oldFrame, oldMatching, oldPointer
	})
	overlayPointerDisplay = func() mygo.Display { return displayA }
	overlayDisplayMatching = func(mygo.Rectangle) mygo.Display { return displayB }

	for _, tc := range []struct {
		name       string
		frame      macx.WindowFrame
		focused    bool
		want       mygo.Rectangle
		wantSource string
	}{
		{name: "focus on B with pointer on A", frame: macx.WindowFrame{X: 1500, Y: 100, Width: 1000, Height: 700}, focused: true, want: displayB.WorkArea, wantSource: "focused window"},
		{name: "no focused window falls back to the pointer", want: displayA.WorkArea, wantSource: "pointer"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			overlayFocusedWindowFrame = func() (macx.WindowFrame, bool) { return tc.frame, tc.focused }
			a := &app{}
			a.ov.native = true
			got, source := a.overlayDisplay()
			if got != tc.want {
				t.Errorf("display work area = %v, want %v", got, tc.want)
			}
			if !strings.Contains(source, tc.wantSource) {
				t.Errorf("display description = %q, want source %q", source, tc.wantSource)
			}
		})
	}
}

func TestPillAnchorKeepsFocusedDisplayBesideKeptCard(t *testing.T) {
	displayA := mygo.Rectangle{X: 0, Y: 0, Width: 1440, Height: 900}
	displayB := mygo.Rectangle{X: 1440, Y: 0, Width: 1440, Height: 900}
	for _, tc := range []struct {
		name       string
		cardArea   mygo.Rectangle
		wantBeside bool
	}{
		{name: "card on the focused display", cardArea: displayB, wantBeside: true},
		{name: "kept card on another display", cardArea: displayA, wantBeside: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := &app{}
			a.prefs.Overlay.Position = "bottom"
			a.pill.area = displayB
			a.ov.area = tc.cardArea
			a.ov.shown, a.ov.mode = true, ovAnswer
			a.ov.frame = mygo.Rectangle{X: tc.cardArea.X + 100, Y: 600, Width: 600, Height: 250}

			got := a.pillAnchor(300, 80)
			want := layout.Frame(displayB, a.prefs.Overlay, 300, 80)
			if tc.wantBeside {
				want.Y = a.ov.frame.Y - want.Height
			}
			if got != want {
				t.Fatalf("pill anchor = %v, want %v (card area %v, pill area %v)", got, want, tc.cardArea, displayB)
			}
		})
	}
}

// TestPillClearOfRewriteBoxCard: a Rewrite answer shows in the Rewrite
// box's window, and the overlay window is then not shown. A recording
// started with that card kept puts the pill beside the box, not over it.
func TestPillClearOfRewriteBoxCard(t *testing.T) {
	area := mygo.Rectangle{X: 0, Y: 30, Width: 1440, Height: 799}
	for _, position := range []string{"bottom", "top"} {
		t.Run(position, func(t *testing.T) {
			a := &app{}
			a.prefs.Overlay.Position = position
			a.pill.area = area
			a.box.winShown, a.box.area = true, area
			a.box.frame = mygo.Rectangle{X: 482, Y: 255, Width: 476, Height: 496}

			pill, box := a.pillFrame(556, 112), a.box.frame
			if pill.X < box.X+box.Width && box.X < pill.X+pill.Width && pill.Y < box.Y+box.Height && box.Y < pill.Y+pill.Height {
				t.Fatalf("pill %v covers the Rewrite box %v", pill, box)
			}
			if pill.Y+pill.Height != box.Y && pill.Y != box.Y+box.Height { // above it while there is room, at either position
				t.Fatalf("pill %v is not adjacent to the Rewrite box %v", pill, box)
			}
		})
	}
}

func TestPillDisplayLogs(t *testing.T) {
	var out bytes.Buffer
	log.SetOutput(&out)
	t.Cleanup(func() { log.SetOutput(os.Stderr) })

	a := &app{}
	a.sess.ID, a.sess.Kind, a.sess.State = "s1", store.KindTranscribe, session.Recording
	a.pillOverlaySync(ovAnswer)
	a.sess.State = session.Done
	a.pillOverlaySync(ovAnswer)

	got := out.String()
	if !strings.Contains(got, "pill: show display=") {
		t.Errorf("show log is missing the chosen display: %q", got)
	}
	if !strings.Contains(got, "pill: hide") {
		t.Errorf("hide log is missing: %q", got)
	}
}
