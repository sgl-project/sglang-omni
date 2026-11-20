package main

// The answer card’s sizes: the feature page’s setting, the corner
// button’s cycle, and the fill size’s work-area clamp.

import (
	"bytes"
	"fmt"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/ink"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestCardSizes: a feature's page sets its own answer card size; a larger
// card is wider, and any card grows with a long answer to the work
// area's height, its ink still from padding to padding. The other features keep theirs
// (#3644 FB41: a long explanation read badly in the small card).
func TestCardSizes(t *testing.T) {
	if hotkey.Transcribe != string(store.KindTranscribe) || hotkey.Translate != string(store.KindTranslate) || hotkey.Rewrite != string(store.KindRewrite) {
		t.Fatal("Features.CardSizes's keys: the pages' feature ids are not store.Kind's")
	}
	a, tt := pageTester(t, "/rewrite")
	if err := tt.ClickControl("Large"); err != nil {
		t.Fatal(err)
	}
	frames(tt)
	if got := a.settings.Get().Features.CardSizes; got[hotkey.Rewrite] != 2 || len(got) != 1 {
		t.Fatalf("card sizes %v", got)
	}
	long := strings.Repeat("A goroutine is a lightweight thread managed by the Go runtime. ", 90)
	a.ov.area = a.overlayArea() // as when the card shows: its size depends on the screen's room
	for _, c := range []struct {
		kind store.Kind
		size layout.CardSize
	}{{store.KindRewrite, layout.CardSizes[2]}, {store.KindTranslate, layout.CardSizes[0]}} {
		a.answer = answerCard{open: true, kind: c.kind, text: long}
		want := layout.CardChrome + a.cardBodyMax(a.cachedCardRef())
		if w, h, _ := a.cardSize(); w != c.size.W || h != want {
			t.Errorf("%s: %d×%d, want %d×%d", c.kind, w, h, c.size.W, want)
		}
	}
	defer func() { a.measureHook = guessText }()
	a.answer = answerCard{open: true, kind: store.KindRewrite, text: long}
	a.overlaySync()
	img := renderOverlay(t, a)
	if _, bad := ink.Check(img, 2, cardInkWant(ink.Want{W: a.ov.w, H: a.ov.h}, a.cachedCardRef())); len(bad) > 0 || a.ov.w != layout.CardSizes[2].W {
		t.Errorf("the large card, %d wide: %v", a.ov.w, bad)
	}
	if dir := os.Getenv("VOXTNEW_INK"); dir != "" {
		var b bytes.Buffer
		png.Encode(&b, img)
		os.WriteFile(filepath.Join(dir, "card-large.png"), b.Bytes(), 0o644)
	}
}

// TestCardResizeOnCard: the card steps through its sizes from the card
// itself (#3644 FB41: a long answer needed a trip to the feature's page
// for a larger card): Small, Medium, Large, Fill and back to Small,
// saved as the feature's size, and the card takes each at once.
func TestCardResizeOnCard(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.answer = answerCard{open: true, kind: store.KindRewrite, text: strings.Repeat("A goroutine is a lightweight thread managed by the Go runtime. ", 30)}
	a.overlaySync()
	tt := hudTester(a.overlayView, layout.CardFillW(ovTestArea, a.prefs.Overlay.Margin)+2*layout.Pad, 900)
	for i, c := range []struct {
		label string
		step  int
	}{{"Make the card larger", 1}, {"Make the card larger", 2}, {"Make the card larger", 3}, {"Make the card smaller", 0}} {
		frames(tt)
		if err := tt.ClickControl(c.label); err != nil {
			t.Fatalf("step %d: %v (controls %q)", i, err, tt.AccessLabels())
		}
		frames(tt)
		got := a.settings.Get().Features.CardSizes
		_, kept := got[hotkey.Rewrite]
		if want := layout.CardSizes[c.step].Width(ovTestArea, a.prefs.Overlay.Margin); got[hotkey.Rewrite] != c.step || kept != (c.step != 0) || a.ov.w != want {
			t.Errorf("step %d: sizes %v, the card %d wide, want %d", i, got, a.ov.w, want)
		}
	}
}

// TestCardFillSize: the fill size takes the width the work area leaves
// for the card, never less than the large card, the card's window
// stays inside the work area, and a rendered fill card passes the ink
// check (its width is no CardSizes literal).
func TestCardFillSize(t *testing.T) {
	saved := ovTestArea
	t.Cleanup(func() { ovTestArea = saved })
	for _, area := range []mygo.Rectangle{
		{X: 0, Y: 25, Width: 2560, Height: 1415},
		{X: 0, Y: 25, Width: 1280, Height: 600},
		{X: 0, Y: 25, Width: 900, Height: 600},
		{X: 0, Y: 25, Width: 520, Height: 600},
	} {
		ovTestArea = area
		a, _, _ := newTestApp(t)
		a.updateSettings(func(s *settings.Settings) { s.Features.CardSizes = map[string]int{hotkey.Rewrite: 3} })
		a.answer = answerCard{open: true, kind: store.KindRewrite, text: "Short."}
		a.overlaySync()
		if want := layout.CardFillW(area, a.prefs.Overlay.Margin); a.ov.w != want {
			t.Errorf("area %+v: the card %d wide, want the fill width %d", area, a.ov.w, want)
		}
		if f := a.ov.frame; f.X < area.X || f.X+f.Width > area.X+area.Width {
			t.Errorf("area %+v: the card's window %+v reaches past it", area, f)
		}
		img := renderOverlay(t, a)
		if _, bad := ink.Check(img, 2, cardInkWant(ink.Want{W: a.ov.w, H: a.ov.h}, a.cachedCardRef())); len(bad) > 0 {
			t.Errorf("area %+v: the fill card's ink: %v", area, bad)
		}
		if dir := os.Getenv("VOXTNEW_INK"); dir != "" {
			var b bytes.Buffer
			png.Encode(&b, img)
			os.WriteFile(filepath.Join(dir, fmt.Sprintf("card-fill-%d.png", area.Width)), b.Bytes(), 0o644)
		}
	}
}

// TestCardFillInk: the fill card passes the same pixel geometry checks as
// the fixed widths, including its text and the resized corner control.
func TestCardFillInk(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.updateSettings(func(s *settings.Settings) { s.Features.CardSizes = map[string]int{hotkey.Rewrite: 3} })
	a.answer = answerCard{open: true, kind: store.KindRewrite, text: strings.Repeat("A goroutine is a lightweight thread. ", 30)}
	a.overlaySync()
	img := renderOverlay(t, a)
	if _, bad := ink.Check(img, 2, cardInkWant(ink.Want{W: a.ov.w, H: a.ov.h}, a.cachedCardRef())); len(bad) > 0 {
		t.Errorf("fill card ink: %v", bad)
	}
}

// TestCardMessageLeavesRoomForResizeButton: a long status message and the
// resize control never cover each other.
func TestCardMessageLeavesRoomForResizeButton(t *testing.T) {
	a, _, _ := newTestApp(t)
	message := "No text field is focused. Click where the text should go, then insert."
	a.answer = answerCard{open: true, kind: store.KindRewrite, text: "Short.", msg: message}
	a.overlaySync()
	tt := hudTester(a.overlayView, a.ov.w+2*layout.Pad, a.ov.h+2*layout.Pad)
	frames(tt)
	messageBounds, ok := tt.Find(message)
	if !ok {
		t.Fatalf("no status message in %q", firstTexts(tt))
	}
	button, ok := tt.Find(i18n.T("Make the card larger"))
	if !ok {
		t.Fatal("no resize control")
	}
	if messageBounds.X+messageBounds.W > button.X && messageBounds.Y+messageBounds.H > button.Y {
		t.Errorf("status message %+v overlaps resize control %+v", messageBounds, button)
	}
}

// TestCardFillFollowsDraggedDisplay: dragging Fill to another display
// recomputes its width from that display's work area.
func TestCardFillFollowsDraggedDisplay(t *testing.T) {
	saved := ovTestArea
	t.Cleanup(func() { ovTestArea = saved })
	ovTestArea = mygo.Rectangle{X: 0, Y: 25, Width: 2560, Height: 1415}
	a, _, _ := newTestApp(t)
	a.updateSettings(func(s *settings.Settings) { s.Features.CardSizes = map[string]int{hotkey.Rewrite: 3} })
	a.answer = answerCard{open: true, kind: store.KindRewrite, text: "Short."}
	a.overlaySync()
	if a.ov.w != 2332 {
		t.Fatalf("initial fill width %d, want 2332", a.ov.w)
	}
	ovTestArea = mygo.Rectangle{X: 2560, Y: 25, Width: 1440, Height: 875}
	bounds := mygo.Rectangle{X: 2700, Y: 100, Width: a.ov.w + 2*layout.Pad, Height: a.ov.h + 2*layout.Pad}
	a.overlayDraggedTo(bounds)
	if want := layout.CardFillW(ovTestArea, a.prefs.Overlay.Margin); a.ov.w != want {
		t.Errorf("after dragging to the narrower display: width %d, want %d", a.ov.w, want)
	}
	if frame := a.ov.frame; frame.X < ovTestArea.X || frame.X+frame.Width > ovTestArea.X+ovTestArea.Width {
		t.Errorf("after dragging: window %+v leaves work area %+v", frame, ovTestArea)
	}
	a.overlayRestyle()
	if a.ov.frame.X != bounds.X || a.ov.w != 1212 {
		t.Errorf("after another restyle: frame %+v, want dragged x %d and width 1212", a.ov.frame, bounds.X)
	}
}

// TestCardFitsTheScreen: a large card with a long answer on a short
// screen keeps its text to the room the work area leaves, instead of
// reaching past the screen's other edge.
func TestCardFitsTheScreen(t *testing.T) {
	saved := ovTestArea
	t.Cleanup(func() { ovTestArea = saved })
	ovTestArea = mygo.Rectangle{X: 0, Y: 25, Width: 1280, Height: 600}
	a, _, _ := newTestApp(t)
	a.updateSettings(func(s *settings.Settings) { s.Features.CardSizes = map[string]int{hotkey.Rewrite: 2} })
	a.answer = answerCard{open: true, kind: store.KindRewrite, text: strings.Repeat("A goroutine is a lightweight thread managed by the Go runtime. ", 90)}
	a.overlaySync()
	if f := a.ov.frame; f.Y < ovTestArea.Y || f.Y+f.Height > ovTestArea.Y+ovTestArea.Height {
		t.Errorf("the card's window %+v is not inside the work area %+v", f, ovTestArea)
	}
}
