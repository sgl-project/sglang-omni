package main

import (
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestLiveTextSwitch: Settings → Overlay style → Show live text off (Voxt
// 1.x's realtimeTextDisplayEnabled) leaves the words out of the pill, not
// a note the pill must show; another overlay setting changed keeps it.
func TestLiveTextSwitch(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.navigate("/settings/overlay")
	frames(tt)
	a.sess = session.Snapshot{ID: "s1", State: session.Recording, Partial: "so the meeting moves"}
	if text, _ := pillLine(a.pillSession()); text != "so the meeting moves" {
		t.Fatalf("on: %q", text)
	}
	for _, label := range []string{"Show live text", "Show the recording time"} {
		if err := tt.ClickControl(label); err != nil {
			t.Fatal(err)
		}
		frames(tt)
	}
	if o := a.settings.Get().Overlay; !o.HideLiveText || !o.ShowTime {
		t.Fatalf("settings %+v", o)
	}
	if text, _ := pillLine(a.pillSession()); text != "" {
		t.Errorf("off: %q", text)
	}
	a.pillNote.text, a.pillNote.id = "The sound could not be muted", "s1"
	if text, _ := pillLine(a.pillSession()); text != "The sound could not be muted" {
		t.Errorf("off, the note: %q", text)
	}
}

// TestLiveTextCard: with Show live text off, a follow-up in the answer
// card shows neither the words being said nor the answer being written,
// as the pill does not.
// TestLiveTextNoLLMDraft: after stop, the pill keeps the live ASR text,
// while the speech model finishes and while the language model works, and
// never replays the language model's draft (FB6). It went blank while the
// language model worked (#3644).
func TestLiveTextNoLLMDraft(t *testing.T) {
	a, _, _ := newTestApp(t)
	stream := "meet at three tomorrow"
	a.sess = session.Snapshot{State: session.Recording, Partial: stream}
	if text, _ := pillLine(a.pillSession()); text != stream {
		t.Fatalf("recording: %q", text)
	}
	a.sess = session.Snapshot{State: session.Composing, Partial: stream, Draft: "Meet at three."}
	if text, _ := pillLine(a.pillSession()); text != stream {
		t.Fatalf("composing: %q", text)
	}
	a.sess = session.Snapshot{State: session.Recognizing, Partial: stream}
	if text, _ := pillLine(a.pillSession()); text != stream {
		t.Fatalf("recognizing: %q", text)
	}
}

func TestLiveTextCard(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.updateSettings(func(s *settings.Settings) { s.Overlay.HideLiveText = true })
	steps := cardSteps()
	words := func() (all []string) {
		for _, b := range a.cardNow().bubbles {
			all = append(all, b.text)
		}
		return all
	}
	for i, st := range steps[:9] { // to "follow-up-writing"
		st.apply(a)
		switch st.name {
		case "follow-up-recording", "follow-up-writing":
			if w := words(); slices.Contains(w, "shorter") && st.name == "follow-up-recording" || slices.Contains(w, "Ten minutes late") {
				t.Errorf("%d %s: %q", i, st.name, w)
			}
		}
	}
}

// TestLiveLines: Overlay style → Live text layout → Lines shows the live
// text wide from the first word, wrapped, the face growing with its lines
// up to layout.LinesMax and no higher. With one line (the default) the
// face grew sideways and the start of what was said scrolled away
// (#3644 FB5).
func TestLiveLines(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.navigate("/settings/overlay")
	frames(tt)
	a.sess = session.Snapshot{ID: "s1", State: session.Recording, Partial: "Meet at three"}
	if w, h, _ := a.overlaySize(ovRecording); w >= layout.LinesW || h != layout.LiveH {
		t.Fatalf("one line: %d×%d", w, h)
	}
	if err := tt.ClickControl("Lines"); err != nil {
		t.Fatal(err)
	}
	frames(tt)
	if !a.settings.Get().Overlay.LiveLines {
		t.Fatal("not saved")
	}
	_, one, _ := a.overlaySize(ovRecording)
	if w, _, _ := a.overlaySize(ovRecording); w != layout.LinesW {
		t.Errorf("lines: %d wide", w)
	}
	a.sess.Partial = strings.Repeat("and then the meeting moves to Thursday ", 12)
	_, some, _ := a.overlaySize(ovRecording)
	a.sess.Partial = strings.Repeat("and then the meeting moves to Thursday ", 200)
	_, most, _ := a.overlaySize(ovRecording)
	if !(one < some && some < most) || most != layout.LinesHeight(layout.LinesMax*layout.LineH) {
		t.Errorf("heights: one line %d, a few %d, many %d (most %d)", one, some, most, layout.LinesHeight(layout.LinesMax*layout.LineH))
	}
}

// TestRewriteWritesIntoCard: Rewrite's first answer, when it will end in
// the card, goes to the answer card as the model writes it, the card as
// high as the words so far (it grows with them, #3644 FB41), the pill
// until then, with Show live text off, and when the answer goes in at
// once (the card switch off). The pill showed a loader until the whole
// answer was in (#3644).
func TestRewriteWritesIntoCard(t *testing.T) {
	a, _, _ := newTestApp(t)
	s := session.Snapshot{ID: "r1", Kind: store.KindRewrite, State: session.Composing, Card: true, Partial: "make it shorter", Selection: "A long paragraph."}
	a.sess = s
	if m := a.overlayModeNow(); m != ovWorking {
		t.Fatalf("before the first words: mode %v", m)
	}
	s.Draft, s.Title = "Short.", "Shorter"
	a.sess = s
	if m := a.overlayModeNow(); m != ovAnswer {
		t.Fatalf("writing: mode %v", m)
	}
	w, h, _ := a.cardSize()
	if v := a.cardNow(); v.text != "Short." || !v.busy || w != layout.CardSizes[0].W || h <= layout.CardChrome || h >= layout.CardChrome+layout.CardBodyMax {
		t.Errorf("the writing card: %q busy %v, %d×%d", v.text, v.busy, w, h)
	}
	a.prefs.Overlay.HideLiveText = true
	if m := a.overlayModeNow(); m != ovWorking {
		t.Errorf("Show live text off: mode %v", m)
	}
	a.prefs.Overlay.HideLiveText = false
	s.Card = false
	a.sess = s
	if m := a.overlayModeNow(); m != ovWorking {
		t.Errorf("going in at once: mode %v", m)
	}
}

// TestRewriteWritingFollows: as the model writes a long first answer into
// the card, the card's text follows the newest words; it stayed at the
// top, and the answer streamed out of sight below its last line.
func TestRewriteWritingFollows(t *testing.T) {
	a, _, _ := newTestApp(t)
	shortScreen(t, a)
	long := strings.Repeat("The meeting moves to Thursday at three. ", 30)
	s := session.Snapshot{ID: "r1", Kind: store.KindRewrite, State: session.Composing, Card: true, Partial: "make it longer", Selection: "x", Draft: long[:200]}
	snap(a, s)
	tt := hudTester(a.overlayView, a.ov.w+2*layout.Pad, a.ov.h+2*layout.Pad)
	frames(tt)
	s.Draft = long
	snap(a, s)
	frames(tt)
	if st := a.ov.cardScroll; st.MaxY <= 0 || st.Y < st.MaxY-2 {
		t.Errorf("the writing card shows %v of %v: not the newest words", st.Y, st.MaxY)
	}
}

// TestLiveLinesScroll: live text in lines longer than the face takes the
// wheel: the overlay stops letting the pointer through, the wheel scrolls
// up to what came before and new words leave it there; back at the end,
// it follows the newest words again. The lines before the last ten were
// out of reach (#3644).
func TestLiveLinesScroll(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.prefs.Overlay.LiveLines = true
	words := strings.Repeat("and then the meeting moves to Thursday ", 40)
	a.sess = session.Snapshot{ID: "s1", State: session.Recording, Partial: "short"}
	a.overlaySync()
	if !a.ov.passThrough {
		t.Fatal("a short text keeps the pointer from passing through")
	}
	a.sess.Partial = words
	a.overlaySync()
	if a.ov.passThrough {
		t.Fatal("the overflowing text lets the pointer through: the wheel cannot reach it")
	}
	tt := hudTester(a.overlayView, a.ov.w+2*layout.Pad, a.ov.h+2*layout.Pad)
	frames(tt)
	l := &a.ov.lines
	if !l.follow || l.st.MaxY <= 0 || l.st.Y != l.st.MaxY {
		t.Fatalf("at the start: %+v", *l)
	}
	x, y := float32(a.ov.w/2+layout.Pad), float32(a.ov.h+layout.Pad-layout.LivePadY-20)
	tt.Move(x, y)
	tt.Scroll(x, y, 0, -60)
	frames(tt)
	up := l.st.Y
	if l.follow || up >= l.st.MaxY {
		t.Fatalf("after the wheel up: %+v", *l)
	}
	a.sess.Partial = words + "and more words"
	frames(tt)
	if l.st.Y != up || l.follow {
		t.Errorf("new words moved the text read: %v → %v", up, l.st.Y)
	}
	tt.Advance(time.Second)
	tt.Scroll(x, y, 0, 10000)
	frames(tt)
	a.sess.Partial = words + "and more words, and the end"
	frames(tt)
	if !l.follow || l.st.Y != l.st.MaxY {
		t.Errorf("back at the end: %+v", *l)
	}
}

// TestLiveLinesOverflowMeasured: the live text guessed to fit ten lines
// lets the pointer through; when its measured height says it does not,
// the window takes the pointer again, so that the wheel scrolls it rather
// than the app under the overlay.
func TestLiveLinesOverflowMeasured(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.measureHook = nil // the guesses until the view measured, then the measures
	a.prefs.Overlay.LiveLines = true
	text := strings.Repeat("meet at three ", 20)
	a.sess = session.Snapshot{ID: "s1", State: session.Recording, Partial: text}
	a.overlaySync()
	if !a.ov.passThrough {
		t.Fatal("a text guessed to fit keeps the pointer from passing through")
	}
	line, _ := pillLine(a.pillSession())
	if a.ov.textW == nil {
		a.ov.textW = map[ovSized]ovMeasured{}
	}
	a.ov.textW[liveLinesKey(line)] = ovMeasured{w: layout.LinesTextW, h: (layout.LinesMax + 2) * layout.LineH}
	a.overlayRestyle()
	if a.ov.passThrough {
		t.Error("the text measured past ten lines still lets the pointer through")
	}
}
