package main

import (
	"slices"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
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
// TestLiveTextNoLLMDraft: after stop, the pill keeps live ASR text and
// does not replay the language model draft (FB6).
func TestLiveTextNoLLMDraft(t *testing.T) {
	a, _, _ := newTestApp(t)
	stream := "meet at three tomorrow"
	a.sess = session.Snapshot{State: session.Recording, Partial: stream}
	if text, _ := pillLine(a.pillSession()); text != stream {
		t.Fatalf("recording: %q", text)
	}
	a.sess = session.Snapshot{State: session.Composing, Partial: stream, Draft: "Meet at three."}
	if text, _ := pillLine(a.pillSession()); text != "" {
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
