package main

import (
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
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
	a.muteNote.text, a.muteNote.id = "The sound could not be muted", "s1"
	if text, _ := pillLine(a.pillSession()); text != "The sound could not be muted" {
		t.Errorf("off, the note: %q", text)
	}
}
