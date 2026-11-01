package uitext

import (
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

func TestHistoryFormat(t *testing.T) {
	for ms, want := range map[int64]string{0: "", 850: "850 ms", 1180: "1.2 s", 59_000: "59.0 s", 65_000: "1:05"} {
		if got := StageTime(ms); got != want {
			t.Errorf("StageTime(%d) = %q, want %q", ms, got, want)
		}
	}
	if got := ClockMS(7400); got != "0:07" {
		t.Errorf("ClockMS = %q", got)
	}
	for _, c := range []struct {
		e      store.Entry
		title  string
		failed bool
	}{
		{store.Entry{Result: "\n  first  \nsecond"}, "first", false},
		{store.Entry{Status: store.StatusFailed, Error: "boom\nmore"}, "boom", true},
		{store.Entry{RawText: "heard"}, "heard", false},
		{store.Entry{}, "No text", true},
	} {
		if title, failed := EntryTitle(c.e); title != c.title || failed != c.failed {
			t.Errorf("EntryTitle(%+v) = %q, %v", c.e, title, failed)
		}
	}
}

// TestDeliveryLabel: the history detail names the app a result went
// into when the user moved there from the start app.
func TestDeliveryLabel(t *testing.T) {
	for _, c := range []struct {
		e    store.Entry
		want string
	}{
		{store.Entry{Delivery: "insert", AppName: "Slack"}, "Inserted"},
		{store.Entry{Delivery: "insert", AppName: "Slack", InsertedBundle: "com.apple.TextEdit", InsertedName: "TextEdit"}, "Inserted into TextEdit"},
		{store.Entry{Delivery: "card"}, "Shown in the answer card"},
		{store.Entry{Delivery: "copy"}, "Copied"},
		{store.Entry{}, ""},
	} {
		if got := DeliveryLabel(c.e); got != c.want {
			t.Errorf("%+v: %q, want %q", c.e, got, c.want)
		}
	}
	if got := EntryMeta(store.Entry{AppName: "Slack", InsertedName: "TextEdit"}); !strings.Contains(got, "Slack") || strings.Contains(got, "TextEdit") {
		t.Errorf("the list names the start app: %q", got)
	}
}

func TestSoundText(t *testing.T) {
	for _, c := range []struct {
		dsp  string
		gain float64
		want string
	}{
		{"", 0, ""},
		{"highpass+level", 12, "Hum filtered, volume +12.0 dB"},
		{"highpass+level", -2.5, "Hum filtered, volume -2.5 dB"},
		{"highpass+level", 0, "Hum filtered, volume unchanged"},
		{"highpass", 0, "Hum filtered"},
		{"level", 3, "Volume +3.0 dB"},
		{"level", 0, "Volume unchanged"},
	} {
		if got := SoundText(store.Entry{DSP: c.dsp, GainDB: c.gain}); got != c.want {
			t.Errorf("%q %v: %q, want %q", c.dsp, c.gain, got, c.want)
		}
	}
}
