package settings

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// TestHistoryRetention: the zero value is Voxt's default (on, 90 days,
// any number); unknown numbers are the defaults; off keeps everything.
func TestHistoryRetention(t *testing.T) {
	now := time.Date(2026, 10, 17, 12, 0, 0, 0, time.UTC)
	var h History
	if h.Days() != 90 || h.Count() != 0 || !h.Cutoff(now).Equal(now.AddDate(0, 0, -90)) || h.PerKind() != 0 {
		t.Errorf("zero: %d %d %v %d", h.Days(), h.Count(), h.Cutoff(now), h.PerKind())
	}
	h = History{KeepDays: 7, KeepCount: 300}
	if h.Days() != 7 || h.PerKind() != 300 || !h.Cutoff(now).Equal(now.AddDate(0, 0, -7)) {
		t.Errorf("7 days, 300: %d %d", h.Days(), h.PerKind())
	}
	if (History{KeepDays: 12, KeepCount: 42}).Days() != 90 || (History{KeepCount: 42}).Count() != 0 {
		t.Error("unknown choices are not the defaults")
	}
	h.CleanupOff = true
	if !h.Cutoff(now).IsZero() || h.PerKind() != 0 {
		t.Errorf("off: %v %d", h.Cutoff(now), h.PerKind())
	}
	if !h.SameCleanup(History{CleanupOff: true, KeepDays: 30}) || h.SameCleanup(History{}) ||
		!(History{}).SameCleanup(History{KeepDays: 90, AudioDir: "/x"}) || (History{}).SameCleanup(History{KeepCount: 50}) {
		t.Error("SameCleanup")
	}
}

// TestHistoryDocument: older documents have no history part (Voxt's
// defaults, no migration); the choices survive a save and a read, and a
// default part is not written.
func TestHistoryDocument(t *testing.T) {
	s, err := Migrate([]byte(`{"version":4,"privacy":{"keepHistory":true}}`))
	if err != nil || s.History != (History{}) || s.Version != Version {
		t.Fatalf("older: %+v %v", s.History, err)
	}
	if data, _ := json.Marshal(Defaults()); strings.Contains(string(data), `"history"`) {
		t.Errorf("defaults write a history part: %s", data)
	}
	s.History = History{KeepDays: 30, KeepCount: 100, AudioDir: "/Volumes/Rec/VoxtNew"}
	data, _ := json.Marshal(s)
	back, err := Migrate(data)
	if err != nil || back.History != s.History {
		t.Errorf("round trip %+v %v", back.History, err)
	}
}
