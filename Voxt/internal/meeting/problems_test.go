package meeting

import (
	"testing"
	"time"
)

func TestProblemSystemSilentOnce(t *testing.T) {
	clock := fakeClock{t: time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)}
	ps := problemState{clock: clock.Now}
	mode := ModeMeeting

	ps.noteMicText()
	for i := 0; i < 29; i++ {
		ps.noteLevel(SourceSystem, -120)
		clock.t = clock.t.Add(time.Second)
		if p := ps.pick(mode, nil); p != nil {
			t.Fatalf("at %ds: early problem %+v", i+1, p)
		}
	}
	ps.noteLevel(SourceSystem, -120)
	clock.t = clock.t.Add(time.Second)
	p := ps.pick(mode, nil)
	if p == nil || p.Key != ProblemKeySystemSilent {
		t.Fatalf("want silent problem at 30s, got %+v", p)
	}
	ps.noteLevel(SourceSystem, -120)
	clock.t = clock.t.Add(10 * time.Second)
	if p := ps.pick(mode, nil); p != nil {
		t.Fatalf("prompt should show only once, got %+v", p)
	}
}

func TestProblemSystemSilentClearsWhenSignal(t *testing.T) {
	clock := fakeClock{t: time.Now()}
	ps := problemState{clock: clock.Now}
	ps.noteMicText()
	for range 20 {
		ps.noteLevel(SourceSystem, -120)
		clock.t = clock.t.Add(time.Second)
	}
	ps.noteLevel(SourceSystem, -40)
	clock.t = clock.t.Add(20 * time.Second)
	for range 20 {
		ps.noteLevel(SourceSystem, -120)
		clock.t = clock.t.Add(time.Second)
	}
	if p := ps.pick(ModeMeeting, nil); p != nil {
		t.Fatalf("quiet timer reset after signal: %+v", p)
	}
}

func TestProblemTrackDegradedWins(t *testing.T) {
	ps := problemState{clock: time.Now}
	track := &Problem{Key: "meeting.track.degraded.system"}
	ps.noteMicText()
	for range 31 {
		ps.noteLevel(SourceSystem, -120)
	}
	if p := ps.pick(ModeMeeting, track); p.Key != track.Key {
		t.Fatalf("track problem %v", p)
	}
}

func TestProblemDiskAndBacklog(t *testing.T) {
	ps := problemState{clock: time.Now}
	ps.noteFreeSpace(diskMinStartBytes)
	if p := ps.pick(ModeMeeting, nil); p != nil {
		t.Fatal("space above start floor does not warn")
	}
	ps.noteFreeSpace(diskMinStartBytes - 1)
	if p := ps.pick(ModeMeeting, nil); p == nil || p.Key != ProblemKeyDiskLow {
		t.Fatalf("disk low: %+v", p)
	}
	ps.reset()
	ps.noteBacklog(liveBacklogThreshold + time.Second)
	if p := ps.pick(ModeMeeting, nil); p == nil || p.Key != ProblemKeyLiveBacklog {
		t.Fatalf("backlog: %+v", p)
	}
}

type fakeClock struct{ t time.Time }

func (c *fakeClock) Now() time.Time { return c.t }
