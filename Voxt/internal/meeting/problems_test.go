package meeting

import (
	"context"
	"testing"
	"time"
)

func TestProblemSystemSilenceDoesNotCountPausedTime(t *testing.T) {
	clock := fakeClock{t: time.Unix(1_700_000_000, 0)}
	m := NewManager(Config{Now: clock.Now})
	m.id, m.mode, m.state = "paused-meeting", ModeMeeting, StateRecording
	m.snap.State = StateRecording
	m.timeline = newTimeline(clock.Now, clock.Now())
	m.noteMicText()
	m.noteLevel(SourceSystem, -120)
	clock.t = clock.t.Add(10 * time.Second)
	if err := m.Pause(context.Background(), m.id); err != nil {
		t.Fatal(err)
	}
	clock.t = clock.t.Add(time.Minute)
	m.noteFreeSpace(diskMinStartBytes + 1)
	if p := m.Current().Problem; p != nil {
		t.Fatalf("pause produced a recording warning: %+v", p)
	}
}

func TestProblemSystemSilenceConditions(t *testing.T) {
	for _, mode := range []Mode{ModeMeeting, ModeRecording, ModeSubtitles} {
		for _, hasText := range []bool{false, true} {
			for _, dbfs := range []float64{-120, -80, 0} {
				clock := fakeClock{t: time.Unix(1_700_000_000, 0)}
				ps := newProblemState(clock.Now)
				if hasText {
					ps.noteMicText()
				}
				ps.noteLevel(SourceSystem, dbfs)
				clock.t = clock.t.Add(30*time.Second - time.Nanosecond)
				if p := ps.pick(mode, nil); p != nil {
					t.Fatalf("early warning: mode=%s text=%v level=%g", mode, hasText, dbfs)
				}
				clock.t = clock.t.Add(time.Nanosecond)
				want := hasText && mode != ModeRecording && dbfs == -120
				if p := ps.pick(mode, nil); (p != nil) != want {
					t.Fatalf("warning=%+v want=%v: mode=%s text=%v level=%g", p, want, mode, hasText, dbfs)
				}
				ps.reset()
				if p := ps.pick(mode, nil); p != nil {
					t.Fatalf("warning survived a new meeting: %+v", p)
				}
			}
		}
	}
}

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
