package meeting_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
)

func testManagerHold(t *testing.T, modelHold interface{ Hold() (release func()) }) (*meeting.Manager, context.Context, *fakeClock, string) {
	t.Helper()
	ctx := context.Background()
	dir := t.TempDir()
	s, err := sqlitestore.Open(ctx, filepath.Join(dir, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	clk := newFakeClock(time.Unix(1_700_000_000, 0))
	cfg := meeting.Config{
		Mic:       newFakeAudio(nil),
		System:    newFakeAudio(nil),
		Store:     s.Meetings(),
		Dir:       filepath.Join(dir, "meetings"),
		Now:       clk.Now,
		Privacy:   func() settings.Privacy { return settings.Privacy{KeepHistory: true} },
		FreeSpace: func(string) (int64, error) { return meeting.DiskMinStartBytes() + 1, nil },
		ModelHold: modelHold,
	}
	return meeting.NewManager(cfg), ctx, clk, dir
}

func testManager(t *testing.T, micPCM, sysPCM []byte, sysFail bool) (*meeting.Manager, context.Context, *fakeClock, string) {
	t.Helper()
	ctx := context.Background()
	dir := t.TempDir()
	s, err := sqlitestore.Open(ctx, filepath.Join(dir, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	clk := newFakeClock(time.Unix(1_700_000_000, 0))
	sys := newFakeAudio(sysPCM)
	if sysFail {
		sys.startErr = errors.New("permission denied")
	}
	mic := newFakeAudio(micPCM)
	cfg := meeting.Config{
		Mic:     mic,
		System:  sys,
		Store:   s.Meetings(),
		Dir:     filepath.Join(dir, "meetings"),
		Now:     clk.Now,
		Privacy: func() settings.Privacy { return settings.Privacy{KeepHistory: true} },
		FreeSpace: func(string) (int64, error) {
			return meeting.DiskMinStartBytes() + 1, nil
		},
	}
	return meeting.NewManager(cfg), ctx, clk, dir
}

func TestStartPauseResumeFinish(t *testing.T) {
	want := []byte{1, 2, 3, 4}
	m, ctx, clk, _ := testManager(t, want, want, false)
	id, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeMeeting})
	if err != nil || id == "" {
		t.Fatalf("Start: %v", err)
	}
	clk.Advance(time.Second)
	beforePause := m.Current().Elapsed
	if err := m.Pause(ctx, id); err != nil {
		t.Fatal(err)
	}
	if m.Current().State != meeting.StatePaused {
		t.Fatalf("state: %s", m.Current().State)
	}
	clk.Advance(5 * time.Second)
	if m.Current().Elapsed != beforePause {
		t.Fatalf("pause froze elapsed: got %v want %v", m.Current().Elapsed, beforePause)
	}
	if err := m.Resume(ctx, id); err != nil {
		t.Fatal(err)
	}
	clk.Advance(time.Second)
	if err := m.Finish(ctx, id); err != nil {
		t.Fatal(err)
	}
	if m.Current().State != meeting.StateIdle {
		t.Fatalf("after finish: %s", m.Current().State)
	}
}

func TestSecondStartBusy(t *testing.T) {
	m, ctx, _, _ := testManager(t, nil, nil, false)
	if _, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeRecording}); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeRecording}); !errors.Is(err, meeting.ErrBusy) {
		t.Fatalf("want busy, got %v", err)
	}
}

func TestSystemStartFailsMicStopped(t *testing.T) {
	m, ctx, _, _ := testManager(t, []byte{9}, nil, true)
	var startErr *meeting.StartError
	_, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeMeeting})
	if !errors.As(err, &startErr) || startErr.Source != meeting.SourceSystem {
		t.Fatalf("Start: %v", err)
	}
	if m.Current().State.Active() {
		t.Fatal("meeting should not be active after failed start")
	}
}

func TestPCMWrittenToWAV(t *testing.T) {
	pcm := make([]byte, asr.BytesPerSecond)
	for i := range pcm {
		pcm[i] = byte(i % 251)
	}
	m, ctx, _, root := testManager(t, pcm, nil, false)
	id, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeRecording})
	if err != nil {
		t.Fatal(err)
	}
	time.Sleep(20 * time.Millisecond)
	if err := m.Finish(ctx, id); err != nil {
		t.Fatal(err)
	}
	wavPath := filepath.Join(root, "meetings", id, "mic.wav")
	got, err := os.ReadFile(wavPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) < 44+len(pcm) {
		t.Fatalf("wav short: %d", len(got))
	}
	if string(got[44:44+len(pcm)]) != string(pcm) {
		t.Fatal("pcm mismatch in wav")
	}
}

type fakeModelHold struct {
	holds int
}

func (f *fakeModelHold) Hold() (release func()) {
	f.holds++
	return func() { f.holds-- }
}

type meetTestStreamer struct {
	text string
}

func (m *meetTestStreamer) Caps() asr.Caps { return asr.Caps{Stream: true} }

func (m *meetTestStreamer) Open(context.Context, asr.StreamRequest) (asr.Stream, error) {
	return &meetTestStream{text: m.text, events: make(chan asr.Event, 4)}, nil
}

type meetTestStream struct {
	text   string
	events chan asr.Event
}

func (s *meetTestStream) Send([]byte) error {
	select {
	case s.events <- asr.Event{Kind: asr.EventPartial, Text: s.text}:
	default:
	}
	return nil
}

func (s *meetTestStream) Finish() error {
	s.events <- asr.Event{Kind: asr.EventFinal, Text: s.text, Result: asr.Result{Text: s.text}}
	close(s.events)
	return nil
}

func (s *meetTestStream) Close() {}

func (s *meetTestStream) Events() <-chan asr.Event { return s.events }

func liveTestPCM(t *testing.T) []byte {
	t.Helper()
	rate := 16000
	speechMs := 1500
	silenceMs := 900
	speech := make([]byte, speechMs*rate/1000*2)
	for i := range speechMs * rate / 1000 {
		if i%80 < 40 {
			speech[i*2], speech[i*2+1] = 0x60, 0
		}
	}
	silence := make([]byte, silenceMs*rate/1000*2)
	return append(speech, silence...)
}

func TestManagerLiveSegmentStore(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	s, err := sqlitestore.Open(ctx, filepath.Join(dir, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	streamer := &meetTestStreamer{text: "hello live"}
	cfg := meeting.Config{
		Mic:     newFakeAudio(liveTestPCM(t)),
		Store:   s.Meetings(),
		Dir:     filepath.Join(dir, "meetings"),
		Privacy: func() settings.Privacy { return settings.Privacy{KeepHistory: true} },
		FreeSpace: func(string) (int64, error) {
			return meeting.DiskMinStartBytes() + 1, nil
		},
		ASR: func(context.Context, meeting.Source) (meeting.TrackASR, error) {
			return meeting.TrackASR{Caps: streamer.Caps(), Streamer: streamer}, nil
		},
	}
	m := meeting.NewManager(cfg)
	id, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeRecording})
	if err != nil {
		t.Fatal(err)
	}
	// The final segment comes when the stream finishes: wait until it is open.
	for deadline := time.Now().Add(10 * time.Second); m.Current().Live[meeting.SourceMic] == ""; time.Sleep(5 * time.Millisecond) {
		if time.Now().After(deadline) {
			t.Fatal("no live text from the microphone")
		}
	}
	if err := m.Finish(ctx, id); err != nil {
		t.Fatal(err)
	}
	mt, segs, err := s.Meetings().Meeting(ctx, id)
	if err != nil {
		t.Fatal(err)
	}
	if mt.ID != id {
		t.Fatalf("meeting id %q", mt.ID)
	}
	if len(segs) == 0 {
		t.Fatal("expected committed segments in store")
	}
	found := false
	for _, g := range segs {
		if g.Text == "hello live" {
			found = true
			if g.Speaker != "me" {
				t.Fatalf("speaker %q", g.Speaker)
			}
		}
	}
	if !found {
		t.Fatalf("segments: %#v", segs)
	}
}

func TestManagerModelHold(t *testing.T) {
	hold := &fakeModelHold{}
	m, ctx, _, _ := testManagerHold(t, hold)
	id, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeRecording})
	if err != nil || hold.holds != 1 {
		t.Fatalf("Start: err=%v holds=%d", err, hold.holds)
	}
	if err := m.Finish(ctx, id); err != nil {
		t.Fatal(err)
	}
	if hold.holds != 0 {
		t.Fatalf("after finish holds=%d", hold.holds)
	}
}
