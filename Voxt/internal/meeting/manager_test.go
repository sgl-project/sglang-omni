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
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
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
	return testManagerMic(t, newFakeAudio(micPCM), sysPCM, sysFail)
}

func testManagerMic(t *testing.T, mic session.Audio, sysPCM []byte, sysFail bool) (*meeting.Manager, context.Context, *fakeClock, string) {
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

type deadlineStopAudio struct {
	*fakeAudio
	missingDeadline bool
}

func (a *deadlineStopAudio) Stop(ctx context.Context, id string) error {
	if _, ok := ctx.Deadline(); !ok {
		a.missingDeadline = true
		return errors.New("capture stop has no deadline")
	}
	return a.fakeAudio.Stop(context.Background(), id)
}

func TestSystemStartFailureBoundsMicStop(t *testing.T) {
	mic := &deadlineStopAudio{fakeAudio: newFakeAudio([]byte{9})}
	m, ctx, _, _ := testManagerMic(t, mic, nil, true)
	errc := make(chan error, 1)
	go func() {
		_, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeMeeting})
		errc <- err
	}()
	select {
	case err := <-errc:
		var startErr *meeting.StartError
		if !errors.As(err, &startErr) || startErr.Source != meeting.SourceSystem || startErr.Err.Error() != "permission denied" {
			t.Fatalf("Start: %v", err)
		}
		if m.Current().State.Active() {
			t.Fatal("meeting should not be active after failed start")
		}
		if mic.missingDeadline {
			t.Fatal("microphone stop has no deadline")
		}
	case <-time.After(7 * time.Second):
		t.Fatal("Start still hangs after the system capture failed")
	}
}

func testManagerStreams(t *testing.T, mic, system session.Audio, micStreamer, systemStreamer asr.Streamer) (*meeting.Manager, context.Context) {
	t.Helper()
	ctx := context.Background()
	dir := t.TempDir()
	s, err := sqlitestore.Open(ctx, filepath.Join(dir, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	cfg := meeting.Config{
		Mic:     mic,
		System:  system,
		Store:   s.Meetings(),
		Dir:     filepath.Join(dir, "meetings"),
		Privacy: func() settings.Privacy { return settings.Privacy{KeepHistory: true} },
		FreeSpace: func(string) (int64, error) {
			return meeting.DiskMinStartBytes() + 1, nil
		},
		ASR: func(_ context.Context, src meeting.Source) (meeting.TrackASR, error) {
			streamer := micStreamer
			if src == meeting.SourceSystem {
				streamer = systemStreamer
			}
			return meeting.TrackASR{Caps: streamer.Caps(), Streamer: streamer}, nil
		},
	}
	return meeting.NewManager(cfg), ctx
}

type finishGateStreamer struct {
	first          chan struct{}
	finishStarted  chan struct{}
	releaseFinish  chan struct{}
	finishReturned chan struct{}
}

func newFinishGateStreamer() *finishGateStreamer {
	first := make(chan struct{}, 1)
	first <- struct{}{}
	return &finishGateStreamer{
		first:          first,
		finishStarted:  make(chan struct{}),
		releaseFinish:  make(chan struct{}),
		finishReturned: make(chan struct{}),
	}
}

func (s *finishGateStreamer) Caps() asr.Caps { return asr.Caps{Stream: true} }

func (s *finishGateStreamer) Open(context.Context, asr.StreamRequest) (asr.Stream, error) {
	select {
	case <-s.first:
		return &finishGateStream{streamer: s, events: make(chan asr.Event, 1)}, nil
	default:
		return &meetTestStream{events: make(chan asr.Event, 1)}, nil
	}
}

type finishGateStream struct {
	streamer *finishGateStreamer
	events   chan asr.Event
}

func (s *finishGateStream) Send([]byte) error { return nil }

func (s *finishGateStream) Finish() error {
	close(s.streamer.finishStarted)
	<-s.streamer.releaseFinish
	close(s.events)
	close(s.streamer.finishReturned)
	return nil
}

func (s *finishGateStream) Close() {}

func (s *finishGateStream) Events() <-chan asr.Event { return s.events }

func (s *finishGateStreamer) release() {
	select {
	case <-s.releaseFinish:
	default:
		close(s.releaseFinish)
	}
}

func TestSystemStartFailureKeepsCaptureErrorDuringSlowFinal(t *testing.T) {
	system := newFakeAudio(nil)
	system.startErr = errors.New("permission denied")
	streamer := newFinishGateStreamer()
	m, ctx := testManagerStreams(t, newFakeAudio([]byte{9}), system, &meetTestStreamer{}, streamer)
	t.Cleanup(streamer.release)
	errc := make(chan error, 1)
	go func() {
		_, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeMeeting})
		errc <- err
	}()
	select {
	case <-streamer.finishStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("the system track did not reach its slow Final")
	}
	select {
	case err := <-errc:
		var startErr *meeting.StartError
		if !errors.As(err, &startErr) || startErr.Source != meeting.SourceSystem || startErr.Err.Error() != "permission denied" {
			t.Fatalf("Start replaced the capture error: %v", err)
		}
	case <-time.After(12 * time.Second):
		t.Fatal("Start still hangs behind the slow Final")
	}
}

func TestStoppedTrackDoesNotReconnectAfterPauseSetModeResume(t *testing.T) {
	mic := newFakeAudio([]byte{9})
	streamer := newFinishGateStreamer()
	m, ctx := testManagerStreams(t, mic, newFakeAudio([]byte{8}), streamer, &meetTestStreamer{})
	t.Cleanup(streamer.release)
	id, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeRecording})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Finish(ctx, id)
	if err := m.Pause(ctx, id); err != nil {
		t.Fatal(err)
	}
	if err := m.SetMode(ctx, id, meeting.ModeMeeting); err != nil {
		t.Fatal(err)
	}
	if err := m.Resume(ctx, id); err != nil {
		t.Fatal(err)
	}
	streamer.release()
	select {
	case <-streamer.finishReturned:
	case <-time.After(2 * time.Second):
		t.Fatal("the replaced track did not finish")
	}
	<-time.After(250 * time.Millisecond)
	if jobs := mic.jobIDs(); len(jobs) != 2 {
		t.Fatalf("the replaced track reconnected: jobs %v", jobs)
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
			speech[i*2], speech[i*2+1] = 0, 0x10 // about -18 dBFS: the VAD hears it
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

// Each track keeps its last session.Bars levels, as bar heights, for its
// waveform in the meeting's window; the other track's stay its own.
func TestSnapshotKeepsLevelHistory(t *testing.T) {
	mic := newFakeAudio(nil)
	mic.levelsSent = make(chan struct{})
	for i := range session.Bars + 4 {
		mic.levels = append(mic.levels, -60+float64(i)*3)
	}
	m, ctx, _, _ := testManagerMic(t, mic, nil, false)
	id, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeRecording})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Finish(ctx, id)
	// A full buffer can still contain the first levels. Wait for the
	// complete sequence, then for the manager to take in the last level.
	select {
	case <-mic.levelsSent:
	case <-time.After(2 * time.Second):
		t.Fatal("microphone did not finish sending its level events")
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		levels := m.Current().Levels[meeting.SourceMic]
		if len(levels) > 0 && levels[len(levels)-1] == session.Level(-3) {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	got := m.Current().Levels
	if h := got[meeting.SourceMic]; len(h) != session.Bars || h[0] != session.Level(-48) || h[len(h)-1] != session.Level(-3) {
		t.Fatalf("mic levels %v: want the last %d, -48 to -3 dBFS", h, session.Bars)
	}
	if len(got[meeting.SourceSystem]) != 0 {
		t.Fatalf("system levels %v in a microphone-only meeting", got[meeting.SourceSystem])
	}
}
