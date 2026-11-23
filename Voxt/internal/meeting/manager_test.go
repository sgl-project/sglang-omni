package meeting_test

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
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
		Mic:      mic,
		System:   system,
		Store:    s.Meetings(),
		Dir:      filepath.Join(dir, "meetings"),
		Privacy:  func() settings.Privacy { return settings.Privacy{KeepHistory: true} },
		Settings: func() settings.Meeting { return settings.Meeting{NoFinalPass: true, NoAutoSummary: true} },
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
	text          string
	first         chan struct{}
	finishStarted chan struct{}
	releaseFinish chan struct{}
}

func newFinishGateStreamer() *finishGateStreamer {
	first := make(chan struct{}, 1)
	first <- struct{}{}
	return &finishGateStreamer{
		first:         first,
		finishStarted: make(chan struct{}),
		releaseFinish: make(chan struct{}),
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
	s.events <- asr.Event{Kind: asr.EventFinal, Text: s.streamer.text, Result: asr.Result{Text: s.streamer.text}}
	close(s.events)
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

type slowReconnectAudio struct {
	*fakeAudio
	thirdStarted chan struct{}
	allowThird   chan struct{}
}

func newSlowReconnectAudio(pcm []byte) *slowReconnectAudio {
	return &slowReconnectAudio{
		fakeAudio:    newFakeAudio(pcm),
		thirdStarted: make(chan struct{}),
		allowThird:   make(chan struct{}),
	}
}

func (a *slowReconnectAudio) Record(ctx context.Context, id string, params proto.CaptureParams, fn func(session.AudioEvent)) (proto.CaptureResult, error) {
	if len(a.jobIDs()) == 2 {
		close(a.thirdStarted)
		<-a.allowThird
	}
	return a.fakeAudio.Record(ctx, id, params, fn)
}

func TestStoppedTrackDoesNotReconnectAfterPauseSetModeResume(t *testing.T) {
	mic := newSlowReconnectAudio([]byte{9})
	streamer := newFinishGateStreamer()
	m, ctx := testManagerStreams(t, mic, newFakeAudio([]byte{8}), streamer, &meetTestStreamer{})
	t.Cleanup(streamer.release)
	id, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeRecording})
	if err != nil {
		t.Fatal(err)
	}
	oldDone := meeting.TrackDoneForTest(m, meeting.SourceMic)
	if oldDone == nil {
		t.Fatal("the first microphone track is not installed")
	}
	defer func() {
		close(mic.allowThird)
		_ = m.Finish(ctx, id)
	}()
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
	case <-oldDone:
	case <-mic.thirdStarted:
		t.Fatal("the replaced track started a slow reconnect")
	case <-time.After(2 * time.Second):
		t.Fatal("the replaced track did not settle")
	}
	if jobs := mic.jobIDs(); len(jobs) != 2 {
		t.Fatalf("the replaced track reconnected: jobs %v", jobs)
	}
}

func TestLateFinalDoesNotCommitIntoNextMeeting(t *testing.T) {
	mic := newFakeAudio([]byte{9})
	streamer := newFinishGateStreamer()
	streamer.text = "late words from A"
	m, ctx := testManagerStreams(t, mic, newFakeAudio([]byte{8}), streamer, &meetTestStreamer{})
	t.Cleanup(streamer.release)
	idA, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeRecording})
	if err != nil {
		t.Fatal(err)
	}
	oldDone := meeting.TrackDoneForTest(m, meeting.SourceMic)
	if oldDone == nil {
		t.Fatal("meeting A's microphone track is not installed")
	}
	finished := make(chan error, 1)
	go func() { finished <- m.Finish(ctx, idA) }()
	select {
	case <-streamer.finishStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("meeting A did not reach its held Final")
	}
	select {
	case err := <-finished:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(7 * time.Second):
		t.Fatal("meeting A did not finish behind its held Final")
	}
	idB, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeRecording})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Finish(ctx, idB)
	streamer.release()
	select {
	case <-oldDone:
	case <-time.After(2 * time.Second):
		t.Fatal("meeting A's late Final did not finish")
	}
	if segments := m.Current().Segments; len(segments) != 0 {
		t.Fatalf("meeting B received meeting A's late Final: %+v", segments)
	}
}

type pcmCrossAudio struct {
	mu    sync.Mutex
	calls int
	stops map[string]chan struct{}
}

func (a *pcmCrossAudio) Record(ctx context.Context, id string, _ proto.CaptureParams, fn func(session.AudioEvent)) (proto.CaptureResult, error) {
	a.mu.Lock()
	a.calls++
	call := a.calls
	stop := make(chan struct{})
	if a.stops == nil {
		a.stops = map[string]chan struct{}{}
	}
	a.stops[id] = stop
	a.mu.Unlock()
	fn(session.AudioEvent{Started: &proto.CaptureStartedParams{Job: id, SampleRate: asr.SampleRate}})
	if call == 1 {
		fn(session.AudioEvent{PCM: &proto.PCMParams{Job: id, Data: []byte{1, 1}}})
		fn(session.AudioEvent{PCM: &proto.PCMParams{Job: id, Data: []byte{2, 2}}})
	}
	select {
	case <-ctx.Done():
		return proto.CaptureResult{}, ctx.Err()
	case <-stop:
		return proto.CaptureResult{Reason: proto.CaptureStopped}, nil
	}
}

func (a *pcmCrossAudio) Stop(_ context.Context, id string) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if stop := a.stops[id]; stop != nil {
		close(stop)
		delete(a.stops, id)
	}
	return nil
}

type pcmCrossStreamer struct {
	mu      sync.Mutex
	opens   int
	release chan struct{}
	started chan struct{}
	newPCM  chan []byte
}

func (s *pcmCrossStreamer) Caps() asr.Caps { return asr.Caps{Stream: true} }

func (s *pcmCrossStreamer) Open(context.Context, asr.StreamRequest) (asr.Stream, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.opens++
	return &pcmCrossStream{owner: s, old: s.opens == 1, events: make(chan asr.Event, 1)}, nil
}

type pcmCrossStream struct {
	owner  *pcmCrossStreamer
	old    bool
	mu     sync.Mutex
	sends  int
	once   sync.Once
	events chan asr.Event
}

func (s *pcmCrossStream) Send(pcm []byte) error {
	s.mu.Lock()
	s.sends++
	call := s.sends
	s.mu.Unlock()
	if s.old && call == 1 {
		close(s.owner.started)
		<-s.owner.release
	}
	if !s.old {
		s.owner.newPCM <- bytes.Clone(pcm)
	}
	return nil
}

func (s *pcmCrossStream) Finish() error {
	s.once.Do(func() {
		if s.old {
			s.events <- asr.Event{Kind: asr.EventFinal, Text: "queued words from A", Result: asr.Result{Text: "queued words from A"}}
		}
		close(s.events)
	})
	return nil
}

func (s *pcmCrossStream) Close()                   {}
func (s *pcmCrossStream) Events() <-chan asr.Event { return s.events }

func TestOldPCMDoesNotEnterNextMeeting(t *testing.T) {
	testOldPCMIsolation(t, false, meeting.ModeRecording)
}

func TestOldPCMDoesNotEnterNextMeetingAfterFinalFailure(t *testing.T) {
	testOldPCMIsolation(t, true, meeting.ModeSubtitles)
}

func testOldPCMIsolation(t *testing.T, failFinal bool, nextMode meeting.Mode) {
	audio := &pcmCrossAudio{}
	streamer := &pcmCrossStreamer{release: make(chan struct{}), started: make(chan struct{}), newPCM: make(chan []byte, 1)}
	release := func() {
		select {
		case <-streamer.release:
		default:
			close(streamer.release)
		}
	}
	t.Cleanup(release)
	ctx := context.Background()
	dir := t.TempDir()
	s, err := sqlitestore.Open(ctx, filepath.Join(dir, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	var settingsMu sync.Mutex
	noFinalPass := !failFinal
	m := meeting.NewManager(meeting.Config{
		Mic: audio, System: newFakeAudio(nil), Store: s.Meetings(), Dir: filepath.Join(dir, "meetings"),
		Privacy: func() settings.Privacy { return settings.Privacy{KeepHistory: true} },
		Settings: func() settings.Meeting {
			settingsMu.Lock()
			defer settingsMu.Unlock()
			return settings.Meeting{NoFinalPass: noFinalPass, NoAutoSummary: true}
		},
		FreeSpace: func(string) (int64, error) { return meeting.DiskMinStartBytes() + 1, nil },
		ASR: func(_ context.Context, src meeting.Source) (meeting.TrackASR, error) {
			if src == meeting.SourceSystem {
				systemStreamer := &meetTestStreamer{}
				return meeting.TrackASR{Caps: systemStreamer.Caps(), Streamer: systemStreamer}, nil
			}
			return meeting.TrackASR{Caps: streamer.Caps(), Streamer: streamer}, nil
		},
	})
	idA, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeRecording})
	if err != nil {
		t.Fatal(err)
	}
	oldDone := meeting.TrackDoneForTest(m, meeting.SourceMic)
	select {
	case <-streamer.started:
	case <-time.After(2 * time.Second):
		t.Fatal("old PCM did not reach the old transcriber")
	}
	interrupted := make(chan struct{}, 1)
	cancelSubscribe := func() {}
	if failFinal {
		cancelSubscribe = m.Subscribe(func(s meeting.Snapshot) {
			if s.State == meeting.StateInterrupted {
				select {
				case interrupted <- struct{}{}:
				default:
				}
			}
		})
	}
	finished := make(chan error, 1)
	go func() { finished <- m.Finish(ctx, idA) }()
	select {
	case err := <-finished:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(7 * time.Second):
		t.Fatal("meeting A did not finish")
	}
	if failFinal {
		select {
		case <-interrupted:
		case <-time.After(2 * time.Second):
			t.Fatal("meeting A's final pass did not fail")
		}
		cancelSubscribe()
		settingsMu.Lock()
		noFinalPass = true
		settingsMu.Unlock()
	}
	idB, err := m.Start(ctx, meeting.StartRequest{Mode: nextMode})
	if err != nil {
		t.Fatal(err)
	}
	release()
	select {
	case <-oldDone:
	case <-time.After(2 * time.Second):
		t.Fatal("old track did not settle")
	}
	if segments := m.Current().Segments; len(segments) != 0 {
		t.Errorf("meeting B received meeting A's queued transcript: %+v", segments)
	}
	var crossedPCM []byte
	select {
	case crossedPCM = <-streamer.newPCM:
	default:
	}
	if err := m.Finish(ctx, idB); err != nil {
		t.Fatal(err)
	}
	wav, err := os.ReadFile(filepath.Join(dir, "meetings", idB, "mic.wav"))
	if err != nil && !os.IsNotExist(err) {
		t.Fatal(err)
	}
	if crossedPCM != nil {
		t.Errorf("meeting A PCM was sent to meeting B transcriber: %v", crossedPCM)
	}
	if bytes.Contains(wav, []byte{2, 2}) {
		t.Errorf("meeting A PCM was written to meeting B WAV")
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
