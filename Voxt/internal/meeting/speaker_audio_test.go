package meeting

import (
	"context"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/speaker"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
)

type voiceTestModel struct {
	ready   <-chan struct{}
	entered chan struct{}
	at      time.Duration
}

func (m *voiceTestModel) Open(ctx context.Context) (speaker.Stream, error) {
	select {
	case m.entered <- struct{}{}:
	default:
	}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-m.ready:
		return m, nil
	}
}
func (m *voiceTestModel) Push(_ context.Context, pcm []byte) ([]speaker.Frame, error) {
	end := m.at + pcmDuration(pcm)
	var frames []speaker.Frame
	for m.at < end {
		scores := [8]float32{0.99, 0, 0, 0}
		if m.at >= 1100*time.Millisecond {
			scores = [8]float32{0, 0.99, 0, 0}
		}
		next := min(m.at+80*time.Millisecond, end)
		frames = append(frames, speaker.Frame{Start: m.at, End: next, Scores: scores})
		m.at = next
	}
	return frames, nil
}
func (m *voiceTestModel) Close() error { return nil }

func TestSpeakerQueueIsBoundedWhileModelLoads(t *testing.T) {
	model := &voiceTestModel{ready: make(chan struct{}), entered: make(chan struct{}, 1)}
	warnings := make(chan string, 1)
	run := newVoiceRun(model, func([]speaker.Frame) { t.Error("unloaded model produced frames") }, func(s string) { warnings <- s })
	defer run.close()
	<-model.entered
	// No inference result is needed for this producer to return. The
	// classifier stops at its limit rather than blocking ASR/capture.
	run.offer(0, make([]byte, asr.BytesPerSecond*40))
	select {
	case <-run.done:
	case <-time.After(5 * time.Second):
		t.Fatal("overflow did not cancel the loading model")
	}
	if run.bytes.Load() != 0 {
		t.Fatal("failed classifier retained queued audio")
	}
	select {
	case <-warnings:
	default:
		t.Fatal("missing classifier failure warning")
	}
}

func TestManagerSpeakersWhileRecording(t *testing.T) {
	ctx := context.Background()
	db, err := sqlitestore.Open(ctx, filepath.Join(t.TempDir(), "db"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	row, err := db.Meetings().CreateMeeting(ctx, store.Meeting{ID: "voices", Mode: store.MeetingModeMeeting, Status: store.MeetingRecording})
	if err != nil {
		t.Fatal(err)
	}
	ready := make(chan struct{})
	model := &voiceTestModel{ready: ready, entered: make(chan struct{}, 1)}
	now := time.Unix(1700000002, 0)
	m := NewManager(Config{Store: db.Meetings(), Now: func() time.Time { return now }, SpeakerModel: model, Settings: func() settings.Meeting { return settings.Meeting{Speakers: true} }})
	m.id, m.seq, m.mode, m.state = "voices", row.Seq, ModeMeeting, StateRecording
	m.snap = Snapshot{ID: "voices", State: StateRecording}
	m.timeline = newTimeline(m.cfg.Now, now.Add(-2*time.Second))
	run := m.speakerRun()
	defer run.close()
	<-model.entered
	// Fast recognition commits text while speaker model loading is held.
	m.commitSegment(Segment{ID: "a", Source: SourceSystem, Start: 0, End: time.Second, Text: "First voice."}, nil)
	m.commitSegment(Segment{ID: "b", Source: SourceSystem, Start: 1100 * time.Millisecond, End: 2 * time.Second, Text: "Second voice."}, nil)
	m.segWrites.Wait()
	if len(m.Current().Segments) != 2 {
		t.Fatal("speaker loading blocked transcription")
	}
	if err := db.Meetings().UpdateSegmentTranslation(ctx, row.Seq, 0, "第一位。"); err != nil {
		t.Fatal(err)
	}
	zero, bob := time.Duration(0), 1100*time.Millisecond
	m.ObserveSpeakers("voices", SpeakerObservation{Active: []string{"Alice"}, At: &zero})
	m.ObserveSpeakers("voices", SpeakerObservation{Active: []string{"Bob"}, At: &bob})
	run.offer(0, make([]byte, asr.BytesPerSecond*2))
	close(ready)
	drainCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	run.drain(drainCtx)
	if drainCtx.Err() != nil {
		t.Fatal(drainCtx.Err())
	}
	m.ObserveSpeakers("voices", SpeakerObservation{}) // hidden meeting window
	snapshot := m.Current()
	if snapshot.Segments[0].Speaker != "s1" || snapshot.Segments[1].Speaker != "s2" || snapshot.LiveSpeakers[SourceSystem] != "s2" {
		t.Fatalf("audio IDs lost with hidden window: %+v", snapshot)
	}
	if snapshot.Speakers["s1"] != "Alice" || snapshot.Speakers["s2"] != "Bob" {
		t.Fatalf("names: %+v", snapshot.Speakers)
	}
	// The normal automatic final pass must not replace established IDs
	// with plain text or the four-slot post-meeting diarizer.
	m.cfg.ASR = func(context.Context, Source) (TrackASR, error) {
		t.Error("final ASR replaced live acoustic rows")
		return TrackASR{}, nil
	}
	m.cfg.Settings = func() settings.Meeting { return settings.Meeting{Speakers: true, NoAutoSummary: true} }
	m.cfg.Dir = t.TempDir()
	m.timeline = newTimeline(m.cfg.Now, now.Add(-3*time.Minute))
	ended := make(chan struct{}, 1)
	m.Subscribe(func(s Snapshot) {
		if s.State == StateIdle {
			select {
			case ended <- struct{}{}:
			default:
			}
		}
	})
	if err := m.Finish(ctx, "voices"); err != nil {
		t.Fatal(err)
	}
	select {
	case <-ended:
	case <-time.After(5 * time.Second):
		t.Fatal("automatic finalization did not end")
	}
	_, segs, err := db.Meetings().Meeting(ctx, "voices")
	if err != nil || len(segs) != 2 || segs[0].Speaker != "s1" || segs[1].Speaker != "s2" || segs[0].Text != "First voice." || segs[0].Translation != "第一位。" {
		t.Fatalf("stored labels overwrote text/translation: %+v, %v", segs, err)
	}
}

type speakerTestCapture struct {
	input   chan []byte
	stopped chan struct{}
	once    sync.Once
}

func (c *speakerTestCapture) Record(ctx context.Context, id string, _ proto.CaptureParams, fn func(session.AudioEvent)) (proto.CaptureResult, error) {
	fn(session.AudioEvent{Started: &proto.CaptureStartedParams{Job: id, SampleRate: asr.SampleRate}})
	for {
		select {
		case <-ctx.Done():
			return proto.CaptureResult{}, ctx.Err()
		case <-c.stopped:
			return proto.CaptureResult{}, nil
		case pcm := <-c.input:
			fn(session.AudioEvent{PCM: &proto.PCMParams{Job: id, Data: pcm}})
		}
	}
}
func (c *speakerTestCapture) Stop(context.Context, string) error {
	c.once.Do(func() { close(c.stopped) })
	return nil
}

type speakerBlockedASR struct {
	liveFakeStreamer
	blocked <-chan struct{}
	entered chan struct{}
}

func (s *speakerBlockedASR) Open(ctx context.Context, req asr.StreamRequest) (asr.Stream, error) {
	base, err := s.liveFakeStreamer.Open(ctx, req)
	if err != nil {
		return nil, err
	}
	return &speakerBlockedStream{Stream: base, blocked: s.blocked, entered: s.entered}, nil
}

type speakerBlockedStream struct {
	asr.Stream
	blocked <-chan struct{}
	entered chan struct{}
}

func (s *speakerBlockedStream) Send(pcm []byte) error {
	select {
	case s.entered <- struct{}{}:
	default:
	}
	<-s.blocked
	return s.Stream.Send(pcm)
}

func TestCaptureFeedsSpeakersWhileASRSendIsBlocked(t *testing.T) {
	ctx := context.Background()
	db, err := sqlitestore.Open(ctx, filepath.Join(t.TempDir(), "db"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	ready := make(chan struct{})
	close(ready)
	model := &voiceTestModel{ready: ready, entered: make(chan struct{}, 1)}
	blocked := make(chan struct{})
	var release sync.Once
	backend := &speakerBlockedASR{blocked: blocked, entered: make(chan struct{}, 1)}
	mic := &speakerTestCapture{stopped: make(chan struct{})}
	system := &speakerTestCapture{input: make(chan []byte, 2), stopped: make(chan struct{})}
	now := time.Unix(1700000000, 0)
	m := NewManager(Config{Mic: mic, System: system, Store: db.Meetings(), Dir: t.TempDir(), Now: func() time.Time { return now }, SpeakerModel: model, Settings: func() settings.Meeting { return settings.Meeting{Speakers: true} }, ASR: func(_ context.Context, src Source) (TrackASR, error) {
		var streamer asr.Streamer = &liveFakeStreamer{}
		if src == SourceSystem {
			streamer = backend
		}
		return TrackASR{Caps: streamer.Caps(), Streamer: streamer}, nil
	}})
	updated := make(chan struct{}, 1)
	m.Subscribe(func(s Snapshot) {
		if s.SpeakerAt >= 400*time.Millisecond {
			select {
			case updated <- struct{}{}:
			default:
			}
		}
	})
	id, err := m.Start(ctx, StartRequest{Mode: ModeMeeting})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { release.Do(func() { close(blocked) }); m.Discard(ctx, id) }()
	system.input <- make([]byte, 6400)
	select {
	case <-backend.entered:
	case <-time.After(5 * time.Second):
		t.Fatal("ASR did not receive its first packet")
	}
	system.input <- make([]byte, 6400)
	select {
	case <-updated:
	case <-time.After(5 * time.Second):
		t.Fatal("blocked ASR stopped acoustic speaker updates")
	}
}
