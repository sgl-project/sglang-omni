package meeting_test

import (
	"context"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

// Without Silero, Quiet goes by the level against the room's floor:
// levels held SpeechOverFloorDB above the quietest part of the last half
// minute set it back to 0; a lone loud level (a click) does not.
func TestSnapshotQuiet(t *testing.T) {
	levels := slices.Repeat([]float64{-70}, 50)
	levels = append(levels, -20) // a click
	levels = append(levels, slices.Repeat([]float64{-70}, 50)...)
	levels = append(levels, -20, -20, -20, -20, -20) // half a second of speech
	levels = append(levels, slices.Repeat([]float64{-70}, 20)...)
	quiet := quietOver(t, nil, levels)
	if quiet[101] != 10100*time.Millisecond {
		t.Fatalf("quiet %v after the click, want 10.1s", quiet[101])
	}
	if got := quiet[len(quiet)-1]; got != 2*time.Second {
		t.Fatalf("quiet %v after the speech, want 2s", got)
	}
}

// A room's steady noise becomes its floor, however loud, so it counts as
// quiet; speech over it does not (11-25: a MacBook microphone averaged -32
// to -42 dBFS with nobody speaking, and -35 dBFS of noise read as speech
// to the energy detector).
func TestSnapshotQuietIgnoresSteadyNoise(t *testing.T) {
	levels := slices.Repeat([]float64{-35}, 1200)                  // two minutes
	levels = append(levels, -15, -15, -15, -15, -15)               // speech
	levels = append(levels, slices.Repeat([]float64{-35}, 300)...) // half a minute
	quiet := quietOver(t, &meetTestStreamer{text: "noise"}, levels)
	if quiet[1200] != 2*time.Minute || quiet[len(quiet)-1] != 30*time.Second {
		t.Fatalf("quiet %v before the speech and %v after it, want 2m and 30s", quiet[1200], quiet[len(quiet)-1])
	}
}

// With Silero the live detector decides, not the level: loud levels it
// hears no speech in add to Quiet.
func TestSnapshotQuietGoesBySilero(t *testing.T) {
	levels := slices.Repeat([]float64{-70, -10, -10, -10, -10, -70}, 10)
	quiet := quietOver(t, &meetTestStreamer{text: "music"}, levels, noSpeech{})
	if got := quiet[len(quiet)-1]; got != 6*time.Second {
		t.Fatalf("quiet %v: Silero heard nobody for 6s", got)
	}
}

type noSpeech struct{}

func (noSpeech) Feed(pcm []byte) ([]float64, error) { return make([]float64, len(pcm)/1024), nil }
func (noSpeech) Chunk() int                         { return 512 }
func (noSpeech) Threshold() float64                 { return 0.5 }
func (noSpeech) Close()                             {}

// quietOver records a microphone-only meeting's Quiet before each of
// levels, 100 ms apart, and after the last; streamer (nil: no recognition)
// and vad (Silero) are its live recognition.
func quietOver(t *testing.T, streamer *meetTestStreamer, levels []float64, vad ...vad.ModelStream) []time.Duration {
	t.Helper()
	ctx := context.Background()
	dir := t.TempDir()
	s, err := sqlitestore.Open(ctx, filepath.Join(dir, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	mic := newFakeAudio(make([]byte, asr.BytesPerSecond)) // a second of silence
	mic.levelsSent = make(chan struct{})
	mic.levels = levels
	clk := newFakeClock(time.Unix(1_700_000_000, 0))
	cfg := meeting.Config{
		Mic: mic, Store: s.Meetings(), Dir: filepath.Join(dir, "meetings"), Now: clk.Now,
		Privacy:   func() settings.Privacy { return settings.Privacy{KeepHistory: true} },
		FreeSpace: func(string) (int64, error) { return meeting.DiskMinStartBytes() + 1, nil },
	}
	if streamer != nil {
		cfg.ASR = func(context.Context, meeting.Source) (meeting.TrackASR, error) {
			ta := meeting.TrackASR{Caps: streamer.Caps(), Streamer: streamer}
			if len(vad) > 0 {
				ta.VAD = vad[0]
			}
			return ta, nil
		}
	}
	m := meeting.NewManager(cfg)
	var quiet []time.Duration
	mic.beforeLevel = func() {
		quiet = append(quiet, m.Current().Quiet)
		clk.Advance(100 * time.Millisecond)
	}
	id, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeRecording})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Finish(ctx, id)
	select {
	case <-mic.levelsSent:
	case <-time.After(2 * time.Second):
		t.Fatal("microphone did not finish sending its level events")
	}
	return append(quiet, m.Current().Quiet) // levels are taken in as they come
}

// A live meeting writes the speech model's Traditional characters in the
// user's Simplified script, in the live line and in the stored segment.
func TestManagerLiveScript(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	s, err := sqlitestore.Open(ctx, filepath.Join(dir, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	streamer := &meetTestStreamer{text: "這個會議為什麼"}
	m := meeting.NewManager(meeting.Config{
		Mic: newFakeAudio(liveTestPCM(t)), Store: s.Meetings(), Dir: filepath.Join(dir, "meetings"),
		Privacy:   func() settings.Privacy { return settings.Privacy{KeepHistory: true} },
		FreeSpace: func(string) (int64, error) { return meeting.DiskMinStartBytes() + 1, nil },
		ASR: func(context.Context, meeting.Source) (meeting.TrackASR, error) {
			return meeting.TrackASR{Caps: streamer.Caps(), Streamer: streamer, Script: "zh-Hans"}, nil
		},
	})
	lines := make(chan string, 64)
	defer m.Subscribe(func(s meeting.Snapshot) {
		if text := s.Live[meeting.SourceMic]; text != "" {
			select {
			case lines <- text:
			default:
			}
		}
	})()
	id, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeRecording})
	if err != nil {
		t.Fatal(err)
	}
	var live string
	select {
	case live = <-lines:
	case <-time.After(10 * time.Second):
		t.Fatal("no live text from the microphone")
	}
	if live != "这个会议为什么" {
		t.Errorf("live line %q", live)
	}
	if err := m.Finish(ctx, id); err != nil {
		t.Fatal(err)
	}
	_, segs, err := s.Meetings().Meeting(ctx, id)
	if err != nil || len(segs) == 0 {
		t.Fatalf("segments %v, %v", segs, err)
	}
	for _, g := range segs {
		if g.Text != "这个会议为什么" {
			t.Errorf("stored %q", g.Text)
		}
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
