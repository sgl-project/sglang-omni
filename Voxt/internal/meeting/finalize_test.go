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
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
)

func rateSamples(d time.Duration) int {
	return int(d.Seconds() * float64(asr.SampleRate))
}

type seqFileASR struct {
	texts  []string
	n      int
	failAt int
}

func (s *seqFileASR) Caps() asr.Caps { return asr.Caps{File: true} }

func (s *seqFileASR) Transcribe(context.Context, asr.FileRequest) (asr.Result, error) {
	s.n++
	if s.failAt > 0 && s.n == s.failAt {
		return asr.Result{}, errors.New("transcribe failed")
	}
	text := ""
	if s.n-1 < len(s.texts) {
		text = s.texts[s.n-1]
	}
	return asr.Result{Text: text}, nil
}

func writeWAV(t *testing.T, dir, name string, pcm []byte) {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, asr.WAV(pcm, asr.SampleRate), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestFinalizeReplacesMicSegments(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	db, err := sqlitestore.Open(ctx, filepath.Join(root, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	meetDir := filepath.Join(root, "meetings", "m1")
	if err := os.MkdirAll(meetDir, 0o755); err != nil {
		t.Fatal(err)
	}
	pcm := tonePCM(rateSamples(2 * time.Second))
	writeWAV(t, meetDir, "mic.wav", pcm)
	writeWAV(t, meetDir, "system.wav", tonePCM(rateSamples(time.Second)))

	mt := store.Meeting{
		ID: "m1", CreatedAt: time.Now(), Status: store.MeetingDone,
		Mode: store.MeetingModeMeeting, Audio: "m1", DurationMS: 3000,
	}
	created, err := db.Meetings().CreateMeeting(ctx, mt)
	if err != nil {
		t.Fatal(err)
	}
	_ = db.Meetings().AppendSegments(ctx, created.Seq, []store.MeetingSegment{
		{Source: store.SegmentMic, Text: "live mic", StartMS: 0, EndMS: 500},
		{Source: store.SegmentSystem, Text: "live sys", StartMS: 0, EndMS: 500},
	})

	micASR := &seqFileASR{texts: []string{"final mic"}}
	sysASR := &seqFileASR{texts: []string{"final sys"}}
	cfg := meeting.Config{
		Store:    db.Meetings(),
		Dir:      filepath.Join(root, "meetings"),
		Settings: func() settings.Meeting { return settings.Meeting{} },
		Privacy:  func() settings.Privacy { return settings.Privacy{KeepHistory: true} },
		ASR: func(ctx context.Context, src meeting.Source) (meeting.TrackASR, error) {
			switch src {
			case meeting.SourceMic:
				return meeting.TrackASR{Caps: micASR.Caps(), File: micASR}, nil
			case meeting.SourceSystem:
				return meeting.TrackASR{Caps: sysASR.Caps(), File: sysASR}, nil
			default:
				return meeting.TrackASR{}, nil
			}
		},
	}

	if err := meeting.FinalizePass(ctx, cfg, created, nil); err != nil {
		t.Fatal(err)
	}
	_, segs, err := db.Meetings().Meeting(ctx, "m1")
	if err != nil {
		t.Fatal(err)
	}
	var micText, sysText string
	for _, s := range segs {
		switch s.Source {
		case store.SegmentMic:
			if s.Final {
				micText = s.Text
			}
		case store.SegmentSystem:
			if s.Final {
				sysText = s.Text
			}
		}
	}
	if micText != "final mic" || sysText != "final sys" {
		t.Fatalf("segments mic=%q sys=%q", micText, sysText)
	}
}

func TestFinalizeFailureKeepsLiveAndRetry(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	db, err := sqlitestore.Open(ctx, filepath.Join(root, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	meetDir := filepath.Join(root, "meetings", "m2")
	if err := os.MkdirAll(meetDir, 0o755); err != nil {
		t.Fatal(err)
	}
	long := tonePCM(rateSamples(30 * time.Second))
	writeWAV(t, meetDir, "mic.wav", long)
	mt, _ := db.Meetings().CreateMeeting(ctx, store.Meeting{
		ID: "m2", CreatedAt: time.Now(), Status: store.MeetingInterrupted,
		Mode: store.MeetingModeMeeting, Audio: "m2",
	})
	_ = db.Meetings().AppendSegments(ctx, mt.Seq, []store.MeetingSegment{
		{Source: store.SegmentMic, Text: "live", StartMS: 0, EndMS: 1000},
	})

	micASR := &seqFileASR{texts: []string{"one", "two"}, failAt: 2}
	cfg := meeting.Config{
		Store:    db.Meetings(),
		Dir:      filepath.Join(root, "meetings"),
		Settings: func() settings.Meeting { return settings.Meeting{} },
		Privacy:  func() settings.Privacy { return settings.Privacy{} },
		ASR: func(ctx context.Context, src meeting.Source) (meeting.TrackASR, error) {
			return meeting.TrackASR{Caps: micASR.Caps(), File: micASR}, nil
		},
	}
	if err := meeting.FinalizePass(ctx, cfg, mt, nil); err == nil {
		t.Fatal("expected failure")
	}
	updated, segs, _ := db.Meetings().Meeting(ctx, "m2")
	if updated.Status != store.MeetingInterrupted {
		t.Fatalf("status %s", updated.Status)
	}
	if updated.AudioUntil.IsZero() {
		t.Fatal("expected audio_until on failure when KeepAudio is off")
	}
	for _, s := range segs {
		if s.Text == "live" && !s.Final {
			return
		}
	}
	t.Fatal("live segment missing")
}

func TestReprocessInterruptedMeeting(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	db, err := sqlitestore.Open(ctx, filepath.Join(root, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	meetDir := filepath.Join(root, "meetings", "m3")
	if err := os.MkdirAll(meetDir, 0o755); err != nil {
		t.Fatal(err)
	}
	pcm := tonePCM(rateSamples(2 * time.Second))
	writeWAV(t, meetDir, "mic.wav", pcm)
	writeWAV(t, meetDir, "system.wav", tonePCM(rateSamples(time.Second)))
	mt, err := db.Meetings().CreateMeeting(ctx, store.Meeting{
		ID: "m3", CreatedAt: time.Now(), Status: store.MeetingInterrupted,
		Mode: store.MeetingModeMeeting, Audio: "m3", DurationMS: 2000,
	})
	if err != nil {
		t.Fatal(err)
	}
	_ = db.Meetings().AppendSegments(ctx, mt.Seq, []store.MeetingSegment{
		{Source: store.SegmentMic, Text: "live", StartMS: 0, EndMS: 500},
	})
	micASR := &seqFileASR{texts: []string{"final"}}
	sysASR := &seqFileASR{texts: []string{"sys final"}}
	cfg := meeting.Config{
		Store:    db.Meetings(),
		Dir:      filepath.Join(root, "meetings"),
		Settings: func() settings.Meeting { return settings.Meeting{} },
		Privacy:  func() settings.Privacy { return settings.Privacy{KeepHistory: true} },
		ASR: func(ctx context.Context, src meeting.Source) (meeting.TrackASR, error) {
			switch src {
			case meeting.SourceMic:
				return meeting.TrackASR{Caps: micASR.Caps(), File: micASR}, nil
			case meeting.SourceSystem:
				return meeting.TrackASR{Caps: sysASR.Caps(), File: sysASR}, nil
			default:
				return meeting.TrackASR{}, nil
			}
		},
	}
	mgr := meeting.NewManager(cfg)
	if err := mgr.Reprocess(ctx, "m3"); err != nil {
		t.Fatal(err)
	}
	updated, _, err := db.Meetings().Meeting(ctx, "m3")
	if err != nil || updated.Status != store.MeetingDone {
		t.Fatalf("status %s err %v", updated.Status, err)
	}
}

func TestFinalizeRecordingModeKeepsAudio(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	db, err := sqlitestore.Open(ctx, filepath.Join(root, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	meetDir := filepath.Join(root, "meetings", "m4rec")
	if err := os.MkdirAll(meetDir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeWAV(t, meetDir, "mic.wav", tonePCM(rateSamples(time.Second)))
	mt, _ := db.Meetings().CreateMeeting(ctx, store.Meeting{
		ID: "m4rec", CreatedAt: time.Now(), Status: store.MeetingDone,
		Mode: store.MeetingModeRecording, Audio: "m4rec",
	})
	micASR := &seqFileASR{texts: []string{"ok"}}
	cfg := meeting.Config{
		Store:    db.Meetings(),
		Dir:      filepath.Join(root, "meetings"),
		Settings: func() settings.Meeting { return settings.Meeting{} },
		Privacy:  func() settings.Privacy { return settings.Privacy{KeepHistory: true} },
		ASR: func(context.Context, meeting.Source) (meeting.TrackASR, error) {
			return meeting.TrackASR{Caps: micASR.Caps(), File: micASR}, nil
		},
	}
	if err := meeting.FinalizePass(ctx, cfg, mt, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(meetDir); err != nil {
		t.Fatal("recording mode should keep audio folder")
	}
}

func TestFinalizeDeletesAudioWithoutKeep(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	db, err := sqlitestore.Open(ctx, filepath.Join(root, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	meetDir := filepath.Join(root, "meetings", "m4")
	if err := os.MkdirAll(meetDir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeWAV(t, meetDir, "mic.wav", tonePCM(rateSamples(time.Second)))
	mt, _ := db.Meetings().CreateMeeting(ctx, store.Meeting{
		ID: "m4", CreatedAt: time.Now(), Status: store.MeetingDone,
		Mode: store.MeetingModeMeeting, Audio: "m4",
	})
	micASR := &seqFileASR{texts: []string{"ok"}}
	cfg := meeting.Config{
		Store:    db.Meetings(),
		Dir:      filepath.Join(root, "meetings"),
		Settings: func() settings.Meeting { return settings.Meeting{} },
		Privacy:  func() settings.Privacy { return settings.Privacy{KeepHistory: true} },
		ASR: func(context.Context, meeting.Source) (meeting.TrackASR, error) {
			return meeting.TrackASR{Caps: micASR.Caps(), File: micASR}, nil
		},
	}
	if err := meeting.FinalizePass(ctx, cfg, mt, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(meetDir); !os.IsNotExist(err) {
		t.Fatal("audio folder should be removed")
	}
	updated, _, _ := db.Meetings().Meeting(ctx, "m4")
	if updated.Audio != "" {
		t.Fatalf("audio ref %q", updated.Audio)
	}
}

func tonePCM(samples int) []byte {
	out := make([]byte, samples*2)
	for i := 0; i < samples; i++ {
		out[i*2] = 0x10
		out[i*2+1] = 0x7f
	}
	return out
}
