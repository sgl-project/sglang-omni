package meeting_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
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

type fakeDiarizer struct {
	result   asr.Result
	requests []asr.FileRequest
}

func (f *fakeDiarizer) Caps() asr.Caps { return asr.Caps{File: true} }

func (f *fakeDiarizer) Diarize(_ context.Context, req asr.FileRequest) (asr.Result, error) {
	f.requests = append(f.requests, req)
	return f.result, nil
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
	detector := &countingVAD{}
	cfg := meeting.Config{
		Store:    db.Meetings(),
		Dir:      filepath.Join(root, "meetings"),
		VAD:      detector,
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
	if detector.calls != 2 || detector.sampleCounts[0] != rateSamples(2*time.Second) || detector.sampleCounts[1] != rateSamples(time.Second) {
		t.Fatalf("ordinary VAD calls %d, sample counts %v", detector.calls, detector.sampleCounts)
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

type countingVAD struct {
	calls        int
	sampleCounts []int
	regions      []vad.Segment
}

func (v *countingVAD) Speech(_ context.Context, pcm []byte, _ int) ([]vad.Segment, error) {
	v.calls++
	v.sampleCounts = append(v.sampleCounts, len(pcm)/2)
	if v.regions != nil {
		return v.regions, nil
	}
	return []vad.Segment{{Start: 0, End: len(pcm) / 2}}, nil
}

func (v *countingVAD) Open(context.Context) (vad.ModelStream, error) { return nil, nil }

func TestFinalizeReadsTwoTrackSegmentsInTimeOrder(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	db, err := sqlitestore.Open(ctx, filepath.Join(root, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	meetDir := filepath.Join(root, "meetings", "m-time-order")
	if err := os.MkdirAll(meetDir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeWAV(t, meetDir, "mic.wav", tonePCM(rateSamples(30*time.Second)))
	writeWAV(t, meetDir, "system.wav", tonePCM(rateSamples(30*time.Second)))
	mt, err := db.Meetings().CreateMeeting(ctx, store.Meeting{
		ID: "m-time-order", CreatedAt: time.Now(), Status: store.MeetingDone,
		Mode: store.MeetingModeMeeting, Audio: "m-time-order", DurationMS: 30000,
	})
	if err != nil {
		t.Fatal(err)
	}
	micASR := &seqFileASR{texts: []string{"mic-0", "mic-20"}}
	sysASR := &seqFileASR{texts: []string{"sys-0", "sys-10"}}
	detector := &countingVAD{}
	cfg := meeting.Config{
		Store:    db.Meetings(),
		Dir:      filepath.Join(root, "meetings"),
		VAD:      detector,
		Settings: func() settings.Meeting { return settings.Meeting{} },
		Privacy:  func() settings.Privacy { return settings.Privacy{KeepHistory: true} },
		ASR: func(ctx context.Context, src meeting.Source) (meeting.TrackASR, error) {
			switch src {
			case meeting.SourceMic:
				detector.regions = []vad.Segment{{Start: 0, End: rateSamples(5 * time.Second)}, {Start: rateSamples(20 * time.Second), End: rateSamples(25 * time.Second)}}
				return meeting.TrackASR{Caps: micASR.Caps(), File: micASR}, nil
			case meeting.SourceSystem:
				detector.regions = []vad.Segment{{Start: 0, End: rateSamples(5 * time.Second)}, {Start: rateSamples(10 * time.Second), End: rateSamples(15 * time.Second)}}
				return meeting.TrackASR{Caps: sysASR.Caps(), File: sysASR}, nil
			default:
				return meeting.TrackASR{}, nil
			}
		},
	}
	if err := meeting.FinalizePass(ctx, cfg, mt, nil); err != nil {
		t.Fatal(err)
	}
	_, segments, err := db.Meetings().Meeting(ctx, "m-time-order")
	if err != nil {
		t.Fatal(err)
	}
	want := []struct {
		text string
		idx  int
	}{{"mic-0", 0}, {"sys-0", 2}, {"sys-10", 3}, {"mic-20", 1}}
	if len(segments) != len(want) {
		t.Fatalf("segments %+v", segments)
	}
	for i, expected := range want {
		segment := segments[i]
		if segment.Text != expected.text || segment.Idx != expected.idx {
			t.Errorf("segment[%d] = %+v, want text %q with idx %d", i, segment, expected.text, expected.idx)
		}
		if i > 0 && segments[i-1].StartMS > segment.StartMS {
			t.Errorf("segments are not chronological: %+v", segments)
		}
	}
	if segments[0].StartMS != segments[1].StartMS {
		t.Errorf("the equal-start tie moved: %+v", segments)
	}
}

func TestFinalizeChunkOverridesAndDisabledVAD(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	db, err := sqlitestore.Open(ctx, filepath.Join(root, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	meetDir := filepath.Join(root, "meetings", "m-chunks")
	if err := os.MkdirAll(meetDir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeWAV(t, meetDir, "system.wav", tonePCM(rateSamples(3200*time.Millisecond)))
	mt, err := db.Meetings().CreateMeeting(ctx, store.Meeting{
		ID: "m-chunks", CreatedAt: time.Now(), Status: store.MeetingDone,
		Mode: store.MeetingModeSubtitles, Audio: "m-chunks", DurationMS: 3200,
	})
	if err != nil {
		t.Fatal(err)
	}
	file := &seqFileASR{texts: []string{"one", "two", "three", "four", "five"}}
	detector := &countingVAD{}
	chunks := meeting.DefaultChunkParams()
	chunks.MaxDur = time.Second
	chunks.MinDur = 100 * time.Millisecond
	chunks.Pad = time.Millisecond
	chunks.HardOverlap = time.Millisecond
	cfg := meeting.Config{
		Store: db.Meetings(), Dir: filepath.Join(root, "meetings"), VAD: detector,
		Settings:   func() settings.Meeting { return settings.Meeting{} },
		Privacy:    func() settings.Privacy { return settings.Privacy{KeepHistory: true} },
		DisableVAD: true, FinalizeChunkParams: &chunks,
		ASR: func(context.Context, meeting.Source) (meeting.TrackASR, error) {
			return meeting.TrackASR{Caps: file.Caps(), File: file}, nil
		},
	}
	if err := meeting.FinalizePass(ctx, cfg, mt, nil); err != nil {
		t.Fatal(err)
	}
	if detector.calls != 0 {
		t.Errorf("VAD ran %d times while disabled", detector.calls)
	}
	if file.n < 3 {
		t.Errorf("3.2 seconds at a one-second limit made %d ASR calls", file.n)
	}
}

func TestFinalizeDiarizesSystemAndReloadsSpeakerLabels(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	db, err := sqlitestore.Open(ctx, filepath.Join(root, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	meetDir := filepath.Join(root, "meetings", "m-speakers")
	if err := os.MkdirAll(meetDir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeWAV(t, meetDir, "mic.wav", tonePCM(rateSamples(2*time.Second)))
	writeWAV(t, meetDir, "system.wav", tonePCM(rateSamples(5*time.Second)))
	mt, err := db.Meetings().CreateMeeting(ctx, store.Meeting{
		ID: "m-speakers", CreatedAt: time.Now(), Status: store.MeetingDone,
		Mode: store.MeetingModeMeeting, Audio: "m-speakers", DurationMS: 5000,
	})
	if err != nil {
		t.Fatal(err)
	}
	micASR := &seqFileASR{texts: []string{"final mic"}}
	sysASR := &seqFileASR{texts: []string{"plain system"}}
	wantHints := []string{"SGLang", "Voxt"}
	diarizer := &fakeDiarizer{result: asr.Result{Segments: []asr.SpeakerSegment{
		{Start: 500 * time.Millisecond, End: 1500 * time.Millisecond, Speaker: "S01", Text: "first"},
		{Start: 1500 * time.Millisecond, End: 2500 * time.Millisecond, Speaker: "S02", Text: "second"},
		{Start: 2500 * time.Millisecond, End: 3500 * time.Millisecond, Speaker: "S04", Text: "fourth"},
	}}}
	hold := &fakeModelHold{}
	detector := &countingVAD{}
	cfg := meeting.Config{
		Store:     db.Meetings(),
		Dir:       filepath.Join(root, "meetings"),
		VAD:       detector,
		Settings:  func() settings.Meeting { return settings.Meeting{} },
		Privacy:   func() settings.Privacy { return settings.Privacy{KeepHistory: true} },
		ModelHold: hold,
		ASR: func(ctx context.Context, src meeting.Source) (meeting.TrackASR, error) {
			if src == meeting.SourceMic {
				return meeting.TrackASR{Caps: micASR.Caps(), File: micASR}, nil
			}
			return meeting.TrackASR{Caps: sysASR.Caps(), File: sysASR, Hints: wantHints}, nil
		},
		Diarizer: func(context.Context) (asr.Diarizer, error) {
			if hold.holds != 0 {
				t.Error("MTD resolved while the selected speech model was still held")
			}
			return diarizer, nil
		},
	}
	if err := meeting.FinalizePass(ctx, cfg, mt, nil); err != nil {
		t.Fatal(err)
	}
	// MTD goes in pieces planned from speech regions, so the system track
	// runs VAD too; the single full-range region still makes one request.
	if detector.calls != 2 || detector.sampleCounts[0] != rateSamples(2*time.Second) || detector.sampleCounts[1] != rateSamples(5*time.Second) {
		t.Fatalf("VAD calls %d, sample counts %v; want the microphone, then the system track", detector.calls, detector.sampleCounts)
	}
	if len(diarizer.requests) != 1 || len(diarizer.requests[0].Audio) != rateSamples(5*time.Second)*2 {
		t.Fatalf("diarizer requests %+v", diarizer.requests)
	}
	if !slices.Equal(diarizer.requests[0].Hints, wantHints) {
		t.Errorf("diarizer hot words = %q, want %q", diarizer.requests[0].Hints, wantHints)
	}
	if sysASR.n != 0 {
		t.Fatal("the plain system transcriber ran instead of MTD")
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	db, err = sqlitestore.Open(ctx, filepath.Join(root, "db"))
	if err != nil {
		t.Fatal(err)
	}
	_, segs, err := db.Meetings().Meeting(ctx, "m-speakers")
	if err != nil {
		t.Fatal(err)
	}
	var mic int
	var exports []meeting.ExportSegment
	for _, seg := range segs {
		if seg.Source == store.SegmentMic {
			mic++
			if seg.Speaker != "me" || seg.Text != "final mic" {
				t.Errorf("mic segment %+v", seg)
			}
			continue
		}
		exports = append(exports, meeting.ExportSegment{Start: time.Duration(seg.StartMS) * time.Millisecond, Speaker: seg.Speaker, Text: seg.Text})
	}
	if mic != 1 || len(exports) != 3 {
		t.Fatalf("segments %+v", segs)
	}
	for i, want := range []struct {
		speaker, text string
		startMS       int64
	}{{"s1", "first", 500}, {"s2", "second", 1500}, {"s4", "fourth", 2500}} {
		if exports[i].Speaker != want.speaker || exports[i].Text != want.text || exports[i].Start.Milliseconds() != want.startMS {
			t.Errorf("system %d: %+v", i, exports[i])
		}
	}
	transcript := meeting.ExportTxt(exports, nil)
	for _, want := range []string{"Speaker 1", "Speaker 2", "Speaker 4"} {
		if !strings.Contains(transcript, want) {
			t.Errorf("reloaded transcript %q lacks %q", transcript, want)
		}
	}
}

func TestFinalizeWithoutMTDUsesPlainSystemTrack(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	db, err := sqlitestore.Open(ctx, filepath.Join(root, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	meetDir := filepath.Join(root, "meetings", "m-plain")
	if err := os.MkdirAll(meetDir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeWAV(t, meetDir, "system.wav", tonePCM(rateSamples(time.Second)))
	mt, err := db.Meetings().CreateMeeting(ctx, store.Meeting{
		ID: "m-plain", CreatedAt: time.Now(), Status: store.MeetingDone,
		Mode: store.MeetingModeMeeting, Audio: "m-plain", DurationMS: 1000,
	})
	if err != nil {
		t.Fatal(err)
	}
	sysASR := &seqFileASR{texts: []string{"plain system"}}
	diarizer := &fakeDiarizer{}
	cfg := meeting.Config{
		Store:    db.Meetings(),
		Dir:      filepath.Join(root, "meetings"),
		Settings: func() settings.Meeting { return settings.Meeting{} },
		Privacy:  func() settings.Privacy { return settings.Privacy{KeepHistory: true} },
		ASR: func(context.Context, meeting.Source) (meeting.TrackASR, error) {
			return meeting.TrackASR{Caps: sysASR.Caps(), File: sysASR}, nil
		},
		Diarizer: func(context.Context) (asr.Diarizer, error) { return nil, nil },
	}
	if err := meeting.FinalizePass(ctx, cfg, mt, nil); err != nil {
		t.Fatal(err)
	}
	_, segs, err := db.Meetings().Meeting(ctx, "m-plain")
	if err != nil {
		t.Fatal(err)
	}
	if sysASR.n != 1 || len(diarizer.requests) != 0 || len(segs) != 1 || segs[0].Text != "plain system" || segs[0].Speaker != "" {
		t.Fatalf("segments %+v, plain calls %d, diarizer calls %d", segs, sysASR.n, len(diarizer.requests))
	}
}

func TestManagerFinalizeReleasesRecordingHoldBeforeMTD(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	db, err := sqlitestore.Open(ctx, filepath.Join(root, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	pcm := tonePCM(rateSamples(time.Second))
	micASR := &seqFileASR{texts: []string{"mic"}}
	sysASR := &seqFileASR{texts: []string{"plain"}}
	hold := &fakeModelHold{}
	called := make(chan struct{})
	cfg := meeting.Config{
		Mic:       newFakeAudio(pcm),
		System:    newFakeAudio(pcm),
		Store:     db.Meetings(),
		Dir:       filepath.Join(root, "meetings"),
		Settings:  func() settings.Meeting { return settings.Meeting{NoAutoSummary: true} },
		Privacy:   func() settings.Privacy { return settings.Privacy{KeepHistory: true} },
		ModelHold: hold,
		ASR: func(_ context.Context, src meeting.Source) (meeting.TrackASR, error) {
			if src == meeting.SourceMic {
				return meeting.TrackASR{Caps: micASR.Caps(), File: micASR}, nil
			}
			return meeting.TrackASR{Caps: sysASR.Caps(), File: sysASR}, nil
		},
		Diarizer: func(context.Context) (asr.Diarizer, error) {
			defer close(called)
			if hold.holds != 0 {
				t.Errorf("MTD runs with %d selected-model holds", hold.holds)
			}
			return &fakeDiarizer{result: asr.Result{Segments: []asr.SpeakerSegment{{End: time.Second, Speaker: "S01", Text: "other"}}}}, nil
		},
	}
	mgr := meeting.NewManager(cfg)
	id, err := mgr.Start(ctx, meeting.StartRequest{Mode: meeting.ModeMeeting})
	if err != nil {
		t.Fatal(err)
	}
	if hold.holds != 1 {
		t.Fatalf("after Start: %d holds", hold.holds)
	}
	if err := mgr.Finish(ctx, id); err != nil {
		t.Fatal(err)
	}
	select {
	case <-called:
	case <-time.After(10 * time.Second):
		t.Fatal("MTD did not run")
	}
	for deadline := time.Now().Add(10 * time.Second); ; time.Sleep(5 * time.Millisecond) {
		mt, _, err := db.Meetings().Meeting(ctx, id)
		if err == nil && mt.Status == store.MeetingDone {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("meeting %+v, err %v", mt, err)
		}
	}
	if hold.holds != 0 {
		t.Fatalf("after Finish: %d holds", hold.holds)
	}
	if micASR.n == 0 {
		t.Fatal("the selected model did not transcribe the microphone")
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
		Settings: func() settings.Meeting { return settings.Meeting{NoAutoSummary: true} },
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

// #3982: the final pass drops mic segments that are the system speech
// heard from the speakers, and keeps the user's own words.
func TestFinalizeDropsMicEcho(t *testing.T) {
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
	long := tonePCM(rateSamples(30 * time.Second)) // two chunks per track
	writeWAV(t, meetDir, "mic.wav", long)
	writeWAV(t, meetDir, "system.wav", long)
	mt, _ := db.Meetings().CreateMeeting(ctx, store.Meeting{
		ID: "m3", CreatedAt: time.Now(), Status: store.MeetingDone,
		Mode: store.MeetingModeMeeting, Audio: "m3",
	})
	echo := "然后一下出来，就是那整整齐齐的这个感觉。"
	own := "我自己不是做app这个背景的，所以觉得有点抽象。"
	micASR := &seqFileASR{texts: []string{"然后一下出来，就是那整整齐的这个感觉。", own}}
	sysASR := &seqFileASR{texts: []string{echo, "嗯。"}}
	cfg := meeting.Config{
		Store:    db.Meetings(),
		Dir:      filepath.Join(root, "meetings"),
		Settings: func() settings.Meeting { return settings.Meeting{} },
		Privacy:  func() settings.Privacy { return settings.Privacy{KeepHistory: true} },
		ASR: func(ctx context.Context, src meeting.Source) (meeting.TrackASR, error) {
			if src == meeting.SourceMic {
				return meeting.TrackASR{Caps: micASR.Caps(), File: micASR}, nil
			}
			return meeting.TrackASR{Caps: sysASR.Caps(), File: sysASR}, nil
		},
	}
	if err := meeting.FinalizePass(ctx, cfg, mt, nil); err != nil {
		t.Fatal(err)
	}
	_, segs, err := db.Meetings().Meeting(ctx, "m3")
	if err != nil {
		t.Fatal(err)
	}
	var mic, sys []string
	for _, s := range segs {
		if s.Source == store.SegmentMic {
			mic = append(mic, s.Text)
		} else {
			sys = append(sys, s.Text)
		}
	}
	if len(mic) != 1 || mic[0] != own {
		t.Fatalf("mic segments %q, want only %q", mic, own)
	}
	if len(sys) != 2 || sys[0] != echo {
		t.Fatalf("system segments %q", sys)
	}
}
