package meeting

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
)

// importFileASR counts plain file transcriptions: an import must never
// use one (MTD labels the mixed track, or the import fails).
type importFileASR struct{ n int }

func (f *importFileASR) Caps() asr.Caps { return asr.Caps{File: true} }

func (f *importFileASR) Transcribe(context.Context, asr.FileRequest) (asr.Result, error) {
	f.n++
	return asr.Result{Text: "plain"}, nil
}

// importDiarizer is MTD with scripted speakers; block set has it work
// until the context ends (the cancel test).
type importDiarizer struct {
	mu       sync.Mutex
	result   asr.Result
	requests []asr.FileRequest
	block    chan struct{}
}

func (d *importDiarizer) Caps() asr.Caps { return asr.Caps{File: true} }

func (d *importDiarizer) Diarize(ctx context.Context, req asr.FileRequest) (asr.Result, error) {
	d.mu.Lock()
	d.requests = append(d.requests, req)
	d.mu.Unlock()
	if d.block != nil {
		close(d.block)
		<-ctx.Done()
		return asr.Result{}, ctx.Err()
	}
	return d.result, nil
}

func (d *importDiarizer) got() []asr.FileRequest {
	d.mu.Lock()
	defer d.mu.Unlock()
	return append([]asr.FileRequest(nil), d.requests...)
}

func importFixture(t *testing.T, chat *fixingChat, diarizer asr.Diarizer, plain *importFileASR) (*Manager, store.Meetings) {
	t.Helper()
	ctx := context.Background()
	s, err := sqlitestore.Open(ctx, filepath.Join(t.TempDir(), "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	m := NewManager(Config{
		Store:    s.Meetings(),
		Dir:      filepath.Join(t.TempDir(), "meetings"),
		Settings: func() settings.Meeting { return settings.Meeting{} },
		ASR: func(context.Context, Source) (TrackASR, error) {
			return TrackASR{Caps: plain.Caps(), File: plain}, nil
		},
		Diarizer: func(context.Context) (asr.Diarizer, error) { return diarizer, nil },
		LLM:      func(context.Context, string) (SummaryChat, error) { return chat, nil },
	})
	return m, s.Meetings()
}

func importPCM(seconds int) []byte {
	return silencePCM(time.Duration(seconds*1000)*time.Millisecond, asr.SampleRate)
}

// An imported recording gets MTD's speaker labels on every voice (S05
// keeps its number: a file can have more voices than a call's other
// side), then the correction and the summary, each asking once.
func TestImportDiarizesCorrectsAndSummarizes(t *testing.T) {
	chat := &fixingChat{}
	diarizer := &importDiarizer{result: asr.Result{Segments: []asr.SpeakerSegment{
		{Start: 0, End: time.Second, Speaker: "S01", Text: "Box one"},
		{Start: time.Second, End: 2 * time.Second, Speaker: "S02", Text: "two"},
		{Start: 2 * time.Second, End: 3 * time.Second, Speaker: "S05", Text: "three"},
	}}}
	plain := &importFileASR{}
	m, repo := importFixture(t, chat, diarizer, plain)
	ctx := context.Background()
	at := time.Date(2026, 11, 10, 9, 30, 0, 0, time.Local)
	lastP := 0.0
	id, err := m.Import(ctx, ImportRequest{Name: "zoom-standup", CreatedAt: at, PCM: importPCM(3)}, func(p float64) { lastP = p })
	if err != nil {
		t.Fatal(err)
	}
	if plain.n != 0 {
		t.Fatal("plain transcription ran for an import")
	}
	reqs := diarizer.got()
	if len(reqs) != 1 || len(reqs[0].Audio) != len(importPCM(3)) {
		t.Fatalf("MTD did not get the whole file in one request: %d requests", len(reqs))
	}
	if len(chat.reqs) != 2 {
		t.Fatalf("language model requests %d, want one correction and one summary", len(chat.reqs))
	}
	mt, segs, err := repo.Meeting(ctx, id)
	if err != nil {
		t.Fatal(err)
	}
	if mt.Status != store.MeetingDone || mt.Mode != store.MeetingModeImport {
		t.Fatalf("status %s, mode %s", mt.Status, mt.Mode)
	}
	var got []string
	for _, s := range segs {
		got = append(got, s.Speaker+":"+s.Text)
	}
	if strings.Join(got, "|") != "s1:Voxt one|s2:two|s5:three" {
		t.Fatalf("segments %q", got)
	}
	if label := SpeakerLabel("s5", nil); label != "Speaker 5" {
		t.Fatalf("SpeakerLabel(s5) %q", label)
	}
	if mt.Title != "T" || mt.Summary == "" {
		t.Fatalf("the summary did not name and fill the meeting: title %q", mt.Title)
	}
	if mt.DurationMS != 3000 || mt.CreatedAt.UnixMilli() != at.UnixMilli() {
		t.Fatalf("duration %d ms, date %v", mt.DurationMS, mt.CreatedAt)
	}
	if lastP != 1 {
		t.Errorf("progress ended at %v", lastP)
	}
}

// Without MTD an import fails instead of falling back to plain
// transcription: the file's speakers would be lost silently.
func TestImportWithoutMTDFailsInsteadOfPlainText(t *testing.T) {
	chat := &fixingChat{}
	plain := &importFileASR{}
	m, repo := importFixture(t, chat, nil, plain)
	ctx := context.Background()
	id, err := m.Import(ctx, ImportRequest{Name: "call", CreatedAt: time.Now(), PCM: importPCM(1)}, nil)
	if !errors.Is(err, ErrNoDiarizer) {
		t.Fatalf("err %v", err)
	}
	if plain.n != 0 {
		t.Fatal("an import fell back to plain transcription")
	}
	mt, segs, err := repo.Meeting(ctx, id)
	if err != nil {
		t.Fatal(err)
	}
	if mt.Status != store.MeetingInterrupted || len(segs) != 0 {
		t.Fatalf("status %s, %d segments", mt.Status, len(segs))
	}
	if len(chat.reqs) != 0 {
		t.Fatal("the language model ran for a failed import")
	}
}

// Cancelling an import stops MTD and never reaches the correction; the
// meeting stays interrupted for Continue processing.
func TestImportCancelStopsTheWork(t *testing.T) {
	chat := &fixingChat{}
	block := make(chan struct{})
	diarizer := &importDiarizer{block: block}
	plain := &importFileASR{}
	m, repo := importFixture(t, chat, diarizer, plain)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	var id string
	go func() {
		var err error
		id, err = m.Import(ctx, ImportRequest{Name: "long", CreatedAt: time.Now(), PCM: importPCM(1)}, nil)
		done <- err
	}()
	<-block // MTD is working on the file
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("err %v", err)
	}
	if len(chat.reqs) != 0 {
		t.Fatal("the correction or summary ran after the cancel")
	}
	mt, _, err := repo.Meeting(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	if mt.Status != store.MeetingInterrupted {
		t.Fatalf("status %s", mt.Status)
	}
}

// Without the automatic summary the meeting keeps the file's name, date
// and duration (the summary names it when it runs, as a recorded
// meeting's does).
func TestImportKeepsFileNameDateAndDuration(t *testing.T) {
	chat := &fixingChat{}
	diarizer := &importDiarizer{result: asr.Result{Segments: []asr.SpeakerSegment{
		{Start: 0, End: time.Second, Speaker: "S01", Text: "hi"},
	}}}
	plain := &importFileASR{}
	m, repo := importFixture(t, chat, diarizer, plain)
	m.cfg.Settings = func() settings.Meeting { return settings.Meeting{NoAutoSummary: true} }
	ctx := context.Background()
	at := time.Date(2026, 11, 10, 9, 30, 0, 0, time.Local)
	id, err := m.Import(ctx, ImportRequest{Name: "zoom-standup", CreatedAt: at, PCM: importPCM(3)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	mt, _, err := repo.Meeting(ctx, id)
	if err != nil {
		t.Fatal(err)
	}
	if mt.Title != "zoom-standup" {
		t.Fatalf("title %q", mt.Title)
	}
	if mt.CreatedAt.UnixMilli() != at.UnixMilli() {
		t.Fatalf("date %v, want the file's %v", mt.CreatedAt, at)
	}
	if mt.DurationMS != 3000 {
		t.Fatalf("duration %d ms", mt.DurationMS)
	}
	if len(chat.reqs) != 0 {
		t.Fatal("the language model ran with the automatic summary off")
	}
}
