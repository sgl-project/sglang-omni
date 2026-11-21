package meeting

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
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

// blockingSummaryChat lets the test cancel while the summary is running.
type blockingSummaryChat struct {
	fixingChat
	started chan struct{}
}

func (c *blockingSummaryChat) Chat(ctx context.Context, req llm.Request) (llm.Response, error) {
	if req.Messages[0].Role != llm.System {
		close(c.started)
		<-ctx.Done()
		return llm.Response{}, ctx.Err()
	}
	return c.fixingChat.Chat(ctx, req)
}

func importFixture(t *testing.T, chat SummaryChat, diarizer asr.Diarizer, plain *importFileASR) (*Manager, store.Meetings) {
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
	pcm := silencePCM(time.Duration(seconds*1000)*time.Millisecond, asr.SampleRate)
	for i := 0; i < len(pcm); i += 2 {
		pcm[i], pcm[i+1] = 0x10, 0x7f
	}
	return pcm
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

// A one-hour import finishes without allocating several whole-track copies,
// and no MTD request exceeds the two-minute request bound.
func TestImportOneHourBoundsPCMBufferingAndRequests(t *testing.T) {
	diarizer := &importDiarizer{result: asr.Result{Segments: []asr.SpeakerSegment{
		{Start: 0, End: time.Second, Speaker: "S01", Text: "hi"},
	}}}
	m, repo := importFixture(t, &fixingChat{}, diarizer, &importFileASR{})
	m.cfg.Settings = func() settings.Meeting { return settings.Meeting{NoAutoSummary: true} }
	pcm := importPCM(3600)
	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	id, err := m.Import(context.Background(), ImportRequest{Name: "one-hour", CreatedAt: time.Now(), PCM: pcm}, nil)
	runtime.ReadMemStats(&after)
	if err != nil {
		t.Fatal(err)
	}
	mt, _, err := repo.Meeting(context.Background(), id)
	if err != nil || mt.Status != store.MeetingDone {
		t.Fatalf("status %s, err %v", mt.Status, err)
	}
	requests := diarizer.got()
	if len(requests) != 31 {
		t.Fatalf("%d MTD requests, want 30 full chunks and the overlap tail", len(requests))
	}
	for i, req := range requests {
		if len(req.Audio) > int((diarizeChunkMax+vadOverlapBatch).Seconds())*asr.BytesPerSecond {
			t.Fatalf("request %d holds %d bytes", i, len(req.Audio))
		}
	}
	allocated := after.TotalAlloc - before.TotalAlloc
	t.Logf("allocated %.1f MiB across %d MTD requests", float64(allocated)/(1<<20), len(requests))
	if allocated > uint64(len(pcm)*5/4) {
		t.Fatalf("allocated %.1f MiB for a %.1f MiB track, want at most one full PCM copy plus working memory", float64(allocated)/(1<<20), float64(len(pcm))/(1<<20))
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

// Cancelling during the summary leaves an interrupted import with its WAV,
// so Continue processing can finish it instead of showing a done meeting
// with no summary and no way to reprocess it.
func TestImportCancelDuringSummaryCanContinue(t *testing.T) {
	chat := &blockingSummaryChat{started: make(chan struct{})}
	diarizer := &importDiarizer{result: asr.Result{Segments: []asr.SpeakerSegment{
		{Start: 0, End: time.Second, Speaker: "S01", Text: "Box one"},
	}}}
	m, repo := importFixture(t, chat, diarizer, &importFileASR{})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	var id string
	go func() {
		var err error
		id, err = m.Import(ctx, ImportRequest{Name: "long", CreatedAt: time.Now(), PCM: importPCM(1)}, nil)
		done <- err
	}()
	<-chat.started
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("err %v", err)
	}
	mt, _, err := repo.Meeting(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	if mt.Status != store.MeetingInterrupted || mt.Audio == "" {
		t.Fatalf("status %s, audio %q", mt.Status, mt.Audio)
	}
	if _, err := os.Stat(filepath.Join(m.cfg.Dir, mt.Audio, "system.wav")); err != nil {
		t.Fatalf("the interrupted import lost its WAV: %v", err)
	}
	m.cfg.LLM = func(context.Context, string) (SummaryChat, error) { return &fixingChat{}, nil }
	if err := m.Reprocess(context.Background(), id); err != nil {
		t.Fatal(err)
	}
	mt, _, err = repo.Meeting(context.Background(), id)
	if err != nil || mt.Status != store.MeetingDone || mt.Title != "T" || mt.Summary == "" {
		t.Fatalf("after Continue: status %s, title %q, summary %q, err %v", mt.Status, mt.Title, mt.Summary, err)
	}
}

// A cancel that lands after the pass but before the summary starts must
// not leave the pass's early done status behind either.
func TestFinishImportCancelBeforeSummary(t *testing.T) {
	m, repo := importFixture(t, &fixingChat{}, &importDiarizer{}, &importFileASR{})
	mt, err := repo.CreateMeeting(context.Background(), store.Meeting{
		ID: "early-cancel", CreatedAt: time.Now(), Status: store.MeetingDone, Mode: store.MeetingModeImport,
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := m.finishImport(ctx, mt, true); !errors.Is(err, context.Canceled) {
		t.Fatalf("err %v", err)
	}
	updated, _, err := repo.Meeting(context.Background(), mt.ID)
	if err != nil || updated.Status != store.MeetingInterrupted {
		t.Fatalf("status %s, err %v", updated.Status, err)
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

// MTD alone can import: the selected meeting model missing (or without
// file recognition) must not stop the import before MTD runs.
func TestImportWorksWithMTDAlone(t *testing.T) {
	for _, tc := range []struct {
		name string
		asr  func(context.Context, Source) (TrackASR, error)
	}{
		{"model not downloaded", func(context.Context, Source) (TrackASR, error) {
			return TrackASR{}, errors.New("model not installed")
		}},
		{"no file recognition", func(context.Context, Source) (TrackASR, error) {
			return TrackASR{Caps: asr.Caps{}}, nil
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			chat := &fixingChat{}
			diarizer := &importDiarizer{result: asr.Result{Segments: []asr.SpeakerSegment{
				{Start: 0, End: time.Second, Speaker: "S01", Text: "hi"},
			}}}
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
				ASR:      tc.asr,
				Diarizer: func(context.Context) (asr.Diarizer, error) { return diarizer, nil },
				LLM:      func(context.Context, string) (SummaryChat, error) { return chat, nil },
			})
			id, err := m.Import(ctx, ImportRequest{Name: "mtd-only", CreatedAt: time.Now(), PCM: importPCM(1)}, nil)
			if err != nil {
				t.Fatal(err)
			}
			if len(diarizer.got()) != 1 {
				t.Fatalf("MTD requests %d", len(diarizer.got()))
			}
			if mt, _, err := s.Meetings().Meeting(ctx, id); err != nil || mt.Status != store.MeetingDone {
				t.Fatalf("status %v, %v", mt, err)
			}
		})
	}
}

// The "skip the live final pass" setting is for recorded meetings: an
// import's only pass still runs with it on.
func TestImportIgnoresNoFinalPass(t *testing.T) {
	chat := &fixingChat{}
	diarizer := &importDiarizer{result: asr.Result{Segments: []asr.SpeakerSegment{
		{Start: 0, End: time.Second, Speaker: "S01", Text: "hi"},
	}}}
	plain := &importFileASR{}
	m, repo := importFixture(t, chat, diarizer, plain)
	m.cfg.Settings = func() settings.Meeting { return settings.Meeting{NoFinalPass: true} }
	ctx := context.Background()
	id, err := m.Import(ctx, ImportRequest{Name: "x", CreatedAt: time.Now(), PCM: importPCM(1)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(diarizer.got()) != 1 {
		t.Fatalf("the pass did not run: %d MTD requests", len(diarizer.got()))
	}
	mt, segs, err := repo.Meeting(ctx, id)
	if err != nil || mt.Status != store.MeetingDone || len(segs) == 0 {
		t.Fatalf("status %v, %d segments, %v", mt, len(segs), err)
	}
}

// Continue processing after a cancel finishes the correction and the
// summary too, not only the transcription.
func TestReprocessCorrectsAndSummarizes(t *testing.T) {
	chat := &fixingChat{}
	block := make(chan struct{})
	diarizer := &importDiarizer{block: block, result: asr.Result{Segments: []asr.SpeakerSegment{
		{Start: 0, End: time.Second, Speaker: "S01", Text: "Box one"},
	}}}
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
	<-block
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("err %v", err)
	}
	diarizer.block = nil
	if err := m.Reprocess(context.Background(), id); err != nil {
		t.Fatal(err)
	}
	if len(chat.reqs) != 2 {
		t.Fatalf("language model requests %d, want the correction and the summary", len(chat.reqs))
	}
	mt, segs, err := repo.Meeting(context.Background(), id)
	if err != nil || mt.Status != store.MeetingDone || len(segs) != 1 || segs[0].Text != "Voxt one" {
		t.Fatalf("status %v, segments %+v, %v", mt, segs, err)
	}
	if mt.Title != "T" || mt.Summary == "" {
		t.Fatalf("no summary: title %q", mt.Title)
	}
}

// Continue processing after a cancel runs the pass even with "skip the
// live final pass" on: an import's only pass is this one, and the
// meeting ends done, not finishing forever.
func TestReprocessImportIgnoresNoFinalPass(t *testing.T) {
	chat := &fixingChat{}
	block := make(chan struct{})
	diarizer := &importDiarizer{block: block, result: asr.Result{Segments: []asr.SpeakerSegment{
		{Start: 0, End: time.Second, Speaker: "S01", Text: "Box one"},
	}}}
	plain := &importFileASR{}
	m, repo := importFixture(t, chat, diarizer, plain)
	m.cfg.Settings = func() settings.Meeting { return settings.Meeting{NoFinalPass: true} }
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	var id string
	go func() {
		var err error
		id, err = m.Import(ctx, ImportRequest{Name: "long", CreatedAt: time.Now(), PCM: importPCM(1)}, nil)
		done <- err
	}()
	<-block
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("err %v", err)
	}
	diarizer.block = nil
	if err := m.Reprocess(context.Background(), id); err != nil {
		t.Fatal(err)
	}
	if len(diarizer.got()) != 2 {
		t.Fatalf("the pass did not run again: %d MTD requests", len(diarizer.got()))
	}
	mt, segs, err := repo.Meeting(context.Background(), id)
	if err != nil || mt.Status != store.MeetingDone || len(segs) != 1 {
		t.Fatalf("status %v, %d segments, %v", mt, len(segs), err)
	}
}

// Continue processing tells the caller when the correction or the
// summary fails, as the import does; the meeting itself is done.
func TestReprocessSurfacesCorrectionFailure(t *testing.T) {
	chat := &fixingChat{}
	diarizer := &importDiarizer{result: asr.Result{Segments: []asr.SpeakerSegment{
		{Start: 0, End: time.Second, Speaker: "S01", Text: "hi"},
	}}}
	plain := &importFileASR{}
	m, repo := importFixture(t, chat, diarizer, plain)
	ctx := context.Background()
	id, err := m.Import(ctx, ImportRequest{Name: "x", CreatedAt: time.Now(), PCM: importPCM(1)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	m.cfg.LLM = func(context.Context, string) (SummaryChat, error) { return nil, errors.New("model down") }
	if err := m.Reprocess(ctx, id); err == nil {
		t.Fatal("a failed correction came back as a success")
	}
	mt, _, err := repo.Meeting(ctx, id)
	if err != nil || mt.Status != store.MeetingDone {
		t.Fatalf("status %v, %v", mt, err)
	}
}

// Keep a history off refuses the import instead of storing the whole
// transcript anyway.
func TestImportRefusedWithoutHistory(t *testing.T) {
	chat := &fixingChat{}
	diarizer := &importDiarizer{}
	plain := &importFileASR{}
	m, repo := importFixture(t, chat, diarizer, plain)
	m.cfg.Privacy = func() settings.Privacy { return settings.Privacy{} }
	_, err := m.Import(context.Background(), ImportRequest{Name: "x", CreatedAt: time.Now(), PCM: importPCM(1)}, nil)
	if !errors.Is(err, ErrNoHistory) {
		t.Fatalf("err %v", err)
	}
	if ms, _ := repo.ListMeetings(context.Background(), store.MeetingQuery{}); len(ms) != 0 {
		t.Fatalf("%d meetings stored", len(ms))
	}
}

// A failed correction or summary is the import's error, so it surfaces
// instead of looking like a success; the meeting stays for a retry.
func TestImportCorrectionFailureSurfaces(t *testing.T) {
	diarizer := &importDiarizer{result: asr.Result{Segments: []asr.SpeakerSegment{
		{Start: 0, End: time.Second, Speaker: "S01", Text: "hi"},
	}}}
	plain := &importFileASR{}
	m, repo := importFixture(t, &fixingChat{}, diarizer, plain)
	m.cfg.LLM = func(context.Context, string) (SummaryChat, error) { return nil, errors.New("no language model") }
	id, err := m.Import(context.Background(), ImportRequest{Name: "x", CreatedAt: time.Now(), PCM: importPCM(1)}, nil)
	if err == nil {
		t.Fatal("the failed correction looked like a success")
	}
	mt, _, err := repo.Meeting(context.Background(), id)
	if err != nil {
		t.Fatalf("the meeting is gone: %v", err)
	}
	if mt.Status != store.MeetingDone {
		t.Fatalf("a correction failure left status %s, want done for the Correct action", mt.Status)
	}
}
