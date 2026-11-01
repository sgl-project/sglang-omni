package main

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/local"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// histRetry is a failed entry whose recording was kept for a retry.
const histRetry = "01HIST0000000000000000RTY1"

// seedRetryEntry writes histRetry and its recording (a second of tone):
// recognition took too long on a long dictation, the recording kept for
// a week.
func seedRetryEntry(t *testing.T, a *app) store.Entry {
	t.Helper()
	hs := a.lib.store.History()
	pcm := make([]byte, asr.SampleRate*2)
	for i := 0; i < len(pcm); i += 2 {
		pcm[i+1] = byte(int8(40 * ((i/64)%2*2 - 1)))
	}
	if err := os.MkdirAll(hs.AudioDir(), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(hs.AudioDir(), histRetry+".wav"), asr.WAV(pcm, asr.SampleRate), 0o644); err != nil {
		t.Fatal(err)
	}
	e := store.Entry{
		ID: histRetry, Kind: store.KindTranscribe, CreatedAt: time.Now().Add(-2 * time.Minute), Status: store.StatusFailed,
		AppBundle: "com.apple.Notes", AppName: "Notes", Error: "asr : timeout: no result within 5m51s (4m51s of audio)",
		ASRProvider: "asr.accurate", ASRModel: "asr.accurate", AudioMS: 291500, ASRMS: 351000, TotalMS: 352000,
		Audio: histRetry + ".wav", AudioUntil: time.Date(2026, 10, 24, 20, 54, 0, 0, time.Local),
	}
	if err := hs.Put(context.Background(), e); err != nil {
		t.Fatal(err)
	}
	return e
}

// retryApp is a test app whose speech model writes text, and the
// transcriptions it got.
func retryApp(t *testing.T, text string) (*app, *ui.Tester, *fakeTranscriptions) {
	t.Helper()
	tr := &fakeTranscriptions{text: text}
	old := newLocalServices
	newLocalServices = func(*app) local.Services { return &fakeLocal{ASRP: tr} }
	t.Cleanup(func() { newLocalServices = old })
	a, _, tt := newTestApp(t)
	a.updateSettings(func(s *settings.Settings) { s.Features.Transcribe.Enhance = false })
	return a, tt, tr
}

// TestHistoryRetryRow: a failed entry with a recording has Retry on its
// row in place of Failed; one without keeps Failed, which says why. Retry
// makes it again in place: its text, status and time, the recording kept
// only for it gone.
func TestHistoryRetryRow(t *testing.T) {
	a, tt, tr := retryApp(t, "Recovered dictation.")
	seedHistory(t, a)
	seedRetryEntry(t, a)
	a.navigate("/history")
	frames(tt)
	if !hasExact(tt, "Retry") || !hasExact(tt, "Failed") {
		t.Fatalf("marks: %q", firstTexts(tt))
	}
	// While it runs, the row says what it does.
	a.hist.retry = entryRetry{id: histRetry, stage: session.Composing}
	if !has(tt, "Cleaning up…") {
		t.Errorf("no progress on the row: %q", firstTexts(tt))
	}
	a.hist.retry = entryRetry{}
	click(t, tt, "Retry")
	if tr.count() != 1 {
		t.Fatalf("%d transcriptions", tr.count())
	}
	a.loadHistory()
	if !has(tt, "Recovered dictation.") || hasExact(tt, "Retry") || !hasPrefix(tt, "Recognised again in") {
		t.Fatalf("after Retry: %q", firstTexts(tt))
	}
	e, err := a.lib.store.History().Get(context.Background(), histRetry)
	if err != nil || e.Status != store.StatusDone || e.Result != "Recovered dictation." || e.Error != "" || e.Audio != "" || !e.AudioUntil.IsZero() {
		t.Fatalf("entry %+v %v", e, err)
	}
	if _, err := os.Stat(filepath.Join(a.lib.store.History().AudioDir(), histRetry+".wav")); !os.IsNotExist(err) {
		t.Errorf("the recording kept for the retry is still there: %v", err)
	}
	if len(a.hist.list) != 5 {
		t.Errorf("entries %v: a retry adds none", historyIDs(a))
	}
}

// TestHistoryRetryDetail: the details of a failed entry with a recording
// say how long the recording is kept and offer Retry; a done entry with
// a recording offers Transcribe again; a failed one without, nothing.
// The menu has the same action.
func TestHistoryRetryDetail(t *testing.T) {
	a, tt, tr := retryApp(t, "Again.")
	seedHistory(t, a)
	seedRetryEntry(t, a)
	a.navigate("/history")
	retries := func() int {
		frames(tt)
		n := 0
		for _, s := range tt.Texts() {
			if s == "Retry" {
				n++
			}
		}
		return n
	}
	row := retries()
	a.hist.sel = histRetry
	frames(tt)
	if !hasPrefix(tt, "Recording kept because recognition failed; deleted after") {
		t.Errorf("no retention line: %q", tt.Texts())
	}
	if n := retries(); row == 0 || n != 2*row {
		t.Errorf("Retry %d times on the row, %d with the details: want the details' too", row, n)
	}
	got := openMenu(t, tt, "asr : timeout: no result within 5m51s (4m51s of audio)")
	if !slices.Contains(got, "Retry") {
		t.Fatalf("menu %q", got)
	}
	choose(t, tt, "Retry")
	if tr.count() != 1 {
		t.Fatalf("%d transcriptions", tr.count())
	}

	// A done entry: Transcribe again; its file is gone here, which it says.
	a.hist.sel = histInserted
	frames(tt)
	if got := openMenu(t, tt, "Let's move the design review to Thursday afternoon."); !slices.Contains(got, "Transcribe again") {
		t.Errorf("menu %q", got)
	}
	tt.CloseMenu()
	click(t, tt, "Transcribe again")
	if !has(tt, "The recording is gone.") || tr.count() != 1 {
		t.Errorf("a missing file: %q, %d transcriptions", firstTexts(tt), tr.count())
	}
	if e, _ := a.lib.store.History().Get(context.Background(), histInserted); e.Status != store.StatusDone || e.Result == "Again." {
		t.Errorf("the entry changed: %+v", e)
	}

	a.hist.sel = histFailed
	frames(tt)
	if hasExact(tt, "Retry") || hasExact(tt, "Transcribe again") {
		t.Errorf("a failed entry without a recording offers a retry: %q", firstTexts(tt))
	}
	for _, s := range tt.Texts() {
		if strings.HasPrefix(s, "Recording kept") {
			t.Errorf("retention line without a recording: %q", s)
		}
	}
}

// TestRetryNotice: a failed recognition whose entry kept the recording
// offers Retry in the overlay's notice: it opens the entry in History and
// retries it.
func TestRetryNotice(t *testing.T) {
	a, tt, tr := retryApp(t, "From the notice.")
	e := seedRetryEntry(t, a)
	text, _, _ := sessionProblemText(&session.Problem{Code: session.ProblemASR, Err: &asr.Error{Kind: asr.KindTimeout}})
	a.notice = &sessionNotice{text: text, action: actionRetry, entry: e.ID}
	if got := noticeActionLabel(actionRetry); got != "Retry" {
		t.Errorf("label %q", got)
	}
	a.noticeAction(actionRetry)
	frames(tt)
	if a.router.Path() != "/history" || a.hist.sel != histRetry || tr.count() != 1 {
		t.Fatalf("path %s, selected %q, %d transcriptions", a.router.Path(), a.hist.sel, tr.count())
	}
	if got, _ := a.lib.store.History().Get(context.Background(), histRetry); got.Result != "From the notice." {
		t.Errorf("entry %+v", got)
	}
}

// snapshotHistoryRetry saves History with a failed entry that kept its
// recording (Retry on its row) above one that did not (Failed), and its
// details; then the entry goes, so that the later snapshots are as before.
func snapshotHistoryRetry(a *app, save func(string)) {
	hs := a.lib.store.History()
	e := store.Entry{
		ID: histRetry, Kind: store.KindTranscribe, CreatedAt: time.Now().Add(-2 * time.Minute), Status: store.StatusFailed,
		AppBundle: "com.apple.Notes", AppName: "Notes", Error: "asr : timeout: no result within 5m51s (4m51s of audio)",
		ASRProvider: "asr.accurate", ASRModel: "asr.accurate", AudioMS: 291500, ASRMS: 351000, TotalMS: 352000,
		Audio: histRetry + ".wav", AudioUntil: time.Date(2026, 10, 24, 20, 54, 0, 0, time.Local),
	}
	_ = hs.Put(context.Background(), e)
	a.loadHistory()
	save("history-retry")
	a.hist.sel = histRetry
	save("history-retry-detail")
	a.hist.sel = ""
	_ = hs.Delete(context.Background(), histRetry)
	a.historyChanged()
	a.loadHistory()
}
