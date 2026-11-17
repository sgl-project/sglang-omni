package main

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// importLocal serves the meeting's speech model (fakeTranscriptions) and
// MTD (importDiarizeFake), both installed.
type importLocal struct {
	*fakeLocal
	mtd asr.Provider
}

func (l *importLocal) Installed(kind local.Kind) []local.Model {
	out := l.fakeLocal.Installed(kind)
	if kind == local.ASR && l.mtd != nil {
		out = append(out, local.Model{Kind: local.ASR, ID: models.ASRMoss})
	}
	return out
}

func (l *importLocal) ASR(ctx context.Context, id string) (asr.Provider, local.Model, error) {
	if id == models.ASRMoss && l.mtd != nil {
		return l.mtd, local.Model{Kind: local.ASR, ID: models.ASRMoss}, nil
	}
	return l.fakeLocal.ASR(ctx, id)
}

// importDiarizeFake is MTD on a two-speaker file.
type importDiarizeFake struct{ fakeTranscriptions }

func (f *importDiarizeFake) Diarize(context.Context, asr.FileRequest) (asr.Result, error) {
	return asr.Result{Segments: []asr.SpeakerSegment{
		{Start: 0, End: time.Second, Speaker: "S01", Text: "welcome everyone"},
		{Start: time.Second, End: 2 * time.Second, Speaker: "S02", Text: "thanks for having me"},
	}}, nil
}

// Import recording… turns a file into a meeting: MTD labels its speakers,
// the correction and the summary each ask the language model once, and
// the meeting keeps the file's name, date and duration.
func TestMeetingImportLabelsSpeakersAndSummarizes(t *testing.T) {
	a, _, tt := newTestApp(t)
	chat := &fakeChat{texts: []string{
		"[]",
		`{"transcript_summary": {"title": "Weekly sync", "content": "Two speakers met."}, "todo_list": []}`,
	}}
	a.local = &importLocal{fakeLocal: withChat(&fakeLocal{ASRP: &fakeTranscriptions{}}, chat), mtd: &importDiarizeFake{}}
	wav := filepath.Join(t.TempDir(), "zoom-standup.wav")
	pcm := make([]byte, 2*asr.SampleRate*3)
	if err := os.WriteFile(wav, asr.WAV(pcm, asr.SampleRate), 0o644); err != nil {
		t.Fatal(err)
	}
	at := time.Date(2026, 11, 10, 9, 30, 0, 0, time.Local)
	if err := os.Chtimes(wav, at, at); err != nil {
		t.Fatal(err)
	}
	savedC, savedD := chooseRecordingFile, decodeRecording
	t.Cleanup(func() { chooseRecordingFile, decodeRecording = savedC, savedD })
	chooseRecordingFile = func(a *app, then func(string)) { then(wav) }
	decodeRecording = func(context.Context, string) ([]byte, error) { return pcm, nil }
	a.prefs.Features.Meeting.Lang = "zh"
	tt.SetSize(windowWidth, 900)
	a.navigate("/meetings")
	frames(tt)
	if err := tt.ClickControl("Import recording…"); err != nil {
		t.Fatalf("%v (%q)", err, firstTexts(tt))
	}
	frames(tt)
	repo := a.lib.store.Meetings()
	list, err := repo.ListMeetings(context.Background(), store.MeetingQuery{})
	if err != nil || len(list) != 1 {
		t.Fatalf("meetings %+v %v", list, err)
	}
	mt := list[0]
	if mt.Mode != store.MeetingModeImport || mt.Status != store.MeetingDone {
		t.Fatalf("mode %s, status %s", mt.Mode, mt.Status)
	}
	if mt.Title != "Weekly sync" {
		t.Fatalf("title %q: the summary did not name the meeting", mt.Title)
	}
	if mt.CreatedAt.UnixMilli() != at.UnixMilli() || mt.DurationMS != 3000 {
		t.Fatalf("date %v, duration %d ms: not the file's", mt.CreatedAt, mt.DurationMS)
	}
	_, segs, err := repo.Meeting(context.Background(), mt.ID)
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]string{}
	for _, s := range segs {
		got[s.Speaker] = s.Text
	}
	if got["s1"] != "welcome everyone" || got["s2"] != "thanks for having me" || len(got) != 2 {
		t.Fatalf("segments %+v", segs)
	}
	if n := len(chat.requests()); n != 2 {
		t.Fatalf("language model requests %d, want one correction and one summary", n)
	}
	if a.mv.detail == nil || a.mv.detail.id != mt.ID {
		t.Fatal("the imported meeting did not open")
	}
	frames(tt)
	if _, ok := tt.Find("Speaker 1"); !ok {
		t.Fatalf("no speaker labels in the details (%q)", firstTexts(tt))
	}
}

// Without MTD, Import recording… says so and opens Speech models; no file
// is asked for and no meeting is made (an import must never fall back to
// plain transcription: the speakers would be lost silently).
func TestMeetingImportExplainsMissingMTD(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.local = &fakeLocal{ASRP: &fakeTranscriptions{}}
	saved := chooseRecordingFile
	t.Cleanup(func() { chooseRecordingFile = saved })
	chooseRecordingFile = func(a *app, then func(string)) { t.Error("the file dialog opened without MTD") }
	tt.SetSize(windowWidth, 900)
	a.navigate("/meetings")
	frames(tt)
	if err := tt.ClickControl("Import recording…"); err != nil {
		t.Fatalf("%v (%q)", err, firstTexts(tt))
	}
	frames(tt)
	if _, ok := tt.Find("MOSS-Transcribe-Diarize is not downloaded"); !ok {
		t.Fatalf("no MTD dialog (%q)", firstTexts(tt))
	}
	if err := tt.ClickControl("Open Speech models"); err != nil {
		t.Fatal(err)
	}
	frames(tt)
	if a.shownPath() != modelsPath {
		t.Fatalf("went to %s", a.shownPath())
	}
	list, _ := a.lib.store.Meetings().ListMeetings(context.Background(), store.MeetingQuery{})
	if len(list) != 0 {
		t.Fatalf("%d meetings without a file", len(list))
	}
}

// With Keep a history off, Import recording… says so and offers Privacy
// settings; no file is asked for and no meeting is made.
func TestMeetingImportExplainsHistoryOff(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.local = &importLocal{fakeLocal: &fakeLocal{ASRP: &fakeTranscriptions{}}, mtd: &importDiarizeFake{}}
	a.updateSettings(func(s *settings.Settings) { s.Privacy.KeepHistory = false })
	saved := chooseRecordingFile
	t.Cleanup(func() { chooseRecordingFile = saved })
	chooseRecordingFile = func(a *app, then func(string)) { t.Error("the file dialog opened with the history off") }
	tt.SetSize(windowWidth, 900)
	a.navigate("/meetings")
	frames(tt)
	if err := tt.ClickControl("Import recording…"); err != nil {
		t.Fatalf("%v (%q)", err, firstTexts(tt))
	}
	frames(tt)
	if _, ok := tt.Find("Keep a history is off"); !ok {
		t.Fatalf("no history dialog (%q) noMTD=%v noHistory=%v keep=%v", firstTexts(tt), a.mv.importNoMTD, a.mv.importNoHistory, a.prefs.Privacy.KeepHistory)
	}
	if err := tt.ClickControl("Open Privacy settings"); err != nil {
		t.Fatal(err)
	}
	frames(tt)
	if a.shownPath() != "/settings/privacy" {
		t.Fatalf("went to %s", a.shownPath())
	}
	list, _ := a.lib.store.Meetings().ListMeetings(context.Background(), store.MeetingQuery{})
	if len(list) != 0 {
		t.Fatalf("%d meetings without a file", len(list))
	}
}

// Cancel reaches the decoder: a long afconvert stops instead of running
// on while the page shows the import as cancelled.
func TestMeetingImportDecodeCancels(t *testing.T) {
	a, _, _ := newTestApp(t)
	savedD := decodeRecording
	t.Cleanup(func() { decodeRecording = savedD })
	started := make(chan struct{})
	decodeRecording = func(ctx context.Context, _ string) ([]byte, error) {
		close(started)
		<-ctx.Done()
		return nil, ctx.Err()
	}
	ctx, cancel := context.WithCancel(context.Background())
	imp := &meetingImport{name: "long.m4a", cancel: cancel}
	done := make(chan error, 1)
	go func() {
		_, err := a.runImport(ctx, imp, "long.m4a")
		done <- err
	}()
	<-started
	imp.cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("err %v", err)
	}
}
