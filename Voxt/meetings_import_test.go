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

func decodedImportPCM(pcm []byte) func(context.Context, string, string) ([]byte, error) {
	return func(_ context.Context, _, wavPath string) ([]byte, error) {
		if err := os.WriteFile(wavPath, asr.WAV(pcm, asr.SampleRate), 0o644); err != nil {
			return nil, err
		}
		return pcm, nil
	}
}

type importTimeoutDiarizeFake struct{ importDiarizeFake }

func (f *importTimeoutDiarizeFake) Diarize(context.Context, asr.FileRequest) (asr.Result, error) {
	return asr.Result{}, context.DeadlineExceeded
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
	a.prefs.General.MainLanguage = "en"
	wav := filepath.Join(t.TempDir(), "zoom-standup.wav")
	pcm := make([]byte, 2*asr.SampleRate*3)
	for i := 0; i < len(pcm); i += 2 {
		pcm[i], pcm[i+1] = 0x10, 0x7f
	}
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
	decodeRecording = decodedImportPCM(pcm)
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

func TestMeetingImportExplainsLanguage(t *testing.T) {
	for _, language := range []string{"", "ko"} {
		t.Run(language, func(t *testing.T) {
			a, _, tt := newTestApp(t)
			a.local = &importLocal{fakeLocal: &fakeLocal{ASRP: &fakeTranscriptions{}}, mtd: &importDiarizeFake{}}
			a.prefs.General.MainLanguage = language
			saved := chooseRecordingFile
			t.Cleanup(func() { chooseRecordingFile = saved })
			chooseRecordingFile = func(a *app, then func(string)) {
				t.Error("the file dialog opened with an unknown or unsupported language")
			}
			tt.SetSize(windowWidth, 900)
			a.navigate("/meetings")
			frames(tt)
			if err := tt.ClickControl("Import recording…"); err != nil {
				t.Fatalf("%v (%q)", err, firstTexts(tt))
			}
			frames(tt)
			if _, ok := tt.Find("Choose the recording's language"); !ok {
				t.Fatalf("no language dialog (%q)", firstTexts(tt))
			}
			if _, ok := tt.Find("MOSS-Transcribe-Diarize is not downloaded"); ok {
				t.Fatalf("installed MTD reported as missing (%q)", firstTexts(tt))
			}
			if err := tt.ClickControl("Open Meeting settings"); err != nil {
				t.Fatal(err)
			}
			frames(tt)
			if !a.mvUI.setOpen {
				t.Fatal("Meeting settings did not open")
			}
			list, _ := a.lib.store.Meetings().ListMeetings(context.Background(), store.MeetingQuery{})
			if len(list) != 0 {
				t.Fatalf("%d meetings without a file", len(list))
			}
		})
	}
}

func TestMeetingImportLanguageFailureDoesNotSayMTDMissing(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.local = &importLocal{fakeLocal: &fakeLocal{ASRP: &fakeTranscriptions{}}, mtd: &importDiarizeFake{}}
	saved := decodeRecording
	t.Cleanup(func() { decodeRecording = saved })
	decodeRecording = decodedImportPCM(make([]byte, 2*asr.SampleRate))
	a.importRecording("unknown-language.wav")
	if !a.mvUI.importNoLanguage || a.mvUI.importNoMTD {
		t.Fatalf("language failure: noLanguage=%v, noMTD=%v", a.mvUI.importNoLanguage, a.mvUI.importNoMTD)
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
		t.Fatalf("no history dialog (%q) noMTD=%v noHistory=%v keep=%v", firstTexts(tt), a.mvUI.importNoMTD, a.mvUI.importNoHistory, a.prefs.Privacy.KeepHistory)
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
	a.prefs.General.MainLanguage = "en"
	a.prefs.Features.Meeting.NoAutoSummary = true
	a.local = &importLocal{fakeLocal: &fakeLocal{ASRP: &fakeTranscriptions{}}, mtd: &importDiarizeFake{}}
	savedD := decodeRecording
	t.Cleanup(func() { decodeRecording = savedD })
	started := make(chan struct{})
	decodeRecording = func(ctx context.Context, _, wavPath string) ([]byte, error) {
		pcm := make([]byte, 2*asr.SampleRate)
		for i := 0; i < len(pcm); i += 2 {
			pcm[i], pcm[i+1] = 0x10, 0x7f
		}
		wav := asr.WAV(pcm, asr.SampleRate)
		metadataWAV := append([]byte{}, wav[:36]...)
		metadataWAV = append(metadataWAV, []byte("FLLR\x04\x00\x00\x00\x00\x00\x00\x00")...)
		metadataWAV = append(metadataWAV, wav[36:]...)
		if err := os.WriteFile(wavPath, metadataWAV, 0o644); err != nil {
			return nil, err
		}
		close(started)
		<-ctx.Done()
		return nil, ctx.Err()
	}
	ctx, cancel := context.WithCancel(context.Background())
	imp := &meetingImport{name: "long.m4a", cancel: cancel}
	done := make(chan error, 1)
	var id string
	go func() {
		var err error
		id, err = a.runImport(ctx, imp, "long.m4a")
		done <- err
	}()
	<-started
	imp.cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("err %v", err)
	}
	if id == "" {
		t.Fatal("decode cancellation returned no interrupted meeting")
	}
	mt, _, err := a.lib.store.Meetings().Meeting(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	if mt.Status != store.MeetingInterrupted || mt.Audio == "" {
		t.Fatalf("status %s, audio %q", mt.Status, mt.Audio)
	}
	if _, err := os.Stat(filepath.Join(a.dataDir(), "meetings", mt.Audio, "system.wav")); err != nil {
		t.Fatalf("decoded audio was not kept: %v", err)
	}
	if err := a.meetMgr.Reprocess(context.Background(), id); err != nil {
		t.Fatalf("Continue processing: %v", err)
	}
	mt, _, err = a.lib.store.Meetings().Meeting(context.Background(), id)
	if err != nil || mt.Status != store.MeetingDone {
		t.Fatalf("after Continue processing: status %s, err %v", mt.Status, err)
	}
}

// An import cannot sit for hours: its whole decode, transcription and summary
// get at most thirty minutes before the meeting is left for Continue processing.
func TestMeetingImportBoundsTotalWait(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.prefs.General.MainLanguage = "en"
	saved := decodeRecording
	t.Cleanup(func() { decodeRecording = saved })
	deadlines := make(chan time.Duration, 1)
	decodeRecording = func(ctx context.Context, _, _ string) ([]byte, error) {
		deadline, _ := ctx.Deadline()
		deadlines <- time.Until(deadline)
		return nil, context.DeadlineExceeded
	}
	a.importRecording("long.m4a")
	select {
	case remaining := <-deadlines:
		if remaining <= 29*time.Minute || remaining > 30*time.Minute {
			t.Fatalf("import deadline is %v away, want at most 30 minutes", remaining)
		}
	case <-time.After(time.Second):
		t.Fatal("the decoder did not start")
	}
}

// A timeout after the meeting exists is an interruption, not a fresh failure
// toast: the reloaded list offers Continue processing with the audio kept.
func TestMeetingImportTimeoutKeepsInterruptedMeeting(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.local = &importLocal{fakeLocal: &fakeLocal{ASRP: &fakeTranscriptions{}}, mtd: &importTimeoutDiarizeFake{}}
	a.prefs.General.MainLanguage = "en"
	saved := decodeRecording
	t.Cleanup(func() { decodeRecording = saved })
	pcm := make([]byte, 2*asr.SampleRate)
	for i := 0; i < len(pcm); i += 2 {
		pcm[i], pcm[i+1] = 0x10, 0x7f
	}
	decodeRecording = decodedImportPCM(pcm)
	a.importRecording("long.wav")
	if len(a.toasts) != 0 {
		t.Fatalf("an interrupted import showed a failure: %q", a.toasts)
	}
	meetings, err := a.lib.store.Meetings().ListMeetings(context.Background(), store.MeetingQuery{})
	if err != nil || len(meetings) != 1 {
		t.Fatalf("meetings %+v, err %v", meetings, err)
	}
	if meetings[0].Status != store.MeetingInterrupted || meetings[0].Audio == "" {
		t.Fatalf("status %s, audio %q", meetings[0].Status, meetings[0].Audio)
	}
}
