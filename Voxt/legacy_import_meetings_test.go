package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/egoist/mygo/plugins/sqlite"

	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
)

func TestLegacyIsFileTranscriptHeuristic(t *testing.T) {
	mixed := legacyMeetingEntry{
		DisplayTitle: "weekly.mp3",
		TranscriptSegments: []legacyTranscriptSegment{
			{Text: "hi", AudioSource: "mixed"},
		},
	}
	if !legacyIsFileTranscript(mixed, "") {
		t.Fatal("mixed-only file import should be skipped")
	}
	live := legacyMeetingEntry{
		DisplayTitle: "Standup",
		TranscriptSegments: []legacyTranscriptSegment{
			{Text: "hi", AudioSource: "microphone"},
		},
	}
	if legacyIsFileTranscript(live, "") {
		t.Fatal("mic segment is not a file import")
	}
}

func TestMergeLegacyMeetingsLiveSession(t *testing.T) {
	ctx := context.Background()
	lib := copyFixture(t, "hist")
	dbPath := legacyDatabase(lib)
	end := 1.25
	entry := legacyMeetingEntry{
		MeetingCaptureMode: "meeting",
		DisplayTitle:       "Sprint review",
		TranscriptionModel: "mlx-community/Qwen3-ASR-0.6B",
		EnhancementModel:   "openai",
		AudioDurationSeconds: ptrF(120.0),
		TranscriptSegments: []legacyTranscriptSegment{
			{
				Speaker: "me", SpeakerID: "me", SpeakerDisplayName: "Alex",
				AudioSource: "microphone", StartSeconds: 0, EndSeconds: &end,
				Text: "Shipped the import.", TranslatedText: "",
			},
			{
				Speaker: "them", SpeakerID: "guest-1", SpeakerDisplayName: "Sam",
				AudioSource: "systemAudio", StartSeconds: 2, EndSeconds: ptrF(4.0),
				Text: "Looks good.", TranslatedText: "不错",
			},
		},
		TranscriptSummary: &legacyTranscriptSummary{
			Title: "Sprint review", Body: "Import work landed.", TodoItems: []string{"Write migration note"},
			GeneratedAt: time.Date(2026, 1, 2, 15, 0, 0, 0, time.UTC),
		},
		TranscriptSummaryChatMessages: []legacySummaryChatMessage{
			{Role: "user", Content: "What did we decide?", CreatedAt: time.Date(2026, 1, 2, 15, 5, 0, 0, time.UTC)},
		},
	}
	insertLegacyTranscript(t, dbPath, "meet1", entry, 1_700_000_000.0, "meet1/archive.wav")
	dataDir := t.TempDir()
	st, err := sqlitestore.Open(ctx, dataDir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { st.Close() })
	stats, err := mergeLegacyMeetings(ctx, st.Meetings(), dataDir, lib, dbPath)
	if err != nil {
		t.Fatal(err)
	}
	if stats.imported != 1 || stats.fileSkipped != 0 || stats.incomplete != 1 { // hist fixture row a4
		t.Fatalf("stats %+v", stats)
	}
	got, segs, err := st.Meetings().Meeting(ctx, "legacy-meet1")
	if err != nil {
		t.Fatal(err)
	}
	if got.Title != "Sprint review" || got.ASRModel == "" || got.DurationMS != 120_000 || len(segs) != 2 {
		t.Fatalf("meeting %+v segs %d", got, len(segs))
	}
	if segs[0].Speaker != "me" || segs[0].Text != "Shipped the import." || segs[0].StartMS != 0 || segs[0].EndMS != 1250 {
		t.Fatalf("seg0 %+v", segs[0])
	}
	if segs[1].Source != store.SegmentSystem || segs[1].Translation != "不错" {
		t.Fatalf("seg1 %+v", segs[1])
	}
	var speakers map[string]string
	if err := json.Unmarshal([]byte(got.Speakers), &speakers); err != nil || speakers["me"] != "Alex" || speakers["guest-1"] != "Sam" {
		t.Fatalf("speakers %q %+v", got.Speakers, speakers)
	}
	sum, ok := meeting.ParseStoredSummary(got.Summary)
	if !ok || sum.Title != "Sprint review" || sum.Body != "Import work landed." || len(sum.Todos) != 1 {
		t.Fatalf("summary %+v ok=%v", sum, ok)
	}
	stats2, err := mergeLegacyMeetings(ctx, st.Meetings(), dataDir, lib, dbPath)
	if err != nil || stats2.imported != 0 {
		t.Fatalf("second import %+v %v", stats2, err)
	}
}

func TestMergeLegacyMeetingsFileTranscriptSkipped(t *testing.T) {
	ctx := context.Background()
	lib := copyFixture(t, "hist")
	dbPath := legacyDatabase(lib)
	entry := legacyMeetingEntry{
		DisplayTitle: "recording.wav",
		TranscriptSegments: []legacyTranscriptSegment{
			{Text: "file import only", AudioSource: "mixed", StartSeconds: 0},
		},
	}
	insertLegacyTranscript(t, dbPath, "file1", entry, 1_700_000_100.0, "")
	dataDir := t.TempDir()
	st, err := sqlitestore.Open(ctx, dataDir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { st.Close() })
	stats, err := mergeLegacyMeetings(ctx, st.Meetings(), dataDir, lib, dbPath)
	if err != nil {
		t.Fatal(err)
	}
	if stats.fileSkipped != 1 {
		t.Fatalf("stats %+v", stats)
	}
	if _, _, err := st.Meetings().Meeting(ctx, "legacy-file1"); err == nil {
		t.Fatal("file transcript should not become a meeting")
	}
}

func TestMergeLegacyMeetingsTextWithoutAudio(t *testing.T) {
	ctx := context.Background()
	lib := copyFixture(t, "hist")
	dbPath := legacyDatabase(lib)
	end := 0.5
	entry := legacyMeetingEntry{
		MeetingCaptureMode: "meeting",
		TranscriptSegments: []legacyTranscriptSegment{
			{Speaker: "me", AudioSource: "microphone", StartSeconds: 0, EndSeconds: &end, Text: "No recording left."},
		},
	}
	insertLegacyTranscript(t, dbPath, "text1", entry, 1_700_000_200.0, "")
	dataDir := t.TempDir()
	st, err := sqlitestore.Open(ctx, dataDir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { st.Close() })
	stats, err := mergeLegacyMeetings(ctx, st.Meetings(), dataDir, lib, dbPath)
	if err != nil || stats.imported != 1 {
		t.Fatalf("%+v %v", stats, err)
	}
	got, _, err := st.Meetings().Meeting(ctx, "legacy-text1")
	if err != nil || got.Audio != "" {
		t.Fatalf("%+v %v", got, err)
	}
}

func TestCopyLegacyMeetingAudio(t *testing.T) {
	voxtDir := t.TempDir()
	audioRoot := filepath.Join(voxtDir, "transcription-history-audio")
	if err := os.MkdirAll(filepath.Dir(filepath.Join(audioRoot, "m1/a.wav")), 0o755); err != nil {
		t.Fatal(err)
	}
	src := filepath.Join(audioRoot, "m1/a.wav")
	if err := os.WriteFile(src, []byte("RIFFxxxx"), 0o644); err != nil {
		t.Fatal(err)
	}
	dataDir := t.TempDir()
	folder, err := copyLegacyMeetingAudio(voxtDir, dataDir, "legacy-x", "m1/a.wav", []store.MeetingSegment{
		{Source: store.SegmentMic},
	})
	if err != nil || folder != "legacy-x" {
		t.Fatalf("%q %v", folder, err)
	}
	if _, err := os.Stat(filepath.Join(dataDir, "meetings", "legacy-x", "mic.wav")); err != nil {
		t.Fatal(err)
	}
	_, err = copyLegacyMeetingAudio(voxtDir, dataDir, "legacy-y", "m1/a.wav", []store.MeetingSegment{
		{Source: store.SegmentSystem},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dataDir, "meetings", "legacy-y", "system.wav")); err != nil {
		t.Fatal(err)
	}
}

func insertLegacyTranscript(t *testing.T, dbPath string, id string, entry legacyMeetingEntry, created float64, audioRel string) {
	t.Helper()
	raw, err := json.Marshal(entry)
	if err != nil {
		t.Fatal(err)
	}
	text := ""
	for _, s := range entry.TranscriptSegments {
		if s.Text != "" {
			text = s.Text
			break
		}
	}
	ctx := context.Background()
	db, err := sqlite.Open(ctx, dbPath, sqlite.OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	q := `INSERT INTO history_entries (
		id, text, createdAt, kind, transcriptionEngine, transcriptionModel,
		enhancementMode, enhancementModel, isTranslation, entryJSON,
		displayTitle, transcriptAudioRelativePath
	) VALUES (?, ?, ?, 'transcript', 'omni', ?, 'off', ?, 0, ?, ?, ?)
	ON CONFLICT(id) DO UPDATE SET text=excluded.text, entryJSON=excluded.entryJSON,
		displayTitle=excluded.displayTitle, transcriptAudioRelativePath=excluded.transcriptAudioRelativePath`
	_, err = db.Execute(ctx, q, id, text, created,
		entry.TranscriptionModel, entry.EnhancementModel, string(raw),
		entry.DisplayTitle, audioRel)
	if err != nil {
		t.Fatal(err)
	}
}

func ptrF(v float64) *float64 { return &v }
