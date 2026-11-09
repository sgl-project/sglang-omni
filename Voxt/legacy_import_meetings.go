package main

// Importing 1.x meeting transcripts from history_entries into meetings and
// meeting_segments (legacy_import.go). File-import transcripts stay out; see
// legacyIsFileTranscript.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/egoist/mygo/plugins/sqlite"

	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type legacyMeetingStats struct {
	imported, fileSkipped, incomplete int
}

type legacyTranscriptSegment struct {
	Speaker            string   `json:"speaker"`
	SpeakerID          string   `json:"speakerID"`
	SpeakerDisplayName string   `json:"speakerDisplayName"`
	AudioSource        string   `json:"audioSource"`
	StartSeconds       float64  `json:"startSeconds"`
	EndSeconds         *float64 `json:"endSeconds"`
	Text               string   `json:"text"`
	TranslatedText     string   `json:"translatedText"`
}

type legacyTranscriptSummary struct {
	Title       string    `json:"title"`
	Body        string    `json:"body"`
	Content     string    `json:"content"`
	TodoItems   []string  `json:"todoItems"`
	GeneratedAt time.Time `json:"generatedAt"`
}

type legacySummaryChatMessage struct {
	Role      string    `json:"role"`
	Content   string    `json:"content"`
	CreatedAt time.Time `json:"createdAt"`
}

type legacyMeetingEntry struct {
	TranscriptSegments             []legacyTranscriptSegment    `json:"transcriptSegments"`
	MeetingCaptureMode             string                       `json:"meetingCaptureMode"`
	DisplayTitle                   string                       `json:"displayTitle"`
	TranscriptSummary              *legacyTranscriptSummary     `json:"transcriptSummary"`
	TranscriptSummaryChatMessages  []legacySummaryChatMessage   `json:"transcriptSummaryChatMessages"`
	TranscriptionChatMessages      []legacySummaryChatMessage   `json:"transcriptionChatMessages"`
	TranscriptAudioRelativePath    string                       `json:"transcriptAudioRelativePath"`
	AudioRelativePath              string                       `json:"audioRelativePath"`
	TranscriptionModel             string                       `json:"transcriptionModel"`
	EnhancementModel               string                       `json:"enhancementModel"`
	AudioDurationSeconds           *float64                     `json:"audioDurationSeconds"`
}

type legacyMeetingRow struct {
	id, text, displayTitle, entryJSON                         string
	audioRel, transcriptAudioRel, transcriptionModel, llmModel string
	createdAt, audioDuration                                   float64
}

// mergeLegacyMeetings imports transcript history rows into the meetings store.
// Ids are legacy-<old id>; an existing meeting is left unchanged (idempotent).
func mergeLegacyMeetings(ctx context.Context, repo store.Meetings, dataDir, voxtDir, dbPath string) (legacyMeetingStats, error) {
	rows, err := readLegacyMeetingRows(ctx, dbPath)
	if err != nil {
		return legacyMeetingStats{}, err
	}
	var stats legacyMeetingStats
	for _, row := range rows {
		entry := legacyMeetingEntry{}
		if strings.TrimSpace(row.entryJSON) != "" {
			_ = json.Unmarshal([]byte(row.entryJSON), &entry)
		}
		if row.displayTitle != "" && entry.DisplayTitle == "" {
			entry.DisplayTitle = row.displayTitle
		}
		if row.transcriptionModel != "" && entry.TranscriptionModel == "" {
			entry.TranscriptionModel = row.transcriptionModel
		}
		if row.llmModel != "" && entry.EnhancementModel == "" {
			entry.EnhancementModel = row.llmModel
		}
		if entry.AudioDurationSeconds == nil && row.audioDuration > 0 {
			entry.AudioDurationSeconds = &row.audioDuration
		}
		audioRel := cmpNonEmpty(entry.TranscriptAudioRelativePath, entry.AudioRelativePath, row.transcriptAudioRel, row.audioRel)
		if legacyIsFileTranscript(entry, row.displayTitle) {
			stats.fileSkipped++
			continue
		}
		if !legacyIsLiveMeeting(entry, row, audioRel) {
			stats.incomplete++
			continue
		}
		meetID := "legacy-" + row.id
		if _, _, err := repo.Meeting(ctx, meetID); err == nil {
			continue
		} else if !errors.Is(err, core.ErrNotFound) {
			return stats, err
		}
		m, segs, err := legacyStoreMeeting(row, entry, meetID, audioRel)
		if err != nil {
			return stats, err
		}
		if audioRel != "" {
			if folder, err := copyLegacyMeetingAudio(voxtDir, dataDir, meetID, audioRel, segs); err != nil {
				log.Printf("legacy import: meeting %s audio: %v", meetID, err)
			} else if folder != "" {
				m.Audio = folder
			}
		}
		created, err := repo.CreateMeeting(ctx, m)
		if err != nil {
			return stats, fmt.Errorf("meeting %s: %w", meetID, err)
		}
		if len(segs) > 0 {
			if err := repo.AppendSegments(ctx, created.Seq, segs); err != nil {
				return stats, err
			}
		}
		stats.imported++
	}
	return stats, nil
}

func readLegacyMeetingRows(ctx context.Context, path string) ([]legacyMeetingRow, error) {
	db, err := sqlite.Open(ctx, path, sqlite.OpenOptions{ReadOnly: true})
	if err != nil {
		return nil, err
	}
	defer db.Close()
	rows, err := db.Query(ctx, `SELECT id, text, createdAt, COALESCE(displayTitle, ''), COALESCE(transcriptionModel, ''),
		COALESCE(enhancementModel, ''), COALESCE(audioDurationSeconds, 0), COALESCE(audioRelativePath, ''),
		COALESCE(transcriptAudioRelativePath, ''), COALESCE(entryJSON, '')
		FROM history_entries WHERE kind = 'transcript' ORDER BY createdAt`)
	if err != nil {
		if strings.Contains(err.Error(), "no such table") {
			return nil, nil
		}
		return nil, err
	}
	out := make([]legacyMeetingRow, 0, len(rows.Values))
	for _, v := range rows.Values {
		out = append(out, legacyMeetingRow{
			id:                 fmt.Sprint(v[0]),
			text:               fmt.Sprint(v[1]),
			createdAt:          number(v[2]),
			displayTitle:       fmt.Sprint(v[3]),
			transcriptionModel: fmt.Sprint(v[4]),
			llmModel:           fmt.Sprint(v[5]),
			audioDuration:      number(v[6]),
			audioRel:           fmt.Sprint(v[7]),
			transcriptAudioRel: fmt.Sprint(v[8]),
			entryJSON:          fmt.Sprint(v[9]),
		})
	}
	return out, nil
}

// shortcut: file imports use displayTitle plus mixed-only segments only; user-set
// titles on live meetings could be skipped — upgrade if that shows up in the field.
func legacyIsFileTranscript(entry legacyMeetingEntry, columnTitle string) bool {
	title := strings.TrimSpace(firstNonEmpty(entry.DisplayTitle, columnTitle))
	if title == "" || len(entry.TranscriptSegments) == 0 {
		return false
	}
	for _, s := range entry.TranscriptSegments {
		src := strings.TrimSpace(s.AudioSource)
		if src != "" && src != "mixed" {
			return false
		}
	}
	return true
}

func legacyIsLiveMeeting(entry legacyMeetingEntry, row legacyMeetingRow, audioRel string) bool {
	if len(legacyMeaningfulSegments(entry.TranscriptSegments)) > 0 {
		return true
	}
	if entry.TranscriptSummary != nil && (strings.TrimSpace(entry.TranscriptSummary.Title) != "" ||
		strings.TrimSpace(firstNonEmpty(entry.TranscriptSummary.Body, entry.TranscriptSummary.Content)) != "") {
		return true
	}
	if len(entry.TranscriptSummaryChatMessages) > 0 || len(entry.TranscriptionChatMessages) > 0 {
		return true
	}
	if audioRel != "" && strings.TrimSpace(firstNonEmpty(entry.MeetingCaptureMode, "meeting")) != "" {
		return true
	}
	return false
}

func legacyStoreMeeting(row legacyMeetingRow, entry legacyMeetingEntry, id, audioRel string) (store.Meeting, []store.MeetingSegment, error) {
	created := time.Unix(0, int64(row.createdAt*1e9))
	mode := store.MeetingModeMeeting
	switch strings.TrimSpace(entry.MeetingCaptureMode) {
	case "subtitles":
		mode = store.MeetingModeSubtitles
	case "recording":
		mode = store.MeetingModeRecording
	}
	title := strings.TrimSpace(firstNonEmpty(entry.DisplayTitle, row.displayTitle))
	if title == "" && entry.TranscriptSummary != nil {
		title = strings.TrimSpace(entry.TranscriptSummary.Title)
	}
	durMS := int64(0)
	if entry.AudioDurationSeconds != nil {
		durMS = int64(*entry.AudioDurationSeconds * 1000)
	} else if row.audioDuration > 0 {
		durMS = int64(row.audioDuration * 1000)
	}
	speakers := legacySpeakerMap(entry.TranscriptSegments)
	speakersJSON, _ := json.Marshal(speakers)
	summaryJSON := legacySummaryJSON(entry.TranscriptSummary)
	chatJSON := legacyChatJSON(entry)
	m := store.Meeting{
		ID: id, CreatedAt: created, EndedAt: created, Status: store.MeetingDone, Mode: mode, Title: title,
		DurationMS: durMS, ASRModel: strings.TrimSpace(entry.TranscriptionModel),
		LLMModel: strings.TrimSpace(entry.EnhancementModel), Speakers: string(speakersJSON),
		Summary: summaryJSON, Chat: chatJSON,
	}
	segs := legacySegments(entry.TranscriptSegments)
	return m, segs, nil
}

func legacyMeaningfulSegments(segs []legacyTranscriptSegment) []legacyTranscriptSegment {
	var out []legacyTranscriptSegment
	for _, s := range segs {
		if strings.TrimSpace(s.Text) != "" {
			out = append(out, s)
		}
	}
	return out
}

func legacySpeakerMap(segs []legacyTranscriptSegment) map[string]string {
	names := map[string]string{}
	for _, s := range segs {
		key := legacySegmentSpeakerKey(s)
		if dn := strings.TrimSpace(s.SpeakerDisplayName); dn != "" {
			names[key] = dn
		}
	}
	return names
}

func legacySegmentSpeakerKey(s legacyTranscriptSegment) string {
	if id := strings.TrimSpace(s.SpeakerID); id != "" {
		return id
	}
	if sp := strings.TrimSpace(s.Speaker); sp != "" {
		return sp
	}
	return "them"
}

func legacySegments(segs []legacyTranscriptSegment) []store.MeetingSegment {
	out := make([]store.MeetingSegment, 0, len(segs))
	for _, s := range legacyMeaningfulSegments(segs) {
		startMS := int64(s.StartSeconds * 1000)
		endMS := startMS + 1000
		if s.EndSeconds != nil {
			endMS = int64(*s.EndSeconds * 1000)
		}
		if endMS < startMS {
			endMS = startMS
		}
		src := store.SegmentMic
		switch strings.TrimSpace(s.AudioSource) {
		case "systemAudio":
			src = store.SegmentSystem
		case "microphone":
			src = store.SegmentMic
		case "mixed":
			src = store.SegmentMic
		}
		if s.AudioSource == "" && strings.TrimSpace(s.Speaker) == "them" {
			src = store.SegmentSystem
		}
		tr := strings.TrimSpace(s.TranslatedText)
		out = append(out, store.MeetingSegment{
			Idx: -1, Source: src, Speaker: legacySegmentSpeakerKey(s),
			StartMS: startMS, EndMS: endMS, Text: strings.TrimSpace(s.Text),
			Translation: tr, Final: true,
		})
	}
	return out
}

func legacySummaryJSON(s *legacyTranscriptSummary) string {
	if s == nil {
		return ""
	}
	body := strings.TrimSpace(firstNonEmpty(s.Body, s.Content))
	title := strings.TrimSpace(s.Title)
	if title == "" && body == "" && len(s.TodoItems) == 0 {
		return ""
	}
	sum := meeting.Summary{Title: title, Body: body, Todos: s.TodoItems}
	if !s.GeneratedAt.IsZero() {
		sum.GeneratedAt = s.GeneratedAt
	}
	raw, err := sum.EncodeJSON()
	if err != nil {
		return ""
	}
	return raw
}

func legacyChatJSON(entry legacyMeetingEntry) string {
	msgs := entry.TranscriptSummaryChatMessages
	if len(msgs) == 0 {
		msgs = entry.TranscriptionChatMessages
	}
	if len(msgs) == 0 {
		return "[]"
	}
	type stored struct {
		Role      string `json:"role"`
		Content   string `json:"content"`
		CreatedAt int64  `json:"created_at"`
	}
	var out []stored
	for _, m := range msgs {
		c := strings.TrimSpace(m.Content)
		if c == "" {
			continue
		}
		at := m.CreatedAt
		if at.IsZero() {
			at = time.Now()
		}
		out = append(out, stored{Role: strings.TrimSpace(m.Role), Content: c, CreatedAt: at.UnixMilli()})
	}
	if len(out) == 0 {
		return "[]"
	}
	b, _ := json.Marshal(out)
	return string(b)
}

func legacyHistoryAudioRoot(voxtDir string) string {
	custom := strings.TrimSpace(legacyPreference(voxtDir, "historyAudioStorageRootPath"))
	if custom != "" {
		return custom
	}
	return filepath.Join(voxtDir, "transcription-history-audio")
}

func copyLegacyMeetingAudio(voxtDir, dataDir, meetID, rel string, segs []store.MeetingSegment) (string, error) {
	rel = strings.TrimPrefix(strings.TrimSpace(rel), "/")
	if rel == "" {
		return "", nil
	}
	src := filepath.Join(legacyHistoryAudioRoot(voxtDir), rel)
	if _, err := os.Stat(src); err != nil {
		return "", nil
	}
	destDir := filepath.Join(dataDir, "meetings", meetID)
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return "", err
	}
	destName := "mic.wav"
	for _, s := range segs {
		if s.Source == store.SegmentSystem {
			destName = "system.wav"
			break
		}
	}
	dest := filepath.Join(destDir, destName)
	in, err := os.Open(src)
	if err != nil {
		return "", err
	}
	defer in.Close()
	out, err := os.Create(dest)
	if err != nil {
		return "", err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return "", err
	}
	if err := out.Close(); err != nil {
		return "", err
	}
	return meetID, nil
}

func firstNonEmpty(parts ...string) string {
	for _, p := range parts {
		if s := strings.TrimSpace(p); s != "" {
			return s
		}
	}
	return ""
}

func cmpNonEmpty(parts ...string) string { return firstNonEmpty(parts...) }
