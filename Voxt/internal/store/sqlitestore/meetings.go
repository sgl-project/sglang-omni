package sqlitestore

import (
	"context"
	"fmt"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/egoist/mygo/plugins/sqlite"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

var meetingStatuses = []store.MeetingStatus{
	store.MeetingRecording, store.MeetingInterrupted, store.MeetingAsking,
	store.MeetingFinishing, store.MeetingDone, store.MeetingFailed,
}
var meetingModes = []store.MeetingMode{store.MeetingModeMeeting, store.MeetingModeSubtitles, store.MeetingModeRecording}
var segmentSources = []store.SegmentSource{store.SegmentMic, store.SegmentSystem}

type meetingsRepo struct{ db *sqlite.DB }

const meetingColumns = `seq, id, created_at, ended_at, status, mode, title, duration_ms, asr_model, diar_model, llm_model,
speakers, summary, chat, audio, audio_until`

const segmentColumns = `meeting_seq, idx, source, speaker, start_ms, end_ms, text, translation, final`

func (r *meetingsRepo) CreateMeeting(ctx context.Context, m store.Meeting) (store.Meeting, error) {
	if err := validID(m.ID); err != nil {
		return store.Meeting{}, err
	}
	if err := validMeeting(&m); err != nil {
		return store.Meeting{}, err
	}
	if m.CreatedAt.IsZero() {
		m.CreatedAt = time.Now()
	}
	if m.Speakers == "" {
		m.Speakers = "{}"
	}
	if m.Chat == "" {
		m.Chat = "[]"
	}
	_, err := r.db.Execute(ctx,
		`INSERT INTO meetings (id, created_at, ended_at, status, mode, title, duration_ms, asr_model, diar_model, llm_model,
speakers, summary, chat, audio, audio_until) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		m.ID, m.CreatedAt.UnixMilli(), ms(m.EndedAt), string(m.Status), string(m.Mode), m.Title, m.DurationMS,
		m.ASRModel, m.DiarModel, m.LLMModel, m.Speakers, m.Summary, m.Chat, m.Audio, untilMS(m.AudioUntil))
	if err != nil {
		return store.Meeting{}, wrap("meeting "+m.ID, err)
	}
	rows, err := r.db.Query(ctx, `SELECT `+meetingColumns+` FROM meetings WHERE id = ?`, m.ID)
	if err != nil || len(rows.Values) != 1 {
		return store.Meeting{}, wrap("meeting "+m.ID, err)
	}
	return scanMeeting(rows.Values[0]), nil
}

func (r *meetingsRepo) UpdateMeeting(ctx context.Context, m store.Meeting) error {
	if m.Seq <= 0 {
		return fmt.Errorf("meeting seq: %w", core.ErrInvalid)
	}
	if err := validID(m.ID); err != nil {
		return err
	}
	if err := validMeeting(&m); err != nil {
		return err
	}
	if m.Speakers == "" {
		m.Speakers = "{}"
	}
	if m.Chat == "" {
		m.Chat = "[]"
	}
	res, err := r.db.Execute(ctx,
		`UPDATE meetings SET id = ?, created_at = ?, ended_at = ?, status = ?, mode = ?, title = ?, duration_ms = ?,
asr_model = ?, diar_model = ?, llm_model = ?, speakers = ?, summary = ?, chat = ?, audio = ?, audio_until = ?
WHERE seq = ?`,
		m.ID, m.CreatedAt.UnixMilli(), ms(m.EndedAt), string(m.Status), string(m.Mode), m.Title, m.DurationMS,
		m.ASRModel, m.DiarModel, m.LLMModel, m.Speakers, m.Summary, m.Chat, m.Audio, untilMS(m.AudioUntil), m.Seq)
	if err != nil {
		return wrap("meeting "+m.ID, err)
	}
	if res.Changes == 0 {
		return fmt.Errorf("meeting %s: %w", m.ID, core.ErrNotFound)
	}
	return nil
}

func (r *meetingsRepo) AppendSegments(ctx context.Context, seq int64, segs []store.MeetingSegment) error {
	if seq <= 0 || len(segs) == 0 {
		return nil
	}
	rows, err := r.db.Query(ctx, `SELECT COALESCE(MAX(idx), -1) FROM meeting_segments WHERE meeting_seq = ?`, seq)
	if err != nil {
		return wrap("meeting segments", err)
	}
	next := int(asInt(rows.Values[0][0])) + 1
	stmts := make([]sqlite.Statement, 0, len(segs))
	for _, s := range segs {
		if err := validSegment(s); err != nil {
			return err
		}
		idx := next
		next++
		stmts = append(stmts, sqlite.Statement{SQL: `INSERT INTO meeting_segments (meeting_seq, idx, source, speaker, start_ms, end_ms, text, translation, final)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			Args: []any{seq, idx, string(s.Source), s.Speaker, s.StartMS, s.EndMS, s.Text, s.Translation, bool01(s.Final)}})
	}
	_, err = r.db.Transaction(ctx, stmts)
	return wrap("meeting segments", err)
}

func (r *meetingsRepo) ReplaceSegments(ctx context.Context, seq int64, src store.SegmentSource, segs []store.MeetingSegment) error {
	if seq <= 0 {
		return fmt.Errorf("meeting seq: %w", core.ErrInvalid)
	}
	if !slices.Contains(segmentSources, src) {
		return fmt.Errorf("segment source %q: %w", src, core.ErrInvalid)
	}
	for _, s := range segs {
		if s.Source != src {
			return fmt.Errorf("segment source %q != %q: %w", s.Source, src, core.ErrInvalid)
		}
		if err := validSegment(s); err != nil {
			return err
		}
	}
	stmts := []sqlite.Statement{{SQL: `DELETE FROM meeting_segments WHERE meeting_seq = ? AND source = ?`, Args: []any{seq, string(src)}}}
	for i, s := range segs {
		stmts = append(stmts, sqlite.Statement{SQL: `INSERT INTO meeting_segments (meeting_seq, idx, source, speaker, start_ms, end_ms, text, translation, final)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			Args: []any{seq, i, string(s.Source), s.Speaker, s.StartMS, s.EndMS, s.Text, s.Translation, bool01(s.Final)}})
	}
	_, err := r.db.Transaction(ctx, stmts)
	return wrap("meeting segments replace", err)
}

func (r *meetingsRepo) Meeting(ctx context.Context, id string) (store.Meeting, []store.MeetingSegment, error) {
	rows, err := r.db.Query(ctx, `SELECT `+meetingColumns+` FROM meetings WHERE id = ?`, id)
	if err != nil {
		return store.Meeting{}, nil, wrap("meeting "+id, err)
	}
	if len(rows.Values) == 0 {
		return store.Meeting{}, nil, fmt.Errorf("meeting %s: %w", id, core.ErrNotFound)
	}
	m := scanMeeting(rows.Values[0])
	segRows, err := r.db.Query(ctx, `SELECT `+segmentColumns+` FROM meeting_segments WHERE meeting_seq = ? ORDER BY idx`, m.Seq)
	if err != nil {
		return store.Meeting{}, nil, wrap("meeting "+id, err)
	}
	segs := make([]store.MeetingSegment, 0, len(segRows.Values))
	for _, row := range segRows.Values {
		segs = append(segs, scanSegment(row))
	}
	return m, segs, nil
}

func (r *meetingsRepo) ListMeetings(ctx context.Context, q store.MeetingQuery) ([]store.Meeting, error) {
	var where []string
	var args []any
	if s := strings.TrimSpace(q.Search); s != "" {
		if useFTS(s) {
			where = append(where, `seq IN (SELECT meeting_seq FROM meeting_segments WHERE rowid IN (
SELECT rowid FROM meeting_segments_fts WHERE meeting_segments_fts MATCH ?))`)
			args = append(args, ftsPhrase(s))
		} else {
			p := likePattern(s)
			where = append(where, `seq IN (SELECT meeting_seq FROM meeting_segments WHERE text LIKE ? ESCAPE '\' OR translation LIKE ? ESCAPE '\')`)
			args = append(args, p, p)
		}
	}
	sql := `SELECT ` + meetingColumns + ` FROM meetings`
	if len(where) > 0 {
		sql += ` WHERE ` + strings.Join(where, ` AND `)
	}
	sql += ` ORDER BY created_at DESC, id DESC`
	sql, args = page(sql, args, q.Limit, q.Offset)
	rows, err := r.db.Query(ctx, sql, args...)
	if err != nil {
		return nil, wrap("meetings", err)
	}
	out := make([]store.Meeting, 0, len(rows.Values))
	for _, row := range rows.Values {
		out = append(out, scanMeeting(row))
	}
	return out, nil
}

func (r *meetingsRepo) DeleteMeeting(ctx context.Context, id string) error {
	res, err := r.db.Execute(ctx, `DELETE FROM meetings WHERE id = ?`, id)
	if err != nil {
		return wrap("meeting "+id, err)
	}
	if res.Changes == 0 {
		return fmt.Errorf("meeting %s: %w", id, core.ErrNotFound)
	}
	return nil
}

func (r *meetingsRepo) Unfinished(ctx context.Context) ([]store.Meeting, error) {
	rows, err := r.db.Query(ctx, `SELECT `+meetingColumns+` FROM meetings WHERE status IN ('recording','asking','finishing') ORDER BY created_at DESC`)
	if err != nil {
		return nil, wrap("meetings unfinished", err)
	}
	out := make([]store.Meeting, 0, len(rows.Values))
	for _, row := range rows.Values {
		out = append(out, scanMeeting(row))
	}
	return out, nil
}

func (r *meetingsRepo) ExpiredAudio(ctx context.Context, now time.Time) ([]store.Meeting, error) {
	rows, err := r.db.Query(ctx, `SELECT `+meetingColumns+` FROM meetings WHERE audio != '' AND audio_until > 0 AND audio_until <= ? ORDER BY audio_until`,
		now.UnixMilli())
	if err != nil {
		return nil, wrap("meetings audio", err)
	}
	out := make([]store.Meeting, 0, len(rows.Values))
	for _, row := range rows.Values {
		out = append(out, scanMeeting(row))
	}
	return out, nil
}

func scanMeeting(row []any) store.Meeting {
	str := func(i int) string { return asString(row[i]) }
	return store.Meeting{
		Seq: asInt(row[0]), ID: str(1), CreatedAt: fromMS(row[2]), EndedAt: fromMS(row[3]),
		Status: store.MeetingStatus(str(4)), Mode: store.MeetingMode(str(5)), Title: str(6),
		DurationMS: asInt(row[7]), ASRModel: str(8), DiarModel: str(9), LLMModel: str(10),
		Speakers: str(11), Summary: str(12), Chat: str(13), Audio: str(14), AudioUntil: untilTime(row[15]),
	}
}

func scanSegment(row []any) store.MeetingSegment {
	return store.MeetingSegment{
		MeetingSeq: asInt(row[0]), Idx: int(asInt(row[1])), Source: store.SegmentSource(asString(row[2])),
		Speaker: asString(row[3]), StartMS: asInt(row[4]), EndMS: asInt(row[5]),
		Text: asString(row[6]), Translation: asString(row[7]), Final: asInt(row[8]) != 0,
	}
}

func validMeeting(m *store.Meeting) error {
	if !slices.Contains(meetingStatuses, m.Status) {
		return fmt.Errorf("meeting status %q: %w", m.Status, core.ErrInvalid)
	}
	if !slices.Contains(meetingModes, m.Mode) {
		return fmt.Errorf("meeting mode %q: %w", m.Mode, core.ErrInvalid)
	}
	if m.Audio != "" && filepath.Base(m.Audio) != m.Audio {
		return fmt.Errorf("meeting audio %q: %w", m.Audio, core.ErrInvalid)
	}
	return nil
}

func validSegment(s store.MeetingSegment) error {
	if !slices.Contains(segmentSources, s.Source) {
		return fmt.Errorf("segment source %q: %w", s.Source, core.ErrInvalid)
	}
	return nil
}

func bool01(b bool) int {
	if b {
		return 1
	}
	return 0
}
