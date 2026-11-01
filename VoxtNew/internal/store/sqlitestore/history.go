package sqlitestore

import (
	"context"
	"fmt"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/egoist/mygo/plugins/sqlite"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// historyRepo stores finished sessions (the history table: migration 001,
// rule_source 002, sent_ms 003, dsp and gain_db 004, audio_until 005;
// the app a result was inserted into rides in delivery, joinDelivery);
// its recordings folder and clean-up are in history_retention.go.
type historyRepo struct {
	db  *sqlite.DB
	dir string // the store's audio directory (DefaultAudioDir)

	mu     sync.Mutex
	custom string // the one SetAudioDir set; "" = dir
}

// historyColumns are the columns an Entry has, in scanEntry's order.
const historyColumns = `id, kind, created_at, status, error, app_bundle, app_name, url_host, rule_id, rule_name,
raw_text, source_text, prompt_text, result, target_lang, delivery, asr_provider, asr_model, llm_provider, llm_model,
audio_ms, asr_ms, llm_ms, total_ms, audio, rule_source, sent_ms, dsp, gain_db, audio_until`

var historyUpsert = func() string {
	cols := strings.Split(strings.ReplaceAll(historyColumns, "\n", " "), ",")
	var set []string
	for i := range cols {
		cols[i] = strings.TrimSpace(cols[i])
		if cols[i] != "id" {
			set = append(set, cols[i]+" = excluded."+cols[i])
		}
	}
	return `INSERT INTO history (` + strings.Join(cols, ", ") + `) VALUES (` + strings.TrimSuffix(strings.Repeat("?, ", len(cols)), ", ") + `)
ON CONFLICT(id) DO UPDATE SET ` + strings.Join(set, ", ")
}()

func (r *historyRepo) Put(ctx context.Context, e store.Entry) error {
	if err := validID(e.ID); err != nil {
		return err
	}
	if !slices.Contains(store.Kinds, e.Kind) {
		return fmt.Errorf("history %s: kind %q: %w", e.ID, e.Kind, core.ErrInvalid)
	}
	if e.Status == "" {
		e.Status = store.StatusDone
	}
	if e.Audio != "" && filepath.Base(e.Audio) != e.Audio {
		return fmt.Errorf("history %s: audio %q: %w", e.ID, e.Audio, core.ErrInvalid)
	}
	_, err := r.db.Execute(ctx, historyUpsert,
		e.ID, string(e.Kind), e.CreatedAt.UnixMilli(), string(e.Status), e.Error, e.AppBundle, e.AppName, e.URLHost, e.RuleID, e.RuleName,
		e.RawText, e.SourceText, e.PromptText, e.Result, e.TargetLang, joinDelivery(e), e.ASRProvider, e.ASRModel, e.LLMProvider, e.LLMModel,
		e.AudioMS, e.ASRMS, e.LLMMS, e.TotalMS, e.Audio, e.RuleSource, e.SentMS, e.DSP, e.GainDB, untilMS(e.AudioUntil))
	return wrap("history "+e.ID, err)
}

// joinDelivery is the delivery column: Delivery, then the app the result
// was inserted into when it is not the entry's app, "insert|<bundle
// id>|<name>" (bundle ids have no "|"). No migration: an older VoxtNew
// reading it shows no delivery for that entry, nothing worse.
func joinDelivery(e store.Entry) string {
	if e.InsertedBundle == "" && e.InsertedName == "" {
		return e.Delivery
	}
	return e.Delivery + "|" + e.InsertedBundle + "|" + e.InsertedName
}

// splitDelivery reads joinDelivery's column into e.
func splitDelivery(e *store.Entry, col string) {
	parts := strings.SplitN(col, "|", 3)
	e.Delivery = parts[0]
	if len(parts) == 3 {
		e.InsertedBundle, e.InsertedName = parts[1], parts[2]
	}
}

func scanEntry(row []any) store.Entry {
	str := func(i int) string { return asString(row[i]) }
	e := store.Entry{
		ID: str(0), Kind: store.Kind(str(1)), CreatedAt: fromMS(row[2]), Status: store.Status(str(3)), Error: str(4),
		AppBundle: str(5), AppName: str(6), URLHost: str(7), RuleID: str(8), RuleName: str(9),
		RawText: str(10), SourceText: str(11), PromptText: str(12), Result: str(13), TargetLang: str(14),
		ASRProvider: str(16), ASRModel: str(17), LLMProvider: str(18), LLMModel: str(19),
		AudioMS: asInt(row[20]), ASRMS: asInt(row[21]), LLMMS: asInt(row[22]), TotalMS: asInt(row[23]), Audio: str(24),
		RuleSource: str(25), SentMS: asInt(row[26]), DSP: str(27), GainDB: asFloat(row[28]),
		AudioUntil: untilTime(row[29]),
	}
	splitDelivery(&e, str(15))
	return e
}

// untilMS is AudioUntil as stored: Unix ms, 0 for none.
func untilMS(t time.Time) int64 {
	if t.IsZero() {
		return 0
	}
	return t.UnixMilli()
}

// untilTime is untilMS's inverse.
func untilTime(v any) time.Time {
	if ms := asInt(v); ms > 0 {
		return time.UnixMilli(ms)
	}
	return time.Time{}
}

func (r *historyRepo) Get(ctx context.Context, id string) (store.Entry, error) {
	rows, err := r.db.Query(ctx, `SELECT `+historyColumns+` FROM history WHERE id = ?`, id)
	if err != nil {
		return store.Entry{}, wrap("history "+id, err)
	}
	if len(rows.Values) == 0 {
		return store.Entry{}, fmt.Errorf("history %s: %w", id, core.ErrNotFound)
	}
	return scanEntry(rows.Values[0]), nil
}

func (r *historyRepo) List(ctx context.Context, q store.HistoryQuery) ([]store.Entry, error) {
	var where []string
	var args []any
	if s := strings.TrimSpace(q.Search); s != "" {
		if useFTS(s) {
			where = append(where, `seq IN (SELECT rowid FROM history_fts WHERE history_fts MATCH ?)`)
			args = append(args, ftsPhrase(s))
		} else {
			p := likePattern(s)
			where = append(where, `(raw_text LIKE ? ESCAPE '\' OR source_text LIKE ? ESCAPE '\' OR prompt_text LIKE ? ESCAPE '\' OR result LIKE ? ESCAPE '\')`)
			args = append(args, p, p, p, p)
		}
	}
	if q.Kind != "" {
		where, args = append(where, `kind = ?`), append(args, string(q.Kind))
	}
	if q.WithAudio {
		where = append(where, `audio != ''`)
	}
	sql := `SELECT ` + historyColumns + ` FROM history`
	if len(where) > 0 {
		sql += ` WHERE ` + strings.Join(where, ` AND `)
	}
	sql += ` ORDER BY created_at DESC, id DESC`
	sql, args = page(sql, args, q.Limit, q.Offset)
	rows, err := r.db.Query(ctx, sql, args...)
	if err != nil {
		return nil, wrap("history", err)
	}
	out := make([]store.Entry, 0, len(rows.Values))
	for _, row := range rows.Values {
		out = append(out, scanEntry(row))
	}
	return out, nil
}

// Delete removes the row, then the audio file (a missing file is fine).
func (r *historyRepo) Delete(ctx context.Context, id string) error {
	rows, err := r.db.Query(ctx, `DELETE FROM history WHERE id = ? RETURNING audio`, id)
	if err != nil {
		return wrap("history "+id, err)
	}
	if len(rows.Values) == 0 {
		return fmt.Errorf("history %s: %w", id, core.ErrNotFound)
	}
	return r.removeAudio(asString(rows.Values[0][0]))
}

func (r *historyRepo) DeleteAll(ctx context.Context, kind store.Kind) (int, error) {
	sql, args := `DELETE FROM history RETURNING audio`, []any(nil)
	if kind != "" {
		sql, args = `DELETE FROM history WHERE kind = ? RETURNING audio`, []any{string(kind)}
	}
	rows, err := r.db.Query(ctx, sql, args...)
	if err != nil {
		return 0, wrap("history", err)
	}
	return len(rows.Values), r.removeAudios(rows.Values)
}

func (r *historyRepo) ExpireAudio(ctx context.Context, now time.Time) (int, error) {
	// RETURNING gives the new values: the names first, then one guarded
	// update a row (a recording retried meanwhile stays).
	rows, err := r.db.Query(ctx, `SELECT seq, audio FROM history WHERE audio != '' AND audio_until > 0 AND audio_until <= ?`, now.UnixMilli())
	if err != nil || len(rows.Values) == 0 {
		return 0, wrap("history audio", err)
	}
	stmts := make([]sqlite.Statement, 0, len(rows.Values))
	for _, row := range rows.Values {
		stmts = append(stmts, sqlite.Statement{SQL: `UPDATE history SET audio = '', audio_until = 0 WHERE seq = ? AND audio = ? AND audio_until > 0`, Args: []any{row[0], row[1]}})
	}
	res, err := r.db.Transaction(ctx, stmts)
	if err != nil {
		return 0, wrap("history audio", err)
	}
	var gone [][]any
	for i, x := range res {
		if x.Changes > 0 {
			gone = append(gone, []any{rows.Values[i][1]})
		}
	}
	return len(gone), r.removeAudios(gone)
}
