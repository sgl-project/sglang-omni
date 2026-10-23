package sqlitestore

// The history's recordings folder and its clean-up: where recordings go
// (the store's audio/, or a folder the user chose), finding one wherever
// it was kept, deleting them all while the entries stay, and the
// automatic clean-up of old entries (by age, and by count per kind). A
// delete is one statement, so the rows and their full-text index (its
// triggers) change together; the files go after it: a file left behind
// is an orphan, never an entry without its row.

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"

	"github.com/egoist/mygo/plugins/sqlite"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

func (r *historyRepo) DefaultAudioDir() string { return r.dir }

func (r *historyRepo) AudioDir() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.custom != "" {
		return r.custom
	}
	return r.dir
}

func (r *historyRepo) SetAudioDir(dir string) {
	if dir != "" && filepath.Clean(dir) == filepath.Clean(r.dir) {
		dir = ""
	}
	r.mu.Lock()
	r.custom = dir
	r.mu.Unlock()
}

func (r *historyRepo) AudioPath(name string) string {
	p := filepath.Join(r.AudioDir(), name)
	if _, err := os.Stat(p); err != nil {
		if q := filepath.Join(r.dir, name); q != p {
			if _, err := os.Stat(q); err == nil {
				return q
			}
		}
	}
	return p
}

// removeAudio deletes a recording, from the folder in use and from the
// store's own (a missing file is fine).
func (r *historyRepo) removeAudio(file string) error {
	if file == "" || filepath.Base(file) != file {
		return nil
	}
	var first error
	for _, d := range []string{r.AudioDir(), r.dir} {
		if err := os.Remove(filepath.Join(d, file)); err != nil && !errors.Is(err, fs.ErrNotExist) && first == nil {
			first = fmt.Errorf("removing %s: %w", file, err)
		}
	}
	return first
}

// removeAudios deletes the recordings rows name (one per row, first
// column) and returns the first error.
func (r *historyRepo) removeAudios(rows [][]any) error {
	var first error
	for _, row := range rows {
		if err := r.removeAudio(asString(row[0])); err != nil && first == nil {
			first = err
		}
	}
	return first
}

func (r *historyRepo) DeleteAudio(ctx context.Context) (int, error) {
	// The names first, then one guarded update a row, in one transaction:
	// a recording kept in between (a new entry, or one that changed) stays.
	rows, err := r.db.Query(ctx, `SELECT seq, audio FROM history WHERE audio != ''`)
	if err != nil || len(rows.Values) == 0 {
		return 0, wrap("history audio", err)
	}
	stmts := make([]sqlite.Statement, 0, len(rows.Values))
	for _, row := range rows.Values {
		stmts = append(stmts, sqlite.Statement{SQL: `UPDATE history SET audio = '', audio_until = 0 WHERE seq = ? AND audio = ?`, Args: []any{row[0], row[1]}})
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

// pruneWhere is the condition of the entries a retention drops, and its
// arguments; "" when it keeps everything.
func pruneWhere(rt store.Retention) (string, []any) {
	var cond string
	var args []any
	if !rt.Before.IsZero() {
		cond, args = `created_at < ?`, append(args, rt.Before.UnixMilli())
	}
	if rt.PerKind > 0 {
		if cond != "" {
			cond += ` OR `
		}
		cond += `seq IN (SELECT seq FROM (SELECT seq, ROW_NUMBER() OVER (PARTITION BY kind ORDER BY created_at DESC, id DESC) AS n FROM history) WHERE n > ?)`
		args = append(args, rt.PerKind)
	}
	return cond, args
}

func (r *historyRepo) Prune(ctx context.Context, rt store.Retention) (int, error) {
	cond, args := pruneWhere(rt)
	if cond == "" {
		return 0, nil
	}
	rows, err := r.db.Query(ctx, `DELETE FROM history WHERE `+cond+` RETURNING audio`, args...)
	if err != nil {
		return 0, wrap("history clean-up", err)
	}
	return len(rows.Values), r.removeAudios(rows.Values)
}

func (r *historyRepo) CountPrune(ctx context.Context, rt store.Retention) (int, error) {
	cond, args := pruneWhere(rt)
	if cond == "" {
		return 0, nil
	}
	rows, err := r.db.Query(ctx, `SELECT COUNT(*) FROM history WHERE `+cond, args...)
	if err != nil {
		return 0, wrap("history clean-up", err)
	}
	if len(rows.Values) != 1 {
		return 0, nil
	}
	return int(asInt(rows.Values[0][0])), nil
}
