package sqlitestore

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/egoist/mygo/plugins/sqlite"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// notesRepo stores the user's notes (notes).
type notesRepo struct{ db *sqlite.DB }

func (r *notesRepo) Put(ctx context.Context, n store.Note) error {
	if err := validID(n.ID); err != nil {
		return err
	}
	n.Text = strings.TrimSpace(n.Text)
	if n.Text == "" {
		return fmt.Errorf("note %s: empty: %w", n.ID, core.ErrInvalid)
	}
	if n.CreatedAt.IsZero() {
		n.CreatedAt = time.Now()
	}
	_, err := r.db.Execute(ctx, `INSERT INTO notes (id, text, created_at) VALUES (?, ?, ?)
ON CONFLICT(id) DO UPDATE SET text = excluded.text, created_at = excluded.created_at`,
		n.ID, n.Text, n.CreatedAt.UnixMilli())
	return wrap("note "+n.ID, err)
}

func (r *notesRepo) List(ctx context.Context) ([]store.Note, error) {
	rows, err := r.db.Query(ctx, `SELECT id, text, created_at FROM notes ORDER BY created_at DESC, id DESC`)
	if err != nil {
		return nil, wrap("notes", err)
	}
	out := make([]store.Note, 0, len(rows.Values))
	for _, row := range rows.Values {
		out = append(out, store.Note{ID: asString(row[0]), Text: asString(row[1]), CreatedAt: fromMS(row[2])})
	}
	return out, nil
}

func (r *notesRepo) Delete(ctx context.Context, id string) error {
	res, err := r.db.Execute(ctx, `DELETE FROM notes WHERE id = ?`, id)
	if err != nil {
		return wrap("note "+id, err)
	}
	if res.Changes == 0 {
		return fmt.Errorf("note %s: %w", id, core.ErrNotFound)
	}
	return nil
}
