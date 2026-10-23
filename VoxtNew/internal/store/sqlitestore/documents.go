package sqlitestore

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/egoist/mygo/plugins/sqlite"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
)

// documentRepo stores small JSON documents by key (store.Documents).
type documentRepo struct{ db *sqlite.DB }

func (r *documentRepo) Get(ctx context.Context, key string) ([]byte, bool, error) {
	rows, err := r.db.Query(ctx, `SELECT doc FROM documents WHERE key = ?`, key)
	if err != nil {
		return nil, false, wrap("document "+key, err)
	}
	if len(rows.Values) == 0 {
		return nil, false, nil
	}
	return asBytes(rows.Values[0][0]), true, nil
}

// Put stores a JSON document under key.
func (r *documentRepo) Put(ctx context.Context, key string, doc []byte) error {
	if key == "" || !json.Valid(doc) {
		return fmt.Errorf("document %q: %w", key, core.ErrInvalid)
	}
	_, err := r.db.Execute(ctx, `INSERT INTO documents (key, doc, updated_at) VALUES (?, ?, ?)
ON CONFLICT(key) DO UPDATE SET doc = excluded.doc, updated_at = excluded.updated_at`, key, string(doc), time.Now().UnixMilli())
	return wrap("document "+key, err)
}

// Delete removes a document; deleting a missing one is a no-op.
func (r *documentRepo) Delete(ctx context.Context, key string) error {
	_, err := r.db.Execute(ctx, `DELETE FROM documents WHERE key = ?`, key)
	return wrap("document "+key, err)
}
