package sqlitestore

import (
	"context"
	"fmt"

	"github.com/egoist/mygo/plugins/sqlite"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
)

// UpdateSegmentSpeakers attaches late audio labels without replacing words,
// translations, ordering or timing. A concurrent first insert can be absent;
// commitSegment rechecks the latest label immediately after that insert.
func (r *meetingsRepo) UpdateSegmentSpeakers(ctx context.Context, seq int64, labels map[int]string) error {
	if seq <= 0 {
		return fmt.Errorf("meeting seq: %w", core.ErrInvalid)
	}
	stmts := make([]sqlite.Statement, 0, len(labels))
	for idx, id := range labels {
		if idx < 0 {
			return fmt.Errorf("meeting segment: %w", core.ErrInvalid)
		}
		stmts = append(stmts, sqlite.Statement{SQL: `UPDATE meeting_segments SET speaker = ? WHERE meeting_seq = ? AND idx = ?`, Args: []any{id, seq, idx}})
	}
	if len(stmts) == 0 {
		return nil
	}
	_, err := r.db.Transaction(ctx, stmts)
	return wrap("meeting speaker labels", err)
}
