package sqlitestore

import (
	"context"
	"testing"
)

// A partial unique index leaves rows outside it without the key the
// store's ON CONFLICT needs, so it does not count.
func TestUniqueColumnIgnoresPartialIndex(t *testing.T) {
	ctx := context.Background()
	s := openTest(t, t.TempDir())
	if _, err := s.db.Execute(ctx, "CREATE TABLE p (id TEXT, done INTEGER)"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.db.Execute(ctx, "CREATE UNIQUE INDEX p_partial ON p(id) WHERE done = 1"); err != nil {
		t.Fatal(err)
	}
	if ok, err := hasUniqueColumn(ctx, s.db, "p", "id"); err != nil || ok {
		t.Fatalf("hasUniqueColumn = %v, %v; a partial index must not count", ok, err)
	}
	if _, err := s.db.Execute(ctx, "CREATE UNIQUE INDEX p_full ON p(id)"); err != nil {
		t.Fatal(err)
	}
	if ok, err := hasUniqueColumn(ctx, s.db, "p", "id"); err != nil || !ok {
		t.Fatalf("hasUniqueColumn = %v, %v; a full index must count", ok, err)
	}
}
