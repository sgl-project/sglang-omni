package historyapi

import (
	"context"
	"path/filepath"
	"slices"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// fakeHistory is the part of store.History the package reads, in memory,
// with the store's meaning of a query.
type fakeHistory struct {
	store.History
	mu   sync.Mutex
	list []store.Entry
	// afterPage runs after each List (a deletion or cancellation between pages).
	afterPage func(context.Context, *fakeHistory)
}

func (f *fakeHistory) selected(q store.HistoryQuery) []store.Entry {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := slices.Clone(f.list)
	sort.SliceStable(out, func(i, j int) bool {
		if !out[i].CreatedAt.Equal(out[j].CreatedAt) {
			return out[i].CreatedAt.Before(out[j].CreatedAt)
		}
		return out[i].ID < out[j].ID
	})
	return slices.DeleteFunc(out, func(e store.Entry) bool {
		switch {
		case q.Kind != "" && e.Kind != q.Kind,
			q.Finished && (e.Status != store.StatusDone || e.Result == ""),
			!q.Before.IsZero() && !e.CreatedAt.Before(q.Before):
			return true
		case q.After.IsZero():
			return false
		case q.AfterID != "":
			return e.CreatedAt.Before(q.After) || e.CreatedAt.Equal(q.After) && e.ID <= q.AfterID
		}
		return e.CreatedAt.Before(q.After)
	})
}

func (f *fakeHistory) List(ctx context.Context, q store.HistoryQuery) ([]store.Entry, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	out := f.selected(q)
	if !q.Oldest {
		slices.Reverse(out)
	}
	out = out[min(q.Offset, len(out)):]
	if q.Limit > 0 && len(out) > q.Limit {
		out = out[:q.Limit]
	}
	if f.afterPage != nil {
		f.afterPage(ctx, f)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

func (f *fakeHistory) Count(_ context.Context, q store.HistoryQuery) (int, error) {
	return len(f.selected(q)), nil
}

var testStart = time.Date(2026, 11, 8, 9, 0, 0, 0, time.UTC)

func doneEntry(i int) store.Entry {
	return store.Entry{ID: entryID(i), Kind: store.KindTranscribe, Status: store.StatusDone,
		CreatedAt: testStart.Add(time.Duration(i) * time.Second), Result: "text " + entryID(i), RawText: "heard"}
}

func entryID(i int) string {
	return "e" + string(rune('a'+i/676)) + string(rune('a'+i/26%26)) + string(rune('a'+i%26))
}

// startServer serves f on a listener in a new folder; the returned stop
// waits for it to end.
func startServer(t *testing.T, s *Server) (path string, cancel func()) {
	t.Helper()
	path = filepath.Join(t.TempDir(), SocketName)
	ln, err := Listen(path)
	if err != nil {
		t.Fatal(err)
	}
	if s.Policy == nil {
		s.Policy = func() Policy { return Policy{KeepHistory: true, KeepDays: 90} }
	}
	ctx, stop := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		s.Serve(ctx, ln)
	}()
	cancel = func() {
		stop()
		<-done
	}
	t.Cleanup(cancel)
	return path, cancel
}
