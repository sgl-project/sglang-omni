package historyapi

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func exportAll(t *testing.T, path string, req Request) []Row {
	t.Helper()
	var got []Row
	n, err := Export(path, req, func(r Row) error { got = append(got, r); return nil })
	if err != nil {
		t.Fatalf("export: %v", err)
	}
	if n != len(got) {
		t.Fatalf("count %d, got %d rows", n, len(got))
	}
	return got
}

// TestServeFileMode: Listen makes the socket 0600, and replaces a stale
// one left by a crash.
func TestServeFileMode(t *testing.T) {
	path := socketPath(t)
	stale, err := net.Listen("unix", path)
	if err != nil {
		t.Fatal(err)
	}
	stale.(*net.UnixListener).SetUnlinkOnClose(false)
	stale.Close()
	ln, err := Listen(path)
	if err != nil {
		t.Fatalf("stale socket not replaced: %v", err)
	}
	defer ln.Close()
	fi, err := os.Stat(path)
	if err != nil || fi.Mode().Perm() != 0o600 {
		t.Fatalf("mode %v, %v", fi, err)
	}
	// Anything else at the path is not ours to delete.
	file := filepath.Join(t.TempDir(), "notasocket")
	os.WriteFile(file, []byte("x"), 0o600)
	if _, err := Listen(file); err == nil {
		t.Error("listened over a plain file")
	}
	if _, err := os.Stat(file); err != nil {
		t.Error("deleted a plain file")
	}
}

// TestExportBounds: a half-open range, oldest first, kind respected.
func TestExportBounds(t *testing.T) {
	f := &fakeHistory{}
	for i := range 5 {
		e := doneEntry(i)
		if i == 2 {
			e.Kind = store.KindRewrite
		}
		f.list = append(f.list, e)
	}
	failed := doneEntry(6)
	failed.Status = store.StatusFailed
	cancelled := doneEntry(7)
	cancelled.Status = store.StatusCancelled
	empty := doneEntry(8)
	empty.Result = ""
	f.list = append(f.list, failed, cancelled, empty)
	path, _ := startServer(t, &Server{History: f})
	got := exportAll(t, path, Request{Since: f.list[1].CreatedAt.UnixMilli(), Until: f.list[4].CreatedAt.UnixMilli()})
	var ids []string
	for _, r := range got {
		ids = append(ids, r.ID)
	}
	want := []string{entryID(1), entryID(2), entryID(3)}
	if strings.Join(ids, ",") != strings.Join(want, ",") {
		t.Errorf("got %v want %v", ids, want)
	}
	if got := exportAll(t, path, Request{Since: 1, Kind: "rewrite"}); len(got) != 1 || got[0].ID != entryID(2) {
		t.Errorf("kind: %+v", got)
	}
	// Failed, cancelled and empty results never go out.
	if got := exportAll(t, path, Request{Since: 1}); len(got) != 5 {
		t.Errorf("all finished: %d", len(got))
	}
}

// TestExportPages: 1201 rows across three pages, none lost or repeated.
func TestExportPages(t *testing.T) {
	f := &fakeHistory{}
	for i := range 1201 {
		f.list = append(f.list, doneEntry(i))
	}
	path, _ := startServer(t, &Server{History: f})
	got := exportAll(t, path, Request{Since: 1})
	if len(got) != 1201 {
		t.Fatalf("%d rows", len(got))
	}
	seen := map[string]bool{}
	for _, r := range got {
		if seen[r.ID] {
			t.Fatalf("%s twice", r.ID)
		}
		seen[r.ID] = true
	}
}

// TestWalkKeepsPlaceAfterDelete: rows deleted between pages (the
// clean-up, oldest first) cannot make the next page skip one.
func TestWalkKeepsPlaceAfterDelete(t *testing.T) {
	f := &fakeHistory{}
	for i := range PageSize + 10 {
		f.list = append(f.list, doneEntry(i))
	}
	once := sync.Once{}
	f.afterPage = func(f *fakeHistory) {
		once.Do(func() {
			f.mu.Lock()
			defer f.mu.Unlock()
			f.list = f.list[5:] // the oldest five go
		})
	}
	seen := 0
	err := Walk(context.Background(), f, store.HistoryQuery{}, func() bool { return false }, func(Row) error { seen++; return nil })
	if err != nil || seen != PageSize+10 {
		t.Errorf("saw %d, want %d (%v)", seen, PageSize+10, err)
	}
}

// TestCancelMidStream: cancelling mid-read ends it without an end line,
// which the client reports as interrupted, and the partial read is logged.
func TestCancelMidStream(t *testing.T) {
	f := &fakeHistory{}
	for i := range PageSize + 5 {
		f.list = append(f.list, doneEntry(i))
	}
	var mu sync.Mutex
	var reads []Read
	path, cancel := startServer(t, &Server{History: f, Audit: func(r Read) { mu.Lock(); reads = append(reads, r); mu.Unlock() }})
	first := make(chan struct{})
	n := 0
	go func() { <-first; cancel() }()
	_, err := Export(path, Request{Since: 1}, func(Row) error {
		if n == 0 {
			close(first)
			time.Sleep(50 * time.Millisecond) // let the cancel land
		}
		n++
		return nil
	})
	if !errors.Is(err, ErrInterrupted) {
		t.Fatalf("err %v (after %d rows)", err, n)
	}
	if n >= PageSize+5 {
		t.Errorf("all %d rows came through", n)
	}
	cancel()
	mu.Lock()
	defer mu.Unlock()
	if len(reads) != 1 {
		t.Errorf("reads %+v", reads)
	}
}

// TestCancelUnblocksStuckWrite: turning the switch off ends a read at
// once, even one blocked writing to a client that stopped reading.
func TestCancelUnblocksStuckWrite(t *testing.T) {
	f := &fakeHistory{}
	big := strings.Repeat("x", 1<<20)
	for i := range 20 {
		e := doneEntry(i)
		e.Result = big
		f.list = append(f.list, e)
	}
	path, cancel := startServer(t, &Server{History: f, Timeout: 20 * time.Second})
	stuck, err := net.Dial("unix", path)
	if err != nil {
		t.Fatal(err)
	}
	defer stuck.Close()
	req, _ := json.Marshal(Request{Op: OpExport, Since: 1})
	stuck.Write(append(req, '\n'))
	time.Sleep(100 * time.Millisecond) // the server is blocked in a write
	start := time.Now()
	cancel()
	if d := time.Since(start); d > 5*time.Second {
		t.Errorf("the server took %v to stop", d)
	}
}

// TestStuckClientReleased: a client that never reads its reply holds up
// the next only until the write timeout.
func TestStuckClientReleased(t *testing.T) {
	f := &fakeHistory{}
	big := strings.Repeat("x", 1<<20)
	for i := range 20 {
		e := doneEntry(i)
		e.Result = big
		f.list = append(f.list, e)
	}
	path, _ := startServer(t, &Server{History: f, Timeout: 150 * time.Millisecond})
	stuck, err := net.Dial("unix", path)
	if err != nil {
		t.Fatal(err)
	}
	defer stuck.Close()
	req, _ := json.Marshal(Request{Op: OpExport, Since: 1})
	stuck.Write(append(req, '\n'))
	time.Sleep(20 * time.Millisecond)
	start := time.Now()
	if got := exportAll(t, path, Request{Since: 1, Kind: "translate"}); len(got) != 0 {
		t.Errorf("%d rows", len(got))
	}
	if d := time.Since(start); d > 5*time.Second {
		t.Errorf("second client waited %v", d)
	}
}

// TestAuditHasNoText: the record says how many and who, never what.
func TestAuditHasNoText(t *testing.T) {
	f := &fakeHistory{list: []store.Entry{secretEntry()}}
	var mu sync.Mutex
	var reads []Read
	path, _ := startServer(t, &Server{History: f, Peer: func(net.Conn) string { return "launchd" },
		Audit: func(r Read) { mu.Lock(); reads = append(reads, r); mu.Unlock() }})
	exportAll(t, path, Request{Since: 1, As: "  daily\x1b[31m \n"})
	if _, err := FetchStatus(path); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(reads) != 2 {
		t.Fatalf("reads %+v", reads)
	}
	r := reads[0]
	if r.Op != OpExport || r.Count != 1 || r.Program != "launchd" || r.As != "daily[31m" {
		t.Errorf("%+v", r)
	}
	if reads[1].Op != OpStatus || reads[1].Count != 0 {
		t.Errorf("status logged as %+v", reads[1])
	}
	data, _ := json.Marshal(r)
	for _, s := range []string{"final text", "heard words", "selected-secret", "the instruction"} {
		if strings.Contains(string(data), s) {
			t.Errorf("leaked %q in %s", s, data)
		}
	}
}

// TestSelectionFollowsPolicy reads the setting on each row, not once.
func TestSelectionFollowsPolicy(t *testing.T) {
	f := &fakeHistory{list: []store.Entry{secretEntry()}}
	on := true
	var mu sync.Mutex
	path, _ := startServer(t, &Server{History: f, Policy: func() Policy { mu.Lock(); defer mu.Unlock(); return Policy{Selection: on} }})
	if got := exportAll(t, path, Request{Since: 1}); got[0].Selection != "selected-secret" {
		t.Errorf("%+v", got[0])
	}
	mu.Lock()
	on = false
	mu.Unlock()
	if got := exportAll(t, path, Request{Since: 1}); got[0].Selection != "" {
		t.Errorf("%+v", got[0])
	}
}

func TestStatus(t *testing.T) {
	failed := doneEntry(0)
	failed.Status = store.StatusFailed
	f := &fakeHistory{list: []store.Entry{failed, doneEntry(1), doneEntry(2)}}
	path, _ := startServer(t, &Server{History: f, Policy: func() Policy { return Policy{KeepHistory: true, KeepDays: 90} }})
	st, err := FetchStatus(path)
	if err != nil {
		t.Fatal(err)
	}
	if st.Schema != 1 || !st.KeepHistory || st.KeepDays != 90 || st.Entries != 2 || !st.Oldest.Equal(doneEntry(1).CreatedAt) {
		t.Errorf("%+v", st)
	}
}

// TestBadRequest: an unknown op or kind is refused and not logged.
func TestBadRequest(t *testing.T) {
	logged := false
	path, _ := startServer(t, &Server{History: &fakeHistory{}, Audit: func(Read) { logged = true }})
	for _, req := range []Request{{Op: "drop"}, {Op: OpExport, Kind: "nope"}} {
		data, _ := json.Marshal(req)
		_, err := exchangeRaw(path, data)
		if !errors.Is(err, ErrInterrupted) {
			t.Errorf("%+v: %v", req, err)
		}
	}
	if logged {
		t.Error("a refused request was logged")
	}
}

// exchangeRaw sends data as the request line and returns the end state.
func exchangeRaw(path string, data []byte) (int, error) {
	var req Request
	json.Unmarshal(data, &req)
	return exchange(path, req, func([]byte) error { return nil })
}
