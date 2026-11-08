package learn

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
)

var bookStart = time.Date(2026, 11, 7, 8, 0, 0, 0, time.UTC)

func newBook(t *testing.T) (*Book, *sqlitestore.Store, *time.Time) {
	t.Helper()
	st, err := sqlitestore.Open(context.Background(), t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { st.Close() })
	b := NewBook(st.Documents(), st.Dictionary())
	now := bookStart
	b.now = func() time.Time { return now }
	return b, st, &now
}

func propose(t *testing.T, b *Book, s ...Suggestion) int {
	t.Helper()
	n, err := b.Propose(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	return n
}

func pending(t *testing.T, b *Book) []Suggestion {
	t.Helper()
	got, err := b.Pending(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	return got
}

func pendingTerms(t *testing.T, b *Book) []string {
	t.Helper()
	var out []string
	for _, s := range pending(t, b) {
		out = append(out, s.Term)
	}
	return out
}

func putTerm(t *testing.T, st *sqlitestore.Store, term store.Term) {
	t.Helper()
	term.CreatedAt, term.UpdatedAt = bookStart, bookStart
	if err := st.Dictionary().Put(context.Background(), term); err != nil {
		t.Fatal(err)
	}
}

func dismiss(t *testing.T, b *Book, key string) {
	t.Helper()
	if err := b.Dismiss(context.Background(), key); err != nil {
		t.Fatal(err)
	}
}

func TestProposeOrderAndRepeat(t *testing.T) {
	b, _, _ := newBook(t)
	if n := propose(t, b,
		Suggestion{Term: "Claude Code", Heard: []string{"Cloud Code"}, Source: FromEdit},
		Suggestion{Term: "  Ghostty ", Source: FromHistory},
	); n != 2 {
		t.Fatalf("new %d", n)
	}
	if got := pendingTerms(t, b); !slices.Equal(got, []string{"Ghostty", "Claude Code"}) {
		t.Fatalf("newest first: %q", got)
	}
	if n := propose(t, b, Suggestion{Term: "claude code", Heard: []string{"cloud code", "Clawed Code"}, Source: FromEdit}); n != 0 {
		t.Fatalf("a repeat counted as new: %d", n)
	}
	got := pending(t, b)
	if len(got) != 2 || got[0].Key() != "claude code" || !slices.Equal(got[0].Heard, []string{"Cloud Code", "Clawed Code"}) {
		t.Fatalf("repeat: %+v", got)
	}
}

func TestProposeDropsWhatTheDictionaryHas(t *testing.T) {
	b, st, _ := newBook(t)
	putTerm(t, st, store.Term{ID: "1", Term: "Kubernetes", Aliases: []string{"cooper netties"}, Enabled: true})
	for _, term := range []string{"kubernetes", "Cooper Netties"} {
		if n := propose(t, b, Suggestion{Term: term, Source: FromEdit}); n != 0 || len(pending(t, b)) != 0 {
			t.Errorf("%q proposed: %d, %+v", term, n, pending(t, b))
		}
	}
}

func TestDismissedStaysAwayFor90Days(t *testing.T) {
	b, _, now := newBook(t)
	propose(t, b, Suggestion{Term: "Wechart", Source: FromEdit})
	dismiss(t, b, "wechart")
	if got := pending(t, b); len(got) != 0 {
		t.Fatalf("still pending: %+v", got)
	}
	for _, tc := range []struct {
		days int
		want int
	}{{1, 0}, {90, 0}, {91, 1}} {
		*now = bookStart.AddDate(0, 0, tc.days)
		if n := propose(t, b, Suggestion{Term: "Wechart", Source: FromEdit}); n != tc.want {
			t.Fatalf("day %d: new %d, want %d", tc.days, n, tc.want)
		}
	}
}

func TestLimitsDropTheOldest(t *testing.T) {
	b, _, now := newBook(t)
	for i := range 201 {
		propose(t, b, Suggestion{Term: fmt.Sprintf("Term%d", i), Source: FromHistory})
	}
	got := pendingTerms(t, b)
	if len(got) != 200 || got[0] != "Term200" || slices.Contains(got, "Term0") {
		t.Fatalf("pending %d, first %q", len(got), got[0])
	}
	dismiss(t, b, "oldest")
	*now = bookStart.AddDate(0, 0, 1)
	for i := range 500 {
		dismiss(t, b, fmt.Sprintf("word%d", i))
	}
	if n := propose(t, b, Suggestion{Term: "oldest", Source: FromEdit}); n != 1 {
		t.Errorf("the oldest dismissal was kept (new %d)", n)
	}
	if n := propose(t, b, Suggestion{Term: "word0", Source: FromEdit}); n != 0 {
		t.Errorf("a newer dismissal went (new %d)", n)
	}
}

func TestAcceptNewTermHasNoAliases(t *testing.T) {
	b, st, _ := newBook(t)
	s := Suggestion{Term: "Claude Code", Heard: []string{"Cloud Code"}, Source: FromEdit}
	propose(t, b, s, Suggestion{Term: "Ghostty", Source: FromHistory})
	saved, err := b.Accept(context.Background(), s.Key(), s.AsTerm())
	if err != nil {
		t.Fatal(err)
	}
	terms, err := st.Dictionary().List(context.Background(), "")
	if err != nil {
		t.Fatal(err)
	}
	if len(terms) != 1 || terms[0].Term != "Claude Code" || len(terms[0].Aliases) != 0 || !terms[0].Enabled || terms[0].ID == "" || terms[0].CreatedAt.IsZero() || saved.ID != terms[0].ID {
		t.Fatalf("saved %+v, terms %+v", saved, terms)
	}
	if got := pendingTerms(t, b); !slices.Equal(got, []string{"Ghostty"}) {
		t.Errorf("pending %q", got)
	}
}

func TestAcceptKnownTermMerges(t *testing.T) {
	b, st, _ := newBook(t)
	putTerm(t, st, store.Term{ID: "1", Term: "Voxt", Aliases: []string{"Vox T"}, Enabled: true})
	s := Suggestion{Term: "VOXT", Source: FromEdit}
	saved, err := b.Accept(context.Background(), s.Key(), store.Term{Term: "VOXT", Aliases: []string{"Box T"}, Enabled: true})
	if err != nil {
		t.Fatal(err)
	}
	terms, err := st.Dictionary().List(context.Background(), "")
	if err != nil {
		t.Fatal(err)
	}
	if len(terms) != 1 || terms[0].ID != "1" || terms[0].Term != "Voxt" || !slices.Equal(terms[0].Aliases, []string{"Vox T", "Box T"}) || saved.ID != "1" {
		t.Fatalf("saved %+v, terms %+v", saved, terms)
	}
}

func TestProposeFromTwoGoroutinesKeepsBoth(t *testing.T) {
	b, _, _ := newBook(t)
	start := make(chan struct{})
	var wg sync.WaitGroup
	for g := range 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			for i := range 10 {
				if _, err := b.Propose(context.Background(), []Suggestion{{Term: fmt.Sprintf("G%dT%d", g, i), Source: FromHistory}}); err != nil {
					t.Error(err)
				}
			}
		}()
	}
	close(start)
	wg.Wait()
	if got := pendingTerms(t, b); len(got) != 20 {
		t.Fatalf("%d of 20: %q", len(got), got)
	}
}

// interleavedDocuments runs interleave once, right after a read: the other
// writer between this one's read and write.
type interleavedDocuments struct {
	store.Documents
	interleave func()
}

func (d *interleavedDocuments) Get(ctx context.Context, key string) ([]byte, bool, error) {
	data, ok, err := d.Documents.Get(ctx, key)
	if interleave := d.interleave; interleave != nil {
		d.interleave = nil
		interleave()
	}
	return data, ok, err
}

func TestAWriterBetweenAnothersReadAndWriteIsNotLost(t *testing.T) {
	for name, write := range map[string]func(b *Book) error{
		"propose": func(b *Book) error {
			_, err := b.Propose(context.Background(), []Suggestion{{Term: "First", Source: FromEdit}})
			return err
		},
		"accept": func(b *Book) error {
			_, err := b.Accept(context.Background(), "first", store.Term{Term: "First", Enabled: true})
			return err
		},
		"dismiss": func(b *Book) error { return b.Dismiss(context.Background(), "first") },
	} {
		t.Run(name, func(t *testing.T) {
			b, st, _ := newBook(t)
			docs := &interleavedDocuments{Documents: st.Documents()}
			b.docs = docs
			done := make(chan struct{})
			docs.interleave = func() {
				guarded := !b.mu.TryLock()
				if !guarded {
					b.mu.Unlock()
				}
				go func() {
					defer close(done)
					if _, err := b.Propose(context.Background(), []Suggestion{{Term: "Second", Source: FromEdit}}); err != nil {
						t.Error(err)
					}
				}()
				if !guarded {
					<-done // no lock held: the other writer lands inside this read
				}
			}
			if err := write(b); err != nil {
				t.Fatal(err)
			}
			<-done
			if got := pendingTerms(t, b); !slices.Contains(got, "Second") {
				t.Errorf("the other writer's suggestion was lost: %q", got)
			}
		})
	}
}

func TestDocumentKeepsOnlyWhatTheRowShows(t *testing.T) {
	b, st, _ := newBook(t)
	propose(t, b, Suggestion{Term: "Claude Code", Heard: []string{"Cloud Code"}, Source: FromEdit})
	dismiss(t, b, "wechart")
	data, ok, err := st.Documents().Get(context.Background(), documentKey)
	if err != nil || !ok {
		t.Fatal(ok, err)
	}
	var doc map[string]any
	if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"app", "at", "entry"} {
		if _, found := doc[field]; found {
			t.Errorf("document has %q: %s", field, data)
		}
	}
	items := doc["items"].([]any)
	if len(items) != 1 {
		t.Fatalf("items %v", items)
	}
	for field := range items[0].(map[string]any) {
		if !slices.Contains([]string{"term", "heard", "source"}, field) {
			t.Errorf("item has %q: %s", field, data)
		}
	}
	if day := doc["dismissed"].(map[string]any)["wechart"]; !strings.HasPrefix(fmt.Sprint(day), "2026-11-07") {
		t.Errorf("dismissed %v", day)
	}
}
