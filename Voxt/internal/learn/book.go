package learn

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dict"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

const (
	documentKey   = "dict:suggestions"
	maxPending    = 200
	maxDismissed  = 500
	dismissedDays = 90
)

// bookDocument is the stored document. Dismissed maps a folded spelling
// to the UTC day it was dismissed, which sorts as it reads.
type bookDocument struct {
	Version   int               `json:"v"`
	Items     []Suggestion      `json:"items"`
	Dismissed map[string]string `json:"dismissed"`
	Scanned   scanCheckpoint    `json:"scanned,omitzero"`
}

// scanCheckpoint is the last history entry a scan read, in the order of
// History.List: an id is not a time (a legacy entry's is no ULID).
type scanCheckpoint struct {
	At time.Time `json:"at"`
	ID string    `json:"id"`
}

// remove takes the suggestion key out of the items.
func (d *bookDocument) remove(key string) (Suggestion, bool) {
	i := slices.IndexFunc(d.Items, func(s Suggestion) bool { return s.Key() == key })
	if i < 0 {
		return Suggestion{}, false
	}
	s := d.Items[i]
	d.Items = slices.Delete(d.Items, i, i+1)
	return s, true
}

// Book is the suggestions, the dismissed terms and the history scan's
// checkpoint: one document and its one owner. Goroutine-safe.
type Book struct {
	docs  store.Documents
	terms store.Dictionary
	mu    sync.Mutex // held across each read-modify-write of the document
	now   func() time.Time
}

// NewBook opens the book over the library's documents and dictionary.
func NewBook(docs store.Documents, terms store.Dictionary) *Book {
	return &Book{docs: docs, terms: terms, now: time.Now}
}

// load reads the document without the dismissals past their 90 days.
func (b *Book) load(ctx context.Context) (bookDocument, error) {
	doc := bookDocument{Version: 1, Dismissed: map[string]string{}}
	data, ok, err := b.docs.Get(ctx, documentKey)
	if err != nil || !ok {
		return doc, err
	}
	if err := json.Unmarshal(data, &doc); err != nil {
		return doc, fmt.Errorf("learn: reading the suggestions: %w", err)
	}
	oldest := b.now().UTC().AddDate(0, 0, -dismissedDays).Format(time.DateOnly)
	maps.DeleteFunc(doc.Dismissed, func(_, day string) bool { return day < oldest })
	return doc, nil
}

func (b *Book) save(ctx context.Context, doc bookDocument) error {
	data, err := json.Marshal(doc)
	if err != nil {
		return err
	}
	return b.docs.Put(ctx, documentKey, data)
}

// scanned is the history scan's checkpoint (zero before the first scan).
func (b *Book) scanned(ctx context.Context) (scanCheckpoint, error) {
	doc, err := b.load(ctx)
	return doc.Scanned, err
}

// setScanned moves the checkpoint.
func (b *Book) setScanned(ctx context.Context, checkpoint scanCheckpoint) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	doc, err := b.load(ctx)
	if err != nil {
		return err
	}
	doc.Scanned = checkpoint
	return b.save(ctx, doc)
}

// Pending lists the suggestions, newest first.
func (b *Book) Pending(ctx context.Context) ([]Suggestion, error) {
	doc, err := b.load(ctx)
	return doc.Items, err
}

// Propose adds suggestions and reports how many are new. One whose term
// is in the dictionary (as a term or an alias) is dropped, so is one
// dismissed in the last 90 days; a repeat merges Heard and moves to the
// top. At most 200 pending: the oldest go.
func (b *Book) Propose(ctx context.Context, proposed []Suggestion) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	terms, err := b.terms.List(ctx, "")
	if err != nil {
		return 0, err
	}
	known := map[string]bool{}
	for _, t := range terms {
		known[fold(t.Term)] = true
		for _, alias := range t.Aliases {
			known[fold(alias)] = true
		}
	}
	doc, err := b.load(ctx)
	if err != nil {
		return 0, err
	}
	added := 0
	for _, s := range proposed {
		s.Term = dict.Clean(store.Term{Term: s.Term}).Term
		key := s.Key()
		if _, dismissed := doc.Dismissed[key]; dismissed || known[key] {
			continue
		}
		repeat, found := doc.remove(key)
		if !found {
			added++
		}
		s.Heard = dict.Clean(store.Term{Term: s.Term, Aliases: append(repeat.Heard, s.Heard...)}).Aliases
		doc.Items = slices.Insert(doc.Items, 0, s)
	}
	doc.Items = doc.Items[:min(len(doc.Items), maxPending)]
	return added, b.save(ctx, doc)
}

// Accept merges t into the dictionary (dict.Merge, the import's rule) and
// removes the suggestion key, in that order: after a failure in between
// the suggestion stays and accepting it again merges.
func (b *Book) Accept(ctx context.Context, key string, t store.Term) (store.Term, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	have, err := b.terms.List(ctx, "")
	if err != nil {
		return store.Term{}, err
	}
	t = dict.Clean(t)
	saved := t
	if i := slices.IndexFunc(have, func(x store.Term) bool { return fold(x.Term) == fold(t.Term) }); i >= 0 {
		saved = have[i]
	}
	put, _, _ := dict.Merge(have, []store.Term{t})
	now := b.now()
	for _, term := range put {
		if term.ID == "" {
			term.ID, term.CreatedAt = core.NewID(now), now
		}
		term.UpdatedAt = now
		if err := b.terms.Put(ctx, term); err != nil {
			return store.Term{}, err
		}
		saved = term
	}
	doc, err := b.load(ctx)
	if err != nil {
		return store.Term{}, err
	}
	doc.remove(key)
	return saved, b.save(ctx, doc)
}

// Dismiss hides the suggestion key for 90 days. At most 500 dismissals:
// the oldest go.
func (b *Book) Dismiss(ctx context.Context, key string) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	doc, err := b.load(ctx)
	if err != nil {
		return err
	}
	doc.remove(key)
	doc.Dismissed[key] = b.now().UTC().Format(time.DateOnly)
	if over := len(doc.Dismissed) - maxDismissed; over > 0 {
		keys := slices.SortedFunc(maps.Keys(doc.Dismissed), func(x, y string) int {
			return cmp.Or(cmp.Compare(doc.Dismissed[x], doc.Dismissed[y]), cmp.Compare(x, y))
		})
		for _, oldest := range keys[:over] {
			delete(doc.Dismissed, oldest)
		}
	}
	return b.save(ctx, doc)
}
