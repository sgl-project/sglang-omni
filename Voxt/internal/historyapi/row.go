// Package historyapi lets programs on this Mac read the finished history
// through the running app, over a Unix socket (docs/history-access.md).
// The fields a row carries are listed here, in Row, and nowhere else:
// a new field of store.Entry does not come along by itself.
//
// Imports: store.
package historyapi

import (
	"context"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// PageSize is how many entries one store read returns: a read holds the
// database's one connection for a page at a time, and sessions writing
// the history go in between.
const PageSize = 500

// Row is one exported entry. Within schema 1 fields are only ever added.
type Row struct {
	ID          string     `json:"id"`
	CreatedAt   time.Time  `json:"created_at"`
	Kind        store.Kind `json:"kind"`
	App         string     `json:"app,omitempty"`
	Text        string     `json:"text"`
	Heard       string     `json:"heard"`
	Instruction string     `json:"instruction,omitempty"`
	TargetLang  string     `json:"target_lang,omitempty"`
	Selection   string     `json:"selection,omitempty"`
}

// FromEntry is what an entry exports; the selected text only when
// selection is on.
func FromEntry(e store.Entry, selection bool) Row {
	r := Row{ID: e.ID, CreatedAt: e.CreatedAt.Local(), Kind: e.Kind, App: e.AppName, Text: e.Result,
		Heard: e.RawText, Instruction: e.PromptText, TargetLang: e.TargetLang}
	if selection {
		r.Selection = e.SourceText
	}
	return r
}

// Walk gives each row of q to each, oldest first and finished entries
// only, a page at a time. Paging by the last row, not an offset: the
// clean-up deleting old rows meanwhile cannot make it skip one.
func Walk(ctx context.Context, h store.History, q store.HistoryQuery, selection func() bool, each func(Row) error) error {
	q.Finished, q.Oldest, q.Limit, q.Offset = true, true, PageSize, 0
	for {
		list, err := h.List(ctx, q)
		if err != nil {
			return err
		}
		for _, e := range list {
			if err := each(FromEntry(e, selection())); err != nil {
				return err
			}
		}
		if len(list) < PageSize {
			return nil
		}
		last := list[len(list)-1]
		q.After, q.AfterID = last.CreatedAt, last.ID
	}
}
