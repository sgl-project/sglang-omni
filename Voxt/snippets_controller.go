package main

// The Snippets page's side: reading and saving the snippets
// (internal/snippet), which sessions read for themselves
// (snippetsReader), and the page's actions.

import (
	"context"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/snippet"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type snippetsState struct {
	snip   snippetsData
	snipUI snippetsLocalUI
}

// snippetsData is the list as last read or saved.
type snippetsData struct {
	list            snippet.List
	loaded, loading bool
	err             string
	saves           snippetSaves
}

// snippetSaves orders the writes: a slower write of an older list does
// not replace a newer one.
type snippetSaves struct {
	mu            sync.Mutex
	next, written int
}

// snippetsReader is what a session reads its snippets with.
func snippetsReader(docs store.Documents) func(ctx context.Context) ([]snippet.Snippet, error) {
	return func(ctx context.Context) ([]snippet.Snippet, error) {
		l, err := readSnippets(ctx, docs)
		return l.Snippets, err
	}
}

func readSnippets(ctx context.Context, docs store.Documents) (snippet.List, error) {
	data, ok, err := docs.Get(ctx, snippet.Key)
	if err != nil || !ok {
		return snippet.List{}, err
	}
	return snippet.Unmarshal(data)
}

// loadSnippets reads the list once, as the page opens.
func (a *app) loadSnippets() {
	s := &a.snip
	if s.loaded || s.loading || a.lib == nil || a.lib.store == nil {
		return
	}
	s.loading = true
	docs := a.lib.store.Documents()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		l, err := readSnippets(ctx, docs)
		a.post(func() {
			s := &a.snip
			s.loading = false
			if err != nil {
				s.err = i18n.T("Could not read the snippets: %v", err)
				return
			}
			s.list, s.loaded, s.err = l, true, ""
		})
	})
}

// saveSnippets stores the list in the background. A list that was not
// read is never written: it would replace the saved one.
func (a *app) saveSnippets() {
	if a.lib == nil || a.lib.store == nil || !a.snip.loaded {
		return
	}
	data, err := a.snip.list.Marshal()
	if err != nil {
		return
	}
	saves := &a.snip.saves
	saves.mu.Lock()
	saves.next++
	seq := saves.next
	saves.mu.Unlock()
	docs := a.lib.store.Documents()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		saves.mu.Lock()
		if seq < saves.written {
			saves.mu.Unlock()
			return
		}
		err := docs.Put(ctx, snippet.Key, data)
		if err == nil {
			saves.written = seq
		}
		saves.mu.Unlock()
		a.post(func() {
			a.snip.err = ""
			if err != nil {
				a.snip.err = i18n.T("Could not save the snippets: %v", err)
			}
		})
	})
}

// saveSnippet adds a snippet (id "") or changes one, and says why it
// could not ("" when it did).
func (a *app) saveSnippet(id, trigger, text string) string {
	if !a.snip.loaded {
		return i18n.T("Could not read the snippets, so nothing was saved.")
	}
	s := snippet.Snippet{ID: id, Trigger: strings.TrimSpace(trigger), Text: strings.TrimRight(text, " \n")}
	if problem := snippet.Check(a.snip.list, s); problem != "" {
		return snippetProblemText(i18n.CurrentTranslator(), problem)
	}
	list := &a.snip.list.Snippets
	if id == "" {
		s.ID = strconv.FormatInt(time.Now().UnixNano(), 36)
		*list = append(*list, s)
	} else {
		for i := range *list {
			if (*list)[i].ID == id {
				(*list)[i] = s
			}
		}
	}
	a.saveSnippets()
	return ""
}

// deleteSnippet removes the snippet id.
func (a *app) deleteSnippet(id string) {
	if !a.snip.loaded {
		return
	}
	list := a.snip.list.Snippets[:0]
	for _, s := range a.snip.list.Snippets {
		if s.ID != id {
			list = append(list, s)
		}
	}
	a.snip.list.Snippets = list
	a.saveSnippets()
}

func (a *app) snippetsPageModel() snippetsViewState {
	s := &a.snip
	return buildSnippetsViewState(i18n.CurrentTranslator(), s.list, s.loaded, s.err, a.snipUI.query, a.snipUI.order)
}
