package learn

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

const (
	scanBatchEntries = 12
	scanRecordChars  = 320 // of an entry's text: its start is enough to find its names
)

var scanSchema = func() *llm.Schema {
	s := llm.Object(map[string]*llm.Schema{"terms": llm.Array(llm.String())}, "terms")
	s.Name = "scan"
	return s
}()

// Languages are the user's: settings.General's main language and others.
type Languages struct {
	Main   string
	Others []string
}

// Scanner proposes the names and terms of the Transcribe history the last
// scan did not read: 12 entries a request, oldest first.
type Scanner struct {
	History store.History
	Book    *Book
}

// scanRecord is what a scan sends of an entry: its result, or the speech
// when it has none, cut to scanRecordChars.
func scanRecord(e store.Entry) string {
	text := e.Result
	if strings.TrimSpace(text) == "" {
		text = e.RawText
	}
	text = strings.TrimSpace(text)
	if characters := []rune(text); len(characters) > scanRecordChars {
		text = string(characters[:scanRecordChars])
	}
	return text
}

// unread are the finished Transcribe entries past the checkpoint, oldest
// first, in the order of History.List (created_at, id).
func (s *Scanner) unread(ctx context.Context) ([]store.Entry, error) {
	entries, err := s.History.List(ctx, store.HistoryQuery{Kind: store.KindTranscribe})
	if err != nil {
		return nil, err
	}
	checkpoint, err := s.Book.scanned(ctx)
	if err != nil {
		return nil, err
	}
	entries = slices.DeleteFunc(entries, func(e store.Entry) bool {
		return e.Status != store.StatusDone || scanRecord(e) == "" ||
			cmp.Or(e.CreatedAt.Compare(checkpoint.At), cmp.Compare(e.ID, checkpoint.ID)) <= 0
	})
	slices.SortFunc(entries, func(x, y store.Entry) int {
		return cmp.Or(x.CreatedAt.Compare(y.CreatedAt), cmp.Compare(x.ID, y.ID))
	})
	return entries, nil
}

// Pending counts the entries a scan would read.
func (s *Scanner) Pending(ctx context.Context) (int, error) {
	entries, err := s.unread(ctx)
	return len(entries), err
}

// Scan reads them with the model open gives, in the user's languages,
// moving the checkpoint after each request. A model other than modelID,
// the one the user was told about, is an error before anything is sent.
// At ctx's end or a request's failure it stops, keeping what it found; it
// reports how many suggestions are new.
func (s *Scanner) Scan(ctx context.Context, modelID string, open func(context.Context) (local.Chat, local.Model, error),
	languages Languages, progress func(done, total, found int)) (int, error) {
	chat, model, err := open(ctx)
	if err != nil {
		return 0, err
	}
	if model.ID != modelID {
		return 0, fmt.Errorf("learn: the model opened is %s, not %s", model.ID, modelID)
	}
	entries, err := s.unread(ctx)
	if err != nil {
		return 0, err
	}
	found := 0
	for start := 0; start < len(entries); start += scanBatchEntries {
		if err := ctx.Err(); err != nil {
			return found, err
		}
		batch := entries[start:min(start+scanBatchEntries, len(entries))]
		records := make([]string, len(batch))
		for i, e := range batch {
			records[i] = scanRecord(e)
		}
		messages := compose.Scan(records, languages.Main, languages.Others).Messages()
		request := llm.Request{Format: llm.FormatJSON, Schema: scanSchema}
		for _, m := range messages {
			request.Messages = append(request.Messages, llm.Message{Role: llm.Role(m.Role), Content: m.Content})
		}
		response, err := chat.Chat(ctx, request)
		if err != nil {
			return found, err
		}
		var answer struct {
			Terms []string `json:"terms"`
		}
		if err := json.Unmarshal(llm.ExtractJSON(response.Text), &answer); err != nil {
			return found, &llm.Error{Code: llm.CodeInvalidJSON, Err: err}
		}
		var proposed []Suggestion
		seen := map[string]bool{}
		for _, term := range answer.Terms {
			key := fold(term)
			if len(proposed) < compose.MaxScanTerms && !seen[key] && keep(term, records) {
				seen[key] = true
				proposed = append(proposed, Suggestion{Term: strings.TrimSpace(term), Source: FromHistory})
			}
		}
		added, err := s.Book.Propose(ctx, proposed)
		if err != nil {
			return found, err
		}
		found += added
		last := batch[len(batch)-1]
		if err := s.Book.setScanned(ctx, scanCheckpoint{At: last.CreatedAt, ID: last.ID}); err != nil {
			return found, err
		}
		progress(start+len(batch), len(entries), found)
	}
	return found, nil
}
