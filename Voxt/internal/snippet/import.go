package snippet

import (
	"encoding/json"
	"errors"
	"strings"
)

// Import limits: what a file may hold.
const (
	MaxImportBytes = 3 << 20
	MaxImportItems = 1000
)

// ErrNotSnippets is a file Import cannot read as snippets.
var ErrNotSnippets = errors.New("not a list of snippets")

// Import reads snippets from a JSON file: a list of objects, or Voxt's
// own {"snippets": [...]}. An object's trigger is its "trigger",
// "keyword" or "name" (Wispr Flow's), its text its "text", "expansion",
// "content" or "snippet". Items without both, or over the limits, are
// skipped and counted.
func Import(data []byte) (out []Snippet, skipped int, err error) {
	if len(data) > MaxImportBytes {
		return nil, 0, ErrNotSnippets
	}
	var items []map[string]any
	if err := json.Unmarshal(data, &items); err != nil {
		var doc struct {
			Snippets []map[string]any `json:"snippets"`
		}
		if json.Unmarshal(data, &doc) != nil || doc.Snippets == nil {
			return nil, 0, ErrNotSnippets
		}
		items = doc.Snippets
	}
	for i, item := range items {
		s := Snippet{
			Trigger: strings.TrimSpace(field(item, "trigger", "keyword", "name")),
			Text:    strings.TrimRight(strings.ReplaceAll(field(item, "text", "expansion", "content", "snippet"), "\r\n", "\n"), " \r\n"),
		}
		if i >= MaxImportItems || len(normalize(s.Trigger)) == 0 || strings.TrimSpace(s.Text) == "" ||
			len([]rune(s.Trigger)) > MaxTrigger || len([]rune(s.Text)) > MaxText {
			skipped++
			continue
		}
		out = append(out, s)
	}
	return out, skipped, nil
}

// field is item's first string among keys.
func field(item map[string]any, keys ...string) string {
	for _, k := range keys {
		if v, ok := item[k].(string); ok && strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

// Merge adds in to l: a snippet whose trigger l has (as Check compares
// them) changes that one's text, the others are added with newID's IDs.
func Merge(l List, in []Snippet, newID func() string) (out List, added, updated int) {
	out.Snippets = append([]Snippet(nil), l.Snippets...)
	for _, s := range in {
		key := string(normalize(s.Trigger))
		found := false
		for i := range out.Snippets {
			if string(normalize(out.Snippets[i].Trigger)) == key {
				if out.Snippets[i].Text != s.Text {
					out.Snippets[i].Text = s.Text
					updated++
				}
				found = true
				break
			}
		}
		if !found {
			s.ID = newID()
			out.Snippets = append(out.Snippets, s)
			added++
		}
	}
	return out, added, updated
}
