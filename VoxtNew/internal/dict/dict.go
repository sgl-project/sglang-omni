// Package dict is the user's dictionary as the rest of VoxtNew uses it
// (plan §2.3, D4; 0.1 is kept by hand): checking a term, the hints sent to
// speech recognition, and importing and exporting the terms as CSV or
// plain text. The terms live in the store (store.Dictionary); the
// language model's glossary is the composer's (compose.Glossary).
//
// Imports: store.
package dict

import (
	"bytes"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"slices"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// Limits of a term.
const (
	MaxTermChars  = 80
	MaxAliases    = 10
	MaxNoteChars  = 500
	MaxImportRows = 10000
)

// Errors of Check.
var (
	ErrEmpty   = errors.New("dict: the term is empty")
	ErrLong    = errors.New("dict: the term is too long")
	ErrNewline = errors.New("dict: a term is one line")
)

// Clean trims a term, its aliases (one line each, without blanks, the
// term itself and duplicates, ignoring case; at most MaxAliases) and its
// note.
func Clean(t store.Term) store.Term {
	t.Term = oneLine(t.Term)
	t.Note = strings.TrimSpace(t.Note)
	var aliases []string
	for _, a := range t.Aliases {
		a = oneLine(a)
		if a == "" || strings.EqualFold(a, t.Term) || slices.ContainsFunc(aliases, func(b string) bool { return strings.EqualFold(a, b) }) {
			continue
		}
		if len(aliases) == MaxAliases {
			break
		}
		aliases = append(aliases, a)
	}
	t.Aliases = aliases
	return t
}

func oneLine(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

// Check says whether a term can be kept (after Clean).
func Check(t store.Term) error {
	switch {
	case strings.TrimSpace(t.Term) == "":
		return ErrEmpty
	case strings.ContainsAny(strings.TrimSpace(t.Term), "\r\n"):
		return ErrNewline
	case utf8.RuneCountInString(strings.TrimSpace(t.Term)) > MaxTermChars:
		return ErrLong
	}
	return nil
}

// SplitAliases reads aliases typed in one field: separated by commas
// (also "，", "、") or semicolons.
func SplitAliases(s string) []string {
	f := func(r rune) bool { return r == ',' || r == ';' || r == '，' || r == '、' || r == '；' || r == '\n' }
	var out []string
	for _, a := range strings.FieldsFunc(s, f) {
		if a = strings.TrimSpace(a); a != "" {
			out = append(out, a)
		}
	}
	return out
}

// Hints are the terms sent to speech recognition: the enabled ones, each
// once (ignoring case), the most recently changed first (a service that
// takes fewer cuts the list from the end). Not the aliases: they are what
// recognition gets wrong.
func Hints(terms []store.Term) []string {
	ts := slices.Clone(terms)
	sort.SliceStable(ts, func(i, j int) bool { return ts[i].UpdatedAt.After(ts[j].UpdatedAt) })
	var out []string
	seen := map[string]bool{}
	for _, t := range ts {
		w := oneLine(t.Term)
		if !t.Enabled || w == "" || seen[strings.ToLower(w)] {
			continue
		}
		seen[strings.ToLower(w)] = true
		out = append(out, w)
	}
	return out
}

// ---- Import and export ----

// Formats.
const (
	CSV  = "csv"
	Text = "txt"
)

// FormatOf is the format of a file by its name (".txt" is Text, else
// CSV).
func FormatOf(name string) string {
	if strings.HasSuffix(strings.ToLower(name), ".txt") {
		return Text
	}
	return CSV
}

// csvHeader is the first row of an exported CSV.
var csvHeader = []string{"term", "aliases", "note", "enabled"}

// Export writes terms in format: CSV with a header (term, aliases joined
// by "; ", note, enabled as true/false), or text, a term per line with its
// aliases after " = ", disabled terms commented out.
func Export(terms []store.Term, format string) []byte {
	var b bytes.Buffer
	if format == Text {
		b.WriteString("# VoxtNew dictionary: a term per line, \"Term = alias, alias\" for what speech recognition hears instead.\n")
		for _, t := range terms {
			line := t.Term
			if len(t.Aliases) > 0 {
				line += " = " + strings.Join(t.Aliases, ", ")
			}
			if !t.Enabled {
				line = "# " + line
			}
			b.WriteString(line + "\n")
		}
		return b.Bytes()
	}
	b.WriteString("\ufeff") // Excel reads UTF-8 with a BOM
	w := csv.NewWriter(&b)
	w.Write(csvHeader)
	for _, t := range terms {
		w.Write([]string{t.Term, strings.Join(t.Aliases, "; "), t.Note, fmt.Sprint(t.Enabled)})
	}
	w.Flush()
	return b.Bytes()
}

// ErrFormat: the file is not a dictionary Import reads.
var ErrFormat = errors.New("dict: not a dictionary file")

// Import reads terms (without ids or times) from data in format. CSV:
// the columns term, aliases, note, enabled, by a header naming them or in
// that order; text: a term per line, "Term = alias, alias", "#" for a
// comment. Rows without a term are skipped (skipped counts them).
func Import(data []byte, format string) (terms []store.Term, skipped int, err error) {
	data = bytes.TrimPrefix(data, []byte("\ufeff"))
	if !utf8.Valid(data) {
		return nil, 0, fmt.Errorf("%w: not UTF-8 text", ErrFormat)
	}
	add := func(t store.Term) {
		t = Clean(t)
		if Check(t) != nil {
			skipped++
			return
		}
		terms = append(terms, t)
	}
	if format == Text {
		for _, line := range strings.Split(string(data), "\n") {
			line = strings.TrimSpace(line)
			if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, "//") {
				continue
			}
			term, aliases, _ := strings.Cut(line, " = ")
			if t, a, ok := strings.Cut(line, "\t"); ok && !strings.Contains(line, " = ") {
				term, aliases = t, a
			}
			add(store.Term{Term: term, Aliases: SplitAliases(aliases), Enabled: true})
			if len(terms) > MaxImportRows {
				return nil, 0, fmt.Errorf("%w: more than %d terms", ErrFormat, MaxImportRows)
			}
		}
		return terms, skipped, nil
	}
	r := csv.NewReader(bytes.NewReader(data))
	r.FieldsPerRecord = -1
	r.LazyQuotes = true
	r.TrimLeadingSpace = true
	col := map[string]int{"term": 0, "aliases": 1, "note": 2, "enabled": 3}
	first := true
	for {
		rec, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, 0, fmt.Errorf("%w: %v", ErrFormat, err)
		}
		if first {
			first = false
			if h := headerColumns(rec); h != nil {
				col = h
				continue
			}
		}
		get := func(name string) string {
			if i, ok := col[name]; ok && i < len(rec) {
				return rec[i]
			}
			return ""
		}
		add(store.Term{Term: get("term"), Aliases: SplitAliases(get("aliases")), Note: get("note"), Enabled: parseBool(get("enabled"), true)})
		if len(terms) > MaxImportRows {
			return nil, 0, fmt.Errorf("%w: more than %d terms", ErrFormat, MaxImportRows)
		}
	}
	return terms, skipped, nil
}

// headerColumns maps a CSV header's names to their columns, nil when rec
// is not a header (it has no "term" column).
func headerColumns(rec []string) map[string]int {
	names := map[string]string{
		"term": "term", "word": "term", "词条": "term", "术语": "term",
		"aliases": "aliases", "alias": "aliases", "别名": "aliases",
		"note": "note", "notes": "note", "备注": "note",
		"enabled": "enabled", "enable": "enabled", "启用": "enabled",
	}
	out := map[string]int{}
	for i, h := range rec {
		if n, ok := names[strings.ToLower(strings.TrimSpace(h))]; ok {
			if _, dup := out[n]; !dup {
				out[n] = i
			}
		}
	}
	if _, ok := out["term"]; !ok {
		return nil
	}
	return out
}

func parseBool(s string, def bool) bool {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "1", "true", "yes", "y", "on", "是", "启用":
		return true
	case "0", "false", "no", "n", "off", "否", "停用", "禁用":
		return false
	}
	return def
}

// Merge applies imported terms to the dictionary have: a new spelling
// (ignoring case) is added as given, a known one gets the aliases it
// lacked and the note when it had none. It returns the terms to save
// (without ids for the new ones) and how many were added and updated.
func Merge(have, imported []store.Term) (put []store.Term, added, updated int) {
	byTerm := map[string]int{}
	cur := slices.Clone(have)
	for i, t := range cur {
		byTerm[strings.ToLower(strings.TrimSpace(t.Term))] = i
	}
	changed := map[int]bool{}
	for _, in := range imported {
		key := strings.ToLower(strings.TrimSpace(in.Term))
		i, ok := byTerm[key]
		if !ok {
			byTerm[key] = len(cur)
			cur = append(cur, in)
			changed[len(cur)-1] = true
			added++
			continue
		}
		t := cur[i]
		before := len(t.Aliases)
		t = Clean(store.Term{ID: t.ID, Term: t.Term, Aliases: append(slices.Clone(t.Aliases), in.Aliases...), Note: t.Note,
			Enabled: t.Enabled, CreatedAt: t.CreatedAt, UpdatedAt: t.UpdatedAt})
		note := false
		if t.Note == "" && in.Note != "" {
			t.Note, note = in.Note, true
		}
		if len(t.Aliases) != before || note {
			cur[i] = t
			if !changed[i] && i < len(have) {
				updated++
			}
			changed[i] = true
		}
	}
	for i := range cur {
		if changed[i] {
			put = append(put, cur[i])
		}
	}
	return put, added, updated
}
