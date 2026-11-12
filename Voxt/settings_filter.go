package main

import (
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// settingsSearchQuery is set while a settings section builds; settingRow
// hides rows that do not match (FB3). settingsSearchHits counts the ones
// that do, so the section can say when none does.
var (
	settingsSearchQuery string
	settingsSearchHits  int
)

// settingsFind is Find in settings while a section builds (#3869): the
// matches of query in the names shown are marked (settingsName), n counts
// them as they build, cur is the current one (from 1, 0 for none), and
// reveal brings it into view in this frame.
var settingsFind struct {
	query  string
	n, cur int
	reveal bool
}

// settingsName is a row's or heading's name, the matches of Find in
// settings marked.
func settingsName(c *ui.Context, name string) ui.Element {
	f := &settingsFind
	m := len(findFolded(name, f.query))
	if m == 0 {
		return ui.Text(c, name)
	}
	cur := -1
	if f.cur > f.n && f.cur <= f.n+m {
		cur = f.cur - f.n - 1
	}
	f.n += m
	el := markedText(c, name, f.query, cur)
	if cur >= 0 && f.reveal {
		el.ScrollIntoView()
	}
	return el
}

// settingsSearchShows reports whether a row or heading with these texts
// matches settingsSearchQuery, counting it in settingsSearchHits.
func settingsSearchShows(parts ...string) bool {
	if !settingsTextMatches(settingsSearchQuery, parts...) {
		return false
	}
	if settingsSearchQuery != "" {
		settingsSearchHits++
	}
	return true
}

// settingsTextMatches reports whether every word of query is in parts, or
// in their English source text (a Chinese label is found by its English
// name too).
func settingsTextMatches(query string, parts ...string) bool {
	query = strings.ToLower(strings.TrimSpace(query))
	if query == "" {
		return true
	}
	var b strings.Builder
	for _, p := range parts {
		if strings.TrimSpace(p) == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(p)
		if en := i18n.Source(p); en != "" {
			b.WriteByte(' ')
			b.WriteString(en)
		}
	}
	hay := strings.ToLower(b.String())
	for _, term := range strings.Fields(query) {
		if !strings.Contains(hay, term) {
			return false
		}
	}
	return true
}
