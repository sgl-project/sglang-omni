package main

import (
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// settingsSearchQuery is set while a settings section builds; settingRow
// hides rows that do not match (FB3). settingsSearchHits counts the ones
// that do, so the section can say when none does.
var (
	settingsSearchQuery string
	settingsSearchHits  int
)

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
