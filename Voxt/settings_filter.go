package main

import "strings"

// settingsSearchQuery is set while a settings section builds; settingRow
// hides rows that do not match (FB3).
var settingsSearchQuery string

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
	}
	hay := strings.ToLower(b.String())
	for _, term := range strings.Fields(query) {
		if !strings.Contains(hay, term) {
			return false
		}
	}
	return true
}
