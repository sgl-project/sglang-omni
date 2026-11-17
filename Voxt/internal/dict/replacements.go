package dict

import "strings"

// Replacements holds directional rules. Text inserted by a rule is not
// scanned again, so overlapping sources are replaced once in input order.
type Replacements struct {
	patterns []replacePattern
}

// ParseReplacements accepts one source → replacement per line, with -> as
// an ASCII alternative. A nonzero line number identifies an invalid rule.
func ParseReplacements(text string) (Replacements, int) {
	var rules Replacements
	for i, line := range strings.Split(text, "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		from, to, ok := strings.Cut(line, "→")
		if !ok {
			from, to, ok = strings.Cut(line, "->")
		}
		from, to = strings.TrimSpace(from), strings.TrimSpace(to)
		if !ok || from == "" || to == "" {
			return Replacements{}, i + 1
		}
		rules.patterns = addPattern(rules.patterns, from, to)
	}
	sortPatterns(rules.patterns)
	return rules, 0
}

func (r Replacements) Replace(text string) string {
	return applyPatterns(text, r.patterns)
}
