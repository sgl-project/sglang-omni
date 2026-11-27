package compose

import (
	"strings"
	"unicode"
)

// withGlossary is a user message msg after the glossary block, if any.
func (o Output) withGlossary(msg string) string {
	if o.Glossary == "" {
		return msg
	}
	return o.Glossary + "\n\n" + msg
}

// Glossary lists the terms that fit the limits, one per line ("- Term
// (heard as: alias, alias)"): a dictionary of at most GlossaryAllUpTo
// terms whole, in its order, whatever text says (the system prompt stays
// the same from one dictation to the next, which the model's prompt cache
// keeps); a larger one only with the terms found in text (their spelling
// or an alias, ignoring case and spaces). "" when there are none.
func Glossary(terms []Term, text, set string) string {
	norm := func(s string) string {
		return strings.Map(func(r rune) rune {
			if unicode.IsSpace(r) || r == '-' || r == '_' {
				return -1
			}
			return unicode.ToLower(r)
		}, s)
	}
	ntext := norm(text)
	found := func(t Term) bool {
		if ntext == "" {
			return false
		}
		for _, s := range append([]string{t.Term}, t.Aliases...) {
			if n := norm(s); n != "" && strings.Contains(ntext, n) {
				return true
			}
		}
		return false
	}
	all := distinctTerms(terms)
	var mentioned []Term
	for _, t := range all {
		if found(t) {
			mentioned = append(mentioned, t)
		}
	}
	heard := "heard as"
	sep := ", "
	if set == Chinese {
		heard, sep = "可能被听成", "、"
	}
	if len(all) > GlossaryAllUpTo {
		all = mentioned
	}
	var lines []string
	size := 0
	for _, t := range all {
		line := "- " + t.Term
		var aliases []string
		for _, a := range t.Aliases {
			if a = strings.TrimSpace(a); a != "" && !strings.EqualFold(a, t.Term) {
				aliases = append(aliases, a)
			}
		}
		if len(aliases) > 0 {
			if set == Chinese {
				line += "（" + heard + "：" + strings.Join(aliases, sep) + "）"
			} else {
				line += " (" + heard + ": " + strings.Join(aliases, sep) + ")"
			}
		}
		if len(lines) == MaxGlossaryTerms || size+len(line) > MaxGlossaryChars {
			break
		}
		lines = append(lines, line)
		size += len(line) + 1
	}
	return strings.Join(lines, "\n")
}

// distinctTerms are terms trimmed, each once (ignoring case), without
// empty ones.
func distinctTerms(terms []Term) []Term {
	var out []Term
	seen := map[string]bool{}
	for _, t := range terms {
		t.Term = strings.TrimSpace(t.Term)
		if t.Term == "" || seen[strings.ToLower(t.Term)] {
			continue
		}
		seen[strings.ToLower(t.Term)] = true
		out = append(out, t)
	}
	return out
}
