package compose

import (
	"strconv"
	"strings"
)

// MaxScanTerms is the most terms a history scan's request may return.
const MaxScanTerms = 12

// Scan is the history scan's request: the records and the user's
// languages, nothing of the dictionary. The answer is
// {"terms": ["OpenAI", "MCP"]}.
func Scan(records []string, mainLanguage string, others []string) Output {
	set := TemplateSet(mainLanguage)
	vars := Variables(Input{MainLanguage: mainLanguage, OtherLanguages: others}, set)
	wrapped := make([]string, len(records))
	for i, record := range records {
		wrapped[i] = wrap("record", record)
	}
	return Output{
		System:   Render(template(set, "scan"), vars, map[string]string{"MAX_TERMS": strconv.Itoa(MaxScanTerms)}),
		User:     strings.Join(wrapped, "\n"),
		Language: set,
		Layers:   []string{"scan"},
	}
}
