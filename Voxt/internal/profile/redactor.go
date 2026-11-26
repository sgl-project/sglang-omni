package profile

import (
	"encoding/json"
	"sort"
	"strings"
	"sync"
)

const redactedPlaceholder = "[redacted]"

// Redactor removes configured service details and credentials from every
// diagnostic produced by a profiling run.
type Redactor struct {
	mu     sync.Mutex
	values map[string]struct{}
}

func NewRedactor(values ...string) *Redactor {
	r := &Redactor{values: map[string]struct{}{}}
	r.Register(values...)
	return r
}

func (r *Redactor) Register(values ...string) {
	r.mu.Lock()
	for _, value := range values {
		if value != "" {
			r.values[value] = struct{}{}
		}
	}
	r.mu.Unlock()
}

func (r *Redactor) orderedValues() []string {
	values := make([]string, 0, len(r.values))
	for value := range r.values {
		values = append(values, value)
	}
	sort.Slice(values, func(i, j int) bool { return len(values[i]) > len(values[j]) })
	return values
}

func (r *Redactor) Clean(text string) string {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, value := range r.orderedValues() {
		text = strings.ReplaceAll(text, value, redactedPlaceholder)
	}
	return text
}

// CleanJSON uses each value's JSON string encoding, so credentials with
// quotes or control characters cannot evade the final report scrub.
func (r *Redactor) CleanJSON(text string) string {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, value := range r.orderedValues() {
		encoded, _ := json.Marshal(value)
		escaped := string(encoded[1 : len(encoded)-1])
		text = strings.ReplaceAll(text, escaped, redactedPlaceholder)
	}
	return text
}
