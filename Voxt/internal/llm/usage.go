package llm

import (
	"slices"
	"strings"
	"sync"
	"time"
	"unicode"
	"unicode/utf8"
)

// EstimateTokens is a rough token count for s when a backend reports
// none: about one token per CJK character and per four bytes of other text.
func EstimateTokens(s string) int {
	n, rest := 0, 0
	for _, r := range s {
		if r >= 0x2E80 && (unicode.Is(unicode.Han, r) || unicode.Is(unicode.Hiragana, r) ||
			unicode.Is(unicode.Katakana, r) || unicode.Is(unicode.Hangul, r)) {
			n++
			continue
		}
		rest += utf8.RuneLen(r)
	}
	return n + (rest+3)/4
}

func estimateUsage(msgs []Message, text string) Usage {
	p := 0
	for _, m := range msgs {
		p += EstimateTokens(m.Content) + 4
	}
	c := EstimateTokens(text)
	return Usage{PromptTokens: p, CompletionTokens: c, TotalTokens: p + c, Estimated: true}
}

// UsageKey groups usage records.
type UsageKey struct {
	Provider string
	Model    string
	Skill    Skill
}

// UsageRecord is the running total for one key.
type UsageRecord struct {
	UsageKey
	Calls    int
	Failures int
	Attempts int
	Usage    Usage
	Elapsed  time.Duration // summed wall time, retries included
	Last     time.Time
}

// Meter accumulates usage. Goroutine-safe; a nil *Meter ignores records.
type Meter struct {
	mu   sync.Mutex
	recs map[UsageKey]*UsageRecord
	// OnRecord, if set, is called after every call (outside the lock) with
	// the key, that call's usage and its error code ("" on success). The
	// app persists usage stats from it. It must not block.
	OnRecord func(k UsageKey, u Usage, code string)
	now      func() time.Time
}

func (m *Meter) record(provider string, req Request, resp Response, err error, elapsed time.Duration) {
	if m == nil {
		return
	}
	model := resp.Model
	if model == "" {
		model = req.Model
	}
	k := UsageKey{Provider: provider, Model: model, Skill: req.Skill}
	now := time.Now
	if m.now != nil {
		now = m.now
	}
	m.mu.Lock()
	if m.recs == nil {
		m.recs = map[UsageKey]*UsageRecord{}
	}
	r := m.recs[k]
	if r == nil {
		r = &UsageRecord{UsageKey: k}
		m.recs[k] = r
	}
	r.Calls++
	if err != nil {
		r.Failures++
	}
	r.Attempts += max(resp.Attempts, 1)
	r.Usage = r.Usage.Add(resp.Usage)
	r.Elapsed += elapsed
	r.Last = now()
	cb := m.OnRecord
	m.mu.Unlock()
	if cb != nil {
		cb(k, resp.Usage, Code(err))
	}
}

// Snapshot returns a copy of every record, sorted by backend, model, skill.
func (m *Meter) Snapshot() []UsageRecord {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	out := make([]UsageRecord, 0, len(m.recs))
	for _, r := range m.recs {
		out = append(out, *r)
	}
	m.mu.Unlock()
	slices.SortFunc(out, func(a, b UsageRecord) int {
		if c := strings.Compare(a.Provider, b.Provider); c != 0 {
			return c
		}
		if c := strings.Compare(a.Model, b.Model); c != 0 {
			return c
		}
		return strings.Compare(string(a.Skill), string(b.Skill))
	})
	return out
}

// Total sums every record.
func (m *Meter) Total() Usage {
	var u Usage
	for _, r := range m.Snapshot() {
		u = u.Add(r.Usage)
	}
	return u
}

// Reset clears the records.
func (m *Meter) Reset() {
	if m == nil {
		return
	}
	m.mu.Lock()
	m.recs = nil
	m.mu.Unlock()
}
