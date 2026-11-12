package meeting

import (
	"strings"
	"time"
	"unicode"
	"unicode/utf8"
)

const (
	silenceFinalizeLive = 750 * time.Millisecond
	maximumLiveSegment  = 8 * time.Second
)

// Committer turns streaming partial ASR text into sentence commits (Voxt 1.x live rules).
type Committer struct {
	cumulative  string
	committed   string
	phraseStart time.Duration
}

// Partial records the latest cumulative hypothesis from EventPartial.
func (c *Committer) Partial(text string) {
	c.cumulative = text
}

// Live returns the text that is not committed yet (for overlay “speaking” line).
func (c *Committer) Live() string {
	return uncommittedSuffix(c.cumulative, c.committed)
}

// BeginPhrase marks the timeline start of the current uncommitted phrase.
func (c *Committer) BeginPhrase(at time.Duration) {
	c.phraseStart = at
}

// Tick commits when silence reaches 0.75 s or the phrase hits 8 s without a boundary.
func (c *Committer) Tick(now, silence time.Duration) (Segment, bool) {
	tail := uncommittedSuffix(c.cumulative, c.committed)
	if tail == "" {
		return Segment{}, false
	}
	phrase := now - c.phraseStart
	if silence < silenceFinalizeLive && phrase < maximumLiveSegment {
		return Segment{}, false
	}
	seg := Segment{
		Text:  tail,
		Start: c.phraseStart,
		End:   now,
		Final: false,
	}
	c.committed = c.cumulative
	c.phraseStart = now
	return seg, true
}

// Final commits any tail left at EventFinal.
func (c *Committer) Final(text string, now time.Duration) (Segment, bool) {
	c.cumulative = text
	tail := uncommittedSuffix(c.cumulative, c.committed)
	if tail == "" {
		return Segment{}, false
	}
	seg := Segment{
		Text:  tail,
		Start: c.phraseStart,
		End:   now,
		Final: true,
	}
	c.committed = c.cumulative
	c.phraseStart = now
	return seg, true
}

// uncommittedSuffix is the tail of cumulative that was not committed yet.
func uncommittedSuffix(cumulative, committed string) string {
	cumulative = strings.TrimSpace(cumulative)
	if cumulative == "" {
		return ""
	}
	if committed == "" {
		return cumulative
	}
	// The model may revise the committed tail after a pause (#3431): emit only
	// what follows the longest shared prefix, so the sentence is not repeated.
	i := 0
	for i < len(committed) && i < len(cumulative) && committed[i] == cumulative[i] {
		i++
	}
	for i > 0 && i < len(cumulative) && !utf8.RuneStart(cumulative[i]) {
		i--
	}
	return trimLeadPunct(cumulative[i:])
}

func trimLeadPunct(s string) string {
	return strings.TrimLeftFunc(s, func(r rune) bool {
		return unicode.IsSpace(r) || r == '。' || r == '，' || r == '、' || r == '.' || r == ','
	})
}
