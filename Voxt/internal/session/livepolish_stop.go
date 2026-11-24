package session

import (
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
)

// note (luojiaxuan): finish ends the live clean-up with the final text (as cleaned, aliases
// replaced): the clean-ups it keeps, as parts and their answers, and the
// text after them (reuse). A request under way for exactly the text after
// the blocks is awaited: it is ahead of a new one for the same text; any
// other is cancelled. A clean-up asked with thinking other than the final
// text's (ThinkingOffUnder) does not count.
func (lp *livePolish) finish(text string) (parts []compose.Part, answers []string, tail string) {
	chars := utf8.RuneCountInString(text)
	thinkOff := lp.thinkOffUnder > 0 && chars < lp.thinkOffUnder
	lp.mu.Lock()
	lp.closed = true
	lp.cond.Broadcast()
	cs, open, all := lp.chains(text, thinkOff)
	keep := lp.reuse(cs, text)
	var wait *liveFlight
	if f := lp.flight; f != nil {
		if all && f.epoch == lp.epoch && f.thinkOff == thinkOff && f.src == open {
			wait = f
		} else {
			f.cancel()
		}
	}
	blocks := slices.Clone(lp.blocks)
	asked, dropped := lp.asked, lp.dropped
	lp.mu.Unlock()
	if wait != nil {
		<-wait.done
		if wait.ok {
			keep = append(blocks, wait.liveResult)
		}
	}
	lp.cancel()
	<-lp.workerDone
	parts, answers, tail = laid(keep, text)
	lp.r.logf("info", "the live clean-up covered %d of %d characters (%d requests while recording, %d cancelled as the text changed; awaited the request under way: %t)",
		chars-utf8.RuneCountInString(tail), chars, asked, dropped, wait != nil)
	return parts, answers, tail
}

// note (luojiaxuan): chain is clean-ups that start the final text, in order, and the byte
// of the text after them (the end of one of its sentences).
type chain struct {
	results []liveResult
	end     int
}

// note (luojiaxuan): chains are the ways the clean-ups start text (lp.mu held): with none;
// with each committed block that starts it after the ones before, or one
// of the block's shorter stretches (its prefixes) in its place; then with
// all the blocks and one of the open block's clean-ups. Only clean-ups
// asked with thinking as thinkOff count. open is the text after the
// blocks, and all whether they all start text.
func (lp *livePolish) chains(text string, thinkOff bool) (cs []chain, open string, all bool) {
	cs = []chain{{}}
	var before []liveResult
	after := func(candidates []liveResult, rest string) {
		at := len(text) - len(rest)
		for _, c := range candidates {
			if c.thinkOff == thinkOff && endsSentence(rest, c.src) {
				cs = append(cs, chain{append(slices.Clone(before), c), at + len(c.src)})
			}
		}
	}
	end := 0
	for _, b := range lp.blocks {
		rest := strings.TrimLeftFunc(text[end:], unicode.IsSpace)
		after(b.prefixes, rest)
		if b.thinkOff != thinkOff || !endsSentence(rest, b.src) {
			return cs, "", false
		}
		before = append(before, b)
		end = len(text) - len(rest) + len(b.src)
		cs = append(cs, chain{slices.Clone(before), end})
	}
	open = strings.TrimLeftFunc(text[end:], unicode.IsSpace)
	after(lp.done, open)
	return cs, open, true
}

// note (luojiaxuan): reuse is the chain whose clean-ups Stop keeps (lp.mu held): the
// longest; when it leaves text to clean, the longest that ends
// lp.reclean sentences before it, so that what was said last can correct
// those sentences, cleaned again with it.
func (lp *livePolish) reuse(cs []chain, text string) []liveResult {
	longest := slices.MaxFunc(cs, func(a, b chain) int { return a.end - b.end })
	if lp.reclean == 0 || strings.TrimSpace(text[longest.end:]) == "" {
		return longest.results
	}
	var ends []int
	for _, e := range compose.SentenceEnds(text) {
		if e < longest.end {
			ends = append(ends, e)
		}
	}
	upTo := 0
	if len(ends) >= lp.reclean {
		upTo = ends[len(ends)-lp.reclean]
	}
	best := cs[0]
	for _, c := range cs {
		if c.end <= upTo && c.end > best.end {
			best = c
		}
	}
	return best.results
}

// note (luojiaxuan): liveContext is what a request reads of the cleaned text before its
// stretch: its end, from a sentence's start, at most compose.PieceChars
// characters (a block's size), however long the dictation, which keeps the
// whole text to insert.
func liveContext(cleaned string) string {
	if utf8.RuneCountInString(cleaned) <= compose.PieceChars {
		return cleaned
	}
	for _, end := range compose.SentenceEnds(cleaned) {
		if rest := strings.TrimLeftFunc(cleaned[end:], unicode.IsSpace); rest != "" && utf8.RuneCountInString(rest) <= compose.PieceChars {
			return rest
		}
	}
	runes := []rune(cleaned)
	return string(runes[len(runes)-compose.PieceChars:])
}

// note (luojiaxuan): chainIn is how many of blocks start text, in order, each at one of its
// sentence ends, and the byte of text after them.
func chainIn(text string, blocks []liveResult) (int, int) {
	end := 0
	for k, b := range blocks {
		rest := text[end:]
		trimmed := strings.TrimLeftFunc(rest, unicode.IsSpace)
		if !endsSentence(trimmed, b.src) {
			return k, end
		}
		end += len(rest) - len(trimmed) + len(b.src)
	}
	return len(blocks), end
}

// note (luojiaxuan): endsSentence says whether text starts with src and src ends at one of
// text's sentence ends, or is all of it.
func endsSentence(text, src string) bool {
	if src == "" || !strings.HasPrefix(text, src) {
		return false
	}
	return len(src) == len(text) || slices.Contains(compose.SentenceEnds(text), len(src))
}

// note (luojiaxuan): joinResults is the blocks' cleaned text, joined as text, which they
// start, joins them.
func joinResults(blocks []liveResult, text string) string {
	parts, answers, _ := laid(blocks, text)
	return compose.JoinParts(parts, answers)
}

// note (luojiaxuan): laid are clean-ups that start text as parts of it (each one's last part
// After the white space after it in text), and the text after them.
func laid(results []liveResult, text string) (parts []compose.Part, answers []string, tail string) {
	rest := strings.TrimLeftFunc(text, unicode.IsSpace)
	for _, b := range results {
		after := rest[len(b.src):]
		rest = strings.TrimLeftFunc(after, unicode.IsSpace)
		ps := slices.Clone(b.parts)
		ps[len(ps)-1].After = after[:len(after)-len(rest)]
		parts, answers = append(parts, ps...), append(answers, b.answers...)
	}
	return parts, answers, rest
}
