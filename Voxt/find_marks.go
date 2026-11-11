package main

// One Find for the search boxes (#3869), as Find in a word processor:
// every match of the query is marked in red, the current one white on
// red; Return goes to the next match, Shift+Return to the previous, and
// "N of M" counts them. Matches ignore case (findFolded).

import (
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// finder is a search box's place among its matches: cur is the current
// one, from 1, 0 until Return; step counts the Returns pressed since the
// last frame (+1 each, -1 with Shift).
type finder struct {
	cur, step int
}

// field is searchField stepping f with Return and Shift+Return; typing
// starts the count again.
func (f *finder) field(c *ui.Context, key string, value *string, placeholder string, maxW float32) ui.Element {
	in := searchField(c, key, value, placeholder, maxW)
	in.OnTextKey(func(k ui.TextKey) bool {
		if k.Key != ui.KeyEnter || (k.Mods != 0 && k.Mods != ui.Shift) {
			return false
		}
		if k.Mods == ui.Shift {
			f.step--
		} else {
			f.step++
		}
		return true
	})
	if in.Changed() {
		f.cur = 0
	}
	return in
}

// advance takes the step pressed over n matches, wrapping around, and
// reports whether the current match moved (to bring it into view).
func (f *finder) advance(n int) bool {
	step := f.step
	f.step = 0
	if f.cur > n {
		f.cur = 0 // the list changed under it
	}
	if step == 0 || n == 0 {
		return false
	}
	from := f.cur
	if from == 0 && step < 0 {
		from = n + 1 // Shift+Return first goes to the last match
	}
	f.cur = ((from-1+step)%n+n)%n + 1
	return true
}

// count is the caption beside the field: "Not found", "3 matches", or
// "2 of 3" once Return moved to one.
func (f *finder) count(c *ui.Context, query string, n int) {
	switch {
	case strings.TrimSpace(query) == "":
	case n == 0:
		caption(c, i18n.T("Not found")).SingleLine().Shrink(0)
	case f.cur > 0:
		caption(c, i18n.T("%d of %d", f.cur, n)).SingleLine().Shrink(0)
	default:
		caption(c, i18n.N(n, "%d match", "%d matches", n)).SingleLine().Shrink(0)
	}
}

// findHits counts the matches of query in the texts each of n rows
// shows: hits[i] is the number before row i, hits[n] the total.
func findHits(query string, n int, texts func(i int) []string) []int {
	hits := make([]int, n+1)
	for i := range n {
		hits[i+1] = hits[i]
		for _, s := range texts(i) {
			hits[i+1] += len(findFolded(s, query))
		}
	}
	return hits
}

// rowOf is the row holding match cur (from 1) and the match's index
// within the row; -1, -1 for none.
func rowOf(hits []int, cur int) (row, at int) {
	for i := 1; i < len(hits); i++ {
		if cur > hits[i-1] && cur <= hits[i] {
			return i - 1, cur - hits[i-1] - 1
		}
	}
	return -1, -1
}

// markSpans splits text into spans over base: the matches of query in
// red (the theme's Danger) and bold on a red tint, match number cur
// (from 0; -1 for none) white on red.
func markSpans(red ui.Color, base ui.Span, text, query string, cur int) []ui.Span {
	r := []rune(text)
	var out []ui.Span
	plain := func(from, to int) {
		if to > from {
			s := base
			s.Text = string(r[from:to])
			out = append(out, s)
		}
	}
	at := 0
	for i, m := range findFolded(text, query) {
		plain(at, m[0])
		s := base
		s.Text, s.Weight, s.Color, s.Background = string(r[m[0]:m[1]]), 700, red, red.Alpha(0.16)
		if i == cur {
			s.Color, s.Background = ui.Hex("#ffffff"), red
		}
		out = append(out, s)
		at = m[1]
	}
	plain(at, len(r))
	if len(out) == 0 {
		out = append(out, base) // an empty text
	}
	return out
}

// markedText is text as a RichText with the matches of query marked
// (markSpans), labelled with the text for assistive technology.
func markedText(c *ui.Context, text, query string, cur int) ui.Element {
	return ui.RichText(c, markSpans(tok(c.Theme()).Danger, ui.Span{}, text, query, cur)...).Label(text)
}

// curIn is the current match's index within one of a row's texts: at is
// its index within the row (rowOf, -1 for none), before the number of
// matches in the row's texts before this one.
func curIn(at, before int) int {
	if at < before {
		return -1
	}
	return at - before
}
