package markdown

import (
	"strings"
	"time"
	"unicode/utf8"

	"github.com/egoist/mygo/ui"
)

// Reveal is how Stream shows text as it comes: rather than in the bursts
// a network delivers, at an even pace that catches up with what came
// within CatchUp, a unit at a time, each fading in, as the chat apps show
// replies.
type Reveal struct {
	// Unit is what shows at a time: a word by default.
	Unit RevealUnit
	// CatchUp is how long text that came waits at most before it shows:
	// 250 ms when zero. Text comes in faster the more there is waiting.
	CatchUp time.Duration
	// Fade is how long text takes to fade in: 300 ms when zero, none when
	// negative, nor when the desktop asks for less motion.
	Fade time.Duration
}

// RevealUnit is what Reveal shows at a time.
type RevealUnit uint8

const (
	// ByWord shows a word and the spaces after it at a time.
	ByWord RevealUnit = iota
	// ByCharacter shows a character at a time.
	ByCharacter
	// BySentence shows a sentence at a time, once it is complete: up to
	// a period, a question or exclamation mark followed by a space, an
	// ideographic full stop, or the end of a line.
	BySentence
	// ByLine shows a line at a time, once it is complete.
	ByLine
	// ByParagraph shows a paragraph, a block between blank lines, at a
	// time, once it is complete.
	ByParagraph
	// AtOnce shows text as it comes, without pacing.
	AtOnce
)

// DefaultReveal is the Reveal of Stream without Options.Reveal.
var DefaultReveal = Reveal{}

// revealState is how much of the source a view shows, and since when
// what it shows came in.
type revealState struct {
	// revealing tells a stream being revealed, or done with text left to
	// reveal.
	revealing bool
	src       string
	shown     int
	carry     float64
	last      time.Time
	marks     []revealMark
	fade      time.Duration
}

// revealMark notes that the source from at on showed at time t.
type revealMark struct {
	at int
	t  time.Time
}

// step returns how much of src shows this frame, and whether more is to
// come in.
func (rs *revealState) step(c *ui.Context, src string, streaming bool, r *Reveal) (int, bool) {
	if r == nil {
		r = &DefaultReveal
	}
	now := c.Now()
	// Text a stream left to reveal still comes in once it is done; other
	// text a finished view is given shows at once, as does an edit.
	edited := rs.shown > len(src) || !strings.HasPrefix(src, rs.src[:rs.shown])
	pending := rs.revealing && rs.shown < len(rs.src) && !edited
	rs.src = src
	if !streaming && !pending || r.Unit == AtOnce || edited {
		rs.shown, rs.last, rs.revealing = len(src), now, streaming
		return rs.shown, false
	}
	rs.revealing = true
	target := len(src)
	if streaming && r.Unit != ByWord && r.Unit != ByCharacter {
		// Only complete sentences, lines or paragraphs, while more is
		// coming.
		target = rs.shown
		for i := rs.shown; ; {
			e := unitEnd(src, i, len(src), r.Unit)
			if e < 0 {
				break
			}
			target, i = e, e
		}
	}
	if rs.shown >= target {
		// Nothing to show until more text comes, which builds a frame.
		rs.last = now
		return rs.shown, false
	}
	dt := now.Sub(rs.last)
	if rs.last.IsZero() || dt > 100*time.Millisecond {
		dt = 16 * time.Millisecond
	}
	rs.last = now
	catchUp := r.CatchUp
	if catchUp <= 0 {
		catchUp = 250 * time.Millisecond
	}
	backlog := target - rs.shown
	// As many bytes a second as show the backlog within catchUp, and at
	// least a few hundred, so that a trickle does not crawl.
	rate := max(float64(backlog)/catchUp.Seconds(), 240)
	rs.carry += rate * dt.Seconds()
	n := int(rs.carry)
	if n == 0 {
		return rs.shown, true
	}
	rs.carry -= float64(n)
	next := min(rs.shown+n, target)
	next = snap(src, next, target, r.Unit)
	if next > rs.shown {
		rs.marks = append(rs.marks, revealMark{rs.shown, now})
		rs.shown = next
	}
	return rs.shown, rs.shown < len(src)
}

// snap moves offset i of src forward to the end of the unit it is in.
func snap(src string, i, target int, u RevealUnit) int {
	switch u {
	case ByWord:
		for i < target && src[i] != ' ' && src[i] != '\n' {
			i++
		}
		for i < target && (src[i] == ' ' || src[i] == '\n') {
			i++
		}
	case BySentence, ByLine, ByParagraph:
		if e := unitEnd(src, i, target, u); e >= 0 {
			i = e
		} else {
			i = target
		}
	}
	for i < target && !utf8.RuneStart(src[i]) {
		i++
	}
	return i
}

// unitEnd returns where the first sentence, line or paragraph of src
// ending after i and by limit ends, or -1.
func unitEnd(src string, i, limit int, u RevealUnit) int {
	switch u {
	case ByLine:
		if k := strings.IndexByte(src[i:limit], '\n'); k >= 0 {
			return i + k + 1
		}
	case ByParagraph:
		if k := strings.Index(src[i:limit], "\n\n"); k >= 0 {
			return i + k + 2
		}
	case BySentence:
		for j := i; j < limit; {
			r, n := utf8.DecodeRuneInString(src[j:limit])
			j += n
			switch r {
			case '\n':
				return j
			case '.', '!', '?', '…':
				// Closing quotes and brackets go with it, then a space.
				for j < limit && strings.IndexByte(`"')]*_`, src[j]) >= 0 {
					j++
				}
				if j < limit && (src[j] == ' ' || src[j] == '\n') {
					return j + 1
				}
			case '。', '！', '？':
				return j
			}
		}
	}
	return -1
}

// fader returns the opacity of text at a source offset as it fades in,
// or nil when nothing fades.
func (rs *revealState) fader(c *ui.Context, r *Reveal) func(src int) float32 {
	return rs.fadeAt(c.Now(), c.Preferences().ReduceMotion, r)
}

func (rs *revealState) fadeAt(now time.Time, reduceMotion bool, r *Reveal) func(src int) float32 {
	if r == nil {
		r = &DefaultReveal
	}
	fade := r.Fade
	if fade == 0 {
		fade = 300 * time.Millisecond
	}
	// Marks done fading are gone.
	k := 0
	for k < len(rs.marks) && (fade < 0 || now.Sub(rs.marks[k].t) >= fade) {
		k++
	}
	rs.marks = append(rs.marks[:0], rs.marks[k:]...)
	if len(rs.marks) == 0 || fade < 0 || reduceMotion {
		rs.marks = rs.marks[:0]
		return nil
	}
	rs.fade = fade
	marks := rs.marks
	return func(src int) float32 {
		if src < marks[0].at {
			return 1
		}
		// The last mark at or before src.
		lo, hi := 0, len(marks)
		for lo < hi {
			m := (lo + hi) / 2
			if marks[m].at <= src {
				lo = m + 1
			} else {
				hi = m
			}
		}
		age := now.Sub(marks[lo-1].t)
		a := float32(age) / float32(fade)
		// Quantized, so that the spans of text alike stay one.
		return min(1, float32(int(easeOut(a)*16))/16)
	}
}

func easeOut(t float32) float32 {
	t = min(max(t, 0), 1)
	u := 1 - t
	return 1 - u*u*u
}

// fading reports whether text still fades in.
func (rs *revealState) fading() bool { return len(rs.marks) > 0 }
