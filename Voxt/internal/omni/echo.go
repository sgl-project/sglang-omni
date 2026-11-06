package omni

import (
	"context"
	"strings"
	"unicode"
)

// listMarks may stand between the terms of an echoed list; sentenceEnds
// end the sentence before one.
const (
	listMarks    = ",，、;；/ \t\n　"
	sentenceEnds = "。.!！?？\n…"
)

// echoTail finds the context's terms Qwen3-ASR may have echoed at the
// end of a request's text (behavior.md 3.6a): after the last sentence (or
// as the whole text), a sentence made only of terms, with list marks
// between them. Speech that ends with a term does not match ("我们用
// Omni。"): the term is not a sentence of its own. It returns where the
// sentence starts, its terms, and whether it is certainly an echo: at
// least two terms, in the context's order. cut < 0: none. A text the
// token budget cut off (cutOff) may end in a part of a term.
func echoTail(text string, terms []string, cutOff bool) (cut int, found []string, certain bool) {
	rs := []rune(text)
	folded := make([]rune, len(rs))
	for i, r := range rs {
		folded[i] = foldRune(r)
	}
	keys := make([][]rune, len(terms))
	for i, t := range terms {
		for _, r := range t {
			if !unicode.IsSpace(r) {
				keys[i] = append(keys[i], foldRune(r))
			}
		}
	}
	pos := skipBack(rs, len(rs), listMarks+sentenceEnds)
	if cutOff {
		from := pos
		for from > 0 && !strings.ContainsRune(listMarks, rs[from-1]) {
			from--
		}
		part := string(folded[from:pos])
		for _, k := range keys {
			if from > 0 && part != "" && len(part) < len(string(k)) && strings.HasPrefix(string(k), part) {
				pos = skipBack(rs, from, listMarks)
				break
			}
		}
	}
	var idx []int // indexes in terms, from the end
	start := pos
	for {
		best, bestStart := -1, 0
		for i, k := range keys {
			if s, ok := matchBack(folded, pos, k); ok && (best < 0 || len(k) > len(keys[best])) {
				best, bestStart = i, s
			}
		}
		if best < 0 {
			break
		}
		idx, start = append(idx, best), bestStart
		next := skipBack(rs, start, listMarks)
		if next == start {
			break
		}
		pos = next
	}
	if len(idx) == 0 {
		return -1, nil, false
	}
	c := skipBack(rs, start, " \t　")
	if c > 0 && !strings.ContainsRune(sentenceEnds, rs[c-1]) {
		return -1, nil, false
	}
	certain = len(idx) >= 2
	found = make([]string, len(idx))
	for i, f := range idx {
		found[len(idx)-1-i] = terms[f]
		if i > 0 && f >= idx[i-1] { // read in order, the terms come as in the context
			certain = false
		}
	}
	return len(string(rs[:c])), found, certain
}

// spokenTokens counts text's words: a Chinese, Japanese or Korean
// character each, a run of letters or digits of other scripts each;
// fillers (嗯, 呃, um…) do not count.
func spokenTokens(text string) int {
	n, word := 0, []rune{}
	flush := func() {
		if len(word) > 0 && !fillers[strings.ToLower(string(word))] {
			n++
		}
		word = word[:0]
	}
	for _, r := range text {
		switch {
		case !latinWord(r) && (unicode.IsLetter(r) || unicode.IsDigit(r)):
			flush()
			if !fillers[string(r)] {
				n++
			}
		case latinWord(r):
			word = append(word, r)
		default:
			flush()
		}
	}
	flush()
	return n
}

var fillers = map[string]bool{"嗯": true, "呃": true, "啊": true, "额": true, "唔": true, "哦": true, "噢": true, "呣": true,
	"um": true, "uh": true, "hmm": true, "mm": true, "ah": true, "er": true, "eh": true}

// foldRune folds case and full width.
func foldRune(r rune) rune {
	if r >= 0xFF01 && r <= 0xFF5E {
		r -= 0xFEE0
	}
	return unicode.ToLower(r)
}

// skipBack is i moved back over the runes in set.
func skipBack(rs []rune, i int, set string) int {
	for i > 0 && strings.ContainsRune(set, rs[i-1]) {
		i--
	}
	return i
}

// matchBack is where key starts when it ends at folded[end], spaces in
// the text skipped; a key that starts with a letter or digit must not be
// the end of a longer word.
func matchBack(folded []rune, end int, key []rune) (int, bool) {
	if len(key) == 0 {
		return 0, false
	}
	i := end
	for k := len(key) - 1; k >= 0; k-- {
		for i > 0 && unicode.IsSpace(folded[i-1]) && k < len(key)-1 {
			i--
		}
		if i == 0 || folded[i-1] != key[k] {
			return 0, false
		}
		i--
	}
	if i > 0 && latinWord(key[0]) && latinWord(folded[i-1]) {
		return 0, false
	}
	return i, true
}

func latinWord(r rune) bool { return r < 0x2E80 && (unicode.IsLetter(r) || unicode.IsDigit(r)) }

// trimEcho is a request's text without the context's terms echoed at
// its end (echoTail; cutOff: the token budget ended the text). The
// evidence is the same audio recognised without the context (plain):
//   - the terms stay when plain writes one of them more often than the
//     text before them does (the speech has it);
//   - a certain echo goes otherwise (also when plain fails);
//   - one that may be speech goes when plain holds fewer words than the
//     text before the terms and half of theirs (it heard nothing more),
//     or when the voice activity model finds no speech in the audio.
func (a *ASR) trimEcho(ctx context.Context, l *lease, req finalRequest, text string, cutOff bool, terms []string) string {
	cut, found, certain := echoTail(text, terms, cutOff)
	if cut < 0 {
		return text
	}
	kept := strings.TrimSpace(text[:cut])
	plainReq := req
	plainReq.context = ""
	plain, err := transcribe(ctx, l, plainReq, a.rt.cfg.StallTimeout)
	drop := certain
	switch {
	case err != nil:
	case heardAgain(plain.text, kept, found):
		drop = false
	case !certain:
		said := 0
		for _, f := range found {
			said += spokenTokens(f)
		}
		drop = 2*spokenTokens(plain.text) < 2*spokenTokens(kept)+said || a.noSpeech(ctx, req.pcm)
	}
	if !drop {
		return text
	}
	a.rt.logf("info", "%s: dropped %d dictionary terms the model echoed after the speech", a.spec.ID, len(found))
	return kept
}

// heardAgain says whether plain writes one of terms more often than kept
// does (ignoring case and spaces).
func heardAgain(plain, kept string, terms []string) bool {
	fold := func(s string) string {
		return strings.Map(func(r rune) rune {
			if unicode.IsSpace(r) {
				return -1
			}
			return foldRune(r)
		}, s)
	}
	p, k := fold(plain), fold(kept)
	for _, t := range terms {
		if f := fold(t); f != "" && strings.Count(p, f) > strings.Count(k, f) {
			return true
		}
	}
	return false
}

// noSpeech says whether the voice activity model that comes with the
// speech model finds no speech in pcm (false without one, or when it
// fails).
func (a *ASR) noSpeech(ctx context.Context, pcm []byte) bool {
	if a.vad == nil {
		return false
	}
	segs, err := a.vad.Speech(ctx, pcm, sampleRate)
	return err == nil && len(segs) == 0
}
