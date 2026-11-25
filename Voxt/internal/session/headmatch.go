package session

import (
	"regexp"
	"slices"
	"strings"
	"unicode/utf8"
)

// A clean-up asked about the live text serves the final text too when
// the two say the same but for punctuation, or differ in one or two places
// by a few words (headChanges): the words are put right in the answer
// (putRight). Anything else is asked again.

const (
	// headSpans is how many stretches of words may differ, headChangeWords
	// the most words put right in one, which go into the answer as they
	// were recognised, and headWords how long what differs, with the words
	// between, may be.
	headSpans       = 2
	headChangeWords = 3
	headWords       = 60
	// headContext is the most words on each side that place a difference
	// in an answer.
	headContext = 5
)

// spokenWord is a word of a text: a Chinese, Japanese or Korean
// character, a word of other letters and digits ("3.5", "don't"), or a
// mark that means something (a question mark, %). The marks that only
// punctuate are none.
type spokenWord struct {
	text       string // lower case
	start, end int
}

var spokenWordRE = regexp.MustCompile(`[\p{Han}\p{Hiragana}\p{Katakana}\p{Hangul}]|(?:[^\P{L}\p{Han}\p{Hiragana}\p{Katakana}\p{Hangul}]|\p{N})+(?:['’.](?:[^\P{L}\p{Han}\p{Hiragana}\p{Katakana}\p{Hangul}]|\p{N})+)*|[^\s\p{L}\p{N}，。！；：、,.!;:…“”"「」]`)

func spokenWords(text string) []spokenWord {
	var words []spokenWord
	for _, m := range spokenWordRE.FindAllStringIndex(text, -1) {
		words = append(words, spokenWord{strings.ToLower(text[m[0]:m[1]]), m[0], m[1]})
	}
	return words
}

func sameWord(a, b spokenWord) bool { return a.text == b.text }

// headChange is words old[i1:i2] said as with in the other text.
type headChange struct {
	i1, i2 int
	with   string
}

var (
	// headFillers are words the clean-up leaves out.
	headFillers = map[string]bool{"uh": true, "um": true, "er": true, "erm": true, "hmm": true, "呃": true, "嗯": true, "啊": true, "额": true, "哎": true, "诶": true, "唉": true, "吧": true, "呀": true}
	// headNumber is a number or a part of one, headCue a word that
	// corrects what was said before or fills a pause, and headLetters a
	// word spelled out: the answer may have changed more than the word.
	headNumber  = regexp.MustCompile(`\p{N}|[零〇一二两三四五六七八九十百千万亿点]{2}`)
	headCue     = regexp.MustCompile(`不|没|错|改|等等|是说|就是|那个|\b(?:no|not|wait|sorry|actually|mean|rather|instead|oh|never|don't)\b`)
	headLetters = regexp.MustCompile(`(?i)\b[a-z](?:[ .,-]+[a-z])+\b`)
	headWord    = regexp.MustCompile(`[\p{L}\p{N}]`)
	headDigits  = strings.NewReplacer("零", "0", "〇", "0", "一", "1", "二", "2", "两", "2", "三", "3", "四", "4", "五", "5", "六", "6", "七", "7", "八", "8", "九", "9", "点", ".", ",", "", "，", "", " ", "")
)

// headChanges are the differences of the words of text from old's that an
// answer to old must be put right for, and false when text needs a
// request of its own.
func headChanges(old []spokenWord, oldText, text string) ([]headChange, bool) {
	now := spokenWords(text)
	var changes []headChange
	spans, ok := differences(old, now)
	if !ok {
		return nil, false
	}
	for _, s := range spans {
		was, is := "", ""
		if s.i2 > s.i1 {
			was = oldText[old[s.i1].start:old[s.i2-1].end]
		}
		if s.j2 > s.j1 {
			is = text[now[s.j1].start:now[s.j2-1].end]
		}
		filler := func(w spokenWord) bool { return headFillers[w.text] }
		switch {
		case !slices.ContainsFunc(old[s.i1:s.i2], not(filler)) && !slices.ContainsFunc(now[s.j1:s.j2], not(filler)):
		case headDigits.Replace(strings.ToLower(was)) == headDigits.Replace(strings.ToLower(is)):
		case s.i2-s.i1 > headChangeWords || s.j2-s.j1 > headChangeWords || headLetters.MatchString(is) || abbreviated(oldText, old, s.i1, s.i2) || abbreviated(text, now, s.j1, s.j2) ||
			headNumber.MatchString(was) || headNumber.MatchString(is) || headCue.MatchString(strings.ToLower(was)) || headCue.MatchString(strings.ToLower(is)) ||
			slices.ContainsFunc(old[s.i1:s.i2], markWord) || slices.ContainsFunc(now[s.j1:s.j2], markWord):
			return nil, false
		default:
			changes = append(changes, headChange{s.i1, s.i2, is})
		}
	}
	return changes, true
}

// abbreviated says whether words[i:j], within text, end in a mark of their
// own ("Mrs."), which is not the sentence's.
func abbreviated(text string, words []spokenWord, i, j int) bool {
	return j > i && j < len(words) && strings.HasPrefix(text[words[j-1].end:], ".")
}

func not[T any](f func(T) bool) func(T) bool { return func(v T) bool { return !f(v) } }

// markWord says whether w is a mark, not a word: its place in the answer
// may be gone.
func markWord(w spokenWord) bool {
	return !headWord.MatchString(w.text)
}

// span is words i1:i2 of one text, j1:j2 of the other.
type span struct{ i1, i2, j1, j2 int }

// differences are where a and b differ, at most headSpans.
func differences(a, b []spokenWord) ([]span, bool) {
	pre := 0
	for pre < len(a) && pre < len(b) && sameWord(a[pre], b[pre]) {
		pre++
	}
	end := 0
	for end < len(a)-pre && end < len(b)-pre && sameWord(a[len(a)-1-end], b[len(b)-1-end]) {
		end++
	}
	x, y := a[pre:len(a)-end], b[pre:len(b)-end]
	if len(x) > headWords || len(y) > headWords {
		return nil, false
	}
	// length[i][j] is the longest common run of words of x[i:] and y[j:].
	length := make([][]int, len(x)+1)
	for i := range length {
		length[i] = make([]int, len(y)+1)
	}
	for i := len(x) - 1; i >= 0; i-- {
		for j := len(y) - 1; j >= 0; j-- {
			if sameWord(x[i], y[j]) {
				length[i][j] = length[i+1][j+1] + 1
			} else {
				length[i][j] = max(length[i+1][j], length[i][j+1])
			}
		}
	}
	var out []span
	for i, j := 0, 0; i < len(x) || j < len(y); {
		if i < len(x) && j < len(y) && sameWord(x[i], y[j]) {
			i, j = i+1, j+1
			continue
		}
		s := span{pre + i, pre + i, pre + j, pre + j}
		for i < len(x) || j < len(y) {
			if i < len(x) && j < len(y) && sameWord(x[i], y[j]) {
				break
			}
			if j >= len(y) || i < len(x) && length[i+1][j] >= length[i][j+1] {
				i++
			} else {
				j++
			}
		}
		s.i2, s.j2 = pre+i, pre+j
		if out = append(out, s); len(out) > headSpans {
			return nil, false
		}
	}
	return out, true
}

// putRight is answer, the clean-up of old, with the changes: each is
// placed by the words around it, which the answer must hold once.
func putRight(answer string, old []spokenWord, changes []headChange) (string, bool) {
	words := spokenWords(answer)
	type edit struct {
		start, end int
		with       string
	}
	var edits []edit
	for _, c := range changes {
		at := -1
		for k := 1; k <= headContext && at < 0; k++ {
			lo, hi := max(0, c.i1-k), min(len(old), c.i2+k)
			if lo == c.i1 && hi == c.i2 {
				break
			}
			seq, found, n := old[lo:hi], 0, 0
			for s := 0; s+len(seq) <= len(words); s++ {
				if slices.EqualFunc(words[s:s+len(seq)], seq, sameWord) {
					found, n = s, n+1
				}
			}
			if n == 0 {
				return "", false
			}
			if n == 1 {
				at = found + c.i1 - lo
			}
		}
		if at < 0 {
			return "", false
		}
		e := edit{with: c.with}
		switch n := c.i2 - c.i1; {
		case n > 0:
			e.start, e.end = words[at].start, words[at+n-1].end
			if c.with == "" && e.start > 0 && answer[e.start-1] == ' ' {
				e.start--
			}
		case at > 0:
			e.start = words[at-1].end
			e.end = e.start
			last, _ := utf8.DecodeLastRuneInString(answer[:e.start])
			first, _ := utf8.DecodeRuneInString(c.with)
			if last < utf8.RuneSelf && first < utf8.RuneSelf {
				e.with = " " + c.with
			}
		default:
			return "", false
		}
		edits = append(edits, e)
	}
	slices.SortFunc(edits, func(a, b edit) int { return b.start - a.start })
	for i, e := range edits {
		if i > 0 && e.end > edits[i-1].start {
			return "", false // they touch
		}
		answer = answer[:e.start] + e.with + answer[e.end:]
	}
	return answer, true
}

// tailAfter is the rest of the live text after the stretches the speech
// model finalised, found by the end of final in live, and false when the
// live text does not reach there or has it twice: it may say the same
// words again.
func tailAfter(live, final string) (string, bool) {
	l, f := spokenWords(live), spokenWords(final)
	for k := min(8, len(f)); k >= 3; k-- {
		end, n, at := f[len(f)-k:], 0, -1
		for s := 0; s+k <= len(l); s++ {
			// Where the final text's end would be, give or take.
			if d := s + k - len(f); d*d > 16+len(f)*len(f)/25 || !slices.EqualFunc(l[s:s+k], end, sameWord) {
				continue
			}
			at, n = s+k, n+1
		}
		if n > 1 {
			return "", false
		}
		if n == 1 {
			return strings.TrimLeft(live[l[at-1].end:], " ,.!;:，。！；：、…"), true
		}
	}
	return "", false
}

// joinSpeech is two stretches of speech as the speech model joins them: a
// space between words of scripts written with spaces.
func joinSpeech(a, b string) string {
	a, b = strings.TrimSpace(a), strings.TrimSpace(b)
	last, _ := utf8.DecodeLastRuneInString(a)
	first, _ := utf8.DecodeRuneInString(b)
	if a != "" && b != "" && last < 0x2E80 && first < 0x2E80 {
		return a + " " + b
	}
	return a + b
}
