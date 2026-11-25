package compose

import (
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/Voxt/internal/snippet"
)

// Speech is cleaned up in pieces (docs/eval.md): where whole sentences
// change language, a small model loses the clean-up's rules (it returns
// the text as it is, or translates it into the language it starts in);
// and a long answer takes a slowed-down model half a minute, while a
// piece's answer reaches the overlay sooner and, when it fails, only its
// piece stays as recognised.
const (
	// LongWords: speech longer than this, in words, is composed in
	// pieces (as long as each other: 301 words are two of about 150).
	LongWords = 300
	// PieceWords is a piece's most words, unless one sentence is
	// longer and has no clause or space to cut at.
	PieceWords = 300
	// PieceChars is the most characters of a block of the live clean-up
	// (the session's livepolish.go): one piece by any count.
	PieceChars = 300
)

// words is text's length as the pieces count it: each character of a
// script written without spaces (Chinese, Japanese, Korean, Thai…) and
// of full-width punctuation is a word, and so is a run of letters and
// digits of any other script ("don't" is one).
func words(text string) int {
	n, inWord := 0, false
	for _, r := range text {
		switch {
		case unspaced(r):
			n, inWord = n+1, false
		case unicode.IsLetter(r) || unicode.IsDigit(r) || unicode.IsMark(r):
			if !inWord {
				n++
			}
			inWord = true
		case inWord && (r == '\'' || r == '’'):
		default:
			inWord = false
		}
	}
	return n
}

// unspaced reports a rune of a script written without spaces, or
// full-width punctuation.
func unspaced(r rune) bool {
	return isCJK(r) || unicode.In(r, unicode.Thai, unicode.Lao, unicode.Khmer, unicode.Myanmar) ||
		(r >= 0x3000 && r <= 0x303F) || (r >= 0xFF00 && r <= 0xFFEF)
}

// Piece is a part of a long input: its text and what followed it in the
// input ("", a space, a line or a paragraph break).
type Piece struct {
	Text, After string
}

// Part is one request of an input composed in parts (BuildParts).
type Part struct {
	Output
	Piece
}

// BuildParts composes in as one request or, for Transcribe, one per
// piece of the speech (LanguagePieces: where the language changes, and
// over LongWords words). Translate and Rewrite are one request: a
// translation has one target language, and in pieces it was no better
// and slower (docs/eval.md).
// Every part has the whole input's system prompt and its piece as the user
// message (after the glossary picked from the whole text, when it is there):
// the parts share their prefix, which the model's prompt cache keeps.
//
// note (luojiaxuan): with Input.Before, each part's user message opens with
// the cleaned text before the input (before.go).
func BuildParts(in Input) ([]Part, error) {
	out, err := Build(in)
	if err != nil {
		return nil, err
	}
	if in.Task != Transcribe {
		return []Part{{Output: out, Piece: Piece{Text: in.Text}}}, nil
	}
	pieces := languagePieces(in.Text, in.PieceLimit)
	parts := make([]Part, len(pieces))
	for i, p := range pieces {
		o, piece := out, in
		piece.Text = p.Text
		o.User = out.withGlossary(userMessage(piece))
		parts[i] = Part{Output: o, Piece: p}
	}
	return parts, nil
}

// JoinParts joins the answers of parts, in order (fewer answers than
// parts: those so far). Empty answers are skipped. Between two answers
// goes the input's line or paragraph break, else a line break when one
// of them has lines of its own, a space after an answer that ends in
// Latin script, or nothing (Chinese).
func JoinParts(parts []Part, answers []string) string {
	var b strings.Builder
	prev, after := "", ""
	for i, a := range answers {
		a = strings.TrimSpace(a)
		if a == "" {
			continue
		}
		if prev != "" {
			last, _ := utf8.DecodeLastRuneInString(prev)
			switch {
			case strings.Contains(after, "\n\n"):
				b.WriteString("\n\n")
			case strings.Contains(after, "\n") || strings.Contains(prev, "\n") || strings.Contains(a, "\n"):
				b.WriteString("\n")
			case last < utf8.RuneSelf:
				b.WriteString(" ")
			}
		}
		b.WriteString(a)
		prev = a
		if i < len(parts) {
			after = parts[i].After
		}
	}
	return b.String()
}

// boundaries split a text, strongest first: after a sentence (its
// closing quotes too) or a line, after a clause, at a space.
var boundaries = []*regexp.Regexp{
	regexp.MustCompile(`(?:[。！？!?；;…]+[”’"'」』）)\]]*|\.[”’"'」』）)\]]*(?:\s|$))\s*|\n\s*`),
	regexp.MustCompile(`[，,、：:]\s*`),
	regexp.MustCompile(`\s+`),
}

// Pieces cuts text into pieces of at most PieceWords words when it
// is longer than LongWords (else one piece): at sentence ends, a piece
// ending at a paragraph once it is half as long as planned, a sentence
// longer than a piece at its clauses, then its spaces, then anywhere.
// The pieces are about as long as each other. Joined with their After,
// they are text, trimmed.
func Pieces(text string) []Piece {
	return piecesOf(text, 0)
}

// piecesOf is Pieces with a piece word cap other than PieceWords
// (limit <= 0: the default).
func piecesOf(text string, limit int) []Piece {
	text = strings.TrimSpace(text)
	long, cap := LongWords, PieceWords
	if limit > 0 {
		long, cap = limit, limit
	}
	if words(text) <= long {
		return []Piece{{Text: text}}
	}
	return pack(text, 0, cap)
}

// LanguagePieces is Pieces, also cut wherever whole sentences change
// language, at any length: runs of sentences whose words are mostly
// Chinese, Japanese or Korean apart from runs of other sentences (a
// Chinese sentence with English words is Chinese; one without letters
// goes with the run before it), each run cut as Pieces cuts it.
func LanguagePieces(text string) []Piece {
	return languagePieces(text, 0)
}

// languagePieces is LanguagePieces with a piece word cap other than
// PieceWords (limit <= 0: the default).
func languagePieces(text string, limit int) []Piece {
	text = strings.TrimSpace(text)
	var runs []Piece
	runCJK, runLetters := false, false
	for _, u := range units(text, 0) {
		cjk, letters := cjkWords(u.Text)
		if len(runs) > 0 && (!letters || !runLetters || cjk == runCJK) {
			last := &runs[len(runs)-1]
			last.Text += last.After + u.Text
			last.After = u.After
			if letters && !runLetters {
				runCJK, runLetters = cjk, true
			}
			continue
		}
		runs = append(runs, u)
		runCJK, runLetters = cjk, letters
	}
	if len(runs) <= 1 {
		return piecesOf(text, limit)
	}
	var out []Piece
	for _, r := range runs {
		ps := piecesOf(r.Text, limit)
		ps[len(ps)-1].After = r.After
		out = append(out, ps...)
	}
	return out
}

// cjkWords: whether text is mostly Chinese, Japanese or Korean, counting
// words (an ideograph, kana or Hangul syllable, or a word of letters), and
// whether it has words at all.
func cjkWords(text string) (cjk, words bool) {
	part, n := share(text, isCJK)
	return n > 0 && part >= 0.5, n > 0
}

func isCJK(r rune) bool {
	return unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Hangul)
}

// pack cuts text at the boundaries of level and below.
func pack(text string, level, cap int) []Piece {
	total := words(text)
	n := (total + cap - 1) / cap
	target := total / n
	var out []Piece
	var cur strings.Builder
	size, after := 0, ""
	flush := func() {
		if cur.Len() > 0 {
			out = append(out, Piece{Text: cur.String(), After: after})
		}
		cur.Reset()
		size, after = 0, ""
	}
	for _, u := range units(text, level) {
		un := words(u.Text)
		if un > cap {
			flush()
			var sub []Piece
			if level+1 < len(boundaries) {
				sub = pack(u.Text, level+1, cap)
			} else {
				sub = hardCut(u.Text, cap)
			}
			sub[len(sub)-1].After = u.After
			out = append(out, sub...)
			continue
		}
		if size+un > cap {
			flush()
		}
		if cur.Len() > 0 {
			cur.WriteString(after)
		}
		cur.WriteString(u.Text)
		size += un
		after = u.After
		if size >= target || strings.Contains(after, "\n\n") && size >= target/2 {
			flush()
		}
	}
	flush()
	if len(out) > 0 {
		out[len(out)-1].After = ""
	}
	return out
}

// units are text cut after each boundary of level, the white space after
// a cut in After.
func units(text string, level int) []Piece {
	var out []Piece
	start := 0
	for _, m := range boundaries[level].FindAllStringIndex(text, -1) {
		if inMark(text, m[0]) {
			continue
		}
		end := m[1]
		body := strings.TrimRight(text[start:end], " \t\r\n")
		if body == "" {
			if len(out) > 0 {
				out[len(out)-1].After += text[start:end]
			}
		} else {
			out = append(out, Piece{Text: body, After: text[start+len(body) : end]})
		}
		start = end
	}
	if start < len(text) {
		out = append(out, Piece{Text: text[start:]})
	}
	return out
}

// hardCut cuts text into pieces of equal length, the last resort, never
// inside a snippet's mark.
func hardCut(text string, cap int) []Piece {
	var at []int // the byte each rune starts at, and the end
	for i := range text {
		at = append(at, i)
	}
	runes := len(at)
	at = append(at, len(text))
	n := (runes + cap - 1) / cap
	size := (runes + n - 1) / n
	var out []Piece
	for i := 0; i < runes; {
		j := min(i+size, runes)
		for j < runes && inMark(text, at[j]) {
			j++
		}
		out = append(out, Piece{Text: text[at[i]:at[j]]})
		i = j
	}
	return out
}

// inMark says whether byte i of text is inside the mark around a
// snippet's trigger (snippet.Mark), which a piece must not cut: the
// model would lose the trigger.
func inMark(text string, i int) bool {
	return strings.LastIndex(text[:i], snippet.Open) > strings.LastIndex(text[:i], snippet.Close)
}
