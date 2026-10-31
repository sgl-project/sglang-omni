package eval

import (
	"strconv"
	"strings"
	"unicode"
)

// Fillers are the hesitations the error rates ignore (on both sides: a
// recogniser that drops "um" is not wrong) and a clean-up must remove.
var Fillers = []string{"嗯", "呃", "额", "啊", "唔", "um", "uh", "er", "erm", "ah", "hmm"}

// Score is an alignment of a transcript (the hypothesis) with its
// reference: tokens are Chinese (and Japanese, Korean) characters one by
// one and Latin words, after Normalize.
type Score struct {
	N                    int // the reference's tokens
	Sub, Del, Ins        int
	RefTokens, HypTokens []string `json:"-"`
	ins                  []string
}

// Edits is substitutions, deletions and insertions.
func (s Score) Edits() int { return s.Sub + s.Del + s.Ins }

// Rate is the error rate: Edits over N (0 when both are empty).
func (s Score) Rate() float64 {
	if s.N == 0 {
		if s.Edits() == 0 {
			return 0
		}
		return 1
	}
	return float64(s.Edits()) / float64(s.N)
}

// Inserted are the hypothesis's tokens the reference has not.
func (s Score) Inserted() []string { return s.ins }

// Add sums two scores (a set's rate is its edits over its tokens).
func (s Score) Add(o Score) Score {
	return Score{N: s.N + o.N, Sub: s.Sub + o.Sub, Del: s.Del + o.Del, Ins: s.Ins + o.Ins}
}

// Compare aligns hyp with ref (Levenshtein over Tokens).
func Compare(ref, hyp string) Score {
	r, h := Tokens(ref), Tokens(hyp)
	// d[i][j]: the distance of r[:i] and h[:j].
	d := make([][]int, len(r)+1)
	for i := range d {
		d[i] = make([]int, len(h)+1)
		d[i][0] = i
	}
	for j := range d[0] {
		d[0][j] = j
	}
	for i := 1; i <= len(r); i++ {
		for j := 1; j <= len(h); j++ {
			cost := 1
			if r[i-1] == h[j-1] {
				cost = 0
			}
			d[i][j] = min(d[i-1][j-1]+cost, d[i-1][j]+1, d[i][j-1]+1)
		}
	}
	s := Score{N: len(r), RefTokens: r, HypTokens: h}
	for i, j := len(r), len(h); i > 0 || j > 0; {
		switch {
		case i > 0 && j > 0 && d[i][j] == d[i-1][j-1] && r[i-1] == h[j-1]:
			i, j = i-1, j-1
		case i > 0 && j > 0 && d[i][j] == d[i-1][j-1]+1:
			s.Sub++
			i, j = i-1, j-1
		case i > 0 && d[i][j] == d[i-1][j]+1:
			s.Del++
			i--
		default:
			s.Ins++
			s.ins = append([]string{h[j-1]}, s.ins...)
			j--
		}
	}
	return s
}

// Tokens is text's tokens after Normalize, without fillers.
func Tokens(text string) []string {
	var out []string
	var word []rune
	flush := func() {
		if len(word) > 0 {
			out = append(out, string(word))
			word = word[:0]
		}
	}
	for _, r := range Normalize(text) {
		switch {
		case isCJK(r):
			flush()
			out = append(out, string(r))
		case unicode.IsLetter(r) || unicode.IsDigit(r) || r == '\'' && len(word) > 0:
			word = append(word, r)
		default:
			flush()
		}
	}
	flush()
	kept := out[:0]
	for i := 0; i < len(out); i++ {
		t := strings.Trim(out[i], "'")
		if n, ok := englishNumber(t); ok {
			// "twenty five" is 25.
			if n >= 20 && n%10 == 0 && i+1 < len(out) {
				if u, ok := englishNumber(out[i+1]); ok && u < 10 {
					n, i = n+u, i+1
				}
			}
			t = strconv.Itoa(n)
		}
		if t == "" || t == "percent" || isFiller(t) {
			continue
		}
		kept = append(kept, t)
	}
	return kept
}

func isFiller(t string) bool {
	for _, f := range Fillers {
		if t == f {
			return true
		}
	}
	return false
}

func isCJK(r rune) bool {
	return unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Hangul)
}

// Normalize folds what a transcript may write either way: full-width
// Latin letters, digits and punctuation, case, and Chinese numerals
// (written as digits: "三百五十" and "350" are the same).
func Normalize(text string) string {
	var b strings.Builder
	rs := []rune(text)
	for i := 0; i < len(rs); i++ {
		r := rs[i]
		if r >= 0xFF01 && r <= 0xFF5E {
			r -= 0xFEE0
		} else if r == 0x3000 {
			r = ' '
		}
		if _, ok := cnDigit[r]; ok || cnUnit[r] > 0 {
			j := i
			for j < len(rs) && (cnDigit[rs[j]] > 0 || rs[j] == '零' || rs[j] == '〇' || cnUnit[rs[j]] > 0) {
				j++
			}
			b.WriteString(strconv.Itoa(chineseNumber(rs[i:j])))
			i = j - 1
			continue
		}
		b.WriteRune(unicode.ToLower(r))
	}
	return b.String()
}

var cnDigit = map[rune]int{'零': 0, '〇': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}

var cnUnit = map[rune]int{'十': 10, '百': 100, '千': 1000, '万': 10000}

// chineseNumber is a run of Chinese numerals' value: with units
// ("三百五十", "十二") or digit by digit ("二〇二六").
func chineseNumber(rs []rune) int {
	units := false
	for _, r := range rs {
		units = units || cnUnit[r] > 0
	}
	if !units {
		n := 0
		for _, r := range rs {
			n = n*10 + cnDigit[r]
		}
		return n
	}
	total, section, digit := 0, 0, 0
	for _, r := range rs {
		switch u := cnUnit[r]; {
		case u == 10000:
			total += (section + digit) * u
			section, digit = 0, 0
		case u > 0:
			if digit == 0 {
				digit = 1 // "十二": one ten
			}
			section += digit * u
			digit = 0
		default:
			digit = cnDigit[r]
		}
	}
	return total + section + digit
}

var englishNumbers = map[string]int{
	"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
	"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
	"seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
	"sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}

func englishNumber(t string) (int, bool) {
	n, ok := englishNumbers[t]
	return n, ok
}

// TermsFound are the terms text writes exactly as the dictionary does
// (case and spaces included).
func TermsFound(terms []string, text string) []string {
	var out []string
	for _, t := range terms {
		if strings.Contains(text, t) {
			out = append(out, t)
		}
	}
	return out
}
