package diff

import (
	"unicode"
	"unicode/utf8"
)

// Range is a range of bytes, Start to End.
type Range struct{ Start, End int }

// Words compares two texts word by word, as review tools highlight what
// changed within lines: words (letters, digits and underscores), runs of
// whitespace and other characters one at a time. It returns the byte
// ranges of a and b that changed, joined across the whitespace between
// them, and how much of the two texts stayed the same, from 0 to 1, by
// their bytes.
func Words(a, b string) (ra, rb []Range, same float64) {
	return tokens(a, b, wordTokens)
}

// Chars compares two texts character by character, as Words does by words.
func Chars(a, b string) (ra, rb []Range, same float64) {
	return tokens(a, b, charTokens)
}

func tokens(a, b string, split func(string, []Range) []Range) (ra, rb []Range, same float64) {
	ta, tb := split(a, nil), split(b, nil)
	ids := make(map[string]int32, len(ta)+len(tb))
	conv := func(s string, t []Range) []int32 {
		out := make([]int32, len(t))
		for i, r := range t {
			v := s[r.Start:r.End]
			id, ok := ids[v]
			if !ok {
				id = int32(len(ids))
				ids[v] = id
			}
			out[i] = id
		}
		return out
	}
	ia, ib := conv(a, ta), conv(b, tb)
	hunks := Ints(ia, ib)
	changed := 0
	for _, h := range hunks {
		if h.A0 < h.A1 {
			ra = join(ra, Range{ta[h.A0].Start, ta[h.A1-1].End}, a)
			changed += ta[h.A1-1].End - ta[h.A0].Start
		}
		if h.B0 < h.B1 {
			rb = join(rb, Range{tb[h.B0].Start, tb[h.B1-1].End}, b)
			changed += tb[h.B1-1].End - tb[h.B0].Start
		}
	}
	if total := len(a) + len(b); total > 0 {
		same = 1 - float64(changed)/float64(total)
	} else {
		same = 1
	}
	return ra, rb, same
}

// join appends r to rs, merged with the last range when only whitespace
// is between them.
func join(rs []Range, r Range, s string) []Range {
	if n := len(rs); n > 0 {
		last := &rs[n-1]
		gap := s[last.End:r.Start]
		blank := true
		for _, c := range gap {
			if c == '\n' || !unicode.IsSpace(c) {
				blank = false
				break
			}
		}
		if blank {
			last.End = r.End
			return rs
		}
	}
	return append(rs, r)
}

func isWord(r rune) bool {
	return r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r)
}

func wordTokens(s string, out []Range) []Range {
	for i := 0; i < len(s); {
		r, n := utf8.DecodeRuneInString(s[i:])
		j := i + n
		switch {
		case isWord(r):
			for j < len(s) {
				r, n := utf8.DecodeRuneInString(s[j:])
				if !isWord(r) {
					break
				}
				j += n
			}
		case r != '\n' && unicode.IsSpace(r):
			for j < len(s) {
				r, n := utf8.DecodeRuneInString(s[j:])
				if r == '\n' || !unicode.IsSpace(r) {
					break
				}
				j += n
			}
		}
		out = append(out, Range{i, j})
		i = j
	}
	return out
}

func charTokens(s string, out []Range) []Range {
	for i := 0; i < len(s); {
		_, n := utf8.DecodeRuneInString(s[i:])
		// Combining marks stay with their base.
		j := i + n
		for j < len(s) {
			r, n := utf8.DecodeRuneInString(s[j:])
			if !unicode.Is(unicode.Mn, r) && !unicode.Is(unicode.Me, r) && r != 0x200d && !(r >= 0xfe00 && r <= 0xfe0f) {
				break
			}
			j += n
		}
		out = append(out, Range{i, j})
		i = j
	}
	return out
}
