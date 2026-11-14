package i18n

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// Plural is a plural category of the Unicode CLDR: the form of a word or a
// phrase a language uses for a number. English has two for cardinal
// numbers, "1 file" (One) and "2 files" (Other), and four for ordinal ones,
// "1st", "2nd", "3rd" and "4th"; Japanese has one, Arabic six.
type Plural uint8

const (
	PluralZero Plural = iota
	PluralOne
	PluralTwo
	PluralFew
	PluralMany
	PluralOther
)

var pluralNames = [...]string{"zero", "one", "two", "few", "many", "other"}

// String returns the category's keyword in CLDR and in messages: "zero",
// "one", "two", "few", "many" or "other".
func (p Plural) String() string {
	if int(p) < len(pluralNames) {
		return pluralNames[p]
	}
	return "other"
}

// ParsePlural returns the category a keyword names, as "few", and whether
// it names one.
func ParsePlural(keyword string) (Plural, bool) {
	for i, name := range pluralNames {
		if name == keyword {
			return Plural(i), true
		}
	}
	return PluralOther, false
}

// Operands are the operands of a number that the plural rules of CLDR
// read (Unicode Technical Standard #35, Part 3, Plural Operand Meanings),
// of its absolute value as it is written: 1.50 has two visible fraction
// digits that 1.5 does not have, and in some languages another plural.
//
// An integer part or a fraction of more than 18 digits keeps its last 18,
// plus 10^18, so that it equals no number of the rules, which are smaller,
// and its remainders by the powers of ten the rules divide by stay right.
type Operands struct {
	// I is the integer digits: 3 for 3.25.
	I uint64
	// V and W are the numbers of visible fraction digits, with trailing
	// zeros and without: 2 and 1 for 1.50.
	V, W int
	// F and T are the visible fraction digits as an integer, with trailing
	// zeros and without: 50 and 5 for 1.50.
	F, T uint64
	// C is the exponent of compact decimal notation, as 6 for "1.2M"
	// written 1.2c6, whose integer digits are then those of 1200000.
	C int
}

// big is added to operands of more than 18 digits.
const big = 1_000_000_000_000_000_000

// NewOperands returns the operands of n: an integer of any Go type, a
// float32 or float64 as strconv writes it in the fewest digits (1.5, never
// 1.50), a string of a decimal number, which keeps the fraction digits it
// writes ("1.50", "-3", "1.2c6" or "1.2e6" in CLDR's compact notation), or
// Operands. A minus sign is ignored.
func NewOperands(n any) (Operands, error) {
	switch n := n.(type) {
	case Operands:
		return n, nil
	case int:
		return intOperands(int64(n)), nil
	case int8:
		return intOperands(int64(n)), nil
	case int16:
		return intOperands(int64(n)), nil
	case int32:
		return intOperands(int64(n)), nil
	case int64:
		return intOperands(n), nil
	case uint:
		return uintOperands(uint64(n)), nil
	case uint8:
		return uintOperands(uint64(n)), nil
	case uint16:
		return uintOperands(uint64(n)), nil
	case uint32:
		return uintOperands(uint64(n)), nil
	case uint64:
		return uintOperands(n), nil
	case uintptr:
		return uintOperands(uint64(n)), nil
	case float32:
		return floatOperands(float64(n), 32)
	case float64:
		return floatOperands(n, 64)
	case string:
		return parseOperands(n)
	}
	return Operands{}, fmt.Errorf("i18n: %T is not a number", n)
}

func intOperands(n int64) Operands {
	if n < 0 {
		// -(math.MinInt64) overflows int64 but not uint64.
		return uintOperands(uint64(-(n + 1)) + 1)
	}
	return uintOperands(uint64(n))
}

func uintOperands(n uint64) Operands {
	if n >= big {
		n = n%big + big
	}
	return Operands{I: n}
}

func floatOperands(f float64, bits int) (Operands, error) {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return Operands{}, fmt.Errorf("i18n: %v is not a finite number", f)
	}
	return parseOperands(strconv.FormatFloat(f, 'f', -1, bits))
}

// parseOperands reads a decimal number: an optional sign, digits, an
// optional fraction, and an optional exponent of compact notation.
func parseOperands(s string) (Operands, error) {
	bad := func() (Operands, error) {
		return Operands{}, fmt.Errorf("i18n: %q is not a decimal number", s)
	}
	num := strings.TrimSpace(s)
	num = strings.TrimPrefix(strings.TrimPrefix(num, "-"), "+")
	exp := 0
	if i := strings.IndexAny(num, "ceCE"); i >= 0 {
		e, err := strconv.Atoi(num[i+1:])
		if err != nil || e < 0 || e > 100 {
			return bad()
		}
		num, exp = num[:i], e
	}
	intPart, frac, hasPoint := strings.Cut(num, ".")
	if intPart == "" && !hasPoint || hasPoint && frac == "" || !digits(intPart) || !digits(frac) {
		return bad()
	}
	// The exponent moves the point to the right.
	if exp > 0 {
		shift := min(exp, len(frac))
		intPart += frac[:shift]
		frac = frac[shift:]
		intPart += strings.Repeat("0", exp-shift)
	}
	o := Operands{C: exp, V: len(frac)}
	o.I = digitsValue(intPart)
	o.F = digitsValue(frac)
	trimmed := strings.TrimRight(frac, "0")
	o.W = len(trimmed)
	o.T = digitsValue(trimmed)
	return o, nil
}

func digits(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}

// digitsValue returns the value of a string of decimal digits, or its last
// 18 digits plus 10^18 when it has more significant ones.
func digitsValue(s string) uint64 {
	s = strings.TrimLeft(s, "0")
	isBig := len(s) > 18
	if isBig {
		s = s[len(s)-18:]
	}
	var v uint64
	for i := 0; i < len(s); i++ {
		v = v*10 + uint64(s[i]-'0')
	}
	if isBig {
		v += big
	}
	return v
}

// PluralCategory returns the plural category of a cardinal number in the
// language of tag, as "1 file" and "2 files": the number of something.
// The rules are those of CLDR for the language, or for the language
// without its region and script when CLDR has none of its own (pt-PT has
// its own, pt-BR is pt's); a language CLDR has no rules for has the one
// category Other.
func PluralCategory(tag string, n Operands) Plural {
	return cardinalRule(pluralRuleOf(tag, cardinalLangs[:], cardinalRules[:]), &n)
}

// OrdinalCategory returns the plural category of an ordinal number in the
// language of tag, as "1st" and "2nd": the place of something in a row.
func OrdinalCategory(tag string, n Operands) Plural {
	return ordinalRule(pluralRuleOf(tag, ordinalLangs[:], ordinalRules[:]), &n)
}

// PluralCategories returns the categories of cardinal numbers in the
// language of tag, in the order of Plural: [PluralOne PluralOther] for
// English.
func PluralCategories(tag string) []Plural {
	return categories(cardinalCategories[pluralRuleOf(tag, cardinalLangs[:], cardinalRules[:])])
}

// OrdinalCategories returns the categories of ordinal numbers in the
// language of tag: [PluralOne PluralTwo PluralFew PluralOther] for
// English.
func OrdinalCategories(tag string) []Plural {
	return categories(ordinalCategories[pluralRuleOf(tag, ordinalLangs[:], ordinalRules[:])])
}

func categories(mask uint8) []Plural {
	var out []Plural
	for p := PluralZero; p <= PluralOther; p++ {
		if mask&(1<<p) != 0 {
			out = append(out, p)
		}
	}
	return out
}

// pluralRuleOf returns the rule set of tag among those of langs, which
// rules index: that of the tag, or of the tag without its last subtag,
// down to its language, as ICU looks plural rules up (CLDR gives plural
// rules no parent locales of their own). Rule set 0 has Other only.
func pluralRuleOf(tag string, langs []string, rules []uint8) uint8 {
	t := Normalize(tag)
	t = stripExtensions(t)
	for t != "" {
		if i, ok := search(langs, t); ok {
			return rules[i]
		}
		i := strings.LastIndexByte(t, '-')
		if i < 0 {
			break
		}
		t = t[:i]
	}
	return 0
}

// search finds s in the sorted list.
func search(list []string, s string) (int, bool) {
	lo, hi := 0, len(list)
	for lo < hi {
		m := int(uint(lo+hi) >> 1)
		if list[m] < s {
			lo = m + 1
		} else {
			hi = m
		}
	}
	return lo, lo < len(list) && list[lo] == s
}

// stripExtensions returns a normalized tag without its extensions and
// private use subtags, as "-u-ca-buddhist".
func stripExtensions(tag string) string {
	parts := strings.Split(tag, "-")
	for i, p := range parts {
		if i > 0 && len(p) == 1 {
			return strings.Join(parts[:i], "-")
		}
	}
	return tag
}
