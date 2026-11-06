package i18n

import (
	"math"
	"strconv"
	"strings"
	"sync"
	"unicode/utf8"
)

// NumberOptions are how FormatNumber writes a number. The zero value
// writes it as Intl.NumberFormat does by default: grouped, with up to
// three fraction digits, rounded half away from zero.
type NumberOptions struct {
	// Percent writes the number times 100, with the locale's percent sign
	// where its pattern puts it: 0.25 is "25%" in English, "25 %" in
	// German, "%25" in Turkish. Percents have no fraction digits unless
	// MaxFractionDigits says otherwise.
	Percent bool
	// Integer rounds to a whole number, as MaxFractionDigits 0 would.
	Integer bool
	// MinFractionDigits and MaxFractionDigits bound the digits after the
	// decimal separator: 1.5 with MinFractionDigits 2 is "1.50", 1.2345
	// with MaxFractionDigits 2 is "1.23". MaxFractionDigits 0 is the
	// style's default (3, none for percents), or MinFractionDigits when
	// that is more.
	MinFractionDigits, MaxFractionDigits int
	// MinIntegerDigits pads the integer part with zeros to as many
	// digits: 7 with 2 is "07".
	MinIntegerDigits int
	// NoGrouping leaves out the separators of thousands: "12345".
	NoGrouping bool
	// HalfEven rounds a number halfway between two to the even one, as
	// ICU's MessageFormat does ("2.5" is "2"), rather than away from zero
	// ("3"), as Intl.NumberFormat does.
	HalfEven bool
}

// FormatNumber writes a number in the way of locale: its decimal and
// grouping separators, its grouping (the Indian 12,34,567 too, and none
// of four digits where the locale groups from five, as Spanish and
// Polish do), its minus and percent signs, and the digits of its default
// numbering system (Arabic-Indic in Persian, Bengali in Bengali; a locale
// with "-u-nu-latn" writes Latin digits). n is an integer of any Go type,
// a float32 or float64 (its fewest digits that read back as it, which
// are then rounded: 1.005 is "1.01" with two fraction digits, as Intl
// writes it), or a string of a decimal number, as "1234.5", exact
// however many digits it has.
//
// NaN and infinities are the locale's symbols of them; a value that is
// not a number is written by fmt.
func FormatNumber(locale string, n any, o NumberOptions) string {
	return NewNumberFormatter(locale, o).Format(n)
}

// A NumberFormatter writes numbers in a locale with options, for writing
// many: FormatNumber keeps one per locale and options.
type NumberFormatter struct {
	sys      *numberSystem
	pat      *numberPattern
	o        NumberOptions
	min, max int
}

type numberFormatterKey struct {
	locale string
	o      NumberOptions
}

var numberFormatters struct {
	mu sync.RWMutex
	m  map[numberFormatterKey]*NumberFormatter
}

// NewNumberFormatter returns the formatter of locale and o.
func NewNumberFormatter(locale string, o NumberOptions) *NumberFormatter {
	key := numberFormatterKey{locale, o}
	numberFormatters.mu.RLock()
	f := numberFormatters.m[key]
	numberFormatters.mu.RUnlock()
	if f != nil {
		return f
	}
	k := keyOf(locale)
	sys := k.dat().numbers(k.latn)
	f = &NumberFormatter{sys: sys, pat: &sys.decimal, o: o}
	if o.Percent {
		f.pat = &sys.percent
	}
	f.min, f.max = o.MinFractionDigits, o.MaxFractionDigits
	if f.min < 0 {
		f.min = 0
	}
	switch {
	case o.Integer:
		f.min, f.max = 0, 0
	case f.max <= 0 && f.min == 0:
		f.max = f.pat.maxFrac
	}
	if f.max < f.min {
		f.max = f.min
	}
	numberFormatters.mu.Lock()
	if numberFormatters.m == nil || len(numberFormatters.m) > 512 {
		numberFormatters.m = map[numberFormatterKey]*NumberFormatter{}
	}
	numberFormatters.m[key] = f
	numberFormatters.mu.Unlock()
	return f
}

// Format writes n (see FormatNumber).
func (f *NumberFormatter) Format(n any) string {
	switch v := n.(type) {
	case float64:
		if s, ok := f.special(v); ok {
			return s
		}
	case float32:
		if s, ok := f.special(float64(v)); ok {
			return s
		}
	}
	num, err := toNumber(n)
	if err != nil {
		return valueString("", n)
	}
	return f.formatDecimal(num.decimal)
}

// special writes NaN and the infinities.
func (f *NumberFormatter) special(v float64) (string, bool) {
	switch {
	case math.IsNaN(v):
		// In the pattern's affixes, as ICU writes it: "NaN%", "%NaN".
		return f.pat.prefix + f.sys.nan + f.pat.suffix, true
	case math.IsInf(v, 1):
		return f.pat.prefix + f.sys.infinity + f.pat.suffix, true
	case math.IsInf(v, -1):
		return f.pat.negPrefix + f.sys.infinity + f.pat.negSuffix, true
	}
	return "", false
}

// formatDecimal writes a decimal number, "-1234.5".
func (f *NumberFormatter) formatDecimal(decimal string) string {
	min, max := f.min, f.max
	neg, intPart, frac := splitDecimal(decimal)
	if f.o.Percent {
		intPart, frac = shiftLeft(intPart, frac, 2)
	}
	intPart, frac = roundDigits(intPart, frac, max, f.o.HalfEven)
	for len(frac) > min && frac[len(frac)-1] == '0' {
		frac = frac[:len(frac)-1]
	}
	for len(frac) < min {
		frac += "0"
	}
	var b strings.Builder
	b.Grow(len(intPart) + len(frac) + 16)
	p := f.pat
	if neg {
		b.WriteString(p.negPrefix)
	} else {
		b.WriteString(p.prefix)
	}
	f.sys.writeInteger(&b, intPart, p, !f.o.NoGrouping, f.o.MinIntegerDigits)
	if frac != "" {
		b.WriteString(f.sys.decimalSep)
		f.sys.writeDigits(&b, frac)
	}
	if neg {
		b.WriteString(p.negSuffix)
	} else {
		b.WriteString(p.suffix)
	}
	return b.String()
}

// splitDecimal splits a decimal number into its sign, integer digits
// (without leading zeros, "0" for none) and fraction digits.
func splitDecimal(s string) (neg bool, intPart, frac string) {
	if strings.HasPrefix(s, "-") {
		neg, s = true, s[1:]
	} else {
		s = strings.TrimPrefix(s, "+")
	}
	intPart, frac, _ = strings.Cut(s, ".")
	intPart = strings.TrimLeft(intPart, "0")
	if intPart == "" {
		intPart = "0"
	}
	return neg, intPart, frac
}

// shiftLeft multiplies a number by 10^n.
func shiftLeft(intPart, frac string, n int) (string, string) {
	for frac != "" && n > 0 {
		intPart += frac[:1]
		frac = frac[1:]
		n--
	}
	intPart += strings.Repeat("0", n)
	intPart = strings.TrimLeft(intPart, "0")
	if intPart == "" {
		intPart = "0"
	}
	return intPart, frac
}

// roundDigits rounds a number to max fraction digits, half away from
// zero or half to even.
func roundDigits(intPart, frac string, max int, halfEven bool) (string, string) {
	if len(frac) <= max {
		return intPart, frac
	}
	drop := frac[max:]
	keep := frac[:max]
	up := false
	switch {
	case drop[0] > '5':
		up = true
	case drop[0] == '5':
		if strings.Trim(drop[1:], "0") != "" || !halfEven {
			up = true
		} else {
			// Exactly half: to the even last digit.
			last := intPart[len(intPart)-1]
			if keep != "" {
				last = keep[len(keep)-1]
			}
			up = (last-'0')%2 == 1
		}
	}
	if !up {
		return intPart, keep
	}
	digits := []byte(intPart + keep)
	i := len(digits) - 1
	for ; i >= 0; i-- {
		if digits[i] == '9' {
			digits[i] = '0'
			continue
		}
		digits[i]++
		break
	}
	s := string(digits)
	if i < 0 {
		s = "1" + s
	}
	return s[:len(s)-len(keep)], s[len(s)-len(keep):]
}

// numberSystem is how a locale writes numbers in a numbering system.
type numberSystem struct {
	digits                                  [10]string
	latin                                   bool
	decimalSep, group, minus, plus          string
	percentSign, exponential, infinity, nan string
	minGrouping                             int
	decimal, percent                        numberPattern
}

// numbers returns the numbering system of the locale: its default, or
// latn.
func (d *localeData) numbers(latn bool) *numberSystem {
	d.numOnce.Do(func() {
		minGrouping, _ := strconv.Atoi(d.v[slotMinGrouping])
		for i := range d.num {
			base := slotNumbers + i*11
			v := d.v[base : base+11]
			if v[0] == "" {
				continue // the default is latn
			}
			s := &numberSystem{
				decimalSep: v[1], group: v[2], minus: v[3], plus: v[4], percentSign: v[5],
				exponential: v[6], infinity: v[7], nan: v[8], minGrouping: max(minGrouping, 1),
				latin: v[0] == "0123456789",
			}
			digits := v[0]
			for j := range s.digits {
				_, size := utf8.DecodeRuneInString(digits)
				s.digits[j], digits = digits[:size], digits[size:]
			}
			s.decimal = parseNumberPattern(v[9], s)
			s.percent = parseNumberPattern(v[10], s)
			d.num[i] = s
		}
		if d.num[1] == nil {
			d.num[1] = d.num[0]
		}
	})
	if latn {
		return d.num[1]
	}
	return d.num[0]
}

// writeDigits writes ASCII digits in the system's.
func (s *numberSystem) writeDigits(b *strings.Builder, digits string) {
	if s.latin {
		b.WriteString(digits)
		return
	}
	for i := 0; i < len(digits); i++ {
		b.WriteString(s.digits[digits[i]-'0'])
	}
}

// writeInteger writes the integer digits, grouped as the pattern groups
// them when group.
func (s *numberSystem) writeInteger(b *strings.Builder, digits string, p *numberPattern, group bool, minInt int) {
	for len(digits) < max(p.minInt, minInt) {
		digits = "0" + digits
	}
	if !group || p.primary == 0 || len(digits) < p.primary+s.minGrouping {
		s.writeDigits(b, digits)
		return
	}
	// The sizes of the groups, from the left.
	n := len(digits)
	first := (n - p.primary) % p.secondary
	if first == 0 {
		first = p.secondary
	}
	pos := 0
	for size := first; pos < n; {
		s.writeDigits(b, digits[pos:pos+size])
		pos += size
		if pos < n {
			b.WriteString(s.group)
		}
		if n-pos == p.primary {
			size = p.primary
		} else {
			size = p.secondary
		}
	}
}

// numberPattern is a pattern of CLDR's decimal or percent formats, as
// "#,##,##0.###" or "#,##0 %", with the locale's symbols in its prefixes
// and suffixes.
type numberPattern struct {
	prefix, suffix, negPrefix, negSuffix string
	primary, secondary                   int // the sizes of groups, 0 for none
	minInt, minFrac, maxFrac             int
}

func parseNumberPattern(pattern string, s *numberSystem) numberPattern {
	pos, neg, hasNeg := strings.Cut(pattern, ";")
	var p numberPattern
	var number string
	p.prefix, number, p.suffix = splitAffixes(pos, s)
	if hasNeg {
		p.negPrefix, _, p.negSuffix = splitAffixes(neg, s)
	} else {
		p.negPrefix, p.negSuffix = s.minus+p.prefix, p.suffix
	}
	intPart, frac, _ := strings.Cut(number, ".")
	for _, c := range frac {
		if c == '0' {
			p.minFrac++
		}
		p.maxFrac++
	}
	if i := strings.LastIndexByte(intPart, ','); i >= 0 {
		p.primary = len(intPart) - i - 1
		p.secondary = p.primary
		if j := strings.LastIndexByte(intPart[:i], ','); j >= 0 {
			p.secondary = i - j - 1
		}
	}
	p.minInt = strings.Count(intPart, "0")
	return p
}

// splitAffixes splits a subpattern into its prefix, number and suffix,
// the affixes with the system's symbols for %, - and +, and without
// quotes.
func splitAffixes(sub string, s *numberSystem) (prefix, number, suffix string) {
	start, end := -1, -1
	quoted := false
	for i := 0; i < len(sub); i++ {
		c := sub[i]
		if c == '\'' {
			quoted = !quoted
			continue
		}
		if !quoted && strings.IndexByte("#0,.@", c) >= 0 {
			if start < 0 {
				start = i
			}
			end = i + 1
		}
	}
	if start < 0 {
		return affix(sub, s), "", ""
	}
	return affix(sub[:start], s), sub[start:end], affix(sub[end:], s)
}

func affix(a string, s *numberSystem) string {
	var b strings.Builder
	quoted := false
	for i := 0; i < len(a); {
		r, size := utf8.DecodeRuneInString(a[i:])
		i += size
		switch {
		case r == '\'':
			if i < len(a) && a[i] == '\'' {
				b.WriteByte('\'')
				i++
				continue
			}
			quoted = !quoted
		case quoted:
			b.WriteRune(r)
		case r == '%':
			b.WriteString(s.percentSign)
		case r == '-':
			b.WriteString(s.minus)
		case r == '+':
			b.WriteString(s.plus)
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}
