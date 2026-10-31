package i18n

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

// Format writes the message with args, the values of its arguments by
// name, for locale: the language whose plural rules pick the cases of
// plural and selectordinal arguments, and whose conventions numbers and
// dates follow. An argument args lacks is written as it is in the message,
// {name}. The pseudo locales (IsPseudo) transform the message, which keeps
// the plural rules of English.
//
// Values: a string, an integer or floating-point number, a string of a
// decimal number (which keeps its fraction digits: "1.50"), a bool, a
// time.Time, an error, a fmt.Stringer, or anything fmt.Sprint writes.
func (m *Message) Format(locale string, args map[string]any) string {
	if IsPseudo(locale) {
		return m.format(locale, "en", args)
	}
	return m.format(locale, locale, args)
}

// format formats the message with the plural rules of pluralLocale, the
// language it is in, and numbers and dates for locale. A pseudo locale
// transforms it.
func (m *Message) format(locale, pluralLocale string, args map[string]any) string {
	f := formatter{locale: locale, pluralLocale: pluralLocale, args: args, pseudo: pseudoFor(locale)}
	if f.pseudo != nil {
		f.locale = pluralLocale
	}
	var b strings.Builder
	f.nodes(&b, m.nodes, nil)
	if f.pseudo != nil {
		return f.pseudo.wrap(b.String())
	}
	return b.String()
}

// formatter formats a message.
type formatter struct {
	locale       string // of numbers and dates
	pluralLocale string // whose plural rules apply: the message's language
	args         map[string]any
	pseudo       pseudoMethod
}

// pluralValue is the number of the plural argument whose case is being
// written, for #.
type pluralValue struct {
	num number
	ok  bool
	raw any
}

func (f *formatter) nodes(b *strings.Builder, nodes []node, pound *pluralValue) {
	for i := range nodes {
		n := &nodes[i]
		switch n.kind {
		case nodeText:
			if f.pseudo != nil {
				b.WriteString(f.pseudo.text(n.text))
			} else {
				b.WriteString(n.text)
			}
		case nodePound:
			if pound == nil {
				b.WriteByte('#')
				continue
			}
			var s string
			if pound.ok {
				s = formatNumber(f.locale, pound.num, "")
			} else {
				s = valueString(f.locale, pound.raw)
			}
			f.placeholder(b, s)
		default:
			f.argument(b, n, pound)
		}
	}
}

// placeholder writes what an argument wrote.
func (f *formatter) placeholder(b *strings.Builder, s string) {
	if f.pseudo != nil {
		s = f.pseudo.placeholder(s)
	}
	b.WriteString(s)
}

func (f *formatter) argument(b *strings.Builder, n *node, pound *pluralValue) {
	v, ok := f.args[n.text]
	if !ok {
		b.WriteString("{" + n.text + "}")
		return
	}
	switch n.kind {
	case nodeArg:
		f.placeholder(b, valueString(f.locale, v))
	case nodeNumber:
		if num, err := toNumber(v); err == nil {
			f.placeholder(b, formatNumber(f.locale, num, n.style))
		} else {
			f.placeholder(b, valueString(f.locale, v))
		}
	case nodeDate, nodeTime:
		if t, ok := v.(time.Time); ok {
			kind := "date"
			if n.kind == nodeTime {
				kind = "time"
			}
			f.placeholder(b, formatTime(f.locale, t, kind, n.style))
		} else {
			f.placeholder(b, valueString(f.locale, v))
		}
	case nodeSelect:
		key := selectKey(v)
		c := findCase(n.cases, key)
		f.nodes(b, c.nodes, pound)
	case nodePlural, nodeSelectOrdinal:
		num, err := toNumber(v)
		if err != nil {
			// Not a number: the other case, whose # writes the value.
			f.nodes(b, findCase(n.cases, "other").nodes, &pluralValue{raw: v})
			return
		}
		// An exact case matches the number itself, a keyword the number
		// less the offset.
		var chosen *msgCase
		if exact, err := parseOperands(num.decimal); err == nil && !strings.HasPrefix(num.decimal, "-") {
			for i := range n.cases {
				c := &n.cases[i]
				if strings.HasPrefix(c.key, "=") && c.exact.I == exact.I && c.exact.T == exact.T && c.exact.W == exact.W {
					chosen = c
					break
				}
			}
		}
		shown := num.minus(n.offset)
		if chosen == nil {
			// The category of the number as # writes it, as ICU picks
			// it: 1.0004 is written "1", whose category is one.
			o := shown.shownOperands()
			var cat Plural
			if n.kind == nodePlural {
				cat = PluralCategory(f.pluralLocale, o)
			} else {
				cat = OrdinalCategory(f.pluralLocale, o)
			}
			chosen = findCase(n.cases, cat.String())
		}
		f.nodes(b, chosen.nodes, &pluralValue{num: shown, ok: true})
	}
}

// findCase returns the case of key, else other, which every argument has.
func findCase(cases []msgCase, key string) *msgCase {
	var other *msgCase
	for i := range cases {
		switch cases[i].key {
		case key:
			return &cases[i]
		case "other":
			other = &cases[i]
		}
	}
	return other
}

// selectKey returns the case a value of a select argument picks.
func selectKey(v any) string {
	switch v := v.(type) {
	case string:
		return v
	case bool:
		return strconv.FormatBool(v)
	case fmt.Stringer:
		return v.String()
	}
	if num, err := toNumber(v); err == nil {
		return num.decimal
	}
	return fmt.Sprint(v)
}

// number is a decimal number as it is written, with its sign: "-1.50".
// An exact one, given as a string, shows its fraction digits as they
// are; others up to three of them, as ICU writes numbers by default.
type number struct {
	decimal string
	exact   bool
}

// shownOperands returns the operands of the number as a message writes it
// by default: with its own fraction digits when exact, else rounded to
// three, without trailing zeros, as Intl.PluralRules reads it.
func (n number) shownOperands() Operands {
	if n.exact {
		o, _ := parseOperands(n.decimal)
		return o
	}
	_, intPart, frac := splitDecimal(n.decimal)
	intPart, frac = roundDigits(intPart, frac, 3, false)
	frac = strings.TrimRight(frac, "0")
	if frac != "" {
		intPart += "." + frac
	}
	o, _ := parseOperands(intPart)
	return o
}

var errNotNumber = errors.New("i18n: not a number")

// toNumber returns v as a decimal number when it is one: an integer, a
// finite float, a string of a decimal number, or Operands.
func toNumber(v any) (number, error) {
	switch v := v.(type) {
	case int:
		return number{decimal: strconv.FormatInt(int64(v), 10)}, nil
	case int8:
		return number{decimal: strconv.FormatInt(int64(v), 10)}, nil
	case int16:
		return number{decimal: strconv.FormatInt(int64(v), 10)}, nil
	case int32:
		return number{decimal: strconv.FormatInt(int64(v), 10)}, nil
	case int64:
		return number{decimal: strconv.FormatInt(v, 10)}, nil
	case uint:
		return number{decimal: strconv.FormatUint(uint64(v), 10)}, nil
	case uint8:
		return number{decimal: strconv.FormatUint(uint64(v), 10)}, nil
	case uint16:
		return number{decimal: strconv.FormatUint(uint64(v), 10)}, nil
	case uint32:
		return number{decimal: strconv.FormatUint(uint64(v), 10)}, nil
	case uint64:
		return number{decimal: strconv.FormatUint(v, 10)}, nil
	case uintptr:
		return number{decimal: strconv.FormatUint(uint64(v), 10)}, nil
	case float32:
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return number{}, errNotNumber
		}
		return number{decimal: strconv.FormatFloat(float64(v), 'f', -1, 32)}, nil
	case float64:
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return number{}, errNotNumber
		}
		return number{decimal: strconv.FormatFloat(v, 'f', -1, 64)}, nil
	case string:
		s := strings.TrimSpace(v)
		if !isDecimal(s) {
			return number{}, errNotNumber
		}
		return number{decimal: s, exact: true}, nil
	case Operands:
		s := strconv.FormatUint(v.I, 10)
		if v.I >= big {
			s = strconv.FormatUint(v.I-big, 10)
		}
		if v.V > 0 {
			f := strconv.FormatUint(v.F%big, 10)
			s += "." + strings.Repeat("0", max(v.V-len(f), 0)) + f
		}
		return number{decimal: s, exact: true}, nil
	}
	return number{}, errNotNumber
}

// isDecimal reports whether s is a decimal number: an optional minus
// sign, digits, and an optional fraction.
func isDecimal(s string) bool {
	s = strings.TrimPrefix(s, "-")
	i, f, point := strings.Cut(s, ".")
	return i != "" && digits(i) && digits(f) && (!point || f != "")
}

// minus returns n less offset, with as many fraction digits as n has.
func (n number) minus(offset int64) number {
	if offset == 0 {
		return n
	}
	if !strings.Contains(n.decimal, ".") {
		if i, err := strconv.ParseInt(n.decimal, 10, 64); err == nil && i >= math.MinInt64+offset {
			return number{decimal: strconv.FormatInt(i-offset, 10), exact: n.exact}
		}
	}
	f, err := strconv.ParseFloat(n.decimal, 64)
	if err != nil {
		return n
	}
	_, frac, _ := strings.Cut(n.decimal, ".")
	return number{decimal: strconv.FormatFloat(f-float64(offset), 'f', len(frac), 64), exact: n.exact}
}

// valueString writes the value of a simple argument.
func valueString(locale string, v any) string {
	switch v := v.(type) {
	case string:
		return v
	case time.Time:
		return formatTime(locale, v, "datetime", "short")
	case error:
		return v.Error()
	case fmt.Stringer:
		return v.String()
	case bool:
		return strconv.FormatBool(v)
	case nil:
		return ""
	}
	if num, err := toNumber(v); err == nil {
		return formatNumber(locale, num, "")
	}
	return fmt.Sprint(v)
}

// formatNumber writes a number for locale in an ICU number style: ""
// (the default: up to three fraction digits, or those of an exact
// number), "integer", "percent", or a skeleton of ICU's number
// skeletons ("::percent .00"), of which it reads percent, integer,
// precision-integer, the precision of fraction digits (.00, .0#, .##)
// and group-off, and writes the rest as the default. Integers and
// percents round half to even, as ICU's MessageFormat does; the default,
// half away from zero, as Intl.NumberFormat does.
func formatNumber(locale string, n number, style string) string {
	var o NumberOptions
	stems := []string{style}
	if rest, ok := strings.CutPrefix(style, "::"); ok {
		stems = strings.Fields(rest)
	}
	fraction := false
	for _, stem := range stems {
		switch {
		case stem == "integer" || stem == "precision-integer":
			o.Integer, o.HalfEven = true, true
		case stem == "percent":
			o.Percent, o.HalfEven = true, true
		case stem == "group-off":
			o.NoGrouping = true
		case strings.HasPrefix(stem, ".") && strings.Trim(stem[1:], "0#") == "" && len(stem) > 1:
			o.MinFractionDigits = strings.Count(stem, "0")
			o.MaxFractionDigits = len(stem) - 1
			fraction = true
		}
	}
	if n.exact && !fraction && !o.Integer && !o.Percent {
		_, frac, _ := strings.Cut(n.decimal, ".")
		o.MinFractionDigits, o.MaxFractionDigits = min(len(frac), 100), min(len(frac), 100)
	}
	return NewNumberFormatter(locale, o).formatDecimal(n.decimal)
}

// timeFormat formats the dates and times of messages: formatDate, once
// linkTimes linked it.
var timeFormat atomic.Pointer[func(locale string, t time.Time, kind, style string) string]

// linkTimes links the formatting of dates and times into messages, for
// the catalogs and messages an app makes: NewCatalog, NewLibraryCatalog
// and ParseMessage call it. MyGo's own strings, which every app formats,
// have no dates, so an app that formats none does not link the patterns
// and names of dates of CLDR's locales.
func linkTimes() {
	if timeFormat.Load() == nil {
		f := formatDate
		timeFormat.Store(&f)
	}
}

// formatTime writes a time of a message for locale (see formatDate), as
// RFC 3339 before linkTimes linked the formatting of dates, which no
// message of MyGo's own strings needs.
func formatTime(locale string, t time.Time, kind, style string) string {
	if f := timeFormat.Load(); f != nil {
		return (*f)(locale, t, kind, style)
	}
	return t.Format(time.RFC3339)
}

// formatDate writes a time for locale: its date, its time, or both
// ("datetime"), in an ICU style, "short", "medium", "long" or "full" (""
// is "medium"), or a skeleton ("::yMMMd").
func formatDate(locale string, t time.Time, kind, style string) string {
	if skel, ok := strings.CutPrefix(style, "::"); ok {
		return FormatDate(locale, t, DateOptions{Skeleton: strings.TrimSpace(skel)})
	}
	s := MediumStyle
	switch style {
	case "short":
		s = ShortStyle
	case "long":
		s = LongStyle
	case "full":
		s = FullStyle
	}
	switch kind {
	case "time":
		return FormatDate(locale, t, DateOptions{Time: s})
	case "datetime":
		return FormatDate(locale, t, DateOptions{Date: ShortStyle, Time: ShortStyle})
	}
	return FormatDate(locale, t, DateOptions{Date: s})
}
