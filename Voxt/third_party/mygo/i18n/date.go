package i18n

import (
	"strconv"
	"strings"
	"sync"
	"time"
)

// Style is the length of a date or a time as CLDR writes them by style.
type Style uint8

const (
	// NoStyle leaves the date, or the time, out.
	NoStyle Style = iota
	// ShortStyle is numeric: "1/5/26", "1:07 PM".
	ShortStyle
	// MediumStyle abbreviates: "Jan 5, 2026", "1:07:09 PM".
	MediumStyle
	// LongStyle writes the month out: "January 5, 2026", "1:07:09 PM UTC".
	LongStyle
	// FullStyle writes everything: "Monday, January 5, 2026",
	// "1:07:09 PM Coordinated Universal Time".
	FullStyle
)

// index returns the style's index in CLDR's lists, full first.
func (s Style) index() int { return int(FullStyle - s) }

// DateOptions are how FormatDate writes a time: its date, its time or
// both by style, as Intl.DateTimeFormat's dateStyle and timeStyle, or the
// fields of a skeleton, as its other options.
type DateOptions struct {
	// Date and Time are the styles of the date and of the time.
	Date, Time Style
	// Skeleton names the fields to write when Date and Time are
	// NoStyle, in CLDR's letters and any order: "yMMMd" (Jan 5, 2026),
	// "yMMMM" (January 2026), "MMMd", "Ed" (5 Mon), "jm" (the hour and
	// minute in the locale's hour cycle: 1:07 PM, 13:07), "Hm", "hms".
	// The locale's pattern that suits it best is found as ICU's
	// DateTimePatternGenerator finds it. Letters: y, M, L (a month
	// standing alone), d, E (weekday), j, h, H, K, k, a, m, s, z. No
	// style and no skeleton is "yMd". An hour letter other than j is that
	// hour cycle (h 1 to 12, H 0 to 23, K 0 to 11, k 1 to 24), as V8
	// reads Intl's hourCycle. EEEEEE, CLDR's short weekday ("Mo"), has no
	// option of Intl: pages write the abbreviated one ("Mon") in its place.
	Skeleton string
	// HourCycle is 12 or 24 to write times so whatever the locale's
	// hours, as the user's regional settings say (Region.HourCycle); 0
	// keeps the locale's, or the one its extension names ("-u-hc-h23").
	HourCycle int
}

// FormatDate writes t, in its location, in the way of locale, in the
// Gregorian calendar: the locale's names of months, weekdays and day
// periods, its patterns, and its digits.
//
// Time zones other than UTC are written as offsets from GMT in the
// locale's format, "GMT+8", "GMT+05:30", where ICU would name some
// ("China Standard Time").
func FormatDate(locale string, t time.Time, o DateOptions) string {
	return NewDateFormatter(locale, o).Format(t)
}

// A DateFormatter writes times in a locale with options, for writing many:
// FormatDate keeps one per locale and options.
type DateFormatter struct {
	key     *localeKey
	data    *localeData
	sys     *numberSystem
	pattern string
	items   []patItem
	hasMin  bool
	hasSec  bool
}

type dateFormatterKey struct {
	locale string
	o      DateOptions
}

var dateFormatters struct {
	mu sync.RWMutex
	m  map[dateFormatterKey]*DateFormatter
}

// NewDateFormatter returns the formatter of locale and o.
func NewDateFormatter(locale string, o DateOptions) *DateFormatter {
	key := dateFormatterKey{locale, o}
	dateFormatters.mu.RLock()
	f := dateFormatters.m[key]
	dateFormatters.mu.RUnlock()
	if f != nil {
		return f
	}
	k := keyOf(locale)
	d := k.dates()
	f = &DateFormatter{key: k, data: d, sys: d.numbers(k.latn)}
	f.pattern = datePattern(k, d, o)
	f.items = parsePattern(f.pattern)
	for _, it := range f.items {
		f.hasMin = f.hasMin || it.char == 'm'
		f.hasSec = f.hasSec || it.char == 's'
	}
	dateFormatters.mu.Lock()
	if dateFormatters.m == nil || len(dateFormatters.m) > 512 {
		dateFormatters.m = map[dateFormatterKey]*DateFormatter{}
	}
	dateFormatters.m[key] = f
	dateFormatters.mu.Unlock()
	return f
}

// Pattern returns the pattern of CLDR the formatter writes, as "MMM d, y".
func (f *DateFormatter) Pattern() string { return f.pattern }

// hourOf returns the hour letter of an hour cycle, given as 12 or 24 or
// by the locale's extension, or 0.
func hourOf(k *localeKey, cycle int) byte {
	switch cycle {
	case 12:
		return 'h'
	case 24:
		return 'H'
	}
	return k.hourCycle
}

// datePattern returns the pattern of options, as V8's Intl.DateTimeFormat
// works it out.
func datePattern(k *localeKey, d *localeData, o DateOptions) string {
	gen := *d.generator()
	gen.defaultHour = k.defaultHour()
	hc := hourOf(k, o.HourCycle)
	if o.Date == NoStyle && o.Time == NoStyle {
		skel := o.Skeleton
		if skel == "" {
			skel = "yMd"
		}
		if o.HourCycle == 0 {
			// An hour letter of the skeleton is an hour cycle, as V8
			// reads the options Intl makes of it: "hm" is 12:30 AM in
			// Japanese too, whose own patterns have 0:30.
			if i := strings.IndexAny(skel, "hHkK"); i >= 0 {
				hc = skel[i]
			}
		}
		if hc != 0 {
			skel = replaceHours(strings.ReplaceAll(skel, "j", "h"), hc)
		}
		return replaceHours(gen.bestPattern(skel), hc)
	}
	var p string
	switch {
	case o.Time == NoStyle:
		p = d.v[slotDateFormats+o.Date.index()]
	case o.Date == NoStyle:
		p = d.v[slotTimeFormats+o.Time.index()]
	default:
		// ICU puts them together with CLDR's atTime patterns, else the
		// standard ones.
		glue := d.v[slotAtTimeFormats+o.Date.index()]
		if glue == "" {
			glue = d.v[slotDateTimeFormats+o.Date.index()]
		}
		p = strings.NewReplacer("{1}", d.v[slotDateFormats+o.Date.index()], "{0}", d.v[slotTimeFormats+o.Time.index()]).Replace(glue)
	}
	if hc == 0 || o.Time == NoStyle || patternHour(p) == hc {
		return p
	}
	skel := replaceHours(patternSkeleton(p), hc)
	return replaceHours(gen.bestPattern(skel), hc)
}

// Format writes t.
func (f *DateFormatter) Format(t time.Time) string {
	var b strings.Builder
	b.Grow(len(f.pattern) + 16)
	for i := range f.items {
		it := &f.items[i]
		if it.char == 0 {
			b.WriteString(it.text)
			continue
		}
		f.field(&b, it.char, it.n, t)
	}
	s := b.String()
	if strings.Contains(s, "\u202f") {
		// V8 writes the narrow no-break spaces of CLDR's times (1:07 PM)
		// as spaces, for the pages that expected spaces: so does this,
		// so that native UI and pages agree.
		s = strings.ReplaceAll(s, "\u202f", " ")
	}
	return s
}

// num writes v with at least n digits, in the locale's digits.
func (f *DateFormatter) num(b *strings.Builder, v, n int) {
	if v < 0 {
		b.WriteString(f.sys.minus)
		v = -v
	}
	s := strconv.Itoa(v)
	for i := len(s); i < n; i++ {
		b.WriteString(f.sys.digits[0])
	}
	f.sys.writeDigits(b, s)
}

func (f *DateFormatter) field(b *strings.Builder, c byte, n int, t time.Time) {
	v := &f.data.v
	switch c {
	case 'G':
		era := 1
		if t.Year() <= 0 {
			era = 0
		}
		w := 0
		switch {
		case n == 4:
			w = 2
		case n == 5:
			w = 1
		}
		b.WriteString(v[slotEras+w*2+era])
	case 'y', 'Y', 'u':
		year := t.Year()
		if c == 'Y' {
			year = f.weekYear(t)
		}
		if c != 'u' && year <= 0 {
			year = 1 - year
		}
		if n == 2 {
			f.num(b, year%100, 2)
		} else {
			f.num(b, year, n)
		}
	case 'M', 'L':
		m := int(t.Month())
		if n <= 2 {
			f.num(b, m, n)
			return
		}
		ctx := 0
		if c == 'L' {
			ctx = 1
		}
		w := 0 // abbreviated
		switch n {
		case 4:
			w = 2
		case 5:
			w = 1
		}
		b.WriteString(v[slotMonths+(ctx*3+w)*12+m-1])
	case 'd':
		f.num(b, t.Day(), n)
	case 'D':
		f.num(b, t.YearDay(), n)
	case 'E', 'e', 'c':
		if c != 'E' && n <= 2 {
			first, _ := weekData(f.key.regionOrLikely())
			if f.key.firstDay >= 0 {
				first = f.key.firstDay
			}
			f.num(b, (int(t.Weekday())-int(first)+7)%7+1, n)
			return
		}
		ctx := 0
		if c == 'c' {
			ctx = 1
		}
		w := 0 // abbreviated
		switch n {
		case 4:
			w = 3
		case 5:
			w = 1
		case 6:
			w = 2
		}
		b.WriteString(v[slotDays+(ctx*4+w)*7+int(t.Weekday())])
	case 'a', 'b':
		w := 0
		switch {
		case n == 4:
			w = 2
		case n == 5:
			w = 1
		}
		p := 0
		if t.Hour() >= 12 {
			p = 1
		}
		b.WriteString(v[slotDayPeriods+w*len(periodNames)+p])
	case 'B':
		f.flexiblePeriod(b, n, t)
	case 'h':
		h := t.Hour() % 12
		if h == 0 {
			h = 12
		}
		f.num(b, h, n)
	case 'H':
		f.num(b, t.Hour(), n)
	case 'K':
		f.num(b, t.Hour()%12, n)
	case 'k':
		h := t.Hour()
		if h == 0 {
			h = 24
		}
		f.num(b, h, n)
	case 'm':
		f.num(b, t.Minute(), n)
	case 's':
		f.num(b, t.Second(), n)
	case 'S':
		frac := strconv.Itoa(t.Nanosecond() + 1e9)[1:]
		for len(frac) < n {
			frac += "0"
		}
		f.sys.writeDigits(b, frac[:n])
	case 'z', 'O', 'v', 'V', 'Z', 'X', 'x':
		f.zone(b, c, n, t)
	default:
		b.WriteString(strings.Repeat(string(c), n))
	}
}

// weekYear returns the year of the week of t, by the week data of the
// locale's region: the first week of a year is the first that has at
// least the region's minimal days of it.
func (f *DateFormatter) weekYear(t time.Time) int {
	first, minDays := weekData(f.key.regionOrLikely())
	if f.key.firstDay >= 0 {
		first = f.key.firstDay
	}
	day := time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, time.UTC)
	for _, y := range [...]int{t.Year() + 1, t.Year(), t.Year() - 1} {
		jan1 := time.Date(y, 1, 1, 0, 0, 0, 0, time.UTC)
		before := (int(jan1.Weekday()) - int(first) + 7) % 7
		start := jan1.AddDate(0, 0, -before)
		if 7-before < minDays {
			start = start.AddDate(0, 0, 7)
		}
		if !day.Before(start) {
			return y
		}
	}
	return t.Year()
}

// flexiblePeriod writes the day period of CLDR's rules for the locale's
// language ("in the morning", "清晨"), as ICU writes B: noon only at noon
// sharp, midnight never, and AM or PM for a language without rules.
func (f *DateFormatter) flexiblePeriod(b *strings.Builder, n int, t time.Time) {
	w := 0
	switch {
	case n == 4 || n > 5:
		w = 2
	case n == 5:
		w = 1
	}
	rules := periodRules(f.key.lang, f.data.tag)
	v := &f.data.v
	if rules == nil {
		f.field(b, 'a', n, t)
		return
	}
	h := t.Hour()
	p := -1
	if h == 12 && (!f.hasMin || t.Minute() == 0) && (!f.hasSec || t.Second() == 0) && rules.noon {
		p = 3
	}
	if p < 0 || v[slotDayPeriods+w*len(periodNames)+p] == "" {
		p = rules.byHour[h]
	}
	name := ""
	if p >= 2 {
		name = v[slotDayPeriods+w*len(periodNames)+p]
	}
	if name == "" {
		f.field(b, 'a', min(n, 5), t)
		return
	}
	b.WriteString(name)
}

// dayPeriodRules are the periods of the hours of a day.
type dayPeriodRules struct {
	byHour [24]int // the index of each hour's period in periodNames
	noon   bool
}

var periodRuleCache sync.Map // string -> *dayPeriodRules

// periodRules returns the rules of day periods of a locale: of the
// locale, else of its language; nil when CLDR has none MyGo keeps.
func periodRules(lang, tag string) *dayPeriodRules {
	loadDates()
	i, ok := search(tables.periodLangs, tag)
	if !ok {
		i, ok = search(tables.periodLangs, lang)
	}
	if !ok {
		return nil
	}
	if r, ok := periodRuleCache.Load(i); ok {
		return r.(*dayPeriodRules)
	}
	r := parsePeriodRules(fmtPeriodRules[i])
	periodRuleCache.Store(i, r)
	return r
}

// parsePeriodRules reads the rules of fmtPeriodRules: period@minute, or
// period:from-before in minutes, a span that may cross midnight
// (night1:1260-360 is 21:00 to 06:00).
func parsePeriodRules(text string) *dayPeriodRules {
	r := &dayPeriodRules{}
	for h := range r.byHour {
		r.byHour[h] = 0
		if h >= 12 {
			r.byHour[h] = 1
		}
	}
	for _, rule := range strings.Fields(text) {
		if p, at, ok := strings.Cut(rule, "@"); ok {
			if pi, _ := strconv.Atoi(p); pi == 3 && at == "720" {
				r.noon = true
			}
			continue
		}
		p, span, _ := strings.Cut(rule, ":")
		from, before, _ := strings.Cut(span, "-")
		pi, _ := strconv.Atoi(p)
		lo, _ := strconv.Atoi(from)
		hi, _ := strconv.Atoi(before)
		if hi <= lo {
			hi += 24 * 60
		}
		for m := lo; m < hi; m += 60 {
			r.byHour[m/60%24] = pi
		}
	}
	return r
}

// zone writes the time zone of t: UTC's names for UTC, else its offset
// from GMT in the locale's format.
func (f *DateFormatter) zone(b *strings.Builder, c byte, n int, t time.Time) {
	v := &f.data.v
	name, off := t.Zone()
	long := n >= 4 || c == 'V'
	if (c == 'z' || c == 'v') && off == 0 && (t.Location() == time.UTC || name == "UTC") {
		if long {
			b.WriteString(v[slotZone+3])
		} else {
			b.WriteString(v[slotZone+4])
		}
		return
	}
	if off == 0 {
		b.WriteString(v[slotZone+1])
		return
	}
	hourFormat := v[slotZone+2]
	pos, neg, _ := strings.Cut(hourFormat, ";")
	p := pos
	if off < 0 {
		p, off = neg, -off
	}
	hours, mins := off/3600, off%3600/60
	hourDigits := 2 // ICU's long form has two digits of hours whatever the pattern
	if !long {
		// The short form: one digit of hours at least, and the pattern
		// cut after its hours, as ICU's truncateOffsetPattern cuts it,
		// when there are no minutes.
		hourDigits = 1
		if mins == 0 {
			if i := strings.LastIndex(p[:max(strings.Index(p, "mm"), 0)], "H"); i >= 0 {
				p = p[:i+1]
			}
		}
	}
	var o strings.Builder
	for i := 0; i < len(p); {
		switch {
		case p[i] == 'H':
			f.num(&o, hours, hourDigits)
			for i < len(p) && p[i] == 'H' {
				i++
			}
		case strings.HasPrefix(p[i:], "mm"):
			f.num(&o, mins, 2)
			i += 2
		default:
			o.WriteByte(p[i])
			i++
		}
	}
	b.WriteString(strings.Replace(v[slotZone], "{0}", o.String(), 1))
}
