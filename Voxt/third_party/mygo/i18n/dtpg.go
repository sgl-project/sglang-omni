package i18n

import "strings"

// A date-time pattern generator, as ICU's DateTimePatternGenerator: it
// finds, among a locale's patterns of CLDR, the one that best suits a
// skeleton (the fields a date shows, as "yMMMd" or "jm", in any order),
// adjusts the widths of its fields to the skeleton's, and puts a date and
// a time together when no one pattern has both. Intl.DateTimeFormat
// formats its options so: this follows ICU's code (dtptngen.cpp), its
// order of patterns included, so that the same skeleton gives the same
// pattern.

// The fields of a skeleton, in ICU's order (UDateTimePatternField).
const (
	fieldEra = iota
	fieldYear
	fieldQuarter
	fieldMonth
	fieldWeekOfYear
	fieldWeekOfMonth
	fieldWeekday
	fieldDayOfYear
	fieldDayOfWeekInMonth
	fieldDay
	fieldDayPeriod
	fieldHour
	fieldMinute
	fieldSecond
	fieldFractionalSecond
	fieldZone
	fieldCount
)

const (
	dtNarrow  = -0x101
	dtShorter = -0x102
	dtShort   = -0x103
	dtLong    = -0x104
	dtNumeric = 0x100
	dtDelta   = 0x10

	extraField   = 0x10000
	missingField = 0x1000

	dateMask = 1<<fieldDayPeriod - 1
	timeMask = 1<<fieldCount - 1 - dateMask
)

// dtType is a row of ICU's dtTypes: a pattern letter from a length on, its
// field and its type.
type dtType struct {
	char   byte
	field  int8
	typ    int16
	minLen int8
}

var dtTypes = [...]dtType{
	{'G', fieldEra, dtShort, 1}, {'G', fieldEra, dtLong, 4}, {'G', fieldEra, dtNarrow, 5},
	{'y', fieldYear, dtNumeric, 1}, {'Y', fieldYear, dtNumeric + dtDelta, 1},
	{'u', fieldYear, dtNumeric + 2*dtDelta, 1}, {'r', fieldYear, dtNumeric + 3*dtDelta, 1},
	{'U', fieldYear, dtShort, 1}, {'U', fieldYear, dtLong, 4}, {'U', fieldYear, dtNarrow, 5},
	{'Q', fieldQuarter, dtNumeric, 1}, {'Q', fieldQuarter, dtShort, 3}, {'Q', fieldQuarter, dtLong, 4}, {'Q', fieldQuarter, dtNarrow, 5},
	{'q', fieldQuarter, dtNumeric + dtDelta, 1}, {'q', fieldQuarter, dtShort - dtDelta, 3}, {'q', fieldQuarter, dtLong - dtDelta, 4}, {'q', fieldQuarter, dtNarrow - dtDelta, 5},
	{'M', fieldMonth, dtNumeric, 1}, {'M', fieldMonth, dtShort, 3}, {'M', fieldMonth, dtLong, 4}, {'M', fieldMonth, dtNarrow, 5},
	{'L', fieldMonth, dtNumeric + dtDelta, 1}, {'L', fieldMonth, dtShort - dtDelta, 3}, {'L', fieldMonth, dtLong - dtDelta, 4}, {'L', fieldMonth, dtNarrow - dtDelta, 5},
	{'l', fieldMonth, dtNumeric + dtDelta, 1},
	{'w', fieldWeekOfYear, dtNumeric, 1},
	{'W', fieldWeekOfMonth, dtNumeric, 1},
	{'E', fieldWeekday, dtShort, 1}, {'E', fieldWeekday, dtLong, 4}, {'E', fieldWeekday, dtNarrow, 5}, {'E', fieldWeekday, dtShorter, 6},
	{'c', fieldWeekday, dtNumeric + 2*dtDelta, 1}, {'c', fieldWeekday, dtShort - 2*dtDelta, 3}, {'c', fieldWeekday, dtLong - 2*dtDelta, 4}, {'c', fieldWeekday, dtNarrow - 2*dtDelta, 5}, {'c', fieldWeekday, dtShorter - 2*dtDelta, 6},
	{'e', fieldWeekday, dtNumeric + dtDelta, 1}, {'e', fieldWeekday, dtShort - dtDelta, 3}, {'e', fieldWeekday, dtLong - dtDelta, 4}, {'e', fieldWeekday, dtNarrow - dtDelta, 5}, {'e', fieldWeekday, dtShorter - dtDelta, 6},
	{'d', fieldDay, dtNumeric, 1},
	{'g', fieldDay, dtNumeric + dtDelta, 1},
	{'D', fieldDayOfYear, dtNumeric, 1},
	{'F', fieldDayOfWeekInMonth, dtNumeric, 1},
	{'a', fieldDayPeriod, dtShort, 1}, {'a', fieldDayPeriod, dtLong, 4}, {'a', fieldDayPeriod, dtNarrow, 5},
	{'b', fieldDayPeriod, dtShort - dtDelta, 1}, {'b', fieldDayPeriod, dtLong - dtDelta, 4}, {'b', fieldDayPeriod, dtNarrow - dtDelta, 5},
	{'B', fieldDayPeriod, dtShort - 3*dtDelta, 1}, {'B', fieldDayPeriod, dtLong - 3*dtDelta, 4}, {'B', fieldDayPeriod, dtNarrow - 3*dtDelta, 5},
	{'H', fieldHour, dtNumeric + 10*dtDelta, 1},
	{'k', fieldHour, dtNumeric + 11*dtDelta, 1},
	{'h', fieldHour, dtNumeric, 1},
	{'K', fieldHour, dtNumeric + dtDelta, 1},
	{'J', fieldHour, dtNumeric + 5*dtDelta, 1},
	{'j', fieldHour, dtNumeric + 6*dtDelta, 1},
	{'C', fieldHour, dtNumeric + 7*dtDelta, 1},
	{'m', fieldMinute, dtNumeric, 1},
	{'s', fieldSecond, dtNumeric, 1},
	{'A', fieldSecond, dtNumeric + dtDelta, 1},
	{'S', fieldFractionalSecond, dtNumeric, 1},
	{'v', fieldZone, dtShort - 2*dtDelta, 1}, {'v', fieldZone, dtLong - 2*dtDelta, 4},
	{'z', fieldZone, dtShort, 1}, {'z', fieldZone, dtLong, 4},
	{'Z', fieldZone, dtNarrow - dtDelta, 1}, {'Z', fieldZone, dtLong - dtDelta, 4}, {'Z', fieldZone, dtShort - dtDelta, 5},
	{'O', fieldZone, dtShort - dtDelta, 1}, {'O', fieldZone, dtLong - dtDelta, 4},
	{'V', fieldZone, dtShort - dtDelta, 1}, {'V', fieldZone, dtLong - dtDelta, 2}, {'V', fieldZone, dtLong - 1 - dtDelta, 3}, {'V', fieldZone, dtLong - 2 - dtDelta, 4},
	{'X', fieldZone, dtNarrow - dtDelta, 1}, {'X', fieldZone, dtShort - dtDelta, 2}, {'X', fieldZone, dtLong - dtDelta, 4},
	{'x', fieldZone, dtNarrow - dtDelta, 1}, {'x', fieldZone, dtShort - dtDelta, 2}, {'x', fieldZone, dtLong - dtDelta, 4},
}

// canonicalIndex returns the row of dtTypes of a field of n letters c,
// or -1.
func canonicalIndex(c byte, n int) int {
	for i := 0; i < len(dtTypes); i++ {
		if dtTypes[i].char != c {
			continue
		}
		if i+1 < len(dtTypes) && dtTypes[i+1].char == c && int(dtTypes[i+1].minLen) <= n {
			continue
		}
		return i
	}
	return -1
}

// patItem is a field of a pattern (a letter repeated) or literal text.
type patItem struct {
	char byte   // the letter of a field, 0 for literal text
	n    int    // the field's length
	text string // the literal text, quotes removed
	raw  string // the item as the pattern writes it
}

// parsePattern splits a pattern of CLDR's dates into fields and literal
// text. ASCII letters are fields; text in apostrophes is literal, and two
// apostrophes are one.
func parsePattern(p string) []patItem {
	var items []patItem
	var lit, raw strings.Builder
	flush := func() {
		if raw.Len() > 0 {
			items = append(items, patItem{text: lit.String(), raw: raw.String()})
			lit.Reset()
			raw.Reset()
		}
	}
	for i := 0; i < len(p); {
		c := p[i]
		switch {
		case c == '\'':
			if i+1 < len(p) && p[i+1] == '\'' {
				lit.WriteByte('\'')
				raw.WriteString("''")
				i += 2
				continue
			}
			j := i + 1
			raw.WriteByte('\'')
			for j < len(p) {
				if p[j] == '\'' {
					if j+1 < len(p) && p[j+1] == '\'' {
						lit.WriteByte('\'')
						raw.WriteString("''")
						j += 2
						continue
					}
					break
				}
				lit.WriteByte(p[j])
				raw.WriteByte(p[j])
				j++
			}
			raw.WriteByte('\'')
			i = j + 1
		case c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z':
			flush()
			j := i
			for j < len(p) && p[j] == c {
				j++
			}
			items = append(items, patItem{char: c, n: j - i, raw: p[i:j]})
			i = j
		default:
			lit.WriteByte(c)
			raw.WriteByte(c)
			i++
		}
	}
	flush()
	return items
}

// skeleton is ICU's PtnSkeleton: a skeleton's field per type, as written
// (original) and at its row's least length (base), and their types.
type skeleton struct {
	typ            [fieldCount]int
	original, base [fieldCount]string
	addedDayPeriod bool
}

// fieldMask returns the fields the skeleton has.
func (s *skeleton) fieldMask() int {
	m := 0
	for i, t := range s.typ {
		if t != 0 {
			m |= 1 << i
		}
	}
	return m
}

func (s *skeleton) baseString() string {
	var b strings.Builder
	for _, f := range s.base {
		b.WriteString(f)
	}
	return b.String()
}

func (s *skeleton) originalString() string {
	var b strings.Builder
	for _, f := range s.original {
		b.WriteString(f)
	}
	return b.String()
}

// newSkeleton reads a skeleton, or a pattern as one, as ICU's
// DateTimeMatcher.set does.
func newSkeleton(p string) *skeleton {
	s := &skeleton{}
	for _, it := range parsePattern(p) {
		if it.char == 0 {
			continue
		}
		ci := canonicalIndex(it.char, it.n)
		if ci < 0 {
			continue
		}
		row := dtTypes[ci]
		s.original[row.field] = it.raw
		s.base[row.field] = strings.Repeat(string(row.char), int(row.minLen))
		t := int(row.typ)
		if t > 0 {
			t += it.n
		}
		s.typ[row.field] = t
	}
	// A 12-hour hour needs a day period; a 24-hour one has none.
	if h := s.original[fieldHour]; h != "" {
		if h[0] == 'h' || h[0] == 'K' {
			if s.original[fieldDayPeriod] == "" {
				s.original[fieldDayPeriod] = "a"
				s.base[fieldDayPeriod] = "a"
				s.typ[fieldDayPeriod] = dtShort
				s.addedDayPeriod = true
			}
		} else if s.original[fieldDayPeriod] != "" {
			s.original[fieldDayPeriod], s.base[fieldDayPeriod], s.typ[fieldDayPeriod] = "", "", 0
		}
	}
	return s
}

// distance is ICU's DateTimeMatcher.getDistance: how far a pattern's
// skeleton is from the one asked for, of the fields of mask.
func (s *skeleton) distance(other *skeleton, mask int) (dist, missing int) {
	for i := 0; i < fieldCount; i++ {
		my := 0
		if mask&(1<<i) != 0 {
			my = s.typ[i]
		}
		o := other.typ[i]
		switch {
		case my == o:
		case my == 0:
			dist += extraField
		case o == 0:
			dist += missingField
			missing |= 1 << i
		default:
			d := my - o
			if d < 0 {
				d = -d
			}
			dist += d
		}
	}
	return dist, missing
}

// dtpEntry is a pattern of the generator, with its skeleton and whether
// CLDR gave the skeleton (availableFormats) or it was read from the
// pattern.
type dtpEntry struct {
	pattern   string
	skel      *skeleton
	base      string
	specified bool
}

// patternGenerator holds a locale's patterns in ICU's order: by the first
// letter of their base skeleton, A to Z then a to z, then as they came.
type patternGenerator struct {
	buckets     [52][]*dtpEntry
	defaultHour byte
	dateTime    [4]string // the glue of a date and a time by style
}

func bucketOf(base string) int {
	if base == "" {
		return -1
	}
	c := base[0]
	switch {
	case c >= 'A' && c <= 'Z':
		return int(c - 'A')
	case c >= 'a' && c <= 'z':
		return 26 + int(c-'a')
	}
	return -1
}

// add adds a pattern as ICU's addPatternWithSkeleton and PatternMap.add
// do: a pattern replaces the one of the same base skeleton and types,
// whose skeleton it keeps, unless it may not.
func (g *patternGenerator) add(pattern, skel string, override bool) {
	var s *skeleton
	if skel != "" {
		s = newSkeleton(skel)
	} else {
		s = newSkeleton(pattern)
	}
	base := s.baseString()
	b := bucketOf(base)
	if b < 0 {
		return
	}
	specified := skel != ""
	for _, e := range g.buckets[b] {
		if e.base == base {
			if (!e.specified || specified && !override) && !override {
				return
			}
			break
		}
	}
	orig := s.originalString()
	for _, e := range g.buckets[b] {
		if e.skel.originalString() == orig {
			if !override || specified && e.specified {
				return
			}
			break
		}
	}
	for _, e := range g.buckets[b] {
		if e.base == base && e.skel.typ == s.typ {
			e.pattern, e.specified = pattern, specified
			return
		}
	}
	g.buckets[b] = append(g.buckets[b], &dtpEntry{pattern: pattern, skel: s, base: base, specified: specified})
}

// generator returns the locale's pattern generator.
func (d *localeData) generator() *patternGenerator {
	d.dtpgOnce.Do(func() {
		g := &patternGenerator{}
		for _, c := range "GyQMwWEDFdaHmsSv" {
			g.add(string(c), "", false)
		}
		for i := 0; i < 4; i++ {
			g.add(d.v[slotDateFormats+i], "", false)
			g.add(d.v[slotTimeFormats+i], "", false)
		}
		loadDates()
		for i, skel := range tables.skeletons {
			if p := d.v[slotSkeletons+i]; p != "" {
				g.add(p, skel, true)
			}
		}
		for i := range g.dateTime {
			g.dateTime[i] = d.v[slotAtTimeFormats+i]
			if g.dateTime[i] == "" {
				g.dateTime[i] = d.v[slotDateTimeFormats+i]
			}
		}
		d.dtpg = g
	})
	return d.dtpg
}

// bestRaw is ICU's getBestRaw: the entry nearest the skeleton in the
// fields of mask.
func (g *patternGenerator) bestRaw(s *skeleton, mask int) (best *dtpEntry, missing int) {
	best, _, missing = g.bestRawDist(s, mask)
	return best, missing
}

func (g *patternGenerator) bestRawDist(s *skeleton, mask int) (best *dtpEntry, bestDist, missing int) {
	bestDist = 1<<31 - 1
	bestMissing := -1
	for _, list := range g.buckets {
		for _, e := range list {
			d, m := s.distance(e.skel, mask)
			if d < bestDist || d == bestDist && bestMissing < m {
				best, bestDist, bestMissing, missing = e, d, m, m
				if d == 0 {
					return best, 0, missing
				}
			}
		}
	}
	return best, bestDist, missing
}

// bestPattern is ICU's getBestPattern with UDATPG_MATCH_HOUR_FIELD_LENGTH,
// as Intl.DateTimeFormat asks it.
func (g *patternGenerator) bestPattern(skel string) string {
	skel = g.mapHourMetacharacters(skel)
	s := newSkeleton(skel)
	e, dist, missing := g.bestRawDist(s, -1)
	if e == nil {
		return ""
	}
	if missing == 0 && dist < extraField {
		return g.adjustFieldTypes(e.pattern, e, s)
	}
	needed := s.fieldMask()
	date := g.bestAppending(s, needed&dateMask)
	tm := g.bestAppending(s, needed&timeMask)
	switch {
	case date == "":
		return tm
	case tm == "":
		return date
	}
	style := 3
	switch len(s.base[fieldMonth]) {
	case 4:
		style = 1
		if s.base[fieldWeekday] != "" {
			style = 0
		}
	case 3:
		style = 2
	}
	return strings.NewReplacer("{1}", date, "{0}", tm).Replace(g.dateTime[style])
}

// bestAppending is ICU's getBestAppending: the best pattern of the fields
// of mask, and the patterns of the fields it lacks after it.
func (g *patternGenerator) bestAppending(s *skeleton, mask int) string {
	if mask == 0 {
		return ""
	}
	e, missing := g.bestRaw(s, mask)
	if e == nil {
		return ""
	}
	result := g.adjustFieldTypes(e.pattern, e, s)
	last := 0
	for missing != 0 && missing != last {
		last = missing
		start := missing
		e2, m2 := g.bestRaw(s, missing)
		if e2 == nil {
			break
		}
		missing = m2
		if start&^missing == 0 {
			break
		}
		// CLDR's appendItems put most fields after the pattern with a
		// space or in parentheses with their names; a space it is.
		result += " " + g.adjustFieldTypes(e2.pattern, e2, s)
	}
	return result
}

// mapHourMetacharacters replaces j with the locale's hour letter, as
// ICU's mapSkeletonMetacharacters does, with the day period a 12-hour
// hour needs.
func (g *patternGenerator) mapHourMetacharacters(skel string) string {
	if !strings.ContainsAny(skel, "jJC") {
		return skel
	}
	var b strings.Builder
	for i := 0; i < len(skel); i++ {
		c := skel[i]
		switch c {
		case 'j', 'C':
			extra := 0
			for i+1 < len(skel) && skel[i+1] == c {
				extra++
				i++
			}
			hourLen := 1 + extra&1
			periodLen := 1
			if extra >= 2 {
				periodLen = 3 + extra>>1
			}
			hour := g.defaultHour
			if c == 'C' {
				hour = 'h'
				if g.defaultHour == 'H' || g.defaultHour == 'k' {
					hour = g.defaultHour
				}
			}
			if hour == 'H' || hour == 'k' {
				periodLen = 0
			}
			b.WriteString(strings.Repeat("a", periodLen))
			b.WriteString(strings.Repeat(string(hour), hourLen))
		case 'J':
			b.WriteByte('H')
		default:
			b.WriteByte(c)
		}
	}
	return b.String()
}

// adjustFieldTypes is ICU's adjustFieldTypes: it gives the fields of a
// pattern the widths and letters the skeleton asks for.
func (g *patternGenerator) adjustFieldTypes(pattern string, e *dtpEntry, req *skeleton) string {
	var b strings.Builder
	for _, it := range parsePattern(pattern) {
		if it.char == 0 {
			b.WriteString(it.raw)
			continue
		}
		ci := canonicalIndex(it.char, it.n)
		if ci < 0 {
			b.WriteString(it.raw)
			continue
		}
		row := dtTypes[ci]
		field := int(row.field)
		if req.typ[field] == 0 {
			b.WriteString(it.raw)
			continue
		}
		reqField := req.original[field]
		reqChar := reqField[0]
		reqLen := len(reqField)
		if reqChar == 'E' && reqLen < 3 {
			reqLen = 3
		}
		adjLen := reqLen
		switch {
		case field == fieldMinute || field == fieldSecond:
			adjLen = it.n
		case e.specified && reqChar != 'c' && reqChar != 'e':
			skelLen := len(e.skel.original[field])
			patNumeric := row.typ > 0
			skelNumeric := e.skel.typ[field] > 0
			if skelLen == reqLen || patNumeric != skelNumeric {
				adjLen = it.n
			}
		}
		c := reqChar
		if field == fieldHour || field == fieldMonth || field == fieldWeekday || field == fieldYear && reqChar != 'Y' {
			c = it.char
		}
		if field == fieldHour && g.defaultHour != 0 {
			switch {
			case reqChar == g.defaultHour:
				c = g.defaultHour
			case reqChar == 'h' && g.defaultHour == 'K':
				c = 'K'
			case reqChar == 'H' && g.defaultHour == 'k':
				c = 'k'
			case reqChar == 'k' && g.defaultHour == 'H':
				c = 'H'
			case reqChar == 'K' && g.defaultHour == 'h':
				c = 'h'
			}
		}
		b.WriteString(strings.Repeat(string(c), adjLen))
	}
	return b.String()
}

// patternSkeleton is ICU's staticGetSkeleton: the skeleton of a pattern,
// its fields in ICU's order, without the day period it added.
func patternSkeleton(pattern string) string {
	s := newSkeleton(pattern)
	out := s.originalString()
	if s.addedDayPeriod {
		out = strings.Replace(out, "a", "", 1)
	}
	return out
}

// replaceHours replaces the hour letters of a pattern, outside its quoted
// text, with hour, as V8 does for an hour cycle Intl.DateTimeFormat is
// given.
func replaceHours(pattern string, hour byte) string {
	if hour == 0 {
		return pattern
	}
	b := []byte(pattern)
	quoted := false
	for i, c := range b {
		switch {
		case c == '\'':
			quoted = !quoted
		case !quoted && (c == 'h' || c == 'H' || c == 'k' || c == 'K'):
			b[i] = hour
		}
	}
	return string(b)
}

// patternHour returns the first hour letter of a pattern, or 0.
func patternHour(pattern string) byte {
	for _, it := range parsePattern(pattern) {
		switch it.char {
		case 'h', 'H', 'k', 'K':
			return it.char
		}
	}
	return 0
}
