package i18n

import (
	"compress/flate"
	"io"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/text/language"
)

// The formatting data of the locales of CLDR's modern coverage, which
// format_tables.go holds as the slots in which each locale differs from
// its parent (see i18n/internal/gen), in two halves: the numbers, which
// every app that formats a message uses, and the dates, with the names
// of languages, which an app links only if it formats them. A locale's
// data is put together the first time it formats something, its dates
// the first time it formats a date, and kept.

// localeData is the formatting data of a locale.
type localeData struct {
	index int    // in tables.locales
	tag   string // the locale of CLDR, as "en-GB"
	// v holds the slots of the numbers, and those of the dates once
	// withDates filled them.
	v [slotCount]string

	numOnce sync.Once
	num     [2]*numberSystem // the default numbering system, and latn

	datesOnce sync.Once

	dtpgOnce sync.Once
	dtpg     *patternGenerator
}

// tables are the tables of format_tables.go, inflated and split on first
// use: the locales and the numbers, then the dates.
var tables struct {
	once                           sync.Once
	locales, hourKeys, weekRegions []string
	parents                        string
	numbers                        pool

	datesOnce                           sync.Once
	dates                               pool
	autonymTags, skeletons, periodLangs []string
	autonyms                            string
}

// pool is the entries of a half of the data, and its strings.
type pool struct {
	entries, strings string
	starts           []uint32 // where each string starts
}

// unpack inflates a blob of packed tables and returns its tables.
func unpack(packed string, lens ...int) []string {
	raw, err := io.ReadAll(flate.NewReader(strings.NewReader(packed)))
	if err != nil {
		panic("i18n: the formatting data does not inflate: " + err.Error())
	}
	all := string(raw)
	out := make([]string, len(lens))
	for i, n := range lens {
		out[i], all = all[:n], all[n:]
	}
	return out
}

// newPool returns the pool of entries, the lengths of count strings and
// the strings.
func newPool(entries, lens, strs string, count int) pool {
	p := pool{entries: entries, strings: strs, starts: make([]uint32, count+1)}
	for i := 0; i < count; i++ {
		p.starts[i+1] = p.starts[i] + uint32(lens[i])
	}
	return p
}

// at returns string i of the pool.
func (p *pool) at(i uint32) string { return p.strings[p.starts[i]:p.starts[i+1]] }

// loadTables loads the locales, their numbers and the week and time data.
func loadTables() {
	tables.once.Do(func() {
		t := unpack(fmtPackedNumbers, packedLocalesLen, packedParentsLen, packedNumberEntriesLen, packedNumberLensLen, packedNumberPoolLen)
		tables.locales = append([]string{""}, strings.Fields(t[0])...)
		tables.parents = t[1]
		tables.numbers = newPool(t[2], t[3], t[4], fmtNumberStringCount)
		tables.hourKeys = strings.Fields(fmtHourKeys)
		tables.weekRegions = strings.Fields(fmtWeekRegions)
	})
}

// loadDates loads the tables of the dates and of the names of languages,
// and those of loadTables.
func loadDates() {
	loadTables()
	tables.datesOnce.Do(func() {
		t := unpack(fmtPackedDates, packedDateEntriesLen, packedDateLensLen, packedDatePoolLen, packedAutonymTagsLen, packedAutonymsLen)
		tables.dates = newPool(t[0], t[1], t[2], fmtDateStringCount)
		tables.autonymTags = strings.Fields(t[3])
		tables.autonyms = t[4]
		tables.skeletons = strings.Fields(fmtSkeletons)
		tables.periodLangs = strings.Fields(fmtPeriodLangs)
	})
}

// uint16At returns the big-endian uint16 at index i of a table of them.
func uint16At(table string, i int) int {
	return int(table[2*i])<<8 | int(table[2*i+1])
}

var localeDatas [fmtLocaleCount]atomic.Pointer[localeData]

// dataAt returns the data of locale i, with its numbers.
func dataAt(i int) *localeData {
	if d := localeDatas[i].Load(); d != nil {
		return d
	}
	loadTables()
	d := &localeData{index: i, tag: tables.locales[i]}
	d.fill(&tables.numbers, fmtNumberStarts[:])
	if !localeDatas[i].CompareAndSwap(nil, d) {
		return localeDatas[i].Load()
	}
	return d
}

// withDates returns d with the slots of its dates, filled the first time.
func (d *localeData) withDates() *localeData {
	d.datesOnce.Do(func() {
		loadDates()
		d.fill(&tables.dates, fmtDateStarts[:])
	})
	return d
}

// fill sets the slots of a half of the data, p, whose entries of each
// locale start at starts: the locale's entries, then its parents', the
// root's last; the nearest wins.
func (d *localeData) fill(p *pool, starts []uint16) {
	var set [slotCount]bool
	for j := d.index; ; j = uint16At(tables.parents, j) {
		for e := int(starts[j]); e < int(starts[j+1]); e++ {
			v := uint32(p.entries[3*e])<<16 | uint32(p.entries[3*e+1])<<8 | uint32(p.entries[3*e+2])
			if s := v >> 15; !set[s] {
				set[s] = true
				d.v[s] = p.at(v & (1<<15 - 1))
			}
		}
		if j == 0 {
			break
		}
	}
}

// localeKey is how a locale formats: the locale of CLDR whose data it
// uses, and the preferences of its Unicode extension ("-u-hc-h23").
type localeKey struct {
	data      int          // index in tables.locales
	hourCycle byte         // 'h', 'H', 'K', 'k', or 0 for the locale's
	firstDay  time.Weekday // or -1 for the region's
	latn      bool         // -u-nu-latn: Latin digits
	region    string       // the region whose week and time data apply
	lang      string       // the language, for the rules of day periods
}

var localeKeys sync.Map // string -> *localeKey

// keyOf returns how locale formats: the data of the locale of CLDR that
// suits it best (en-GB for en-GB, en-001 for en-AU... down to the root of
// its language), else of English.
func keyOf(locale string) *localeKey {
	if k, ok := localeKeys.Load(locale); ok {
		return k.(*localeKey)
	}
	k := makeKey(locale)
	if v, loaded := localeKeys.LoadOrStore(locale, k); loaded {
		return v.(*localeKey)
	}
	return k
}

func makeKey(locale string) *localeKey {
	k := &localeKey{firstDay: -1}
	t, ok := parse(locale)
	if ok {
		if ext, ok := t.Extension('u'); ok {
			readUnicodeExtension(k, ext.String())
		}
		if IsPseudo(t.String()) {
			t = language.English
		}
	}
	tag := stripExtensions(t.String())
	if !ok || tag == "und" {
		tag = "en"
	}
	base, _ := t.Base()
	k.lang = base.String()
	// The region: the tag's, else the language's likely one.
	if r, conf := t.Region(); conf != language.No {
		k.region = r.String()
	}
	// The tag with the script it is likely written in where it names
	// none and that differs from its language's: zh-TW is zh-Hant-TW,
	// whose data is zh-Hant's, not zh's.
	candidates := []string{tag}
	if s, conf := t.Script(); conf != language.Exact && conf != language.No {
		if ls, _ := language.Make(k.lang).Script(); ls != s {
			if r, rc := t.Region(); rc == language.Exact {
				if c, err := language.Compose(base, s, r); err == nil {
					candidates = []string{c.String(), tag}
				}
			} else if c, err := language.Compose(base, s); err == nil {
				candidates = []string{c.String(), tag}
			}
		}
	}
	k.data = -1
	loadTables()
	for _, c := range candidates {
		for _, p := range chain(c) {
			if i, ok := search(tables.locales, p); ok && p != "" {
				k.data = i
				break
			}
		}
		if k.data >= 0 {
			break
		}
	}
	if k.data < 0 {
		k.data, _ = search(tables.locales, "en")
		k.lang = "en"
	}
	return k
}

// readUnicodeExtension reads the keys of a Unicode extension MyGo
// formats with: hc (the hour cycle), fw (the first day of the week) and
// nu-latn (Latin digits).
func readUnicodeExtension(k *localeKey, ext string) {
	parts := strings.Split(ext, "-")
	for i := 1; i+1 < len(parts); i++ {
		switch parts[i] {
		case "hc":
			switch parts[i+1] {
			case "h12":
				k.hourCycle = 'h'
			case "h23":
				k.hourCycle = 'H'
			case "h11":
				k.hourCycle = 'K'
			case "h24":
				k.hourCycle = 'k'
			}
		case "fw":
			for d, name := range [...]string{"sun", "mon", "tue", "wed", "thu", "fri", "sat"} {
				if parts[i+1] == name {
					k.firstDay = time.Weekday(d)
				}
			}
		case "nu":
			k.latn = parts[i+1] == "latn"
		}
	}
}

func (k *localeKey) dat() *localeData { return dataAt(k.data) }

// dates returns the data of the locale, with its dates.
func (k *localeKey) dates() *localeData { return k.dat().withDates() }

// weekData returns the first day of the week and the minimal days of the
// first week of the year in a region, by CLDR's week data.
func weekData(region string) (time.Weekday, int) {
	loadTables()
	i, ok := search(tables.weekRegions, region)
	if !ok {
		i, _ = search(tables.weekRegions, "001")
	}
	return time.Weekday(fmtWeek[i] & 15), int(fmtWeek[i] >> 4)
}

// preferredHour returns the hour symbol CLDR's time data prefers for a
// language in a region: 'h' (1 to 12), 'H' (0 to 23), 'K' (0 to 11) or
// 'k' (1 to 24).
func preferredHour(lang, region string) byte {
	loadTables()
	for _, key := range [...]string{lang + "-" + region, region, "001"} {
		if i, ok := search(tables.hourKeys, key); ok {
			return fmtHours[i]
		}
	}
	return 'H'
}

// defaultHour returns the hour symbol of the locale: its extension's,
// else its region's.
func (k *localeKey) defaultHour() byte {
	if k.hourCycle != 0 {
		return k.hourCycle
	}
	return preferredHour(k.lang, k.regionOrLikely())
}

// regionOrLikely returns the locale's region, else "001".
func (k *localeKey) regionOrLikely() string {
	if k.region != "" {
		return k.region
	}
	return "001"
}

// FirstWeekday returns the day weeks start on in locale, as calendars
// show them: that of its region by CLDR's week data, or of the region its
// language is most likely spoken in, Sunday for "en" (the United States)
// and Monday for "en-GB" and "de", or the one its extension names
// ("en-US-u-fw-mon").
func FirstWeekday(locale string) time.Weekday {
	k := keyOf(locale)
	if k.firstDay >= 0 {
		return k.firstDay
	}
	d, _ := weekData(k.regionOrLikely())
	return d
}

// HourCycle returns how locale writes times, 12 for times with AM and PM
// and 24 for times without, by CLDR's time data for its region, or the
// one its extension names ("en-US-u-hc-h23").
func HourCycle(locale string) int {
	switch keyOf(locale).defaultHour() {
	case 'h', 'K':
		return 12
	}
	return 24
}

// RegionFor returns the regional formats CLDR gives locale: the first day
// of the week and the hour cycle of its region. The system's settings,
// which mygo.App.Region reports, may differ.
func RegionFor(locale string) Region {
	return Region{Locale: Normalize(locale), FirstWeekday: FirstWeekday(locale), HourCycle: HourCycle(locale)}
}

// Width is the width of the names of months, weekdays and day periods.
type Width uint8

const (
	// Wide names are whole: "January", "Monday".
	Wide Width = iota
	// Abbreviated names are short: "Jan", "Mon".
	Abbreviated
	// Short names are shorter still, for weekdays only: "Mo"; months
	// have abbreviated names in their place.
	Short
	// Narrow names are a letter or so: "J", "M", for calendars.
	Narrow
)

// monthWidth returns the index of a width among the widths of months.
func monthWidth(w Width) int {
	switch w {
	case Abbreviated, Short:
		return 0
	case Narrow:
		return 1
	}
	return 2
}

func dayWidth(w Width) int {
	switch w {
	case Abbreviated:
		return 0
	case Narrow:
		return 1
	case Short:
		return 2
	}
	return 3
}

// Months returns the names of the months in locale, January first, in a
// width. A name that stands alone, as a calendar's heading or a list of
// months, takes standAlone: some languages write it otherwise inside a
// date ("января" of "5 января" but "январь" alone, in Russian).
func Months(locale string, w Width, standAlone bool) [12]string {
	d := keyOf(locale).dates()
	ctx := 0
	if standAlone {
		ctx = 1
	}
	base := slotMonths + (ctx*3+monthWidth(w))*12
	var out [12]string
	copy(out[:], d.v[base:base+12])
	return out
}

// Weekdays returns the names of the days of the week in locale, by
// time.Weekday (Sunday first), in a width, standing alone or inside a
// date as Months does.
func Weekdays(locale string, w Width, standAlone bool) [7]string {
	d := keyOf(locale).dates()
	ctx := 0
	if standAlone {
		ctx = 1
	}
	base := slotDays + (ctx*4+dayWidth(w))*7
	var out [7]string
	copy(out[:], d.v[base:base+7])
	return out
}

// DayPeriods returns the names of the halves of the day in locale, before
// noon and after, in a width: "AM" and "PM" in English, "上午" and "下午" in
// Chinese.
func DayPeriods(locale string, w Width) (am, pm string) {
	d := keyOf(locale).dates()
	base := slotDayPeriods + monthWidth(w)*len(periodNames)
	return d.v[base], d.v[base+1]
}

// periodNames are the day periods of slotDayPeriods, in their order.
var periodNames = [...]string{"am", "pm", "midnight", "noon", "morning1", "morning2", "afternoon1", "afternoon2", "evening1", "evening2", "night1", "night2"}

// Autonym returns the name of a language in itself, for menus of
// languages: "English", "British English", "Deutsch", "español
// latinoamericano", "简体中文", "繁體中文", "日本語", in CLDR's form, as
// Intl.DisplayNames writes it (lowercase in the languages that write
// names so: "français"). A locale of a region or script has its own name
// where CLDR has it ("zh-Hant" is "繁體中文"), else the name of the language
// with them ("português (Brasil)"). It returns "" for a language CLDR has
// no data of in its modern coverage.
func Autonym(tag string) string {
	t := stripExtensions(Normalize(tag))
	if IsPseudo(t) {
		return t
	}
	loadDates()
	if i, ok := search(tables.autonymTags, t); ok {
		return tables.dates.at(uint32(uint16At(tables.autonyms, i)))
	}
	// A locale CLDR has no name of: the name of the locale whose data it
	// formats with, zh-Hant's for zh-TW, sr-Latn-ME's for sr-ME, else of
	// its parents.
	if k := keyOf(t); k.lang == Base(t) {
		if i, ok := search(tables.autonymTags, tables.locales[k.data]); ok {
			return tables.dates.at(uint32(uint16At(tables.autonyms, i)))
		}
	}
	for _, c := range chain(t) {
		if i, ok := search(tables.autonymTags, c); ok {
			return tables.dates.at(uint32(uint16At(tables.autonyms, i)))
		}
	}
	return ""
}

// FormatLocales returns the locales of CLDR's modern coverage that this
// package has formatting data of, sorted: "af", "ar", "ar-AE"... A
// locale it lacks formats as the nearest of them, by Parent, or as
// English.
func FormatLocales() []string {
	loadTables()
	return append([]string(nil), tables.locales[1:]...)
}
