package main

import (
	"bytes"
	"compress/flate"
	"encoding/json"
	"fmt"
	"go/format"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

// The formatting data of a locale is a record of slots, strings each:
// the names of months, weekdays, day periods and eras, the patterns of
// dates and times, the symbols and patterns of numbers, and the names of
// time zones' offsets. The generated tables hold, for each locale, only
// the slots in which it differs from its parent locale (or, for the
// locales without one, from a synthetic root that holds the value most
// of them have), as strings of one pool: the locales of a language share
// almost everything.

// Months: [context][width][month], context format then stand-alone,
// width abbreviated, narrow, wide.
var (
	monthContexts = []string{"format", "stand-alone"}
	monthWidths   = []string{"abbreviated", "narrow", "wide"}
	// Days: [context][width][weekday], Sunday first.
	dayWidths = []string{"abbreviated", "narrow", "short", "wide"}
	dayKeys   = []string{"sun", "mon", "tue", "wed", "thu", "fri", "sat"}
	// Day periods: [width][period], of the format context.
	periodWidths = []string{"abbreviated", "narrow", "wide"}
	periodKeys   = []string{"am", "pm", "midnight", "noon", "morning1", "morning2", "afternoon1", "afternoon2", "evening1", "evening2", "night1", "night2"}
	// Eras: [width][era], width abbreviated, narrow, wide.
	eraTables = []string{"eraAbbr", "eraNarrow", "eraNames"}
	styles    = []string{"full", "long", "medium", "short"}
	// Zone: the localized GMT format, and the names of UTC.
	zoneKeys = []string{"gmtFormat", "gmtZeroFormat", "hourFormat", "utcLong", "utcShort"}
	// Numbers: [system][symbol], system the default one, then latn when
	// the default is another.
	numberKeys = []string{"digits", "decimal", "group", "minusSign", "plusSign", "percentSign", "exponential", "infinity", "nan", "decimalPattern", "percentPattern"}
)

// slots lays the record out.
type slots struct {
	months, days, periods, eras, dateFormats, timeFormats, dateTimeFormats, atTimeFormats, zone, numbers, minGrouping, skeletons, count int
	skeletonKeys                                                                                                                        []string
}

func newSlots(skeletons []string) slots {
	var s slots
	n := 0
	take := func(k int) int { at := n; n += k; return at }
	s.months = take(len(monthContexts) * len(monthWidths) * 12)
	s.days = take(len(monthContexts) * len(dayWidths) * 7)
	s.periods = take(len(periodWidths) * len(periodKeys))
	s.eras = take(len(eraTables) * 2)
	s.dateFormats = take(4)
	s.timeFormats = take(4)
	s.dateTimeFormats = take(4)
	s.atTimeFormats = take(4)
	s.zone = take(len(zoneKeys))
	s.numbers = take(2 * len(numberKeys))
	s.minGrouping = take(1)
	s.skeletons = take(len(skeletons))
	s.count = n
	s.skeletonKeys = skeletons
	return s
}

// skeletonRe matches the keys of availableFormats MyGo keeps: those of
// the fields it formats.
var skeletonRe = regexp.MustCompile(`^[yMLdEchHKkamsv]+$`)

// patternLettersRe matches the pattern letters MyGo formats, outside
// quotes.
var patternLetters = "GyYMLdEcaBhHKkmszv"

type gregorian struct {
	Months          map[string]map[string]map[string]string `json:"months"`
	Days            map[string]map[string]map[string]string `json:"days"`
	DayPeriods      map[string]map[string]map[string]string `json:"dayPeriods"`
	Eras            map[string]map[string]string            `json:"eras"`
	DateFormats     map[string]string                       `json:"dateFormats"`
	TimeFormats     map[string]string                       `json:"timeFormats"`
	DateTimeFormats struct {
		Full, Long, Medium, Short string
		AvailableFormats          map[string]string `json:"availableFormats"`
	} `json:"dateTimeFormats"`
	AtTime struct {
		Standard map[string]string `json:"standard"`
	} `json:"dateTimeFormats-atTime"`
}

type numbersData map[string]json.RawMessage

// locale is what the generator reads of a locale.
type localeRecord struct {
	tag    string
	values []string // by slot, "" when the locale has none
	usesB  bool     // its patterns have flexible day periods
}

func writeFormatTables(out, cache string, core map[string][]byte) error {
	var coverage struct {
		Effective map[string]string `json:"effectiveCoverageLevels"`
	}
	if err := json.Unmarshal(core["coverageLevels.json"], &coverage); err != nil {
		return fmt.Errorf("coverageLevels.json: %v", err)
	}
	var tags []string
	for tag, level := range coverage.Effective {
		if level == "modern" && tag != "und" {
			tags = append(tags, tag)
		}
	}
	slices.Sort(tags)
	wanted := map[string]bool{}
	for _, t := range tags {
		wanted[t] = true
	}
	inMain := func(name string, files ...string) bool {
		rest, ok := strings.CutPrefix(name, "main/")
		if !ok {
			return false
		}
		dir, file, _ := strings.Cut(rest, "/")
		return wanted[dir] && slices.Contains(files, file)
	}

	dates, err := readPackage(datesPackage, cache, func(n string) bool { return inMain(n, "ca-gregorian.json", "timeZoneNames.json") })
	if err != nil {
		return err
	}
	numbers, err := readPackage(numbersPackage, cache, func(n string) bool { return inMain(n, "numbers.json") })
	if err != nil {
		return err
	}
	names, err := readPackage(localeNamesPackage, cache, func(n string) bool {
		return inMain(n, "languages.json", "scripts.json", "territories.json", "variants.json", "localeDisplayNames.json")
	})
	if err != nil {
		return err
	}
	supp := func(name string, v any) error {
		if err := json.Unmarshal(core["supplemental/"+name], v); err != nil {
			return fmt.Errorf("%s: %v", name, err)
		}
		return nil
	}
	var numberingSystems struct {
		Supplemental struct {
			NumberingSystems map[string]struct {
				Digits string `json:"_digits"`
				Type   string `json:"_type"`
			} `json:"numberingSystems"`
		} `json:"supplemental"`
	}
	if err := supp("numberingSystems.json", &numberingSystems); err != nil {
		return err
	}

	// The skeletons of availableFormats any locale has.
	grego := map[string]*gregorian{}
	skelSet := map[string]bool{}
	for _, tag := range tags {
		var doc struct {
			Main map[string]struct {
				Dates struct {
					Calendars struct {
						Gregorian gregorian `json:"gregorian"`
					} `json:"calendars"`
				} `json:"dates"`
			} `json:"main"`
		}
		data, ok := dates["main/"+tag+"/ca-gregorian.json"]
		if !ok {
			return fmt.Errorf("cldr-dates-full has no gregorian calendar of %s", tag)
		}
		if err := json.Unmarshal(data, &doc); err != nil {
			return fmt.Errorf("%s ca-gregorian.json: %v", tag, err)
		}
		g := doc.Main[tag].Dates.Calendars.Gregorian
		grego[tag] = &g
		for k := range g.DateTimeFormats.AvailableFormats {
			if skeletonRe.MatchString(k) {
				skelSet[k] = true
			}
		}
	}
	skeletons := slices.Sorted(func(yield func(string) bool) {
		for k := range skelSet {
			if !yield(k) {
				return
			}
		}
	})
	sl := newSlots(skeletons)

	var records []*localeRecord
	for _, tag := range tags {
		r := &localeRecord{tag: tag, values: make([]string, sl.count)}
		g := grego[tag]
		for c, ctx := range monthContexts {
			for w, width := range monthWidths {
				for m := 1; m <= 12; m++ {
					r.values[sl.months+(c*len(monthWidths)+w)*12+m-1] = g.Months[ctx][width][strconv.Itoa(m)]
				}
			}
			for w, width := range dayWidths {
				for d, key := range dayKeys {
					r.values[sl.days+(c*len(dayWidths)+w)*7+d] = g.Days[ctx][width][key]
				}
			}
		}
		var patterns []string
		for i, s := range styles {
			r.values[sl.dateFormats+i] = g.DateFormats[s]
			r.values[sl.timeFormats+i] = g.TimeFormats[s]
			r.values[sl.atTimeFormats+i] = g.AtTime.Standard[s]
			patterns = append(patterns, g.DateFormats[s], g.TimeFormats[s])
		}
		r.values[sl.dateTimeFormats+0] = g.DateTimeFormats.Full
		r.values[sl.dateTimeFormats+1] = g.DateTimeFormats.Long
		r.values[sl.dateTimeFormats+2] = g.DateTimeFormats.Medium
		r.values[sl.dateTimeFormats+3] = g.DateTimeFormats.Short
		for i, k := range skeletons {
			p := g.DateTimeFormats.AvailableFormats[k]
			r.values[sl.skeletons+i] = p
			if p != "" {
				patterns = append(patterns, p)
			}
		}
		for _, p := range patterns {
			for _, c := range patternFields(p) {
				if !strings.ContainsRune(patternLetters, c) {
					return fmt.Errorf("%s: the pattern %q has the field %c, which MyGo does not format", tag, p, c)
				}
				if c == 'B' {
					r.usesB = true
				}
			}
		}
		for w, width := range periodWidths {
			for p, key := range periodKeys {
				if key != "am" && key != "pm" && !r.usesB {
					continue
				}
				r.values[sl.periods+w*len(periodKeys)+p] = g.DayPeriods["format"][width][key]
			}
		}
		for w, table := range eraTables {
			r.values[sl.eras+w*2] = g.Eras[table]["0"]
			r.values[sl.eras+w*2+1] = g.Eras[table]["1"]
		}

		// Time zones.
		var tz struct {
			Main map[string]struct {
				Dates struct {
					TimeZoneNames struct {
						GMTFormat     string `json:"gmtFormat"`
						GMTZeroFormat string `json:"gmtZeroFormat"`
						HourFormat    string `json:"hourFormat"`
						Zone          struct {
							Etc struct {
								UTC struct {
									Long, Short struct {
										Standard string `json:"standard"`
									}
								} `json:"UTC"`
							} `json:"Etc"`
						} `json:"zone"`
					} `json:"timeZoneNames"`
				} `json:"dates"`
			} `json:"main"`
		}
		if err := json.Unmarshal(dates["main/"+tag+"/timeZoneNames.json"], &tz); err != nil {
			return fmt.Errorf("%s timeZoneNames.json: %v", tag, err)
		}
		z := tz.Main[tag].Dates.TimeZoneNames
		for i, v := range []string{z.GMTFormat, z.GMTZeroFormat, z.HourFormat, z.Zone.Etc.UTC.Long.Standard, z.Zone.Etc.UTC.Short.Standard} {
			r.values[sl.zone+i] = v
		}

		// Numbers.
		var nd struct {
			Main map[string]struct {
				Numbers numbersData `json:"numbers"`
			} `json:"main"`
		}
		if err := json.Unmarshal(numbers["main/"+tag+"/numbers.json"], &nd); err != nil {
			return fmt.Errorf("%s numbers.json: %v", tag, err)
		}
		n := nd.Main[tag].Numbers
		var system, minGrouping string
		if err := json.Unmarshal(n["defaultNumberingSystem"], &system); err != nil {
			return fmt.Errorf("%s: defaultNumberingSystem: %v", tag, err)
		}
		if err := json.Unmarshal(n["minimumGroupingDigits"], &minGrouping); err != nil {
			return fmt.Errorf("%s: minimumGroupingDigits: %v", tag, err)
		}
		r.values[sl.minGrouping] = minGrouping
		systems := []string{system}
		if system != "latn" {
			systems = append(systems, "latn")
		}
		for i, sys := range systems {
			ns, ok := numberingSystems.Supplemental.NumberingSystems[sys]
			if !ok || ns.Type != "numeric" || len([]rune(ns.Digits)) != 10 {
				return fmt.Errorf("%s: the numbering system %s has no decimal digits", tag, sys)
			}
			var symbols map[string]string
			var decimal, percent struct {
				Standard string `json:"standard"`
			}
			for key, v := range map[string]any{"symbols": &symbols, "decimalFormats": &decimal, "percentFormats": &percent} {
				raw, ok := n[key+"-numberSystem-"+sys]
				if !ok {
					return fmt.Errorf("%s has no %s of %s", tag, key, sys)
				}
				if err := json.Unmarshal(raw, v); err != nil {
					return fmt.Errorf("%s %s of %s: %v", tag, key, sys, err)
				}
			}
			base := sl.numbers + i*len(numberKeys)
			for k, key := range numberKeys {
				var v string
				switch key {
				case "digits":
					v = ns.Digits
				case "decimalPattern":
					v = decimal.Standard
				case "percentPattern":
					v = percent.Standard
				default:
					v = symbols[key]
				}
				if v == "" {
					return fmt.Errorf("%s has no %s of %s", tag, key, sys)
				}
				r.values[base+k] = v
			}
		}
		records = append(records, r)
	}

	// The locale each inherits from: its nearest ancestor of the list.
	parents, err := parentLocales(core)
	if err != nil {
		return err
	}
	likely, err := likelySubtags(core)
	if err != nil {
		return err
	}
	index := map[string]int{}
	for i, r := range records {
		index[r.tag] = i
	}
	parentOf := make([]int, len(records)) // -1 for none
	for i, r := range records {
		parentOf[i] = -1
		for t := parentTag(r.tag, parents, likely); t != ""; t = parentTag(t, parents, likely) {
			if j, ok := index[t]; ok {
				parentOf[i] = j
				break
			}
		}
	}

	// The synthetic root: of the locales without a parent, the value most
	// have, in each slot most of them fill.
	root := make([]string, sl.count)
	var tops []int
	for i := range records {
		if parentOf[i] < 0 {
			tops = append(tops, i)
		}
	}
	for s := range root {
		counts := map[string]int{}
		for _, i := range tops {
			if v := records[i].values[s]; v != "" {
				counts[v]++
			}
		}
		best, bestN := "", 0
		for v, n := range counts {
			if n > bestN || n == bestN && v < best {
				best, bestN = v, n
			}
		}
		if bestN*2 > len(tops) {
			root[s] = best
		}
	}

	// The data is split in two halves, each with its own pool of
	// strings and its own entries: the numbers, which every app that
	// formats a message links, and the dates and the names of languages,
	// which only apps that format them do.
	numberHalf := newHalf(func(s int) bool { return s >= sl.numbers && s < sl.skeletons })
	dateHalf := newHalf(func(s int) bool { return s < sl.numbers || s >= sl.skeletons })
	for _, h := range []*half{numberHalf, dateHalf} {
		h.diff(root, make([]string, sl.count))
		// A locale's entries over its parent's resolved values are its
		// own values, "" included: a slot it lacks is "" even where its
		// parent's is not.
		for i, r := range records {
			base := root
			if parentOf[i] >= 0 {
				base = records[parentOf[i]].values
			}
			h.diff(r.values, base)
		}
		if len(h.pool) >= 1<<15 || sl.count >= 1<<9 || len(h.entries) >= 1<<16 {
			return fmt.Errorf("%d strings or %d slots do not fit the entries", len(h.pool), sl.count)
		}
	}

	// Week data and hour cycles, by region.
	var week struct {
		Supplemental struct {
			WeekData struct {
				MinDays  map[string]string `json:"minDays"`
				FirstDay map[string]string `json:"firstDay"`
			} `json:"weekData"`
		} `json:"supplemental"`
	}
	if err := supp("weekData.json", &week); err != nil {
		return err
	}
	var timeData struct {
		Supplemental struct {
			TimeData map[string]struct {
				Allowed   string `json:"_allowed"`
				Preferred string `json:"_preferred"`
			} `json:"timeData"`
		} `json:"supplemental"`
	}
	if err := supp("timeData.json", &timeData); err != nil {
		return err
	}
	var dayPeriodRules struct {
		Supplemental struct {
			RuleSet map[string]map[string]map[string]string `json:"dayPeriodRuleSet"`
		} `json:"supplemental"`
	}
	if err := supp("dayPeriods.json", &dayPeriodRules); err != nil {
		return err
	}

	var b bytes.Buffer
	fmt.Fprintf(&b, "// Code generated by go generate (i18n/internal/gen) from CLDR %s. DO NOT EDIT.\n\npackage i18n\n\n", cldrVersion)
	b.WriteString("// The slots of the formatting data of a locale (see i18n/internal/gen).\nconst (\n")
	for _, c := range []struct {
		name string
		v    int
	}{
		{"slotMonths", sl.months}, {"slotDays", sl.days}, {"slotDayPeriods", sl.periods}, {"slotEras", sl.eras},
		{"slotDateFormats", sl.dateFormats}, {"slotTimeFormats", sl.timeFormats},
		{"slotDateTimeFormats", sl.dateTimeFormats}, {"slotAtTimeFormats", sl.atTimeFormats},
		{"slotZone", sl.zone}, {"slotNumbers", sl.numbers}, {"slotMinGrouping", sl.minGrouping},
		{"slotSkeletons", sl.skeletons}, {"slotCount", sl.count},
	} {
		fmt.Fprintf(&b, "\t%s = %d\n", c.name, c.v)
	}
	b.WriteString(")\n\n")
	b.WriteString("// fmtSkeletons are the skeletons of the patterns of availableFormats, sorted,\n// separated by spaces.\nconst fmtSkeletons = ")
	writeLongString(&b, strings.Join(skeletons, " "))
	b.WriteString("\n\n")

	var localeTags []string
	for _, r := range records {
		localeTags = append(localeTags, r.tag)
	}
	fmt.Fprintf(&b, "// fmtLocaleCount is the number of locales with formatting data.\nconst fmtLocaleCount = %d\n\n", len(localeTags)+1)
	// The big tables are packed, compressed, in two blobs: the
	// locales and the numbers (fmtPackedNumbers), the dates and the names
	// of languages (fmtPackedDates).
	var packed []packedSection
	packed = append(packed, packedSection{"Locales", "the locales, sorted and separated by spaces, after the synthetic root \"\" (index 0)", strings.Join(localeTags, " ")})
	var pb []byte
	pb = append(pb, 0, 0)
	for i := range records {
		pb = append(pb, byte((parentOf[i]+1)>>8), byte(parentOf[i]+1))
	}
	packed = append(packed, packedSection{"Parents", "the index of the locale each inherits from, 0 for the root and for the root's, two bytes each, big-endian", string(pb)})
	numberSections, err := numberHalf.sections("Number", "numbers")
	if err != nil {
		return err
	}
	packed = append(packed, numberSections...)
	if err := writePacked(&b, "fmtPackedNumbers", "the locales and their numbers", packed); err != nil {
		return err
	}

	// Autonyms: of each locale, and of the default content locales of
	// its language (en-US, zh-Hans-CN), in itself.
	autonyms, err := makeAutonyms(tags, names, core, parents, likely)
	if err != nil {
		return err
	}
	autoTags := slices.Sorted(func(yield func(string) bool) {
		for t := range autonyms {
			if !yield(t) {
				return
			}
		}
	})
	var ab []byte
	for _, t := range autoTags {
		i := dateHalf.intern(autonyms[t])
		ab = append(ab, byte(i>>8), byte(i))
	}
	if len(dateHalf.pool) >= 1<<15 {
		return fmt.Errorf("%d strings do not fit the entries", len(dateHalf.pool))
	}
	dateSections, err := dateHalf.sections("Date", "dates")
	if err != nil {
		return err
	}
	packed = append(dateSections,
		packedSection{"AutonymTags", "the locales whose names in themselves Autonyms has, sorted and separated by spaces", strings.Join(autoTags, " ")},
		packedSection{"Autonyms", "the index of each name in the pool of the dates, two bytes each", string(ab)})
	if err := writePacked(&b, "fmtPackedDates", "the dates of the locales and the names of languages", packed); err != nil {
		return err
	}
	for _, h := range []struct {
		name string
		h    *half
	}{{"Number", numberHalf}, {"Date", dateHalf}} {
		fmt.Fprintf(&b, "// fmt%sStarts are where the entries of each locale start in those of the\n// %ss, counted in entries.\nvar fmt%sStarts = [...]uint16{\n", h.name, strings.ToLower(h.name), h.name)
		writeInts(&b, h.h.starts)
		b.WriteString("}\n\n")
		fmt.Fprintf(&b, "// fmt%sStringCount is the number of strings of the pool of the %ss.\nconst fmt%sStringCount = %d\n\n", h.name, strings.ToLower(h.name), h.name, len(h.h.pool))
	}

	// Week data.
	regionSet := map[string]bool{}
	for r := range week.Supplemental.WeekData.FirstDay {
		if !strings.Contains(r, "-") {
			regionSet[r] = true
		}
	}
	for r := range week.Supplemental.WeekData.MinDays {
		regionSet[r] = true
	}
	regions := slices.Sorted(func(yield func(string) bool) {
		for r := range regionSet {
			if !yield(r) {
				return
			}
		}
	})
	var weekVals []int
	for _, r := range regions {
		fd, ok := week.Supplemental.WeekData.FirstDay[r]
		if !ok {
			fd = week.Supplemental.WeekData.FirstDay["001"]
		}
		md, ok := week.Supplemental.WeekData.MinDays[r]
		if !ok {
			md = week.Supplemental.WeekData.MinDays["001"]
		}
		d := slices.Index(dayKeys, fd)
		m, err := strconv.Atoi(md)
		if d < 0 || err != nil || m < 1 || m > 7 {
			return fmt.Errorf("weekData of %s: %s, %s", r, fd, md)
		}
		weekVals = append(weekVals, d|m<<4)
	}
	b.WriteString("// fmtWeekRegions are the regions of CLDR's week data, sorted and\n// separated by spaces, and fmtWeek their first day of the week\n// (time.Weekday) in the low 4 bits and the minimal days of the first\n// week in the high ones, a byte each.\nconst fmtWeekRegions = ")
	writeLongString(&b, strings.Join(regions, " "))
	wb := make([]byte, len(weekVals))
	for i, v := range weekVals {
		wb[i] = byte(v)
	}
	b.WriteString("\n\nconst fmtWeek = ")
	writeLongString(&b, string(wb))
	b.WriteString("\n\n")

	// Hour cycles.
	hourKeys := slices.Sorted(func(yield func(string) bool) {
		for r := range timeData.Supplemental.TimeData {
			if !yield(r) {
				return
			}
		}
	})
	var prefs strings.Builder
	for _, k := range hourKeys {
		p := timeData.Supplemental.TimeData[k].Preferred
		if len(p) != 1 || !strings.Contains("hHKk", p) {
			return fmt.Errorf("timeData of %s prefers %q", k, p)
		}
		prefs.WriteString(p)
	}
	b.WriteString("// fmtHourKeys are the regions and locales of CLDR's time data, sorted\n// and separated by spaces, and fmtHours the hour symbol each prefers.\nconst fmtHourKeys = ")
	writeLongString(&b, strings.Join(hourKeys, " "))
	fmt.Fprintf(&b, "\n\nconst fmtHours = %q\n\n", prefs.String())

	// Day period rules of the languages whose patterns have B.
	ruleLangs := map[string]bool{}
	for _, r := range records {
		if r.usesB {
			lang, _, _ := strings.Cut(r.tag, "-")
			if _, ok := dayPeriodRules.Supplemental.RuleSet[r.tag]; ok {
				lang = r.tag
			}
			if _, ok := dayPeriodRules.Supplemental.RuleSet[lang]; !ok {
				return fmt.Errorf("%s has flexible day periods but CLDR no rules for them", r.tag)
			}
			ruleLangs[lang] = true
		}
	}
	var ruleTags, ruleTexts []string
	for _, lang := range slices.Sorted(func(yield func(string) bool) {
		for l := range ruleLangs {
			if !yield(l) {
				return
			}
		}
	}) {
		var parts []string
		set := dayPeriodRules.Supplemental.RuleSet[lang]
		for _, name := range slices.Sorted(func(yield func(string) bool) {
			for n := range set {
				if !yield(n) {
					return
				}
			}
		}) {
			p := slices.Index(periodKeys, name)
			if p < 0 {
				return fmt.Errorf("day periods of %s: unknown period %s", lang, name)
			}
			rule := set[name]
			if at, ok := rule["_at"]; ok {
				parts = append(parts, fmt.Sprintf("%d@%d", p, minutes(at)))
			} else {
				parts = append(parts, fmt.Sprintf("%d:%d-%d", p, minutes(rule["_from"]), minutes(rule["_before"])))
			}
		}
		ruleTags = append(ruleTags, lang)
		ruleTexts = append(ruleTexts, strings.Join(parts, " "))
	}
	b.WriteString("// fmtPeriodLangs are the languages whose patterns have flexible day\n// periods (B), sorted and separated by spaces, and fmtPeriodRules their\n// rules: period@minute, or period:from-before in minutes, the period an\n// index of slotDayPeriods.\nconst fmtPeriodLangs = ")
	writeLongString(&b, strings.Join(ruleTags, " "))
	b.WriteString("\n\nvar fmtPeriodRules = [...]string{\n")
	for _, t := range ruleTexts {
		fmt.Fprintf(&b, "\t%q,\n", t)
	}
	b.WriteString("}\n")

	// The locales, for scripts/format-golden.ts.
	locs, _ := json.Marshal(tags)
	autos, _ := json.Marshal(autoTags)
	list := fmt.Sprintf("{\n \"locales\": %s,\n \"autonyms\": %s\n}\n", locs, autos)
	if err := os.WriteFile(filepath.Join(out, "testdata", "format_locales.json"), []byte(list), 0o644); err != nil {
		return err
	}

	src, err := format.Source(b.Bytes())
	if err != nil {
		return fmt.Errorf("formatting format_tables.go: %v", err)
	}
	return os.WriteFile(filepath.Join(out, "format_tables.go"), src, 0o644)
}

// half is the entries of the slots of one half of the data, over a
// pool of strings of its own.
type half struct {
	has     func(slot int) bool
	pool    []string // in the order of first use; "" is 0
	index   map[string]int
	entries []uint32 // [0] the root, then the locales in order
	starts  []int
}

func newHalf(has func(slot int) bool) *half {
	return &half{has: has, pool: []string{""}, index: map[string]int{"": 0}, starts: []int{0}}
}

func (h *half) intern(s string) int {
	if i, ok := h.index[s]; ok {
		return i
	}
	h.pool = append(h.pool, s)
	h.index[s] = len(h.pool) - 1
	return len(h.pool) - 1
}

// diff adds the entries of a locale: its slots of the half whose values
// differ from base's.
func (h *half) diff(values, base []string) {
	for s := range values {
		if h.has(s) && values[s] != base[s] {
			h.entries = append(h.entries, uint32(s)<<20|uint32(h.intern(values[s])))
		}
	}
	h.starts = append(h.starts, len(h.entries))
}

// sections returns the entries and the pool of the half as the sections
// <name>Entries, <name>Lens and <name>Pool.
func (h *half) sections(name, what string) ([]packedSection, error) {
	var eb []byte
	for _, e := range h.entries {
		slot, str := e>>20, e&(1<<20-1)
		eb = append(eb, byte((slot<<15|str)>>16), byte((slot<<15|str)>>8), byte(slot<<15|str))
	}
	var all strings.Builder
	var lens []byte
	for _, s := range h.pool {
		if len(s) > 255 {
			return nil, fmt.Errorf("the string %q is longer than 255 bytes", s)
		}
		all.WriteString(s)
		lens = append(lens, byte(len(s)))
	}
	return []packedSection{
		{name + "Entries", "the slots of the " + what + " in which a locale differs from its parent, three bytes each, big-endian: the slot in the top 9 bits, the index of its string in the 15 others", string(eb)},
		{name + "Lens", "the lengths of the strings of the pool of the " + what + ", a byte each", string(lens)},
		{name + "Pool", "the strings of the " + what, all.String()},
	}, nil
}

// packedSection is a table of a blob of packed tables.
type packedSection struct {
	name, doc, data string
}

// writePacked writes the sections compressed with DEFLATE as the
// constant name, and the length of each.
func writePacked(b *bytes.Buffer, name, what string, sections []packedSection) error {
	var raw bytes.Buffer
	fmt.Fprintf(b, "// %s holds %s:\n// tables compressed with DEFLATE, one after the other, which package\n// i18n inflates once, as it first uses them. Their lengths:\nconst (\n", name, what)
	for _, s := range sections {
		fmt.Fprintf(b, "\tpacked%sLen = %d // %s\n", s.name, len(s.data), s.doc)
		raw.WriteString(s.data)
	}
	fmt.Fprintf(b, ")\n\nconst %s = ", name)
	var z bytes.Buffer
	w, err := flate.NewWriter(&z, flate.BestCompression)
	if err != nil {
		return err
	}
	w.Write(raw.Bytes())
	if err := w.Close(); err != nil {
		return err
	}
	writeLongString(b, z.String())
	b.WriteString("\n\n")
	return nil
}

func minutes(hhmm string) int {
	h, m, _ := strings.Cut(hhmm, ":")
	hi, _ := strconv.Atoi(h)
	mi, _ := strconv.Atoi(m)
	return hi*60 + mi
}

// patternFields returns the letters of the fields of a date pattern,
// outside its quoted literals: ASCII letters.
func patternFields(p string) []rune {
	var out []rune
	quoted := false
	for _, c := range p {
		switch {
		case c == '\'':
			quoted = !quoted
		case !quoted && (c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z'):
			out = append(out, c)
		}
	}
	return out
}

// writeLongString writes s as a sum of quoted lines.
func writeLongString(b *bytes.Buffer, s string) {
	const width = 64
	first := true
	for len(s) > 0 {
		n := 0
		for i := range s {
			if i > width {
				break
			}
			n = i
		}
		if len(s) <= width+4 {
			n = len(s)
		}
		if n == 0 {
			n = len(s)
		}
		if !first {
			b.WriteString(" +\n\t")
		}
		b.WriteString(strconv.Quote(s[:n]))
		s = s[n:]
		first = false
	}
	if first {
		b.WriteString(`""`)
	}
}

// parentLocales returns CLDR's parent locales of the main data.
func parentLocales(core map[string][]byte) (map[string]string, error) {
	var parents struct {
		Supplemental struct {
			ParentLocales struct {
				ParentLocale map[string]string `json:"parentLocale"`
			} `json:"parentLocales"`
		} `json:"supplemental"`
	}
	if err := json.Unmarshal(core["supplemental/parentLocales.json"], &parents); err != nil {
		return nil, err
	}
	return parents.Supplemental.ParentLocales.ParentLocale, nil
}

func likelySubtags(core map[string][]byte) (map[string]string, error) {
	var likely struct {
		Supplemental struct {
			LikelySubtags map[string]string `json:"likelySubtags"`
		} `json:"supplemental"`
	}
	if err := json.Unmarshal(core["supplemental/likelySubtags.json"], &likely); err != nil {
		return nil, err
	}
	return likely.Supplemental.LikelySubtags, nil
}

// parentTag returns the parent of a locale as package i18n's Parent does,
// with CLDR's own likely subtags: the root ("") for a language in a
// script other than its likely one.
func parentTag(tag string, parents, likely map[string]string) string {
	if p, ok := parents[tag]; ok {
		if p == "root" || p == "und" {
			return ""
		}
		return p
	}
	lang, rest, ok := strings.Cut(tag, "-")
	if !ok {
		return ""
	}
	if len(rest) == 4 {
		if max := likely[lang]; max != "" && strings.Split(max, "-")[1] != rest {
			return ""
		}
	}
	return tag[:strings.LastIndexByte(tag, '-')]
}

// makeAutonyms returns the names of the locales, and of the default
// content locales of their languages, in themselves, as ICU's
// LocaleDisplayNames writes them with dialect names (Intl.DisplayNames'
// default): "British English", "español latinoamericano", "中文（繁體，台灣）".
func makeAutonyms(tags []string, files map[string][]byte, core map[string][]byte, parents, likely map[string]string) (map[string]string, error) {
	var dc struct {
		DefaultContent []string `json:"defaultContent"`
	}
	if err := json.Unmarshal(core["defaultContent.json"], &dc); err != nil {
		return nil, fmt.Errorf("defaultContent.json: %v", err)
	}
	shipped := map[string]bool{}
	for _, t := range tags {
		shipped[t] = true
	}
	all := slices.Clone(tags)
	for _, t := range dc.DefaultContent {
		// The default content of a shipped locale: en-US of en,
		// zh-Hans-CN of zh-Hans, whose data is theirs.
		if p := parentTag(t, parents, likely); shipped[p] || shipped[t[:max(strings.LastIndexByte(t, '-'), 0)]] {
			all = append(all, t)
		}
	}
	type table map[string]string
	load := func(tag, file, key string) (table, error) {
		data, ok := files["main/"+tag+"/"+file]
		if !ok {
			if file == "languages.json" {
				return nil, fmt.Errorf("cldr-localenames-full has no %s of %s", file, tag)
			}
			return table{}, nil // a locale without names of variants
		}
		var doc struct {
			Main map[string]struct {
				Names map[string]json.RawMessage `json:"localeDisplayNames"`
			} `json:"main"`
		}
		if err := json.Unmarshal(data, &doc); err != nil {
			return nil, fmt.Errorf("%s %s: %v", tag, file, err)
		}
		var t table
		if err := json.Unmarshal(doc.Main[tag].Names[key], &t); err != nil {
			return nil, fmt.Errorf("%s %s %s: %v", tag, file, key, err)
		}
		return t, nil
	}
	out := map[string]string{}
	for _, tag := range all {
		// The data of the locale, or of its nearest shipped ancestor.
		dataTag := tag
		for !shipped[dataTag] {
			dataTag = parentTag(dataTag, parents, likely)
			if dataTag == "" {
				return nil, fmt.Errorf("no names for %s", tag)
			}
		}
		langs, err := load(dataTag, "languages.json", "languages")
		if err != nil {
			return nil, err
		}
		scripts, err := load(dataTag, "scripts.json", "scripts")
		if err != nil {
			return nil, err
		}
		territories, err := load(dataTag, "territories.json", "territories")
		if err != nil {
			return nil, err
		}
		variants, err := load(dataTag, "variants.json", "variants")
		if err != nil {
			return nil, err
		}
		var patterns struct {
			LocalePattern   string `json:"localePattern"`
			LocaleSeparator string `json:"localeSeparator"`
		}
		{
			data := files["main/"+dataTag+"/localeDisplayNames.json"]
			var doc struct {
				Main map[string]struct {
					Names struct {
						Pattern json.RawMessage `json:"localeDisplayPattern"`
					} `json:"localeDisplayNames"`
				} `json:"main"`
			}
			if err := json.Unmarshal(data, &doc); err != nil {
				return nil, fmt.Errorf("%s localeDisplayNames.json: %v", dataTag, err)
			}
			if err := json.Unmarshal(doc.Main[dataTag].Names.Pattern, &patterns); err != nil {
				return nil, fmt.Errorf("%s localeDisplayPattern: %v", dataTag, err)
			}
		}
		parts := strings.Split(tag, "-")
		lang := parts[0]
		var script, region string
		var vars []string
		for _, p := range parts[1:] {
			switch {
			case len(p) == 4 && p[0] >= 'A' && p[0] <= 'Z':
				script = p
			case region == "" && (len(p) == 2 || len(p) == 3 && p[0] >= '0' && p[0] <= '9'):
				region = p
			default:
				vars = append(vars, p)
			}
		}
		// ICU's dialect names: lang-script-region when the tag has both
		// (ICU tries nothing else then), else lang-script or lang-region.
		name := ""
		hasScript, hasRegion := script != "", region != ""
		switch {
		case hasScript && hasRegion:
			if n := langs[lang+"-"+script+"-"+region]; n != "" {
				name, hasScript, hasRegion = n, false, false
			}
		case hasScript:
			if n := langs[lang+"-"+script]; n != "" {
				name, hasScript = n, false
			}
		case hasRegion:
			if n := langs[lang+"-"+region]; n != "" {
				name, hasRegion = n, false
			}
		}
		if name == "" {
			name = langs[lang]
			if name == "" {
				name = lang
			}
		}
		open, close := "(", ")"
		ropen, rclose := "[", "]"
		if strings.Contains(patterns.LocalePattern, "（") {
			open, close, ropen, rclose = "（", "）", "［", "］"
		}
		fix := func(s string) string {
			return strings.ReplaceAll(strings.ReplaceAll(s, open, ropen), close, rclose)
		}
		var rest []string
		if hasScript {
			n := scripts[script]
			if n == "" {
				n = script
			}
			rest = append(rest, fix(n))
		}
		if hasRegion {
			n := territories[region]
			if n == "" {
				n = region
			}
			rest = append(rest, fix(n))
		}
		for _, v := range vars {
			n := variants[strings.ToUpper(v)]
			if n == "" {
				n = strings.ToUpper(v)
			}
			rest = append(rest, fix(n))
		}
		if len(rest) > 0 {
			r := rest[0]
			for _, x := range rest[1:] {
				r = strings.NewReplacer("{0}", r, "{1}", x).Replace(patterns.LocaleSeparator)
			}
			name = strings.NewReplacer("{0}", name, "{1}", r).Replace(patterns.LocalePattern)
		}
		out[tag] = name
	}
	return out, nil
}
