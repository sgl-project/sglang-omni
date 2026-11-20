package ui

import (
	"strconv"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/egoist/mygo/i18n"
)

// TimeInput creates a field of the time of day of *tm, in the hour cycle
// of the user's regional formats (Context.Region): 15:04, or 3:04 PM with
// the window's language's names of the halves of the day, before the
// hours where the language writes them so (下午3:04). Its hours, minutes
// and half of the day each take the focus, as the segments of AppKit's
// date picker: Up and Down step the one with the focus round the clock,
// digits typed set it, going on to the minutes once the hours are typed,
// the first letter of a half of the day chooses it, and Left and Right
// move between them. Changed reports a new time, which keeps the date and
// the location of *tm. Assistive technology sees spin buttons, named
// after the field's Label and "hours", "minutes" and "AM/PM". It reads
// left to right in an interface laid out right to left too, as times are
// written there.
//
//	ui.TimeInput(c, &app.alarm).Label("Alarm")
func coreTimeInput(c *context, tm *time.Time) *node {
	t := c.theme
	region, locale := c.formats()
	twelve := region.HourCycle == 12
	// The order of the segments and what separates the hours from the
	// minutes, as the locale's pattern of hours and minutes writes them.
	sep, periodFirst := ":", false
	if twelve {
		sep, periodFirst = timeLayout(i18n.NewDateFormatter(locale, i18n.DateOptions{Skeleton: "jmm", HourCycle: 12}).Pattern())
	} else {
		sep, _ = timeLayout(i18n.NewDateFormatter(locale, i18n.DateOptions{Skeleton: "jmm", HourCycle: 24}).Pattern())
	}
	am, pm := i18n.DayPeriods(locale, i18n.Abbreviated)
	f := coreRow(c).Direction(LeftToRight).AlignItems(Center).Padding(t.Space(1.5), t.Space(2)).Radius(t.Radius).Background(t.Surface).Border(1, t.Border).Shrink(0).Role(RoleGroup)
	f.widget = "TimeInput"
	set := func(h, m int) {
		y, mo, d := tm.Date()
		next := time.Date(y, mo, d, h, m, tm.Second(), tm.Nanosecond(), tm.Location())
		if !next.Equal(*tm) {
			*tm = next
			f.st.markChanged()
			c.rt.consumed = true
		}
	}
	texts := c.texts()
	// segs are the hours, the minutes and, in 12 hours, the half of the
	// day; order is the order they show in.
	var segs [3]*node
	order := []int{0, 1}
	if twelve {
		order = []int{0, 1, 2}
		if periodFirst {
			order = []int{2, 0, 1}
		}
	}
	f.Children(func() {
		for i, k := range order {
			if k == 1 {
				coreText(c, sep).TextColor(t.TextMuted).Padding(0, 1).Role(RoleNone)
			} else if i > 0 {
				coreBox(c).Width(t.Space(1)).Role(RoleNone)
			}
			seg := coreBox(c).Padding(0, t.Space(0.5)).Radius(t.Space(1)).Focusable().FocusRing(false).Role(RoleStepper)
			seg.flags |= flagTypeSelect
			seg.nameFrom, seg.nameJoin = f, true
			var text string
			switch k {
			case 0:
				h, lo, hi := tm.Hour(), 0, 23
				if twelve {
					h, lo, hi = (h+11)%12+1, 1, 12
					text = c.formatNumber(locale, h, 1)
				} else {
					text = c.formatNumber(locale, h, 2)
				}
				seg.label = texts.Hours()
				seg.hasRange, seg.accRange, seg.accStep = true, [3]float64{float64(lo), float64(hi), float64(h)}, 1
			case 1:
				seg.label = texts.Minutes()
				seg.hasRange, seg.accRange, seg.accStep = true, [3]float64{0, 59, float64(tm.Minute())}, 1
				text = c.formatNumber(locale, tm.Minute(), 2)
			case 2:
				seg.label = texts.DayPeriod()
				text = am
				if tm.Hour() >= 12 {
					text = pm
				}
				seg.accValue = text
			}
			if seg.Focused() {
				seg.Background(t.Accent).TextColor(t.AccentText)
			}
			seg.Children(func() { coreText(c, text).FontFeatures("tnum") })
			segs[k] = seg
		}
	})
	f.afterInput(func() {
		h, m := tm.Hour(), tm.Minute()
		for i, k := range order {
			seg := segs[k]
			step := 0
			switch {
			case seg.Shortcut(0, KeyUp):
				step = 1
			case seg.Shortcut(0, KeyDown):
				step = -1
			case seg.Shortcut(0, KeyRight) && i+1 < len(order):
				segs[order[i+1]].Focus()
				c.rt.focusVisible = true
			case seg.Shortcut(0, KeyLeft) && i > 0:
				segs[order[i-1]].Focus()
				c.rt.focusVisible = true
			}
			if k == 2 {
				// The half of the day: Up and Down switch it, as does
				// the first letter of the other's name.
				pmNow := h >= 12
				want := pmNow
				if step != 0 {
					want = !pmNow
				}
				if s := seg.st; s.typing && s.typed != "" {
					r, _ := utf8.DecodeLastRuneInString(s.typed)
					switch {
					case startsWith(pm, r) || unicode.ToLower(r) == 'p':
						want = true
					case startsWith(am, r) || unicode.ToLower(r) == 'a':
						want = false
					}
					s.typed = ""
				}
				if want != pmNow {
					if want {
						h += 12
					} else {
						h -= 12
					}
					set(h, m)
				}
				continue
			}
			value, top := &h, 24
			if k == 1 {
				value, top = &m, 60
			}
			if step != 0 {
				if k == 0 && twelve {
					// Round the twelve hours of the same half of the day.
					half := *value / 12 * 12
					*value = half + (*value%12+step+12)%12
				} else {
					*value = (*value + step + top) % top
				}
				set(h, m)
			}
			// Digits typed: the first sets the segment, a second makes two
			// digits of them, or starts anew past the top; both typed, or one
			// that can't begin two, go on to the minutes.
			if s := seg.st; s.typing && s.typed != "" {
				top := top
				if k == 0 && twelve {
					top = 13
				}
				typed := s.typed
				n, err := strconv.Atoi(typed)
				if err != nil || len(typed) > 2 || n >= top {
					typed = typed[len(typed)-1:]
					n, err = strconv.Atoi(typed)
					s.typed = typed
				}
				if err == nil {
					switch {
					case k == 0 && twelve && n == 0:
						// The first digit of 01 to 09.
					case k == 0 && twelve:
						*value = h/12*12 + n%12
						set(h, m)
					default:
						*value = n
						set(h, m)
					}
					if (len(typed) == 2 || n*10 >= top) && k == 0 {
						s.typed = ""
						for j := i + 1; j < len(order); j++ {
							if order[j] == 1 {
								segs[1].Focus()
							}
						}
					}
				}
			}
		}
	})
	f.styleFn = func(f *node) {
		for _, k := range order {
			if segs[k].Focused() {
				f.borderC = t.Accent
			}
		}
	}
	return f
}

// timeLayout reads a pattern of hours and minutes of CLDR, as "h:mm a",
// "ah:mm" or "H 'h' mm": the text that separates the hours from the
// minutes, its quotes removed, and whether the half of the day comes
// before the hours.
func timeLayout(pattern string) (sep string, periodFirst bool) {
	// The fields (a letter repeated, outside quotes) and the literal text
	// between them.
	type item struct {
		field byte
		text  string
	}
	var items []item
	var lit strings.Builder
	quoted := false
	for i := 0; i < len(pattern); i++ {
		c := pattern[i]
		switch {
		case c == '\'':
			if i+1 < len(pattern) && pattern[i+1] == '\'' {
				lit.WriteByte('\'')
				i++
			} else {
				quoted = !quoted
			}
		case !quoted && (c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z'):
			items = append(items, item{text: lit.String()})
			lit.Reset()
			for i+1 < len(pattern) && pattern[i+1] == c {
				i++
			}
			items = append(items, item{field: c})
		default:
			lit.WriteByte(c)
		}
	}
	hour, minute, period := -1, -1, -1
	for i, it := range items {
		switch it.field {
		case 'h', 'H', 'k', 'K':
			if hour < 0 {
				hour = i
			}
		case 'm':
			if minute < 0 {
				minute = i
			}
		case 'a', 'b', 'B':
			if period < 0 {
				period = i
			}
		}
	}
	sep = ":"
	if hour >= 0 && minute > hour {
		var b strings.Builder
		for _, it := range items[hour+1 : minute] {
			b.WriteString(it.text)
		}
		sep = b.String()
	}
	return sep, period >= 0 && hour >= 0 && period < hour
}

// startsWith reports whether a name of a half of the day starts with r,
// ignoring case.
func startsWith(name string, r rune) bool {
	first, _ := utf8.DecodeRuneInString(name)
	return unicode.ToLower(first) == unicode.ToLower(r)
}
