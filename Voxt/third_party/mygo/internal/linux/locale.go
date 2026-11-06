//go:build linux

package linux

import (
	"os"
	"strings"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"

	"github.com/egoist/mygo/internal/platform"
)

// The languages and regional formats of the user are those of the
// environment, as gettext and glibc read them: no setting of the desktop
// changes them while the app runs.

// PreferredLanguages returns the languages of the environment as gettext
// reads them: LANGUAGE, a list separated by colons, unless the locale is C,
// then the locale of messages, LC_ALL, LC_MESSAGES or LANG.
func (a appController) PreferredLanguages() []string {
	locale := envLocale("LC_MESSAGES")
	var out []string
	if locale != "" {
		for _, l := range strings.Split(os.Getenv("LANGUAGE"), ":") {
			if l != "" {
				out = append(out, l)
			}
		}
		out = append(out, locale)
	}
	return out
}

// envLocale returns the locale of a category as glibc reads the
// environment, LC_ALL, then the category's, then LANG, or "" for the C
// locale.
func envLocale(category string) string {
	for _, env := range []string{"LC_ALL", category, "LANG"} {
		if v := os.Getenv(env); v != "" {
			if v == "C" || v == "POSIX" || strings.HasPrefix(v, "C.") {
				return ""
			}
			return v
		}
	}
	return ""
}

// SavedLocale and SaveLocale leave the app's language to the core's file.
func (a appController) SavedLocale() (string, bool) { return "", false }
func (a appController) SaveLocale(string) bool      { return false }

// SetLanguages gives WebKitGTK the languages, as the Accept-Language of
// its requests and the navigator.language of pages: now if it is loaded,
// else as it loads for the first web view. GTK's widgets, as menus, whose
// items take no direction from the window, are laid out in the direction.
func (a appController) SetLanguages(tags []string, rtl bool) {
	dir := int32(1) // GTK_TEXT_DIR_LTR
	if rtl {
		dir = 2
	}
	if gtkWidgetSetDefaultDirection != nil {
		gtkWidgetSetDefaultDirection(dir)
	}
	a.b.languages = append([]string(nil), tags...)
	if libWebKit != 0 {
		a.b.applyWebLanguages()
	}
}

// applyWebLanguages gives the default web context the app's languages.
func (b *Backend) applyWebLanguages() {
	if len(b.languages) == 0 || webkitWebContextSetPreferredLanguages == nil {
		return
	}
	strs := make([]*byte, len(b.languages)+1) // NULL-terminated
	for i, l := range b.languages {
		strs[i] = cs(l)
	}
	webkitWebContextSetPreferredLanguages(webkitWebContextGetDefault(), &strs[0])
}

// Region returns the formats of LC_TIME, as glibc has them for the
// environment's locale: the first day of the week and the hour cycle of
// its time format. A libc without them, as musl, or a locale the system
// has not generated, leaves them to the core.
func (a appController) Region() platform.Region {
	r := platform.Region{Locale: envLocale("LC_TIME"), FirstWeekday: -1}
	if r.Locale == "" {
		return r
	}
	langinfo.once.Do(loadLanginfo)
	if langinfo.newlocale == nil {
		return r
	}
	const lcTimeMask = 1 << 2 // LC_TIME is 2 in glibc and musl
	loc := langinfo.newlocale(lcTimeMask, "", 0)
	if loc == 0 {
		return r // not generated
	}
	defer langinfo.freelocale(loc)
	if f := langinfo.get(tFmt, loc); f != 0 {
		r.HourCycle = hourCycleOf(goStr(f))
	}
	// _NL_TIME_FIRST_WEEKDAY counts from _NL_TIME_WEEK_1STDAY, a date: 1
	// is the day of that date, Sunday for 19971130 and Monday for
	// 19971201, as GTK's calendar reads them.
	if first := langinfo.get(nlTimeFirstWeekday, loc); first != 0 {
		day := int(*(*byte)(*(*unsafe.Pointer)(unsafe.Pointer(&first))))
		origin := uint32(langinfo.get(nlTimeWeek1stDay, loc))
		start := -1
		switch origin {
		case 19971130:
			start = 0
		case 19971201:
			start = 1
		}
		if start >= 0 && day >= 1 && day <= 7 {
			r.FirstWeekday = (start + day - 1) % 7
		}
	}
	return r
}

// Items of nl_langinfo for LC_TIME in glibc: _NL_ITEM(LC_TIME, index).
const (
	tFmt               = 2<<16 | 42
	nlTimeWeek1stDay   = 2<<16 | 102
	nlTimeFirstWeekday = 2<<16 | 104
)

// langinfoLib is libc's locale objects, bound once.
type langinfoLib struct {
	once       sync.Once
	newlocale  func(mask int32, locale string, base uintptr) uintptr
	freelocale func(loc uintptr)
	langinfoL  uintptr
}

var langinfo langinfoLib

func loadLanginfo() {
	sym := func(name string) uintptr {
		p, err := purego.Dlsym(purego.RTLD_DEFAULT, name)
		if err != nil {
			return 0
		}
		return p
	}
	n, f, l := sym("newlocale"), sym("freelocale"), sym("nl_langinfo_l")
	if n == 0 || f == 0 || l == 0 {
		return
	}
	purego.RegisterFunc(&langinfo.newlocale, n)
	purego.RegisterFunc(&langinfo.freelocale, f)
	langinfo.langinfoL = l
}

// get calls nl_langinfo_l, whose result is a string, or for some items a
// number, in a pointer.
func (l *langinfoLib) get(item int, loc uintptr) uintptr {
	r, _, _ := purego.SyscallN(l.langinfoL, uintptr(item), loc)
	if r != 0 && item != nlTimeWeek1stDay && *(*byte)(*(*unsafe.Pointer)(unsafe.Pointer(&r))) == 0 {
		return 0 // an empty string: an item the libc has not
	}
	return r
}

// hourCycleOf returns the hour cycle of a strftime format: 12 for %I, %l
// and %r, 24 for %H, %k, %R and %T, 0 for none.
func hourCycleOf(format string) int {
	for i := 0; i+1 < len(format); i++ {
		if format[i] != '%' {
			continue
		}
		i++
		// Flags and modifiers between % and the conversion.
		for i < len(format) && strings.IndexByte("_-0^#EO", format[i]) >= 0 {
			i++
		}
		if i >= len(format) {
			break
		}
		switch format[i] {
		case 'I', 'l', 'r':
			return 12
		case 'H', 'k', 'R', 'T':
			return 24
		}
	}
	return 0
}
