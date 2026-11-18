//go:build windows && (amd64 || arm64)

package windows

import (
	"slices"
	"strconv"
	"syscall"
	"unsafe"

	"github.com/egoist/mygo/internal/platform"
)

// The languages and regional formats of the user: the display languages
// of Settings > Time & language, and the regional formats, which may be
// another language's, as English Windows with the formats of China.

const muiLanguageName = 0x8 // MUI_LANGUAGE_NAME

// PreferredLanguages returns the user's display languages, in order
// (GetUserPreferredUILanguages), or the display language alone on a
// Windows that lists none.
func (a appController) PreferredLanguages() []string {
	if langs := multiString(procGetUserPreferredUILanguages); len(langs) > 0 {
		return langs
	}
	buf := make([]uint16, 85) // LOCALE_NAME_MAX_LENGTH
	if id, _, _ := procGetUserDefaultUILanguage.Call(); id != 0 {
		if n, _, _ := procLCIDToLocaleName.Call(id, uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)), 0); n > 0 {
			return []string{syscall.UTF16ToString(buf)}
		}
	}
	return nil
}

// multiString calls a function of the MUI that writes a list of language
// names, each ending in a NUL and the list in another, into a buffer it
// is first asked the size of.
func multiString(proc *syscall.LazyProc) []string {
	var count, size uint32
	if ok, _, _ := proc.Call(muiLanguageName, uintptr(unsafe.Pointer(&count)), 0, uintptr(unsafe.Pointer(&size))); ok == 0 || size == 0 {
		return nil
	}
	buf := make([]uint16, size)
	if ok, _, _ := proc.Call(muiLanguageName, uintptr(unsafe.Pointer(&count)), uintptr(unsafe.Pointer(&buf[0])), uintptr(unsafe.Pointer(&size))); ok == 0 {
		return nil
	}
	var out []string
	start := 0
	for i, c := range buf {
		if c == 0 {
			if i > start {
				out = append(out, syscall.UTF16ToString(buf[start:i]))
			}
			start = i + 1
		}
	}
	return out
}

// The items of GetLocaleInfoEx.
const (
	localeSName           = 0x5c
	localeSTimeFormat     = 0x1003
	localeIFirstDayOfWeek = 0x100c
)

// localeInfo returns an item of the user's regional formats, with the
// changes they made to them.
func localeInfo(item uint32) string {
	buf := make([]uint16, 128)
	n, _, _ := procGetLocaleInfoEx.Call(0, uintptr(item), uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf))) // LOCALE_NAME_USER_DEFAULT
	if n == 0 {
		return ""
	}
	return syscall.UTF16ToString(buf[:n])
}

// Region reads the user's regional formats: their locale, the first day of
// the week (0 for Monday to 6 for Sunday) and the time format.
func (a appController) Region() platform.Region {
	r := platform.Region{Locale: localeInfo(localeSName), FirstWeekday: -1}
	if d, err := strconv.Atoi(localeInfo(localeIFirstDayOfWeek)); err == nil && d >= 0 && d <= 6 {
		r.FirstWeekday = (d + 1) % 7
	}
	r.HourCycle = timeFormatCycle(localeInfo(localeSTimeFormat))
	return r
}

// timeFormatCycle returns the hour cycle of a Windows time format: 12 for
// h, 24 for H, 0 for none. Quoted text is not a field.
func timeFormatCycle(format string) int {
	quoted := false
	for _, c := range format {
		switch {
		case c == '\'':
			quoted = !quoted
		case quoted:
		case c == 'h':
			return 12
		case c == 'H':
			return 24
		}
	}
	return 0
}

// SavedLocale and SaveLocale leave the app's language to the core's file.
func (a appController) SavedLocale() (string, bool) { return "", false }
func (a appController) SaveLocale(string) bool      { return false }

// SetLanguages makes the app's languages the process's preferred display
// languages, which the system's own texts, as those of message boxes and
// file dialogs, are in where Windows has them, and the language of the
// WebView2 environment, which the first web view creates: one created
// already keeps its language, as WebView2 cannot change it.
func (a appController) SetLanguages(tags []string, _ bool) {
	if slices.Equal(tags, a.b.languages) {
		return
	}
	a.b.languages = slices.Clone(tags)
	var list []uint16
	for _, t := range tags {
		list = append(list, utf16z(t)...)
	}
	list = append(list, 0)
	var n uint32
	procSetProcessPreferredUILanguages.Call(muiLanguageName, uintptr(unsafe.Pointer(&list[0])), uintptr(unsafe.Pointer(&n)))
}
