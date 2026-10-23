package uitext

// Sizes, paths, times and the width of text in display units, and the
// menu bar's short labels of results (TrayLabel).

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// HumanBytes shows a size: "1.2 GB", "340 MB", "12 KB", "512 B".
func HumanBytes(n int64) string {
	switch {
	case n >= 1e9:
		return fmt.Sprintf("%.1f GB", float64(n)/1e9)
	case n >= 1e6:
		return fmt.Sprintf("%.1f MB", float64(n)/1e6)
	case n >= 1e3:
		return fmt.Sprintf("%.0f KB", float64(n)/1e3)
	}
	return fmt.Sprintf("%d B", n)
}

// MemoryBytes shows memory as Activity Monitor does, in units of 1024:
// "3.4 GB", "512 MB" (a size on disk is HumanBytes').
func MemoryBytes(n int64) string {
	if n >= 1<<30 {
		return fmt.Sprintf("%.1f GB", float64(n)/(1<<30))
	}
	return fmt.Sprintf("%.0f MB", float64(n)/(1<<20))
}

// ShortPath shows a path under the home folder as "~/…".
func ShortPath(p string) string {
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		if rel, err := filepath.Rel(home, p); err == nil && !filepath.IsAbs(rel) && rel != ".." && len(rel) < len(p) && rel[:min(2, len(rel))] != ".." {
			return "~/" + rel
		}
	}
	return p
}

// Names joins names for a sentence: "AudioSmith", "AudioSmith and
// Dicta", "A, B and C" ("A、B 和 C").
func Names(names []string) string {
	if len(names) <= 1 {
		return strings.Join(names, "")
	}
	head := strings.Join(names[:len(names)-1], i18n.X("list", ", "))
	return i18n.T("%s and %s", head, names[len(names)-1])
}

// OrDash shows "—" for an empty value.
func OrDash(s string) string {
	if s == "" {
		return "—"
	}
	return s
}

// TimeAgo says how long ago t was, briefly.
func TimeAgo(t time.Time) string {
	d := time.Since(t)
	switch {
	case d < time.Minute:
		return i18n.T("just now")
	case d < time.Hour:
		return i18n.N(int(d.Minutes()), "%d minute ago", "%d minutes ago", int(d.Minutes()))
	case d < 24*time.Hour:
		return i18n.N(int(d.Hours()), "%d hour ago", "%d hours ago", int(d.Hours()))
	}
	return i18n.DateTime(t)
}

// ClockMS is a length as m:ss.
func ClockMS(ms int64) string {
	s := (ms + 500) / 1000
	return fmt.Sprintf("%d:%02d", s/60, s%60)
}

// StageTime is a stage's time: "850 ms", "1.2 s", "1:05"; "" for none.
func StageTime(ms int64) string {
	switch {
	case ms <= 0:
		return ""
	case ms < 1000:
		return fmt.Sprintf("%d ms", ms)
	case ms < 60_000:
		return fmt.Sprintf("%.1f s", float64(ms)/1000)
	}
	return ClockMS(ms)
}

// RecordingTime is the recording's length and, when the speech model
// got less of it (the silence left out), how much: "7.5 s, 2.5 s sent".
func RecordingTime(e store.Entry) string {
	rec := StageTime(e.AudioMS)
	if rec == "" || e.SentMS <= 0 || e.SentMS >= e.AudioMS-50 {
		return rec
	}
	return i18n.T("%s, %s sent", rec, StageTime(e.SentMS))
}

// TrayRecentWidth is how wide each of the menu bar's recent results
// shows (TrayLabel), in display units (a CJK letter 2, any other 1), the
// "…" included: about as wide as "Paste the last result ⌃⌘V", so a long
// result does not widen the menu (10-15: whole results made it as wide
// as the screen).
const TrayRecentWidth = 26

// TrayLabel is a result as a menu item shows it: one line, its spaces
// and line breaks one space, at most TrayRecentWidth display units wide,
// cut with "…" (Voxt cuts at a rendered width of 100 pt; a native menu's
// text cannot be measured here, so the units stand in for it).
func TrayLabel(text string) string {
	line := strings.Join(strings.Fields(text), " ")
	if DisplayWidth(line) <= TrayRecentWidth {
		return line
	}
	room, w := TrayRecentWidth-1, 0 // the "…" takes one
	cut := len(line)
	for i, r := range line {
		if w+runeWidth(r) > room {
			cut = i
			break
		}
		w += runeWidth(r)
	}
	// Not in the middle of a (Latin) word: back to its space, if near.
	if latinAt(line, cut-1) && latinAt(line, cut) {
		if sp := strings.LastIndexByte(line[:cut], ' '); sp > 0 && DisplayWidth(line[sp:cut]) <= 10 {
			cut = sp
		}
	}
	return strings.TrimRight(line[:cut], " ,.;:，。；：、") + "…"
}

// latinAt says whether the rune at byte i of s (or the one ending there,
// for the last byte of one) is a Latin letter or digit.
func latinAt(s string, i int) bool {
	if i < 0 || i >= len(s) {
		return false
	}
	for i > 0 && !utf8.RuneStart(s[i]) {
		i--
	}
	r, _ := utf8.DecodeRuneInString(s[i:])
	return r < 0x2E80 && (unicode.IsLetter(r) || unicode.IsDigit(r))
}

func runeWidth(r rune) int {
	switch {
	case WideRune(r):
		return 2
	case unicode.Is(unicode.Mn, r) || r == 0x200B || r == 0xFE0F:
		return 0
	}
	return 1
}

// DisplayWidth is how wide s shows in display units: a wide (CJK) letter
// 2, a combining mark 0, any other 1.
func DisplayWidth(s string) int {
	w := 0
	for _, r := range s {
		w += runeWidth(r)
	}
	return w
}

// WideRune says whether r is a wide (CJK) letter: as wide as the font
// is high.
func WideRune(r rune) bool {
	return unicode.Is(unicode.Han, r) || unicode.In(r, unicode.Hiragana, unicode.Katakana, unicode.Hangul) ||
		r >= 0xFF00 && r <= 0xFFEF || r >= 0x3000 && r <= 0x303F
}
