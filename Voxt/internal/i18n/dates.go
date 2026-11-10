package i18n

import (
	"fmt"
	"time"
)

// DateTime formats a moment for lists and cards in the active language:
// day, month and time, with the year when it is not the current one
// ("Jan 2 15:04" / "Jan 2, 2025 15:04"; "1月2日 15:04" / "2025年1月2日 15:04").
// Month names are English: Go's time package has no other.
func DateTime(t time.Time) string {
	t = t.Local()
	sameYear := t.Year() == time.Now().Year()
	switch Locale() {
	case "zh-Hans":
		if sameYear {
			return fmt.Sprintf("%d月%d日 %s", t.Month(), t.Day(), t.Format("15:04"))
		}
		return fmt.Sprintf("%d年%d月%d日 %s", t.Year(), t.Month(), t.Day(), t.Format("15:04"))
	}
	if sameYear {
		return t.Format("Jan 2 15:04")
	}
	return t.Format("Jan 2, 2006 15:04")
}

// Date formats a day in the active language ("Jan 2, 2006" / "2006年1月2日").
func Date(t time.Time) string {
	t = t.Local()
	if Locale() == "zh-Hans" {
		return fmt.Sprintf("%d年%d月%d日", t.Year(), t.Month(), t.Day())
	}
	return t.Format("Jan 2, 2006")
}
