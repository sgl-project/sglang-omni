// Package activity counts the finished results of each day, for Home's
// activity calendar: the days Voxt was used and how much each day had. It keeps counts only, no text,
// in a document of its own (Key), so the history's clean-up does not take
// the older days with it.
package activity

import (
	"encoding/json"
	"time"
)

// Key is the document the log is stored under (store.Documents).
const Key = "activity:days"

// Day is one day's finished results.
type Day struct {
	Count int `json:"count"`
}

// Log is the days with at least one finished result, by local date
// (DayKey).
type Log struct {
	Days map[string]Day `json:"days"`
}

// DayKey is t's local date, the log's key for its day.
func DayKey(t time.Time) string { return t.Format("2006-01-02") }

// Add counts a finished result on t's day.
func (l *Log) Add(t time.Time) {
	if l.Days == nil {
		l.Days = map[string]Day{}
	}
	key := DayKey(t)
	day := l.Days[key]
	day.Count++
	l.Days[key] = day
}

// Get is t's day (zero when nothing finished then).
func (l Log) Get(t time.Time) Day { return l.Days[DayKey(t)] }

// Marshal and Unmarshal are the stored form.
func (l Log) Marshal() ([]byte, error) { return json.Marshal(l) }

func Unmarshal(data []byte) (Log, error) {
	var l Log
	err := json.Unmarshal(data, &l)
	return l, err
}

// Levels is how many shades the calendar has for a day with results.
const Levels = 4

// Level is a day's shade: 0 for none, then 1–Levels as it had more
// results (1–2, 3–5, 6–10, more).
func Level(count int) int {
	switch {
	case count <= 0:
		return 0
	case count <= 2:
		return 1
	case count <= 5:
		return 2
	case count <= 10:
		return 3
	}
	return 4
}

// Weeks are the calendar's columns: weeks of seven days from Sunday, the
// last one holding end's day, weeks columns in all.
func Weeks(end time.Time, weeks int) [][7]time.Time {
	last := midnight(end)
	last = last.AddDate(0, 0, -int(last.Weekday())) // that week's Sunday
	first := last.AddDate(0, 0, -7*(weeks-1))
	out := make([][7]time.Time, weeks)
	for week := range out {
		for weekday := range 7 {
			out[week][weekday] = first.AddDate(0, 0, 7*week+weekday)
		}
	}
	return out
}

func midnight(t time.Time) time.Time {
	y, m, d := t.Date()
	return time.Date(y, m, d, 0, 0, 0, 0, t.Location())
}
