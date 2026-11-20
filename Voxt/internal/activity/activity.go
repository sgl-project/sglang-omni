// Package activity counts the finished results of each day, for Home's
// activity calendar: the days Voxt was used, the current and longest run
// of such days, and how much each day had. It keeps counts only, no text,
// in a document of its own (Key), so the history's clean-up does not take
// the older days with it.
package activity

import (
	"encoding/json"
	"time"
)

// Key is the document the log is stored under (store.Documents).
const Key = "activity:days"

// Day is one day's finished results: how many, and their characters.
type Day struct {
	Count int `json:"count"`
	Chars int `json:"chars,omitempty"`
}

// Log is the days with at least one finished result, by local date
// (DayKey).
type Log struct {
	Days map[string]Day `json:"days"`
}

// DayKey is t's local date, the log's key for its day.
func DayKey(t time.Time) string { return t.Format("2006-01-02") }

// Add counts a finished result of chars characters on t's day.
func (l *Log) Add(t time.Time, chars int) {
	if l.Days == nil {
		l.Days = map[string]Day{}
	}
	key := DayKey(t)
	day := l.Days[key]
	day.Count++
	day.Chars += max(chars, 0)
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

// Stats are the calendar's three numbers.
type Stats struct {
	// Active is how many days had a result; Current the run of such days
	// that ends today, or yesterday while today has none yet; Longest the
	// longest run.
	Active, Current, Longest int
}

// Stats counts the log's days up to today.
func (l Log) Stats(today time.Time) Stats {
	var s Stats
	for _, day := range l.Days {
		if day.Count > 0 {
			s.Active++
		}
	}
	if s.Active == 0 {
		return s
	}
	day := midnight(today)
	if l.Get(day).Count == 0 {
		day = day.AddDate(0, 0, -1) // today is not over: the run may go on
	}
	for l.Get(day).Count > 0 {
		s.Current++
		day = day.AddDate(0, 0, -1)
	}
	// The longest run: walk each run from its first day.
	for key, d := range l.Days {
		start, err := time.ParseInLocation("2006-01-02", key, today.Location())
		if err != nil || d.Count == 0 || l.Get(start.AddDate(0, 0, -1)).Count > 0 {
			continue
		}
		run := 0
		for day := start; l.Get(day).Count > 0; day = day.AddDate(0, 0, 1) {
			run++
		}
		s.Longest = max(s.Longest, run)
	}
	return s
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
