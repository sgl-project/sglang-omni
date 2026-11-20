package activity

import (
	"testing"
	"time"
)

func day(s string) time.Time {
	t, err := time.ParseInLocation("2006-01-02 15:04", s+" 12:00", time.Local)
	if err != nil {
		panic(err)
	}
	return t
}

func logOf(days ...string) Log {
	var l Log
	for _, d := range days {
		l.Add(day(d), 10)
	}
	return l
}

func TestStats(t *testing.T) {
	for _, tc := range []struct {
		name  string
		log   Log
		today string
		want  Stats
	}{
		{name: "empty", log: Log{}, today: "2026-11-20", want: Stats{}},
		{name: "today only", log: logOf("2026-11-20"), today: "2026-11-20", want: Stats{1, 1, 1}},
		{name: "run up to yesterday stays current", log: logOf("2026-11-18", "2026-11-19"), today: "2026-11-20", want: Stats{2, 2, 2}},
		{name: "a missed day ends the run", log: logOf("2026-11-17", "2026-11-18"), today: "2026-11-20", want: Stats{2, 0, 2}},
		{name: "longest is an older run", log: logOf("2026-10-01", "2026-10-02", "2026-10-03", "2026-11-19", "2026-11-20"), today: "2026-11-20", want: Stats{5, 2, 3}},
		{name: "across a month and a year", log: logOf("2025-12-31", "2026-01-01", "2026-01-02"), today: "2026-01-02", want: Stats{3, 3, 3}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.log.Stats(day(tc.today)); got != tc.want {
				t.Fatalf("got %+v, want %+v", got, tc.want)
			}
		})
	}
}

func TestAddAndStore(t *testing.T) {
	var l Log
	l.Add(day("2026-11-07"), 12)
	l.Add(day("2026-11-07"), 3)
	l.Add(day("2026-11-08"), -1)
	data, err := l.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	back, err := Unmarshal(data)
	if err != nil {
		t.Fatal(err)
	}
	if got := back.Get(day("2026-11-07")); got != (Day{Count: 2, Chars: 15}) {
		t.Fatalf("Nov 7: %+v", got)
	}
	if got := back.Get(day("2026-11-08")); got != (Day{Count: 1}) {
		t.Fatalf("Nov 8: %+v", got)
	}
}

func TestLevel(t *testing.T) {
	for count, want := range map[int]int{0: 0, 1: 1, 2: 1, 3: 2, 5: 2, 6: 3, 10: 3, 11: 4, 100: 4} {
		if got := Level(count); got != want {
			t.Errorf("Level(%d) = %d, want %d", count, got, want)
		}
	}
}

func TestWeeks(t *testing.T) {
	weeks := Weeks(day("2026-11-20"), 3) // a Friday
	if len(weeks) != 3 {
		t.Fatalf("%d weeks", len(weeks))
	}
	if got := DayKey(weeks[2][0]); got != "2026-11-15" {
		t.Errorf("last week starts %s, want Sunday Nov 15", got)
	}
	if got := DayKey(weeks[0][0]); got != "2026-11-01" {
		t.Errorf("first week starts %s, want Nov 1", got)
	}
	if got := DayKey(weeks[2][5]); got != "2026-11-20" {
		t.Errorf("the end's day is %s", got)
	}
}
