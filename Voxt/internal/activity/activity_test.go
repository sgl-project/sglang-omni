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

func TestAddAndStore(t *testing.T) {
	var l Log
	l.Add(day("2026-11-07"), Result{})
	l.Add(day("2026-11-07"), Result{})
	l.Add(day("2026-11-08"), Result{})
	data, err := l.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	back, err := Unmarshal(data)
	if err != nil {
		t.Fatal(err)
	}
	if got := back.Get(day("2026-11-07")); got != (Day{Count: 2}) {
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

func TestStats(t *testing.T) {
	var l Log
	l.Add(day("2026-11-18"), Result{Chars: 100, Spoken: 90, Audio: 30 * time.Second})
	l.Add(day("2026-11-19"), Result{Chars: 40, Spoken: 60, Audio: 30 * time.Second})
	l.Add(day("2026-11-19"), Result{Chars: 7}) // typed, no speech
	if got := l.Stats(day("2026-11-20")); got != (Stats{MonthChars: 147, Chars: 147, PerMinute: 150, Streak: 2}) {
		t.Fatalf("a run up to yesterday: %+v", got)
	}
	if got := l.Stats(day("2026-11-21")); got.Streak != 0 {
		t.Fatalf("a missed day ends the run: %+v", got)
	}
	if got := (Log{}).Stats(day("2026-11-21")); got != (Stats{}) {
		t.Fatalf("empty: %+v", got)
	}
}
