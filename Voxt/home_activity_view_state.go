package main

import (
	"fmt"
	"strconv"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/activity"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// homeActivityWeeks is how many weeks a calendar page shows: half a year.
const homeActivityWeeks = 26

type homeActivityStatViewState struct {
	Value string `json:"value"`
	Unit  string `json:"unit"`
	Label string `json:"label"`
}

type homeActivityDayViewState struct {
	// Level is the square's shade (activity.Level); Future a day after
	// today, drawn empty.
	Level  int    `json:"level"`
	Tip    string `json:"tip"`
	Future bool   `json:"future,omitempty"`
}

type homeActivityMonthViewState struct {
	Week  int    `json:"week"`
	Label string `json:"label"`
}

type homeActivityViewState struct {
	Stats    []homeActivityStatViewState   `json:"stats"`
	Weekdays [7]string                     `json:"weekdays"`
	Weeks    [][7]homeActivityDayViewState `json:"weeks"`
	Months   []homeActivityMonthViewState  `json:"months"`
	// Levels is how many shades a day with results has (the legend).
	Levels int `json:"levels"`
	// CanBack: an older page has days; CanForward: this is not today's.
	CanBack    bool `json:"can_back"`
	CanForward bool `json:"can_forward"`
}

// buildHomeActivityViewState lays out page (0: the weeks up to today) of
// the log's calendar, and its numbers as of now.
func buildHomeActivityViewState(translator *i18n.Translator, log activity.Log, page int, now time.Time) homeActivityViewState {
	stats := log.Stats(now)
	stat := func(n int, label string) homeActivityStatViewState {
		return homeActivityStatViewState{Value: strconv.Itoa(n), Unit: translator.N(n, "day", "days"), Label: label}
	}
	state := homeActivityViewState{
		Stats: []homeActivityStatViewState{
			stat(stats.Active, translator.T("Active days")), stat(stats.Current, translator.T("Current streak")),
			stat(stats.Longest, translator.T("Longest streak")),
		},
		CanForward: page > 0, Levels: activity.Levels,
	}
	names := []rune("SMTWTFS")
	if translator.Locale() == "zh-Hans" {
		names = []rune("日一二三四五六")
	}
	for weekday, name := range names {
		state.Weekdays[weekday] = string(name)
	}
	today := activity.DayKey(now)
	end := now.AddDate(0, 0, -7*homeActivityWeeks*page)
	weeks := activity.Weeks(end, homeActivityWeeks)
	state.Weeks = make([][7]homeActivityDayViewState, len(weeks))
	for week, days := range weeks {
		for weekday, day := range days {
			key := activity.DayKey(day)
			count := log.Get(day).Count
			state.Weeks[week][weekday] = homeActivityDayViewState{
				Level: activity.Level(count), Future: key > today,
				Tip: homeActivityDayTip(translator, day, count),
			}
			if day.Day() == 1 {
				state.Months = append(state.Months, homeActivityMonthViewState{Week: week, Label: homeActivityMonth(translator, day.Month())})
			}
		}
	}
	first := activity.DayKey(weeks[0][0])
	for key, day := range log.Days {
		if day.Count > 0 && key < first {
			state.CanBack = true
			break
		}
	}
	return state
}

// homeActivityDayTip is a square's tooltip: its date and its results.
func homeActivityDayTip(translator *i18n.Translator, day time.Time, count int) string {
	date := day.Format("Jan 2, 2006")
	if translator.Locale() == "zh-Hans" {
		date = fmt.Sprintf("%d年%d月%d日", day.Year(), day.Month(), day.Day())
	}
	if count == 0 {
		return translator.T("%s: no results", date)
	}
	return translator.T("%s: %s", date, translator.N(count, "%d result", "%d results", count))
}

func homeActivityMonth(translator *i18n.Translator, month time.Month) string {
	if translator.Locale() == "zh-Hans" {
		return fmt.Sprintf("%d月", month)
	}
	return month.String()[:3]
}
