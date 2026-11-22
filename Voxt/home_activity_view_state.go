package main

import (
	"fmt"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/activity"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// homeActivityWeeks is how many weeks a calendar page shows: about five
// months, so the days keep a readable size beside the numbers.
const homeActivityWeeks = 22

// homeActivityStatViewState is one of the numbers beside the calendar:
// "12,345" and "characters".
type homeActivityStatViewState struct {
	Value string `json:"value"`
	Label string `json:"label"`
}

type homeActivityDayViewState struct {
	// Level is the square's shade (activity.Level); Future a day after
	// today, drawn empty.
	Level int `json:"level"`
	// Date and Count are the two lines of the square's tooltip: its day,
	// and its results or that it had none.
	Date   string `json:"date"`
	Count  string `json:"count"`
	Future bool   `json:"future,omitempty"`
}

type homeActivityMonthViewState struct {
	Week  int    `json:"week"`
	Label string `json:"label"`
}

type homeActivityViewState struct {
	// Stats are today's and the last 30 days' characters, speech rate and
	// average recognition time.
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
// the log's calendar.
func buildHomeActivityViewState(translator *i18n.Translator, log activity.Log, page int, now time.Time, usage homeUsage) homeActivityViewState {
	state := homeActivityViewState{CanForward: page > 0, Levels: activity.Levels, Stats: homeActivityStats(translator, log.Stats(now), usage)}
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
				Date: homeActivityDate(translator, day), Count: homeActivityCount(translator, count),
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

// homeActivityDate is a square's day, as its tooltip's first line says it.
func homeActivityDate(translator *i18n.Translator, day time.Time) string {
	if translator.Locale() == "zh-Hans" {
		return fmt.Sprintf("%d年%d月%d日", day.Year(), day.Month(), day.Day())
	}
	return day.Format("Jan 2, 2006")
}

// homeActivityCount is a square's results, its tooltip's second line.
func homeActivityCount(translator *i18n.Translator, count int) string {
	if count == 0 {
		return translator.T("No activity")
	}
	return translator.N(count, "%d result", "%d results", count)
}

func homeActivityMonth(translator *i18n.Translator, month time.Month) string {
	if translator.Locale() == "zh-Hans" {
		return fmt.Sprintf("%d月", month)
	}
	return month.String()[:3]
}

// homeActivityStats are Home's four usage numbers. Speed and latency are
// dashes until their timings exist.
func homeActivityStats(translator *i18n.Translator, s activity.Stats, usage homeUsage) []homeActivityStatViewState {
	speed, latency := "—", "—"
	if s.PerMinute > 0 {
		speed = groupDigits(s.PerMinute)
	}
	if usage.asrCount > 0 {
		latency = translator.T("%.1f s", float64(usage.asrMS)/float64(usage.asrCount)/1000)
	}
	return []homeActivityStatViewState{
		{Value: groupDigits(s.TodayChars), Label: translator.T("Today")},
		{Value: groupDigits(s.MonthChars), Label: translator.T("30 days")},
		{Value: speed, Label: translator.T("Per minute")},
		{Value: latency, Label: translator.T("Latency")},
	}
}
