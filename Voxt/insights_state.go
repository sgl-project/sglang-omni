package main

// Home's Insights (insights_view.go): the totals the activity log keeps,
// above its calendar, and a card that shares them. Nothing new is
// stored: every number comes from the log's days (internal/activity).

import (
	"fmt"
	"sort"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/activity"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// insightsTypingRate is the typing speed "time saved" compares speech
// with: characters a minute, a typical rate for typing Chinese with
// pinyin (and about 25 words a minute of English).
const insightsTypingRate = 40

// insightsWeeks is how many weeks the share card's calendar shows.
const insightsWeeks = 13

// insightsPart is a number and its unit, as "9 hours 42 minutes" shows
// two of them.
type insightsPart struct {
	Value string `json:"value"`
	Unit  string `json:"unit"`
}

type insightsStatViewState struct {
	Parts []insightsPart `json:"parts"`
	Label string         `json:"label"`
}

type insightsViewState struct {
	// Stats are the characters dictated, the time saved, the speed and
	// the time spoken; Days the active days and the current and longest
	// streaks.
	Stats []insightsStatViewState `json:"stats"`
	Days  []insightsStatViewState `json:"days"`
	// Weeks are the share card's last insightsWeeks weeks, Sunday first,
	// each day's level (activity.Level), -1 for days still to come.
	Weeks [][7]int `json:"weeks"`
	// Months label the weeks a month starts in, "" for the others.
	Months []string `json:"months"`
}

func buildInsightsViewState(translator *i18n.Translator, log activity.Log, now time.Time) insightsViewState {
	stats, longest := log.Stats(now), insightsLongest(log)
	var chars, active int
	var audio time.Duration
	for _, day := range log.Days {
		chars += day.Chars
		audio += time.Duration(day.AudioMS) * time.Millisecond
		if day.Count > 0 {
			active++
		}
	}
	saved := time.Duration(float64(chars)/insightsTypingRate*float64(time.Minute)) - audio
	if saved < 0 || audio == 0 {
		saved = 0 // nothing timed yet, or slower than typing
	}
	speed := "—"
	if stats.PerMinute > 0 {
		speed = groupDigits(stats.PerMinute)
	}
	state := insightsViewState{
		Stats: []insightsStatViewState{
			{Parts: []insightsPart{{groupDigits(chars), translator.N(chars, "character", "characters")}}, Label: translator.T("Characters dictated")},
			{Parts: insightsDuration(translator, saved), Label: translator.T("Time saved")},
			{Parts: []insightsPart{{speed, translator.T("characters a minute")}}, Label: translator.T("Average speed")},
			{Parts: insightsDuration(translator, audio), Label: translator.T("Time spoken")},
		},
		Days: []insightsStatViewState{
			{Parts: []insightsPart{{groupDigits(active), translator.N(active, "day", "days")}}, Label: translator.T("Active days")},
			{Parts: []insightsPart{{groupDigits(stats.Streak), translator.N(stats.Streak, "day", "days")}}, Label: translator.T("Current streak")},
			{Parts: []insightsPart{{groupDigits(longest), translator.N(longest, "day", "days")}}, Label: translator.T("Longest streak")},
		},
	}
	today := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, now.Location())
	lastMonth := time.Month(0)
	for _, week := range activity.Weeks(now, insightsWeeks) {
		var levels [7]int
		for i, day := range week {
			levels[i] = activity.Level(log.Get(day).Count)
			if day.After(today) {
				levels[i] = -1
			}
		}
		state.Weeks = append(state.Weeks, levels)
		month := ""
		if m := week[0].Month(); m != lastMonth {
			if lastMonth != 0 {
				month = homeActivityMonth(translator, m)
			}
			lastMonth = m
		}
		state.Months = append(state.Months, month)
	}
	return state
}

// insightsDuration is d as hours and minutes ("2 hours 34 minutes"), or
// minutes alone under an hour.
func insightsDuration(translator *i18n.Translator, d time.Duration) []insightsPart {
	minutes := int(d.Round(time.Minute) / time.Minute)
	hours, minutes := minutes/60, minutes%60
	parts := []insightsPart{}
	if hours > 0 {
		parts = append(parts, insightsPart{fmt.Sprint(hours), translator.N(hours, "hour", "hours")})
	}
	if minutes > 0 || hours == 0 {
		parts = append(parts, insightsPart{fmt.Sprint(minutes), translator.N(minutes, "minute", "minutes")})
	}
	return parts
}

// insightsLongest is the longest run of days with a result.
func insightsLongest(log activity.Log) int {
	keys := make([]string, 0, len(log.Days))
	for key, day := range log.Days {
		if day.Count > 0 {
			keys = append(keys, key)
		}
	}
	sort.Strings(keys)
	longest, run := 0, 0
	var prev time.Time
	for _, key := range keys {
		day, err := time.ParseInLocation("2006-01-02", key, time.Local)
		if err != nil {
			continue
		}
		if !prev.IsZero() && prev.AddDate(0, 0, 1).Equal(day) {
			run++
		} else {
			run = 1
		}
		longest = max(longest, run)
		prev = day
	}
	return longest
}
