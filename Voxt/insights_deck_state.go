package main

// The insights cards (insights_deck.go): six pictures of the activity
// log's totals, one at a time, each to copy or save. This file builds
// what each card shows; like Home's Insights (insights_state.go), every
// number comes from the log's days.

import (
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/activity"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// The cards, in their order.
const (
	insightsCardHighlights = "highlights" // the overview: the four totals, the days and the calendar
	insightsCardCharacters = "characters" // the characters dictated, large
	insightsCardStreak     = "streak"     // the current streak over the last seven days
	insightsCardSpeed      = "speed"      // the speed on a dial, against typing
	insightsCardSaved      = "saved"      // the time saved: typing's bar over speaking's
	insightsCardDays       = "days"       // the last weeks' calendar
)

var insightsCardKinds = []string{
	insightsCardHighlights, insightsCardCharacters, insightsCardStreak,
	insightsCardSpeed, insightsCardSaved, insightsCardDays,
}

const (
	// insightsPageChars is a book's page, for the characters card's line.
	insightsPageChars = 500
	// insightsNovelChars is a novel's length.
	insightsNovelChars = 100000
	// insightsDialTimes is the speed dial's full scale, in typing speeds.
	insightsDialTimes = 8
	// insightsSettled is an Elapsed past every card's entrance: what a
	// copied picture and Reduce motion show.
	insightsSettled = time.Hour
)

// insightsWeekdayViewState is one of the streak card's last seven days.
type insightsWeekdayViewState struct {
	Label  string `json:"label"`
	Active bool   `json:"active,omitempty"`
	Today  bool   `json:"today,omitempty"`
}

// insightsBarViewState is one of the saved card's bars: Fraction of the
// longest.
type insightsBarViewState struct {
	Label    string  `json:"label"`
	Value    string  `json:"value"`
	Fraction float32 `json:"fraction"`
}

// insightsFaceViewState is one card. Kind says which of the fields under
// Date it shows.
type insightsFaceViewState struct {
	Kind  string `json:"kind"`
	Theme int    `json:"theme"`
	Title string `json:"title"`
	// Move is the lockup's mascot's move as the card comes up
	// (mascotPoseAt).
	Move int `json:"move"`
	// Date is the day the card was made, in its corner.
	Date string `json:"date"`

	// Stats are the overview's four totals, and the days card's two;
	// Days the overview's active days and streaks.
	Stats []insightsStatViewState `json:"stats,omitempty"`
	Days  []insightsStatViewState `json:"days,omitempty"`
	// Number is the card's large number (characters, days of streak,
	// characters a minute), Unit its unit; Known false shows a dash.
	Number int    `json:"number,omitempty"`
	Unit   string `json:"unit,omitempty"`
	Known  bool   `json:"known,omitempty"`
	// Line says the characters in pages; Notes are the facts under it.
	Line  string   `json:"line,omitempty"`
	Notes []string `json:"notes,omitempty"`
	// Week is the streak card's last seven days, today last.
	Week []insightsWeekdayViewState `json:"week,omitempty"`
	// Dial is how far the speed dial goes (0–1), Times the speed in
	// typing speeds ("4.6×"), TimesLabel what it compares with.
	Dial       float32 `json:"dial,omitempty"`
	Times      string  `json:"times,omitempty"`
	TimesLabel string  `json:"times_label,omitempty"`
	// Parts are the time saved; Bars typing's time and speaking's.
	Parts []insightsPart         `json:"parts,omitempty"`
	Bars  []insightsBarViewState `json:"bars,omitempty"`
	// Weeks and Months are the overview's and the days card's calendar
	// (insightsViewState).
	Weeks  [][7]int `json:"weeks,omitempty"`
	Months []string `json:"months,omitempty"`
}

// insightsDeckViewState is the cards' window: the card shown, how long
// it has been up (its entrance plays over Elapsed), and the side it
// came in from.
type insightsDeckViewState struct {
	Card    int                   `json:"card"`
	Count   int                   `json:"count"`
	Face    insightsFaceViewState `json:"face"`
	Elapsed time.Duration         `json:"elapsed"`
	// Slide is -1 when the card came from the left (Previous), 1 from
	// the right, 0 when the window opened on it.
	Slide  int  `json:"slide,omitempty"`
	Copied bool `json:"copied,omitempty"`
	// Posted is the site the card was just copied for (insightsSites):
	// the window says to paste it there.
	Posted string `json:"posted,omitempty"`
}

// buildInsightsFace is card kind of the log as of now, in theme.
func buildInsightsFace(translator *i18n.Translator, log activity.Log, now time.Time, kind string, theme int) insightsFaceViewState {
	ins := buildInsightsViewState(translator, log, now)
	stats := log.Stats(now)
	var chars, count, active int
	for _, day := range log.Days {
		chars += day.Chars
		count += day.Count
		if day.Count > 0 {
			active++
		}
	}
	face := insightsFaceViewState{Kind: kind, Theme: theme, Date: homeActivityDate(translator, now)}
	face.Move = max(slices.Index(insightsCardKinds, kind), 0) % 3
	switch kind {
	case insightsCardCharacters:
		face.Title = translator.T("Characters dictated")
		face.Number, face.Known = chars, true
		face.Unit = translator.N(chars, "character", "characters")
		face.Line = insightsPagesLine(translator, chars)
		face.Notes = []string{
			translator.N(count, "%d dictation", "%d dictations", count),
			translator.N(active, "%d active day", "%d active days", active),
		}
	case insightsCardStreak:
		face.Title = translator.T("Current streak")
		face.Number, face.Known = stats.Streak, true
		face.Unit = translator.N(stats.Streak, "day", "days")
		face.Week = insightsLastWeek(translator, log, now)
		face.Notes = []string{ins.Days[2].Label + " · " + insightsJoin(ins.Days[2].Parts)}
	case insightsCardSpeed:
		face.Title = translator.T("Average speed")
		face.Number, face.Known = stats.PerMinute, stats.PerMinute > 0
		face.Unit = translator.T("characters a minute")
		face.Times, face.TimesLabel = "—", translator.T("vs. typing")
		if face.Known {
			times := float64(stats.PerMinute) / insightsTypingRate
			face.Times = strings.TrimSuffix(fmt.Sprintf("%.1f", times), ".0") + "×"
			face.Dial = float32(min(1, times/insightsDialTimes))
		}
	case insightsCardSaved:
		face.Title = translator.T("Time saved")
		typing, audio := insightsTyping(log)
		face.Parts = ins.Stats[1].Parts
		// Nothing said yet: neither bar has a length.
		full, speaking := float32(0), float32(0)
		if typing > 0 && audio > 0 {
			full, speaking = 1, float32(min(1, float64(audio)/float64(typing)))
		}
		face.Bars = []insightsBarViewState{
			{Label: translator.T("Typing it"), Value: insightsJoin(insightsDuration(translator, typing)), Fraction: full},
			{Label: translator.T("Saying it"), Value: insightsJoin(insightsDuration(translator, audio)), Fraction: speaking},
		}
	case insightsCardDays:
		face.Title = translator.T("Active days")
		face.Stats = []insightsStatViewState{ins.Days[0], ins.Days[2]}
		face.Weeks, face.Months = ins.Weeks, ins.Months
	default:
		face.Kind = insightsCardHighlights
		face.Title = translator.T("Me and Voxt, lately")
		face.Stats, face.Days = ins.Stats, ins.Days
		face.Weeks, face.Months = ins.Weeks, ins.Months
	}
	return face
}

// insightsPagesLine says chars as pages of a book.
func insightsPagesLine(translator *i18n.Translator, chars int) string {
	switch pages := chars / insightsPageChars; {
	case chars >= insightsNovelChars:
		return translator.T("Enough for a whole novel")
	case pages >= 1:
		return translator.N(pages, "About %d page of a book", "About %d pages of a book", pages)
	}
	return translator.T("The first page is under way")
}

// insightsLastWeek is the seven days up to now's, each with its
// weekday's letter.
func insightsLastWeek(translator *i18n.Translator, log activity.Log, now time.Time) []insightsWeekdayViewState {
	names := []rune("SMTWTFS")
	if translator.Locale() == "zh-Hans" {
		names = []rune("日一二三四五六")
	}
	week := make([]insightsWeekdayViewState, 7)
	for i := range week {
		day := now.AddDate(0, 0, i-6)
		week[i] = insightsWeekdayViewState{Label: string(names[day.Weekday()]), Active: log.Get(day).Count > 0, Today: i == 6}
	}
	return week
}

// insightsJoin is parts on one line: "2 hours 34 minutes".
func insightsJoin(parts []insightsPart) string {
	words := make([]string, 0, 2*len(parts))
	for _, part := range parts {
		words = append(words, part.Value, part.Unit)
	}
	return strings.Join(words, " ")
}
