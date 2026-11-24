package main

import "time"

// The mascot's outfits change by themselves with the calendar: every
// outfit Voxt ships has the days it is worn, so a holiday's clothes come
// on that day without an update or the network.

const (
	outfitChristmas = "christmas"
	outfitNewYear   = "new-year"
	outfitSpring    = "spring-festival"
)

// mascotSeason is the days of the year an outfit is worn, from From to To
// with both included; To before From wraps into the next year.
type mascotSeason struct {
	outfit   string
	from, to monthDay
}

type monthDay struct {
	month time.Month
	day   int
}

func (d monthDay) before(o monthDay) bool {
	return d.month < o.month || d.month == o.month && d.day < o.day
}

// mascotSeasons are the outfits worn on the same days every year.
var mascotSeasons = []mascotSeason{
	{outfitChristmas, monthDay{time.December, 15}, monthDay{time.December, 26}},
	{outfitNewYear, monthDay{time.December, 31}, monthDay{time.January, 2}},
}

// springFestival is the first day of the lunar year, which moves.
var springFestival = map[int]monthDay{
	2027: {time.February, 6},
	2028: {time.January, 26},
	2029: {time.February, 13},
	2030: {time.February, 3},
	2031: {time.January, 23},
}

// mascotOutfitOn is the outfit the mascot wears on now's day, "" for none:
// the Spring Festival from its eve for a week, else a season of the year.
func mascotOutfitOn(now time.Time) string {
	day := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, time.UTC)
	if first, ok := springFestival[now.Year()]; ok {
		start := time.Date(now.Year(), first.month, first.day, 0, 0, 0, 0, time.UTC).AddDate(0, 0, -1)
		if !day.Before(start) && day.Before(start.AddDate(0, 0, 8)) {
			return outfitSpring
		}
	}
	today := monthDay{now.Month(), now.Day()}
	for _, s := range mascotSeasons {
		inside := !today.before(s.from) && !s.to.before(today)
		if s.to.before(s.from) { // wraps into the next year
			inside = !today.before(s.from) || !s.to.before(today)
		}
		if inside {
			return s.outfit
		}
	}
	return ""
}
