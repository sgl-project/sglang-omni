package main

// The insights cards' faces, 600×400 DIPs each: the highlights, the
// characters, the streak, the speed, the time saved and the days. Each
// has its own layout, theme (insights_themes.go) and entrance, which
// plays over the time the card has been up: numbers count up, the dial
// and the bars grow from nothing, the week's days and the calendar's
// squares come one after another. A copied picture is the card settled.

import (
	"image"
	"math"
	"strconv"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

const (
	insightsCardW, insightsCardH = 600, 400
	insightsPad                  = 28
	insightsLockupH              = 28
	// insightsEntrance is how long a card's entrance takes at most:
	// frames keep coming until then.
	insightsEntrance = 1800 * time.Millisecond
)

// insightsMS is the entrances' unit.
const insightsMS = time.Millisecond

// renderInsightsCard is the card settled, at 3× (1800×1200 pixels): what
// Copy copies and Save saves. Its corners are square, as renderShareCard's.
func renderInsightsCard(face *insightsFaceViewState) image.Image {
	return ui.Render(func(c *ui.Context) { insightsCardFace(c, face, insightsSettled, false) }, insightsCardW, insightsCardH, 3)
}

// insightsCardFace draws face as it is elapsed after it came up; rounded
// for the window, square for a picture.
func insightsCardFace(c *ui.Context, face *insightsFaceViewState, elapsed time.Duration, rounded bool) {
	radius := float32(shareCardRadius)
	if !rounded {
		radius = 0
	}
	inks := insightsThemeInks(face.Theme)
	ui.Box(c.Key("insights-card")).Size(insightsCardW, insightsCardH).Radius(radius).Clip().Children(func() {
		insightsBackdrop(c, face)
		body := ui.Box(c.Key("insights-"+face.Kind)).Absolute().Left(0).Top(0).Size(insightsCardW, insightsCardH)
		body.Children(func() {
			switch face.Kind {
			case insightsCardCharacters:
				insightsCharactersCard(c, face, inks, elapsed)
			case insightsCardStreak:
				insightsStreakCard(c, face, inks, elapsed)
			case insightsCardSpeed:
				insightsSpeedCard(c, face, inks, elapsed)
			case insightsCardSaved:
				insightsSavedCard(c, face, inks, elapsed)
			case insightsCardDays:
				insightsDaysCard(c, face, inks, elapsed)
			default:
				insightsHighlightsCard(c, face, inks, elapsed)
			}
		})
	})
}

// insightsFoot is the card's foot: the lockup on one side, the day on
// the other.
func insightsFoot(c *ui.Context, face *insightsFaceViewState, inks insightsInks, elapsed time.Duration, lockupRight bool) {
	ui.Row(c.Key("insights-foot")).Absolute().Left(insightsPad).Top(insightsCardH-24-insightsLockupH).
		Size(insightsCardW-2*insightsPad, insightsLockupH).AlignItems(ui.Center).Children(func() {
		date := func() {
			ui.Text(c, face.Date).FontSize(13).FontWeight(500).TextColor(inks.Soft).FontFeatures("tnum").SingleLine()
		}
		if lockupRight {
			date()
			ui.Spacer(c)
			insightsLockup(c, insightsLockupH, face.Move, elapsed)
			return
		}
		insightsLockup(c, insightsLockupH, face.Move, elapsed)
		ui.Spacer(c)
		date()
	})
}

// insightsTitle is a card's title on its background.
func insightsTitle(c *ui.Context, title string, inks insightsInks) ui.Element {
	return ui.Text(c, title).FontSize(27).FontWeight(700).TextColor(inks.Title).SingleLine()
}

// insightsLabel is a card's title as a small label over a large number.
func insightsLabel(c *ui.Context, title string, inks insightsInks) ui.Element {
	return ui.Text(c, title).FontSize(17).FontWeight(600).TextColor(inks.Title.Alpha(.86)).SingleLine()
}

// insightsRise fades e in and lifts it into place as progress goes to 1.
func insightsRise(e ui.Element, progress float32) ui.Element {
	return e.Opacity(progress).Translate(0, 12*(1-progress))
}

// insightsHighlightsCard is the overview: the four totals in tiles, two
// by two, over the active days and the streaks beside the last weeks'
// calendar. Each tile rises after the one before, and the calendar's
// weeks come one after another.
func insightsHighlightsCard(c *ui.Context, face *insightsFaceViewState, inks insightsInks, elapsed time.Duration) {
	ui.Column(c.Key("body")).Absolute().Left(0).Top(0).Size(insightsCardW, insightsCardH).Padding(22, insightsPad).Gap(10).Children(func() {
		insightsTitle(c, face.Title, inks).Margin(0, 0, 4, 0)
		for row := 0; row < 2; row++ {
			ui.Row(c.Key("row-" + strconv.Itoa(row))).Gap(10).Children(func() {
				for col := 0; col < 2; col++ {
					i := row*2 + col
					if i >= len(face.Stats) {
						continue
					}
					tile := insightsTile(c.Key(i), inks).Grow(1).Basis(0).Height(70).Padding(0, 20).Gap(3).Justify(ui.Center)
					insightsRise(tile, insightsProgress(elapsed, time.Duration(120+70*i)*insightsMS, 420*insightsMS)).Children(func() {
						insightsParts(c, face.Stats[i].Parts, 24, 13, inks.Ink, inks.Ink)
						ui.Text(c, face.Stats[i].Label).FontSize(12).TextColor(inks.Muted).SingleLine()
					})
				}
			})
		}
		days := insightsTile(c.Key("days"), inks).Height(98).Padding(0, 20)
		insightsRise(days, insightsProgress(elapsed, 400*insightsMS, 420*insightsMS)).Children(func() {
			ui.Row(c.Key("days-row")).Grow(1).Gap(24).AlignItems(ui.Center).Children(func() {
				for i, stat := range face.Days {
					ui.Column(c.Key(i)).Gap(2).Shrink(0).Children(func() {
						insightsParts(c, stat.Parts, 22, 12, inks.Ink, inks.Ink)
						ui.Text(c, stat.Label).FontSize(11).TextColor(inks.Muted).SingleLine()
					})
				}
				ui.Spacer(c)
				insightsCalendar(c, face, inks, elapsed, 9, 3, 2.5, false)
			})
		})
	})
	insightsFoot(c, face, inks, elapsed, false)
}

// insightsCalendar is the last weeks as squares of cell DIPs, a column a
// week, shaded by each day's results; its weeks come one after another.
// months writes the months under it.
func insightsCalendar(c *ui.Context, face *insightsFaceViewState, inks insightsInks, elapsed time.Duration, cell, gap, radius float32, months bool) {
	ui.Column(c.Key("weeks")).Gap(7).Shrink(0).Children(func() {
		ui.Row(c.Key("grid")).Gap(gap).Children(func() {
			for w, week := range face.Weeks {
				show := insightsProgress(elapsed, time.Duration(450+45*w)*insightsMS, 320*insightsMS)
				ui.Column(c.Key(w)).Gap(gap).Opacity(show).Translate(0, 8*(1-show)).Children(func() {
					for d, level := range week {
						square := ui.Box(c.Key(d)).Size(cell, cell).Radius(radius)
						switch {
						case level < 0:
						case level == 0:
							square.Background(inks.Track)
						default:
							square.Background(inks.Accent.Alpha(.3 + .7*float32(level)/4))
						}
					}
				})
			}
		})
		if !months {
			return
		}
		ui.Row(c.Key("months")).Gap(gap).Children(func() {
			for w, month := range face.Months {
				ui.Text(c.Key(w), month).FontSize(10).TextColor(inks.Muted).Width(cell).NoWrap()
			}
		})
	})
}

// insightsCharactersCard is the characters dictated, counted up, large on
// the background, over what they come to in pages.
func insightsCharactersCard(c *ui.Context, face *insightsFaceViewState, inks insightsInks, elapsed time.Duration) {
	count := insightsProgress(elapsed, 150*insightsMS, 1000*insightsMS)
	size := float32(112)
	if face.Number >= 1000000 {
		size = 88
	}
	ui.Column(c.Key("body")).Absolute().Left(0).Top(0).Size(insightsCardW, insightsCardH).Padding(30, insightsPad+4).Children(func() {
		insightsLabel(c, face.Title, inks)
		ui.Row(c.Key("number")).Gap(12).AlignItems(ui.End).Margin(2, 0, 0, 0).Children(func() {
			ui.Text(c, insightsCount(face.Number, count)).FontSize(size).FontWeight(800).TextColor(shareWhite).
				FontFeatures("tnum").LineHeight(1.1).SingleLine()
			ui.Text(c, face.Unit).FontSize(22).FontWeight(600).TextColor(inks.Title.Alpha(.86)).SingleLine().Margin(0, 0, size*.2, 0)
		})
		pop := insightsPop(elapsed, 750*insightsMS, 380*insightsMS)
		show := insightsProgress(elapsed, 750*insightsMS, 200*insightsMS)
		ui.Row(c.Key("line")).Radius(999).Background(shareWhite.Alpha(.95)).Padding(9, 18).AlignSelf(ui.Start).Margin(10, 0, 0, 0).
			TransformOrigin(0, .5).Scale(.8+.2*pop, .8+.2*pop).Opacity(show).Children(func() {
			ui.Text(c, face.Line).FontSize(16).FontWeight(600).TextColor(ui.Hex("#1E1B2E")).SingleLine()
		})
		notes := insightsProgress(elapsed, 1000*insightsMS, 400*insightsMS)
		ui.Row(c.Key("notes")).Gap(18).Margin(16, 0, 0, 2).Opacity(notes).Children(func() {
			for i, note := range face.Notes {
				ui.Text(c.Key(i), note).FontSize(14).FontWeight(500).TextColor(inks.Soft).SingleLine()
			}
		})
	})
	insightsFoot(c, face, inks, elapsed, true)
}

// insightsStreakCard is the streak's days, large, over the last seven
// days, which pop in one by one. Its lockup is at the top.
func insightsStreakCard(c *ui.Context, face *insightsFaceViewState, inks insightsInks, elapsed time.Duration) {
	ui.Row(c.Key("head")).Absolute().Left(insightsPad).Top(24).Size(insightsCardW-2*insightsPad, insightsLockupH).AlignItems(ui.Center).Children(func() {
		insightsLockup(c, insightsLockupH, face.Move, elapsed)
		ui.Spacer(c)
		ui.Text(c, face.Date).FontSize(13).FontWeight(500).TextColor(inks.Soft).FontFeatures("tnum").SingleLine()
	})
	count := insightsProgress(elapsed, 150*insightsMS, 800*insightsMS)
	ui.Column(c.Key("body")).Absolute().Left(insightsPad + 4).Top(82).Children(func() {
		insightsLabel(c, face.Title, inks)
		ui.Row(c.Key("number")).Gap(12).AlignItems(ui.End).Children(func() {
			ui.Text(c, insightsCount(face.Number, count)).FontSize(120).FontWeight(800).TextColor(shareWhite).
				FontFeatures("tnum").LineHeight(1.1).SingleLine()
			ui.Text(c, face.Unit).FontSize(28).FontWeight(700).TextColor(inks.Title.Alpha(.9)).SingleLine().Margin(0, 0, 25, 0)
		})
	})
	tile := insightsTile(c.Key("week"), inks).Absolute().Left(insightsPad).Top(insightsCardH-insightsPad-92).
		Size(insightsCardW-2*insightsPad, 92).Padding(0, 22)
	tile.Children(func() {
		ui.Row(c.Key("week-row")).Grow(1).AlignItems(ui.Center).Children(func() {
			insightsWeekDots(c, face, inks, elapsed)
			ui.Spacer(c)
			ui.Column(c.Key("notes")).Gap(4).AlignItems(ui.End).Opacity(insightsProgress(elapsed, 900*insightsMS, 400*insightsMS)).Children(func() {
				for i, note := range face.Notes {
					ui.Text(c.Key(i), note).FontSize(13).FontWeight(600).TextColor(inks.Ink).SingleLine()
				}
			})
		})
	})
}

// insightsWeekDots are the last seven days, today last: a day with a
// result is a tick on its part of the card's gradient, which runs across
// the row.
func insightsWeekDots(c *ui.Context, face *insightsFaceViewState, inks insightsInks, elapsed time.Duration) {
	n := float32(max(len(face.Week), 1))
	ui.Row(c.Key("dots")).Gap(12).Children(func() {
		for i, day := range face.Week {
			pop := insightsPop(elapsed, time.Duration(350+70*i)*insightsMS, 360*insightsMS)
			ui.Column(c.Key(i)).Gap(6).AlignItems(ui.Center).Children(func() {
				dot := ui.Box(c.Key("dot")).Size(34, 34).Radius(17).Center().Scale(pop, pop)
				switch {
				case day.Active:
					dot.Gradient(inks.at(float32(i)/n), inks.at(float32(i+1)/n), 90)
				case day.Today:
					dot.Border(2, inks.at(1).Alpha(.75))
				default:
					dot.Background(inks.Track)
				}
				dot.Children(func() {
					if day.Active {
						ui.Icon(c, icons.ShareCopied.Linear).Size(18, 18).TextColor(inks.On)
					}
				})
				weight, color := 500, inks.Muted
				if day.Today {
					weight, color = 700, inks.Ink
				}
				ui.Text(c, day.Label).FontSize(11).FontWeight(weight).TextColor(color).SingleLine()
			})
		}
	})
}

// insightsArc is an arc of radius r around (cx, cy), from degrees start
// (0 points right, 90 down) over sweep degrees clockwise.
func insightsArc(cx, cy, r float32, start, sweep float64) *ui.Path {
	var path ui.Path
	steps := max(8, int(math.Abs(sweep)/3))
	for i := 0; i <= steps; i++ {
		a := (start + sweep*float64(i)/float64(steps)) * math.Pi / 180
		x, y := cx+r*float32(math.Cos(a)), cy+r*float32(math.Sin(a))
		if i == 0 {
			path.MoveTo(x, y)
		} else {
			path.LineTo(x, y)
		}
	}
	return &path
}

// insightsSpeedCard is the speed beside a dial that fills from nothing
// to it, eight typing speeds being the dial's end.
func insightsSpeedCard(c *ui.Context, face *insightsFaceViewState, inks insightsInks, elapsed time.Duration) {
	grow := insightsProgress(elapsed, 200*insightsMS, 1200*insightsMS)
	ui.Column(c.Key("body")).Absolute().Left(0).Top(0).Size(insightsCardW, insightsCardH).Padding(26, insightsPad).Children(func() {
		insightsTitle(c, face.Title, inks)
		insightsTile(c.Key("tile"), inks).Margin(18, 0, 0, 0).Height(212).Padding(0, 30).Children(func() {
			ui.Row(c.Key("row")).Grow(1).AlignItems(ui.Center).Children(func() {
				ui.Column(c.Key("number")).Gap(2).Children(func() {
					number, weight := "—", 300 // a dash, light, before any speech
					if face.Known {
						number, weight = insightsCount(face.Number, grow), 800
					}
					ui.Text(c, number).FontSize(88).FontWeight(weight).TextColor(inks.Ink).FontFeatures("tnum").LineHeight(1.1).SingleLine()
					ui.Text(c, face.Unit).FontSize(15).FontWeight(600).TextColor(inks.Muted).LetterSpacing(.4).SingleLine()
				})
				ui.Spacer(c)
				insightsDial(c, face, inks, grow)
			})
		})
	})
	insightsFoot(c, face, inks, elapsed, false)
}

// insightsDial is the speed's dial, 208×150: a track of 240°, the part
// of it the speed fills at progress in the card's gradient, a knob at
// its end, and the speed in typing speeds inside.
func insightsDial(c *ui.Context, face *insightsFaceViewState, inks insightsInks, progress float32) {
	const w, h, r, stroke, sweep = 208, 150, 88, 16, 240.0
	const start = 90 + (360-sweep)/2
	filled := float64(face.Dial * progress)
	ui.Box(c.Key("dial")).Size(w, h).Shrink(0).Draw(func(p *ui.Painter, rect ui.Rect) {
		u := rect.W / w
		cx, cy := rect.X+w/2*u, rect.Y+(r+stroke/2+2)*u
		p.StrokePath(insightsArc(cx, cy, r*u, start, sweep), stroke*u, inks.Track)
		if filled <= 0 {
			return
		}
		p.StrokePathGradient(insightsArc(cx, cy, r*u, start, sweep*filled), stroke*u, inks.gradient(90))
		end := (start + sweep*filled) * math.Pi / 180
		var knob ui.Path
		knob.Circle(cx+r*u*float32(math.Cos(end)), cy+r*u*float32(math.Sin(end)), 5*u)
		p.FillPath(&knob, inks.On)
	}).Children(func() {
		ui.Column(c.Key("times")).Absolute().Left(0).Top(58).Width(w).AlignItems(ui.Center).Gap(1).Opacity(min(1, progress*2)).Children(func() {
			ui.Text(c, face.Times).FontSize(34).FontWeight(800).TextColor(inks.Ink).FontFeatures("tnum").SingleLine()
			ui.Text(c, face.TimesLabel).FontSize(13).FontWeight(500).TextColor(inks.Muted).SingleLine()
		})
	})
}

// insightsSavedCard is the time saved, large, over two bars that grow:
// how long typing it would have taken, and how long saying it did.
func insightsSavedCard(c *ui.Context, face *insightsFaceViewState, inks insightsInks, elapsed time.Duration) {
	ui.Column(c.Key("body")).Absolute().Left(0).Top(0).Size(insightsCardW, insightsCardH).Padding(28, insightsPad).Children(func() {
		insightsLabel(c, face.Title, inks).Margin(0, 0, 0, 4)
		ui.Row(c.Key("saved")).Margin(2, 0, 0, 4).Children(func() {
			insightsParts(c, face.Parts, 64, 22, shareWhite, inks.Title.Alpha(.86))
		})
		insightsTile(c.Key("tile"), inks).Margin(14, 0, 0, 0).Padding(20, 24).Gap(16).Children(func() {
			for i, bar := range face.Bars {
				grow := insightsProgress(elapsed, time.Duration(250+180*i)*insightsMS, 900*insightsMS)
				ui.Column(c.Key(i)).Gap(7).Children(func() {
					ui.Row(c.Key("text")).Children(func() {
						ui.Text(c, bar.Label).FontSize(13).FontWeight(600).TextColor(inks.Muted).SingleLine()
						ui.Spacer(c)
						ui.Text(c, bar.Value).FontSize(13).FontWeight(700).TextColor(inks.Ink).FontFeatures("tnum").SingleLine()
					})
					fraction := bar.Fraction * grow
					ui.Box(c.Key("bar")).Height(18).Draw(func(p *ui.Painter, r ui.Rect) {
						p.Fill(r, inks.Track, r.H/2)
						filled := ui.Rect{X: r.X, Y: r.Y, W: max(r.H, r.W*fraction), H: r.H}
						switch {
						case fraction <= 0:
						case i == 0: // typing: plain
							p.Fill(filled, inks.Ink.Alpha(.3), r.H/2)
						default:
							p.FillGradient(filled, inks.gradient(90), r.H/2)
						}
					})
				})
			}
		})
	})
	insightsFoot(c, face, inks, elapsed, true)
}

// insightsDaysCard is the last weeks' calendar, its weeks coming one
// after another, beside the active days and the longest streak.
func insightsDaysCard(c *ui.Context, face *insightsFaceViewState, inks insightsInks, elapsed time.Duration) {
	const cell, gap = 17, 4
	ui.Column(c.Key("body")).Absolute().Left(0).Top(0).Size(insightsCardW, insightsCardH).Padding(26, insightsPad).Children(func() {
		insightsTitle(c, face.Title, inks)
		insightsTile(c.Key("tile"), inks).Margin(18, 0, 0, 0).Height(212).Padding(0, 26).Children(func() {
			ui.Row(c.Key("row")).Grow(1).AlignItems(ui.Center).Children(func() {
				insightsCalendar(c, face, inks, elapsed, cell, gap, 4.5, true)
				ui.Spacer(c)
				ui.Column(c.Key("stats")).Gap(20).Shrink(0).Children(func() {
					for i, stat := range face.Stats {
						insightsRise(ui.Column(c.Key(i)).Gap(2), insightsProgress(elapsed, time.Duration(500+140*i)*insightsMS, 420*insightsMS)).Children(func() {
							insightsParts(c, stat.Parts, 38, 15, inks.Ink, inks.Ink)
							ui.Text(c, stat.Label).FontSize(13).TextColor(inks.Muted).SingleLine()
						})
					}
				})
			})
		})
	})
	insightsFoot(c, face, inks, elapsed, false)
}
