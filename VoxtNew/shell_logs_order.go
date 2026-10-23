package main

// The logs' order (the family's log-selection spec, "Order"): the newest
// line first, at the top, so that the view opens on what just happened.
// A list showing its top follows the lines that come: they appear there.
// A list the user scrolled down, reading older lines, keeps them where
// they are as lines come above (ListState keeps its place by the line at
// its top, keyed by Seq), and a pill over its top says how many came since
// it last showed its top; a click goes back there.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
)

// logView is which list: the Logs page's or the panel's, each with its own
// place.
type logView int

const (
	logPage logView = iota
	logPanel
)

func viewOf(panel bool) logView {
	if panel {
		return logPanel
	}
	return logPage
}

// logListView is a list's place.
type logListView struct {
	list ui.ListState
	// scroll is the list's offset (TrackScroll): 0 at the top. A drag
	// near the edges moves it (logsAutoScroll).
	scroll ui.ScrollState
	// seen is the newest line the list showed at its top: the lines
	// after it are new to the user.
	seen uint64
}

// logAtTop is how far from the top (DIPs) the list still shows it.
const logAtTop = 1

// logsFollow keys the list's rows by their lines and, when the list
// showed its top in the last frame, keeps it there as new lines come.
// Before the list is built.
func (a *app) logsFollow(o *logListView, lines []core.LogEntry) {
	o.list.Key = func(i int) any { return lines[i].Seq }
	if len(lines) == 0 || o.scroll.Y > logAtTop {
		return
	}
	if newest := lines[0].Seq; newest > o.seen {
		// The list keeps its place by the line that was at its top, now
		// under the new ones: show them.
		o.list.ScrollTo(0, ui.Start)
		o.seen = newest
	}
}

// newLines is how many lines came since the list last showed its top.
func newLines(o *logListView, lines []core.LogEntry) int {
	n := 0
	for _, e := range lines {
		if e.Seq <= o.seen {
			break
		}
		n++
	}
	return n
}

// newLinesPill shows, over the top of a list scrolled down, how many lines
// came since it showed its top; a click scrolls back up. Inside the
// list's area, after the list.
func (a *app) newLinesPill(c *ui.Context, o *logListView, lines []core.LogEntry) {
	n := newLines(o, lines)
	if n == 0 || o.scroll.Y <= logAtTop {
		return
	}
	t := c.Theme()
	k := tok(t)
	label := i18n.N(n, "%d new line", "%d new lines", n)
	// The row spans the list's width to centre the pill, and lets the
	// pointer through to the lines beside it.
	ui.Row(c).Absolute().Top(px(t, 8)).Left(0).Right(0).Justify(ui.Center).PassThrough().Children(func() {
		b := ui.ButtonBase(c.Key("logs-new-lines")).Padding(px(t, 4), px(t, 10)).Gap(px(t, 4)).AlignItems(ui.Center).
			Radius(px(t, 12)).Background(k.Ink).Label(label).Tooltip(i18n.T("Show the newest lines"))
		if b.Hovered() {
			b.Background(k.InkHover)
		}
		b.Children(func() {
			ui.Icon(c, icons.AltArrowUp.Linear).Size(px(t, 12), px(t, 12)).TextColor(k.OnInk)
			ui.Text(c, label).FontSize(px(t, fsMicro)).FontWeight(600).TextColor(k.OnInk).SingleLine()
		})
		if b.Clicked() {
			o.list.ScrollTo(0, ui.Start)
			o.seen = lines[0].Seq
		}
	})
}
