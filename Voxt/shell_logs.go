package main

// Logs: the newest lines of the app and of the helper, as a page (/logs)
// and as a panel under any page, filtered by source and text. (From
// AudioSmith.) Lines are copied and selected as shell_logs_select.go says.

import (
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

// logsLocalUI is the logs views' interface-local state: the selection,
// the find box, and bindings into the filters.
type logsLocalUI struct {
	sel    *logSelection
	find   *finder
	query  *string // the text filter
	source *string // the source filter's code
}

// logsPage is the Logs page.
func logsPage(c *ui.Context, state *logsViewState, actions logsActions, local *logsLocalUI) {
	pageHeader(c, i18n.T("Logs"), nil)
	logsView(c, state, actions, local, false)
}

// logsView shows the log; panel is the compact form under a page.
func logsView(c *ui.Context, state *logsViewState, actions logsActions, local *logsLocalUI, panel bool) {
	t := c.Theme()
	k := tok(t)
	lines := state.Lines
	q := strings.TrimSpace(state.Query)
	hits := findHits(q, len(lines), func(i int) []string { return []string{lines[i].Source, lines[i].Message} })
	if local.find.advance(hits[len(lines)]) {
		row, _ := rowOf(hits, local.find.cur)
		local.sel.views[viewOf(panel)].list.ScrollIntoView(row)
	}
	local.sel.prune(lines) // lines filtered out or gone from the log
	ui.Row(c).Padding(px(t, 6), px(t, sp3), px(t, 6), px(t, sp6)).Gap(px(t, 6)).AlignItems(ui.Center).Shrink(0).Children(func() {
		if panel {
			ui.Text(c, i18n.T("Logs")).FontWeight(600).FontSize(px(t, fsCaption)).SingleLine().Shrink(0)
		}
		var labels []string
		for _, s := range state.Sources {
			labels = append(labels, s.Label)
		}
		chosen := state.Source
		sel := ui.Select(c.Key("logs-source"), &chosen, labels).Label(i18n.T("Source")).MinWidth(px(t, 110)).Shrink(1)
		if sel.Changed() {
			for _, s := range state.Sources {
				if s.Label == chosen {
					*local.source = s.Code
				}
			}
		}
		local.find.keys(ui.TextInput(c.Key("logs-search"), local.query).Placeholder(i18n.T("Filter")).Width(px(t, 200)).Shrink(1).MinWidth(px(t, 80)).Label(i18n.T("Filter logs")))
		if hits[len(lines)] > 0 { // none: the list says nothing matches
			local.find.count(c, q, hits[len(lines)])
		}
		ui.Spacer(c)
		if n := local.sel.Count(); n > 0 {
			// While lines are selected: how many, Copy and Deselect.
			ui.Text(c, i18n.N(n, "%d line selected", "%d lines selected", n)).FontSize(px(t, fsMicro)).FontWeight(600).TextColor(k.AccentInk).SingleLine().Shrink(0)
			if quietButton(c.Key("logs-copy-selected"), icons.Copy, i18n.T("Copy")).Tooltip(i18n.N(n, "Copy %d selected line", "Copy %d selected lines", n)).Clicked() {
				actions.copyLogLines(local.sel.selectedSeqs(lines))
			}
			if quietButton(c.Key("logs-deselect"), icons.Close, i18n.T("Deselect")).Clicked() {
				local.sel.clearSelection()
			}
		} else {
			ui.Text(c, state.Count).FontSize(px(t, fsMicro)).TextColor(k.Text3).SingleLine()
			if iconButton(c.Key("logs-copy"), icons.Copy, i18n.T("Copy all lines shown")).Disabled(len(lines) == 0).Clicked() {
				actions.copyLogLines(logSeqs(lines))
			}
		}
		if state.LogFile {
			if iconButton(c.Key("logs-open"), icons.FolderOpen, i18n.T("Open log file")).Clicked() {
				actions.openLogFile()
			}
		}
		if panel {
			if iconButton(c.Key("logs-close"), icons.Close, i18n.T("Close")).Clicked() {
				actions.closeLogsPanel()
			}
		}
	})
	ui.Divider(c)
	if len(lines) == 0 {
		// Compact, not emptyState: the panel can be 120 points high.
		ui.Column(c).Grow(1).Center().Gap(px(t, sp2)).Children(func() {
			q := strings.TrimSpace(state.Query)
			if q == "" {
				ui.Text(c, i18n.T("No log lines yet.")).TextColor(t.TextMuted)
				return
			}
			if r := []rune(q); len(r) > 40 {
				q = string(r[:40]) + "…"
			}
			ui.Text(c, i18n.T("Nothing matches “%s”", q)).TextColor(t.TextMuted).TextAlign(ui.Center)
			if ghostButton(c.Key("logs-clear-filter"), i18n.T("Clear filter")).Clicked() {
				*local.query = ""
			}
		})
		return
	}
	logRows(c, lines, viewOf(panel), q, hits, actions, local)
}

// logRows are the lines, the newest first, in a list that keeps its place
// as lines come (shell_logs_order.go) and that the pointer selects
// (shell_logs_select.go): the line under it lit, the lines selected in the
// accent's tint. The matches of the filter q are marked (hits: findHits
// over each line's source and message).
func logRows(c *ui.Context, lines []logLineViewState, v logView, q string, hits []int, actions logsActions, local *logsLocalUI) {
	t := c.Theme()
	k := tok(t)
	s := local.sel
	o := &s.views[v]
	logsFollow(o, lines) // before the list: where it goes
	// A drag at the list's edge scrolls it (before the list, which takes
	// the offset as it is built).
	edge := s.Dragging() && logsAutoScroll(t, s.drag.y, s.drag.h, &o.scroll)
	pressing := s.Pressing()
	s.spans = s.spans[:0]
	cur, curAt := rowOf(hits, local.find.cur)
	ui.Column(c.Key("logs-area")).Grow(1).MinHeight(0).Children(func() {
		list := ui.List(c.Key("logs-rows"), &o.list, len(lines), func(i int) {
			e := lines[i]
			color := t.Text
			switch e.Level {
			case logLevelError:
				color = t.Danger
			case logLevelWarning:
				color = t.Warning
			case logLevelDebug:
				color = k.Text3
			}
			row := ui.Row(c).Gap(px(t, 8)).Padding(px(t, 1), px(t, 4)).Radius(px(t, 4))
			// The hover waits while the pointer presses: a drag shows
			// what it selects.
			hovered := !pressing && row.Hovered()
			switch selected := s.IsSelected(e.Seq); {
			case selected && hovered:
				row.Background(k.SelectedHover)
			case selected:
				row.Background(k.Selected)
			case hovered:
				row.Background(k.Hover)
			}
			if b := row.Bounds(); b.H > 0 { // in the window, until the list's box is known
				s.spans = append(s.spans, logSpan{seq: e.Seq, top: b.Y, bottom: b.Y + b.H})
			}
			logRowMenu(c, row, e, lines, actions, s)
			at := -1
			if i == cur {
				at = curAt
			}
			row.Children(func() {
				ui.Text(c, e.Time).Font("monospace").FontSize(px(t, fsMicro)).TextColor(k.Text3).Shrink(0)
				markedText(c, e.Source, q, curIn(at, 0)).Font("monospace").FontSize(px(t, fsMicro)).TextColor(k.Text2).Width(px(t, 80)).Shrink(0).SingleLine().Ellipsis("…")
				// Not Selectable: the pointer chooses lines, not characters.
				markedText(c, e.Message, q, curIn(at, len(findFolded(e.Source, q)))).Font("monospace").FontSize(px(t, 11)).TextColor(color).Grow(1).MinWidth(0)
			})
		})
		list.TrackScroll(&o.scroll).Grow(1).MinHeight(0).Padding(px(t, 5), px(t, sp3-4), px(t, 5), px(t, sp6-4)).Focusable().Label(i18n.T("Log lines"))
		// The rows' spans against the list's box, as on screen (the last
		// frame's, as the rows'): the pointer's events come in its box.
		origin := list.Bounds()
		s.origin = origin
		for i := range s.spans {
			s.spans[i].top -= origin.Y
			s.spans[i].bottom -= origin.Y
		}
		list.HandleInput(logsInput(s, lines, actions))
		logsKeys(c, list, lines, actions, s)
		newLinesPill(c, o, lines) // over the list's top
		changed := logsResolve(c, list, lines, actions, s)
		if changed || edge {
			c.AnimationFrame() // show the selection the events made
		}
	})
}
