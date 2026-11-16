package main

// Home (/home): a slogan with Transcribe's shortcut in a key
// cap; what is still missing before VoxtNew can work (a speech model,
// a language model, the Microphone, Accessibility, the shortcut), each
// line the way to fix it, and nothing once all is ready; until VoxtNew
// has given a result once, a Try area that runs a whole transcription on
// the page (Start, or the shortcut pressed while VoxtNew is in front);
// today's count of results and characters; the last five results, to
// copy or open in History.

import (
	"strings"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

// homeLocalUI is the visit-local state shared with Home's Try card.
type homeLocalUI struct {
	block *sessionNotice
	tryOn bool
}

func homePage(c *ui.Context, state *homeViewState, actions homeActions, local *historyLocalUI, tryCard func(*ui.Context)) {
	t := c.Theme()
	if state.TryRecording {
		c.After(250 * time.Millisecond) // the Try area's clock
	}
	columnPageHeader(c, i18n.T("Home"), func() {
		if ghostIconButton(c.Key("home-guide"), icons.Compass, i18n.T("Setup guide")).Label(i18n.T("Open setup guide")).Clicked() {
			actions.openHomeGuide()
		}
	})
	ui.Scroll(c.Key("home-scroll")).Grow(1).MinHeight(0).Children(func() {
		pageColumn(c).Gap(px(t, sp3)).Children(func() {
			homeSlogan(c, state.Slogan, actions)
			homeReadiness(c, state, actions)
			if state.ShowTry && state.TryAvailable && tryCard != nil {
				sectionTitle(c, i18n.T("Try it"))
				tryCard(c)
			}
			homeToday(c, state)
			homeRecentList(c, state, actions, local)
		})
	})
}

// ---- The slogan ----

// homeSlogan is the line under Home's title: what VoxtNew does, then how
// to start it (Transcribe's shortcut in a key cap, or a link to add one).
func homeSlogan(c *ui.Context, state homeSloganViewState, actions homeActions) {
	t := c.Theme()
	k := tok(t)
	text := func(s string, col ui.Color) {
		if s = strings.TrimSpace(s); s != "" {
			ui.Text(c, s).FontSize(px(t, fsBody)).TextColor(col).SingleLine()
		}
	}
	ui.Row(c.Key("home-slogan")).Wrap().GapX(px(t, 6)).GapY(px(t, sp1)).AlignItems(ui.Center).Children(func() {
		text(i18n.T("Speak anywhere and Voxt types at your cursor"), k.Text)
		text("·", k.Text3)
		if state.Keys == "" {
			l := ui.ButtonBase(c.Key("home-slogan-add")).Label(state.Before)
			l.Children(func() {
				lt := ui.Text(c, state.Before).FontSize(px(t, fsBody)).TextColor(k.AccentInk).SingleLine()
				if l.Hovered() {
					lt.Underline()
				}
			})
			if l.Clicked() {
				actions.addHomeShortcut()
			}
			return
		}
		// The hint's words and its key cap, in one piece that wraps whole.
		ui.Row(c).Gap(px(t, 6)).AlignItems(ui.Center).Shrink(0).Children(func() {
			text(state.Before, k.Text2)
			keycap(c, state.Keys)
			text(state.After, k.Text2)
		})
	})
}

// homeToday is today's numbers.
func homeToday(c *ui.Context, state *homeViewState) {
	t := c.Theme()
	k := tok(t)
	sectionTitle(c, i18n.T("Today"))
	if !state.KeepHistory {
		caption(c, i18n.T("History is off, so nothing is counted."))
		return
	}
	tile := func(key, value, label string) {
		fillCard(c.Key(key)).Gap(px(t, 2)).Grow(1).Basis(px(t, 140)).MinWidth(px(t, 120)).Padding(px(t, sp3), px(t, sp4)).Children(func() {
			ui.Text(c, value).FontSize(px(t, fsTitle)).FontWeight(600).SingleLine()
			ui.Text(c, label).FontSize(px(t, fsCaption)).TextColor(k.Text2).SingleLine().Ellipsis("…").MinWidth(0)
		})
	}
	ui.Row(c.Key("home-today")).Wrap().Gap(px(t, sp3)).Children(func() {
		tile("home-count", state.Today.Count, state.Today.CountLabel)
		tile("home-chars", state.Today.Characters, state.Today.CharactersLabel)
		// By feature: each one's icon and count.
		fillCard(c.Key("home-kinds")).Gap(px(t, 2)).Grow(1).Basis(px(t, 140)).MinWidth(px(t, 120)).Padding(px(t, sp3), px(t, sp4)).Children(func() {
			ui.Row(c).Gap(px(t, sp3)).AlignItems(ui.Center).Height(px(t, fsTitle*1.3)).Children(func() {
				for _, kind := range state.Today.Kinds {
					ui.Row(c.Key("kind-" + kind.Kind)).Gap(px(t, 4)).AlignItems(ui.Center).Label(kind.Title + " " + kind.Count).Tooltip(kind.Title).Children(func() {
						ui.Icon(c, homeKindIcon(kind.Kind).Linear).Size(px(t, 16), px(t, 16)).TextColor(k.Text2)
						ui.Text(c, kind.Count).FontSize(px(t, fsSection)).FontWeight(600).SingleLine()
					})
				}
			})
			ui.Text(c, i18n.T("By feature")).FontSize(px(t, fsCaption)).TextColor(k.Text2).SingleLine().Ellipsis("…").MinWidth(0)
		})
	})
}

func homeKindIcon(kind string) icons.Icon {
	switch kind {
	case "translate":
		return translateIcon
	case "rewrite":
		return rewriteIcon
	}
	return transcribeIcon
}

// homeRecentList is the last results: each copies, or opens in History;
// its context menu copies, opens or deletes it.
func homeRecentList(c *ui.Context, state *homeViewState, actions homeActions, local *historyLocalUI) {
	t := c.Theme()
	k := tok(t)
	// No margin above, unlike sectionTitle's: Open History's face makes
	// the row taller than its title, which keeps Today's distance.
	ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
		sectionTitleText(c, i18n.T("Recent results")).Grow(1)
		if ghostButton(c.Key("home-history"), i18n.T("Open History")).Clicked() {
			actions.openHomeHistory()
		}
	})
	if state.Error != "" {
		notice(c, "home-err", noticeError, i18n.T("Could not read the history."), state.Error, nil, nil)
		return
	}
	if len(state.Recent) == 0 {
		if state.Read {
			caption(c, i18n.T("Results appear here after you use a shortcut."))
		}
		return
	}
	c.After(time.Minute) // the times are relative
	card(c.Key("home-recent")).Padding(px(t, sp1)).Gap(0).Children(func() {
		for _, recent := range state.Recent {
			rowState := recent.Row
			row := ui.ButtonBase(c.Key("home-recent-"+rowState.ID)).Padding(px(t, sp2), px(t, sp3)).Radius(px(t, radiusControl)).
				Gap(px(t, sp3)).AlignItems(ui.Center).Label(rowState.Title).Tooltip(i18n.T("Open in History"))
			if row.Hovered() {
				row.Background(k.Hover)
			}
			row.Children(func() {
				ui.Icon(c, homeKindIcon(recent.Kind).Linear).Size(px(t, 16), px(t, 16)).TextColor(k.Text2).Shrink(0).Label(recent.KindTitle)
				ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, 2)).Children(func() {
					ui.Text(c, rowState.Title).SingleLine().Ellipsis("…").MinWidth(0)
					ui.Text(c, rowState.Meta).FontSize(px(t, fsCaption)).TextColor(k.Text3).SingleLine().Ellipsis("…").MinWidth(0)
				})
				if iconButton(c.Key("copy"), icons.Copy, i18n.T("Copy")).Label(i18n.T("Copy: %s", rowState.Title)).Clicked() {
					actions.copyHomeRecent(rowState.ID)
				}
			})
			if row.Clicked() {
				actions.openHomeRecent(rowState.ID)
			}
			entryMenu(c, row, &rowState, actions, local, false, func() { actions.openHomeRecent(rowState.ID) }) // history_menu.go
		}
	})
}
