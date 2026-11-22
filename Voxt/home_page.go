package main

// Home (/home): Transcribe's shortcut and the last speech model used, four
// usage numbers, and the persistent daily activity calendar.

import (
	"strings"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

func homePage(c *ui.Context, state *homeViewState, actions homeActions) {
	t := c.Theme()
	c.After(time.Minute) // the rolling usage windows change at local midnight
	columnPageHeader(c, i18n.T("Home"), func() {
		if ghostIconButton(c.Key("home-guide"), icons.Compass, i18n.T("Setup guide")).Label(i18n.T("Open setup guide")).Clicked() {
			actions.openHomeGuide()
		}
	})
	ui.Scroll(c.Key("home-scroll")).Grow(1).MinHeight(0).Children(func() {
		pageColumn(c).Gap(px(t, sp3)).Children(func() {
			homeSlogan(c, state.Slogan, actions)
			if state.Error != "" {
				notice(c, "home-history-error", noticeError, i18n.T("Could not read the history."), state.Error, nil, nil)
			}
			homeActivitySection(c, state, actions)
		})
	})
}

// homeSlogan is the line under Home's title: what Voxt does, how to start
// it, and the speech model used by the newest spoken result.
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
		} else {
			// The hint's words and its key cap, in one piece that wraps whole.
			ui.Row(c).Gap(px(t, 6)).AlignItems(ui.Center).Shrink(0).Children(func() {
				text(state.Before, k.Text2)
				keycap(c, state.Keys)
				text(state.After, k.Text2)
			})
		}
		if state.Model != "" {
			ui.Spacer(c)
			tagChip(c, state.Model)
		}
	})
}
