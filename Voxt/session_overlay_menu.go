package main

import (
	"time"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

func langMenuHeight() int {
	return min(len(compose.TargetLanguages)*ovLangRowH+2*ovLangPad, ovLangMaxH)
}

func langMenuFrame(card mygo.Rectangle, top bool) mygo.Rectangle {
	faceX, faceY := card.X+layout.Pad, card.Y+layout.Pad
	capsuleRight := faceX + card.Width - 2*layout.Pad - layout.CardPadX - 3*layout.CardBtn - 3*layout.CardHeadGap
	height := langMenuHeight()
	x := capsuleRight - ovLangW
	y := faceY - ovLangGap - height
	if top {
		y = faceY + layout.CardPadTop + layout.CardHeadH + ovLangGap
	}
	return mygo.Rectangle{X: x - layout.Pad, Y: y - layout.Pad, Width: ovLangW + 2*layout.Pad, Height: height + 2*layout.Pad}
}

func thinkMenuFrame(language mygo.Rectangle, top bool, rows int) mygo.Rectangle {
	height := min(rows*ovLangRowH+2*ovLangPad, ovLangMaxH) + 2*layout.Pad
	if !top {
		language.Y += language.Height - height
	}
	language.Height = height
	return language
}

func pillMenuFrame(pill mygo.Rectangle, top bool, rows int) mygo.Rectangle {
	height := min(rows*ovLangRowH+2*ovLangPad, ovLangMaxH)
	x := pill.X + pill.Width - layout.Pad - ovLangW
	y := pill.Y + layout.Pad - ovLangGap - height
	if top {
		y = pill.Y + pill.Height - layout.Pad + ovLangGap
	}
	return mygo.Rectangle{X: x - layout.Pad, Y: y - layout.Pad, Width: ovLangW + 2*layout.Pad, Height: height + 2*layout.Pad}
}

func (a *app) langMenuWanted() bool {
	if !a.answer.ui.langOpen || !a.ov.shown {
		return false
	}
	if a.answer.ui.pillMenu {
		return a.ov.mode == ovWorking && a.ov.projection.view.Pill.CanThink
	}
	card := a.ov.projection.view.Card
	return a.ov.mode == ovAnswer && card != nil &&
		(a.answer.ui.thinkMenu && card.CanThink || !a.answer.ui.thinkMenu && card.CanLanguage)
}

func (a *app) syncLangMenu() {
	a.refreshSessionOverlay(time.Now())
	menu := &a.ov.lang
	if !a.langMenuWanted() || a.ov.projection.view.Menu == nil {
		if !a.ov.shown || a.ov.mode != ovAnswer || a.answer.ui.pillMenu {
			a.answer.ui.langOpen, a.answer.ui.thinkMenu, a.answer.ui.pillMenu = false, false, false
		}
		if menu.shown {
			menu.shown = false
			if menu.win != nil {
				menu.win.Hide()
			}
		}
		return
	}
	state := a.ov.projection.view.Menu
	frame := langMenuFrame(a.ov.frame, a.prefs.Overlay.Position == "top")
	if a.answer.ui.pillMenu {
		frame = pillMenuFrame(a.ov.frame, a.prefs.Overlay.Position == "top", len(state.Options))
	} else if a.answer.ui.thinkMenu {
		frame = thinkMenuFrame(frame, a.prefs.Overlay.Position == "top", len(state.Options))
	} else if a.ov.projection.view.Card.ShowThinking {
		frame.X -= layout.CardBtn + layout.CardHeadGap
	}
	moved := frame != menu.frame || !menu.shown
	menu.frame = frame
	if a.answer.ui.thinkMenu {
		a.ov.ui.langScroll.Y = 0
	} else if !menu.shown {
		a.ov.ui.langScroll.Y = float32(max(0, state.SelectedIndex*ovLangRowH-langMenuHeight()/2))
	}
	menu.shown = true
	if !a.overlayNative() {
		return
	}
	if menu.win == nil {
		content := sessionLanguageMenuContent(
			func() sessionMenuViewState {
				a.refreshSessionOverlay(time.Now())
				if a.ov.projection.view.Menu == nil {
					return sessionMenuViewState{SelectedIndex: -1}
				}
				return *a.ov.projection.view.Menu
			},
			a,
			&a.ov.ui,
		)
		menu.win = mygo.NewWindow(floatingWindowOptions("Voxt languages", frame.Width, frame.Height, hudContent(content)))
	}
	if moved {
		menu.win.SetContentBounds(frame)
	}
	menu.win.Invalidate()
	showFloating(menu.win, frame.X, frame.Y)
}

func (a *app) closeLangWindow() {
	if a.ov.lang.win != nil {
		a.ov.lang.win.Destroy()
		a.ov.lang.win = nil
	}
	a.ov.lang.shown = false
}
