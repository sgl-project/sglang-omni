package main

// The answer card's language menu (10-15, second round): the target
// languages in VoxtNew's own dropdown, not the system's menu. The card's
// window is as large as the card (overlay.go), far too small for the list,
// and growing it would move the card's material in a transaction of its
// own (a frame of the wrong shape); so the list is a window of its own, a
// non-activating floating one like the card's: it never takes the app in
// front's focus, and the card does not move or change while it shows.
//
// It opens from the language capsule: above the card when the overlay
// sits at the bottom of the screen (there is no room under it), under the
// capsule, over the card, when it sits at the top; its right edge under
// the capsule's. A language closes it and translates the card's text
// again; so do the capsule again, Escape (before the card), any of the
// card's other actions, the card going, and another app coming to the
// front.

import (
	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/compose"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// Sizes of the language menu, in DIPs.
const (
	ovLangW      = 200
	ovLangRowH   = 30
	ovLangPad    = 6   // around the rows
	ovLangMaxH   = 336 // then it scrolls
	ovLangGap    = 6   // from the capsule
	ovLangRadius = 12
)

// langMenuState is the language menu's window.
type langMenuState struct {
	win   *mygo.Window
	shown bool
	// frame is the window's, in screen DIPs (the list and the shadow's
	// margin around it).
	frame  mygo.Rectangle
	scroll ui.ScrollState
}

// langMenuWanted says whether the menu shows: the card shows, offers
// another language, and the capsule opened it.
func (a *app) langMenuWanted() bool {
	return a.answer.langOpen && a.ov.shown && a.ov.mode == ovAnswer && a.cardNow().canLang
}

// langMenuHeight is the list's height: all the languages, up to
// ovLangMaxH.
func langMenuHeight() int {
	return min(len(compose.TargetLanguages)*ovLangRowH+2*ovLangPad, ovLangMaxH)
}

// langMenuFrame is the menu window's frame for the card's window frame
// card, the overlay at the top or the bottom of the screen: right under
// the capsule's right edge (the card's header ends with Insert, Copy and
// Close, then the capsule), above the card or under the capsule.
func langMenuFrame(card mygo.Rectangle, top bool) mygo.Rectangle {
	faceX, faceY := card.X+layout.Pad, card.Y+layout.Pad
	capRight := faceX + layout.AnswerW - layout.CardPadX - 3*layout.CardBtn - 3*layout.CardHeadGap
	h := langMenuHeight()
	x := capRight - ovLangW
	y := faceY - ovLangGap - h // above the card's top edge, over the capsule
	if top {
		y = faceY + layout.CardPadTop + layout.CardHeadH + ovLangGap
	}
	return mygo.Rectangle{X: x - layout.Pad, Y: y - layout.Pad, Width: ovLangW + 2*layout.Pad, Height: h + 2*layout.Pad}
}

// toggleLangMenu opens or closes the language menu (the capsule).
func (a *app) toggleLangMenu() {
	a.answer.langOpen = !a.answer.langOpen
	a.syncLangMenu()
}

// closeLangMenu closes the language menu, if it shows.
func (a *app) closeLangMenu() {
	if a.answer.langOpen {
		a.answer.langOpen = false
		a.syncLangMenu()
	}
}

// syncLangMenu shows, places or hides the language menu after a change of
// the card, its place or the menu's state. Main thread.
func (a *app) syncLangMenu() {
	m := &a.ov.lang
	if !a.langMenuWanted() {
		if !a.ov.shown || a.ov.mode != ovAnswer {
			a.answer.langOpen = false
		}
		if m.shown {
			m.shown = false
			if m.win != nil {
				m.win.Hide()
			}
		}
		return
	}
	frame := langMenuFrame(a.ov.frame, a.prefs.Overlay.Position == "top")
	moved := frame != m.frame || !m.shown
	m.frame = frame
	if !m.shown {
		// Scrolled to the chosen language.
		m.scroll.Y = float32(max(0, langIndex(a.cardNow().lang)*ovLangRowH-langMenuHeight()/2))
	}
	m.shown = true
	if !a.overlayNative() {
		return // tests: the state is enough
	}
	if m.win == nil {
		m.win = mygo.NewWindow(floatingWindowOptions("VoxtNew languages", frame.Width, frame.Height, hudContent(a.langMenuView)))
	}
	if moved {
		m.win.SetContentBounds(frame)
	}
	m.win.Invalidate()
	showFloating(m.win, frame.X, frame.Y)
}

// closeLangWindow destroys the menu's window (quit).
func (a *app) closeLangWindow() {
	if a.ov.lang.win != nil {
		a.ov.lang.win.Destroy()
		a.ov.lang.win = nil
	}
	a.ov.lang.shown = false
}

// langIndex is code's place among the target languages (0 when it is not
// one of them).
func langIndex(code string) int {
	for i, c := range compose.TargetLanguages {
		if c == code {
			return i
		}
	}
	return 0
}

// langMenuView is the menu window's content: the list, in the overlay's
// dark look (the app's dropdown: rows of 30, a check by the chosen one,
// the row under the pointer lit), with the shadow's margin around it.
func (a *app) langMenuView(c *ui.Context) {
	c.SetTheme(overlayTheme())
	defer c.Root().Background(ui.Transparent)
	cur := a.cardNow().lang
	ui.Column(c).Fill().Padding(layout.Pad).Children(func() {
		ui.Column(c.Key("ov-lang-face")).Fill().Radius(ovLangRadius).Background(ui.RGBA(30, 30, 34, 0.98)).
			Border(1, ui.RGBA(255, 255, 255, 0.12)).Shadow(0, layout.ShadowY, layout.ShadowBlur, 0, ui.RGBA(0, 0, 0, layout.ShadowA)).
			Label(i18n.T("Translate into")).Children(func() {
			ui.Scroll(c.Key("ov-lang-scroll")).Fill().TrackScroll(&a.ov.lang.scroll).Children(func() {
				ui.Column(c).Padding(ovLangPad - layout.Border).Children(func() {
					for _, code := range compose.TargetLanguages {
						a.langMenuRow(c, code, code == cur)
					}
				})
			})
		})
	})
}

// langMenuRow is one language: its name, a check when it is the card's.
func (a *app) langMenuRow(c *ui.Context, code string, chosen bool) {
	name := uitext.TargetLanguageLabel(code)
	b := ui.ButtonBase(c.Key("ov-lang-"+code)).Height(ovLangRowH).Padding(0, 10).Radius(8).Label(name).Checked(chosen)
	if b.Hovered() {
		b.Background(ui.RGBA(255, 255, 255, 0.08))
	}
	if b.Children(func() {
		ui.Row(c).Fill().Gap(8).AlignItems(ui.Center).Children(func() {
			ui.Text(c, name).FontSize(13).TextColor(ui.RGBA(255, 255, 255, 0.92)).SingleLine().Ellipsis("…").Grow(1).MinWidth(0)
			if chosen {
				ui.Icon(c, icons.Check.Linear).Size(14, 14).TextColor(tok(c.Theme()).Accent).Shrink(0)
			}
		})
	}).Clicked() {
		a.chooseLanguage(code)
	}
}

// chooseLanguage is a language chosen in the menu: it closes, and the
// card's text is translated again into it (the same language: nothing).
func (a *app) chooseLanguage(code string) {
	a.closeLangMenu()
	if code != a.cardNow().lang {
		a.retryAnswer(code)
	}
}
