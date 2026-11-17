package main

// The input box's view (rewrite_box.go): the Rewrite card's dark face
// with the selected text, an editable text box, the pictures'
// thumbnails (each with a remove button), the voice line for the spoken
// instruction, and a line for an error.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// boxThumb is a thumbnail's size in the box.
const boxThumb = 40

// boxSelection is the selection the box shows: the session's while it
// runs, the kept one after a send failed.
func (a *app) boxSelection() string {
	if a.box.session != "" && a.sess.ID == a.box.session {
		return a.sess.Selection
	}
	return a.box.selection
}

// boxView is the input box's content, in its window.
func (a *app) boxView(c *ui.Context) {
	b := &a.box
	svc := c.Services()
	a.boxShare.store(b.text, b.images) // a keystroke is a frame
	root := ui.Column(c).Fill().Padding(12, 14, 12, 14).Gap(8)
	if c.Shortcut(0, ui.KeyEscape) {
		a.closeBox() // wherever the focus is in the box
	}
	if files := root.DroppedFiles(); len(files) > 0 {
		a.boxDrop(files)
	}
	root.Children(func() {
		// The header: what it is, Send while it records or a resend
		// waits, Close.
		ui.Row(c).Height(layout.CardHeadH).Shrink(0).Gap(layout.CardHeadGap).AlignItems(ui.Center).Children(func() {
			ui.Icon(c, icons.VoxtAnswer.Bold).Size(layout.CardIcon, layout.CardIcon).TextColor(ui.RGB(255, 255, 255)).Shrink(0).Label(uitext.KindTitle(store.KindRewrite))
			ui.Text(c, uitext.KindTitle(store.KindRewrite)).FontSize(layout.CardTitleSize).FontWeight(600).TextColor(ui.RGB(255, 255, 255)).SingleLine()
			ui.Box(c).Grow(1).MinWidth(0)
			if b.sending == "" && (b.session != "" && a.sess.ID == b.session && a.sess.State == session.Recording || b.session == "" && b.instruction != "") {
				if cardCapsule(c.Key("box-send"), i18n.T("Send"), i18n.T("Send (Enter)")) {
					a.boxSend()
				}
			}
			if cardIconButton(c.Key("box-close"), i18n.T("Close"), "", icons.Close, true) {
				a.closeBox()
			}
		})
		ui.Box(c).Height(1).Shrink(0).Background(ovCardRule)

		// The text selected when the shortcut fired: the input, as the
		// model gets it first.
		if sel := a.boxSelection(); sel != "" {
			ui.Text(c, sel).FontSize(12).TextColor(ovText2).MaxLines(3).Shrink(0)
		}

		// The text box.
		ta := ui.TextAreaBase(c.Key("box-text"), &b.text).
			Label(i18n.T("More text for the rewrite")).
			Placeholder(i18n.T("Type or paste more text, or drop images")).
			Lines(3, 8).FontSize(layout.CardTextSize).TextColor(ovCardText).Grow(1)
		ta.OnTextKey(func(k ui.TextKey) bool {
			switch {
			case k.Key == ui.KeyEnter && k.Mods == 0:
				a.boxSend()
				return true
			case k.Key == ui.KeyV && k.Mods == ui.Cmd:
				return a.boxPaste(svc)
			}
			return false
		})
		if b.focusText {
			b.focusText = false
			ta.Focus()
		}

		// The pictures.
		if len(b.thumbs) > 0 {
			ui.Row(c).Shrink(0).Gap(8).Children(func() {
				for i, bmp := range b.thumbs {
					a.boxThumbView(c.Key(i), i, bmp)
				}
			})
		}
		if b.limit {
			ui.Text(c, i18n.T("At most %d images", boxMaxImages)).FontSize(12).TextColor(tok(c.Theme()).Danger).Shrink(0)
		}

		// The voice line: the instruction, being heard or kept.
		ui.Row(c).Shrink(0).Gap(6).AlignItems(ui.Center).Children(func() {
			ui.Icon(c, icons.VoxtTranscribe.Bold).Size(14, 14).TextColor(ovText2).Shrink(0).Label(i18n.T("The spoken instruction"))
			ui.Text(c, a.boxVoiceLine()).FontSize(12).TextColor(ovText2).Grow(1).MinWidth(0).SingleLine().Ellipsis("…")
		})

		// An error, kept with the box's content.
		if b.err != "" {
			ui.Row(c).Shrink(0).Gap(6).AlignItems(ui.Center).Children(func() {
				ui.Icon(c, icons.DangerCircle.Linear).Size(12, 12).TextColor(tok(c.Theme()).Danger).Shrink(0)
				ui.Text(c, b.err).FontSize(12).TextColor(ovText2).Grow(1).MinWidth(0).SingleLine().Ellipsis("…").Tooltip(b.err)
			})
		}
	})
}

// boxThumbView is one picture's thumbnail, its remove button over the
// corner.
func (a *app) boxThumbView(c *ui.Context, i int, bmp *ui.Bitmap) {
	label := i18n.T("Remove this image")
	ui.Box(c).Width(boxThumb).Height(boxThumb).Shrink(0).Children(func() {
		if bmp != nil {
			ui.Image(c, bmp).Width(boxThumb).Height(boxThumb).Fit(ui.Cover).Radius(6)
		} else {
			ui.Box(c).Width(boxThumb).Height(boxThumb).Radius(6).Background(ui.RGBA(255, 255, 255, 0.1))
		}
		if ui.ButtonBase(c.Key("rm")).Absolute().Top(-4).Right(-4).Size(16, 16).Radius(8).Center().
			Background(ui.RGBA(0, 0, 0, 0.85)).Border(1, ui.RGBA(255, 255, 255, 0.24)).Label(label).Tooltip(label).
			Children(func() {
				ui.Icon(c, icons.Close.Linear).Size(9, 9).TextColor(ui.RGBA(255, 255, 255, 0.94))
			}).Clicked() {
			a.boxRemoveImage(i)
		}
	})
}

// boxVoiceLine is the voice line's text: the hint, the instruction
// being heard, or the recognised one (Enter sends it again).
func (a *app) boxVoiceLine() string {
	b := &a.box
	if b.session != "" && a.sess.ID == b.session && a.sess.State.Live() {
		switch a.sess.State {
		case session.Recording:
			if p := a.sess.Partial; p != "" {
				return p
			}
			return i18n.T("Listening…")
		default:
			return i18n.T("Recognising…")
		}
	}
	if b.instruction != "" {
		return b.instruction
	}
	if keys := a.featureKeys(hotkey.Rewrite); keys != "" {
		return i18n.T("Say the instruction (%s), then Enter sends", keys)
	}
	return i18n.T("Say the instruction, then Enter sends")
}
