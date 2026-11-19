package main

// The input box's view (rewrite_box.go): the Rewrite card's dark face
// with the selected text, an editable text box, the pictures'
// thumbnails (each with a remove button), the voice line for the spoken
// instruction, and a line for an error.

import (
	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

type boxBitmap = ui.Bitmap

var decodeBoxBitmap = ui.DecodeBitmap

// boxThumb is a thumbnail's size in the box.
const boxThumb = 40

type boxDropWindow interface {
	boxWindow
	OnFileDrop(func(*mygo.FileDropEvent)) func()
}

var newBoxNativeWindow = func(options mygo.WindowOptions) boxDropWindow { return mygo.NewWindow(options) }

// newBoxWindow creates the input box's native panel.
func newBoxWindow(state func() rewriteBoxViewState, actions rewriteBoxActions, local *rewriteBoxLocalUI) boxWindow {
	window := newBoxNativeWindow(mygo.WindowOptions{
		Title: i18n.T("Rewrite"), Width: boxWinW, Height: boxWinH, UseContentSize: true,
		Hidden: true, Frameless: true, Transparent: true, Vibrancy: mygo.VibrancyHUD,
		Panel: true, DisableResize: true, DisableClose: true,
		Content: hudContent(func(context *ui.Context) { rewriteBoxView(context, state(), actions, local) }),
	})
	window.OnFileDrop(func(event *mygo.FileDropEvent) { actions.drop(event.Paths) })
	return window
}

// rewriteBoxView draws the input box from its state, actions and local text.
func rewriteBoxView(context *ui.Context, state rewriteBoxViewState, actions rewriteBoxActions, local *rewriteBoxLocalUI) {
	if local.text != state.Text {
		local.text = state.Text
	}
	services := context.Services()
	root := ui.Column(context).Fill().Padding(12, 14, 12, 14).Gap(8)
	if context.Shortcut(0, ui.KeyEscape) {
		actions.close()
	}
	root.Children(func() {
		// The header: what it is, Send while it records or a resend
		// waits, Close.
		ui.Row(context).Height(layout.CardHeadH).Shrink(0).Gap(layout.CardHeadGap).AlignItems(ui.Center).Children(func() {
			ui.Icon(context, icons.VoxtAnswer.Bold).Size(layout.CardIcon, layout.CardIcon).TextColor(ui.RGB(255, 255, 255)).Shrink(0).Label(i18n.T("Rewrite"))
			ui.Text(context, i18n.T("Rewrite")).FontSize(layout.CardTitleSize).FontWeight(600).TextColor(ui.RGB(255, 255, 255)).SingleLine()
			ui.Box(context).Grow(1).MinWidth(0)
			if state.CanSend {
				if cardCapsule(context.Key("box-send"), i18n.T("Send"), i18n.T("Send (Enter)")) {
					actions.textChanged(local.text)
					actions.send()
				}
			}
			if cardIconButton(context.Key("box-close"), i18n.T("Close"), "", icons.Close, true) {
				actions.close()
			}
		})
		ui.Box(context).Height(1).Shrink(0).Background(ovCardRule)

		// The text selected when the shortcut fired: the input, as the
		// model gets it first.
		if state.Selection != "" {
			ui.Text(context, state.Selection).FontSize(12).TextColor(ovText2).MaxLines(3).Shrink(0)
		}

		// The text box.
		textArea := ui.TextAreaBase(context.Key("box-text"), &local.text).
			Label(i18n.T("More text for the rewrite")).
			Placeholder(i18n.T("Type or paste more text, or drop images")).
			Lines(3, 8).FontSize(layout.CardTextSize).TextColor(ovCardText).Grow(1)
		textArea.OnPaste(func(string) bool { return actions.paste(services) })
		textArea.OnTextKey(func(key ui.TextKey) bool {
			if key.Key == ui.KeyEnter && key.Mods == 0 {
				actions.textChanged(local.text)
				actions.send()
				return true
			}
			return false
		})
		if state.FocusText {
			textArea.Focus()
			actions.focused()
		}

		// The pictures.
		if len(state.Thumbnails) > 0 {
			ui.Row(context).Shrink(0).Gap(8).Children(func() {
				for index, bitmap := range state.Thumbnails {
					boxThumbView(context.Key(index), index, bitmap, actions.remove)
				}
			})
		}
		if state.Limit {
			ui.Text(context, i18n.T("At most %d images", boxMaxImages)).FontSize(12).TextColor(tok(context.Theme()).Danger).Shrink(0)
		}

		// The voice line: the instruction, being heard or kept.
		ui.Row(context).Shrink(0).Gap(6).AlignItems(ui.Center).Children(func() {
			ui.Icon(context, icons.VoxtTranscribe.Bold).Size(14, 14).TextColor(ovText2).Shrink(0).Label(i18n.T("The spoken instruction"))
			ui.Text(context, state.VoiceLine).FontSize(12).TextColor(ovText2).Grow(1).MinWidth(0).SingleLine().Ellipsis("…")
		})

		// An error, kept with the box's content.
		if state.Error != "" {
			ui.Row(context).Shrink(0).Gap(6).AlignItems(ui.Center).Children(func() {
				ui.Icon(context, icons.DangerCircle.Linear).Size(12, 12).TextColor(tok(context.Theme()).Danger).Shrink(0)
				ui.Text(context, state.Error).FontSize(12).TextColor(ovText2).Grow(1).MinWidth(0).SingleLine().Ellipsis("…").Tooltip(state.Error)
			})
		}
	})
	actions.textChanged(local.text)
}

// boxThumbView is one picture's thumbnail, its remove button over the
// corner.
func boxThumbView(context *ui.Context, index int, bitmap *ui.Bitmap, remove func(int)) {
	label := i18n.T("Remove this image")
	ui.Box(context).Width(boxThumb).Height(boxThumb).Shrink(0).Children(func() {
		if bitmap != nil {
			ui.Image(context, bitmap).Width(boxThumb).Height(boxThumb).Fit(ui.Cover).Radius(6)
		} else {
			ui.Box(context).Width(boxThumb).Height(boxThumb).Radius(6).Background(ui.RGBA(255, 255, 255, 0.1))
		}
		if ui.ButtonBase(context.Key("rm")).Absolute().Top(-4).Right(-4).Size(16, 16).Radius(8).Center().
			Background(ui.RGBA(0, 0, 0, 0.85)).Border(1, ui.RGBA(255, 255, 255, 0.24)).Label(label).Tooltip(label).
			Children(func() {
				ui.Icon(context, icons.Close.Linear).Size(9, 9).TextColor(ui.RGBA(255, 255, 255, 0.94))
			}).Clicked() {
			remove(index)
		}
	})
}
