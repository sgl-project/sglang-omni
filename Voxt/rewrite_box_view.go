package main

// The input box's view (rewrite_box_controller.go): the Rewrite card's dark
// face with the selected text, an editable text box, pictures, the voice
// line, and an error.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

const boxThumb = 40

type boxThumbnail = ui.Bitmap
type boxUIContext = ui.Context
type boxUIServices = ui.Services

func boxDecodeBitmap(jpeg []byte) *boxThumbnail {
	bitmap, _ := ui.DecodeBitmap(jpeg)
	return bitmap
}

type boxViewState struct {
	title, selection, voiceLine, err string
	send, limit                      bool
	text                             *string
	focusText                        *bool
	thumbs                           []*ui.Bitmap
}

type boxViewActions struct {
	close  func()
	send   func()
	drop   func([]string)
	paste  func(boxUIServices) bool
	remove func(int)
}

func boxView(context *ui.Context, state boxViewState, actions boxViewActions) {
	root := ui.Column(context).Fill().Padding(12, 14, 12, 14).Gap(8)
	if context.Shortcut(0, ui.KeyEscape) {
		actions.close()
	}
	if files := root.DroppedFiles(); len(files) > 0 {
		actions.drop(files)
	}
	root.Children(func() {
		ui.Row(context).Height(layout.CardHeadH).Shrink(0).Gap(layout.CardHeadGap).AlignItems(ui.Center).Children(func() {
			ui.Icon(context, icons.VoxtAnswer.Bold).Size(layout.CardIcon, layout.CardIcon).TextColor(ui.RGB(255, 255, 255)).Shrink(0).Label(state.title)
			ui.Text(context, state.title).FontSize(layout.CardTitleSize).FontWeight(600).TextColor(ui.RGB(255, 255, 255)).SingleLine()
			ui.Box(context).Grow(1).MinWidth(0)
			if state.send {
				if cardCapsule(context.Key("box-send"), i18n.T("Send"), i18n.T("Send (Enter)")) {
					actions.send()
				}
			}
			if cardIconButton(context.Key("box-close"), i18n.T("Close"), "", icons.Close, true) {
				actions.close()
			}
		})
		ui.Box(context).Height(1).Shrink(0).Background(ovCardRule)
		if state.selection != "" {
			ui.Text(context, state.selection).FontSize(12).TextColor(ovText2).MaxLines(3).Shrink(0)
		}
		textArea := ui.TextAreaBase(context.Key("box-text"), state.text).
			Label(i18n.T("More text for the rewrite")).
			Placeholder(i18n.T("Type or paste more text, or drop images")).
			Lines(3, 8).FontSize(layout.CardTextSize).TextColor(ovCardText).Grow(1)
		textArea.OnTextKey(func(key ui.TextKey) bool {
			switch {
			case key.Key == ui.KeyEnter && key.Mods == 0:
				actions.send()
				return true
			case key.Key == ui.KeyV && key.Mods == ui.Cmd:
				return actions.paste(context.Services())
			}
			return false
		})
		if *state.focusText {
			*state.focusText = false
			textArea.Focus()
		}
		if len(state.thumbs) > 0 {
			ui.Row(context).Shrink(0).Gap(8).Children(func() {
				for index, bitmap := range state.thumbs {
					boxThumbView(context.Key(index), index, bitmap, actions.remove)
				}
			})
		}
		if state.limit {
			ui.Text(context, i18n.T("At most %d images", boxMaxImages)).FontSize(12).TextColor(tok(context.Theme()).Danger).Shrink(0)
		}
		ui.Row(context).Shrink(0).Gap(6).AlignItems(ui.Center).Children(func() {
			ui.Icon(context, icons.VoxtTranscribe.Bold).Size(14, 14).TextColor(ovText2).Shrink(0).Label(i18n.T("The spoken instruction"))
			ui.Text(context, state.voiceLine).FontSize(12).TextColor(ovText2).Grow(1).MinWidth(0).SingleLine().Ellipsis("…")
		})
		if state.err != "" {
			ui.Row(context).Shrink(0).Gap(6).AlignItems(ui.Center).Children(func() {
				ui.Icon(context, icons.DangerCircle.Linear).Size(12, 12).TextColor(tok(context.Theme()).Danger).Shrink(0)
				ui.Text(context, state.err).FontSize(12).TextColor(ovText2).Grow(1).MinWidth(0).SingleLine().Ellipsis("…").Tooltip(state.err)
			})
		}
	})
}

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
