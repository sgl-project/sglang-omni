package main

import (
	"image"
	"image/color"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

func answerInputFixturePreviews(inputs []sessionCardInput) []answerInputPreview {
	if len(inputs) == 0 {
		return nil
	}
	previews := make([]answerInputPreview, 0, len(inputs))
	for _, input := range inputs {
		preview := answerInputPreview{Label: input.Label, Text: input.Text, Chars: input.Chars, Lines: input.Lines, Width: input.Width, Height: input.Height}
		if input.Image {
			placeholder := image.NewRGBA(image.Rect(0, 0, 1, 1))
			placeholder.Set(0, 0, color.RGBA{R: 70, G: 110, B: 150, A: 255})
			preview.Image = ui.NewBitmap(placeholder)
		}
		previews = append(previews, preview)
	}
	return previews
}

func cardInputChips(context *ui.Context, local sessionOverlayLocal) {
	ui.ScrollHorizontal(context.Key("answer-inputs")).Height(layout.CardAttachThumb).Shrink(0).Margin(layout.CardAttachH-layout.CardAttachThumb, 0, 0, 0).Label(i18n.T("Inputs sent with the instruction")).Children(func() {
		ui.Row(context).Gap(6).AlignItems(ui.Center).Children(func() {
			for index, preview := range local.Overlay.cardPreviews {
				button := ui.ButtonBase(context.Key(index)).Height(layout.CardAttachThumb).Padding(0, 6).Radius(5).Background(ui.RGBA(255, 255, 255, 0.08)).Label(preview.Label).Tooltip(i18n.T("Preview %s", preview.Label))
				if button.Children(func() {
					ui.Row(context).Gap(4).AlignItems(ui.Center).Children(func() {
						if preview.Image != nil {
							ui.Image(context, preview.Image).Size(18, 18).Fit(ui.Cover).Radius(3)
						}
						ui.Text(context, preview.Label).FontSize(11).TextColor(ovText2).SingleLine()
					})
				}).Clicked() {
					local.Card.preview = index + 1
				}
			}
		})
	})
}

func cardInputOriginal(context *ui.Context, card *sessionCardViewState, local sessionOverlayLocal, bodyHeight float32) bool {
	index := local.Card.preview - 1
	if index < 0 || index >= len(local.Overlay.cardPreviews) || bodyHeight <= 0 {
		return false
	}
	preview := local.Overlay.cardPreviews[index]
	ui.Column(context).Height(bodyHeight).Shrink(0).Margin(layout.CardRuleGap, 0, 0, 0).Gap(6).Children(func() {
		ui.Row(context).Shrink(0).Gap(8).AlignItems(ui.Center).Children(func() {
			if cardCapsule(context.Key("close-input-preview"), i18n.T("Back to answer"), "") {
				local.Card.preview = 0
			}
			ui.Text(context, preview.Label).FontSize(12).TextColor(ovText2).SingleLine().Ellipsis("…")
			if preview.Text != "" {
				ui.Text(context, i18n.T("%d chars · %d lines", preview.Chars, preview.Lines)).FontSize(11).TextColor(ovText2).SingleLine()
			} else if preview.Image != nil {
				ui.Text(context, i18n.T("%d × %d", preview.Width, preview.Height)).FontSize(11).TextColor(ovText2).SingleLine()
			}
		})
		if preview.Text != "" {
			ui.Scroll(context.Key("original-text")).Grow(1).TrackScroll(&local.Card.previewScroll).Children(func() {
				ui.Text(context, preview.Text).FontSize(layout.CardTextSize).TextColor(ovCardText).Selectable()
			})
		} else if preview.Image != nil {
			ui.ScrollBoth(context.Key("original-image")).Grow(1).TrackScroll(&local.Card.previewScroll).Children(func() {
				ui.Image(context, preview.Image).Width(float32(preview.Width)).Height(float32(preview.Height)).Fit(ui.Contain)
			})
		}
	})
	return true
}
