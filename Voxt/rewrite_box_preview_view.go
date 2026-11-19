package main

import (
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

func boxTextBlockView(context *ui.Context, key string, segment boxTextSegment, local *rewriteBoxLocalUI, remove func()) {
	if !segment.Pasted || segment.Lines <= boxTextCollapseLines {
		if segment.Pasted && remove != nil {
			ui.Row(context).Shrink(0).Gap(6).AlignItems(ui.Center).Children(func() {
				ui.Text(context, segment.Text).FontSize(12).TextColor(ovText2).MaxLines(3).Shrink(0).Grow(1).MinWidth(0)
				if cardIconButton(context.Key("remove-text"), i18n.T("Remove pasted text"), "", icons.Close, true) {
					remove()
				}
			})
		} else {
			ui.Text(context, segment.Text).FontSize(12).TextColor(ovText2).MaxLines(3).Shrink(0)
		}
		return
	}
	expanded := local.expandedText[key]
	ui.Column(context.Key(key)).Shrink(0).Padding(6, 8, 6, 8).Gap(5).Radius(6).Background(ui.RGBA(255, 255, 255, 0.06)).Children(func() {
		ui.Row(context).Shrink(0).Gap(6).AlignItems(ui.Center).Children(func() {
			ui.Text(context, i18n.T("%s · %d chars · %d lines", segment.Source, segment.Chars, segment.Lines)).FontSize(11).FontWeight(600).TextColor(ovText2).Grow(1).MinWidth(0).SingleLine().Ellipsis("…")
			if remove != nil {
				if cardIconButton(context.Key("remove-text"), i18n.T("Remove pasted text"), "", icons.Close, true) {
					remove()
				}
			}
		})
		if expanded {
			ui.Scroll(context.Key("full-text")).MaxHeight(150).TrackScroll(&local.textScroll).Children(func() {
				ui.Text(context, segment.Text).FontSize(12).TextColor(ovCardText).Selectable()
			})
		} else {
			lines := strings.SplitN(segment.Text, "\n", boxTextCollapseLines+1)
			ui.Text(context, strings.Join(lines[:min(len(lines), boxTextCollapseLines)], "\n")).FontSize(12).TextColor(ovText2).MaxLines(boxTextCollapseLines).Shrink(0)
		}
		label := i18n.T("Show all")
		if expanded {
			label = i18n.T("Show less")
		}
		if ui.ButtonBase(context.Key("toggle-text")).Height(18).Label(label).Children(func() {
			ui.Text(context, label).FontSize(11).FontWeight(600).TextColor(ui.RGBA(255, 255, 255, 0.78)).SingleLine()
		}).Clicked() {
			if local.expandedText == nil {
				local.expandedText = map[string]bool{}
			}
			local.expandedText[key] = !expanded
		}
	})
}

func boxFullImagePreview(context *ui.Context, state rewriteBoxViewState, local *rewriteBoxLocalUI) bool {
	index := local.previewImage - 1
	if index < 0 || index >= len(state.Thumbnails) {
		return false
	}
	bitmap := state.Thumbnails[index]
	if bitmap == nil {
		return false
	}
	info := boxImageInfo{Source: i18n.T("Image")}
	info.Width, info.Height = bitmap.Size()
	if index < len(state.ImageInfo) {
		info = state.ImageInfo[index]
	}
	ui.Column(context).Grow(1).Gap(6).Children(func() {
		ui.Row(context).Shrink(0).Gap(8).AlignItems(ui.Center).Children(func() {
			if cardCapsule(context.Key("close-image-preview"), i18n.T("Back to input"), "") {
				local.previewImage = 0
			}
			ui.Text(context, info.Source).FontSize(12).TextColor(ovText2).SingleLine().Ellipsis("…")
			ui.Text(context, i18n.T("%d × %d", info.Width, info.Height)).FontSize(11).TextColor(ovText2).SingleLine()
		})
		ui.ScrollBoth(context.Key("box-original-image")).Grow(1).TrackScroll(&local.previewScroll).Children(func() {
			ui.Image(context, bitmap).Width(float32(info.Width)).Height(float32(info.Height)).Fit(ui.Contain)
		})
	})
	return true
}
