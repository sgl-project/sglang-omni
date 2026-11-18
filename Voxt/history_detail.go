package main

// A history entry's details and the shared delete confirmation.

import (
	"fmt"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

func historyDetail(c *ui.Context, state *historyDetailViewState, actions historyActions, local *historyLocalUI) {
	t := c.Theme()
	colors := tok(t)
	ui.Column(c.Key("hist-detail-"+state.ID)).Padding(px(t, sp3), px(t, sp5), px(t, sp5)).Gap(px(t, sp3)).Children(func() {
		ui.Row(c).Gap(px(t, sp1)).AlignItems(ui.Center).Children(func() {
			sectionHeading(c, i18n.T("Entry details")).Grow(1)
			if state.CanReveal && iconButton(c.Key("hist-reveal"), icons.FolderOpen, i18n.T("Show recording in Finder")).Clicked() {
				actions.showHistoryRecording(state.ID)
			}
			if iconButton(c.Key("hist-delete"), icons.TrashBinMinimalistic, i18n.T("Delete entry…")).Clicked() {
				local.delAsk, local.delID = true, state.ID
			}
			if iconButton(c.Key("panel-close"), icons.Close, i18n.T("Close")).Clicked() {
				actions.closeHistoryDetail()
			}
		})
		if state.ErrorText != "" {
			kind := noticeError
			if state.ErrorWarning {
				kind = noticeWarn
			}
			notice(c, "hist-detail-err", kind, state.ErrorText, state.ErrorDetail, nil, nil)
		}
		entryRetryButton(c, state, actions)
		if state.RecordingKeptText != "" {
			caption(c, state.RecordingKeptText)
		}
		for _, block := range state.TextBlocks {
			ui.Column(c.Key(block.Key)).Gap(px(t, sp1)).Children(func() {
				ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).MinHeight(px(t, 28)).Children(func() {
					sectionTitleText(c, block.Title).Grow(1).MinWidth(0)
					if iconButton(c.Key(block.Key+"-copy"), icons.Copy, block.CopyLabel).Clicked() {
						actions.copyHistoryText(block.Text)
					}
				})
				fillCard(c).Padding(px(t, sp2), px(t, sp3)).Children(func() {
					if !block.Formatted || !actions.transcriptResult(c, state.ID, block.Text, block.Streaming) {
						ui.Text(c, block.Text).Selectable()
					}
				})
			})
		}
		sectionGap := [2]float32{sp1, sp4} // above the first heading, above the others
		ui.Grid(c.Key("hist-facts")).ColumnTracks(ui.FitContent(), ui.Fr(1)).GapX(px(t, sp3)).GapY(px(t, sp1)).Children(func() {
			for sectionIndex, section := range state.Sections {
				sectionTitleText(c, section.Title).ColumnSpan(2).Margin(px(t, sectionGap[min(sectionIndex, 1)]), 0, px(t, 2), 0)
				for _, fact := range section.Facts {
					ui.Text(c, fact.Label).FontSize(px(t, fsCaption)).TextColor(colors.Text2)
					value := ui.Text(c.Key(fmt.Sprintf("hist-fact-%d-%s", sectionIndex, fact.Label)), fact.Value).FontSize(px(t, fsCaption)).Selectable().MinWidth(0)
					if fact.Empty {
						value.TextColor(colors.Text3)
					}
				}
			}
		})
	})
}

func historyDeleteDialog(c *ui.Context, local *historyLocalUI, actions historyActions) {
	if !local.delAsk {
		return
	}
	dialogFrame(c, "hist-del", &local.delAsk, i18n.T("Delete this entry?"), "", 420, func() {
		ui.Text(c, i18n.T("Its text and recording will be deleted. This cannot be undone."))
	}, func() {
		ui.Spacer(c)
		if ui.Button(c.Key("hist-del-cancel"), i18n.T("Cancel")).AutoFocus().Clicked() {
			local.delAsk = false
		}
		if recordButton(c.Key("hist-del-ok"), i18n.T("Delete")).Clicked() {
			local.delAsk = false
			actions.deleteHistoryEntry(local.delID)
		}
	})
}
