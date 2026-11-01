package main

// A history entry's details (history_page.go): the result, the original,
// what made it (services, group, sound, times), and deleting it.

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/local"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// entryModel is the model an entry used, as the details show it: its
// name (the catalogue's words, also once it is deleted), else its id; ""
// for none.
func (a *app) entryModel(kind local.Kind, id, model string) string {
	if id == "" {
		return ""
	}
	name := uitext.ModelName(id)
	for _, m := range a.installed(kind) {
		if m.ID == id {
			name = m.Name
		}
	}
	if model != "" && model != id {
		name += " · " + model
	}
	return name
}

// historyDetail is the details panel of an entry.
func (a *app) historyDetail(c *ui.Context, e store.Entry) {
	t := c.Theme()
	k := tok(t)
	h := &a.hist
	ui.Column(c.Key("hist-detail-"+e.ID)).Padding(px(t, sp3), px(t, sp5), px(t, sp5)).Gap(px(t, sp3)).Children(func() {
		// The panel's header: its title, then Show in Finder (with a
		// recording), Delete and Close.
		ui.Row(c).Gap(px(t, sp1)).AlignItems(ui.Center).Children(func() {
			sectionHeading(c, i18n.T("Entry details")).Grow(1)
			if e.Audio != "" && iconButton(c.Key("hist-reveal"), icons.FolderOpen, i18n.T("Show the recording in Finder")).Clicked() {
				revealInFinder(filepath.Join(a.lib.store.History().AudioDir(), e.Audio))
			}
			if iconButton(c.Key("hist-delete"), icons.TrashBinMinimalistic, i18n.T("Delete the entry…")).Clicked() {
				a.askDeleteEntry(e.ID)
			}
			if iconButton(c.Key("panel-close"), icons.Close, i18n.T("Close")).Clicked() {
				h.sel = ""
			}
		})
		if e.Error != "" {
			kind, text := noticeError, i18n.T("Failed: %s", uitext.FirstLine(e.Error))
			if e.Status != store.StatusFailed {
				kind, text = noticeWarn, i18n.T("Done with a problem: %s", uitext.FirstLine(e.Error))
			}
			notice(c, "hist-detail-err", kind, text, e.Error, nil, nil)
		}
		a.entryRetryButton(c, e) // history_retry.go
		if s := recordingKeptText(e); s != "" {
			caption(c, s)
		}
		// The texts there are, each with Copy.
		textBlock := func(key, title, copyLabel, text string) {
			text = strings.TrimSpace(text)
			if text == "" {
				return
			}
			ui.Column(c.Key(key)).Gap(px(t, sp1)).Children(func() {
				ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).MinHeight(px(t, 28)).Children(func() {
					sectionTitleText(c, title).Grow(1).MinWidth(0)
					if iconButton(c.Key(key+"-copy"), icons.Copy, copyLabel).Clicked() {
						mygoClipboard(c, text)
						a.toast(i18n.T("Copied"))
					}
				})
				fillCard(c).Padding(px(t, sp2), px(t, sp3)).Children(func() {
					ui.Text(c, text).Selectable()
				})
			})
		}
		textBlock("hist-result", i18n.T("Result"), i18n.T("Copy the result"), e.Result)
		if e.PromptText != "" {
			textBlock("hist-prompt", i18n.T("Instruction"), i18n.T("Copy the instruction"), e.PromptText)
		}
		if e.SourceText != "" {
			textBlock("hist-source", i18n.T("Selected text"), i18n.T("Copy the selected text"), e.SourceText)
		}
		textBlock("hist-raw", i18n.T("Recognised text"), i18n.T("Copy the recognised text"), e.RawText)

		// The facts: labels as wide as the longest, values in the rest.
		app := e.AppName
		if e.URLHost != "" {
			app += " · " + e.URLHost
		}
		details := [][2]string{{i18n.T("Time"), i18n.DateTime(e.CreatedAt)}, {i18n.T("App"), app}}
		if e.Kind == store.KindTranslate {
			details = append(details, [2]string{i18n.T("Target language"), uitext.TargetLang(e)})
		}
		details = append(details,
			[2]string{i18n.T("Delivery"), uitext.DeliveryLabel(e)},
			[2]string{i18n.T("Speech recognition"), a.entryModel(local.ASR, e.ASRProvider, e.ASRModel)},
			[2]string{i18n.T("Sound optimization"), uitext.SoundText(e)},
			[2]string{i18n.T("Language model"), a.entryModel(local.LLM, e.LLMProvider, e.LLMModel)},
			[2]string{i18n.T("App rule"), uitext.RuleText(e)})
		sections := []struct {
			title string
			rows  [][2]string
		}{
			{i18n.T("Details"), details},
			{i18n.T("Timing"), [][2]string{
				{i18n.T("Recording"), uitext.RecordingTime(e)},
				{i18n.T("Recognition"), uitext.StageTime(e.ASRMS)},
				{i18n.T("Language model"), uitext.StageTime(e.LLMMS)},
				{i18n.T("Total"), uitext.StageTime(e.TotalMS)},
			}},
		}
		sectionGap := [2]float32{sp1, sp4} // above the first heading, above the others
		ui.Grid(c.Key("hist-facts")).ColumnTracks(ui.FitContent(), ui.Fr(1)).GapX(px(t, sp3)).GapY(px(t, sp1)).Children(func() {
			for i, sec := range sections {
				sectionTitleText(c, sec.title).ColumnSpan(2).Margin(px(t, sectionGap[min(i, 1)]), 0, px(t, 2), 0)
				for _, r := range sec.rows {
					ui.Text(c, r[0]).FontSize(px(t, fsCaption)).TextColor(k.Text2)
					v := ui.Text(c.Key(fmt.Sprintf("hist-fact-%d-%s", i, r[0])), uitext.OrDash(r[1])).FontSize(px(t, fsCaption)).Selectable().MinWidth(0)
					if r[1] == "" {
						v.TextColor(k.Text3)
					}
				}
			}
		})
	})
}

// historyDeleteDialog confirms Delete (askDeleteEntry), on History and
// Home.
func (a *app) historyDeleteDialog(c *ui.Context) {
	h := &a.hist
	if !h.delAsk {
		return
	}
	dialogFrame(c, "hist-del", &h.delAsk, i18n.T("Delete this entry?"), "", 420, func() {
		ui.Text(c, i18n.T("Its texts and its recording are deleted. This cannot be undone."))
	}, func() {
		ui.Spacer(c)
		if ui.Button(c.Key("hist-del-cancel"), i18n.T("Cancel")).AutoFocus().Clicked() {
			h.delAsk = false
		}
		if recordButton(c.Key("hist-del-ok"), i18n.T("Delete")).Clicked() {
			h.delAsk = false
			a.deleteHistoryEntry(h.delID)
		}
	})
}

// deleteHistoryEntry deletes an entry and its recording, in the
// background, then tells the history's observers (History, Home, the
// menu bar).
func (a *app) deleteHistoryEntry(id string) {
	if id == "" {
		return
	}
	hs := a.lib.store.History()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		err := hs.Delete(ctx, id)
		a.post(func() {
			if err != nil && !errors.Is(err, core.ErrNotFound) {
				a.toast(i18n.T("Could not delete the entry: %v", err))
				return
			}
			if a.hist.sel == id {
				a.hist.sel = ""
			}
			a.toast(i18n.T("Entry deleted"))
			a.historyChanged()
		})
	})
}
