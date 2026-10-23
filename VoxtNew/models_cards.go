package main

// The Local models page's view (models_page.go): a card per model, with
// what the ActNow family says of it (family spec §7.10.2–§7.10.4): the
// other apps that share it, another app downloading it, the app that
// deleted it for all apps; and the models of other family apps VoxtNew
// can run (models.Model.Borrowed), shown once they are on this Mac.

import (
	"fmt"

	"github.com/egoist/mygo/ui"
	"github.com/hehehai/actnowkit/hfhub"
	fammodels "github.com/hehehai/actnowkit/models"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/actnow"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

func (a *app) modelsPage(c *ui.Context) {
	t := c.Theme()
	pageHeader(c, i18n.T("Local models"), func() {
		if ghostButton(c.Key("models-network"), i18n.T("Network…")).Tooltip(i18n.T("How model downloads reach Hugging Face")).Clicked() {
			a.openNetwork()
		}
	})
	ui.Scroll(c.Key("models-scroll")).Grow(1).MinHeight(0).Children(func() {
		ui.Column(c).Padding(0, px(t, sp6), px(t, sp6)).Gap(px(t, sp3)).MaxWidth(px(t, 880)).Children(func() {
			caption(c, i18n.T("VoxtNew recognises speech and runs its language models on this Mac: your audio and text never leave it."))
			sectionTitleTip(c, i18n.T("Speech recognition"), i18n.T("Turns what you say into text. Every feature needs one to work by voice."))
			a.modelCards(c, models.ASR)
			sectionTitleTip(c, i18n.T("Language models"), i18n.T("Cleans up, translates and rewrites the text. Without one, Transcribe types the text as recognised."))
			a.modelCards(c, models.LLM)
			a.modelsAdvanced(c) // models_keep.go
		})
	})
	a.modelDeleteDialogs(c) // models_delete.go
	a.networkDialog(c)      // models_network.go
}

// modelCards are kind's models of the catalogue.
func (a *app) modelCards(c *ui.Context, kind models.Kind) {
	for _, m := range models.Catalog() {
		if m.Kind == kind && (!m.Borrowed || a.modelStatus[m.ID].OnDisk) { // another app's: only when it is on this Mac
			a.modelCard(c, m)
		}
	}
}

// modelCard is one model: what it is good at, its languages, its size
// and state, and what can be done with it.
func (a *app) modelCard(c *ui.Context, m models.Model) {
	t := c.Theme()
	k := tok(t)
	st := a.modelStatus[m.ID]
	ic := icons.Soundwave
	if m.Kind == models.LLM {
		ic = icons.ChatRoundLine
	}
	card(c.Key("model-" + m.ID)).Children(func() {
		ui.Row(c).Gap(px(t, sp3)).AlignItems(ui.Start).Children(func() {
			ui.Icon(c, ic.Linear).Size(px(t, 20), px(t, 20)).TextColor(k.Text2).Shrink(0).Margin(px(t, 1), 0, 0, 0)
			ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, sp1)).Children(func() {
				ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
					ui.Text(c, uitext.ModelName(m.ID)).FontWeight(600)
					if m.Default {
						tagChip(c, i18n.T("Default"))
					}
					if st.Installed {
						statusMark(c, icons.CheckCircle, k.Success, i18n.T("Installed"))
					}
				})
				ui.Text(c, uitext.ModelAbout(m.ID))
				if l := uitext.ModelLanguages(m.ID); l != "" {
					ui.Row(c).Gap(px(t, 4)).AlignItems(ui.Center).Children(func() {
						ui.Icon(c, icons.Global.Linear).Size(px(t, 14), px(t, 14)).TextColor(k.Text2).Shrink(0)
						caption(c, l).MinWidth(0)
					})
				}
				size, family := modelSizeText(m, st), modelFamilyText(m, st)
				if st.Downloading && st.WaitingFor != "" { // "Downloading in AudioSmith… 43%" says it all
					size, family = family, ""
				}
				if family != "" {
					caption(c.Key("family"), family)
				}
				ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
					caption(c, size)
					if mem, ok := a.modelMem[m.ID]; ok && st.Installed { // models_memory.go
						ui.Row(c.Key("loaded")).Gap(px(t, 6)).AlignItems(ui.Center).Children(func() {
							statusDot(c, k.Accent)
							caption(c, modelMemText(mem))
						})
					}
				})
			})
			ui.Row(c.Key("actions")).Gap(px(t, sp2)).Shrink(0).Children(func() { a.modelActions(c, m, st) })
		})
		if st.Downloading {
			frac := 0.0
			if st.Total > 0 {
				frac = float64(st.Done) / float64(st.Total)
			}
			ui.Progress(c, frac).Label(i18n.T("Downloading %s", uitext.ModelName(m.ID)))
		}
		if st.Err != nil && !st.Downloading {
			msg, more := uitext.DownloadErrorText(st.Err)
			notice(c, "model-err", noticeError, msg, more, func() {
				if ui.Button(c.Key("retry"), i18n.T("Try again")).Clicked() {
					a.downloadModel(m.ID)
				}
			}, nil)
		}
	})
}

// modelSizeText is a model's size, or how much of it came.
func modelSizeText(m models.Model, st models.Status) string {
	size := uitext.HumanBytes(m.Size())
	switch {
	case st.Downloading:
		return i18n.T("Downloading: %s of %s (%s)", uitext.HumanBytes(st.Done), uitext.HumanBytes(st.Total), fmt.Sprintf("%d%%", percent(st.Done, st.Total)))
	case st.Installed || st.OnDisk:
		return i18n.T("%s on disk", size)
	case st.Done > 0 && st.Elsewhere == "":
		return i18n.T("Stopped: %s of %s downloaded", uitext.HumanBytes(st.Done), size)
	}
	return i18n.T("%s download", size)
}

// percent is done of total in whole percents (0 without a total).
func percent(done, total int64) int {
	if total <= 0 {
		return 0
	}
	return int(done * 100 / total)
}

// modelFamilyText is what the family says of a model ("" nothing): the
// apps that installed a borrowed model VoxtNew does not use yet, the other
// apps that share it, another app downloading what VoxtNew's download
// waits for or the model itself, or the app that deleted it for all apps.
func modelFamilyText(m models.Model, st models.Status) string {
	switch {
	case m.Borrowed && !st.Installed:
		if len(st.Users) == 0 {
			return i18n.T("Already on this Mac")
		}
		return i18n.T("Installed by %s", uitext.Names(st.Users))
	case st.Installed && len(st.Users) > 0:
		return i18n.T("Shared with %s", uitext.Names(st.Users))
	case st.Downloading && st.WaitingFor != "":
		return i18n.T("Downloading in %s… %d%%", ownerName(st.WaitingFor), percent(st.Done, st.Total))
	case !st.Downloading && !st.OnDisk && st.Elsewhere != "":
		return i18n.T("Downloading in %s…", ownerName(st.Elsewhere))
	case !st.Downloading && !st.OnDisk && st.DeletedBy != "":
		return i18n.T("Deleted in %s", ownerName(st.DeletedBy))
	}
	return ""
}

// ownerName is the name of the family app of bundleID for a sentence:
// "AudioSmith", "another app" when the holder of a lock is unknown.
func ownerName(bundleID string) string {
	if bundleID == "" || bundleID == hfhub.UnknownOwner {
		return i18n.T("another app")
	}
	return actnow.AppName(bundleID)
}

// modelMemText says a loaded model's memory use (0: not measured).
func modelMemText(b int64) string {
	if b <= 0 {
		return i18n.T("Loaded")
	}
	return i18n.T("Loaded · %s in memory", uitext.MemoryBytes(b))
}

// modelActions are a card's buttons for the model's state.
func (a *app) modelActions(c *ui.Context, m models.Model, st models.Status) {
	name := uitext.ModelName(m.ID)
	switch {
	case st.Downloading:
		if ui.Button(c.Key("cancel"), i18n.T("Cancel")).Label(i18n.T("Cancel downloading %s", name)).Clicked() {
			a.cancelModel(m.ID)
		}
	case st.Installed:
		label, tip := i18n.T("Delete"), i18n.T("Delete %s…", name)
		if m.Borrowed { // the files are the other app's: VoxtNew lets go of it
			label, tip = i18n.T("Remove"), i18n.T("Remove %s…", name)
		}
		if ghostButton(c.Key("delete"), label).Label(tip).Disabled(a.modelDeciding == m.ID).Clicked() {
			a.askDeleteModel(m.ID) // models_delete.go
		}
	case m.Borrowed:
		if st.OnDisk && ui.Button(c.Key("use"), i18n.T("Use it")).Label(i18n.T("Use %s", name)).Clicked() {
			a.useModel(m.ID)
		}
	default:
		elsewhere := st.Elsewhere != ""
		if st.Done > 0 && !elsewhere && ghostButton(c.Key("discard"), i18n.T("Delete")).Label(i18n.T("Delete what was downloaded of %s", name)).Clicked() {
			a.deleteModel(m.ID, fammodels.ThisApp)
		}
		label := i18n.T("Download")
		switch {
		case st.DeletedBy != "":
			label = i18n.T("Download again")
		case st.Done > 0 && !elsewhere:
			label = i18n.T("Resume")
		}
		if st.Err == nil && secondaryIconButton(c.Key("download"), icons.DownloadMinimalistic, label).Label(i18n.T("Download %s", name)).Disabled(elsewhere).Clicked() { // a failure: the notice's Try again
			a.downloadModel(m.ID)
		}
	}
}
