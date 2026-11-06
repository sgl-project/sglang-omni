package main

// The Speech models page's view (models_page.go): a card per model, with
// what the ActNow family says of it (family spec §7.10.2–§7.10.4): the
// other apps that share it, another app downloading it, the app that
// deleted it for all apps; and the models of other family apps VoxtNew
// can run (models.Model.Borrowed), shown once they are on this Mac.
// System dictation has a card of its own: built into macOS, with nothing
// to download or delete. A speech model's card hides it from the
// features' menus (Voxt 1.x's Hide): the card then waits, folded, in
// Hidden models at the foot of the page, to be shown again.

import (
	"fmt"
	"slices"
	"strconv"

	"github.com/egoist/mygo/ui"
	"github.com/hehehai/actnowkit/hfhub"
	fammodels "github.com/hehehai/actnowkit/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/actnow"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dictation"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
	"github.com/sgl-project/sglang-omni/Voxt/internal/lang"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

func (a *app) modelsPage(c *ui.Context) {
	t := c.Theme()
	columnPageHeader(c, i18n.T("Speech models"), func() {
		if ghostButton(c.Key("models-network"), i18n.T("Network…")).Tooltip(i18n.T("How model downloads reach Hugging Face")).Clicked() {
			a.openNetwork()
		}
	})
	ui.Scroll(c.Key("models-scroll")).Grow(1).MinHeight(0).Children(func() {
		pageColumn(c).Gap(px(t, sp3)).Children(func() {
			a.obBackBar(c, modelsPath) // onboarding.go: left from the setup guide
			caption(c, i18n.T("A speech model turns what you say into text: every feature needs one to work by voice. Those on this Mac keep your audio here; a remote service gets your recording only when you choose it. Language models are on the Language models page."))
			sectionTitle(c, i18n.T("On this Mac"))
			caption(c, i18n.T("Free, and your audio stays here: download a model once, then it runs on this Mac."))
			a.modelCards(c, models.ASR)
			if !a.modelHidden(dictation.ModelID) {
				a.dictationCard(c)
			}
			a.providerCards(c, local.ASR) // remote_page.go
			a.speechContextSection(c)     // models_context.go
			if a.anyModelShown(models.LLM) {
				sectionTitleTip(c, i18n.T("Language models no longer used"), i18n.T("Voxt 2.0 ran these on this Mac; Voxt now uses the language models of the Language models page. Delete them to free the space: another app that uses one keeps it."))
				a.modelCards(c, models.LLM)
			}
			a.hiddenModelCards(c)
			a.modelsAdvanced(c) // models_keep.go
		})
	})
	a.modelDeleteDialogs(c) // models_delete.go
	a.networkDialog(c)      // models_network.go
	a.providerDialogView(c) // remote_dialog.go
}

// modelCards are kind's models of the catalogue that the page shows,
// but the hidden ones.
func (a *app) modelCards(c *ui.Context, kind models.Kind) {
	for _, m := range models.Catalog() {
		if m.Kind == kind && a.modelShown(m) && !a.modelHidden(m.ID) {
			a.modelCard(c, m)
		}
	}
}

// modelHidden says whether the user hid speech model id from the
// features' menus: a model in use never is.
func (a *app) modelHidden(id string) bool {
	return slices.Contains(a.prefs.Models.Hidden, id) && a.usedBy(local.ASR, id) == ""
}

// hiddenModels are the hidden models the page would list, in its order.
func (a *app) hiddenModels() []string {
	var ids []string
	for _, m := range models.Catalog() {
		if m.Kind == models.ASR && a.modelShown(m) && a.modelHidden(m.ID) {
			ids = append(ids, m.ID)
		}
	}
	if a.dictationListed() && a.modelHidden(dictation.ModelID) {
		ids = append(ids, dictation.ModelID)
	}
	return ids
}

// hiddenModelCards is the Hidden models section, folded until opened:
// the hidden models' cards, each with Show.
func (a *app) hiddenModelCards(c *ui.Context) {
	ids := a.hiddenModels()
	if len(ids) == 0 {
		return
	}
	collapsible(c.Key("models-hidden"), i18n.T("Hidden models"), &a.modelHiddenOpen, func() { caption(c, strconv.Itoa(len(ids))) }, func() {
		caption(c, i18n.T("The features' menus leave these models out. Show puts one back."))
		for _, id := range ids {
			if id == dictation.ModelID {
				a.dictationCard(c)
			} else if m, ok := models.Lookup(id); ok {
				a.modelCard(c, m)
			}
		}
	})
}

// hideButton is a speech model's Hide, or Show once hidden; none while
// the model is in use.
func (a *app) hideButton(c *ui.Context, id string) {
	if a.usedBy(local.ASR, id) != "" {
		return
	}
	name := uitext.ModelName(id)
	if slices.Contains(a.prefs.Models.Hidden, id) {
		if ghostButton(c.Key("show"), i18n.T("Show")).Label(i18n.T("Show %s", name)).Tooltip(i18n.T("Offer it in the features' menus again")).Clicked() {
			a.updateSettings(func(s *settings.Settings) {
				s.Models.Hidden = slices.DeleteFunc(slices.Clone(s.Models.Hidden), func(h string) bool { return h == id })
			})
		}
		return
	}
	if ghostButton(c.Key("hide"), i18n.T("Hide")).Label(i18n.T("Hide %s", name)).Tooltip(i18n.T("Leave it out of the features' menus")).Clicked() {
		a.updateSettings(func(s *settings.Settings) { s.Models.Hidden = append(slices.Clone(s.Models.Hidden), id) })
	}
}

// modelShown says whether the page shows m: a retired model only while
// Voxt has some of it on this Mac (to delete it), another app's only when
// it is on this Mac.
func (a *app) modelShown(m models.Model) bool {
	st := a.modelStatus[m.ID]
	switch {
	case m.Retired:
		return st.Installed || st.Downloading || st.Done > 0
	case m.Borrowed:
		return st.OnDisk
	}
	return true
}

// modelTags are m's labels: uitext.ModelTags, then whether a speech
// model knows the user's main language.
func (a *app) modelTags(m models.Model) []string {
	tags := uitext.ModelTags(m.ID)
	if spec, ok := omni.Lookup(m.ID); ok && len(spec.Languages) > 0 {
		if slices.Contains(spec.Languages, lang.Canonical(session.MainLanguage(a.prefs, i18n.Locale()))) {
			tags = append(tags, i18n.T("Supports main language"))
		} else {
			tags = append(tags, i18n.T("Lacks main language"))
		}
	}
	return tags
}

// anyModelShown says whether the page shows a model of kind.
func (a *app) anyModelShown(kind models.Kind) bool {
	for _, m := range models.Catalog() {
		if m.Kind == kind && a.modelShown(m) {
			return true
		}
	}
	return false
}

// dictationListed says whether the page lists System dictation: while
// the helper can recognise.
func (a *app) dictationListed() bool {
	return slices.ContainsFunc(a.installed(local.ASR), func(m local.Model) bool { return m.ID == dictation.ModelID })
}

// dictationCard is System dictation, while the helper can recognise.
func (a *app) dictationCard(c *ui.Context) {
	if !a.dictationListed() {
		return
	}
	t := c.Theme()
	k := tok(t)
	card(c.Key("model-" + dictation.ModelID)).Children(func() {
		ui.Row(c).Gap(px(t, sp3)).AlignItems(ui.Start).Children(func() {
			ui.Icon(c, icons.Soundwave.Linear).Size(px(t, 20), px(t, 20)).TextColor(k.Text2).Shrink(0).Margin(px(t, 1), 0, 0, 0)
			// modelCard's order: the name, "Built-in" where a size would be and
			// where it is used, what it is good at, then the labels at the foot.
			ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, sp2)).Children(func() {
				ui.Column(c).Gap(px(t, 2)).Children(func() {
					ui.Text(c, uitext.ModelName(dictation.ModelID)).FontWeight(600)
					meta := i18n.T("Built-in") // nothing to download
					if used := a.usedBy(local.ASR, dictation.ModelID); used != "" {
						meta += " · " + used
					}
					caption(c, meta)
				})
				ui.Text(c, uitext.ModelAbout(dictation.ModelID))
				ui.Row(c).Gap(px(t, 4)).AlignItems(ui.Center).Children(func() {
					ui.Icon(c, icons.Global.Linear).Size(px(t, 14), px(t, 14)).TextColor(k.Text2).Shrink(0)
					caption(c, uitext.ModelLanguages(dictation.ModelID)).MinWidth(0)
				})
				ui.Row(c.Key("tags")).Gap(px(t, 6)).AlignItems(ui.Center).Wrap().Children(func() {
					for _, tag := range uitext.ModelTags(dictation.ModelID) {
						tagChip(c, tag)
					}
				})
			})
			ui.Row(c.Key("actions")).Gap(px(t, sp2)).Shrink(0).Children(func() { a.hideButton(c, dictation.ModelID) })
		})
	})
}

// modelCard is one model: what it is good at, its languages, its size
// and state, and what can be done with it; a speech model's card carries
// its draft once the model is installed (draftRow).
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
			// Voxt 1.x's order: the name, then its size and where it is used,
			// then what it is good at, then the small labels at the foot.
			ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, sp2)).Children(func() {
				size, family := modelSizeText(m, st), modelFamilyText(m, st)
				if st.Downloading && st.WaitingFor != "" { // "Downloading in AudioSmith… 43%" says it all
					size, family = family, ""
				}
				ui.Column(c).Gap(px(t, 2)).Children(func() {
					ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
						ui.Text(c, uitext.ModelName(m.ID)).FontWeight(600)
						if m.Default && !m.Retired {
							tagChip(c, i18n.T("Default"))
						}
						if st.Installed {
							statusMark(c, icons.CheckCircle, k.Success, i18n.T("Installed"))
						}
					})
					ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
						meta := size
						if used := a.usedBy(local.Kind(m.Kind), m.ID); used != "" {
							meta += " · " + used
						}
						caption(c, meta)
						if mem, ok := a.modelMem[m.ID]; ok && st.Installed { // models_memory.go
							ui.Row(c.Key("loaded")).Gap(px(t, 6)).AlignItems(ui.Center).Children(func() {
								statusDot(c, k.Accent)
								caption(c, modelMemText(mem))
							})
						}
					})
					if family != "" {
						caption(c.Key("family"), family)
					}
				})
				ui.Text(c, uitext.ModelAbout(m.ID))
				if l := uitext.ModelLanguages(m.ID); l != "" {
					ui.Row(c).Gap(px(t, 4)).AlignItems(ui.Center).Children(func() {
						ui.Icon(c, icons.Global.Linear).Size(px(t, 14), px(t, 14)).TextColor(k.Text2).Shrink(0)
						caption(c, l).MinWidth(0)
					})
				}
				if tags := a.modelTags(m); len(tags) > 0 && !m.Retired {
					ui.Row(c.Key("tags")).Gap(px(t, 6)).AlignItems(ui.Center).Wrap().Children(func() {
						for _, tag := range tags {
							tagChip(c, tag)
						}
					})
				}
			})
			ui.Row(c.Key("actions")).Gap(px(t, sp2)).Shrink(0).Children(func() {
				if m.Kind == models.ASR {
					a.hideButton(c, m.ID)
				}
				a.modelActions(c, m, st)
			})
		})
		a.modelProgress(c, "model-err", m, st)
		// The draft's row stays while the draft is on disk or coming, its
		// model deleted first: it can still be cancelled or deleted.
		ds := a.modelStatus[m.Draft]
		if m.Draft != "" && (st.Installed || ds.Installed || ds.OnDisk || ds.Downloading || ds.Done > 0) {
			a.draftRow(c, m.Draft)
		}
	})
}

// modelProgress is a model's download bar while it downloads, and the
// notice of a failed download (key names the notice).
func (a *app) modelProgress(c *ui.Context, key string, m models.Model, st models.Status) {
	if st.Downloading {
		frac := 0.0
		if st.Total > 0 {
			frac = float64(st.Done) / float64(st.Total)
		}
		ui.Progress(c, frac).Label(i18n.T("Downloading %s", uitext.ModelName(m.ID)))
	}
	if st.Err != nil && !st.Downloading {
		msg, more := uitext.DownloadErrorText(st.Err)
		notice(c, key, noticeError, msg, more, func() {
			if ui.Button(c.Key("retry"), i18n.T("Try again")).Clicked() {
				a.downloadModel(m.ID)
			}
		}, nil)
	}
}

// draftRow is a speech model's speculative draft (models.Model.Draft) on
// its card: a small model that proposes several tokens a step for the
// speech model to check, so that it decodes faster.
// Installed, the model's server takes it at its next start
// (models_omni.go); deleting it stops that server first.
func (a *app) draftRow(c *ui.Context, id string) {
	t := c.Theme()
	k := tok(t)
	d, _ := models.Lookup(id)
	st := a.modelStatus[id]
	ui.Row(c.Key("draft")).Gap(px(t, sp3)).AlignItems(ui.Start).Margin(px(t, sp2), 0, 0, px(t, 32)).Children(func() {
		ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, sp1)).Children(func() {
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
				ui.Text(c, i18n.T("Speculative decoding")).FontWeight(600)
				if st.Installed {
					statusMark(c, icons.CheckCircle, k.Success, i18n.T("Installed"))
				}
			})
			caption(c, i18n.T("A small draft model proposes several words at a time and the speech model checks them, so it decodes faster."))
			caption(c.Key("size"), modelSizeText(d, st))
		})
		ui.Row(c.Key("actions")).Gap(px(t, sp2)).Shrink(0).Children(func() { a.modelActions(c, d, st) })
	})
	if st.Downloading || st.Err != nil { // under the row's text, not the card's icon
		ui.Column(c.Key("draft-progress")).Margin(0, 0, 0, px(t, 32)).Children(func() { a.modelProgress(c, "draft-err", d, st) })
	}
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
// apps that installed a model VoxtNew does not use yet (a borrowed one,
// or one of its own another app downloaded), the other
// apps that share it, another app downloading what VoxtNew's download
// waits for or the model itself, or the app that deleted it for all apps.
func modelFamilyText(m models.Model, st models.Status) string {
	switch {
	case (m.Borrowed || st.OnDisk) && !st.Installed:
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
	case m.Retired:
		if st.Done > 0 && ghostButton(c.Key("discard"), i18n.T("Delete")).Label(i18n.T("Delete what was downloaded of %s", name)).Clicked() {
			a.deleteModel(m.ID, fammodels.ThisApp)
		}
	case m.Borrowed || st.OnDisk:
		// On disk but not VoxtNew's: another app installed it. Download
		// has nothing to fetch; using it claims it.
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
