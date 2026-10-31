package main

// The setup guide's models step (onboarding.go): the recommended pack (the
// accurate speech model with its voice detection, the balanced language
// model) and one button that downloads what is missing. The models
// another ActNow app already put on this Mac come first (family spec
// §7.10.1): a card offers to use them ("Use them" claims them for
// VoxtNew, nothing downloads), and a kind whose recommended model is
// missing while one of its alternatives is there offers that one for now,
// the download staying the default (it is a downgrade).
//
// Which models are on disk, and which apps installed them, is a survey of
// the family's library (models.Manager.Survey) run off the main thread
// when the step shows; whether VoxtNew claims them already is the
// models' status (a.modelStatus), so that "Use them" hides the card.

import (
	"context"
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/local"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// obPack is the recommended pack: what the guide downloads.
var obPack = []string{models.ASRAccurate, models.VADSilero, models.LLMBalanced}

// obRecommended is the pack's model of kind.
func obRecommended(kind local.Kind) string {
	if kind == local.LLM {
		return models.LLMBalanced
	}
	return models.ASRAccurate
}

// obSurvey looks for the pack's models and their alternatives in the
// family's library, off the main thread (when the models step shows).
func (a *app) obSurvey() {
	if a.models == nil {
		return
	}
	var ids []string
	for _, id := range obPack {
		ids = append(ids, id)
		if m, ok := models.Lookup(id); ok {
			ids = append(ids, m.Alternatives...)
		}
	}
	m := a.models
	a.async(func() {
		found := m.Survey(ids...)
		a.post(func() { a.ob.found = found })
	})
}

// obFound is model id as the survey found it on disk, while VoxtNew does
// not claim it yet (it would be VoxtNew's model then, no longer an offer).
func (a *app) obFound(id string) (models.Found, bool) {
	if a.modelStatus[id].Mine {
		return models.Found{}, false
	}
	for _, f := range a.ob.found {
		if f.ID == id {
			return f, true
		}
	}
	return models.Found{}, false
}

// obReuse are the pack's models on this Mac that VoxtNew could use
// without downloading: the reuse card's.
func (a *app) obReuse() []models.Found {
	var out []models.Found
	for _, id := range obPack {
		if f, ok := a.obFound(id); ok {
			out = append(out, f)
		}
	}
	return out
}

// obReuseShown says whether the reuse card shows (its Use them is then
// the step's primary button).
func (a *app) obReuseShown() bool { return a.ob.step == obModels && len(a.obReuse()) > 0 }

// obAlternative is the model on this Mac that may stand in for kind's
// recommended one, which is not: "" when there is none, or the
// recommended one is on disk or downloading.
func (a *app) obAlternative(kind local.Kind) (models.Found, bool) {
	id := obRecommended(kind)
	if _, ok := a.obFound(id); ok || a.modelStatus[id].Installed || a.modelStatus[id].Downloading {
		return models.Found{}, false
	}
	m, _ := models.Lookup(id)
	for _, alt := range m.Alternatives {
		if f, ok := a.obFound(alt); ok {
			return f, true
		}
	}
	return models.Found{}, false
}

// obReady are the installed models of kind VoxtNew uses already: a model
// another app installed counts once VoxtNew claims it (until then it is
// an offer of the card or of an alternative line).
func (a *app) obReady(kind local.Kind) []local.Model {
	var out []local.Model
	for _, m := range a.installed(kind) {
		if _, offer := a.obFound(m.ID); !offer {
			out = append(out, m)
		}
	}
	return out
}

// obModelsStep is the step: the reuse card, a row by kind, and Download
// for what is missing.
func (a *app) obModelsStep(c *ui.Context) {
	t := c.Theme()
	reuse := a.obReuse()
	if len(reuse) > 0 {
		a.obReuseCard(c, reuse)
	}
	var want []string
	var size int64
	busy := false
	for _, kind := range []local.Kind{local.ASR, local.LLM} {
		if list := a.obReady(kind); len(list) > 0 {
			a.obModelRow(c, kind, list[0].ID, list[0].Name, true, false)
			continue
		}
		id := obRecommended(kind)
		m, _ := models.Lookup(id)
		st := a.modelStatus[id]
		_, onDisk := a.obFound(id)
		_, alt := a.obAlternative(kind)
		if !st.Downloading && !onDisk && !alt {
			want = append(want, id)
			size += m.Size() - st.Done
		}
		busy = busy || st.Downloading
		a.obModelRow(c, kind, id, uitext.ModelName(id), false, len(reuse) > 0)
	}
	if len(want) > 0 {
		label := i18n.T("Download (%s)", uitext.HumanBytes(size))
		if len(want) > 1 {
			label = i18n.T("Download both (%s)", uitext.HumanBytes(size))
		}
		ui.Row(c).Margin(px(t, sp2), 0, 0, 0).Children(func() {
			if secondaryIconButton(c.Key("ob-download"), icons.DownloadMinimalistic, label).Clicked() {
				for _, id := range want {
					a.downloadModel(id)
				}
			}
		})
	}
	if busy {
		caption(c, i18n.T("The models download in the background: continue, and they are ready when the bar fills."))
	}
}

// obReuseCard is the card of the models already on this Mac: how many,
// who installed them and their size, their names, Use them and Choose
// models….
func (a *app) obReuseCard(c *ui.Context, found []models.Found) {
	t := c.Theme()
	k := tok(t)
	var size int64
	var names, ids []string
	by := ""
	for _, f := range found {
		size += f.Bytes
		ids = append(ids, f.ID)
		names = append(names, uitext.ModelName(f.ID))
		if by == "" && len(f.By) > 0 {
			by = f.By[0]
		}
	}
	sub := i18n.T("%s · no download needed", uitext.HumanBytes(size))
	if by != "" {
		sub = i18n.T("Installed by %s · %s · no download needed", by, uitext.HumanBytes(size))
	}
	card(c.Key("ob-reuse")).Children(func() {
		ui.Row(c).Gap(px(t, sp3)).AlignItems(ui.Start).Children(func() {
			ui.Icon(c, icons.CheckCircle.Linear).Size(px(t, 20), px(t, 20)).TextColor(k.Success).Shrink(0)
			ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, sp2)).Children(func() {
				ui.Column(c).Gap(px(t, 2)).Children(func() {
					ui.Text(c, i18n.N(len(found), "%d model is already on this Mac", "%d models are already on this Mac", len(found))).FontWeight(600)
					caption(c, sub)
				})
				ui.Text(c, strings.Join(names, " · ")).FontSize(px(t, fsCaption))
				ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
					if primaryButton(c.Key("ob-reuse-use"), i18n.T("Use them")).Clicked() {
						a.obUse(ids...)
					}
					if ghostButton(c.Key("ob-reuse-choose"), i18n.T("Choose models…")).Clicked() {
						a.closeOnboarding()
						a.navigate(modelsPath)
					}
				})
			})
		})
	})
}

// obModelRow is the guide's line of a kind: the model VoxtNew uses
// (ready), or the recommended one: on this Mac already (the card's), or
// its size (what is still to download, under the card), its download,
// and the alternative on this Mac.
func (a *app) obModelRow(c *ui.Context, kind local.Kind, id, name string, ready, underCard bool) {
	t := c.Theme()
	k := tok(t)
	title := i18n.T("Speech recognition")
	if kind == local.LLM {
		title = i18n.T("Language model")
	}
	st := a.modelStatus[id]
	found, onDisk := a.obFound(id)
	m, _ := models.Lookup(id)
	ui.Column(c.Key("ob-models-"+string(kind))).Gap(px(t, sp1)).Margin(px(t, sp2), 0, 0, 0).Children(func() {
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
			sectionTitleText(c, title)
			switch {
			case ready:
				statusMark(c, icons.CheckCircle, k.Success, i18n.T("Ready: %s", name))
			case onDisk:
				ui.Text(c, name).FontWeight(600)
				from := i18n.T("Already on this Mac")
				if len(found.By) > 0 {
					from = i18n.T("Installed · from %s", found.By[0])
				}
				statusMark(c, icons.CheckCircle, k.Text2, from)
			default:
				ui.Text(c, name).FontWeight(600)
				left := uitext.HumanBytes(m.Size() - st.Done)
				if underCard {
					tagChip(c, i18n.T("%s still to download", left))
				} else {
					tagChip(c, uitext.HumanBytes(m.Size()))
				}
			}
		})
		if ready || onDisk { // the card's: it says the rest
			return
		}
		caption(c, uitext.ModelAbout(id))
		if alt, ok := a.obAlternative(kind); ok {
			a.obAlternativeLine(c, id, alt)
		}
		if st.Downloading {
			frac := 0.0
			if st.Total > 0 {
				frac = float64(st.Done) / float64(st.Total)
			}
			ui.Progress(c, frac).Label(i18n.T("Downloading %s", name))
		}
		if st.Err != nil && !st.Downloading {
			msg, more := uitext.DownloadErrorText(st.Err)
			notice(c, "ob-model-err", noticeError, msg, more, nil, nil)
		}
	})
}

// obAlternativeLine offers alt, on this Mac, for the recommended model
// want: Download want (the default: alt is a downgrade) or use alt for
// now.
func (a *app) obAlternativeLine(c *ui.Context, want string, alt models.Found) {
	t := c.Theme()
	wantName, altName := uitext.ModelName(want), uitext.ModelName(alt.ID)
	msg := i18n.T("%s isn't installed, but %s is. Use it for now?", wantName, altName)
	if len(alt.By) > 0 {
		msg = i18n.T("%s isn't installed, but %s (from %s) is. Use it for now?", wantName, altName, alt.By[0])
	}
	fillCard(c.Key("ob-alt-"+want)).Padding(px(t, sp2), px(t, sp3)).Gap(px(t, sp2)).Children(func() {
		ui.Text(c, msg).FontSize(px(t, fsCaption))
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
			if secondaryIconButton(c.Key("ob-alt-download"), icons.DownloadMinimalistic, i18n.T("Download %s", wantName)).Clicked() {
				a.downloadModel(want)
			}
			if ghostButton(c.Key("ob-alt-use"), i18n.X("alternative", "Use %s", altName)).Clicked() {
				a.obUse(alt.ID)
			}
		})
	})
}

// obUse claims the models ids for VoxtNew ("Use them", "Use … for now"),
// off the main thread, then downloads what they need that is missing
// (the voice detection of a speech model).
func (a *app) obUse(ids ...string) {
	m := a.models
	a.async(func() {
		err := m.Use(context.Background(), ids...)
		a.post(func() {
			if err != nil {
				a.toast(i18n.T("Could not use the models: %v", err))
			}
			a.modelsChanged()
			for _, id := range ids {
				mod, _ := models.Lookup(id)
				for _, need := range mod.Needs {
					if st := a.modelStatus[need]; !st.Installed && !st.Downloading {
						a.downloadModel(need)
					}
				}
			}
		})
	})
}
