package main

// The setup guide's models step (onboarding.go): the recommended pack (a
// speech model for this Mac's memory with its voice detection), which the
// footer's primary button downloads, then the language model: the one set
// up, or a way to the Language models page (they are remote). The models another
// ActNow app already put on this Mac come first (family spec §7.10.1): a
// card offers to use them ("Use them" claims them for VoxtNew, nothing
// downloads), and while the recommended speech model is missing and one
// of its alternatives is there, the guide offers that one for now, the
// download staying the default (it is a downgrade).
//
// Which models are on disk, and which apps installed them, is a survey of
// the family's library (models.Manager.Survey) run off the main thread
// when the step shows; whether VoxtNew claims them already is the
// models' status (a.modelStatus), so that "Use them" hides the card.

import (
	"context"
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dictation"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// recommendedSpeech is the speech model recommended for a Mac with gb GB
// of memory (0: not known): the small one under 16 GB, where the accurate
// one would take too much of it.
func recommendedSpeech(gb int) string {
	if gb > 0 && gb < 16 {
		return models.ASRFast
	}
	return models.ASRAccurate
}

// obSpeech is the pack's speech model.
func (a *app) obSpeech() string { return recommendedSpeech(a.ob.memoryGB) }

// obPack is the recommended pack: what the guide downloads.
func (a *app) obPack() []string { return []string{a.obSpeech(), models.VADSilero} }

// downloadPercent is how much of a download has come, in percent.
func downloadPercent(st models.Status) int {
	if st.Total <= 0 {
		return 0
	}
	return int(100 * float64(st.Done) / float64(st.Total))
}

// obDownloadLeft is what the recommended speech model still has to
// download, for the buttons that start it.
func (a *app) obDownloadLeft() string {
	id := a.obSpeech()
	m, _ := models.Lookup(id)
	return uitext.HumanBytes(m.Size() - a.modelStatus[id].Done)
}

// obNeedsDownload says that the models step has nothing but the download
// to offer, so its primary button is Download and continue: no speech
// model in use, the recommended one neither on this Mac nor downloading,
// and no alternative to choose instead.
func (a *app) obNeedsDownload() bool {
	if a.ob.step != obModels || len(a.obReady(local.ASR)) > 0 {
		return false
	}
	id := a.obSpeech()
	st := a.modelStatus[id]
	_, alt := a.obAlternative()
	return !alt && !st.Downloading && !st.OnDisk
}

// obSurvey looks for the pack's models and their alternatives in the
// family's library, off the main thread (when the models step shows).
func (a *app) obSurvey() {
	if a.models == nil {
		return
	}
	var ids []string
	for _, id := range a.obPack() {
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
	for _, id := range a.obPack() {
		if f, ok := a.obFound(id); ok {
			out = append(out, f)
		}
	}
	return out
}

// obReuseShown says whether the reuse card shows (its Use them is then
// the step's primary button).
func (a *app) obReuseShown() bool { return a.ob.step == obModels && len(a.obReuse()) > 0 }

// obAlternative is the model on this Mac that may stand in for the
// recommended speech model, which is not: "" when there is none, or the
// recommended one is on disk or downloading.
func (a *app) obAlternative() (models.Found, bool) {
	id := a.obSpeech()
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
// an offer of the card or of an alternative line). System dictation does
// not: the step still offers the recommended speech model.
func (a *app) obReady(kind local.Kind) []local.Model {
	var out []local.Model
	for _, m := range a.installed(kind) {
		if _, offer := a.obFound(m.ID); !offer && m.ID != dictation.ModelID {
			out = append(out, m)
		}
	}
	return out
}

// obModelsStep is the step: the reuse card, a row by kind, and the way to
// the other models. The download itself is the footer's (obNeedsDownload);
// under the reuse card, whose Use them is the primary button, it stays
// here.
func (a *app) obModelsStep(c *ui.Context) {
	t := c.Theme()
	reuse := a.obReuse()
	if len(reuse) > 0 {
		a.obReuseCard(c, reuse)
	}
	if list := a.obReady(local.ASR); len(list) > 0 {
		a.obModelRow(c, list[0].ID, list[0].Name, true, false)
	} else {
		id := a.obSpeech()
		st := a.modelStatus[id]
		_, onDisk := a.obFound(id)
		_, alt := a.obAlternative()
		a.obModelRow(c, id, uitext.ModelName(id), false, len(reuse) > 0)
		if id == models.ASRFast {
			caption(c, i18n.T("Chosen for this Mac's %d GB of memory. The Speech models page has more accurate models that use more memory.", a.ob.memoryGB))
		}
		if len(reuse) > 0 && !st.Downloading && !onDisk && !alt {
			label := i18n.T("Download (%s)", a.obDownloadLeft())
			ui.Row(c).Margin(px(t, sp2), 0, 0, 0).Children(func() {
				if secondaryIconButton(c.Key("ob-download"), icons.DownloadMinimalistic, label).Clicked() {
					a.downloadModel(id)
				}
			})
		}
		if st.Downloading {
			caption(c, i18n.T("The model downloads in the background. You can continue; it is ready when the bar fills."))
		}
	}
	ui.Row(c).Margin(px(t, sp1), 0, 0, 0).Children(func() {
		if ghostButton(c.Key("ob-choose-other"), i18n.T("Choose another model…")).Clicked() {
			a.obLeave(modelsPath)
		}
	})
	a.obLanguageRow(c)
}

// obLanguageRow is the guide's language model: the one the features use,
// or the way to set one up (they are remote: a service's API, or a
// server such as Ollama on this Mac).
func (a *app) obLanguageRow(c *ui.Context) {
	t := c.Theme()
	k := tok(t)
	m, ready := a.featureModel(local.LLM, hotkey.Transcribe)
	ui.Column(c.Key("ob-models-llm")).Gap(px(t, sp1)).Margin(px(t, sp4), 0, 0, 0).Children(func() {
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
			sectionTitleText(c, i18n.T("Language model"))
			if ready {
				statusMark(c, icons.CheckCircle, k.Success, i18n.T("Ready: %s", m.Name))
			}
		})
		if ready {
			return
		}
		caption(c, i18n.T("Cleans up, translates and rewrites the text. Run one on this Mac with Ollama, LM Studio or oMLX, or use a service's API key."))
		ui.Row(c).Children(func() {
			if ui.Button(c.Key("ob-llm-setup"), i18n.T("Set up a language model")).Clicked() {
				a.obLeave(remotePath) // onboarding.go: the page offers the way back
			}
		})
	})
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
						a.obLeave(modelsPath)
					}
				})
			})
		})
	})
}

// obModelRow is the guide's speech model: the one VoxtNew uses (ready),
// or the recommended one: on this Mac already (the card's), or its size
// (what is still to download, under the card), its download, and the
// alternative on this Mac.
func (a *app) obModelRow(c *ui.Context, id, name string, ready, underCard bool) {
	t := c.Theme()
	k := tok(t)
	title := i18n.T("Speech recognition")
	st := a.modelStatus[id]
	found, onDisk := a.obFound(id)
	m, _ := models.Lookup(id)
	ui.Column(c.Key("ob-models-asr")).Gap(px(t, sp1)).Margin(px(t, sp2), 0, 0, 0).Children(func() {
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
		if alt, ok := a.obAlternative(); ok {
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
