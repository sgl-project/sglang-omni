package main

// A feature page's Models section and Transcribe's clean-up layers
// (feature_page.go).

import (
	"slices"
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// ---- Models ----

func (a *app) featureModels(c *ui.Context, feature string) {
	t := c.Theme()
	p := uitext.FeaturePrefix(feature)
	u := a.fui(feature)
	f := featureOf(a.prefs, feature)
	enhance := feature != hotkey.Transcribe || a.prefs.Features.Transcribe.Enhance
	readyA := a.installed(local.ASR)
	readyL := local.ForFeature(a.installed(local.LLM), feature)
	shownA := slices.DeleteFunc(slices.Clone(readyA), func(m local.Model) bool { return a.modelHidden(m.ID) }) // models_cards.go
	optsA, autoA := modelOptions(readyA, shownA)
	optsL, autoL := modelOptions(readyL, readyL)
	fb := a.prefs.Features.Translate.Fallback
	optsF := append([]string{i18n.T("None")}, optsL[1:]...)
	seen := strings.Join([]string{f.ASR, f.LLM, fb, strings.Join(optsA, "\x00"), strings.Join(optsL, "\x00")}, "\x01")
	if u.svcSeen != seen {
		u.svcSeen = seen
		u.asr, u.llm, u.fallback = modelLabel(readyA, f.ASR, autoA), modelLabel(readyL, f.LLM, autoL), modelLabel(readyL, fb, optsF[0])
	}
	sectionTitleTip(c, i18n.T("Models"), i18n.T("Automatic uses the recommended speech model on this Mac (Speech models page) and the first language model set up on the Language models page. A remote speech service is used when you choose it here, or when no speech model is installed on this Mac."))
	opened := false // one "Open Speech models" per page
	openModels := func() {
		if opened {
			return
		}
		opened = true
		if ui.Button(c.Key(p+"-open-models"), i18n.T("Open Speech models")).Clicked() {
			a.navigate(modelsPath)
		}
	}
	if len(readyA) == 0 {
		switch feature {
		case hotkey.Translate:
			notice(c, p+"-no-asr", noticeInfo, i18n.T("No speech model is installed, so only selected text can be translated."), "", openModels, nil)
		case hotkey.Rewrite:
			notice(c, p+"-no-asr", noticeWarn, i18n.T("No speech model is installed. Install one to rewrite."), "", openModels, nil)
		default:
			notice(c, p+"-no-asr", noticeWarn, i18n.T("No speech model is installed. Install one to transcribe."), "", openModels, nil)
		}
	}
	setUpLLM := func() {
		if ui.Button(c.Key(p+"-open-llm"), i18n.T("Set up a language model")).Clicked() {
			a.navigate(remotePath)
		}
	}
	if len(readyL) == 0 {
		switch feature {
		case hotkey.Translate:
			notice(c, p+"-no-llm", noticeWarn, i18n.T("No language model is set up. Set one up to translate."), "", setUpLLM, nil)
		case hotkey.Rewrite:
			notice(c, p+"-no-llm", noticeWarn, i18n.T("No language model is set up. Set one up to rewrite."), "", setUpLLM, nil)
		}
	}
	var asrDesc, asrLabel, llmDesc, llmLabel string
	switch feature {
	case hotkey.Translate:
		asrDesc, asrLabel = i18n.T("Turns what you say into text, when nothing is selected."), i18n.T("Speech model for Translate")
		llmDesc, llmLabel = i18n.T("Translates the text."), i18n.T("Language model for Translate")
	case hotkey.Rewrite:
		asrDesc, asrLabel = i18n.T("Turns your spoken instruction into text."), i18n.T("Speech model for Rewrite")
		llmDesc, llmLabel = i18n.T("Rewrites the selection, or writes what you ask for."), i18n.T("Language model for Rewrite")
	default:
		asrDesc, asrLabel = i18n.T("Turns your recording into text."), i18n.T("Speech model for Transcribe")
		llmDesc, llmLabel = i18n.T("Cleans up the recognised text when \"Clean up with a language model\" is on."), i18n.T("Language model for Transcribe")
	}
	settingRow(c, p+"-asr", i18n.T("Speech recognition"), asrDesc, func() {
		sel := ui.Select(c.Key(p+"-asr-sel"), &u.asr, optsA).Label(asrLabel).Width(px(t, 260)).Shrink(1).MinWidth(px(t, 160)).Disabled(len(readyA) == 0)
		if sel.Changed() {
			id := modelID(optsA, shownA, u.asr)
			a.updateSettings(func(s *settings.Settings) { setFeatureModel(s, local.ASR, feature, id) })
		}
	})
	settingRow(c, p+"-llm", i18n.T("Language model"), llmDesc, func() {
		sel := ui.Select(c.Key(p+"-llm-sel"), &u.llm, optsL).Label(llmLabel).Width(px(t, 260)).Shrink(1).MinWidth(px(t, 160)).
			Disabled(len(readyL) == 0 || !enhance)
		if sel.Changed() {
			id := modelID(optsL, readyL, u.llm)
			a.updateSettings(func(s *settings.Settings) { setFeatureModel(s, local.LLM, feature, id) })
		}
	})
	if feature == hotkey.Translate {
		settingRow(c, p+"-fallback", i18n.T("Fallback model"), i18n.T("If the language model fails, Voxt translates again with this model. History shows which model answered."), func() {
			sel := ui.Select(c.Key(p+"-fallback-sel"), &u.fallback, optsF).Label(i18n.T("Fallback language model for Translate")).Width(px(t, 260)).Shrink(1).MinWidth(px(t, 160)).
				Disabled(len(readyL) < 2)
			if sel.Changed() {
				id := modelID(optsF, readyL, u.fallback)
				a.updateSettings(func(s *settings.Settings) { s.Features.Translate.Fallback = id })
			}
		})
	}
	if feature == hotkey.Transcribe && len(readyL) == 0 && enhance {
		notice(c, "tr-no-llm", noticeInfo, i18n.T("No language model is set up, so the text is typed as recognised, without clean-up."), "", setUpLLM, nil)
	}
}

// modelOptions are a model select's labels: "Automatic (now: X)", X the
// first of the installed models (ready), then the models the menu shows.
func modelOptions(ready, shown []local.Model) (opts []string, auto string) {
	auto = i18n.T("Automatic")
	if len(ready) > 0 {
		auto = i18n.T("Automatic (now: %s)", ready[0].Name)
	}
	opts = []string{auto}
	for _, m := range shown {
		opts = append(opts, m.Name)
	}
	return opts, auto
}

// modelLabel is the select's label of the model id ("" automatic).
func modelLabel(ready []local.Model, id, auto string) string {
	for _, m := range ready {
		if m.ID == id {
			return m.Name
		}
	}
	return auto
}

// modelID is the id a select's label stands for, "" for automatic.
func modelID(opts []string, ready []local.Model, label string) string {
	if j := slices.Index(opts, label); j > 0 {
		return ready[j-1].ID
	}
	return ""
}

// ---- Clean-up ----

func (a *app) trEnhance(c *ui.Context) {
	sectionTitle(c, i18n.T("Clean-up"))
	f := &a.prefs.Features.Transcribe
	settingRow(c, "tr-enhance", i18n.T("Clean up with a language model"),
		i18n.T("Sends the recognised text and the rules below to the language model, and types its answer. When off, or if the model fails or is too slow, the text is typed as recognised."), func() {
			if ui.Switch(c.Key("tr-enhance-switch"), &f.Enhance).Label(i18n.T("Clean up with a language model")).Changed() {
				on := f.Enhance
				a.updateSettings(func(s *settings.Settings) { s.Features.Transcribe.Enhance = on })
			}
		})
	settingRow(c, "tr-cleanup", i18n.T("Basic clean-up"),
		i18n.T("Removes fillers (um, uh, you know) and false starts, and keeps what you said after correcting yourself. Fixes punctuation, spacing and capitals, turns spoken punctuation (\"comma\", \"new line\") into marks, and writes numbers and dates the usual way. Names, terms and code stay as said."), func() {
			if ui.Switch(c.Key("tr-cleanup-switch"), &f.Cleanup).Label(i18n.T("Basic clean-up")).Disabled(!f.Enhance).Changed() {
				on := f.Cleanup
				a.updateSettings(func(s *settings.Settings) { s.Features.Transcribe.Cleanup = on })
			}
		})
	settingRow(c, "tr-structure", i18n.T("Structure"),
		i18n.T("Makes a list when you clearly say steps or several items, and splits a long text into paragraphs. When off, the text keeps the form you said it in."), func() {
			if ui.Switch(c.Key("tr-structure-switch"), &f.Structure).Label(i18n.T("Structure")).Disabled(!f.Enhance).Changed() {
				on := f.Structure
				a.updateSettings(func(s *settings.Settings) { s.Features.Transcribe.Structure = on })
			}
		})
}
