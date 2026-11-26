package main

import (
	"fmt"
	"slices"
	"strings"

	"github.com/hehehai/actnowkit/hfhub"
	"github.com/sgl-project/sglang-omni/Voxt/internal/actnow"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

func modelShown(m models.Model, status models.Status) bool {
	switch {
	case m.Retired:
		return status.Installed || status.Downloading || status.Done > 0
	case m.Borrowed:
		return status.OnDisk
	default:
		return true
	}
}

func modelsUsedBy(translator *i18n.Translator, prefs settings.Settings, installedASR, installedLLM []local.Model, kind local.Kind, id string) string {
	var names []string
	for _, feature := range []string{hotkey.Transcribe, hotkey.Translate, hotkey.Rewrite, hotkey.Meeting} {
		if kind == local.LLM && feature == hotkey.Transcribe && !prefs.Features.Transcribe.Enhance {
			continue
		}
		installed := installedASR
		if kind == local.LLM {
			installed = installedLLM
		}
		if selected, ok := chooseFeatureModel(prefs, installed, kind, feature); ok && selected.ID == id {
			names = append(names, uitext.FeatureTitleFor(translator, feature))
		}
	}
	if len(names) == 0 {
		return ""
	}
	return translator.T("Used by %s", strings.Join(names, " · "))
}

// modelSizeTextFor returns a model's size, or how much of it came.
func modelSizeTextFor(translator *i18n.Translator, m models.Model, status models.Status) string {
	size := uitext.HumanBytes(m.Size())
	switch {
	case status.Downloading:
		return translator.T("Downloading: %s of %s (%s)", uitext.HumanBytes(status.Done), uitext.HumanBytes(status.Total), fmt.Sprintf("%d%%", percent(status.Done, status.Total)))
	case status.Installed || status.OnDisk:
		return translator.T("%s on disk", size)
	case status.Done > 0 && status.Elsewhere == "":
		return translator.T("Stopped: %s of %s downloaded", uitext.HumanBytes(status.Done), size)
	default:
		return translator.T("%s download", size)
	}
}

// percent is done of total in whole percents (0 without a total).
func percent(done, total int64) int {
	if total <= 0 {
		return 0
	}
	return int(done * 100 / total)
}

// modelFamilyTextFor is what the family says of a model ("" nothing).
func modelFamilyTextFor(translator *i18n.Translator, m models.Model, status models.Status) string {
	switch {
	case (m.Borrowed || status.OnDisk) && !status.Installed:
		if len(status.Users) == 0 {
			return translator.T("Already on this Mac")
		}
		return translator.T("Installed by %s", uitext.NamesFor(translator, status.Users))
	case status.Installed && len(status.Users) > 0:
		return translator.T("Shared with %s", uitext.NamesFor(translator, status.Users))
	case status.Downloading && status.WaitingFor != "":
		return translator.T("Downloading in %s… %d%%", ownerNameFor(translator, status.WaitingFor), percent(status.Done, status.Total))
	case !status.Downloading && !status.OnDisk && status.Elsewhere != "":
		return translator.T("Downloading in %s…", ownerNameFor(translator, status.Elsewhere))
	case !status.Downloading && !status.OnDisk && status.DeletedBy != "":
		return translator.T("Deleted in %s", ownerNameFor(translator, status.DeletedBy))
	default:
		return ""
	}
}

// ownerName is the name of the family app of bundleID for a sentence.
func ownerName(bundleID string) string {
	return ownerNameFor(i18n.CurrentTranslator(), bundleID)
}

func ownerNameFor(translator *i18n.Translator, bundleID string) string {
	if bundleID == "" || bundleID == hfhub.UnknownOwner {
		return translator.T("another app")
	}
	return actnow.AppName(bundleID)
}

// modelMemTextFor says a loaded model's memory use (0: not measured).
func modelMemTextFor(translator *i18n.Translator, bytes int64) string {
	if bytes <= 0 {
		return translator.T("Loaded")
	}
	return translator.T("Loaded · %s in memory", uitext.MemoryBytes(bytes))
}

// keepLabelsFor returns the choices' labels, in settings.KeepChoices' order.
func keepLabelsFor(translator *i18n.Translator) []string {
	return []string{
		translator.T("Automatic"), translator.T("5 minutes"), translator.T("15 minutes"),
		translator.T("30 minutes"), translator.T("1 hour"), translator.T("Until Voxt quits"),
	}
}

// keepLabelFor returns the label of choice value (an unknown one is Automatic's).
func keepLabelFor(translator *i18n.Translator, value string) string {
	index := max(slices.Index(settings.KeepChoices, value), 0)
	return keepLabelsFor(translator)[index]
}
