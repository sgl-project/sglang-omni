package main

import (
	"errors"
	"io/fs"
	"strings"

	"github.com/hehehai/actnowkit/hfhub"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dictation"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

func onboardingModelFound(input onboardingModelStateInput, id string) (models.Found, bool) {
	if input.Status[id].Mine {
		return models.Found{}, false
	}
	for _, found := range input.Found {
		if found.ID == id {
			return found, true
		}
	}
	return models.Found{}, false
}

func onboardingModelReuse(input onboardingModelStateInput) []models.Found {
	var reuse []models.Found
	for _, id := range input.PackIDs {
		if found, ok := onboardingModelFound(input, id); ok {
			reuse = append(reuse, found)
		}
	}
	return reuse
}

func onboardingModelAlternative(input onboardingModelStateInput) (models.Found, bool) {
	if _, ok := onboardingModelFound(input, input.SpeechID); ok || input.Status[input.SpeechID].Installed || input.Status[input.SpeechID].Downloading {
		return models.Found{}, false
	}
	model, _ := models.Lookup(input.SpeechID)
	for _, alternative := range model.Alternatives {
		if found, ok := onboardingModelFound(input, alternative); ok {
			return found, true
		}
	}
	return models.Found{}, false
}

func onboardingReadySpeech(input onboardingModelStateInput) []local.Model {
	var ready []local.Model
	for _, model := range input.InstalledASR {
		if _, offer := onboardingModelFound(input, model.ID); !offer && model.ID != dictation.ModelID {
			ready = append(ready, model)
		}
	}
	return ready
}

func onboardingModelName(id string) string {
	if model, ok := models.Lookup(id); ok {
		return model.Name
	}
	return id
}

func buildOnboardingModels(translator *i18n.Translator, input onboardingViewStateInput) onboardingModelsViewState {
	modelInput := input.Models
	reuse := onboardingModelReuse(modelInput)
	alternative, hasAlternative := onboardingModelAlternative(modelInput)
	ready := onboardingReadySpeech(modelInput)
	speechStatus := modelInput.Status[modelInput.SpeechID]
	state := onboardingModelsViewState{
		ReuseShown: len(reuse) > 0, LanguageReady: modelInput.LanguageReady,
		LanguageName: modelInput.Language.Name, NeedsDownload: len(ready) == 0 && !hasAlternative && !speechStatus.Downloading && !speechStatus.OnDisk,
	}
	state.Speech = onboardingSpeechViewState{
		ID: modelInput.SpeechID, Name: onboardingModelName(modelInput.SpeechID),
		About: input.SpeechAbout,
	}
	if len(ready) > 0 {
		state.Speech.ID, state.Speech.Name, state.Speech.Ready = ready[0].ID, ready[0].Name, true
	} else {
		state.Speech.Downloading = speechStatus.Downloading
		model, _ := models.Lookup(modelInput.SpeechID)
		state.DownloadLeft = uitext.HumanBytes(model.Size() - speechStatus.Done)
		if modelInput.PackLeft != "" {
			state.DownloadLeft = modelInput.PackLeft
		}
		found, onDisk := onboardingModelFound(modelInput, modelInput.SpeechID)
		state.Speech.OnDisk = onDisk
		if onDisk {
			state.Speech.InstalledBy = translator.T("Already on this Mac")
			if len(found.By) > 0 {
				state.Speech.InstalledBy = translator.T("Installed · from %s", found.By[0])
			}
		}
		state.Speech.SizeLabel = uitext.HumanBytes(model.Size())
		if len(reuse) > 0 {
			state.Speech.SizeLabel = translator.T("%s still to download", state.DownloadLeft)
			state.Speech.ShowDownload = !speechStatus.Downloading && !onDisk && !hasAlternative
			state.Speech.DownloadLabel = translator.T("Download (%s)", state.DownloadLeft)
		}
		if speechStatus.Total > 0 {
			state.Speech.DownloadFraction = float64(speechStatus.Done) / float64(speechStatus.Total)
		}
		if hasAlternative {
			wantName, alternativeName := onboardingModelName(modelInput.SpeechID), onboardingModelName(alternative.ID)
			alternativeState := &onboardingAlternativeViewState{
				WantID: modelInput.SpeechID, UseID: alternative.ID,
				Message:       translator.T("%s isn't installed, but %s is. Use it for now?", wantName, alternativeName),
				DownloadLabel: translator.T("Download %s", wantName), UseLabel: input.AlternativeUseLabel,
			}
			if len(alternative.By) > 0 {
				alternativeState.Message = translator.T("%s isn't installed, but %s (from %s) is. Use it for now?", wantName, alternativeName, alternative.By[0])
			}
			state.Speech.Alternative = alternativeState
		}
		state.Speech.Error = onboardingNotice(translator, input.SpeechError)
	}
	var size int64
	var names []string
	installedBy := ""
	for _, found := range reuse {
		size += found.Bytes
		state.ReuseIDs = append(state.ReuseIDs, found.ID)
		names = append(names, onboardingModelName(found.ID))
		if installedBy == "" && len(found.By) > 0 {
			installedBy = found.By[0]
		}
	}
	state.ReuseNames = strings.Join(names, " · ")
	if len(reuse) > 0 {
		state.ReuseSubtitle = translator.T("%s · no download needed", uitext.HumanBytes(size))
		if installedBy != "" {
			state.ReuseSubtitle = translator.T("Installed by %s · %s · no download needed", installedBy, uitext.HumanBytes(size))
		}
	}
	return state
}

func onboardingModelAbout(translator *i18n.Translator, id string) string {
	if id == models.ASRAccurate {
		return translator.T("The most accurate. Good with accents, dialects, names and several languages in one sentence. Shows the text as you speak.")
	}
	return ""
}

func onboardingDownloadErrorNotice(translator *i18n.Translator, err error) *sessionNotice {
	if err == nil {
		return nil
	}
	notice := &sessionNotice{more: err.Error()}
	var pathError *fs.PathError
	switch hfhub.Code(err) {
	case hfhub.CodeDiskSpace:
		notice.text = translator.T("Not enough disk space. Free some space, then try again.")
	case hfhub.CodeChecksum:
		notice.text = translator.T("A downloaded file was damaged. Try again.")
	case hfhub.CodeNotFound, hfhub.CodeGated:
		notice.text = translator.T("The download server does not have this model.")
	case hfhub.CodeOffline:
		notice.text = translator.T("The download failed. Check the internet connection, then try again.")
	case hfhub.CodeBusy:
		notice.text = translator.T("Another app is using this model. Try again later.")
	default:
		if errors.As(err, &pathError) {
			notice.text = translator.T("The model could not be saved to the disk.")
		} else {
			notice.text = translator.T("The download failed. Check the internet connection, then try again.")
		}
	}
	return notice
}
