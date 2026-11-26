package uitext

// The words of the model catalogue (internal/models) and of System
// dictation (internal/dictation): what a model is called (its family,
// size and quantisation, as Voxt 1.x listed them: the catalogue's Name),
// which languages it knows, and why a download failed.

import (
	"errors"
	"io/fs"

	"github.com/hehehai/actnowkit/hfhub"

	"github.com/sgl-project/sglang-omni/Voxt/internal/dictation"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
)

// ModelName is what the UI calls model id: "Qwen3-ASR 1.7B (8bit)" (its
// id when it is not in the catalogue).
func ModelName(id string) string {
	return ModelNameFor(i18n.CurrentTranslator(), id)
}

func ModelNameFor(translator *i18n.Translator, id string) string {
	if id == dictation.ModelID {
		return translator.T("System dictation")
	}
	if m, ok := models.Lookup(id); ok {
		return m.Name
	}
	return id
}

// FamilyName gives the display name used on grouped model cards.
func FamilyName(family string) string {
	switch family {
	case "qwen3-asr":
		return "Qwen3-ASR"
	case "whisper":
		return "Whisper"
	}
	return family
}

// ModelLanguages says which languages model id knows ("" for none).
func ModelLanguages(id string) string {
	return ModelLanguagesFor(i18n.CurrentTranslator(), id)
}

func ModelLanguagesFor(translator *i18n.Translator, id string) string {
	switch id {
	case models.ASRAccurate, models.ASRFast, models.ASRQwen6bit:
		return translator.T("30 languages and 22 Chinese dialects, including Chinese, English, Cantonese, Japanese and Korean")
	case models.ASRWhisperTurbo, models.ASRWhisperLarge, models.ASRWhisperSmall:
		return translator.T("About 100 languages, including Chinese, English, Cantonese, Japanese and Korean")
	case models.ASRParakeet:
		return translator.T("25 European languages, including English, German, French, Spanish and Russian")
	case models.ASRSenseVoice:
		return translator.T("Chinese, Cantonese, English, Japanese and Korean")
	case models.ASRCohere:
		return translator.T("14 languages, including Chinese, English, Japanese and Korean")
	case models.ASRNemotron:
		return translator.T("28 languages, including Chinese, English, Japanese and Korean")
	case models.ASRMoss:
		return translator.T("Chinese and English")
	case models.LLMBalanced, models.LLMFast:
		return translator.T("Over 200 languages, best in Chinese and English")
	case models.LLMQwen3:
		return translator.T("Over 100 languages, best in Chinese and English")
	case models.LLMGemma4:
		return translator.T("Over 140 languages")
	case models.LLMTranslate:
		return translator.T("Translates between 33 languages, including Chinese, English, Japanese, Korean and Cantonese")
	case dictation.ModelID:
		return translator.T("The languages of macOS dictation that this Mac recognises on device (System Settings → Keyboard → Dictation)")
	}
	return ""
}

// DownloadErrorTextFor says why a model's download failed, and its detail
// (the error itself, for the ⓘ).
func DownloadErrorTextFor(translator *i18n.Translator, err error) (msg, more string) {
	if err == nil {
		return "", ""
	}
	more = err.Error()
	var pe *fs.PathError
	switch hfhub.Code(err) {
	case hfhub.CodeDiskSpace:
		return translator.T("Not enough disk space. Free some space, then try again."), more
	case hfhub.CodeChecksum:
		return translator.T("A downloaded file was damaged. Try again."), more
	case hfhub.CodeNotFound, hfhub.CodeGated:
		return translator.T("The download server does not have this model."), more
	case hfhub.CodeOffline:
		return translator.T("The download failed. Check the internet connection, then try again."), more
	case hfhub.CodeBusy:
		return translator.T("Another app is using this model. Try again later."), more
	}
	if errors.As(err, &pe) {
		return translator.T("The model could not be saved to the disk."), more
	}
	return translator.T("The download failed. Check the internet connection, then try again."), more
}
