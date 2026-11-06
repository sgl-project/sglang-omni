package uitext

// The words of the model catalogue (internal/models): what a model is
// called (its family, size and quantisation, as Voxt 1.x listed them:
// the catalogue's Name), what it is good at and which languages it knows,
// and why a download failed.

import (
	"errors"
	"io/fs"

	"github.com/hehehai/actnowkit/hfhub"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
)

// ModelName is what the UI calls model id: "Qwen3-ASR 1.7B (8bit)" (its
// id when it is not in the catalogue).
func ModelName(id string) string {
	if m, ok := models.Lookup(id); ok {
		return m.Name
	}
	return id
}

// FamilyName is what the UI calls a model family (models.Model.Family):
// "Qwen3-ASR", "Whisper"… (the family itself when it has no name here).
// Not translated.
func FamilyName(family string) string {
	switch family {
	case "qwen3-asr":
		return "Qwen3-ASR"
	case "whisper":
		return "Whisper"
	case "moss-transcribe-diarize":
		return "MOSS"
	case "parakeet":
		return "Parakeet"
	case "sensevoice":
		return "SenseVoice"
	}
	return family
}

// ModelTags are model id's short labels on its card, as Voxt 1.x showed
// them: where it runs, what it is best at, whether it shows the text
// while you speak. No ratings: they had no basis.
func ModelTags(id string) []string {
	switch id {
	case models.ASRAccurate, models.ASRAccurate6bit:
		return []string{i18n.T("Local"), i18n.T("Accurate"), i18n.T("Realtime")}
	case models.ASRFast:
		return []string{i18n.T("Local"), i18n.T("Fast"), i18n.T("Realtime")}
	case models.ASRWhisperTurbo, models.ASRWhisperSmall, models.ASRParakeet, models.ASRSenseVoice:
		return []string{i18n.T("Local"), i18n.T("Fast")}
	case models.ASRWhisperLarge:
		return []string{i18n.T("Local"), i18n.T("Accurate")}
	case models.ASRMoss:
		return []string{i18n.T("Local"), i18n.T("Realtime")}
	}
	return []string{i18n.T("Local")}
}

// ModelAbout says what model id is good at.
func ModelAbout(id string) string {
	switch id {
	case models.ASRAccurate:
		return i18n.T("The most accurate. Good with accents, dialects, names and several languages in one sentence, and shows the text as you speak.")
	case models.ASRAccurate6bit:
		return i18n.T("The accurate model in 6 bits: a smaller download and less memory, and shows the text as you speak.")
	case models.ASRFast:
		return i18n.T("Quicker and lighter on memory, for clear speech in a quiet place. A good choice for a Mac with 8 GB of memory.")
	case models.ASRWhisperTurbo:
		return i18n.T("Whisper's best balance of speed and accuracy, in many languages. The text comes once you stop speaking.")
	case models.ASRWhisperLarge:
		return i18n.T("The most accurate Whisper, larger and slower than turbo. The text comes once you stop speaking.")
	case models.ASRWhisperSmall:
		return i18n.T("A small Whisper for a Mac with little memory, less accurate. The text comes once you stop speaking.")
	case models.ASRMoss:
		return i18n.T("Made for conversations with several speakers, and shows the text as you speak.")
	case models.ASRParakeet:
		return i18n.T("Fast, for European languages; it finds the language itself. The text comes once you stop speaking.")
	case models.ASRSenseVoice:
		return i18n.T("Fast and light, for Chinese, Cantonese, English, Japanese and Korean. The text comes once you stop speaking.")
	case models.VADSilero:
		return i18n.T("Tells speech from silence, so that only speech is recognised.")
	case models.LLMBalanced:
		return i18n.T("Cleans up dictation, translates and rewrites well: the best balance of quality and speed.")
	case models.LLMFast:
		return i18n.T("Answers sooner and uses less memory, for short clean-ups and simple rewrites.")
	case models.LLMTranslate:
		return i18n.T("Made for translation: quick and natural. Only Translate uses it.")
	case models.LLMQwen3:
		return i18n.T("Cleans up, translates and rewrites well.")
	case models.LLMGemma4:
		return i18n.T("Writes well in many languages, but is slower and needs more memory.")
	}
	return ""
}

// ModelLanguages says which languages model id knows ("" for none).
func ModelLanguages(id string) string {
	switch id {
	case models.ASRAccurate, models.ASRAccurate6bit, models.ASRFast:
		return i18n.T("30 languages and 22 Chinese dialects, including Chinese, English, Cantonese, Japanese and Korean")
	case models.ASRWhisperTurbo, models.ASRWhisperLarge, models.ASRWhisperSmall:
		return i18n.T("Over 90 languages, including Chinese, English, Japanese and Korean")
	case models.ASRMoss:
		return i18n.T("Chinese and English")
	case models.ASRParakeet:
		return i18n.T("25 European languages, including English, German, French, Spanish and Russian")
	case models.ASRSenseVoice:
		return i18n.T("Chinese, Cantonese, English, Japanese and Korean")
	case models.LLMBalanced, models.LLMFast:
		return i18n.T("Over 200 languages, best in Chinese and English")
	case models.LLMQwen3:
		return i18n.T("Over 100 languages, best in Chinese and English")
	case models.LLMGemma4:
		return i18n.T("Over 140 languages")
	case models.LLMTranslate:
		return i18n.T("Translates between 33 languages, including Chinese, English, Japanese, Korean and Cantonese")
	}
	return ""
}

// DownloadErrorText says why a model's download failed, and its detail
// (the error itself, for the ⓘ).
func DownloadErrorText(err error) (msg, more string) {
	if err == nil {
		return "", ""
	}
	more = err.Error()
	var pe *fs.PathError
	switch hfhub.Code(err) {
	case hfhub.CodeDiskSpace:
		return i18n.T("Not enough disk space: free some room, then try again."), more
	case hfhub.CodeChecksum:
		return i18n.T("A downloaded file was damaged. Try again."), more
	case hfhub.CodeNotFound, hfhub.CodeGated:
		return i18n.T("The download server does not have this model."), more
	case hfhub.CodeOffline:
		return i18n.T("The download failed. Check the internet connection, then try again."), more
	case hfhub.CodeBusy:
		return i18n.T("Another app is using this model. Try again later."), more
	}
	if errors.As(err, &pe) {
		return i18n.T("The model could not be saved to the disk."), more
	}
	return i18n.T("The download failed. Check the internet connection, then try again."), more
}
