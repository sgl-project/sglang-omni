package uitext

// The words of the model catalogue (internal/models) and of System
// dictation (internal/dictation): what a model is called (its family,
// size and quantisation, as Voxt 1.x listed them: the catalogue's Name),
// what it is good at and which languages it knows, and why a download
// failed.

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
	if id == dictation.ModelID {
		return i18n.T("System dictation")
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

// ModelTags are model id's short labels on its card, as Voxt 1.x showed
// them: where it runs, what it is best at, whether it shows the text
// while you speak, and whether it takes the dictionary (only Qwen3-ASR
// does). No ratings: they had no basis.
func ModelTags(id string) []string {
	switch id {
	case models.ASRAccurate, models.ASRQwen6bit:
		return []string{i18n.T("Local"), i18n.T("Accurate"), i18n.T("Realtime")}
	case models.ASRFast:
		return []string{i18n.T("Local"), i18n.T("Fast"), i18n.T("Realtime")}
	case models.ASRWhisperLarge, models.ASRCohere:
		return []string{i18n.T("Local"), i18n.T("Accurate"), i18n.T("No dictionary")}
	case models.ASRWhisperTurbo, models.ASRWhisperSmall, models.ASRParakeet, models.ASRSenseVoice:
		return []string{i18n.T("Local"), i18n.T("Fast"), i18n.T("No dictionary")}
	case models.ASRNemotron:
		return []string{i18n.T("Local"), i18n.T("Fast"), i18n.T("Realtime"), i18n.T("No dictionary")}
	case models.ASRMoss:
		return []string{i18n.T("Local"), i18n.T("Realtime"), i18n.T("No dictionary")}
	case dictation.ModelID:
		return []string{i18n.T("Local"), i18n.T("Built-in"), i18n.T("Realtime")}
	}
	return []string{i18n.T("Local")}
}

// ModelAbout says what model id is good at.
func ModelAbout(id string) string {
	switch id {
	case models.ASRAccurate:
		return i18n.T("The most accurate. Good with accents, dialects, names and several languages in one sentence, and shows the text as you speak.")
	case models.ASRFast:
		return i18n.T("Quicker and lighter on memory, for clear speech in a quiet place.")
	case models.ASRQwen6bit:
		return i18n.T("Qwen3-ASR 1.7B in less memory than the 8-bit model, and nearly as accurate. Shows the text as you speak.")
	case models.ASRWhisperTurbo:
		return i18n.T("OpenAI's Whisper, made quicker: good in many languages. The text shows a moment after you speak.")
	case models.ASRWhisperLarge:
		return i18n.T("The largest Whisper: careful in many languages, but slower and heavy on memory.")
	case models.ASRWhisperSmall:
		return i18n.T("A small Whisper: light on memory, less accurate.")
	case models.ASRParakeet:
		return i18n.T("NVIDIA's Parakeet: very quick, for European languages. It does not know Chinese, Japanese or Korean.")
	case models.ASRSenseVoice:
		return i18n.T("Very quick, for Chinese, Cantonese, English, Japanese and Korean.")
	case models.ASRCohere:
		return i18n.T("Cohere's model: accurate, with punctuation, but large. It hears your main language only (English when it does not know it).")
	case models.ASRNemotron:
		return i18n.T("NVIDIA's streaming model: shows the text as you speak, in many languages.")
	case models.ASRMoss:
		return i18n.T("Made for conversations between several people; shows the text as you speak.")
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
	case dictation.ModelID:
		return i18n.T("macOS's own speech recognition, on this Mac: nothing to download, and quick. Less accurate than Qwen3-ASR, with fewer languages; it hears your main language.")
	}
	return ""
}

// ModelLanguages says which languages model id knows ("" for none).
func ModelLanguages(id string) string {
	switch id {
	case models.ASRAccurate, models.ASRFast, models.ASRQwen6bit:
		return i18n.T("30 languages and 22 Chinese dialects, including Chinese, English, Cantonese, Japanese and Korean")
	case models.ASRWhisperTurbo, models.ASRWhisperLarge, models.ASRWhisperSmall:
		return i18n.T("About 100 languages, including Chinese, English, Cantonese, Japanese and Korean")
	case models.ASRParakeet:
		return i18n.T("25 European languages, including English, German, French, Spanish and Russian")
	case models.ASRSenseVoice:
		return i18n.T("Chinese, Cantonese, English, Japanese and Korean")
	case models.ASRCohere:
		return i18n.T("14 languages, including Chinese, English, Japanese and Korean")
	case models.ASRNemotron:
		return i18n.T("28 languages, including Chinese, English, Japanese and Korean")
	case models.ASRMoss:
		return i18n.T("Chinese and English")
	case models.LLMBalanced, models.LLMFast:
		return i18n.T("Over 200 languages, best in Chinese and English")
	case models.LLMQwen3:
		return i18n.T("Over 100 languages, best in Chinese and English")
	case models.LLMGemma4:
		return i18n.T("Over 140 languages")
	case models.LLMTranslate:
		return i18n.T("Translates between 33 languages, including Chinese, English, Japanese, Korean and Cantonese")
	case dictation.ModelID:
		return i18n.T("The languages of macOS dictation that this Mac recognises on device (System Settings → Keyboard → Dictation)")
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
