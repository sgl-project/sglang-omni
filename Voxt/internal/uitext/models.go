package uitext

// The words of a model's failure, local or remote (the notices of a
// session, the Try areas, a remote model's connection test): what went
// wrong and its detail, for the ⓘ.

import (
	"errors"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dictation"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
)

// SpeechName is a remote speech service's name: its brand, untranslated,
// but in Chinese for the Chinese services (as PresetName).
func SpeechName(id string) string {
	switch id {
	case "doubao":
		return i18n.T("Doubao (Volcengine)")
	case providers.ZhipuASR:
		return i18n.T("Zhipu GLM-ASR")
	case "aliyun":
		return i18n.T("Aliyun Bailian ASR")
	case providers.StepFunASR:
		return i18n.T("StepFun")
	case "mimo":
		return i18n.T("Xiaomi MiMo")
	}
	if pr, ok := providers.LookupSpeech(id); ok {
		return pr.Name
	}
	return id
}

// RemoteSpeech says whether err is a remote speech service's (its
// asr.Error names one of providers' speech services).
func RemoteSpeech(err error) bool {
	var ae *asr.Error
	if !errors.As(err, &ae) {
		return false
	}
	_, ok := providers.LookupSpeech(ae.Provider)
	return ok
}

// ModelErrorText says what went wrong with a model, and its detail
// (the error itself, for the ⓘ).
func ModelErrorText(err error) (msg, more string) {
	if err == nil {
		return "", ""
	}
	more = err.Error()
	if errors.Is(err, local.ErrNotInstalled) {
		return i18n.T("The model is not installed."), more
	}
	if errors.Is(err, local.ErrNoRuntime) {
		return i18n.T("This version of Voxt cannot run the model yet."), more
	}
	if errors.Is(err, dictation.ErrNotAllowed) {
		return i18n.T("Voxt may not use Speech Recognition: allow it in Settings → Permissions."), more
	}
	if errors.Is(err, dictation.ErrNoLanguage) {
		return i18n.T("macOS cannot recognise your main language on this Mac: add it in System Settings → Keyboard → Dictation, or choose another speech model."), more
	}
	var ae *asr.Error
	if errors.As(err, &ae) {
		switch remote := RemoteSpeech(err); {
		case remote && ae.Kind == asr.KindAuth:
			return i18n.T("The remote service refused the API key."), more
		case remote && ae.Kind == asr.KindQuota:
			return i18n.T("The remote service refused the request: it is busy, the account is out of credit, or the model is not enabled for it."), more
		}
		switch ae.Kind {
		case asr.KindNetwork, asr.KindServer:
			return i18n.T("The speech model stopped or did not answer. Try again in a moment."), more
		case asr.KindTimeout:
			return i18n.T("The speech model took too long."), more
		case asr.KindAudio:
			return i18n.T("The speech model did not accept the audio."), more
		case asr.KindCancelled:
			return i18n.T("Cancelled."), more
		case asr.KindConfig:
			return i18n.T("The speech model is not set up correctly: check it on the Speech models page."), more
		}
		return i18n.T("The speech model answered something Voxt does not understand."), more
	}
	var le *llm.Error
	if errors.As(err, &le) {
		switch le.Code {
		case llm.CodeNoKey:
			return i18n.T("No API key is saved for the remote model: add one on the Language models page."), more
		case llm.CodeAuth:
			return i18n.T("The remote service refused the API key."), more
		case llm.CodeModelNotFound:
			return i18n.T("The remote service does not have this model."), more
		case llm.CodeConfig:
			return i18n.T("The remote model's address is not allowed: use https://, or http:// to this Mac."), more
		case llm.CodeTimeout:
			return i18n.T("The language model did not answer in time."), more
		case llm.CodeNetwork, llm.CodeServer, llm.CodeOverloaded, llm.CodeRateLimited:
			return i18n.T("The language model stopped or did not answer. Try again in a moment."), more
		case llm.CodeResponsesOnly:
			return i18n.T("This model needs the Responses API. For OpenAI and custom services, choose it under API format in Advanced."), more
		case llm.CodeContextLength:
			return i18n.T("The text is too long for the language model."), more
		case llm.CodeCancelled:
			return i18n.T("Cancelled."), more
		}
		return i18n.T("The language model answered something Voxt does not understand."), more
	}
	return i18n.T("The model stopped or did not answer. Try again in a moment."), more
}
