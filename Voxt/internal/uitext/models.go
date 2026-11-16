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
	translator := i18n.CurrentTranslator()
	switch id {
	case "doubao":
		return translator.T("Doubao (Volcengine)")
	case providers.ZhipuASR:
		return translator.T("Zhipu GLM-ASR")
	case "aliyun":
		return translator.T("Aliyun Bailian ASR")
	case providers.StepFunASR:
		return translator.T("StepFun")
	case "mimo":
		return translator.T("Xiaomi MiMo")
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
	return ModelErrorTextFor(i18n.CurrentTranslator(), err)
}

// ModelErrorTextFor is ModelErrorText with an explicit locale.
func ModelErrorTextFor(translator *i18n.Translator, err error) (msg, more string) {
	if err == nil {
		return "", ""
	}
	more = err.Error()
	if errors.Is(err, local.ErrNotInstalled) {
		return translator.T("The model is not installed."), more
	}
	if errors.Is(err, local.ErrNoRuntime) {
		return translator.T("This version of Voxt cannot run the model yet."), more
	}
	if errors.Is(err, dictation.ErrNotAllowed) {
		return translator.T("Voxt does not have Speech Recognition access. Allow it in Settings → Permissions."), more
	}
	if errors.Is(err, dictation.ErrNoLanguage) {
		return translator.T("macOS cannot recognise your main language on this Mac. Add it in System Settings → Keyboard → Dictation, or choose another speech model."), more
	}
	var ae *asr.Error
	if errors.As(err, &ae) {
		switch remote := RemoteSpeech(err); {
		case remote && ae.Kind == asr.KindAuth:
			return translator.T("The remote service refused the API key."), more
		case remote && ae.Kind == asr.KindQuota:
			return translator.T("The remote service refused the request. It may be busy, the account may be out of credit, or the model is not enabled for it."), more
		}
		switch ae.Kind {
		case asr.KindNetwork, asr.KindServer:
			return translator.T("The speech model stopped or did not answer. Try again in a moment."), more
		case asr.KindTimeout:
			return translator.T("The speech model took too long."), more
		case asr.KindAudio:
			return translator.T("The speech model did not accept the audio."), more
		case asr.KindCancelled:
			return translator.T("Cancelled."), more
		case asr.KindConfig:
			return translator.T("The speech model is not set up correctly. Check it on the Speech models page."), more
		}
		return translator.T("The speech model answered something Voxt does not understand."), more
	}
	var le *llm.Error
	if errors.As(err, &le) {
		switch le.Code {
		case llm.CodeSignIn:
			return translator.T("Your ChatGPT sign-in has expired. Sign in again on the Language models page."), more
		case llm.CodeUsageLimit:
			return translator.T("You have reached your ChatGPT plan's usage limit for now."), more
		case llm.CodeNotEligible:
			return translator.T("ChatGPT refused this account. Its plan, workspace or region cannot use Codex."), more
		case llm.CodeNoKey:
			return translator.T("No API key is saved for the remote model. Add one on the Language models page."), more
		case llm.CodeAuth:
			return translator.T("The remote service refused the API key."), more
		case llm.CodeModelNotFound:
			return translator.T("The remote service does not have this model."), more
		case llm.CodeAddress:
			return translator.T("Nothing answers at the remote model's address. Check the address on the Language models page."), more
		case llm.CodeConfig:
			return translator.T("The remote model's address is not allowed. Use https://, or http:// for an address on this Mac."), more
		case llm.CodeTimeout:
			return translator.T("The language model did not answer in time."), more
		case llm.CodeNetwork, llm.CodeServer, llm.CodeOverloaded, llm.CodeRateLimited:
			return translator.T("The language model stopped or did not answer. Try again in a moment."), more
		case llm.CodeResponsesOnly:
			return translator.T("This model needs the Responses API. For OpenAI and custom services, choose it under API format in Advanced."), more
		case llm.CodeContextLength:
			return translator.T("The text is too long for the language model."), more
		case llm.CodeCancelled:
			return translator.T("Cancelled."), more
		}
		return translator.T("The language model answered something Voxt does not understand."), more
	}
	return translator.T("The model stopped or did not answer. Try again in a moment."), more
}
