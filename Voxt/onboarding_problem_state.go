package main

import (
	"errors"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dictation"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

func onboardingStartNotice(translator *i18n.Translator, notice, wait *sessionNotice) *sessionNotice {
	if notice == nil {
		return nil
	}
	localized := *notice
	switch notice.action {
	case actionLanguage:
		localized.text = translator.T("No language model is set up.")
	case actionModels:
		if wait != nil {
			localized.text = wait.text
		} else {
			localized.text = translator.T("No speech model is installed.")
		}
	case actionHelper:
		localized.text = translator.T("The helper is not running.")
	}
	return &localized
}

func onboardingSessionProblemNotice(translator *i18n.Translator, problem *session.Problem) *sessionNotice {
	if problem == nil {
		return nil
	}
	notice := &sessionNotice{}
	switch problem.Code {
	case session.ProblemNoASR:
		notice.text, notice.action = translator.T("No speech model is installed."), actionModels
	case session.ProblemSpeechWait:
		notice.text, notice.action = problem.Err.Error(), actionModels
	case session.ProblemMicrophone:
		notice.text, notice.more, notice.action = translator.T("Voxt does not have microphone access."), errText(problem.Err), actionPermissions
	case session.ProblemDevice:
		notice.text, notice.more = translator.T("No sound came from any microphone."), errText(problem.Err)
	case session.ProblemHelper:
		notice.text, notice.more, notice.action = translator.T("The helper stopped. Try again in a moment."), errText(problem.Err), actionHelper
	case session.ProblemASR:
		message, more := onboardingModelErrorText(translator, problem.Err)
		notice.more = more
		switch {
		case problem.Degraded && asr.KindOf(problem.Err) == asr.KindTimeout:
			notice.text = translator.T("Full recognition took too long, so the live text was used.")
		case problem.Degraded:
			notice.text = translator.T("Full recognition failed, so the live text was used.")
		case errors.Is(problem.Err, dictation.ErrNotAllowed):
			notice.text, notice.action = translator.T("Speech recognition failed: %s", message), actionPermissions
		default:
			notice.text, notice.action = translator.T("Speech recognition failed: %s", message), actionModels
		}
	case session.ProblemNoLLM:
		notice.text, notice.action = translator.T("No language model is set up."), actionLanguage
	case session.ProblemEmpty:
		notice.text = translator.T("The language model returned an empty answer.")
	case session.ProblemTruncated:
		notice.text = translator.T("The model hit its length limit, so the answer may be incomplete.")
	case session.ProblemLLM:
		if !problem.Degraded {
			message, more := onboardingModelErrorText(translator, problem.Err)
			notice.text, notice.more, notice.action = translator.T("The language model failed: %s", message), more, llmAction(problem.Err, actionLanguage)
			break
		}
		_, notice.more = onboardingModelErrorText(translator, problem.Err)
		notice.text = onboardingDegradedLLMText(translator, problem)
		notice.action = llmAction(problem.Err, actionNone)
	case session.ProblemNoSpeech:
		notice.text = translator.T("Nothing was heard.")
	case session.ProblemUnsupported:
		notice.text = translator.T("This feature is not available.")
	case session.ProblemTooShort:
	default:
		notice.text = translator.T("Something went wrong: %s", errText(problem.Err))
	}
	return notice
}

func onboardingDegradedLLMText(translator *i18n.Translator, problem *session.Problem) string {
	switch {
	case llm.Code(problem.Err) == llm.CodeUsageLimit:
		return translator.T("Your ChatGPT plan has no usage left, so the recognised text was inserted.")
	case session.TimedOut(problem.Err):
		return translator.T("The language model did not answer in time, so the recognised text was inserted.")
	}
	return translator.T("The language model failed, so the recognised text was inserted.")
}

func onboardingModelErrorText(translator *i18n.Translator, err error) (string, string) {
	if err == nil {
		return "", ""
	}
	more := err.Error()
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
	var speechError *asr.Error
	if errors.As(err, &speechError) {
		switch remote := uitext.RemoteSpeech(err); {
		case remote && speechError.Kind == asr.KindAuth:
			return translator.T("The remote service refused the API key."), more
		case remote && speechError.Kind == asr.KindQuota:
			return translator.T("The remote service refused the request. It may be busy, the account may be out of credit, or the model is not enabled for it."), more
		}
		switch speechError.Kind {
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
	var languageError *llm.Error
	if errors.As(err, &languageError) {
		switch languageError.Code {
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
