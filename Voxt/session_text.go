package main

// What a session's problem says, and the action its notice offers
// (session_link.go): the words for the notice in the overlay and the Try
// area.

import (
	"errors"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dictation"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// sessionProblemText says what went wrong in a session, its detail (the
// model's own message) and where the notice's button goes. "" for a
// problem that needs no notice.
func sessionProblemText(p *session.Problem) (text, more, action string) {
	return sessionProblemTextFor(i18n.CurrentTranslator(), p)
}

func sessionProblemTextFor(translator *i18n.Translator, p *session.Problem) (text, more, action string) {
	switch p.Code {
	case session.ProblemNoASR:
		return translator.T("No speech model is installed."), "", actionModels
	case session.ProblemSpeechWait:
		return p.Err.Error(), "", actionModels
	case session.ProblemMicrophone:
		return translator.T("Voxt does not have microphone access."), errText(p.Err), actionPermissions
	case session.ProblemDevice:
		return translator.T("No sound came from any microphone."), errText(p.Err), actionNone
	case session.ProblemHelper:
		return translator.T("The helper stopped. Try again in a moment."), errText(p.Err), actionHelper
	case session.ProblemASR:
		msg, more := uitext.ModelErrorTextFor(translator, p.Err)
		if p.Degraded {
			if asr.KindOf(p.Err) == asr.KindTimeout {
				return translator.T("Full recognition took too long, so the live text was used."), more, actionNone
			}
			return translator.T("Full recognition failed, so the live text was used."), more, actionNone
		}
		if errors.Is(p.Err, dictation.ErrNotAllowed) {
			return translator.T("Speech recognition failed: %s", msg), more, actionPermissions
		}
		return translator.T("Speech recognition failed: %s", msg), more, actionModels
	case session.ProblemNoLLM:
		return translator.T("No language model is set up."), "", actionLanguage
	case session.ProblemEmpty:
		return translator.T("The language model returned an empty answer."), "", actionNone
	case session.ProblemTruncated:
		return translator.T("The model hit its length limit, so the answer may be incomplete."), "", actionNone
	case session.ProblemLLM:
		if !p.Degraded {
			msg, more := uitext.ModelErrorTextFor(translator, p.Err)
			return translator.T("The language model failed: %s", msg), more, llmAction(p.Err, actionLanguage)
		}
		_, more := uitext.ModelErrorTextFor(translator, p.Err)
		return degradedLLMTextFor(translator, p, false), more, llmAction(p.Err, actionNone)
	case session.ProblemNoSpeech:
		return translator.T("Nothing was heard."), "", actionNone
	case session.ProblemUnsupported:
		return translator.T("This feature is not available."), "", actionNone
	case session.ProblemTooShort:
		return "", "", actionNone
	}
	return translator.T("Something went wrong: %s", errText(p.Err)), "", actionNone
}

func errText(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

// noticeActionLabel is a notice's button.
// degradedLLMText says the recognised text went on past a failed language
// model: inserted, or with note, saved to Notes.
func degradedLLMText(p *session.Problem, note bool) string {
	return degradedLLMTextFor(i18n.CurrentTranslator(), p, note)
}

func degradedLLMTextFor(translator *i18n.Translator, p *session.Problem, note bool) string {
	timeout := session.TimedOut(p.Err)
	usage := llm.Code(p.Err) == llm.CodeUsageLimit
	switch {
	case note && usage:
		return translator.T("Your ChatGPT plan has no usage left, so the recognised text was saved to Notes.")
	case usage:
		return translator.T("Your ChatGPT plan has no usage left, so the recognised text was inserted.")
	case note && timeout:
		return translator.T("The language model did not answer in time, so the recognised text was saved to Notes.")
	case note:
		return translator.T("The language model failed, so the recognised text was saved to Notes.")
	case timeout:
		return translator.T("The language model did not answer in time, so the recognised text was inserted.")
	}
	return translator.T("The language model failed, so the recognised text was inserted.")
}

// llmAction is the notice's button for a language model's failure: the
// way out of a signed-in plan's own, else other.
func llmAction(err error, other string) string {
	switch llm.Code(err) {
	case llm.CodeSignIn:
		return actionSignIn
	case llm.CodeUsageLimit:
		return actionUsage
	}
	return other
}

func noticeActionLabel(action string) string {
	switch action {
	case actionSignIn:
		return i18n.T("Sign in")
	case actionUsage:
		return i18n.T("Manage usage")
	case actionModels:
		return i18n.T("Open Speech models")
	case actionLanguage:
		return i18n.T("Open Language models")
	case actionPermissions:
		return i18n.T("Open Permissions")
	case actionHelper:
		return i18n.T("Restart helper")
	case actionRetry:
		return i18n.T("Retry")
	}
	return ""
}
