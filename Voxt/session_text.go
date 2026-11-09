package main

// What a session's problem says, and the action its notice offers
// (session_link.go): the words for the notice in the overlay and the Try
// area.

import (
	"context"
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
	switch p.Code {
	case session.ProblemNoASR:
		return i18n.T("No speech model is installed."), "", actionModels
	case session.ProblemSpeechWait:
		return p.Err.Error(), "", actionModels
	case session.ProblemMicrophone:
		return i18n.T("Voxt may not use the microphone."), errText(p.Err), actionPermissions
	case session.ProblemDevice:
		return i18n.T("No microphone gave any sound."), errText(p.Err), actionNone
	case session.ProblemHelper:
		return i18n.T("The helper stopped: try again in a moment."), errText(p.Err), actionHelper
	case session.ProblemASR:
		msg, more := uitext.ModelErrorText(p.Err)
		if p.Degraded {
			if asr.KindOf(p.Err) == asr.KindTimeout {
				return i18n.T("Used the live text: full recognition took too long."), more, actionNone
			}
			return i18n.T("Used the live text: full recognition failed."), more, actionNone
		}
		if errors.Is(p.Err, dictation.ErrNotAllowed) {
			return i18n.T("Speech recognition failed: %s", msg), more, actionPermissions
		}
		return i18n.T("Speech recognition failed: %s", msg), more, actionModels
	case session.ProblemNoLLM:
		return i18n.T("No language model is set up."), "", actionLanguage
	case session.ProblemEmpty:
		return i18n.T("The language model answered nothing."), "", actionNone
	case session.ProblemTruncated:
		return i18n.T("The model hit its length limit: the answer may be incomplete."), "", actionNone
	case session.ProblemLLM:
		if !p.Degraded {
			msg, more := uitext.ModelErrorText(p.Err)
			return i18n.T("The language model failed: %s", msg), more, llmAction(p.Err, actionLanguage)
		}
		_, more := uitext.ModelErrorText(p.Err)
		return degradedLLMText(p, false), more, llmAction(p.Err, actionNone)
	case session.ProblemNoSpeech:
		return i18n.T("Nothing was heard."), "", actionNone
	case session.ProblemUnsupported:
		return i18n.T("This feature is not available."), "", actionNone
	case session.ProblemTooShort:
		return "", "", actionNone
	}
	return i18n.T("Something went wrong: %s", errText(p.Err)), "", actionNone
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
	var le *llm.Error
	timeout := errors.Is(p.Err, context.DeadlineExceeded) || errors.As(p.Err, &le) && le.Code == llm.CodeTimeout
	usage := llm.Code(p.Err) == llm.CodeUsageLimit
	switch {
	case note && usage:
		return i18n.T("Saved to Notes as recognised: your ChatGPT plan's usage is used up.")
	case usage:
		return i18n.T("Inserted as recognised: your ChatGPT plan's usage is used up.")
	case note && timeout:
		return i18n.T("Saved to Notes as recognised: the language model did not answer in time.")
	case note:
		return i18n.T("Saved to Notes as recognised: the language model failed.")
	case timeout:
		return i18n.T("Inserted as recognised: the language model did not answer in time.")
	}
	return i18n.T("Inserted as recognised: the language model failed.")
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
		return i18n.T("Restart the helper")
	case actionRetry:
		return i18n.T("Retry")
	}
	return ""
}
