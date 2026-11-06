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
			return i18n.T("The language model failed: %s", msg), more, actionLanguage
		}
		_, more := uitext.ModelErrorText(p.Err)
		var le *llm.Error
		if errors.Is(p.Err, context.DeadlineExceeded) || errors.As(p.Err, &le) && le.Code == llm.CodeTimeout {
			return i18n.T("Inserted as recognised: the language model did not answer in time."), more, actionNone
		}
		return i18n.T("Inserted as recognised: the language model failed."), more, actionNone
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
func noticeActionLabel(action string) string {
	switch action {
	case actionModels:
		return i18n.T("Open Local models")
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
