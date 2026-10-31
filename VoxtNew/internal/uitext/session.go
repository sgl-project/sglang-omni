package uitext

// The words of a session (the overlay, the answer card, the Try area):
// what it is doing, its feature, the group that matched, why its answer
// is in the card, its timings, the delivery, the target language.

import (
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/appctx"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/compose"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/deliver"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// DeliveryText says why a result was not inserted.
func DeliveryText(res deliver.Result, target proto.App) string {
	name := target.Name
	if name == "" {
		name = i18n.T("the app")
	}
	switch res.Reason {
	case deliver.ReasonNoAccess:
		return i18n.T("Allow Accessibility to insert results where you type.")
	case deliver.ReasonNotFront, deliver.ReasonNoTarget:
		return i18n.T("%s is not in front any more.", name)
	case deliver.ReasonNotEditable:
		return i18n.T("No text field is focused: click where the text should go, then insert.")
	case deliver.ReasonSecure:
		return i18n.T("A password field is focused: VoxtNew never types into one.")
	case deliver.ReasonFailed:
		return i18n.T("Could not insert: %v", res.Err)
	case deliver.ReasonUnconfirmed:
		return i18n.T("The text did not show up where you type: it is on the clipboard, press ⌘V.")
	case deliver.ReasonSecureInput:
		if s := res.Paste.SecureInput; s != nil && s.Name != "" {
			return i18n.T("Secure input is on in %s, which can block the keys VoxtNew sends: the text is on the clipboard, press ⌘V.", s.Name)
		}
		return i18n.T("Secure input is on, which can block the keys VoxtNew sends: the text is on the clipboard, press ⌘V.")
	}
	return ""
}

// SceneLabel is the app enhancement group a session falls in, as the
// history and the pill's description say it: "Slack · Team chat"; ""
// for none.
func SceneLabel(s session.Snapshot) string {
	if s.Scene == nil || !s.Scene.Matched() {
		return ""
	}
	app := s.Target.Name
	if s.Scene.Source == appctx.SourceURL && s.Scene.Host != "" {
		app = s.Scene.Host // a website rule: the site, not the browser
	}
	if app == "" {
		return s.Scene.GroupName()
	}
	return app + " · " + s.Scene.GroupName()
}

// SceneShort is the group's name alone, as the pill shows it: the user
// knows which app they are in.
func SceneShort(s session.Snapshot) string {
	if s.Scene == nil || !s.Scene.Matched() {
		return ""
	}
	return s.Scene.GroupName()
}

// WorkingText says what a session does after recording (the loader's
// description).
func WorkingText(s session.Snapshot) string {
	switch s.State {
	case session.Reading:
		return i18n.T("Reading the selection…")
	case session.Composing:
		switch {
		case s.Kind == store.KindTranslate && s.Selection != "":
			return i18n.T("Translating the selection…")
		case s.Kind == store.KindTranslate:
			return i18n.T("Translating…")
		case s.Kind == store.KindRewrite && s.Selection != "":
			return i18n.T("Rewriting the selection…")
		case s.Kind == store.KindRewrite:
			return i18n.T("Writing…")
		}
		return i18n.T("Cleaning up…")
	case session.Delivering:
		return i18n.T("Inserting…")
	}
	return i18n.T("Recognizing…")
}

// WaitText says what a session waits for (session.Snapshot.Waiting): the
// starting icon's accessibility label, and the pill's line for WaitMic
// until the live text comes; "" for nothing.
func WaitText(s session.Snapshot) string {
	switch {
	case s.Waiting == session.WaitMic && s.State == session.Recording:
		return i18n.T("Starting the microphone…")
	case s.Waiting == session.WaitModel && (s.State == session.Recording || s.State == session.Recognizing):
		return i18n.T("Loading the speech model…")
	}
	return ""
}

func KindTitle(k store.Kind) string {
	switch k {
	case store.KindTranslate:
		return i18n.T("Translate")
	case store.KindRewrite:
		return i18n.T("Rewrite")
	}
	return i18n.T("Transcribe")
}

// TimingsText is a try's stage times and models: "Recording 3.2 s ·
// Recognition 0.8 s · Clean-up 1.1 s · Accurate speech recognition,
// Balanced language model".
func TimingsText(s session.Snapshot) string {
	secs := func(d time.Duration) string { return i18n.T("%.1f s", d.Seconds()) }
	var parts []string
	tm := s.Timings
	if tm.Audio > 0 {
		parts = append(parts, i18n.T("Recording %s", secs(tm.Audio)))
	}
	if tm.ASR > 0 {
		parts = append(parts, i18n.T("Recognition %s", secs(tm.ASR)))
	}
	if tm.LLM > 0 {
		switch s.Kind {
		case store.KindTranslate:
			parts = append(parts, i18n.T("Translation %s", secs(tm.LLM)))
		case store.KindRewrite:
			parts = append(parts, i18n.T("Writing %s", secs(tm.LLM)))
		default:
			parts = append(parts, i18n.T("Clean-up %s", secs(tm.LLM)))
		}
	}
	if tm.Total > 0 {
		parts = append(parts, i18n.T("Total %s", secs(tm.Total)))
	}
	var svc []string
	for _, m := range []session.Model{s.ASR, s.LLM} {
		if m.Name != "" {
			svc = append(svc, m.Name)
		}
	}
	if len(svc) > 0 {
		parts = append(parts, strings.Join(svc, ", "))
	}
	return strings.Join(parts, " · ")
}

// CardTitle is the card's heading: a Rewrite answer's own title, else
// what the card holds.
func CardTitle(kind store.Kind, title string) string {
	if t := strings.TrimSpace(title); t != "" {
		return t
	}
	switch kind {
	case store.KindTranslate:
		return i18n.T("Translation")
	case store.KindRewrite:
		return i18n.T("AI Answer") // Voxt's, a rewritten selection's too
	}
	return i18n.T("Transcription")
}

// TargetLanguageLabel names a target language in the UI's language.
func TargetLanguageLabel(code string) string {
	return compose.LanguageName(code, i18n.Locale())
}

// WhyInCard says why a result is in the card and not in place, for
// Insert's tip ("" when the user asked for the card, or it went there
// for its length). name is the start app's: not being in front matters
// only for it (a selection made there, or VoxtNew in front).
func WhyInCard(r deliver.Reason, selection bool, name string) string {
	if name == "" {
		name = i18n.T("the app")
	}
	switch r {
	case deliver.ReasonNotFront, deliver.ReasonNoTarget:
		return i18n.T("%s is not in front any more: insert it there, or copy it.", name)
	case deliver.ReasonNotEditable:
		if selection {
			return i18n.T("The selection in %s cannot be edited: copy the result, or click where it should go and insert.", name)
		}
		return i18n.T("No text field is focused: click where it should go, then insert, or copy it.")
	case deliver.ReasonSecure:
		return i18n.T("A password field is focused, and VoxtNew never types into one: copy it, or click another field and insert.")
	case deliver.ReasonNoAccess:
		return i18n.T("Allow Accessibility to insert results where you type.")
	case deliver.ReasonUnconfirmed:
		return i18n.T("The text did not show up where you type. It is on the clipboard: press ⌘V, or insert it again.")
	case deliver.ReasonSecureInput:
		return i18n.T("Secure input is on in another app, which can block the keys VoxtNew sends. The text is on the clipboard: press ⌘V, or insert it again.")
	}
	return ""
}
