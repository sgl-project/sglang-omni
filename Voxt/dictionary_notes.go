package main

// The Dictionary page's Personal dictionary view (Dictionary.Notes):
// Markdown the user writes once, which Transcribe, Translate and Rewrite
// send in the language model's system prompt, after the task's rules
// (compose.Input.Dictionary); speech recognition never gets it. It is
// saved as the Speech context is (autosave in models_context.go). Under it, How
// it is used builds a request from the settings, as a session does, and
// shows its parts and the messages sent.

import (
	"math"
	"strconv"
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// The Dictionary page's views (dictView.tab).
const (
	dictTermsTab = iota
	dictNotesTab
)

// notesPlaceholder is the editor's example: the triples of standard
// spelling, correction mapping and condition (issue #3815).
func notesPlaceholder() string {
	return i18n.T("| Standard spelling | Correction mapping | Condition |\n|---|---|---|\n| 101; I-80 | `one oh one` → 101 | Bay Area highways |\n| Roth | `wrath` → Roth | retirement accounts |\n| NVDA | — | tickers, all caps |")
}

// saveNotes saves the editor as typed.
func (a *app) saveNotes() {
	n := &a.dv.notes
	n.dirty = false
	n.seen = n.text
	if v := n.text; v != a.prefs.Dictionary.Notes {
		a.updateSettings(func(s *settings.Settings) { s.Dictionary.Notes = v })
	}
}

func (a *app) notesView(c *ui.Context) {
	t := c.Theme()
	n := &a.dv.notes
	if saved := a.prefs.Dictionary.Notes; !n.dirty && saved != n.seen {
		n.text, n.seen = saved, saved
	}
	ui.Scroll(c.Key("dict-notes-scroll")).Grow(1).MinHeight(0).Children(func() {
		ui.Column(c).Padding(px(t, sp2), px(t, sp6), px(t, sp6)).Gap(px(t, sp2)).Children(func() {
			caption(c, i18n.T("Write your names, terms and numbers once, in Markdown: for example a table of the standard spelling, what speech recognition writes instead, and when it applies. Transcribe, Translate and Rewrite add it to the language model's prompt. Speech recognition does not get it."))
			off := !a.prefs.Dictionary.Enabled
			formRow(c, "dict-notes-on-row", i18n.T("Add to the language model prompt"), "", func() {
				on := !a.prefs.Dictionary.NotesOff
				if ui.Switch(c.Key("dict-notes-on"), &on).Label(i18n.T("Add to the language model prompt")).Disabled(off).Changed() {
					v := !on
					a.updateSettings(func(s *settings.Settings) { s.Dictionary.NotesOff = v })
				}
			})
			if off {
				notice(c, "dict-notes-off", noticeInfo, i18n.T("The dictionary is off, so the personal dictionary is not sent."), "", nil, nil)
			}
			// As high as its text, however long (llm_debug.go): one scroll bar, the page's.
			in := ui.TextArea(c.Key("dict-notes"), &n.text).Label(i18n.T("Personal dictionary in Markdown")).Placeholder(notesPlaceholder()).Lines(8, math.MaxInt32)
			a.autosave(in, n, a.saveNotes)
			a.notesExample(c)
		})
	})
}

// notesExampleInput is a request of feature (hotkey.Transcribe,
// Translate, Rewrite) as a session builds it from the settings
// (session/run_listen.go, write.go), with the editor as typed and sample
// text; no app scene, which depends on the app in front.
func (a *app) notesExampleInput(feature string) compose.Input {
	p := a.prefs
	p.Dictionary.Notes = a.dv.notes.text
	f := p.Features
	in := compose.Input{
		MainLanguage: session.MainLanguage(p, i18n.Locale()), OtherLanguages: p.General.OtherLanguages,
		Dictionary: session.PromptDictionary(p),
	}
	if p.Dictionary.Enabled && p.Dictionary.Glossary {
		for _, tm := range a.dv.all {
			if tm.Enabled {
				in.Glossary = append(in.Glossary, compose.Term{Term: tm.Term, Aliases: tm.Aliases})
			}
		}
	}
	switch feature {
	case hotkey.Translate:
		in.Task, in.TaskPrompt, in.Cleanup, in.Text = compose.Translate, f.Translate.Prompt, f.Translate.Cleanup, trySample()
		in.TargetLanguage = session.TranslateTarget(p, i18n.Locale(), "")
	case hotkey.Rewrite:
		in.Task, in.TaskPrompt, in.Cleanup = compose.Rewrite, f.Rewrite.Prompt, f.Rewrite.Cleanup
		in.Instruction, in.Text = rewriteSample()
	default:
		in.Task, in.TaskPrompt, in.Cleanup, in.Structure, in.Text = compose.Transcribe, f.Transcribe.Prompt, f.Transcribe.Cleanup, f.Transcribe.Structure, trySample()
	}
	return in
}

// notesFeatures are How it is used's features, in its switch's order.
var notesFeatures = []string{hotkey.Transcribe, hotkey.Translate, hotkey.Rewrite}

// notesExample is How it is used: the system prompt, the dictionary, the
// user's input and the speech recognition text, then the messages sent.
func (a *app) notesExample(c *ui.Context) {
	t := c.Theme()
	d := &a.dv
	sectionTitle(c, i18n.T("How it is used"))
	caption(c, i18n.T("An example with your settings: system prompt + personal dictionary + your input + speech recognition text = what the language model receives. The dictionary comes right after the system prompt, before anything that changes from one request to the next, so the model server can reuse its cache."))
	segmentedFit(c.Key("dict-notes-feature"), &d.notesFeature, i18n.T("Transcribe"), i18n.T("Translate"), i18n.T("Rewrite")).Label(i18n.T("Example for"))
	feature := notesFeatures[d.notesFeature]
	if feature == hotkey.Transcribe && !a.prefs.Features.Transcribe.Enhance {
		caption(c, i18n.T("Clean-up is off, so nothing is sent to a language model."))
	}
	in := a.notesExampleInput(feature)
	out, err := compose.Build(in)
	if err != nil {
		caption(c, err.Error())
		return
	}
	without := in
	without.Dictionary = ""
	plain, _ := compose.Build(without)
	system, dict := plain.System, out.Dictionary
	none := i18n.T("(none)")
	or := func(s string) string {
		if strings.TrimSpace(s) == "" {
			return none
		}
		return s
	}
	userInput, speech := none, in.Text
	if in.Task == compose.Rewrite {
		userInput, speech = in.Text, in.Instruction
	}
	if dict == "" {
		dict = i18n.T("(not sent: empty or switched off)")
	}
	ui.Column(c.Key("dict-notes-example")).Gap(px(t, sp2)).Children(func() {
		promptCard(c, "dict-ex-system", i18n.T("System prompt"), system)
		promptCard(c, "dict-ex-dict", "+ "+i18n.T("Personal dictionary"), dict)
		promptCard(c, "dict-ex-input", "+ "+i18n.T("Your input (the selected text)"), or(userInput))
		promptCard(c, "dict-ex-speech", "+ "+i18n.T("Speech recognition text"), or(speech))
		sectionTitle(c, "= "+i18n.T("What the language model receives"))
		for i, m := range out.Messages() {
			promptCard(c, "dict-ex-msg-"+strconv.Itoa(i), m.Role, m.Content)
		}
	})
}

// promptCard is a part of a request: its title and its text, in a card.
func promptCard(c *ui.Context, key, title, text string) {
	t := c.Theme()
	k := tok(t)
	ui.Column(c.Key(key)).Gap(px(t, 4)).Children(func() {
		ui.Text(c, title).FontSize(px(t, fsCaption)).FontWeight(600).TextColor(k.Text2)
		fillCard(c).Padding(px(t, sp3)).Children(func() {
			ui.Text(c, text).FontSize(px(t, fsCaption)).Font("monospace").Selectable()
		})
	})
}
