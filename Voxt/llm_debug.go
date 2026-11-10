package main

// The language model debug page (D3, Voxt 1.x's LLM debug window): the
// request of Transcribe, Translate, Rewrite or an App enhancement group,
// for typed text, sent to a language model chosen here, with its prompt
// edited here and not saved; it shows the answer, its time and the
// messages sent. "Use this prompt" saves the edit as the feature's rules
// or the group's prompt. The command palette and the Language models
// page open it.

import (
	"context"
	"fmt"
	"math"
	"slices"
	"strings"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/appctx"
	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

const llmDebugPath = "/llm-debug"

// llmDebugTimeout bounds a run: a slow model's long answer.
const llmDebugTimeout = 2 * time.Minute

// llmDebugPromptLines is the prompt field's height when the prompt is
// short; it grows with its text from there.
const llmDebugPromptLines = 8

// llmDebugUI is the page's state.
type llmDebugUI struct {
	model, preset      string // the menus' choices, by title
	filled             string // the preset the prompt was filled from
	prompt             string // as edited here
	input, instruction string
	running            bool
	res                *llmDebugResult
}

type llmDebugResult struct {
	answer, err string
	took        time.Duration
	msgs        []compose.Message
}

// llmDebugPreset is a request the page sends: a feature's, or
// Transcribe's with an App enhancement group (its id) as the scene.
type llmDebugPreset struct {
	title, feature, group string
}

func (a *app) llmDebugPresets() []llmDebugPreset {
	ps := []llmDebugPreset{
		{title: i18n.T("Transcribe"), feature: hotkey.Transcribe},
		{title: i18n.T("Translate"), feature: hotkey.Translate},
		{title: i18n.T("Rewrite"), feature: hotkey.Rewrite},
	}
	for _, g := range a.aeRules.Groups {
		ps = append(ps, llmDebugPreset{title: i18n.T("App enhancement: %s", g.Name), feature: hotkey.Transcribe, group: g.ID})
	}
	return ps
}

// savedPrompt is p's prompt as saved: the group's, or the feature's rules
// (the built-in prompt when it has none).
func (a *app) savedPrompt(p llmDebugPreset) string {
	if p.group != "" {
		g, _ := a.aeRules.Group(p.group)
		return g.Prompt
	}
	if s := promptOf(a.prefs, p.feature); s != "" {
		return s
	}
	return a.promptBuiltin(p.feature)
}

// llmDebugInput is p's request for the typed text, with prompt in place
// of the saved one, as a session builds it (session/write.go).
func (a *app) llmDebugInput(p llmDebugPreset, prompt string) compose.Input {
	u, f := &a.llmDbg, a.prefs.Features
	in := compose.Input{
		MainLanguage: session.MainLanguage(a.prefs, i18n.Locale()), OtherLanguages: a.prefs.General.OtherLanguages,
		Text: strings.TrimSpace(u.input),
	}
	rules := prompt
	if p.group != "" {
		g, _ := a.aeRules.Group(p.group)
		in.Scene = &compose.Scene{AppName: g.Name, Prompt: prompt}
		rules = f.Transcribe.Prompt
	} else if strings.TrimSpace(prompt) == strings.TrimSpace(a.promptBuiltin(p.feature)) {
		rules = "" // note (luojiaxuan): the built-in prompt is built in layers, not sent as rules
	}
	switch p.feature {
	case hotkey.Translate:
		in.Task, in.Cleanup, in.TaskPrompt = compose.Translate, f.Translate.Cleanup, rules
		in.TargetLanguage = session.TranslateTarget(a.prefs, i18n.Locale(), "")
	case hotkey.Rewrite:
		in.Task, in.Cleanup, in.TaskPrompt = compose.Rewrite, f.Rewrite.Cleanup, rules
		in.Instruction, in.Answer = strings.TrimSpace(u.instruction), in.Text == ""
	default:
		in.Task, in.Cleanup, in.Structure, in.TaskPrompt = compose.Transcribe, f.Transcribe.Cleanup, f.Transcribe.Structure, rules
	}
	return in
}

// runLLMDebug sends p's request to setup, off the main thread.
func (a *app) runLLMDebug(p llmDebugPreset, setup settings.Provider) {
	u := &a.llmDbg
	in := a.llmDebugInput(p, u.prompt)
	out, err := compose.Build(in)
	if err != nil {
		u.res = &llmDebugResult{err: err.Error()}
		return
	}
	msgs := out.Messages()
	req := llm.Request{Messages: make([]llm.Message, len(msgs))}
	for i, m := range msgs {
		req.Messages[i] = llm.Message{Role: llm.Role(m.Role), Content: m.Content}
	}
	if in.Answer {
		req.Format = llm.FormatJSON
	}
	u.running, u.res = true, nil
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), llmDebugTimeout)
		defer cancel()
		chat, err := a.openRemote(ctx, setup)
		var resp llm.Response
		if err == nil {
			resp, err = chat.Chat(ctx, req)
		}
		a.post(func() {
			u.running = false
			u.res = &llmDebugResult{answer: resp.Text, took: resp.Latency, msgs: msgs}
			if err != nil {
				u.res.err = errText(err)
			}
		})
	})
}

// useLLMDebugPrompt saves the prompt as edited as p's.
func (a *app) useLLMDebugPrompt(p llmDebugPreset) {
	v := a.llmDbg.prompt
	if p.group != "" {
		a.updateGroup(p.group, func(g *appctx.Group) { g.Prompt = v })
	} else {
		u := a.fui(p.feature)
		u.prompt, u.promptDirty = v, true
		a.savePrompt(p.feature)
	}
	a.toast(i18n.T("Prompt saved"))
}

func (a *app) llmDebugPage(c *ui.Context) {
	t := c.Theme()
	k := tok(t)
	u := &a.llmDbg
	columnPageHeader(c, i18n.T("Language model debug"), nil)
	ui.Scroll(c.Key("llmdbg-scroll")).Grow(1).MinHeight(0).Children(func() {
		pageColumn(c).Gap(px(t, sp4)).Children(func() {
			caption(c, i18n.T("Send your own text to a language model as a feature would, and try edits to its prompt here. Nothing is saved until you click Use this prompt."))
			setups := providers.Ready(a.prefs, func(id string) bool { return a.hasKey(local.LLM, id) })
			if len(setups) == 0 {
				caption(c, i18n.T("Set up a language model first."))
				if ui.Button(c.Key("llmdbg-setup"), i18n.T("Open Language models")).AlignSelf(ui.Start).Clicked() {
					a.navigate(remotePath)
				}
				return
			}
			models := make([]string, len(setups))
			for i, p := range setups {
				models[i] = setupName(local.LLM, p)
			}
			if !slices.Contains(models, u.model) {
				u.model = models[0]
			}
			presets := a.llmDebugPresets()
			titles := make([]string, len(presets))
			for i, p := range presets {
				titles[i] = p.title
			}
			if !slices.Contains(titles, u.preset) {
				u.preset = titles[0]
			}
			ui.Row(c).Gap(px(t, sp3)).Wrap().Children(func() {
				ui.Select(c.Key("llmdbg-model"), &u.model, models).Label(i18n.T("Language model")).Width(px(t, 300))
				ui.Select(c.Key("llmdbg-preset"), &u.preset, titles).Label(i18n.T("Request")).Width(px(t, 300))
			})
			p := presets[slices.Index(titles, u.preset)]
			saved := a.savedPrompt(p)
			if u.filled != u.preset {
				u.prompt, u.filled, u.res = saved, u.preset, nil
			}
			label := i18n.T("System prompt")
			if p.group != "" {
				label = i18n.T("Group prompt")
				caption(c, i18n.T("This is the group's prompt from App enhancement. Edits apply only here until you click Use this prompt, which saves them to the group."))
			} else {
				caption(c, i18n.T("This is the feature's Your rules, the same prompt as on its page. Edits apply only here until you click Use this prompt, which saves them there."))
			}
			// As high as its text, however long, as Your rules on a
			// feature's page: a long prompt (tens of thousands of
			// characters) scrolled in a box of its own inside the page,
			// two scroll bars (11-09).
			ui.TextArea(c.Key("llmdbg-prompt"), &u.prompt).Label(label).Lines(llmDebugPromptLines, math.MaxInt32)
			ui.Row(c).Gap(px(t, sp2)).Children(func() {
				edited := u.prompt != saved
				if ui.Button(c.Key("llmdbg-use"), i18n.T("Use this prompt")).Disabled(!edited).Clicked() {
					a.useLLMDebugPrompt(p)
				}
				if ghostButton(c.Key("llmdbg-reset"), i18n.T("Back to the saved prompt")).Disabled(!edited).Clicked() {
					u.prompt = saved
				}
			})
			input := i18n.T("Recognised text")
			switch p.feature {
			case hotkey.Translate:
				input = i18n.T("Text to translate")
			case hotkey.Rewrite:
				input = i18n.T("Selected text (optional)")
				ui.TextInput(c.Key("llmdbg-instruction"), &u.instruction).Label(i18n.T("Spoken instruction")).
					Placeholder(i18n.T("um make it shorter, no, more formal"))
			}
			ui.TextArea(c.Key("llmdbg-input"), &u.input).Label(input).Lines(4, math.MaxInt32)
			ready := strings.TrimSpace(u.input) != "" || p.feature == hotkey.Rewrite && strings.TrimSpace(u.instruction) != ""
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
				if primaryButton(c.Key("llmdbg-run"), i18n.T("Run")).Disabled(u.running || !ready).Clicked() {
					a.runLLMDebug(p, setups[slices.Index(models, u.model)])
				}
				if u.running {
					ui.Spinner(c).Size(px(t, 16), px(t, 16))
					caption(c, i18n.T("Waiting for the answer…"))
				}
			})
			r := u.res
			if r == nil {
				return
			}
			if r.err != "" {
				notice(c, "llmdbg-err", noticeError, r.err, "", nil, func() { u.res = nil })
			} else {
				sectionTitle(c, i18n.T("Answer in %s", fmt.Sprintf("%.1f s", r.took.Seconds())))
				fillCard(c.Key("llmdbg-answer")).Padding(px(t, sp3)).Children(func() {
					ui.Text(c, r.answer).Selectable()
				})
			}
			if len(r.msgs) == 0 {
				return
			}
			sectionTitle(c, i18n.T("Messages sent"))
			for i, m := range r.msgs {
				ui.Column(c.Key(fmt.Sprintf("llmdbg-msg-%d", i))).Gap(px(t, 4)).Children(func() {
					ui.Text(c, m.Role).FontSize(px(t, fsCaption)).FontWeight(600).TextColor(k.Text2)
					fillCard(c).Padding(px(t, sp3)).Children(func() {
						ui.Text(c, m.Content).FontSize(px(t, fsCaption)).Font("monospace").Selectable()
					})
				})
			}
		})
	})
}
