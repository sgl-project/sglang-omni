package main

// The language model debug page's controller. It builds requests, runs the
// chosen remote model, and saves an edited prompt only on explicit use.

import (
	"context"
	"slices"
	"strings"
	"time"

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

// llmDebugUI is the page's editable widget state.
type llmDebugUI struct {
	model, preset      string // the menus' choices, by title
	filled             string // the preset the prompt was filled from
	prompt             string // as edited here
	input, instruction string
}

// llmDebugState adds the controller's in-flight and completed run state.
type llmDebugState struct {
	llmDebugUI
	running bool
	res     *llmDebugResult
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
		Text: strings.TrimSpace(u.input), Dictionary: session.PromptDictionary(a.prefs),
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

func (a *app) llmDebugSetups() []settings.Provider {
	return providers.Ready(a.prefs, func(id string) bool { return a.hasKey(local.LLM, id) })
}

func (a *app) prepareLLMDebug() {
	u := &a.llmDbg
	setups := a.llmDebugSetups()
	if len(setups) == 0 {
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
	if u.filled != u.preset {
		p := presets[slices.Index(titles, u.preset)]
		u.prompt, u.filled, u.res = a.savedPrompt(p), u.preset, nil
	}
}

func (a *app) runRemoteDebug() {
	a.prepareLLMDebug()
	setups := a.llmDebugSetups()
	if len(setups) == 0 {
		return
	}
	models := make([]string, len(setups))
	for i, p := range setups {
		models[i] = setupName(local.LLM, p)
	}
	presets := a.llmDebugPresets()
	titles := make([]string, len(presets))
	for i, p := range presets {
		titles[i] = p.title
	}
	a.runLLMDebug(presets[slices.Index(titles, a.llmDbg.preset)], setups[slices.Index(models, a.llmDbg.model)])
}

func (a *app) useRemoteDebugPrompt() {
	for _, p := range a.llmDebugPresets() {
		if p.title == a.llmDbg.preset {
			a.useLLMDebugPrompt(p)
			return
		}
	}
}

func (a *app) dismissLLMDebugResult() { a.llmDbg.res = nil }

func (a *app) openRemoteLanguageModels() { a.navigate(remotePath) }
