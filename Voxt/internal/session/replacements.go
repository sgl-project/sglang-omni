package session

import (
	"slices"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dict"
)

func (r *run) replaceWords(text string) string {
	rules, _ := dict.ParseReplacements(r.settings.Dictionary.Replacements)
	return rules.Replace(text)
}

func (r *run) localTranscript(text string) string {
	text = r.replaceWords(text)
	if r.settings.Features.Transcribe.LocalStructure {
		return compose.StructureSpoken(text)
	}
	return text
}

func (r *run) replaceInput(in compose.Input) compose.Input {
	return ReplacePromptText(in, r.settings.Dictionary.Replacements)
}

// ReplacePromptText applies the saved rules to user text before the composer
// splits long transcripts. Notes and earlier turns can carry the same phrases.
func ReplacePromptText(in compose.Input, replacements string) compose.Input {
	if replacements == "" {
		return in
	}
	rules, _ := dict.ParseReplacements(replacements)
	in.Text = rules.Replace(in.Text)
	in.Instruction = rules.Replace(in.Instruction)
	in.TaskPrompt = rules.Replace(in.TaskPrompt)
	in.Dictionary = rules.Replace(in.Dictionary)
	if in.Scene != nil {
		scene := *in.Scene
		scene.AppName, scene.Prompt = rules.Replace(scene.AppName), rules.Replace(scene.Prompt)
		in.Scene = &scene
	}
	in.Glossary = slices.Clone(in.Glossary)
	for i := range in.Glossary {
		term := &in.Glossary[i]
		term.Term = rules.Replace(term.Term)
		term.Aliases = slices.Clone(term.Aliases)
		for j, alias := range term.Aliases {
			term.Aliases[j] = rules.Replace(alias)
		}
	}
	in.Previous = slices.Clone(in.Previous)
	for i := range in.Previous {
		turn := &in.Previous[i]
		turn.Instruction, turn.Selection = rules.Replace(turn.Instruction), rules.Replace(turn.Selection)
		turn.Answer.Title, turn.Answer.Content = rules.Replace(turn.Answer.Title), rules.Replace(turn.Answer.Content)
	}
	return in
}
