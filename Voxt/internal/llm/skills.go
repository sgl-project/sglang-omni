package llm

// Skill is a task the app routes to a model.
type Skill string

const (
	// SkillTranscribeEnhance cleans up a transcript (fillers, self-
	// corrections, punctuation), with the scene and glossary layers.
	SkillTranscribeEnhance Skill = "transcribe_enhance"
	// SkillTranslateText translates speech or selected text.
	SkillTranslateText Skill = "translate_text"
	// SkillRewrite rewrites the selection as told, or writes from a spoken
	// prompt.
	SkillRewrite Skill = "rewrite"
)

// Skills lists every skill, in settings order.
func Skills() []Skill {
	return []Skill{SkillTranscribeEnhance, SkillTranslateText, SkillRewrite}
}

// DefaultTemperature is the sampling temperature a skill uses when neither
// the request nor the model sets one: low for clean-up and translation,
// which must keep to the input, higher for rewriting and writing, which
// may choose words. (Voxt leaves it to the model's default.)
func DefaultTemperature(s Skill) float64 {
	switch s {
	case SkillRewrite:
		return 0.5
	case SkillTranslateText:
		return 0.2
	}
	return 0.1
}
