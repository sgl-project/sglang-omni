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
