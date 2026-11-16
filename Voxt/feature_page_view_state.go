package main

type featureViewState struct {
	Feature          string                    `json:"feature"`
	Prefix           string                    `json:"prefix"`
	Title            string                    `json:"title"`
	Caption          string                    `json:"caption"`
	RefreshRecording bool                      `json:"refresh_recording"`
	Shortcuts        featureShortcutsViewState `json:"shortcuts"`
	Models           featureModelsViewState    `json:"models"`
	Prompt           featurePromptViewState    `json:"prompt"`
	Try              tryCardViewState          `json:"try"`
}
