package uitext

// The remote language models' words (the Language models page, the
// features' model menus, History).

import (
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// PresetName is a remote service's name: its brand, untranslated, but
// the custom server's.
func PresetName(id string) string {
	if id == llm.Custom {
		return i18n.T("Custom (OpenAI-compatible)")
	}
	if pr, ok := llm.LookupPreset(id); ok {
		return pr.Name
	}
	return id
}

// ProviderName is how a remote model is named where a model is chosen or
// shown: its service and model, "OpenAI · gpt-4.1-mini".
func ProviderName(p settings.Provider) string {
	return PresetName(p.ID) + " · " + providers.Model(p)
}
