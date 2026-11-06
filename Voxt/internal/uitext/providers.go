package uitext

// The remote language models' words (the Language models page, the
// features' model menus, History).

import (
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// PresetName is a remote service's name: its brand, untranslated, but in
// Chinese for the Chinese services (as the team and Voxt 1.x call them)
// and the custom server.
func PresetName(id string) string {
	switch id {
	case llm.Custom:
		return i18n.T("Custom (OpenAI-compatible)")
	case "zhipu":
		return i18n.T("Zhipu GLM")
	case "volcengine":
		return i18n.T("Volcengine Ark (Doubao)")
	case "dashscope":
		return i18n.T("Qwen (Alibaba Cloud Model Studio)")
	case "moonshot":
		return i18n.T("Moonshot (Kimi)")
	case "stepfun":
		return i18n.T("StepFun")
	case "mimo":
		return i18n.T("Xiaomi MiMo")
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
