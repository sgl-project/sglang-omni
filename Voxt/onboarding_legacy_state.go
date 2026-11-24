package main

import (
	"github.com/hehehai/actnowkit/shortcuts"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
)

func onboardingServiceName(translator *i18n.Translator, kind local.Kind, id string) string {
	if kind == local.LLM {
		switch id {
		case llm.Custom:
			return translator.T("Custom (OpenAI-compatible)")
		case "zhipu":
			return translator.T("Zhipu GLM")
		case "volcengine":
			return translator.T("Volcengine Ark (Doubao)")
		case "dashscope":
			return translator.T("Qwen (Alibaba Cloud Model Studio)")
		case "moonshot":
			return translator.T("Moonshot (Kimi)")
		case "stepfun":
			return translator.T("StepFun")
		case "mimo":
			return translator.T("Xiaomi MiMo")
		}
		if preset, ok := llm.LookupPreset(id); ok {
			return preset.Name
		}
		return id
	}
	switch id {
	case "doubao":
		return translator.T("Doubao (Volcengine)")
	case providers.ZhipuASR:
		return translator.T("Zhipu GLM-ASR")
	case "aliyun":
		return translator.T("Aliyun Bailian ASR")
	case providers.StepFunASR:
		return translator.T("StepFun")
	case "mimo":
		return translator.T("Xiaomi MiMo")
	}
	if preset, ok := providers.LookupSpeech(id); ok {
		return preset.Name
	}
	return id
}

func onboardingFamilyFunction(translator *i18n.Translator, shortcut shortcuts.Shortcut) string {
	switch shortcut.ID {
	case "transcribe":
		return translator.T("Transcribe")
	case "translate":
		return translator.T("Translate")
	case "rewrite":
		return translator.T("Rewrite")
	case "pasteLast":
		return translator.T("Paste last")
	case "dictation":
		return translator.T("Dictation")
	}
	if shortcut.Title != "" {
		return shortcut.Title
	}
	return shortcut.ID
}

func onboardingConflictLines(translator *i18n.Translator, conflicts []shortcuts.Conflict, feature string, index int) []string {
	id := shortcutID(feature, index)
	var lines []string
	for _, conflict := range conflicts {
		if conflict.Mine.ID != id {
			continue
		}
		line := translator.T("Also used by %s (%s)", familyAppName(conflict.Other), onboardingFamilyFunction(translator, conflict.Theirs))
		seen := false
		for _, existing := range lines {
			if existing == line {
				seen = true
			}
		}
		if !seen {
			lines = append(lines, line)
		}
	}
	return lines
}
