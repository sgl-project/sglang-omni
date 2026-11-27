package main

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func cleanupOffFeatures(selection string) string {
	return `{"transcription":{"asrSelectionID":"dictation","llmEnabled":false,"llmSelectionID":"` + selection + `","prompt":""}}`
}

const openAISetup = `[{"providerID":"openAI","model":"gpt-5.2","endpoint":"https://api.openai.com/v1/responses","storedCredentialPresence":{"apiKey":true}}]`

func TestLegacyCleanupOffApply(t *testing.T) {
	for _, c := range []struct {
		name     string
		features string
		chosen   string
		enhance  bool
	}{
		{"off, nothing chosen", cleanupOffFeatures("remote-llm:openAI"), "", false},
		{"off, a model chosen here", cleanupOffFeatures("remote-llm:openAI"), "deepseek", true},
		{"on", `{"transcription":{"llmEnabled":true,"llmSelectionID":"remote-llm:openAI"}}`, "", true},
		{"not saved", `{"transcription":{"llmSelectionID":"remote-llm:openAI"}}`, "", true},
	} {
		l := parseLegacySettings(openAISetup, "", c.features, "")
		s := settings.Defaults()
		s.Features.Transcribe.LLM = c.chosen
		l.apply(&s, "")
		if s.Features.Transcribe.Enhance != c.enhance {
			t.Errorf("%s: Enhance %v, want %v", c.name, s.Features.Transcribe.Enhance, c.enhance)
		}
	}
}

// Import again keeps what the user chose after the first import, also when
// the first left Transcribe with no language model chosen (a local model
// there), where the "none chosen" rule alone would apply again.
func TestLegacyCleanupOffNotReappliedByImportAgain(t *testing.T) {
	for _, selection := range []string{"remote-llm:openAI", "local-llm:mlx-community/Qwen3.5-4B-OptiQ-4bit"} {
		fakeLegacyPlutil(t, map[string]string{"remoteLLMProviderConfigurations": openAISetup, "featureSettings": cleanupOffFeatures(selection)})
		a, _, _ := newTestApp(t)
		lib := copyFixture(t, "real")
		a.importLegacyFrom(lib)
		if a.lib.settings.Get().Features.Transcribe.Enhance {
			t.Fatalf("%s: clean-up came back on after the import", selection)
		}
		if err := a.lib.settings.Update(context.Background(), func(s *settings.Settings) {
			s.Features.Transcribe.Enhance = true
			s.Features.Transcribe.LLM = ""
		}); err != nil {
			t.Fatal(err)
		}
		a.importLegacyFrom(lib)
		if !a.lib.settings.Get().Features.Transcribe.Enhance {
			t.Errorf("%s: importing again undid the user's choice", selection)
		}
	}
}

// An import that could not read the switch has not decided anything: the
// one that can still applies it.
func TestLegacyCleanupOffWaitsForReadableFeatures(t *testing.T) {
	fakeLegacyPlutil(t, map[string]string{"remoteLLMProviderConfigurations": openAISetup})
	a, _, _ := newTestApp(t)
	lib := copyFixture(t, "real")
	a.importLegacyFrom(lib)
	if !a.lib.settings.Get().Features.Transcribe.Enhance {
		t.Fatal("clean-up off with no features read")
	}
	fakeLegacyPlutil(t, map[string]string{"remoteLLMProviderConfigurations": openAISetup, "featureSettings": cleanupOffFeatures("local-llm:x")})
	a.importLegacyFrom(lib)
	if a.lib.settings.Get().Features.Transcribe.Enhance {
		t.Error("the switch was not applied once it could be read")
	}
}

func TestLegacyCodexLeftOut(t *testing.T) {
	const codexSetup = `[{"providerID":"codex","model":"gpt-5.4","storedCredentialPresence":{"apiKey":false}}]`
	for _, c := range []struct {
		name, setups, features string
		want                   []string
		absent                 []string
	}{
		{"setup only", codexSetup, "", []string{"Codex setup: sign in to ChatGPT (Codex) in Settings → Language models"}, []string{"used by", "not available"}},
		{"all features", codexSetup, `{"transcription":{"llmEnabled":true,"llmSelectionID":"remote-llm:codex"},"translation":{"modelSelectionID":"remote-llm:codex"},"rewrite":{"llmSelectionID":"remote-llm:codex"}}`,
			[]string{"Codex (used by Transcribe, Translate, Rewrite): sign in to ChatGPT (Codex) or choose another language model"}, []string{"Codex setup", "not available"}},
		{"Transcribe had clean-up off", codexSetup, `{"transcription":{"llmEnabled":false,"llmSelectionID":"remote-llm:codex"},"rewrite":{"llmSelectionID":"remote-llm:codex"}}`,
			[]string{"used by Rewrite"}, []string{"Transcribe"}},
		{"no Codex", openAISetup, "", nil, []string{"Codex"}},
	} {
		fakeLegacyPlutil(t, map[string]string{"remoteLLMProviderConfigurations": c.setups, "featureSettings": c.features})
		a, _, _ := newTestApp(t)
		a.importLegacyFrom(copyFixture(t, "real"))
		left := strings.Join(a.ob.legacy.skipped, "; ")
		for _, w := range c.want {
			if !strings.Contains(left, w) {
				t.Errorf("%s: left out %q lacks %q", c.name, left, w)
			}
		}
		for _, w := range c.absent {
			if strings.Contains(left, w) {
				t.Errorf("%s: left out %q has %q", c.name, left, w)
			}
		}
	}
	doc, err := os.ReadFile("docs/voxt-migration.md")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(doc), "has no Codex") || !strings.Contains(string(doc), "ChatGPT (Codex)") {
		t.Error("docs/voxt-migration.md still says Voxt 2 has no Codex, or does not name ChatGPT (Codex)")
	}
}

func TestLegacyHTTPGatewayDocs(t *testing.T) {
	for _, check := range []struct {
		file        string
		wants, olds []string
	}{
		{"docs/voxt-migration.md", []string{"API-key language models can use HTTP on another host", "the Address tip in the service's set-up dialog says so, and the import's result and the Language models settings show no warning"}, []string{"Voxt 2 only connects over https, or http to this Mac", "the Language models page warns that HTTP sends the API key and text without encryption"}},
		{"docs/manual-test.md", []string{"另一台机器的 http 地址可以继续使用", "结果和语言模型设置里都没有 HTTP 的警告"}, []string{"另一台机器的 http 地址写“要改", "语言模型页说明 HTTP 会不加密地发送 API Key 和文字"}},
		{"docs/behavior.md", []string{"API-key 服务接受 http 或 https", "页面和卡片上没有警告"}, []string{"地址只接受 https 或指向本机的 http", "http 不加密地发送 API key 和文字，页面给出警告"}},
		{"docs/architecture.md", []string{"API-key services: http or https; Codex: https off this Mac"}, []string{"ProviderConfig.Resolve (https, or http to this Mac)"}},
		{"CHANGELOG.md", []string{"which the Address tip in the service's set-up dialog says"}, []string{"Voxt warns that HTTP sends the key and text without encryption"}},
	} {
		doc, err := os.ReadFile(check.file)
		if err != nil {
			t.Fatal(err)
		}
		text := string(doc)
		for _, want := range check.wants {
			if !strings.Contains(text, want) {
				t.Errorf("%s lacks %q", check.file, want)
			}
		}
		for _, old := range check.olds {
			if strings.Contains(text, old) {
				t.Errorf("%s still says %q", check.file, old)
			}
		}
	}
	// Only the Address tip in the set-up dialog mentions plain HTTP; no
	// document may claim the Language models page or the import result
	// warns about it (remote_dialog_view_state.go draws the tip).
	for file, claim := range map[string]string{
		"docs/voxt-migration.md": "the Language models page warns that HTTP sends",
		"docs/behavior.md":       "页面给出警告",
		"docs/manual-test.md":    "语言模型页说明 HTTP",
		"CHANGELOG.md":           "Voxt warns that HTTP sends",
	} {
		doc, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(doc), claim) {
			t.Errorf("%s claims a plain-HTTP warning that the app does not show", file)
		}
	}
}

func TestLegacyPromptInputVariables(t *testing.T) {
	for _, c := range []struct{ in, want string }{
		{"Task: {{DICTATED_PROMPT}}\nText: {{SOURCE_TEXT}}", ""},
		{"Keep names.\nText：{{SOURCE_TEXT}}", "Keep names."},
		{"Fix {{RAW_TRANSCRIPTION}} for English", "Fix for English"},
		{"Polish {{SOURCE_TEXT}}.  Be brief.", "Polish . Be brief."},
		{"Two  spaces stay here.\n\n{{SOURCE_TEXT}}\nLast line", "Two  spaces stay here.\n\nLast line"},
		{"Translate into {{TARGET_LANGUAGE}}", "Translate into {{TARGET_LANGUAGE}}"},
		{"Translate plainly. Source: {{SOURCE_TEXT}}\nKeep the tone.", "Translate plainly. Source:\nKeep the tone."},
		{"Apply this request: {{DICTATED_PROMPT}}", "Apply this request:"},
		{"把下面的话翻译成英文：{{SOURCE_TEXT}}", "把下面的话翻译成英文："},
	} {
		if got := legacyOwnPrompt("translation", c.in); got != c.want {
			t.Errorf("%q: %q, want %q", c.in, got, c.want)
		}
	}
}

func writeLegacyPrefs(t *testing.T, lib string) {
	t.Helper()
	prefs := filepath.Join(lib, "Preferences", legacyBundleID+".plist")
	if err := os.MkdirAll(filepath.Dir(prefs), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(prefs, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
}

// Preferences that exist and cannot be read are reported; no preferences
// at all, or readable ones, are not.
func TestLegacyPreferencesUnreadable(t *testing.T) {
	const line = "could not be read"
	failing := func(t *testing.T) {
		t.Helper()
		fakeLegacyPlutil(t, nil)
		script := filepath.Join(t.TempDir(), "plutil")
		if err := os.WriteFile(script, []byte("#!/bin/sh\nexit 1\n"), 0o755); err != nil {
			t.Fatal(err)
		}
		legacyPlutil = script
	}
	skipped := func(lib string) []string {
		a, _, _ := newTestApp(t)
		a.importLegacyFrom(lib)
		return a.ob.legacy.skipped
	}
	has := func(lines []string) bool {
		return slices.ContainsFunc(lines, func(s string) bool { return strings.Contains(s, line) })
	}

	failing(t)
	lib := copyFixture(t, "real")
	writeLegacyPrefs(t, lib)
	if got := skipped(lib); !has(got) {
		t.Errorf("unreadable preferences: %q", got)
	}
	if got := skipped(copyFixture(t, "real")); has(got) {
		t.Errorf("no preferences at all: %q", got)
	}
	fakeLegacyPlutil(t, nil)
	if got := skipped(lib); has(got) {
		t.Errorf("readable preferences: %q", got)
	}
}
