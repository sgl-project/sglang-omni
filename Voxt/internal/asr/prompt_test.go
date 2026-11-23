package asr

import (
	"strings"
	"testing"
)

func TestResolvePromptTemplate(t *testing.T) {
	tpl := "Main: {{USER_MAIN_LANGUAGE}}. Others: {{USER_OTHER_LANGUAGES}}. Terms:\n{{DICTIONARY_TERMS}}"
	got := ResolvePromptTemplate(tpl, "Chinese", "English, Japanese", "Voxt\nSGLang")
	want := "Main: Chinese. Others: English, Japanese. Terms:\nVoxt\nSGLang"
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
	if ResolvePromptTemplate("", "x", "y", "z") != "" {
		t.Fatal("empty template")
	}
}

func TestBuildSpeechPrompt(t *testing.T) {
	global := "Leo works on LLM serving."
	tpl := "Main {{USER_MAIN_LANGUAGE}}, terms: {{DICTIONARY_TERMS}}"
	got := BuildSpeechPrompt(global, tpl, "zh-Hans", nil, []string{"Voxt"})
	if !strings.Contains(got, global) || !strings.Contains(got, "Voxt") || !strings.Contains(got, "Chinese") {
		t.Fatalf("got %q", got)
	}
	got = BuildSpeechPrompt(global, "", "en", []string{"fr"}, []string{"A"})
	if got != global {
		t.Fatalf("no template: got %q", got)
	}
}

func TestOtherLanguagesPrompt(t *testing.T) {
	if OtherLanguagesPrompt(nil) != "None specified" {
		t.Fatal("empty")
	}
	if got := OtherLanguagesPrompt([]string{"en", "ja"}); got != "English, Japanese" {
		t.Fatalf("got %q", got)
	}
}

func TestJoinPromptParts(t *testing.T) {
	if got := JoinPromptParts("a", "", "  b  "); got != "a\n\nb" {
		t.Fatalf("got %q", got)
	}
}
