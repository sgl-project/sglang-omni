package historyapi

import (
	"encoding/json"
	"slices"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func secretEntry() store.Entry {
	return store.Entry{ID: "e1", Kind: store.KindTranslate, CreatedAt: time.Date(2026, 11, 8, 9, 14, 0, 0, time.UTC), Status: store.StatusDone,
		AppBundle: "bundle-secret", AppName: "Obsidian", URLHost: "secret.example.com", RuleID: "rule-secret", RuleName: "rule-name-secret",
		RawText: "heard words", SourceText: "selected-secret", PromptText: "the instruction", Result: "final text", TargetLang: "zh",
		Error: "error-secret", ASRProvider: "asrprov-secret", ASRModel: "asrmodel-secret", LLMProvider: "llmprov-secret", LLMModel: "llmmodel-secret",
		Audio: "secret.wav", Delivery: "insert-secret", AudioMS: 123456}
}

func keysOf(t *testing.T, r Row) []string {
	t.Helper()
	data, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatal(err)
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// TestRowFields: exactly the listed fields, and nothing sensitive; the
// selected text only when asked for.
func TestRowFields(t *testing.T) {
	want := []string{"app", "created_at", "heard", "id", "instruction", "kind", "target_lang", "text"}
	r := FromEntry(secretEntry(), false)
	if got := keysOf(t, r); !slices.Equal(got, want) {
		t.Errorf("keys %v, want %v", got, want)
	}
	data, _ := json.Marshal(r)
	if s := string(data); strings.Contains(s, "secret") || strings.Contains(s, "123456") {
		t.Errorf("leaked: %s", s)
	}
	if r.CreatedAt.Location() != time.Local {
		t.Errorf("created_at in %v", r.CreatedAt.Location())
	}
	r = FromEntry(secretEntry(), true)
	if got := keysOf(t, r); !slices.Contains(got, "selection") || r.Selection != "selected-secret" {
		t.Errorf("with selection: keys %v, %q", got, r.Selection)
	}
}

// TestRowOmitsEmpty: app, instruction, target_lang and selection leave
// the line when empty; the text fields a consumer relies on stay.
func TestRowOmitsEmpty(t *testing.T) {
	got := keysOf(t, FromEntry(store.Entry{ID: "x", Kind: store.KindTranscribe, Result: "r", CreatedAt: testStart}, true))
	if want := []string{"created_at", "heard", "id", "kind", "text"}; !slices.Equal(got, want) {
		t.Errorf("keys %v, want %v", got, want)
	}
}
