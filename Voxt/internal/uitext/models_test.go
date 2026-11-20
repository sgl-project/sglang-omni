package uitext

import (
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
)

func TestResponsesOnlyText(t *testing.T) {
	msg, more := ModelErrorText(&llm.Error{Code: llm.CodeResponsesOnly, Status: 404})
	if !strings.Contains(msg, "API format") || !strings.Contains(more, llm.CodeResponsesOnly) {
		t.Errorf("%q, %q", msg, more)
	}
}

func TestModelErrorTextForTranslator(t *testing.T) {
	translator := i18n.NewTranslator("zh-Hans")
	msg, more := ModelErrorTextFor(translator, local.ErrNotInstalled)
	if msg != translator.T("The model is not installed.") || more != local.ErrNotInstalled.Error() {
		t.Fatalf("translator error text: %q, %q", msg, more)
	}
}

func TestPlanErrorTexts(t *testing.T) {
	for code, want := range map[string]string{
		llm.CodeSignIn:      "Sign in again",
		llm.CodeUsageLimit:  "usage limit",
		llm.CodeNotEligible: "cannot use Codex",
	} {
		if msg, _ := ModelErrorText(&llm.Error{Code: code}); !strings.Contains(msg, want) {
			t.Errorf("%s: %q", code, msg)
		}
	}
}
