package uitext

import (
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
)

func TestResponsesOnlyText(t *testing.T) {
	msg, more := ModelErrorText(&llm.Error{Code: llm.CodeResponsesOnly, Status: 404})
	if !strings.Contains(msg, "API format") || !strings.Contains(more, llm.CodeResponsesOnly) {
		t.Errorf("%q, %q", msg, more)
	}
}

func TestPlanErrorTexts(t *testing.T) {
	for code, want := range map[string]string{
		llm.CodeSignIn:      "sign in again",
		llm.CodeUsageLimit:  "used up",
		llm.CodeNotEligible: "cannot use Codex",
	} {
		if msg, _ := ModelErrorText(&llm.Error{Code: code}); !strings.Contains(msg, want) {
			t.Errorf("%s: %q", code, msg)
		}
	}
}
