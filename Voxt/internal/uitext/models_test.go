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
