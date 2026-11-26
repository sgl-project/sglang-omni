package uitext

import (
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
)

// TestSpeechModelWords: every speech model of the catalogue says which
// languages it knows.
func TestSpeechModelWords(t *testing.T) {
	for _, m := range models.Catalog() {
		if m.Kind != models.ASR {
			continue
		}
		if ModelLanguages(m.ID) == "" {
			t.Errorf("%s: no languages", m.ID)
		}
	}
}
