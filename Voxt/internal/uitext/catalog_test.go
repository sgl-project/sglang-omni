package uitext

import (
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
)

// TestSpeechModelWords: every speech model of the catalogue says what it
// is good at, which languages it knows, and has labels beyond Local.
func TestSpeechModelWords(t *testing.T) {
	for _, m := range models.Catalog() {
		if m.Kind != models.ASR {
			continue
		}
		if ModelAbout(m.ID) == "" || ModelLanguages(m.ID) == "" || len(ModelTags(m.ID)) < 2 {
			t.Errorf("%s: about %q, languages %q, tags %v", m.ID, ModelAbout(m.ID), ModelLanguages(m.ID), ModelTags(m.ID))
		}
	}
}
