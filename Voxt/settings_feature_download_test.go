package main

import (
	"slices"
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
)

// Settings → Transcribe's Download for the recommended speech model
// fetches its draft too (#4069), as the feature pages did before they
// moved into Settings (#4478).
func TestSettingsFeatureRecommendedDownload(t *testing.T) {
	countingHub(t)
	a, _, _ := newTestApp(t)
	drain := queuePosts(a)
	reopenModels(a)
	a.local = onlyDictation{&fakeLocal{}}
	tt := ui.NewTester(a.view, windowWidth, 1200)
	a.navigate("/settings/transcribe")
	frames(tt)
	if !slices.Contains(tt.AccessLabels(), "Download Qwen3-ASR 1.7B (8bit)") {
		t.Fatalf("Settings → Transcribe does not recommend Qwen3-ASR 1.7B (8bit): %q", tt.AccessLabels())
	}
	click(t, tt, "Download Qwen3-ASR 1.7B (8bit)")
	downloadAsked(t, a, drain, models.ASRAccurate)
	downloadAsked(t, a, drain, models.ASRAccurateDraft)
}
