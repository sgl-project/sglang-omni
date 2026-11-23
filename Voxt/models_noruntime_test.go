package main

import (
	"errors"
	"slices"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// TestModelsViewStateNoRuntime: a copy without the omni runtime (the
// Windows port today) offers no download, Use it or Apply to on the
// models page — every one of them ends in ErrNoRuntime at dictation —
// and says to use an online speech service instead. Partials stay
// deletable.
func TestModelsViewStateNoRuntime(t *testing.T) {
	catalog := []models.Model{}
	for _, id := range []string{models.ASRAccurate, models.ASRFast} {
		m, ok := models.Lookup(id)
		if !ok {
			t.Fatalf("catalog has no %s", id)
		}
		catalog = append(catalog, m)
	}
	input := modelsViewStateInput{
		Catalog:   catalog,
		Status:    map[string]models.Status{models.ASRAccurate: {Installed: true, OnDisk: true, Mine: true}, models.ASRFast: {Done: 1024}},
		Settings:  settings.Defaults(),
		NoRuntime: true,
	}
	state := buildModelsViewState(i18n.NewTranslator("en"), input)
	if state.NoRuntimeText == "" {
		t.Fatal("no no-runtime notice")
	}
	var accurate, fast *modelRowViewState
	for _, card := range state.ASR {
		for i := range card.Rows {
			switch card.Rows[i].ID {
			case models.ASRAccurate:
				accurate = &card.Rows[i]
			case models.ASRFast:
				fast = &card.Rows[i]
			}
		}
	}
	if accurate == nil || fast == nil {
		t.Fatalf("rows %v %v", accurate, fast)
	}
	for _, action := range accurate.Actions {
		if action.Kind == modelActionAssign {
			t.Errorf("Apply to without a runtime: %+v", action)
		}
	}
	for _, action := range fast.Actions {
		if action.Kind == modelActionDownload {
			t.Errorf("Download without a runtime: %+v", action)
		}
	}
	if !slices.ContainsFunc(fast.Actions, func(a modelActionViewState) bool { return a.Kind == modelActionDiscard }) {
		t.Errorf("a partial download cannot be deleted: %+v", fast.Actions)
	}
}

// TestProbeOmniRuntime: the startup probe's answer reaches the models
// page input.
func TestProbeOmniRuntime(t *testing.T) {
	a, _, _ := newTestApp(t)
	old := omniLocate
	omniLocate = func() (string, error) { return "", errors.New("no runtime here") }
	t.Cleanup(func() { omniLocate = old })
	a.probeOmniRuntime()
	if !a.modelsViewStateInput().NoRuntime {
		t.Error("the models page does not know the runtime is missing")
	}
}
