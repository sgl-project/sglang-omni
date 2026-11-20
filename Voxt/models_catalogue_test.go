package main

import (
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// TestModelsPageMatchesCatalogue: with every model installed, the Local
// models page lists each speech model of the catalogue once, under its
// family, and shows a draft exactly for the speech models whose catalogue
// entry names one. A draft that no speech model names could never be
// switched on, and a named draft missing from the catalogue would show an
// empty row.
func TestModelsPageMatchesCatalogue(t *testing.T) {
	catalog := models.Catalog()
	status := map[string]models.Status{}
	for _, model := range catalog {
		status[model.ID] = models.Status{Installed: true, OnDisk: true}
	}
	state := buildModelsViewState(i18n.NewTranslator("en"), modelsViewStateInput{Catalog: catalog, Status: status, Settings: settings.Defaults()})

	rows := map[string]modelRowViewState{}
	for _, card := range state.ASR {
		for _, row := range card.Rows {
			if _, twice := rows[row.ID]; twice {
				t.Errorf("%s is listed twice", row.ID)
			}
			rows[row.ID] = row
			model, ok := models.Lookup(row.ID)
			if !ok || model.Kind != models.ASR {
				t.Errorf("the page lists %s, which is not a speech model of the catalogue", row.ID)
			}
			if family, grouped := strings.CutPrefix(card.Key, "family-"); grouped && family != model.Family {
				t.Errorf("%s (family %q) is in the card of family %q", row.ID, model.Family, family)
			}
		}
	}

	parents := map[string]string{} // draft -> the speech model that names it
	for _, model := range catalog {
		if model.Kind == models.Draft {
			if _, listed := parents[model.ID]; !listed {
				parents[model.ID] = ""
			}
			continue
		}
		if model.Kind != models.ASR {
			if model.Draft != "" {
				t.Errorf("%s (%s) names a draft; only speech models have one", model.ID, model.Kind)
			}
			continue
		}
		row, listed := rows[model.ID]
		if !listed {
			t.Errorf("the page does not list %s", model.ID)
			continue
		}
		if model.Draft == "" {
			if row.Draft != nil {
				t.Errorf("%s shows the draft %s, and the catalogue names none", model.ID, row.Draft.ModelID)
			}
			continue
		}
		draft, ok := models.Lookup(model.Draft)
		if !ok || draft.Kind != models.Draft {
			t.Errorf("%s names the draft %s, which is not a draft of the catalogue", model.ID, model.Draft)
		}
		if other := parents[model.Draft]; other != "" && !model.DraftOptIn {
			t.Errorf("%s and %s name the same draft %s", other, model.ID, model.Draft)
		}
		parents[model.Draft] = model.ID
		if row.Draft == nil || row.Draft.ModelID != model.Draft || row.Draft.ParentID != model.ID {
			t.Errorf("%s: the page's draft is %+v, want %s", model.ID, row.Draft, model.Draft)
		}
	}
	for draft, parent := range parents {
		if parent == "" {
			t.Errorf("no speech model names the draft %s", draft)
		}
	}
}
