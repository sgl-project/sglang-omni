package main

import (
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"strings"
	"testing"

	"github.com/hehehai/actnowkit/hfhub"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

func TestModelsFixtureCoverage(t *testing.T) {
	scenarios := modelsFixtureScenarios()
	kinds, runtimes, keepValues, errorCodes := map[string]bool{}, map[string]bool{}, map[string]bool{}, map[string]bool{}
	proxyModes, sourceValues := map[string]bool{}, map[string]bool{}
	hasPathError := false
	for _, scenario := range scenarios {
		for _, model := range scenario.models.Catalog {
			kinds[string(model.Kind)], runtimes[string(model.Runtime)] = true, true
		}
		if strings.HasPrefix(scenario.name, "models-keep-") && scenario.name != "models-keep-unknown" {
			keepValues[scenario.models.Settings.Models.KeepLoaded] = true
		}
		for _, status := range scenario.models.Status {
			if status.Err != nil {
				errorCodes[hfhub.Code(status.Err)] = true
				var pathError *fs.PathError
				if errors.As(status.Err, &pathError) {
					hasPathError = true
				}
			}
		}
		if scenario.models.NetworkOpen {
			proxyModes[modelsProxyConstantName(scenario.models.NetworkForm.proxyMode)] = true
			sourceValues[modelsNetworkSourceName(scenario.models.NetworkForm.source)] = true
		}
	}
	assertMeetingFixtureSet(t, "models kinds", meetingTypedConstants(t, "internal/models/catalog.go", "Kind"), kinds)
	assertMeetingFixtureSet(t, "models runtimes", meetingTypedConstants(t, "internal/models/catalog.go", "Runtime"), runtimes)
	assertMeetingFixtureSet(t, "models keep choices", meetingNamedConstants(t, "internal/settings/models.go", "Keep"), keepValues)
	assertMeetingFixtureSet(t, "models proxy modes", modelsConstantNames(t, "models_network.go", "proxy"), proxyModes)
	assertMeetingFixtureSet(t, "models network sources", map[string]bool{"automatic": true, "official": true, "mirror": true, "custom": true}, sourceValues)
	assertMeetingFixtureSet(t, "models download problem codes", modelsDownloadProblemCodes(t), errorCodes)
	if !hasPathError {
		t.Error("models: no fs.PathError download fixture")
	}

	for _, locale := range i18n.Supported() {
		fixtures := readMeetingFixtures(t, locale)
		for _, name := range []string{
			"models-page-empty", "models-page-loading", "models-page-normal", "models-page-error",
			"models-error-path", "models-error-generic", "models-network-custom-validation", "models-network-password-validation",
			"models-delete-plain", "models-delete-shared", "models-delete-all", "models-hidden-retired-legacy",
		} {
			if _, ok := fixtures[name]; !ok {
				t.Errorf("%s: missing required models fixture %s", locale, name)
			}
		}
		actions, familyKinds := map[string]bool{}, map[string]bool{}
		for name, fixture := range fixtures {
			state := fixture.Models
			if state == nil {
				continue
			}
			if name == "models-hidden-retired-legacy" && len(state.LLM) != 0 {
				t.Errorf("%s: legacy hidden retired model is visible", locale)
			}
			cards := append(append(append([]modelCardViewState{}, state.ASR...), state.LLM...), state.Hidden...)
			if state.Dictation != nil {
				cards = append(cards, *state.Dictation)
			}
			for _, card := range cards {
				for _, row := range card.Rows {
					for _, action := range row.Actions {
						actions[string(action.Kind)] = true
					}
					if row.Draft != nil {
						for _, action := range row.Draft.Actions {
							actions[string(action.Kind)] = true
						}
					}
				}
			}
			if state.Network.Open {
				familyKinds[state.Network.ProxyFamily.Kind], familyKinds[state.Network.SourceFamily.Kind] = true, true
			}
		}
		assertMeetingFixtureSet(t, locale+" models actions", map[string]bool{
			string(modelActionHide): true, string(modelActionShow): true, string(modelActionCancel): true,
			string(modelActionDelete): true, string(modelActionDiscard): true, string(modelActionUse): true, string(modelActionDownload): true,
		}, actions)
		assertMeetingFixtureSet(t, locale+" models network family states", map[string]bool{"": true, "conflict": true, "readonly": true, "follows": true, "own": true}, familyKinds)
	}
}

func modelsDownloadProblemCodes(t *testing.T) map[string]bool {
	t.Helper()
	required := map[string]bool{
		hfhub.CodeDiskSpace: true, hfhub.CodeChecksum: true, hfhub.CodeNotFound: true,
		hfhub.CodeGated: true, hfhub.CodeOffline: true, hfhub.CodeBusy: true, hfhub.CodeDownloadFailed: true,
	}
	syntax, err := parser.ParseFile(token.NewFileSet(), "internal/uitext/catalog.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	declared := map[string]bool{}
	ast.Inspect(syntax, func(node ast.Node) bool {
		selector, ok := node.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		identifier, ok := selector.X.(*ast.Ident)
		if ok && identifier.Name == "hfhub" && strings.HasPrefix(selector.Sel.Name, "Code") {
			switch selector.Sel.Name {
			case "CodeDiskSpace":
				declared[hfhub.CodeDiskSpace] = true
			case "CodeChecksum":
				declared[hfhub.CodeChecksum] = true
			case "CodeNotFound":
				declared[hfhub.CodeNotFound] = true
			case "CodeGated":
				declared[hfhub.CodeGated] = true
			case "CodeOffline":
				declared[hfhub.CodeOffline] = true
			case "CodeBusy":
				declared[hfhub.CodeBusy] = true
			}
		}
		return true
	})
	for code := range declared {
		if !required[code] {
			t.Errorf("models download problem codes: unexpected declaration %s", code)
		}
	}
	for code := range required {
		if code != hfhub.CodeDownloadFailed && !declared[code] {
			t.Errorf("models download problem codes: %s is required but not declared in DownloadErrorTextFor", code)
		}
	}
	return required
}

func modelsConstantNames(t *testing.T, path, prefix string) map[string]bool {
	t.Helper()
	syntax, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	values := map[string]bool{}
	for _, declaration := range syntax.Decls {
		general, ok := declaration.(*ast.GenDecl)
		if !ok || general.Tok != token.CONST {
			continue
		}
		for _, rawSpec := range general.Specs {
			spec := rawSpec.(*ast.ValueSpec)
			for _, name := range spec.Names {
				if strings.HasPrefix(name.Name, prefix) {
					values[name.Name] = true
				}
			}
		}
	}
	if len(values) == 0 {
		t.Fatalf("%s: no constants with prefix %s", path, prefix)
	}
	return values
}

func modelsProxyConstantName(mode int) string {
	switch mode {
	case proxyAuto:
		return "proxyAuto"
	case proxyNone:
		return "proxyNone"
	case proxyCustom:
		return "proxyCustom"
	default:
		return "unknown"
	}
}

func modelsNetworkSourceName(source int) string {
	switch source {
	case 0:
		return "automatic"
	case 1:
		return "official"
	case 2:
		return "mirror"
	default:
		return "custom"
	}
}
