package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/appctx"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

func TestAppEnhancementFixtureCoverage(t *testing.T) {
	for _, locale := range i18n.Supported() {
		checkAppEnhancementFixtureValues(t, locale, readMeetingFixtures(t, locale))
	}
}

func checkAppEnhancementFixtureValues(t *testing.T, locale string, fixtures map[string]meetingFixtureState) {
	t.Helper()
	for _, name := range []string{
		"appenhance-empty", "appenhance-normal", "appenhance-off", "appenhance-error",
		"appenhance-preview-loading", "appenhance-apps-loading", "appenhance-apps-normal",
		"appenhance-apps-error", "appenhance-apps-empty", "appenhance-browsers-normal", "appenhance-delete-ask",
	} {
		if _, ok := fixtures[name]; !ok {
			t.Errorf("%s: missing required app enhancement fixture %s", locale, name)
		}
	}
	sources := map[string]bool{}
	patternProblems := map[string]bool{}
	autoKeyProblems := map[string]bool{}
	engines := map[string]bool{}
	addressStatuses := map[string]bool{}
	statusTones := map[string]bool{}
	featureValues := map[string]map[bool]bool{}
	autoKeyOptions := map[string]bool{}
	for _, fixture := range fixtures {
		state := fixture.AppEnhancement
		if state == nil {
			continue
		}
		sources[state.Preview.MatchSource] = true
		addressStatuses[state.Preview.AddressStatus] = true
		if state.Detail != nil {
			for _, feature := range state.Detail.Features {
				if featureValues[feature.ID] == nil {
					featureValues[feature.ID] = map[bool]bool{}
				}
				featureValues[feature.ID][feature.Checked] = true
			}
			for _, row := range state.Detail.URLs {
				if row.WarningCode != "" {
					patternProblems[row.WarningCode] = true
				}
			}
			if state.Detail.AutoKey.ErrorCode != "" {
				autoKeyProblems[state.Detail.AutoKey.ErrorCode] = true
			}
			for _, option := range state.Detail.AutoKey.Options {
				autoKeyOptions[option.Value] = true
			}
		}
		if state.Browsers != nil {
			for _, row := range state.Browsers.Rows {
				engines[row.Engine] = true
				statusTones[row.StatusTone] = true
			}
		}
	}
	assertMeetingFixtureSet(t, locale+" app enhancement match sources", meetingTypedConstants(t, "internal/appctx/match.go", "Source"), sources)
	assertMeetingFixtureSet(t, locale+" app enhancement pattern problems", appEnhancementPatternProblemCodes(t), patternProblems)
	assertMeetingFixtureSet(t, locale+" app enhancement auto-key problems", map[string]bool{"autoKey": true}, autoKeyProblems)
	assertMeetingFixtureSet(t, locale+" app enhancement browser engines", map[string]bool{appctx.WebKit: true, appctx.Chromium: true, appctx.Gecko: true, "": true}, engines)
	assertMeetingFixtureSet(t, locale+" app enhancement address states", map[string]bool{
		"none": true, "loading": true, "address": true, "raw-address": true, "helper-stopped": true, "no-address": true,
	}, addressStatuses)
	assertMeetingFixtureSet(t, locale+" app enhancement browser status tones", map[string]bool{"success": true, "warning": true, "info": true}, statusTones)
	for _, feature := range []string{appctx.Transcribe, appctx.Translate, appctx.Rewrite} {
		if !featureValues[feature][true] || !featureValues[feature][false] {
			t.Errorf("%s: feature %s does not have checked and unchecked fixtures", locale, feature)
		}
	}
	requiredKeys := map[string]bool{"": true, "Control+Shift+Return": true}
	for _, keys := range autoKeys {
		requiredKeys[keys] = true
	}
	assertMeetingFixtureSet(t, locale+" app enhancement auto-key options", requiredKeys, autoKeyOptions)
}

func appEnhancementPatternProblemCodes(t *testing.T) map[string]bool {
	t.Helper()
	syntax, err := parser.ParseFile(token.NewFileSet(), "internal/appctx/url.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	codes := map[string]bool{}
	for _, declaration := range syntax.Decls {
		general, ok := declaration.(*ast.GenDecl)
		if !ok || general.Tok != token.VAR {
			continue
		}
		for _, rawSpec := range general.Specs {
			spec := rawSpec.(*ast.ValueSpec)
			for _, name := range spec.Names {
				if strings.HasPrefix(name.Name, "ErrPattern") {
					codes[strings.ToLower(strings.TrimPrefix(name.Name, "ErrPattern"))] = true
				}
			}
		}
	}
	if len(codes) == 0 {
		t.Fatal("internal/appctx/url.go: no pattern errors found")
	}
	return codes
}
