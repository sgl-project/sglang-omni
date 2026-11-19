package main

import (
	"reflect"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// TestModelAssignmentPreservesOtherSettings: applying a model writes the
// checked features' model field and nothing else — prompts, switches,
// the Translate fallback and unchecked features stay as they were.
func TestModelAssignmentPreservesOtherSettings(t *testing.T) {
	for _, kind := range []local.Kind{local.ASR, local.LLM} {
		t.Run(string(kind), func(t *testing.T) {
			a, _, _ := newTestApp(t)
			readyModels(a)
			model := a.installed(kind)[0]
			a.updateSettings(func(s *settings.Settings) {
				s.Features.Transcribe.Enhance = false
				s.Features.Translate.Fallback = "keep-fallback"
				s.Features.Meeting.NoAutoSummary = true
				s.Features.Rewrite.Prompt = "keep-prompt"
			})
			before := a.settings.Get()
			a.openModelAssignment(string(kind), model.ID)
			a.modelsLocalUI.assign.selected = [4]bool{true, false, false, true}
			a.applyModelAssignment()
			want := before
			if kind == local.ASR {
				want.Features.Transcribe.ASR = model.ID
				want.Features.Meeting.ASR = model.ID
			} else {
				want.Features.Transcribe.LLM = model.ID
				want.Features.Meeting.LLM = model.ID
			}
			if got := a.settings.Get(); !reflect.DeepEqual(got, want) {
				t.Errorf("settings changed beyond the checked model fields:\ngot %+v\nwant %+v", got.Features, want.Features)
			}
			if a.modelsLocalUI.assign.open {
				t.Error("dialog stayed open after applying")
			}
		})
	}
}

func TestModelAssignmentMeetingUsage(t *testing.T) {
	for _, kind := range []local.Kind{local.ASR, local.LLM} {
		t.Run(string(kind), func(t *testing.T) {
			installed := []local.Model{
				{Kind: kind, ID: "default", Name: "Default"},
				{Kind: kind, ID: "meeting-only", Name: "Meeting only"},
			}
			prefs := settings.Defaults()
			if kind == local.ASR {
				prefs.Features.Meeting.ASR = installed[1].ID
			} else {
				prefs.Features.Meeting.LLM = installed[1].ID
			}
			translator := i18n.CurrentTranslator()
			if got := modelsUsedBy(translator, prefs, installed, installed, kind, installed[1].ID); got != "Used by Meetings" {
				t.Fatalf("meeting-only model usage = %q", got)
			}
			if actions := buildHideAction(translator, modelsViewStateInput{Settings: prefs}, installed[1].ID,
				modelsUsedBy(translator, prefs, installed, installed, kind, installed[1].ID)); len(actions) != 0 {
				t.Fatalf("meeting-only model offers Hide: %+v", actions)
			}
		})
	}
}

func TestModelAssignmentMeetingUnhidesSpeechModel(t *testing.T) {
	a, _, _ := newTestApp(t)
	seedModels(t, a, models.ASRFast, models.ASRAccurate)
	installed := a.installed(local.ASR)
	if len(installed) < 2 {
		t.Fatalf("need two speech models, got %v", installed)
	}
	model := installed[1]
	if got := a.usedBy(local.ASR, model.ID); got != "" {
		t.Fatalf("model already in use before assignment: %q", got)
	}
	a.updateSettings(func(s *settings.Settings) { s.Models.Hidden = []string{model.ID} })
	if !a.modelHidden(model.ID) {
		t.Fatal("unused model was not hidden before assignment")
	}
	a.openModelAssignment(string(local.ASR), model.ID)
	a.modelsLocalUI.assign.selected = [4]bool{false, false, false, true}
	a.applyModelAssignment()
	if a.prefs.Features.Meeting.ASR != model.ID {
		t.Fatal("meeting assignment was not saved")
	}
	if a.modelHidden(model.ID) {
		t.Fatal("a model assigned to Meetings is still hidden")
	}
	for _, option := range a.meetingSettingsModel().ASROptions {
		if option.Code == model.ID {
			return
		}
	}
	t.Fatal("the model assigned to Meetings is absent from the meeting picker")
}

// TestModelAssignmentSelectionAndRevalidation: unsupported features are
// neither preselected nor written, a model that went away is not
// applied, a missing id never opens the dialog, and explicit bindings
// win the preselection.
func TestModelAssignmentSelectionAndRevalidation(t *testing.T) {
	a, _, _ := newTestApp(t)
	readyModels(a)
	translateOnly := local.Model{Kind: local.LLM, ID: "translate-only", Name: "Translation only", Features: []string{hotkey.Translate}}
	fake := a.local.(*fakeLocal)
	fake.MoreLLM = []local.Model{translateOnly}
	a.openModelAssignment(string(local.LLM), translateOnly.ID)
	if a.modelsLocalUI.assign.selected != [4]bool{false, true, false, false} {
		t.Fatalf("unsupported features preselected: %v", a.modelsLocalUI.assign.selected)
	}
	before := a.settings.Get()
	a.modelsLocalUI.assign.selected = [4]bool{true, true, true, true}
	a.applyModelAssignment()
	want := before
	want.Features.Translate.LLM = translateOnly.ID
	if !reflect.DeepEqual(a.settings.Get(), want) {
		t.Error("unsupported features were written")
	}

	a.openModelAssignment(string(local.LLM), translateOnly.ID)
	fake.MoreLLM = nil
	before = a.settings.Get()
	a.applyModelAssignment()
	if !reflect.DeepEqual(a.settings.Get(), before) || !a.modelsLocalUI.assign.open {
		t.Error("a removed model was applied or the dialog closed")
	}

	a.modelsLocalUI.assign = modelAssignmentLocalUI{}
	a.openModelAssignment(string(local.LLM), "gone")
	if a.modelsLocalUI.assign.open {
		t.Error("a missing model opened the dialog through the automatic fallback")
	}

	asr := a.installed(local.ASR)[0]
	a.updateSettings(func(s *settings.Settings) { s.Features.Rewrite.ASR = asr.ID })
	a.openModelAssignment(string(local.ASR), asr.ID)
	if a.modelsLocalUI.assign.selected != [4]bool{false, false, true, false} {
		t.Errorf("explicit bindings not preselected: %v", a.modelsLocalUI.assign.selected)
	}
	before = a.settings.Get()
	a.modelsLocalUI.assign.selected = [4]bool{}
	a.applyModelAssignment()
	if !reflect.DeepEqual(a.settings.Get(), before) {
		t.Error("an empty selection changed settings")
	}
}

// TestModelAssignmentDialogShowsSavedModel: the dialog names the model
// each feature uses now, and a binding to a model that is gone shows its
// id as unavailable rather than the automatic choice.
func TestModelAssignmentDialogShowsSavedModel(t *testing.T) {
	a, _, tt := newTestApp(t)
	readyModels(a)
	tt.SetSize(windowWidth, 1000)
	a.navigate(modelsPath)
	model := a.installed(local.LLM)[0]
	a.openModelAssignment(string(local.LLM), model.ID)
	frames(tt)
	if !has(tt, "Apply to features") || !has(tt, "Current: Automatic (now: "+model.Name+")") {
		t.Fatalf("missing dialog or its saved model: %v", tt.Texts())
	}
	a.updateSettings(func(s *settings.Settings) { s.Features.Translate.LLM = "temporarily-unavailable" })
	frames(tt)
	if !has(tt, "Current: Unavailable: temporarily-unavailable") {
		t.Fatalf("a binding to a gone model shows as automatic: %v", tt.Texts())
	}
}

// TestModelAssignmentRemoteCard: Apply to sits beside Test on a set-up
// remote card and applies that service to every feature the dialog
// checked, without opening the provider dialog.
func TestModelAssignmentRemoteCard(t *testing.T) {
	useKeys(t, &memKeys{m: map[string]string{}})
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 1200)
	a.navigate(remotePath)
	frames(tt)
	scrollSettingsToText(tt, "DeepSeek")
	clickControl(t, tt, "DeepSeek")
	frames(tt)
	a.provDlg.key = "sk-test-assignment"
	click(t, tt, "Save")
	a.local = a.remote
	frames(tt)
	if !has(tt, "Apply to") {
		t.Fatalf("no Apply to on a set-up card: %q", tt.Texts())
	}
	anchor, _ := tt.Find("Find in settings")
	tt.Scroll(anchor.X+anchor.W/2, anchor.Y+anchor.H+100, 0, -10000)
	frames(tt)
	testBounds, testVisible := tt.Find("Test DeepSeek")
	applyBounds, applyVisible := tt.Find("Apply DeepSeek to features")
	if !testVisible || !applyVisible || testBounds.W <= 0 || applyBounds.W <= 0 || testBounds.H <= 0 || applyBounds.H <= 0 {
		t.Fatalf("card actions are not visible: test=%+v apply=%+v", testBounds, applyBounds)
	}
	if testBounds.Y != applyBounds.Y || testBounds.H != applyBounds.H {
		t.Errorf("card actions are not aligned: test=%+v apply=%+v", testBounds, applyBounds)
	}
	clickControl(t, tt, "Apply DeepSeek to features")
	if !a.modelsLocalUI.assign.open || a.provDlg.open {
		t.Fatalf("Apply to also opened the provider dialog: assignment=%v provider=%v", a.modelsLocalUI.assign.open, a.provDlg.open)
	}
	if !has(tt, "Apply to features") {
		t.Fatalf("no dialog: %v", tt.Texts())
	}
	before := a.settings.Get()
	click(t, tt, "Cancel")
	if a.modelsLocalUI.assign.open || !reflect.DeepEqual(a.settings.Get(), before) {
		t.Error("cancel changed settings or stayed open")
	}
	anchor, _ = tt.Find("Find in settings")
	tt.Scroll(anchor.X+anchor.W/2, anchor.Y+anchor.H+100, 0, -10000)
	frames(tt)
	clickControl(t, tt, "Apply DeepSeek to features")
	click(t, tt, "Apply model to Translate")
	click(t, tt, "Apply")
	want := before
	want.Features.Transcribe.LLM = "deepseek"
	want.Features.Rewrite.LLM = "deepseek"
	want.Features.Meeting.LLM = "deepseek"
	if !reflect.DeepEqual(a.settings.Get(), want) {
		t.Errorf("checked features not written: %+v", a.settings.Get().Features)
	}
	if a.provDlg.open {
		t.Error("the assignment opened the provider dialog")
	}
}
