package main

import (
	"reflect"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func TestModelAssignmentPreservesOtherSettings(t *testing.T) {
	for _, kind := range []local.Kind{local.ASR, local.LLM} {
		t.Run(string(kind), func(t *testing.T) {
			a, _, _ := newTestApp(t)
			readyModels(a)
			m := a.installed(kind)[0]
			a.updateSettings(func(s *settings.Settings) {
				s.Features.Transcribe.Enhance = false
				s.Features.Translate.Fallback = "keep-fallback"
				s.Features.Meeting.NoAutoSummary = true
				s.Features.Rewrite.Prompt = "keep-prompt"
			})
			before := a.settings.Get()
			a.openModelAssignment(kind, m.ID)
			a.assignment.selected = [4]bool{true, false, false, true}
			a.applyModelAssignment()
			want := before
			if kind == local.ASR {
				want.Features.Transcribe.ASR = m.ID
				want.Features.Meeting.ASR = m.ID
			} else {
				want.Features.Transcribe.LLM = m.ID
				want.Features.Meeting.LLM = m.ID
			}
			if got := a.settings.Get(); !reflect.DeepEqual(got, want) {
				t.Errorf("settings changed beyond selected model fields:\ngot %+v\nwant %+v", got.Features, want.Features)
			}
			if a.assignment.open {
				t.Error("dialog stayed open after applying")
			}
		})
	}
}

func TestModelAssignmentSelectionAndRevalidation(t *testing.T) {
	a, _, _ := newTestApp(t)
	readyModels(a)
	model := local.Model{Kind: local.LLM, ID: "translate-only", Name: "Translation only", Features: []string{hotkey.Translate}}
	a.local.(*fakeLocal).MoreLLM = []local.Model{model}
	a.openModelAssignment(local.LLM, model.ID)
	if a.assignment.selected != [4]bool{false, true, false, false} {
		t.Fatalf("unsupported features preselected: %v", a.assignment.selected)
	}
	before := a.settings.Get()
	a.assignment.selected = [4]bool{true, true, true, true}
	a.applyModelAssignment()
	want := before
	want.Features.Translate.LLM = model.ID
	if !reflect.DeepEqual(a.settings.Get(), want) {
		t.Error("unsupported assignment changed settings")
	}

	a.openModelAssignment(local.LLM, model.ID)
	a.local.(*fakeLocal).MoreLLM = nil
	before = a.settings.Get()
	a.applyModelAssignment()
	if !reflect.DeepEqual(a.settings.Get(), before) || !a.assignment.open {
		t.Error("removed model was applied or dialog closed")
	}

	a.assignment = modelAssignment{}
	a.openModelAssignment(local.LLM, "gone")
	if a.assignment.open {
		t.Error("missing model opened via automatic fallback")
	}

	m := a.installed(local.ASR)[0]
	a.updateSettings(func(s *settings.Settings) { s.Features.Rewrite.ASR = m.ID })
	a.openModelAssignment(local.ASR, m.ID)
	if a.assignment.selected != [4]bool{false, false, true, false} {
		t.Errorf("explicit existing assignments not preselected: %v", a.assignment.selected)
	}
	before = a.settings.Get()
	a.assignment.selected = [4]bool{}
	a.applyModelAssignment()
	if !reflect.DeepEqual(a.settings.Get(), before) {
		t.Error("empty selection changed settings")
	}
}

func TestModelAssignmentRemoteCard(t *testing.T) {
	useKeys(t, &memKeys{m: map[string]string{}})
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 1000)
	a.navigate(remotePath)
	frames(tt)
	click(t, tt, "DeepSeek")
	a.provDlg.key = "sk-test-assignment"
	click(t, tt, "Save")
	frames(tt)
	click(t, tt, "Apply to")
	if !a.assignment.open || a.provDlg.open {
		t.Fatalf("assignment button also opened provider dialog: assignment=%v provider=%v", a.assignment.open, a.provDlg.open)
	}
	click(t, tt, "Apply")
	for _, feature := range assignmentFeatures {
		if assignmentID(a.settings.Get(), local.LLM, feature) != "deepseek" {
			t.Errorf("%s not persisted", feature)
		}
	}
}

func TestModelAssignmentDialog(t *testing.T) {
	a, _, tt := newTestApp(t)
	readyModels(a)
	tt.SetSize(windowWidth, 1000)
	a.navigate(remotePath)
	m := a.installed(local.LLM)[0]
	a.openModelAssignment(local.LLM, m.ID)
	frames(tt)
	if !has(tt, "Apply to features") || !has(tt, "Automatic (now: "+m.Name+")") {
		t.Fatalf("missing dialog or automatic model: %v", tt.Texts())
	}
	a.updateSettings(func(s *settings.Settings) { s.Features.Translate.LLM = "temporarily-unavailable" })
	frames(tt)
	if !has(tt, "Current: Unavailable: temporarily-unavailable") {
		t.Fatalf("missing explicit binding shown as automatic: %v", tt.Texts())
	}
	before := a.settings.Get()
	click(t, tt, "Cancel")
	if a.assignment.open || !reflect.DeepEqual(a.settings.Get(), before) {
		t.Error("cancel changed settings or stayed open")
	}
	a.openModelAssignment(local.LLM, m.ID)
	frames(tt)
	click(t, tt, "Apply model to Translate")
	click(t, tt, "Apply")
	want := before
	want.Features.Transcribe.LLM = m.ID
	want.Features.Rewrite.LLM = m.ID
	want.Features.Meeting.LLM = m.ID
	if !reflect.DeepEqual(a.settings.Get(), want) {
		t.Errorf("checkbox selection not persisted: %+v", a.settings.Get().Features)
	}
}
