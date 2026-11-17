package main

import (
	"context"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func TestDictionaryProfileControls(t *testing.T) {
	a, tt := dictTester(t)
	seedTerms(t, a)
	a.loadDict()
	frames(tt)
	click(t, tt, "New profile")
	click(t, tt, "Profile name")
	tt.Type("Work")
	click(t, tt, "Save")
	profiles, err := a.lib.store.Dictionary().Profiles(context.Background())
	if err != nil || len(profiles) != 2 || profiles[1].Name != "Work" || !profiles[0].Active {
		t.Fatalf("created profiles %+v, %v; state %+v; view %q", profiles, err, a.dictProfiles, firstTexts(tt))
	}
	if got := terms(t, a); len(got) != 5 || !has(tt, "Work") {
		t.Fatalf("creating changed the active dictionary: %+v", got)
	}
	click(t, tt, "Work")
	frames(tt)
	profiles, _ = a.lib.store.Dictionary().Profiles(context.Background())
	if !profiles[1].Active || len(terms(t, a)) != 0 {
		t.Errorf("Work is not the active empty dictionary: %+v", profiles)
	}
	a.saveTerm(store.Term{Term: "ActiveOnly", Enabled: true}, false)
	if got := terms(t, a); len(got) != 1 || got[0].Term != "ActiveOnly" {
		t.Errorf("term saved outside Work: %+v", got)
	}
	click(t, tt, "Profile actions")
	if err := tt.ChooseMenuItem("Rename “Work”…"); err != nil {
		t.Fatal(err)
	}
	a.dictProfiles.editName = "Office"
	click(t, tt, "Save")
	profiles, _ = a.lib.store.Dictionary().Profiles(context.Background())
	if profiles[1].Name != "Office" || !profiles[1].Active || !has(tt, "Office") {
		t.Errorf("rename did not apply: %+v", profiles)
	}
	click(t, tt, "Profile actions")
	if err := tt.ChooseMenuItem("Delete “Office”…"); err != nil {
		t.Fatal(err)
	}
	if !has(tt, "Delete “Office”?") {
		t.Fatalf("no delete confirmation: %q", firstTexts(tt))
	}
	click(t, tt, "Delete")
	profiles, _ = a.lib.store.Dictionary().Profiles(context.Background())
	if len(profiles) != 1 || !profiles[0].Active || len(terms(t, a)) != 5 {
		t.Errorf("delete did not restore Default: %+v", profiles)
	}
	click(t, tt, "Profile actions")
	if err := tt.ChooseMenuItem("Delete “Default”…"); err == nil || !strings.Contains(err.Error(), "disabled") {
		t.Errorf("the last profile can be deleted: %v", err)
	}
	tt.CloseMenu()
}

func TestDictionaryProfileExternalSwitch(t *testing.T) {
	a, _, _ := newTestApp(t)
	repo := a.lib.store.Dictionary()
	if err := repo.Put(context.Background(), store.Term{ID: "default-term", Term: "DefaultOnly", Enabled: true}); err != nil {
		t.Fatal(err)
	}
	if err := repo.CreateProfile(context.Background(), store.DictionaryProfile{ID: "writing", Name: "Research & Writing"}); err != nil {
		t.Fatal(err)
	}
	if err := repo.ActivateProfile(context.Background(), "writing"); err != nil {
		t.Fatal(err)
	}
	if err := repo.Put(context.Background(), store.Term{ID: "writing-term", Term: "WritingOnly", Enabled: true}); err != nil {
		t.Fatal(err)
	}
	if err := repo.ActivateProfile(context.Background(), "default"); err != nil {
		t.Fatal(err)
	}
	a.navigate("/history")
	// This is the URL opened by the documented `open -g` shell command.
	a.openURL("voxtnew://dictionary/profile?name=Research%20%26%20Writing")
	profiles, err := repo.Profiles(context.Background())
	if err != nil || len(profiles) != 2 || !profiles[1].Active {
		t.Fatalf("external switch: %+v, %v", profiles, err)
	}
	if got, _ := repo.List(context.Background(), ""); len(got) != 1 || got[0].ID != "writing-term" {
		t.Errorf("external switch used the wrong entries: %+v", got)
	}
	if a.router.Path() != "/history" {
		t.Errorf("the switch opened the Dictionary page: %s", a.router.Path())
	}
	a.openURL("voxtnew://dictionary/profile?name=Missing")
	profiles, _ = repo.Profiles(context.Background())
	if !profiles[1].Active || len(a.toasts) == 0 {
		t.Errorf("unknown profile changed the active one or showed no error: %+v, %q", profiles, a.toasts)
	}
}

// A term dialog opened in one profile saves and deletes there, even
// after a switch: the active profile at the time must not retarget
// them (#4052 review).
func TestDictionaryTermDialogStaysInItsProfile(t *testing.T) {
	a, _, _ := newTestApp(t)
	repo := a.lib.store.Dictionary()
	if err := repo.Put(context.Background(), store.Term{ID: "t1", Term: "Alpha", Enabled: true}); err != nil {
		t.Fatal(err)
	}
	if err := repo.CreateProfile(context.Background(), store.DictionaryProfile{ID: "work", Name: "Work"}); err != nil {
		t.Fatal(err)
	}
	// The same id in Work: a mistargeted Delete would take it.
	if err := repo.ActivateProfile(context.Background(), "work"); err != nil {
		t.Fatal(err)
	}
	if err := repo.Put(context.Background(), store.Term{ID: "t1", Term: "WorkAlpha", Enabled: true}); err != nil {
		t.Fatal(err)
	}
	if err := repo.ActivateProfile(context.Background(), "default"); err != nil {
		t.Fatal(err)
	}
	a.loadDictionaryProfiles()

	// Open Alpha's editor in Default, switch to Work, then save.
	a.openTerm(store.Term{ID: "t1", Term: "Alpha", Enabled: true})
	a.dv.edit.term = "AlphaTwo"
	a.switchDictionaryProfile("work")
	a.submitTerm()
	if got := terms(t, a); len(got) != 1 || got[0].Term != "WorkAlpha" {
		t.Fatalf("the save wrote into Work: %+v", got)
	}
	if err := repo.ActivateProfile(context.Background(), "default"); err != nil {
		t.Fatal(err)
	}
	if got := terms(t, a); len(got) != 1 || got[0].Term != "AlphaTwo" {
		t.Fatalf("the save did not update Default: %+v", got)
	}

	// And Delete, with the same id present in Work.
	a.switchDictionaryProfile("default")
	a.openTerm(store.Term{ID: "t1", Term: "AlphaTwo", Enabled: true})
	a.switchDictionaryProfile("work")
	a.deleteTerm(store.Term{ID: a.dv.edit.id, Term: a.dv.edit.original, ProfileID: a.dv.edit.profile})
	if got := terms(t, a); len(got) != 1 || got[0].Term != "WorkAlpha" {
		t.Fatalf("the delete took Work's term: %+v", got)
	}
	if err := repo.ActivateProfile(context.Background(), "default"); err != nil {
		t.Fatal(err)
	}
	if got := terms(t, a); len(got) != 0 {
		t.Fatalf("the delete missed Default's term: %+v", got)
	}
}

// The built-in Default profile shows translated, and the external
// switch takes the shown name too.
func TestDictionaryProfileDefaultTranslated(t *testing.T) {
	a, tt := dictTester(t)
	i18n.SetLocale("zh-Hans")
	defer i18n.SetLocale("en")
	frames(tt)
	if !has(tt, "默认") {
		t.Errorf("the built-in profile is not translated: %q", firstTexts(tt))
	}
	a.openURL("voxtnew://dictionary/profile?name=默认")
	for _, toast := range a.toasts {
		t.Errorf("switch by the translated name toasted %q", toast)
	}
	profiles, err := a.lib.store.Dictionary().Profiles(context.Background())
	if err != nil || len(profiles) != 1 || !profiles[0].Active {
		t.Errorf("switch by the translated name: %+v, %v", profiles, err)
	}
}
