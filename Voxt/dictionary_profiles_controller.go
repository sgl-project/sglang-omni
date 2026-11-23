package main

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type dictionaryProfileState struct {
	list  []store.DictionaryProfile
	reads int // numbers the reads: only the last one lands
}

// dictionaryProfilesFingerprint is the profiles as one comparable string:
// a switch marks the active one in place.
func dictionaryProfilesFingerprint(profiles []store.DictionaryProfile) string {
	var b strings.Builder
	for _, profile := range profiles {
		fmt.Fprintf(&b, "%s\x00%s\x00%t\x00", profile.ID, profile.Name, profile.Active)
	}
	return b.String()
}

func (a *app) activeDictionaryProfile() store.DictionaryProfile {
	for _, profile := range a.dictProfiles.list {
		if profile.Active {
			return profile
		}
	}
	return store.DictionaryProfile{}
}

func (a *app) loadDictionaryProfiles() {
	profiles := &a.dictProfiles
	profiles.reads++
	read, repo := profiles.reads, a.lib.store.Dictionary()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		list, err := repo.Profiles(ctx)
		a.post(func() {
			if read != a.dictProfiles.reads {
				return
			}
			if err != nil {
				a.toast(i18n.T("Could not read the dictionary profiles: %v", err))
				return
			}
			a.dictProfiles.list = list
		})
	})
}

func (a *app) markDictionaryProfileActive(id string) {
	for index := range a.dictProfiles.list {
		a.dictProfiles.list[index].Active = a.dictProfiles.list[index].ID == id
	}
}

func (a *app) switchDictionaryProfile(id string) {
	repo := a.lib.store.Dictionary()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		err := withDictionaryLock(repo, func() error { return repo.ActivateProfile(ctx, id) })
		a.post(func() {
			if err != nil {
				a.toast(i18n.T("Could not switch the dictionary profile: %v", err))
				return
			}
			a.markDictionaryProfileActive(id)
			a.loadDict()
		})
	})
}

// switchDictionaryProfileByName is the external command's entry point. It
// changes no navigation and needs neither the history socket nor a permission.
func (a *app) switchDictionaryProfileByName(name string) {
	repo := a.lib.store.Dictionary()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		profiles, err := repo.Profiles(ctx)
		a.post(func() {
			if err != nil {
				a.toast(i18n.T("Could not switch the dictionary profile: %v", err))
				return
			}
			want := strings.TrimSpace(name)
			for _, profile := range profiles {
				if profile.Name == want || dictionaryProfileName(i18n.CurrentTranslator(), profile) == want {
					a.switchDictionaryProfile(profile.ID)
					return
				}
			}
			a.toast(i18n.T("No dictionary profile named “%s”", name))
		})
	})
}

// openDictionaryProfileEdit opens the profile dialog on id: a new profile
// when id is "".
func (a *app) openDictionaryProfileEdit(id string) {
	edit := dictionaryProfileEdit{open: true}
	for _, profile := range a.dictProfiles.list {
		if id != "" && profile.ID == id {
			edit.id, edit.name = profile.ID, profile.Name
		}
	}
	a.dictUI.profileEdit = edit
}

func profileErrorText(err error) string {
	switch {
	case errors.Is(err, core.ErrExists):
		return i18n.T("A profile with this name already exists.")
	case errors.Is(err, core.ErrInvalid):
		return i18n.T("Type a profile name.")
	default:
		return i18n.T("Could not save the profile: %v", err)
	}
}

func (a *app) saveDictionaryProfile() {
	edit := &a.dictUI.profileEdit
	name := strings.TrimSpace(edit.name)
	if name == "" {
		edit.err = i18n.T("Type a profile name.")
		return
	}
	edit.saving = true
	id, repo := edit.id, a.lib.store.Dictionary()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		err := withDictionaryLock(repo, func() error {
			if id == "" {
				return repo.CreateProfile(ctx, store.DictionaryProfile{ID: core.NewID(time.Now()), Name: name})
			}
			return repo.RenameProfile(ctx, id, name)
		})
		a.post(func() {
			edit := &a.dictUI.profileEdit
			edit.saving = false
			if err != nil {
				edit.err = profileErrorText(err)
				return
			}
			edit.open = false
			a.loadDictionaryProfiles()
		})
	})
}

func (a *app) deleteDictionaryProfile(id string) {
	repo := a.lib.store.Dictionary()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		err := withDictionaryLock(repo, func() error { return repo.DeleteProfile(ctx, id) })
		a.post(func() {
			if err != nil {
				a.toast(i18n.T("Could not delete the profile: %v", err))
				return
			}
			a.loadDictionaryProfiles()
			a.loadDict()
		})
	})
}
