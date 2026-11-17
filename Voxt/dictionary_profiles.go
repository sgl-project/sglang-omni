package main

// Dictionary profiles: named sets of terms. The repository reads and writes
// the active profile, so sessions, meetings, imports and learning follow the
// same choice as this page. Other programs switch it through the existing
// voxtnew:// URL scheme (docs/dictionary-profiles.md).

import (
	"context"
	"errors"
	"strings"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type dictionaryProfileState struct {
	list  []store.DictionaryProfile
	reads int

	editOpen   bool
	editID     string
	editName   string
	editErr    string
	editSaving bool
	deleteOpen bool
	deleteID   string
}

// profileName is a profile's name as the user sees it: the built-in
// Default translates; the user's own names stay as typed.
func profileName(p store.DictionaryProfile) string {
	if p.ID == "default" {
		return i18n.T("Default")
	}
	return p.Name
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
	p := &a.dictProfiles
	p.reads++
	read, repo := p.reads, a.lib.store.Dictionary()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		profiles, err := repo.Profiles(ctx)
		a.post(func() {
			if read != a.dictProfiles.reads {
				return
			}
			if err != nil {
				a.toast(i18n.T("Could not read the dictionary profiles: %v", err))
				return
			}
			a.dictProfiles.list = profiles
		})
	})
}

func (a *app) markDictionaryProfileActive(id string) {
	for i := range a.dictProfiles.list {
		a.dictProfiles.list[i].Active = a.dictProfiles.list[i].ID == id
	}
}

func (a *app) switchDictionaryProfile(id string) {
	repo := a.lib.store.Dictionary()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		err := repo.ActivateProfile(ctx, id)
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
			for _, profile := range profiles {
				if profile.Name == strings.TrimSpace(name) || profileName(profile) == strings.TrimSpace(name) {
					a.switchDictionaryProfile(profile.ID)
					return
				}
			}
			a.toast(i18n.T("No dictionary profile named “%s”", name))
		})
	})
}

func (a *app) dictionaryProfileBar(c *ui.Context) {
	t := c.Theme()
	profiles := a.dictProfiles.list
	if len(profiles) == 0 {
		return
	}
	active := a.activeDictionaryProfile()
	selected, names := 0, make([]string, len(profiles))
	for i, profile := range profiles {
		names[i] = profileName(profile)
		if profile.Active {
			selected = i
		}
	}
	ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
		caption(c, i18n.T("Active profile"))
		if segmentedFit(c.Key("dict-profile"), &selected, names...).Label(i18n.T("Active dictionary profile")).Changed() {
			a.switchDictionaryProfile(profiles[selected].ID)
		}
		if ghostIconButton(c.Key("dict-profile-new"), icons.Add, i18n.T("New profile")).Clicked() {
			p := &a.dictProfiles
			p.editOpen, p.editID, p.editName, p.editErr = true, "", "", ""
		}
		moreMenu(c.Key("dict-profile-more"), i18n.T("Profile actions"), func(m *ui.Menu) {
			if m.Item(i18n.T("Rename “%s”…", profileName(active))).Disabled(active.ID == "").Chosen() {
				p := &a.dictProfiles
				p.editOpen, p.editID, p.editName, p.editErr = true, active.ID, active.Name, ""
			}
			if m.Item(i18n.T("Delete “%s”…", profileName(active))).Disabled(len(profiles) <= 1).Chosen() {
				a.dictProfiles.deleteOpen, a.dictProfiles.deleteID = true, active.ID
			}
		})
	})
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
	p := &a.dictProfiles
	name := strings.TrimSpace(p.editName)
	if name == "" {
		p.editErr = i18n.T("Type a profile name.")
		return
	}
	p.editSaving = true
	id, repo := p.editID, a.lib.store.Dictionary()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		var err error
		if id == "" {
			err = repo.CreateProfile(ctx, store.DictionaryProfile{ID: core.NewID(time.Now()), Name: name})
		} else {
			err = repo.RenameProfile(ctx, id, name)
		}
		a.post(func() {
			p := &a.dictProfiles
			p.editSaving = false
			if err != nil {
				p.editErr = profileErrorText(err)
				return
			}
			p.editOpen = false
			a.loadDictionaryProfiles()
		})
	})
}

func (a *app) deleteDictionaryProfile() {
	id, repo := a.dictProfiles.deleteID, a.lib.store.Dictionary()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		err := repo.DeleteProfile(ctx, id)
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

func (a *app) dictionaryProfileDialogs(c *ui.Context) {
	p := &a.dictProfiles
	if p.editOpen {
		title := i18n.T("New profile")
		if p.editID != "" {
			title = i18n.T("Rename profile")
		}
		dialogFrame(c, "dict-profile-edit", &p.editOpen, title, "", 420, func() {
			fieldInfo(c, i18n.T("Name"), i18n.T("Use different dictionary entries for different situations."), func() {
				in := ui.TextInput(c.Key("dict-profile-name"), &p.editName).Label(i18n.T("Profile name")).Placeholder(i18n.T("Work")).AutoFocus()
				if in.Changed() {
					p.editErr = ""
				}
				if in.Submitted() {
					a.saveDictionaryProfile()
				}
			})
			if p.editErr != "" {
				warnLine(c, p.editErr)
			}
		}, func() {
			ui.Spacer(c)
			if ui.Button(c.Key("dict-profile-cancel"), i18n.T("Cancel")).Clicked() {
				p.editOpen = false
			}
			if primaryButton(c.Key("dict-profile-save"), i18n.T("Save")).Disabled(p.editSaving).Clicked() {
				a.saveDictionaryProfile()
			}
		})
	}
	if p.deleteOpen {
		name := ""
		for _, profile := range p.list {
			if profile.ID == p.deleteID {
				name = profileName(profile)
			}
		}
		dialogFrame(c, "dict-profile-delete", &p.deleteOpen, i18n.T("Delete “%s”?", name), "", 420, func() {
			ui.Text(c, i18n.T("Its dictionary entries will be deleted. This cannot be undone."))
		}, func() {
			ui.Spacer(c)
			if ui.Button(c.Key("dict-profile-delete-cancel"), i18n.T("Cancel")).AutoFocus().Clicked() {
				p.deleteOpen = false
			}
			if recordButton(c.Key("dict-profile-delete-ok"), i18n.T("Delete")).Clicked() {
				p.deleteOpen = false
				a.deleteDictionaryProfile()
			}
		})
	}
}
