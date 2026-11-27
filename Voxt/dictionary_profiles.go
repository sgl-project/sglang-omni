package main

// Dictionary profiles: named sets of terms, switched from the menu beside
// the page's title. The repository reads and writes the active profile, so sessions,
// meetings, imports and learning follow the same choice as this page.
// Other programs switch it through the voxtnew:// URL scheme
// (docs/dictionary-profiles.md; dictionary_profiles_controller.go).

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// dictionaryProfileEdit is the New or Rename profile dialog: a new profile
// when id is "".
type dictionaryProfileEdit struct {
	open, saving  bool
	id, name, err string
}

// dictionaryProfilePicker is the active profile beside the page's title,
// a menu of the profiles with a check by the active one, and below them
// New, Rename and Delete. The page shows it only with more than one
// profile (dictionaryHeader).
func dictionaryProfilePicker(c *ui.Context, state *dictionaryViewState, actions dictionaryActions, local *dictionaryLocalUI) {
	profiles := state.Profiles
	if len(profiles) == 0 {
		return
	}
	active := profiles[0]
	for _, profile := range profiles {
		if profile.Active {
			active = profile
		}
	}
	dropButton(c.Key("dict-profile"), dictionaryIcon, active.Name).Label(i18n.T("Active dictionary profile")).Menu(func(menu *ui.Menu) {
		for _, profile := range profiles {
			if menu.Item(profile.Name).Checked(profile.Active).Chosen() && !profile.Active {
				actions.switchDictionaryProfile(profile.ID)
			}
		}
		menu.Separator()
		if menu.Item(i18n.T("New profile…")).Chosen() {
			actions.openDictionaryProfileEdit("")
		}
		if menu.Item(i18n.T("Rename “%s”…", active.Name)).Disabled(!active.Active).Chosen() {
			actions.openDictionaryProfileEdit(active.ID)
		}
		if menu.Item(i18n.T("Delete “%s”…", active.Name)).Disabled(len(profiles) <= 1).Chosen() {
			local.profileDeleteOpen, local.profileDeleteID = true, active.ID
		}
	})
}

func dictionaryProfileDialogs(c *ui.Context, state *dictionaryViewState, actions dictionaryActions, local *dictionaryLocalUI) {
	edit := &local.profileEdit
	if edit.open {
		title := i18n.T("New profile")
		if edit.id != "" {
			title = i18n.T("Rename profile")
		}
		dialogFrame(c, "dict-profile-edit", &edit.open, title, "", 420, func() {
			fieldInfo(c, i18n.T("Name"), i18n.T("Use different dictionary entries for different situations."), func() {
				input := ui.TextInput(c.Key("dict-profile-name"), &edit.name).Label(i18n.T("Profile name")).Placeholder(i18n.T("Work")).AutoFocus()
				if input.Changed() {
					edit.err = ""
				}
				if input.Submitted() {
					actions.saveDictionaryProfile()
				}
			})
			if edit.err != "" {
				warnLine(c, edit.err)
			}
		}, func() {
			ui.Spacer(c)
			if ui.Button(c.Key("dict-profile-cancel"), i18n.T("Cancel")).Clicked() {
				edit.open = false
			}
			if primaryButton(c.Key("dict-profile-save"), i18n.T("Save")).Disabled(edit.saving).Clicked() {
				actions.saveDictionaryProfile()
			}
		})
	}
	if local.profileDeleteOpen {
		name := ""
		for _, profile := range state.Profiles {
			if profile.ID == local.profileDeleteID {
				name = profile.Name
			}
		}
		dialogFrame(c, "dict-profile-delete", &local.profileDeleteOpen, i18n.T("Delete “%s”?", name), "", 420, func() {
			ui.Text(c, i18n.T("Its dictionary entries will be deleted. This cannot be undone."))
		}, func() {
			ui.Spacer(c)
			if ui.Button(c.Key("dict-profile-delete-cancel"), i18n.T("Cancel")).AutoFocus().Clicked() {
				local.profileDeleteOpen = false
			}
			if recordButton(c.Key("dict-profile-delete-ok"), i18n.T("Delete")).Clicked() {
				local.profileDeleteOpen = false
				actions.deleteDictionaryProfile(local.profileDeleteID)
			}
		})
	}
}
