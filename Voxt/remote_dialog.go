package main

// The Language models page's dialog (remote_page.go; Auralink's service
// dialog): a service's key, address and model, with a connection test
// before saving, and Remove.

import (
	"slices"
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// providerDialog is the open service dialog's form.
type providerDialog struct {
	open               bool
	id                 string
	exists             bool // the service was set up before
	key, endpoint      string
	model, customModel string // model: the select's label
	err, errMore       string // a save or check error
	saving, removeKey  bool
	test               providerTest
}

// otherModel is the model menu's last choice: a model id typed in.
func otherModel() string { return i18n.T("Other model…") }

func (a *app) openProviderDialog(id string) {
	pr, _ := llm.LookupPreset(id)
	p, exists := providers.Setup(a.prefs, id)
	d := providerDialog{open: true, id: id, exists: exists, endpoint: p.Endpoint}
	d.model = providers.Model(p)
	if d.model == "" || !slices.Contains(pr.Models, d.model) {
		d.model, d.customModel = otherModel(), d.model
	}
	a.provDlg = d
	if _, read := a.keyMasks[id]; !read {
		a.loadKeyMasks([]string{id})
	}
}

// setup is the setup the dialog's form describes.
func (d *providerDialog) setup() settings.Provider {
	pr, _ := llm.LookupPreset(d.id)
	p := settings.Provider{ID: d.id, Endpoint: strings.TrimSpace(d.endpoint)}
	model := d.model
	if model == otherModel() {
		model = strings.TrimSpace(d.customModel)
	}
	if model != pr.DefaultModel {
		p.Model = model
	}
	return p
}

func (a *app) providerDialogView(c *ui.Context) {
	d := &a.provDlg
	if !d.open {
		return
	}
	t := c.Theme()
	pr, _ := llm.LookupPreset(d.id)
	name := uitext.PresetName(d.id)
	dialogFrame(c, "remote-dialog", &d.open, name, i18n.T("Remote language model · OpenAI-compatible API"), 560, func() {
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
			what := i18n.T("Recognised and selected text is sent to %s.", name)
			if pr.BaseURL == "" {
				what = i18n.T("Recognised and selected text is sent to the server at the address below.")
			}
			caption(c, what).Grow(1).Basis(px(t, 240))
			if pr.KeyURL != "" {
				ui.Link(c.Key("remote-keyurl"), i18n.T("Get a key"), pr.KeyURL).FontSize(px(t, fsCaption))
			}
		})
		mask := a.keyMasks[d.id]
		keyInfo := i18n.T("Saved in this Mac's Keychain, never in Voxt's settings or logs.")
		if pr.KeyOptional {
			keyInfo = i18n.T("Optional for a server on this Mac. Saved in this Mac's Keychain, never in Voxt's settings or logs.")
		}
		fieldInfo(c, i18n.T("API key"), keyInfo, func() {
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
				ph := i18n.T("Paste the key")
				if mask != "" && !d.removeKey {
					ph = i18n.T("Saved: %s · paste a new one to replace it", mask)
				}
				ui.TextInput(c.Key("remote-key"), &d.key).Password().Placeholder(ph).Label(i18n.T("API key")).Grow(1).MinWidth(0)
				if mask != "" && !d.removeKey && ghostButton(c.Key("remote-key-remove"), i18n.T("Remove")).Label(i18n.T("Remove the saved key")).Clicked() {
					d.removeKey, d.key = true, ""
				}
			})
			if d.removeKey {
				caption(c, i18n.T("The saved key is removed when you save."))
			}
		})
		epInfo := i18n.T("The API root, ending in /v1; leave it empty for the service's own. Only https://, or http:// to this Mac.")
		if pr.BaseURL == "" {
			epInfo = i18n.T("The API root of any OpenAI-compatible server, ending in /v1. Only https://, or http:// to this Mac.")
		}
		fieldInfo(c, i18n.T("Address"), epInfo, func() {
			ph := pr.BaseURL
			if ph == "" {
				ph = "https://…/v1"
			}
			ui.TextInput(c.Key("remote-endpoint"), &d.endpoint).Placeholder(ph).Label(i18n.T("Address"))
		})
		fieldInfo(c, i18n.T("Model"), "", func() {
			opts := append(slices.Clone(pr.Models), otherModel())
			ui.Select(c.Key("remote-model"), &d.model, opts).Label(i18n.T("Model")).Width(px(t, 320))
			if d.model == otherModel() {
				ui.TextInput(c.Key("remote-model-custom"), &d.customModel).Placeholder(i18n.T("Model id")).Label(i18n.T("Model id"))
			}
		})
		if d.err != "" {
			notice(c, "remote-dlg-err", noticeError, d.err, d.errMore, nil, func() { d.err, d.errMore = "", "" })
		}
	}, func() {
		if ui.Button(c.Key("remote-dlg-test"), i18n.T("Test connection")).Disabled(d.test.running || d.saving).Shrink(0).Clicked() {
			a.testDialog()
		}
		// The test's state, beside its button; a long reason is cut, its
		// ⓘ tells the whole of it.
		ui.Row(c.Key("remote-dlg-status")).Gap(px(t, sp1)).AlignItems(ui.Center).Grow(1).Shrink(1).Basis(0).MinWidth(0).Children(func() {
			testStatus(c, d.test, true)
		})
		if d.exists && ghostButton(c.Key("remote-dlg-remove"), i18n.T("Remove")).Label(i18n.T("Remove %s", name)).Disabled(d.saving).Shrink(0).Clicked() {
			a.removeProvider()
		}
		if ghostButton(c.Key("remote-dlg-cancel"), i18n.T("Cancel")).Shrink(0).Clicked() {
			d.open = false
		}
		if primaryButton(c.Key("remote-dlg-save"), i18n.T("Save")).Disabled(d.saving).Shrink(0).Clicked() {
			a.saveProviderDialog()
		}
	})
}

// check says what the form lacks before it can be tested or saved ("" for
// nothing), as an error's text and detail.
func (d *providerDialog) check(p settings.Provider) (msg, more string) {
	if strings.TrimSpace(d.model) == "" || d.model == otherModel() && strings.TrimSpace(d.customModel) == "" {
		return i18n.T("Choose a model, or type its id."), ""
	}
	if _, err := providers.Config(p).Resolve(); err != nil {
		return uitext.ModelErrorText(err)
	}
	return "", ""
}

// testDialog tests what the dialog's form describes, saved or not: with
// the key typed, else the saved one (read in the background), none when
// it is to be removed.
func (a *app) testDialog() {
	d := &a.provDlg
	p := d.setup()
	if d.err, d.errMore = d.check(p); d.err != "" {
		return
	}
	typed, remove, kr := strings.TrimSpace(d.key), d.removeKey, a.keys
	a.testProvider(p, func() (string, error) {
		switch {
		case typed != "":
			return typed, nil
		case remove:
			return "", nil
		}
		return kr.Get(p.ID)
	})
}

// saveProviderDialog checks the form, saves the key (when one was typed
// or removed) and then the setup.
func (a *app) saveProviderDialog() {
	d := &a.provDlg
	p := d.setup()
	if d.err, d.errMore = d.check(p); d.err != "" {
		return
	}
	key, remove := strings.TrimSpace(d.key), d.removeKey
	label := "Voxt " + uitext.PresetName(p.ID) + " API key"
	d.saving = true
	kr := a.keys
	a.async(func() {
		var err error
		switch {
		case key != "":
			err = kr.Set(p.ID, label, key)
		case remove:
			err = kr.Set(p.ID, label, "")
		}
		a.post(func() {
			a.provDlg.saving = false
			if err != nil {
				a.provDlg.err, a.provDlg.errMore = i18n.T("Could not save the key in the Keychain."), err.Error()
				return
			}
			switch {
			case key != "":
				a.keyMasks[p.ID] = maskKey(key)
			case remove:
				a.keyMasks[p.ID] = ""
			}
			delete(a.provTests, p.ID)
			a.updateSettings(func(s *settings.Settings) { providers.Put(s, p) })
			a.syncRemote() // a new key: the settings may not have changed
			a.provDlg = providerDialog{}
			a.toast(i18n.T("%s saved: choose it as the language model of Transcribe, Translate or Rewrite.", uitext.ProviderName(p)))
		})
	})
}

// removeProvider removes the dialog's service: its setup, its key, and
// the features that used it go back to automatic.
func (a *app) removeProvider() {
	d := &a.provDlg
	id, name, kr := d.id, uitext.PresetName(d.id), a.keys
	d.saving = true
	a.async(func() {
		err := kr.Set(id, "", "")
		a.post(func() {
			a.provDlg.saving = false
			if err != nil {
				a.provDlg.err, a.provDlg.errMore = i18n.T("Could not remove the key from the Keychain."), err.Error()
				return
			}
			a.keyMasks[id] = ""
			delete(a.provTests, id)
			a.updateSettings(func(s *settings.Settings) { providers.Remove(s, id) })
			a.provDlg = providerDialog{}
			a.toast(i18n.T("%s removed", name))
		})
	})
}
