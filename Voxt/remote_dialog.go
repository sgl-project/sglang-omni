package main

// The remote services' dialog (remote_page.go; Auralink's service
// dialog): a service's key (and Doubao's App ID), address and model, with
// a connection test before saving, and Remove.

import (
	"reflect"
	"slices"
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// providerDialog is the open service dialog's form.
type providerDialog struct {
	open               bool
	kind               local.Kind
	id                 string
	appID              string
	exists             bool // the service was set up before
	key, endpoint      string
	model, customModel string // model: the select's label
	recognitionPrompt  string
	err, errMore       string // a save or check error
	saving, removeKey  bool
	test               providerTest
	gen                genForm // a language model's generation settings (remote_generation.go)
	// was is the setup the dialog opened with: a click beside the dialog,
	// or Escape, does not close it while the form differs (unsaved).
	was settings.Provider
}

// unsaved says whether the form holds what Save has not kept: a key typed
// or to remove, or a changed address, model or App ID.
func (d *providerDialog) unsaved() bool {
	return d.key != "" || d.removeKey || !reflect.DeepEqual(d.setup(), d.was)
}

// otherModel is the model menu's last choice: a model id typed in.
func otherModel() string { return i18n.T("Other model…") }

func (a *app) openProviderDialog(kind local.Kind, id string) {
	sv := lookupService(kind, id)
	p, exists := a.setupOf(kind, id)
	d := providerDialog{open: true, kind: kind, id: id, exists: exists, endpoint: p.Endpoint, appID: p.AppID, recognitionPrompt: p.RecognitionPrompt, gen: genFormOf(p.Generation)}
	d.gen.api = apiFormatTitle(p.APIFormat)
	d.model = setupModel(kind, p)
	if d.model == "" || !slices.Contains(sv.models, d.model) {
		d.model, d.customModel = otherModel(), d.model
	}
	d.gen.fit(providers.Capabilities(id, d.setup().Model))
	d.was = d.setup()
	a.provDlg = d
	if acct := providers.Account(kind, id); !a.keyRead(acct) {
		a.loadKeyMasks([]string{acct})
	}
}

// keyRead says whether account's key was read (its mask may be "").
func (a *app) keyRead(account string) bool {
	_, read := a.keyMasks[account]
	return read
}

// setup is the setup the dialog's form describes.
func (d *providerDialog) setup() settings.Provider {
	p := settings.Provider{ID: d.id, Endpoint: strings.TrimSpace(d.endpoint), AppID: strings.TrimSpace(d.appID)}
	model := d.model
	if model == otherModel() {
		model = strings.TrimSpace(d.customModel)
	}
	if model != lookupService(d.kind, d.id).model {
		p.Model = model
	}
	switch d.kind {
	case local.LLM:
		p.Generation, _ = d.gen.settings()
		p.APIFormat = valueOf(d.gen.api, apiFormats, apiFormatTitle)
	case local.ASR:
		p.RecognitionPrompt = strings.TrimSpace(d.recognitionPrompt)
	}
	return p
}

func (a *app) providerDialogView(c *ui.Context) {
	d := &a.provDlg
	if !d.open {
		a.cancelChatGPTLogin() // closed while waiting for the browser
		return
	}
	t := c.Theme()
	sv := lookupService(d.kind, d.id)
	name := sv.name
	model := providers.SpeechModel(d.setup())
	streams := d.kind == local.ASR && sv.speech.Streams(model)
	sub := i18n.T("Remote language model · OpenAI-compatible API")
	switch {
	case sv.signIn:
		sub = i18n.T("Remote language model · your ChatGPT plan")
	case streams:
		sub = i18n.T("Remote speech recognition · realtime, while you speak")
	case d.kind == local.ASR:
		sub = i18n.T("Remote speech recognition · the recording, once you stop")
	}
	if dialogFrame(c, "remote-dialog", &d.open, name, sub, 560, func() {
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
			what := i18n.T("Recognised and selected text is sent to %s.", name)
			switch {
			case d.kind == local.ASR:
				what = i18n.T("Your recordings are sent to %s to be recognised.", name)
			case sv.endpoint == "":
				what = i18n.T("Recognised and selected text is sent to the server at the address below.")
			}
			caption(c, what).Grow(1).Basis(px(t, 240))
			if sv.keyURL != "" {
				ui.Link(c.Key("remote-keyurl"), i18n.T("Get a key"), sv.keyURL).FontSize(px(t, fsCaption))
			}
		})
		if sv.appID {
			fieldInfo(c, i18n.T("App ID"), i18n.T("From the app in the Volcengine speech console, with its Access Token as the key below. A key of the newer console needs no App ID."), func() {
				ui.TextInput(c.Key("remote-appid"), &d.appID).Placeholder(i18n.T("Optional")).Label(i18n.T("App ID"))
			})
		}
		if sv.signIn {
			a.chatgptFields(c, d) // remote_chatgpt.go
		}
		mask := a.keyMasks[providers.Account(d.kind, d.id)]
		keyInfo := i18n.T("Saved in this Mac's Keychain, never in Voxt's settings or logs.")
		if sv.keyOptional {
			keyInfo = i18n.T("Optional for a server on this Mac. Saved in this Mac's Keychain, never in Voxt's settings or logs.")
		}
		if !sv.signIn {
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
		}
		epInfo := i18n.T("The API root, ending in /v1; leave it empty for the service's own. Only https://, or http:// to this Mac.")
		switch {
		case streams:
			epInfo = i18n.T("Leave it empty for the service's own. Only wss://, or ws:// to this Mac.")
		case d.kind == local.ASR:
			epInfo = i18n.T("Leave it empty for the service's own. Only https://, or http:// to this Mac.")
		case sv.endpoint == "":
			epInfo = i18n.T("The API root of any OpenAI-compatible server, ending in /v1. Only https://, or http:// to this Mac.")
		}
		if !sv.signIn {
			fieldInfo(c, i18n.T("Address"), epInfo, func() {
				ph := sv.endpoint
				if d.kind == local.ASR {
					ph = sv.speech.EndpointOf(model)
				}
				if ph == "" {
					ph = "https://…/v1"
				}
				ui.TextInput(c.Key("remote-endpoint"), &d.endpoint).Placeholder(ph).Label(i18n.T("Address"))
			})
		}
		fieldInfo(c, i18n.T("Model"), "", func() {
			opts := append(slices.Clone(sv.models), otherModel())
			ui.Select(c.Key("remote-model"), &d.model, opts).Label(i18n.T("Model")).Width(px(t, 320))
			if d.model == otherModel() {
				ui.TextInput(c.Key("remote-model-custom"), &d.customModel).Placeholder(i18n.T("Model id")).Label(i18n.T("Model id"))
			}
		})
		if d.kind == local.ASR {
			fieldInfo(c, i18n.T("Recognition prompt"), i18n.T("Optional. Goes after the global speech context on the Speech models page. Use {{USER_MAIN_LANGUAGE}}, {{USER_OTHER_LANGUAGES}} and {{DICTIONARY_TERMS}} for your languages and dictionary. Services that take no prompt ignore it."), func() {
				ui.TextArea(c.Key("remote-asr-prompt"), &d.recognitionPrompt).Label(i18n.T("Recognition prompt")).
					Placeholder(i18n.T("For example: Technical talk; prefer product names from the dictionary.")).MinHeight(px(t, 72))
			})
		}
		if d.kind == local.LLM && !sv.signIn {
			a.generationFields(c, d) // remote_generation.go
		}
		if d.err != "" {
			notice(c, "remote-dlg-err", noticeError, d.err, d.errMore, nil, func() { d.err, d.errMore = "", "" })
		}
	}, func() {
		if ui.Button(c.Key("remote-dlg-test"), i18n.T("Test connection")).Disabled(d.test.running || d.saving || sv.signIn && !a.hasKey(d.kind, d.id)).Shrink(0).Clicked() {
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
	}).Dismissed() && d.unsaved() {
		d.open = true // a stray click or Escape: keep what was typed
	}
}

// check says what the form lacks before it can be tested or saved ("" for
// nothing), as an error's text and detail.
func (d *providerDialog) check(p settings.Provider) (msg, more string) {
	if strings.TrimSpace(d.model) == "" || d.model == otherModel() && strings.TrimSpace(d.customModel) == "" {
		return i18n.T("Choose a model, or type its id."), ""
	}
	if err := checkSetup(d.kind, p); err != nil {
		return uitext.ModelErrorText(err)
	}
	if d.kind == local.LLM {
		if _, bad := d.gen.settings(); bad != "" {
			return i18n.T("%s is not a number.", bad), ""
		}
		switch providers.CheckGeneration(p.Generation) {
		case providers.ErrJSONSchema:
			return i18n.T("The JSON schema is not a JSON object."), ""
		case providers.ErrOptions:
			return i18n.T("Options JSON is not a JSON object."), ""
		case providers.ErrExtraBody:
			return i18n.T("Extra Body JSON is not a JSON object."), ""
		}
	}
	return "", ""
}

// checkSetup checks a setup's address before it is tested or saved.
func checkSetup(kind local.Kind, p settings.Provider) error {
	if kind == local.ASR {
		_, err := providers.NewASR(p, "check", nil) // any key: only the address is checked here
		return err
	}
	_, err := providers.Config(p).Resolve()
	return err
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
	typed, remove, kr, acct := strings.TrimSpace(d.key), d.removeKey, a.keys, providers.Account(d.kind, d.id)
	a.testProvider(d.kind, p, func() (string, error) {
		switch {
		case typed != "":
			return typed, nil
		case remove:
			return "", nil
		}
		return kr.Get(acct)
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
	key, remove, kind := strings.TrimSpace(d.key), d.removeKey, d.kind
	sv, acct := lookupService(kind, p.ID), providers.Account(kind, p.ID)
	label := "Voxt " + sv.name + " API key"
	d.saving = true
	kr := a.keys
	a.async(func() {
		var err error
		switch {
		case key != "":
			err = kr.Set(acct, label, key)
		case remove:
			err = kr.Set(acct, label, "")
		}
		a.post(func() {
			a.provDlg.saving = false
			if err != nil {
				a.provDlg.err, a.provDlg.errMore = i18n.T("Could not save the key in the Keychain."), err.Error()
				return
			}
			switch {
			case key != "":
				a.keyMasks[acct] = maskKey(key)
			case remove:
				a.keyMasks[acct] = ""
			}
			delete(a.provTests, acct)
			a.updateSettings(func(s *settings.Settings) {
				if kind == local.ASR {
					providers.PutSpeech(s, p)
				} else {
					providers.Put(s, p)
				}
			})
			a.syncRemote() // a new key: the settings may not have changed
			a.provDlg = providerDialog{}
			a.toast(i18n.T("%s saved", setupName(kind, p)))
		})
	})
}

// removeProvider removes the dialog's service: its setup, its key, and
// the features that used it go back to automatic.
func (a *app) removeProvider() {
	d := &a.provDlg
	kind, id, kr := d.kind, d.id, a.keys
	name, acct := lookupService(kind, id).name, providers.Account(kind, id)
	d.saving = true
	a.async(func() {
		var err error
		if id == providers.ChatGPT {
			err = a.chatgpt.SignOut()
		} else {
			err = kr.Set(acct, "", "")
		}
		a.post(func() {
			a.provDlg.saving = false
			if err != nil {
				a.provDlg.err, a.provDlg.errMore = i18n.T("Could not remove the key from the Keychain."), err.Error()
				return
			}
			a.keyMasks[acct] = ""
			delete(a.provTests, acct)
			a.updateSettings(func(s *settings.Settings) {
				if kind == local.ASR {
					providers.RemoveSpeech(s, id)
				} else {
					providers.Remove(s, id)
				}
			})
			a.provDlg = providerDialog{}
			a.toast(i18n.T("%s removed", name))
		})
	})
}
