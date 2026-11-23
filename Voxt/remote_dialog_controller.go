package main

// The remote service dialog's controller. It owns the saved setup, model
// discovery, validation, connection tests, credential writes and removal.

import (
	"context"
	"reflect"
	"slices"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/secrets"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// providerDialogLocalUI is the open service dialog's editable widget state.
type providerDialogLocalUI struct {
	open               bool
	appID              string
	key, endpoint      string
	model, customModel string // model: the select's label
	recognitionPrompt  string
	err, errMore       string // a save or check error
	removeKey          bool
	showKey            bool // the typed key in clear
	gen                genForm
}

// providerDialog is the open service dialog's controller state.
type providerDialog struct {
	providerDialogLocalUI
	kind   local.Kind
	id     string
	exists bool // the service was set up before
	saving bool
	test   providerTest
	served []string // the models the server says it has (listModels); nil: the preset's list
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
func otherModel() string { return otherModelFor(i18n.CurrentTranslator()) }

func otherModelFor(translator *i18n.Translator) string { return translator.T("Other model…") }

func (a *app) openProviderDialog(kind local.Kind, id string) {
	sv := lookupService(kind, id)
	p, exists := a.setupOf(kind, id)
	d := providerDialog{kind: kind, id: id, exists: exists, was: settings.Provider{}}
	d.providerDialogLocalUI = providerDialogLocalUI{open: true, endpoint: p.Endpoint, appID: p.AppID, recognitionPrompt: p.RecognitionPrompt, gen: genFormOf(p.Generation)}
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
	a.listModels()
}

// listModels asks a language model server on this Mac (Ollama, LM Studio,
// oMLX, a custom one) which models it has, for the Model menu: the
// preset's names may not be installed (#3950). With no answer the menu is
// the preset's.
func (a *app) listModels() {
	d := &a.provDlg
	if d.kind != local.LLM || !lookupService(d.kind, d.id).keyOptional {
		return
	}
	p, acct := d.setup(), providers.Account(d.kind, d.id)
	typed, remove, kr, client := strings.TrimSpace(d.key), d.removeKey, a.keys, a.remoteHTTP.Load()
	a.async(func() {
		key := typed
		if key == "" && !remove {
			key, _ = kr.Get(acct)
		}
		models, err := runListModels(context.Background(), p, key, client)
		a.post(func() {
			if !a.provDlg.open || providers.Account(a.provDlg.kind, a.provDlg.id) != acct || a.provDlg.setup().Endpoint != p.Endpoint {
				return // closed, or an answer for an address since changed
			}
			if err != nil || len(models) == 0 {
				a.provDlg.served = nil // back to the preset's list, not another address's
				return
			}
			a.provDlg.offer(models)
		})
	})
}

// runListModels is providers.Models; tests swap it.
var runListModels = providers.Models

// offer puts the server's models in the Model menu. A typed model ID the
// server has becomes its menu item; a new, untouched setup whose default
// the server lacks takes the server's first model.
func (d *providerDialog) offer(models []string) {
	untouched := !d.exists && !d.unsaved()
	d.served = models
	cur := d.model
	if cur == otherModel() {
		cur = strings.TrimSpace(d.customModel)
	}
	switch {
	case slices.Contains(models, cur):
		d.model, d.customModel = cur, ""
	case untouched:
		d.model, d.customModel = models[0], ""
		d.was = d.setup()
	}
}

// modelOptions is the Model menu: the server's models when it listed
// them, else the preset's, then the setup's own model if neither has it,
// then Other.
func (d *providerDialog) modelOptions(preset []string) []string {
	return remoteModelOptions(preset, d.served, d.model, otherModel())
}

func remoteModelOptions(preset, served []string, current, other string) []string {
	opts := slices.Clone(preset)
	if served != nil {
		opts = slices.Clone(served)
	}
	if current != other && current != "" && !slices.Contains(opts, current) {
		opts = append(opts, current)
	}
	return append(opts, other)
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

// check says what the form lacks before it can be tested or saved ("" for
// nothing), as an error's text and detail.
func (d *providerDialog) check(p settings.Provider) (msg, more string) {
	if strings.TrimSpace(d.model) == "" || d.model == otherModel() && strings.TrimSpace(d.customModel) == "" {
		return i18n.T("Choose a model, or type its ID."), ""
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
		case providers.ErrExtraBody:
			return i18n.T("The request parameters are not a JSON object."), ""
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
	typed, _ := secrets.Clean(d.key)
	remove, kr, acct := d.removeKey, a.keys, providers.Account(d.kind, d.id)
	a.testProvider(d.kind, p, func() (string, error) {
		switch {
		case typed != "":
			return typed, nil
		case remove:
			return "", nil
		}
		return kr.Get(acct)
	})
	a.listModels() // the address may have changed since the menu was filled
}

// saveProviderDialog checks the form, saves the key (when one was typed
// or removed) and then the setup.
func (a *app) saveProviderDialog() {
	d := &a.provDlg
	p := d.setup()
	if d.err, d.errMore = d.check(p); d.err != "" {
		return
	}
	key, _ := secrets.Clean(d.key)
	remove, kind := d.removeKey, d.kind
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
				delete(a.keySpaced, acct)
			case remove:
				a.keyMasks[acct] = ""
				delete(a.keySpaced, acct)
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
			delete(a.keySpaced, acct)
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
