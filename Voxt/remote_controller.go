package main

// Composition between app state and the remote area's pure view-state
// builders and views. Shared Speech models callers keep these app facades.

import (
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func (a *app) remoteCardsStateInput(kind local.Kind) remoteCardsStateInput {
	input := remoteCardsStateInput{Kind: kind}
	for _, sv := range services(kind) {
		setup, configured := a.setupOf(kind, sv.id)
		account := providers.Account(kind, sv.id)
		mask, read := a.keyMasks[account]
		test, hasTest := a.provTests[account]
		serviceInput := remoteServiceStateInput{
			Service: sv, Setup: setup, Configured: configured, Missing: a.setupMissing(kind, setup),
			Mask: mask, KeyRead: read, KeySpaced: a.keySpaced[account], Test: test, HasTest: hasTest,
		}
		if configured {
			serviceInput.Used = a.usedBy(kind, sv.id)
		}
		input.Services = append(input.Services, serviceInput)
	}
	return input
}

func (a *app) prepareRemoteDialog() {
	if !a.provDlg.open {
		a.cancelChatGPTLogin()
		return
	}
	setup := a.provDlg.setup()
	a.provDlg.gen.fit(providers.Capabilities(a.provDlg.id, setup.Model))
}

func (a *app) remoteDialogStateInput() *remoteDialogStateInput {
	d := &a.provDlg
	if !d.open {
		return nil
	}
	setup := d.setup()
	account := providers.Account(d.kind, d.id)
	return &remoteDialogStateInput{
		Dialog: d, Service: lookupService(d.kind, d.id), Model: setupModel(d.kind, setup),
		Capabilities: providers.Capabilities(d.id, setup.Model), Mask: a.keyMasks[account],
		KeySpaced: a.keySpaced[account], ChatEmail: a.keyMasks[chatgptAccount()], ChatLogin: a.chatLogin, Unsaved: d.unsaved(),
	}
}

func (a *app) remoteDialogModel() *remoteDialogViewState {
	a.prepareRemoteDialog()
	input := a.remoteDialogStateInput()
	if input == nil {
		return nil
	}
	state := buildRemoteDialogViewState(i18n.CurrentTranslator(), *input)
	return &state
}

func remoteKind(kind string) local.Kind {
	if kind == string(local.ASR) {
		return local.ASR
	}
	return local.LLM
}

func (a *app) openRemoteProvider(kind, id string) { a.openProviderDialog(remoteKind(kind), id) }

func (a *app) testRemoteProvider(kind, id string) {
	modelKind := remoteKind(kind)
	setup, _ := a.setupOf(modelKind, id)
	account := providers.Account(modelKind, id)
	keys := a.keys
	a.testProvider(modelKind, setup, func() (string, error) { return keys.Get(account) })
}

func (a *app) chooseLLMWait(value string, custom bool, minutes int) {
	if custom {
		value = settings.WaitMinutes(minutes)
	}
	a.llmWaitFor = value + "\x00" + i18n.Locale()
	a.updateSettings(func(s *settings.Settings) { s.Models.LLMWait = value })
}

func (a *app) changeLLMWaitMinutes(minutes int) { a.chooseLLMWait("", true, minutes) }

func (a *app) dismissOnboardingBack() { a.ob.awayTo = "" }

func (a *app) openLLMDebug() { a.navigate(llmDebugPath) }

func (a *app) openRemoteKeyURL() {
	if url := lookupService(a.provDlg.kind, a.provDlg.id).keyURL; url != "" {
		openExternal(url)
	}
}

func (a *app) openChatGPTUsage() { openExternal(chatgptUsageURL) }

func (a *app) dismissChatGPTError() {
	a.chatLogin.err, a.chatLogin.more, a.chatLogin.code = "", "", ""
}
