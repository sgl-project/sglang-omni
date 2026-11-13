package main

// The remote services (Auralink's model services): language models and
// speech recognition, the user's setups (settings.Providers), their keys in
// the Keychain, and a.remote, the local.Services that offers the ready
// ones to the features after the local models of their kind
// (internal/providers). The pages and their dialog are remote_page.go and
// remote_dialog.go.
//
// Keys live in the Keychain only (keyring, internal/secrets): the app
// keeps their masks ("sk-…wxyz"), read in the background, never a key; a
// session's open reads the key off the main thread. Masks, tests and the
// keyring go by Keychain account (providers.Account: "llm.openai",
// "asr.openai"), as both lists have services of the same id.

import (
	"context"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/codexauth"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/secrets"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// remotePath is the Language models page's path.
const remotePath = "/language-models"

// service is a remote service as its page and dialog show it: a language
// model's preset (internal/llm) or a speech service (providers.Speech).
type service struct {
	kind     local.Kind
	id, name string
	// endpoint is the default address ("": the user gives one), model the
	// default model ("": none).
	endpoint, model string
	models          []string
	keyOptional     bool
	appID           bool // Doubao: an App ID may come with the key
	signIn          bool // signed in to, not a key (remote_chatgpt.go)
	keyURL          string
	// speech is a speech service's facts (which models stream); zero for
	// a language model.
	speech providers.Speech
}

// services are kind's remote services, in the page's order.
func services(kind local.Kind) []service {
	var out []service
	if kind == local.ASR {
		for _, p := range providers.SpeechPresets() {
			out = append(out, service{kind: kind, id: p.ID, name: uitext.SpeechName(p.ID), endpoint: p.Endpoint, model: p.Model, models: p.Models,
				appID: p.AppID, keyOptional: p.KeyOptional, keyURL: p.KeyURL, speech: p})
		}
		return out
	}
	for _, p := range llm.Presets() {
		out = append(out, service{kind: kind, id: p.ID, name: uitext.PresetName(p.ID), endpoint: p.BaseURL, model: p.DefaultModel,
			models: p.Models, keyOptional: p.KeyOptional, keyURL: p.KeyURL, signIn: p.SignIn})
	}
	return out
}

// lookupService is kind's service id (its id alone when unknown).
func lookupService(kind local.Kind, id string) service {
	for _, sv := range services(kind) {
		if sv.id == id {
			return sv
		}
	}
	return service{kind: kind, id: id, name: id}
}

// setupOf is the user's setup of kind's service id, and whether there is
// one.
func (a *app) setupOf(kind local.Kind, id string) (settings.Provider, bool) {
	if kind == local.ASR {
		return providers.SpeechSetup(a.prefs, id)
	}
	return providers.Setup(a.prefs, id)
}

// setupModel is the model a setup uses: its own, else its service's
// default.
func setupModel(kind local.Kind, p settings.Provider) string {
	if kind == local.ASR {
		return providers.SpeechModel(p)
	}
	return providers.Model(p)
}

// setupName is how a ready setup is named where a model is chosen: its
// service and model ("OpenAI · gpt-4.1-mini").
func setupName(kind local.Kind, p settings.Provider) string {
	return lookupService(kind, p.ID).name + " · " + setupModel(kind, p)
}

// setupMissing says what a setup lacks before it can be used (a speech
// service: its key only), "" when it is ready.
func (a *app) setupMissing(kind local.Kind, p settings.Provider) string {
	switch {
	case kind == local.LLM:
		return providers.Missing(p, a.hasKey(kind, p.ID))
	case a.hasKey(kind, p.ID), providers.SpeechKeyOptional(p):
		return ""
	}
	return providers.MissingKey
}

// keyring keeps the remote services' keys, by Keychain account
// ("llm.openai"). Its calls may block: off the main thread only.
type keyring interface {
	Get(account string) (string, error)
	Set(account, label, value string) error // "" deletes
}

// keychain is the macOS Keychain (internal/secrets).
type keychain struct{}

func (keychain) Get(account string) (string, error) { return secrets.Key(account, "").Get() }
func (keychain) Set(account, label, value string) error {
	return secrets.Key(account, label).Set(value)
}

// newKeyring is the keyring a new app uses; tests swap in a memory one.
var newKeyring = func() keyring { return keychain{} }

// remoteState is the remote models and the Language models page's state.
type remoteState struct {
	// remote is a.local with the remote models (nil when tests set
	// a.local themselves).
	remote *providers.Services
	// remoteHTTP is the remote models' HTTP client, through the proxy
	// the downloads use (family.go networkFor); set on the main thread,
	// read by opens.
	remoteHTTP atomic.Pointer[http.Client]
	keys       keyring
	// proxyPassword is a custom proxy's password, as read or saved
	// (models_proxy_auth.go); networkFor reads it off the main thread too.
	proxyPassword atomic.Pointer[proxySecret]
	// keyMasks are the saved keys' masks by account (the signed-in email
	// for ChatGPT), "" for none; an account not in it was not read yet.
	keyMasks map[string]string
	// provTests are the last connection tests, by account.
	provTests map[string]providerTest
	provDlg   providerDialog
	chatgpt   *codexauth.Session // remote_chatgpt.go
	chatLogin chatgptLogin
}

// providerTest is a connection test running or done.
type providerTest struct {
	running bool
	res     providers.TestResult
	detail  string // the error for the ⓘ, without the key
}

// withRemote is base with the remote models; their keys are read in the
// background.
func (a *app) withRemote(base local.Services) local.Services {
	a.keys = newKeyring()
	a.keyMasks = map[string]string{}
	a.provTests = map[string]providerTest{}
	a.remote = providers.NewServices(base, setupName, a.openRemote, a.openSpeech)
	a.readUnreadKeys()
	a.syncRemote()
	a.chatgpt = a.newChatGPTSession() // remote_chatgpt.go
	a.loadProxyPassword()             // models_proxy_auth.go
	return a.remote
}

// openRemote opens a language model setup for a session (off the main
// thread): its key from the Keychain, then its client.
func (a *app) openRemote(_ context.Context, p settings.Provider) (local.Chat, error) {
	if p.ID == providers.ChatGPT {
		return providers.OpenSignedIn(p, providers.ChatGPTTokens(a.chatgpt), a.remoteHTTP.Load())
	}
	key, err := a.keys.Get(providers.Account(local.LLM, p.ID))
	if err != nil {
		return nil, err
	}
	return providers.Open(p, key, a.remoteHTTP.Load())
}

// openSpeech opens a speech setup for a session, as openRemote.
func (a *app) openSpeech(_ context.Context, p settings.Provider) (asr.Provider, error) {
	key, err := a.keys.Get(providers.Account(local.ASR, p.ID))
	if err != nil {
		return nil, err
	}
	return providers.NewASR(p, key, a.remoteHTTP.Load())
}

// syncRemote gives a.remote the setups that are ready and the network,
// after a change of the settings or the keys, and updates the sidebar's
// card about missing models.
func (a *app) syncRemote() {
	if a.remote == nil {
		return
	}
	a.remoteHTTP.Store(unboundedHeaders(a.networkFor(a.prefs).Client))
	a.remote.SetReady(local.LLM, providers.Ready(a.prefs, func(id string) bool { return a.hasKey(local.LLM, id) }))
	a.remote.SetReady(local.ASR, providers.SpeechReady(a.prefs, func(id string) bool { return a.hasKey(local.ASR, id) }))
	a.checkModels() // shell_notices.go: a ready remote model is a model
}

// unboundedHeaders is c without the downloads' 30 s wait for an answer's
// headers: an upload service answers a whole recording only once it is
// recognised, which can take minutes. The session's deadline, or the
// request's own timeout, bounds the request instead.
func unboundedHeaders(c *http.Client) *http.Client {
	t, ok := c.Transport.(*http.Transport)
	if !ok {
		return c
	}
	t = t.Clone()
	t.ResponseHeaderTimeout = 0
	return &http.Client{Transport: t}
}

// readUnreadKeys reads the keys of the setups whose key was not read yet:
// all of them at the start, then those a change added (the import from
// the previous Voxt), so that their cards and the sidebar say whether a
// key is missing.
func (a *app) readUnreadKeys() {
	if a.keys == nil {
		return // tests with local models of their own
	}
	var accounts []string
	unread := func(kind local.Kind, list []settings.Provider) {
		for _, p := range list {
			if _, read := a.keyMasks[providers.Account(kind, p.ID)]; !read {
				accounts = append(accounts, providers.Account(kind, p.ID))
			}
		}
	}
	unread(local.LLM, a.prefs.Providers.LLM)
	unread(local.ASR, a.prefs.Providers.ASR)
	a.loadKeyMasks(accounts)
}

// loadKeyMasks reads the keys of accounts from the keyring, in the
// background, and keeps their masks.
func (a *app) loadKeyMasks(accounts []string) {
	if len(accounts) == 0 {
		return
	}
	kr := a.keys
	a.async(func() {
		masks := map[string]string{}
		for _, acct := range accounts {
			v, err := kr.Get(acct)
			if err != nil {
				v = ""
			}
			masks[acct] = maskFor(acct, v)
		}
		a.post(func() {
			for k, v := range masks {
				a.keyMasks[k] = v
			}
			a.syncRemote()
		})
	})
}

// pendingSetup is the first of kind's setups that lacks what it needs
// (its key, address or model: providers.Missing*), and what; false for
// none. A key not read from the Keychain yet is not missing.
func (a *app) pendingSetup(kind local.Kind) (settings.Provider, string, bool) {
	list := a.prefs.Providers.LLM
	if kind == local.ASR {
		list = a.prefs.Providers.ASR
	}
	for _, p := range list {
		if _, read := a.keyMasks[providers.Account(kind, p.ID)]; !read {
			continue
		}
		if missing := a.setupMissing(kind, p); missing != "" {
			return p, missing, true
		}
	}
	return settings.Provider{}, "", false
}

// setupNeedsText says what kind's setup p lacks (providers.Missing*).
func setupNeedsText(kind local.Kind, p settings.Provider, missing string) string {
	name := lookupService(kind, p.ID).name
	switch missing {
	case providers.MissingKey:
		if lookupService(kind, p.ID).signIn {
			return i18n.T("%s needs you to sign in again.", name)
		}
		return i18n.T("%s needs its API key. Paste it again to use it.", name)
	case providers.MissingEndpoint:
		return i18n.T("%s needs its address.", name)
	case providers.MissingAddress:
		return i18n.Platform("Change the address of %s. Voxt connects only over https, or http to this Mac, and the address cannot contain a user name or password.", "Change the address of %s. Voxt connects only over https, or http to this PC, and the address cannot contain a user name or password.", name)
	}
	return i18n.T("%s needs a model.", name)
}

// linkCompleteSetup starts the link of a card whose action opens a
// setup's dialog: linkCompleteSetup + kind + "/" + id (not a deep link).
const linkCompleteSetup = "action:complete-setup:"

// completeSetup opens the dialog of kind's setup id on its page.
func (a *app) completeSetup(kind local.Kind, id string) {
	page := remotePath
	if kind == local.ASR {
		page = modelsPath
	}
	a.navigate(page)
	a.openProviderDialog(kind, id)
}

// hasKey reports whether a key is saved for kind's setup id (as last
// read).
func (a *app) hasKey(kind local.Kind, id string) bool {
	return a.keyMasks[providers.Account(kind, id)] != ""
}

func maskKey(v string) string {
	if v == "" {
		return ""
	}
	return secrets.Mask(v)
}

// testProvider runs a connection test of kind's setup p with key (any
// goroutine's), then shows it on its card and, when it is open, in the
// dialog.
func (a *app) testProvider(kind local.Kind, p settings.Provider, key func() (string, error)) {
	acct := providers.Account(kind, p.ID)
	a.provTests[acct] = providerTest{running: true}
	dialog := func() bool { return a.provDlg.open && providers.Account(a.provDlg.kind, a.provDlg.id) == acct }
	if dialog() {
		a.provDlg.test = providerTest{running: true}
	}
	client, logs := a.remoteHTTP.Load(), a.logs
	signedIn := kind == local.LLM && p.ID == providers.ChatGPT
	var tokens llm.TokenSource
	if signedIn {
		tokens = providers.ChatGPTTokens(a.chatgpt)
		key = func() (string, error) { return "", nil } // the sign-in is the session's
	}
	a.async(func() {
		k, err := key()
		res := providers.TestResult{Err: err, At: time.Now()}
		var level, msg string
		switch {
		case err != nil:
		case signedIn:
			res = runChatGPTTest(context.Background(), p, tokens, client)
		case kind == local.ASR:
			res = runSpeechTest(context.Background(), p, k, client)
		default:
			res = runProviderTest(context.Background(), p, k, client)
		}
		if kind == local.ASR {
			level, msg = providers.SpeechTestLog(p, res, k)
		} else {
			level, msg = providers.TestLog(p, res, k)
		}
		if logs != nil {
			logs.Append(core.LogEntry{Time: time.Now(), Level: core.Severity(level), Source: "app", Message: msg})
		}
		tr := providerTest{res: res, detail: providers.ErrorLog(res.Err, k)}
		a.post(func() {
			a.provTests[acct] = tr
			if dialog() {
				a.provDlg.test = tr
			}
		})
	})
}

// runProviderTest is providers.Test, runSpeechTest providers.TestSpeech;
// tests swap them for fakes.
var (
	runProviderTest = providers.Test
	runSpeechTest   = providers.TestSpeech
	runChatGPTTest  = providers.TestSignedIn
)
