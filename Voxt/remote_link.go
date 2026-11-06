package main

// The remote language models (Auralink's model services, language models
// only): the user's setups (settings.Providers), their keys in the
// Keychain, and a.remote, the local.Services that offers the ready ones to
// the features after the local models (internal/providers). The page and
// its dialog are remote_page.go and remote_dialog.go.
//
// Keys live in the Keychain only (keyring, internal/secrets): the app
// keeps their masks ("sk-…wxyz"), read in the background, never a key; a
// session's open reads the key off the main thread.

import (
	"context"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/secrets"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// remotePath is the Remote models page's path.
const remotePath = "/remote-models"

// keyring keeps the remote models' keys, by setup id ("openai"). Its
// calls may block: off the main thread only.
type keyring interface {
	Get(id string) (string, error)
	Set(id, label, value string) error // "" deletes
}

// keychain is the macOS Keychain (internal/secrets).
type keychain struct{}

func (keychain) Get(id string) (string, error)     { return secrets.LLM(id, "").Get() }
func (keychain) Set(id, label, value string) error { return secrets.LLM(id, label).Set(value) }

// newKeyring is the keyring a new app uses; tests swap in a memory one.
var newKeyring = func() keyring { return keychain{} }

// remoteState is the remote models and the Remote models page's state.
type remoteState struct {
	// remote is a.local with the remote models (nil when tests set
	// a.local themselves).
	remote *providers.Services
	// remoteHTTP is the remote models' HTTP client, through the proxy
	// the downloads use (family.go networkFor); set on the main thread,
	// read by opens.
	remoteHTTP atomic.Pointer[http.Client]
	keys       keyring
	// keyMasks are the saved keys' masks by setup id, "" for none; an id
	// not in it was not read yet.
	keyMasks map[string]string
	// provTests are the last connection tests, by setup id.
	provTests map[string]providerTest
	provDlg   providerDialog
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
	a.remote = providers.NewServices(base, uitext.ProviderName, a.openRemote)
	var ids []string
	for _, p := range a.prefs.Providers.LLM {
		ids = append(ids, p.ID)
	}
	a.loadKeyMasks(ids)
	a.syncRemote()
	return a.remote
}

// openRemote opens a setup for a session (off the main thread): its key
// from the Keychain, then its client.
func (a *app) openRemote(_ context.Context, p settings.Provider) (local.Chat, error) {
	key, err := a.keys.Get(p.ID)
	if err != nil {
		return nil, err
	}
	return providers.Open(p, key, a.remoteHTTP.Load())
}

// syncRemote gives a.remote the setups that are ready and the network,
// after a change of the settings or the keys.
func (a *app) syncRemote() {
	if a.remote == nil {
		return
	}
	a.remoteHTTP.Store(a.networkFor(a.prefs).Client)
	a.remote.SetReady(providers.Ready(a.prefs, a.hasKey))
}

// loadKeyMasks reads the keys of ids from the keyring, in the
// background, and keeps their masks.
func (a *app) loadKeyMasks(ids []string) {
	if len(ids) == 0 {
		return
	}
	kr := a.keys
	a.async(func() {
		masks := map[string]string{}
		for _, id := range ids {
			v, err := kr.Get(id)
			if err != nil {
				v = ""
			}
			masks[id] = maskKey(v)
		}
		a.post(func() {
			for k, v := range masks {
				a.keyMasks[k] = v
			}
			a.syncRemote()
		})
	})
}

// hasKey reports whether a key is saved for setup id (as last read).
func (a *app) hasKey(id string) bool { return a.keyMasks[id] != "" }

func maskKey(v string) string {
	if v == "" {
		return ""
	}
	return secrets.Mask(v)
}

// testProvider runs a connection test of setup p with key (any
// goroutine's), then shows it on its card and, when it is open, in the
// dialog.
func (a *app) testProvider(p settings.Provider, key func() (string, error)) {
	a.provTests[p.ID] = providerTest{running: true}
	if a.provDlg.open && a.provDlg.id == p.ID {
		a.provDlg.test = providerTest{running: true}
	}
	client, logs := a.remoteHTTP.Load(), a.logs
	a.async(func() {
		k, err := key()
		res := providers.TestResult{Err: err, At: time.Now()}
		if err == nil {
			res = runProviderTest(context.Background(), p, k, client)
		}
		if logs != nil {
			level, msg := providers.TestLog(p, res, k)
			logs.Append(core.LogEntry{Time: time.Now(), Level: core.Severity(level), Source: "app", Message: msg})
		}
		tr := providerTest{res: res, detail: providers.ErrorLog(res.Err, k)}
		a.post(func() {
			a.provTests[p.ID] = tr
			if a.provDlg.open && a.provDlg.id == p.ID {
				a.provDlg.test = tr
			}
		})
	})
}

// runProviderTest is providers.Test; tests swap it for a fake.
var runProviderTest = providers.Test
