package main

// The ChatGPT (Codex) language model's sign-in: its dialog's fields
// (remote_dialog.go) and the session of internal/codexauth, whose item in
// the Keychain is this setup's "key". The transport gets its tokens from
// that session (providers.ChatGPTTokens).

import (
	"context"
	"errors"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/codexauth"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// chatgptUsageURL is where a ChatGPT plan's usage is managed.
const chatgptUsageURL = "https://chatgpt.com/codex/settings/usage"

var (
	// chatgptEndpoints and openBrowser are swapped in tests.
	chatgptEndpoints = codexauth.OpenAI
	openBrowser      = func(url string) error { return mygo.Shell.OpenExternal(url) }
)

// chatgptLogin is the browser sign-in under way, or how the last ended.
type chatgptLogin struct {
	running   bool
	cancel    context.CancelFunc
	err, more string
}

func chatgptAccount() string { return providers.Account(local.LLM, providers.ChatGPT) }

// chatgptStore is the sign-in's Keychain item, through the app's keyring.
type chatgptStore struct {
	keys  keyring
	label string
}

func (s chatgptStore) Get() (string, error)   { return s.keys.Get(chatgptAccount()) }
func (s chatgptStore) Set(value string) error { return s.keys.Set(chatgptAccount(), s.label, value) }

func (a *app) newChatGPTSession() *codexauth.Session {
	return codexauth.NewSession(codexauth.Config{
		Store:     chatgptStore{keys: a.keys, label: "Voxt ChatGPT (Codex) sign-in"},
		Endpoints: chatgptEndpoints,
		HTTP:      a.remoteHTTP.Load,
		Success:   i18n.T("You are signed in. You can close this tab and go back to Voxt."),
		Failure:   i18n.T("The sign-in did not finish. Go back to Voxt and try again."),
		SignedOut: func() { a.post(func() { a.chatgptGone() }) },
	})
}

// chatgptGone notes that the sign-in is no more.
func (a *app) chatgptGone() {
	a.keyMasks[chatgptAccount()] = ""
	delete(a.provTests, chatgptAccount())
	a.syncRemote()
}

// maskFor is what the app keeps of a saved key: its mask, or for the
// sign-in its email (never any of the item itself, which is a token).
func maskFor(account, value string) string {
	if account != chatgptAccount() || value == "" {
		return maskKey(value)
	}
	if email := codexauth.Email(value); email != "" {
		return email
	}
	return "•••"
}

// signInChatGPT runs the browser sign-in in the background.
func (a *app) signInChatGPT() {
	if a.chatLogin.running {
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	a.chatLogin = chatgptLogin{running: true, cancel: cancel}
	sess := a.chatgpt
	a.async(func() {
		email, err := sess.Login(ctx, openBrowser)
		cancel()
		a.post(func() { a.chatgptLoginDone(email, err) })
	})
}

func (a *app) cancelChatGPTLogin() {
	if a.chatLogin.cancel != nil {
		a.chatLogin.cancel()
	}
}

func (a *app) chatgptLoginDone(email string, err error) {
	a.chatLogin = chatgptLogin{}
	if errors.Is(err, context.Canceled) {
		return
	}
	if err != nil {
		a.chatLogin.err, a.chatLogin.more = chatgptLoginText(err)
		return
	}
	acct := chatgptAccount()
	a.keyMasks[acct] = email
	if email == "" {
		a.keyMasks[acct] = "•••"
	}
	delete(a.provTests, acct)
	// Signing in is the setup: there is nothing else to save, so that a
	// closed dialog does not lose it.
	d := &a.provDlg
	p, _ := providers.Setup(a.prefs, providers.ChatGPT)
	if d.open && d.id == providers.ChatGPT {
		p = d.setup()
		d.exists, d.was = true, p
	}
	a.updateSettings(func(s *settings.Settings) { providers.Put(s, p) })
	a.syncRemote()
	a.toast(i18n.T("Signed in to ChatGPT"))
}

// chatgptLoginText says why the sign-in failed, and its detail.
func chatgptLoginText(err error) (msg, more string) {
	more = err.Error()
	var e *codexauth.Error
	if !errors.As(err, &e) {
		return i18n.T("The sign-in failed."), more
	}
	switch e.Kind {
	case codexauth.KindPortInUse:
		return i18n.T("Another program is using port %d, which the sign-in needs (a running `codex login`, perhaps). Close it and try again.", e.Port), more
	case codexauth.KindDenied:
		return i18n.T("The sign-in was refused in the browser."), more
	case codexauth.KindTimeout:
		return i18n.T("The sign-in timed out: the browser did not come back to Voxt."), more
	case codexauth.KindNetwork:
		return i18n.T("Could not reach ChatGPT's sign-in service: check your network and proxy."), more
	case codexauth.KindStore:
		return i18n.T("Could not save the sign-in in the Keychain."), more
	}
	return i18n.T("ChatGPT did not finish the sign-in: try again."), more
}

// signOutChatGPT deletes the sign-in from this Mac; the setup stays.
func (a *app) signOutChatGPT() {
	sess := a.chatgpt
	a.async(func() {
		err := sess.SignOut()
		a.post(func() {
			if err != nil {
				a.chatLogin.err, a.chatLogin.more = i18n.T("Could not remove the sign-in from the Keychain."), err.Error()
				return
			}
			a.chatgptGone()
			a.toast(i18n.T("Signed out of ChatGPT"))
		})
	})
}

// chatgptFields is the sign-in row of the dialog, where a key field would
// be.
func (a *app) chatgptFields(c *ui.Context, d *providerDialog) {
	t := c.Theme()
	email := a.keyMasks[chatgptAccount()]
	login := &a.chatLogin
	info := i18n.T("Voxt uses your own ChatGPT plan, and has no account of its own. The sign-in is kept in this Mac's Keychain, never in settings or logs, and it stands for your whole ChatGPT account. Signing out deletes it from this Mac.")
	fieldInfo(c, i18n.T("ChatGPT account"), info, func() {
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
			switch {
			case login.running:
				ui.Spinner(c).Size(px(t, 14), px(t, 14)).Shrink(0)
				caption(c, i18n.T("Waiting for the browser…")).Grow(1).MinWidth(0)
				if ghostButton(c.Key("chatgpt-cancel"), i18n.T("Cancel")).Label(i18n.T("Cancel the sign-in")).Clicked() {
					a.cancelChatGPTLogin()
				}
			case email != "":
				caption(c, i18n.T("Signed in as %s", email)).Grow(1).MinWidth(0)
				ui.Link(c.Key("chatgpt-usage"), i18n.T("Manage usage"), chatgptUsageURL).FontSize(px(t, fsCaption))
				if ghostButton(c.Key("chatgpt-signout"), i18n.T("Sign out")).Label(i18n.T("Sign out of ChatGPT")).Clicked() {
					a.signOutChatGPT()
				}
			default:
				caption(c, i18n.T("Opens your browser to sign in with your ChatGPT plan.")).Grow(1).MinWidth(0)
				if ui.Button(c.Key("chatgpt-signin"), i18n.T("Sign in with ChatGPT")).Clicked() {
					a.signInChatGPT()
				}
			}
		})
		if login.err != "" {
			notice(c, "chatgpt-err", noticeError, login.err, login.more, nil, func() { login.err, login.more = "", "" })
		}
	})
}
