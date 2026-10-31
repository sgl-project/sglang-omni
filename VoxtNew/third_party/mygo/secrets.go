package mygo

import (
	"errors"
	"fmt"
	"strings"

	"github.com/egoist/mygo/internal/platform"
)

// SecretsModule keeps small secrets, such as access tokens and API keys,
// in the system's credential store. Use the Secrets singleton, or
// Secrets.Service for the secrets of another service.
type SecretsModule struct {
	// service is the store's service, "" for the app's identifier.
	service string
}

// Secrets keeps small secrets in the system's credential store: the
// login Keychain on macOS, the Secret Service on Linux (GNOME Keyring,
// KWallet, KeePassXC…, through libsecret) and the Credential Manager on
// Windows. The system encrypts them for the user, unlike a file in
// PathUserData:
//
//	if err := mygo.Secrets.Set("github-token", []byte(token)); err != nil { … }
//
//	token, err := mygo.Secrets.Get("github-token")
//	if errors.Is(err, mygo.ErrSecretNotFound) { … ask the user to sign in }
//
// A store keeps a secret under a service and an account, which Secrets
// takes the key for. The service is the app's identifier (identifier of
// mygo.config.ts); under `mygo dev`, whose app is "<identifier>.dev", the
// development app's, so it keeps secrets apart from the installed app;
// and without an identifier, as with `go run`, the app's name
// (App.Name). Service chooses another. Other apps of the user may still
// read them, on Linux without asking.
//
// Where there is no credential store, the methods fail with an error for
// which errors.Is(err, errors.ErrUnsupported) holds: in browsers, and on
// Linux without libsecret, a session bus or a Secret Service on it, as on
// many servers and minimal desktops. An app can then keep the secret
// elsewhere, or ask for it each time.
//
// The methods are safe from any goroutine and work before Run. They may
// block while the system asks the user, e.g. to unlock the keyring or,
// on macOS, to let a rebuilt app read what it stored; on the main thread
// the app keeps handling events meanwhile.
var Secrets SecretsModule

// ErrSecretNotFound is what Secrets.Get fails with when no secret is
// stored for the key.
var ErrSecretNotFound = platform.ErrSecretNotFound

// MaxSecretSize is the largest secret Secrets.Set stores, in bytes: the
// most the Windows Credential Manager keeps, so that an app behaves the
// same everywhere. To keep more, encrypt it into a file of PathUserData
// with a key kept in Secrets.
const MaxSecretSize = 2560

// maxSecretKeyLen is the longest key and service, in bytes, well within
// the Credential Manager's 513 characters for the user name.
const maxSecretKeyLen = 256

// Service returns Secrets for the secrets of service, in place of the
// app's identifier: to share them with another program of the app, such
// as a command line tool, or to read those an Electron app stored with
// keytar, which takes the same service and account and stores them as
// Secrets does, under the service the app passed to it. An empty service
// is the app's.
//
//	token, err := mygo.Secrets.Service("MyApp").Get("refresh-token") // keytar.getPassword("MyApp", "refresh-token")
func (SecretsModule) Service(service string) SecretsModule { return SecretsModule{service: service} }

// serviceName returns the store's service, after checking it.
func (s SecretsModule) serviceName() (string, error) {
	if s.service == "" {
		return appID(), nil
	}
	return s.service, checkSecretName("service", s.service)
}

// Set stores value under key, replacing what it had. The key must not be
// empty or longer than 256 bytes; the value may be at most MaxSecretSize
// bytes.
func (s SecretsModule) Set(key string, value []byte) error {
	service, err := s.check(key)
	if err != nil {
		return err
	}
	if len(value) > MaxSecretSize {
		return fmt.Errorf("mygo: secret %q has %d bytes, more than MaxSecretSize (%d)", key, len(value), MaxSecretSize)
	}
	return offMain(func() error { return wrapSecretErr(backend().Secrets().SetSecret(service, key, value)) })
}

// Get returns the secret stored under key, or ErrSecretNotFound.
func (s SecretsModule) Get(key string) ([]byte, error) {
	service, err := s.check(key)
	if err != nil {
		return nil, err
	}
	type result struct {
		v   []byte
		err error
	}
	r := offMain(func() result {
		v, err := backend().Secrets().Secret(service, key)
		return result{v, wrapSecretErr(err)}
	})
	return r.v, r.err
}

// Delete removes the secret stored under key. Deleting a key without a
// secret is not an error.
func (s SecretsModule) Delete(key string) error {
	service, err := s.check(key)
	if err != nil {
		return err
	}
	return offMain(func() error { return wrapSecretErr(backend().Secrets().DeleteSecret(service, key)) })
}

// check checks key and returns the service.
func (s SecretsModule) check(key string) (string, error) {
	if err := checkSecretName("key", key); err != nil {
		return "", err
	}
	return s.serviceName()
}

func checkSecretName(what, name string) error {
	switch {
	case name == "":
		return fmt.Errorf("mygo: empty secret %s", what)
	case len(name) > maxSecretKeyLen:
		return fmt.Errorf("mygo: secret %s has %d bytes, more than %d", what, len(name), maxSecretKeyLen)
	case strings.ContainsRune(name, 0):
		return fmt.Errorf("mygo: secret %s %q contains NUL", what, name)
	}
	return nil
}

// wrapSecretErr names the store in the errors of the backend, which keep
// what they wrap (errors.ErrUnsupported), and leaves ErrSecretNotFound,
// which needs no more words, as it is.
func wrapSecretErr(err error) error {
	if err == nil || errors.Is(err, ErrSecretNotFound) {
		return err
	}
	return fmt.Errorf("mygo: secrets: %w", err)
}

// offMain runs fn, which may block for long, e.g. while the system asks
// the user something, without blocking the event loop: on the main thread
// of a running app fn runs on another goroutine while await handles
// native events. Elsewhere, before Run and after the loop stopped, it
// runs fn directly.
func offMain[T any](fn func() T) T {
	if !isMainThread() || !App.initialized || loop.stopped() {
		return fn()
	}
	ch := make(chan T, 1)
	go func() { deliver(ch, fn()) }()
	return await(ch)
}
