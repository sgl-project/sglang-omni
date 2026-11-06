// Package secrets keeps Voxt's secrets (the API keys of the remote
// language models) in the macOS login Keychain, as generic password items
// of the service "VoxtNew" (an inside name, as the data folders), one
// account per secret: "llm.<provider>"
// (settings.Provider.ID). Secrets never go into the settings, the
// database or the logs.
//
// It is Auralink's (@748cf55), which runs /usr/bin/security as
// AudioSmith's hub does (hub/token.go, without the Hugging Face token). Every
// function may block: call them off the main thread. Off macOS there is
// no Keychain: Get finds nothing and Set fails with ErrNoKeychain.
//
// Imports: nothing internal.
package secrets

import (
	"bytes"
	"errors"
	"fmt"
	"os/exec"
	"runtime"
	"strings"
)

// Service is the Keychain service of Voxt's secrets.
const Service = "VoxtNew"

// ErrNoKeychain is returned off macOS, where there is no Keychain to use.
var ErrNoKeychain = errors.New("no keychain on this system")

// Item is a generic password item of the login keychain (or of the
// keychain file File, for tests). Label is what Keychain Access shows; ""
// uses "Voxt <account>".
type Item struct{ Service, Account, Label, File string }

// LLM is the Keychain item of a remote language model's key, by its
// settings id: LLM("deepseek", "Voxt DeepSeek API key").
func LLM(provider, label string) Item {
	return Item{Service: Service, Account: "llm." + provider, Label: label}
}

// with appends the keychain file to a security command, when there is one.
func (k Item) with(args ...string) []string {
	if k.File != "" {
		args = append(args, k.File)
	}
	return args
}

// security runs /usr/bin/security with args, feeding it stdin; swapped in
// tests.
var security = func(stdin string, args ...string) ([]byte, error) {
	if runtime.GOOS != "darwin" {
		return nil, ErrNoKeychain
	}
	cmd := exec.Command("/usr/bin/security", args...)
	cmd.Stdin = strings.NewReader(stdin)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		if msg := strings.TrimSpace(stderr.String()); msg != "" {
			return out, fmt.Errorf("security: %s", msg)
		}
		return out, err
	}
	return out, nil
}

// missing reports an error that means the item is not there.
func missing(err error) bool {
	return errors.Is(err, ErrNoKeychain) || strings.Contains(err.Error(), "could not be found") || isExit(err, 44)
}

// Get returns the secret saved, "" when there is none.
func (k Item) Get() (string, error) {
	out, err := security("", k.with("find-generic-password", "-s", k.Service, "-a", k.Account, "-w")...)
	if err != nil {
		if missing(err) {
			return "", nil
		}
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

// Set saves value, replacing what was saved; "" deletes it. The value goes
// to security on its standard input (interactive mode), not its
// arguments, so that other processes cannot read it from the process list.
func (k Item) Set(value string) error {
	value = strings.TrimSpace(value)
	if value == "" {
		return k.Delete()
	}
	if strings.ContainsAny(value, "\"\\\n\r") {
		return errors.New("the key has quotes, backslashes or line breaks, which API keys never have")
	}
	label := k.Label
	if label == "" {
		label = "Voxt " + k.Account
	}
	cmd := fmt.Sprintf("add-generic-password -U -s %q -a %q -l %q -w \"%s\"", k.Service, k.Account, label, value)
	if k.File != "" {
		cmd += fmt.Sprintf(" %q", k.File)
	}
	_, err := security(cmd+"\n", "-i")
	return err
}

// Delete removes the saved secret; none saved is not an error.
func (k Item) Delete() error {
	_, err := security("", k.with("delete-generic-password", "-s", k.Service, "-a", k.Account)...)
	if err != nil && missing(err) {
		return nil
	}
	return err
}

func isExit(err error, code int) bool {
	var ee *exec.ExitError
	return errors.As(err, &ee) && ee.ExitCode() == code
}

// Mask shows a secret without giving it away: "sk-…wxyz".
func Mask(s string) string {
	if len(s) <= 8 {
		return strings.Repeat("•", len(s))
	}
	return s[:3] + "…" + s[len(s)-4:]
}
