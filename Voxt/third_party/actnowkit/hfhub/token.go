package hfhub

import (
	"os"
	"path/filepath"
	"strings"
)

// The Hugging Face token, for gated repos. It is a secret: it is sent
// only to the official endpoint (Options.Official), never logged. This
// package keeps no secrets itself: the app (with the credentials package)
// reads the Keychain and hands ResolveToken its callbacks.

// TokenSource names where a token came from.
type TokenSource string

const (
	TokenNone   TokenSource = ""
	TokenEnv    TokenSource = "env"    // $HF_TOKEN
	TokenShared TokenSource = "shared" // the family's shared credential (spec §8.2)
	TokenApp    TokenSource = "app"    // the app's own (older) Keychain item
	TokenCLI    TokenSource = "file"   // huggingface_hub's token file (huggingface-cli login)
)

// TokenFile is huggingface_hub's token file: $HF_TOKEN_PATH, else
// $HF_HOME/token, else ~/.cache/huggingface/token.
func TokenFile() string {
	if p := os.Getenv("HF_TOKEN_PATH"); p != "" {
		return p
	}
	if d := os.Getenv("HF_HOME"); d != "" {
		return filepath.Join(d, "token")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".cache", "huggingface", "token")
}

// ResolveToken finds the HF token in the order of spec §8.4, and where it
// came from; "" when there is none:
//
//	$HF_TOKEN                       development, CI
//	shared()                        the family's shared credential (the app's credentialRef)
//	own()                           the app's own Keychain item, from before the family
//	huggingface_hub's token file    TokenFile()
//
// shared and own may be nil; an error from one counts as no token there.
// They may block (the Keychain runs /usr/bin/security): call ResolveToken
// off the main thread, and cache the answer for Options.Token.
func ResolveToken(shared, own func() (string, error)) (string, TokenSource) {
	if t := strings.TrimSpace(os.Getenv("HF_TOKEN")); t != "" {
		return t, TokenEnv
	}
	for _, s := range []struct {
		get func() (string, error)
		src TokenSource
	}{{shared, TokenShared}, {own, TokenApp}} {
		if s.get == nil {
			continue
		}
		if t, err := s.get(); err == nil {
			if t = strings.TrimSpace(t); t != "" {
				return t, s.src
			}
		}
	}
	if data, err := os.ReadFile(TokenFile()); err == nil {
		if t := strings.TrimSpace(string(data)); t != "" {
			return t, TokenCLI
		}
	}
	return "", TokenNone
}

// MaskToken shows a token as the UI may: "hf_…wxyz".
func MaskToken(t string) string {
	if len(t) <= 8 {
		return strings.Repeat("•", len(t))
	}
	return t[:3] + "…" + t[len(t)-4:]
}
