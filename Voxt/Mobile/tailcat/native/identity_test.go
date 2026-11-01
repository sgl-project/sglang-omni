package main

import (
	"encoding/json"
	"strings"
	"testing"
)

const knownPrivate = "privkey:e014170e3aad6aaefe4d67667124106901cfd02c007ae10ea6d660a9d2347152"
const knownPublic = "nodekey:c2d83e8a7301f3cb2ca083dd2c8889842bdd494346f548fa89ca822b41c34c5e"

func TestPublicDerivationMatchesKnownVector(t *testing.T) {
	got, err := publicFromPrivate(knownPrivate)
	if err != nil || got != knownPublic {
		t.Fatal("known public derivation mismatch")
	}
}

func TestIdentityUsesPublicDerivation(t *testing.T) {
	first := Identity()
	second := Identity()
	if first == second {
		t.Fatal("identity was not unique")
	}
	for _, raw := range []string{first, second} {
		var doc struct {
			PrivateKey string `json:"privateKey"`
			PublicKey  string `json:"publicKey"`
		}
		if err := json.Unmarshal([]byte(raw), &doc); err != nil {
			t.Fatal("identity json")
		}
		if strings.Contains(raw, "psk:") || strings.Contains(doc.PublicKey, "psk:") {
			t.Fatal("identity exposed preshared key")
		}
		derived, err := publicFromPrivate(doc.PrivateKey)
		if err != nil || derived != doc.PublicKey {
			t.Fatal("identity public derivation mismatch")
		}
		if !strings.HasPrefix(doc.PrivateKey, "privkey:") || !strings.HasPrefix(doc.PublicKey, "nodekey:") {
			t.Fatal("identity key format")
		}
	}
}
