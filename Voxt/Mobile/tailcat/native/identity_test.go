package main

import (
	"encoding/json"
	"strings"
	"testing"
)

const knownPrivate = "privkey:40ab1b58e9076c7a4d9d07291f5edf9d1aa017eb949624ba683317f48a640369"
const knownPublic = "nodekey:50d20b455ecf12bc453f83c2cfdb2a24925d06cf2598dcaa54e91af82ce9f765"

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
