package main

import (
	"encoding/json"

	"tailscale.com/types/key"
)

func publicFromPrivate(privateText string) (string, error) {
	var private key.NodePrivate
	if err := private.UnmarshalText([]byte(privateText)); err != nil {
		return "", err
	}
	if private.IsZero() {
		return "", errZeroKey
	}
	text, err := private.Public().MarshalText()
	if err != nil {
		return "", err
	}
	return string(text), nil
}

func Identity() string {
	private := key.NewNode()
	privateText, err := private.MarshalText()
	if err != nil {
		return `{"error":"identity-failed"}`
	}
	publicText, err := publicFromPrivate(string(privateText))
	if err != nil {
		return `{"error":"identity-failed"}`
	}
	body, err := json.Marshal(struct {
		PrivateKey string `json:"privateKey"`
		PublicKey  string `json:"publicKey"`
	}{
		PrivateKey: string(privateText),
		PublicKey:  publicText,
	})
	if err != nil {
		return `{"error":"identity-failed"}`
	}
	return string(body)
}

func main() {}
