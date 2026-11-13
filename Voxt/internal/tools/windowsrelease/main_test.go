package main

import (
	"bytes"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReleaseConfigurationCommands(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "mygo.json")
	if err := os.WriteFile(configPath, []byte(`{"name":"Voxt","version":"2.0.3","macos":{"minimumSystemVersion":"15.0"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	var stdout bytes.Buffer
	if err := run([]string{"version", configPath}, &stdout); err != nil {
		t.Fatal(err)
	}
	if stdout.String() != "2.0.3\n" {
		t.Fatalf("version output %q", stdout.String())
	}
	if err := run([]string{"set-version", configPath, "2.1.0"}, &stdout); err != nil {
		t.Fatal(err)
	}
	publicKey := base64.StdEncoding.EncodeToString(make([]byte, 32))
	if err := run([]string{"configure-updates", configPath, publicKey, "update.key", "notes.md"}, &stdout); err != nil {
		t.Fatal(err)
	}
	config, err := readObject(configPath)
	if err != nil {
		t.Fatal(err)
	}
	var version, name string
	if err := json.Unmarshal(config["version"], &version); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(config["name"], &name); err != nil {
		t.Fatal(err)
	}
	if version != "2.1.0" || name != "Voxt" || config["macos"] == nil {
		t.Fatalf("configuration was not preserved: %s", config)
	}
	var updates map[string]string
	if err := json.Unmarshal(config["updates"], &updates); err != nil {
		t.Fatal(err)
	}
	if updates["publicKey"] != publicKey || updates["privateKey"] != "update.key" || updates["changelog"] != "notes.md" || updates["github"] != "sgl-project/sglang-omni" || updates["tagPrefix"] != "voxt-v" {
		t.Fatalf("updates configuration %v", updates)
	}
	if err := run([]string{"configure-updates", configPath, "not-a-key", "update.key", "notes.md"}, &stdout); err == nil {
		t.Fatal("invalid update public key was accepted")
	}
}

func TestCheckManifest(t *testing.T) {
	dir := t.TempDir()
	archivePath := filepath.Join(dir, "update.tar.gz")
	if err := os.WriteFile(archivePath, []byte("archive"), 0o644); err != nil {
		t.Fatal(err)
	}
	privateKey := ed25519.NewKeyFromSeed(make([]byte, ed25519.SeedSize))
	publicKey := base64.StdEncoding.EncodeToString(privateKey.Public().(ed25519.PublicKey))
	digest := sha256.Sum256([]byte("archive"))
	signature := base64.StdEncoding.EncodeToString(ed25519.Sign(privateKey, digest[:]))
	manifestPath := filepath.Join(dir, "update-windows-amd64.json")
	manifest := `{"version":"2.1.0","url":"https://example.test/update.tar.gz","size":7,"signature":"` + signature + `"}`
	if err := os.WriteFile(manifestPath, []byte(manifest), 0o644); err != nil {
		t.Fatal(err)
	}
	args := []string{"check-manifest", manifestPath, archivePath, "2.1.0", "https://example.test/update.tar.gz", publicKey}
	if err := run(args, &bytes.Buffer{}); err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name    string
		version string
		url     string
		want    string
	}{
		{"version", "2.0.3", "https://example.test/update.tar.gz", "manifest version"},
		{"url", "2.1.0", "https://example.test/other.tar.gz", "manifest URL"},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := run([]string{"check-manifest", manifestPath, archivePath, test.version, test.url, publicKey}, &bytes.Buffer{})
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error %v, want %s", err, test.want)
			}
		})
	}
	badManifest := strings.Replace(manifest, `"size":7`, `"size":8`, 1)
	if err := os.WriteFile(manifestPath, []byte(badManifest), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := run(args, &bytes.Buffer{}); err == nil || !strings.Contains(err.Error(), "manifest size") {
		t.Fatalf("wrong archive size was accepted: %v", err)
	}
	badManifest = strings.Replace(manifest, signature, base64.StdEncoding.EncodeToString(make([]byte, 64)), 1)
	if err := os.WriteFile(manifestPath, []byte(badManifest), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := run(args, &bytes.Buffer{}); err == nil || !strings.Contains(err.Error(), "does not verify") {
		t.Fatalf("invalid archive signature was accepted: %v", err)
	}
}
