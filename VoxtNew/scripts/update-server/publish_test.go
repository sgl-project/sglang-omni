package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestPublish lays versions out in the bucket as actnow.dev does: betas
// under beta/, releases at the top, and the beta channel's manifest the
// newer of both.
func TestPublish(t *testing.T) {
	bucket := t.TempDir()
	publishVersion := func(version string) (string, error) {
		t.Helper()
		dir := t.TempDir()
		os.WriteFile(filepath.Join(dir, "voxtnew-"+version+"-darwin-arm64.tar.gz"), []byte(version), 0o644)
		os.WriteFile(filepath.Join(dir, "voxtnew-0.0.1-to-"+version+"-darwin-arm64.delta"), []byte(version), 0o644)
		os.WriteFile(filepath.Join(dir, "VoxtNew "+version+" arm64.dmg"), []byte(version), 0o644)
		data, _ := json.Marshal(map[string]string{"version": version})
		os.WriteFile(filepath.Join(dir, "update-darwin-arm64.json"), data, 0o644)
		var out bytes.Buffer
		err := publish(dir, bucket, "voxtnew", &out)
		return strings.TrimSpace(out.String()), err
	}
	channels := func() string {
		v := func(p string) string {
			var m struct{ Version string }
			b, err := os.ReadFile(filepath.Join(bucket, "voxtnew", p, "update-darwin-arm64.json"))
			if err != nil {
				return "none"
			}
			json.Unmarshal(b, &m)
			return m.Version
		}
		return v("") + " " + v("beta")
	}
	for _, c := range []struct{ version, want string }{
		{"0.1.0-beta.2", "none 0.1.0-beta.2"},
		{"0.1.0", "0.1.0 0.1.0"},
		{"0.2.0-beta.1", "0.1.0 0.2.0-beta.1"},
		{"0.1.1", "0.1.1 0.2.0-beta.1"},
	} {
		out, err := publishVersion(c.version)
		if err != nil || channels() != c.want {
			t.Errorf("%s: %v, channels %s, want %s\n%s", c.version, err, channels(), c.want, out)
		}
		if !strings.HasSuffix(strings.Split(out, "\n")[2], "update-darwin-arm64.json") || strings.Contains(out, ".dmg") {
			t.Errorf("%s: the manifest after the files, no disk image:\n%s", c.version, out)
		}
	}
	if _, err := publishVersion("0.1.1"); err == nil {
		t.Error("published 0.1.1 again")
	}
	for _, c := range []struct {
		a, b string
		want int
	}{{"0.1.0", "0.1.0-beta.2", 1}, {"0.1.0-beta.10", "0.1.0-beta.9", 1}, {"0.2.0-beta.1", "0.1.1", 1}, {"1.0.0-rc.1", "1.0.0-beta.3", 1}, {"0.9.9", "0.10.0", -1}, {"1.0.0", "1.0.0", 0}} {
		if got := compareVersions(c.a, c.b); got != c.want {
			t.Errorf("compareVersions(%s, %s) = %d", c.a, c.b, got)
		}
	}
}
