//go:build !windows

package models

import (
	"os"
	"path/filepath"
	"testing"
)

// An old snapshot links its files straight into snapshots/<rev>/, with no
// blobs/ behind them (the pre-blob cache layout). It is still installed:
// the size is all the check asks.
func TestCheckOldLayoutWithoutBlobs(t *testing.T) {
	hub := t.TempDir()
	repoDir := RepoDir(hub, repoA)
	snap := filepath.Join(repoDir, "snapshots", revA)
	if err := os.MkdirAll(snap, 0o755); err != nil {
		t.Fatal(err)
	}
	for name, size := range map[string]int64{"config.json": 10, "model.safetensors": 100} {
		if err := os.WriteFile(filepath.Join(snap, name), make([]byte, size), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	c := Claim{Repo: repoA, Revision: revA, Sizes: map[string]int64{"config.json": 10, "model.safetensors": 100}}
	st := Check(hub, c)
	if st.State != Installed {
		t.Fatalf("old layout snapshot is %v (missing %v), want Installed", st.State, st.Missing)
	}
}
