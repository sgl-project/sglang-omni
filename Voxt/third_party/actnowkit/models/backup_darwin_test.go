//go:build darwin

package models

import (
	"bytes"
	"os"
	"os/exec"
	"strings"
	"testing"
)

// The attribute is the one tmutil writes, and Time Machine sees it.
func TestBackupExclusionDarwin(t *testing.T) {
	// Not under $TMPDIR, which Time Machine already skips: a folder in
	// the home directory, removed afterwards.
	home, _ := os.UserHomeDir()
	dir, err := os.MkdirTemp(home, ".actnowkit-tm-test-")
	if err != nil {
		t.Skip(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	if ex, err := IsBackupExcluded(dir); err != nil || ex {
		t.Fatalf("fresh dir: %v %v", ex, err)
	}
	if err := SetBackupExcluded(dir, true); err != nil {
		t.Fatal(err)
	}
	if ex, err := IsBackupExcluded(dir); err != nil || !ex {
		t.Fatalf("after set: %v %v", ex, err)
	}
	if out, err := exec.Command("tmutil", "isexcluded", dir).CombinedOutput(); err == nil && !strings.Contains(string(out), "[Excluded]") {
		t.Errorf("tmutil isexcluded: %s", out)
	}
	// The same bytes as `tmutil addexclusion`.
	ref, _ := os.MkdirTemp(home, ".actnowkit-tm-ref-")
	t.Cleanup(func() { os.RemoveAll(ref) })
	if err := exec.Command("tmutil", "addexclusion", ref).Run(); err == nil {
		got, _ := exec.Command("xattr", "-px", BackupExcludeXattr, ref).Output()
		ours, _ := exec.Command("xattr", "-px", BackupExcludeXattr, dir).Output()
		if !bytes.Equal(bytes.TrimSpace(got), bytes.TrimSpace(ours)) {
			t.Errorf("value differs from tmutil's:\n%s\n%s", got, ours)
		}
	}
	if err := SetBackupExcluded(dir, false); err != nil {
		t.Fatal(err)
	}
	if ex, _ := IsBackupExcluded(dir); ex {
		t.Fatal("still excluded")
	}
	if err := SetBackupExcluded(dir, false); err != nil {
		t.Fatalf("clearing twice: %v", err)
	}
	if out, err := exec.Command("tmutil", "isexcluded", dir).CombinedOutput(); err == nil && !strings.Contains(string(out), "[Included]") {
		t.Errorf("tmutil isexcluded after clearing: %s", out)
	}
}
