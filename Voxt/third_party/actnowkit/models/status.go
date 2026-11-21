package models

import (
	"os"
	"path/filepath"
)

// State is how much of a model is on disk.
type State string

const (
	Missing   State = "missing"   // nothing of the snapshot
	Partial   State = "partial"   // a repo directory, an unfinished snapshot or missing files
	Installed State = "installed" // complete (spec §7.5)
)

// Status is what Check found.
type Status struct {
	State    State
	Snapshot string   // snapshots/<commit>, when it exists
	Bytes    int64    // bytes of the blobs the snapshot links (each counted once)
	Missing  []string // files listed but absent, or links without a blob
	// Damaged are files whose size is not the size their blob was
	// verified with (a truncated weights file): the model is Partial,
	// and a download or Verify repairs it (spec §7.5).
	Damaged []string
	// Incomplete bytes of unfinished downloads of the repo (blobs/*.incomplete).
	Incomplete int64
}

// weightExts mark a usable model: a snapshot of a main (non-aux) repo
// without any is Partial (a tool may have fetched config.json alone).
var weightExts = []string{".safetensors", ".bin", ".npz", ".gguf", ".onnx", ".pt", ".pth", ".mlmodelc", ".mlpackage"}

// matchAny reports whether a glob ("v3/*") matches any present file.
func matchAny(present map[string]bool, pattern string) bool {
	for p := range present {
		if ok, _ := filepath.Match(pattern, p); ok {
			return true
		}
	}
	return false
}

func nonEmpty(dir string) bool {
	entries, err := os.ReadDir(dir)
	return err == nil && len(entries) > 0
}

// FindInstalled picks the snapshot an app should use for a catalogue entry
// (spec §7.2): the pinned commit when installed, else the first of accept
// (other commits the app's engine has validated) that is. ok is false
// when none is: download the pinned one.
func FindInstalled(hub string, c Claim, accept []string) (rev string, st Status, ok bool) {
	for _, r := range append([]string{c.Revision}, accept...) {
		cc := c
		cc.Revision = r
		if st := Check(hub, cc); st.State == Installed {
			return r, st, true
		}
	}
	return "", Status{}, false
}
