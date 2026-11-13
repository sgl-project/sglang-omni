package models

import (
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
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

// Check says whether c's snapshot is installed, from the files alone (no
// hashing, no network; spec §7.5): the snapshot directory exists, every
// link reaches its blob, every file c.Files lists is there, each file has
// the size recorded for its blob (SizesFile) and the size c.Sizes gives,
// and a main repo has a weights file. Hashes are checked when downloading
// and when the user asks (hfhub Verify).
func Check(hub string, c Claim) Status {
	ref := c.Ref()
	var st Status
	if !ValidRepo(ref.Repo) || ref.Revision == "" {
		st.State = Missing
		return st
	}
	repoDir := RepoDir(hub, ref.Repo)
	if _, err := os.Stat(repoDir); err != nil {
		st.State = Missing
		return st
	}
	if entries, err := os.ReadDir(filepath.Join(repoDir, "blobs")); err == nil {
		for _, e := range entries {
			if strings.HasSuffix(e.Name(), ".incomplete") {
				if info, err := e.Info(); err == nil {
					st.Incomplete += info.Size()
				}
			}
		}
	}
	snap := SnapshotDir(hub, ref)
	if fi, err := os.Stat(snap); err != nil || !fi.IsDir() {
		st.State = Partial
		if st.Incomplete == 0 && !nonEmpty(filepath.Join(repoDir, "blobs")) && !nonEmpty(filepath.Join(repoDir, "snapshots")) {
			st.State = Missing
		}
		return st
	}
	st.Snapshot = snap
	recorded := ReadBlobSizes(hub, ref.Repo)
	present := map[string]bool{}
	seen := map[string]bool{}
	weights := false
	files := 0
	_ = filepath.WalkDir(snap, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		rel, _ := filepath.Rel(snap, p)
		rel = filepath.ToSlash(rel)
		fi, err := os.Stat(p) // follows the link to the blob
		if err != nil {
			st.Missing = append(st.Missing, rel)
			return nil
		}
		files++
		present[rel] = true
		if want, ok := c.Sizes[rel]; ok && want > 0 && fi.Size() != want {
			st.Damaged = append(st.Damaged, rel)
		} else if target, err := ResolveBlob(repoDir, p); err != nil {
			st.Missing = append(st.Missing, rel)
		} else if recorded != nil {
			if want, ok := recorded[filepath.Base(target)]; ok && fi.Size() != want {
				st.Damaged = append(st.Damaged, rel)
			}
		}
		key := p
		if target, err := ResolveBlob(repoDir, p); err == nil {
			key = target
		}
		if !seen[key] {
			seen[key] = true
			st.Bytes += fi.Size()
		}
		if slices.Contains(weightExts, strings.ToLower(filepath.Ext(rel))) {
			weights = true
		}
		return nil
	})
	for _, f := range c.Files {
		if strings.ContainsAny(f, "*?[") {
			if !matchAny(present, f) {
				st.Missing = append(st.Missing, f)
			}
			continue
		}
		if !present[f] && !slices.Contains(st.Missing, f) {
			st.Missing = append(st.Missing, f)
		}
	}
	switch {
	case len(st.Missing) > 0 || len(st.Damaged) > 0 || files == 0:
		st.State = Partial
	case !c.Aux && len(c.Files) == 0 && !weights:
		st.State = Partial
	default:
		st.State = Installed
	}
	return st
}

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
