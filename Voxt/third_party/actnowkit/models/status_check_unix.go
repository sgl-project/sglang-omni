//go:build !windows

package models

import (
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

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
		} else if t, err := os.Readlink(p); err == nil && recorded != nil {
			if want, ok := recorded[filepath.Base(t)]; ok && fi.Size() != want {
				st.Damaged = append(st.Damaged, rel)
			}
		}
		key := p
		if target, err := filepath.EvalSymlinks(p); err == nil {
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
