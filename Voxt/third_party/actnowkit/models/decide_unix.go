//go:build !windows

package models

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

// snapshotBlobs maps the blobs a snapshot links (absolute paths) to their
// sizes.
func snapshotBlobs(snap string) map[string]int64 {
	out := map[string]int64{}
	_ = filepath.WalkDir(snap, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		target, err := filepath.EvalSymlinks(p)
		if err != nil {
			return nil
		}
		if fi, err := os.Stat(target); err == nil {
			out[target] = fi.Size()
		}
		return nil
	})
	return out
}

// usedBlobNames are the blob names (base names of link targets) linked by
// the snapshots of repoDir other than skip.
func usedBlobNames(repoDir, skip string) (used map[string]bool, others int) {
	used = map[string]bool{}
	snaps := filepath.Join(repoDir, "snapshots")
	entries, _ := os.ReadDir(snaps)
	for _, e := range entries {
		if !e.IsDir() || e.Name() == skip || e.Name() == skip+".incomplete" {
			continue
		}
		others++
		_ = filepath.WalkDir(filepath.Join(snaps, e.Name()), func(p string, d fs.DirEntry, err error) error {
			if err == nil && d.Type()&fs.ModeSymlink != 0 {
				if t, err := os.Readlink(p); err == nil {
					used[filepath.Base(t)] = true
				}
			}
			return nil
		})
	}
	return used, others
}

// freeable is what deleting ref's snapshot would free.
func freeable(hub string, ref Ref) (int64, bool) {
	repoDir := RepoDir(hub, ref.Repo)
	used, others := usedBlobNames(repoDir, ref.Revision)
	var n int64
	for p, size := range snapshotBlobs(SnapshotDir(hub, ref)) {
		if fi, err := os.Stat(p); err == nil && linkedElsewhere(fi) {
			continue // an imported blob the Hugging Face cache still holds
		}
		if !used[filepath.Base(p)] {
			n += size
		}
	}
	return n, others == 0
}

// removeSize removes p and returns the space that frees: nothing for a
// file with other hard links (an imported blob the Hugging Face cache
// still holds).
func removeSize(p string) int64 {
	fi, err := os.Lstat(p)
	if err != nil {
		return 0
	}
	if os.Remove(p) != nil || linkedElsewhere(fi) {
		return 0
	}
	return fi.Size()
}

// Orphans lists the claims of apps the registry says are uninstalled
// (needs Store.Apps), for "N models (X GB) are only used by apps that
// are no longer installed" (TextOrphans).
func (s *Store) Orphans() ([]Orphan, error) {
	apps := s.apps()
	if apps == nil {
		return nil, nil
	}
	all, errs := ReadAllClaims(s.Family)
	if len(errs) > 0 {
		return nil, errors.Join(errs...)
	}
	var out []Orphan
	for bid, cf := range all {
		if bid == s.BundleID || installed(apps, bid) || (cf.Root != "" && !SameHub(cf.Root, s.Hub)) || len(cf.Claims) == 0 {
			continue
		}
		o := Orphan{User: userOf(bid, apps), Claims: cf.Claims}
		o.Orphan = true
		seen := map[string]bool{}
		for _, c := range cf.Claims {
			if usedByInstalled(all, apps, bid, c.Ref(), s.Hub) {
				continue
			}
			for p, n := range snapshotBlobs(SnapshotDir(s.Hub, c.Ref())) {
				if !seen[p] {
					seen[p] = true
					o.Bytes += n
				}
			}
		}
		out = append(out, o)
	}
	slices.SortFunc(out, func(a, b Orphan) int { return strings.Compare(a.BundleID, b.BundleID) })
	return out, nil
}

// Usage computes it.
func (s *Store) Usage() (Usage, error) {
	mine, err := s.Claims()
	if err != nil {
		return Usage{}, err
	}
	var u Usage
	seen := map[string]bool{}
	for _, c := range mine.Claims {
		d, err := s.Decide(c.Ref())
		if err != nil {
			continue
		}
		for p, n := range snapshotBlobs(SnapshotDir(s.Hub, c.Ref())) {
			if seen[p] {
				continue
			}
			seen[p] = true
			if d.InUse() {
				u.Shared += n
			} else {
				u.Mine += n
			}
		}
	}
	return u, nil
}
