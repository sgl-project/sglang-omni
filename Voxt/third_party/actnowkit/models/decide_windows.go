//go:build windows

package models

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

type snapshotBlob struct {
	sizeBytes     int64
	snapshotLinks uint32
}

// snapshotBlobs maps the blobs a snapshot links (absolute paths) to their
// sizes and the snapshot entries hard-linked to them.
func snapshotBlobs(snap string) map[string]snapshotBlob {
	out := map[string]snapshotBlob{}
	repoDir := filepath.Dir(filepath.Dir(snap))
	_ = filepath.WalkDir(snap, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		target, err := ResolveBlob(repoDir, p)
		if err != nil {
			return nil
		}
		targetInfo, err := os.Stat(target)
		if err != nil {
			return nil
		}
		blob := snapshotBlob{sizeBytes: targetInfo.Size(), snapshotLinks: out[target].snapshotLinks}
		if linkInfo, err := os.Stat(p); err == nil && os.SameFile(targetInfo, linkInfo) {
			blob.snapshotLinks++
		}
		out[target] = blob
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
			if err == nil && !d.IsDir() {
				if target, err := ResolveBlob(repoDir, p); err == nil {
					used[filepath.Base(target)] = true
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
	for p, blob := range snapshotBlobs(SnapshotDir(hub, ref)) {
		if used[filepath.Base(p)] {
			continue
		}
		if fi, err := os.Stat(p); err == nil && linkedElsewhere(p, fi, blob.snapshotLinks) {
			continue // an imported blob the Hugging Face cache still holds
		}
		n += blob.sizeBytes
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
	linked := linkedElsewhere(p, fi, 0)
	if os.Remove(p) != nil || linked {
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
					o.Bytes += n.sizeBytes
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
				u.Shared += n.sizeBytes
			} else {
				u.Mine += n.sizeBytes
			}
		}
	}
	return u, nil
}
