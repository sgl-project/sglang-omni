package models

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/hehehai/actnowkit/family"
	famruntime "github.com/hehehai/actnowkit/runtime"
)

// User is an app that claims a model.
type User struct {
	BundleID string
	AppID    string // "audiosmith"; "" for an app outside the family table
	Name     string // "AudioSmith", "AudioSmith (dev)"
	Claim    Claim
	// Orphan: the app is uninstalled (spec §7.7); its claim protects
	// nothing, but is listed so that the user can clean it up.
	Orphan bool
}

// Decision is what deleting a model would do (spec §7.6), computed from
// the disk, every app's claims and the running apps' runtime files.
type Decision struct {
	Ref    Ref
	Status Status // the snapshot on disk
	// Mine: this app claims it (MyClaim), for the models MyModels.
	Mine     bool
	MyClaim  Claim
	MyModels []string
	// Users are the other apps with a valid claim: same $HUB, installed.
	Users []User
	// Orphans are the claims of uninstalled apps.
	Orphans []User
	// LoadedBy are the other running apps that hold it in memory.
	LoadedBy []User
	// Unsure: some claims files could not be read, or an installed app
	// that keeps models in this $HUB has no claims file (Unclaimed), so
	// anyone may use it.
	Unsure bool
	// Unclaimed are the installed apps whose manifest names this $HUB but
	// whose claims file is missing (lost with the family folder): they
	// will claim their models again when they start (Store.Heal).
	Unclaimed []User
	// FreeBytes is what deleting the snapshot frees: its blobs that no
	// other snapshot of the repo links.
	FreeBytes int64
	// LastSnapshot: no other snapshot of the repo is left after it, so
	// the repo directory and its engine copy (mlx-audio/) go too.
	LastSnapshot bool
}

// InUse reports whether another app needs the files: deleting "from this
// app" then keeps them (spec §7.6 step 1.4).
func (d Decision) InUse() bool { return len(d.Users) > 0 || len(d.LoadedBy) > 0 || d.Unsure }

// UserNames are the names of Users, LoadedBy and Unclaimed, each once, for the
// dialogs ("AudioSmith also uses this model").
func (d Decision) UserNames() []string {
	var out []string
	for _, u := range append(append(slices.Clone(d.Users), d.LoadedBy...), d.Unclaimed...) {
		if !slices.Contains(out, u.Name) {
			out = append(out, u.Name)
		}
	}
	return out
}

func (s *Store) apps() map[string]AppState {
	if s.Apps == nil {
		return nil
	}
	return s.Apps()
}

// userOf names a bundle id for the UI.
func userOf(bid string, apps map[string]AppState) User {
	u := User{BundleID: bid, Name: bid}
	if a, v, err := family.ParseBundleID(bid); err == nil {
		u.AppID, u.Name = a.ID, a.Name
		if v != family.Release {
			u.Name += " (" + string(v) + ")"
		}
	}
	if st, ok := apps[bid]; ok && st.Name != "" && u.AppID == "" {
		u.Name = st.Name
	}
	return u
}

// installed: an app missing from the map counts as installed.
func installed(apps map[string]AppState, bid string) bool {
	if apps == nil {
		return true
	}
	st, ok := apps[bid]
	return !ok || st.Installed
}

// Decide computes what deleting ref would do. It takes no lock: Delete
// calls it again holding the repo lock.
func (s *Store) Decide(ref Ref) (Decision, error) {
	if !ValidRepo(ref.Repo) || ref.Revision == "" {
		return Decision{}, fmt.Errorf("%s: %w", ref, errBadRepo)
	}
	apps := s.apps()
	all, errs := ReadAllClaims(s.Family)
	d := Decision{Ref: ref, Unsure: len(errs) > 0}
	d.Status = Check(s.Hub, Claim{Repo: ref.Repo, Revision: ref.Revision, Aux: true})
	for bid, cf := range all {
		if cf.Root != "" && !SameHub(cf.Root, s.Hub) {
			continue
		}
		c, ok := cf.Find(ref)
		if !ok {
			continue
		}
		if bid == s.BundleID {
			d.Mine, d.MyClaim, d.MyModels = true, c, c.Users()
			continue
		}
		u := userOf(bid, apps)
		u.Claim = c
		if installed(apps, bid) {
			d.Users = append(d.Users, u)
		} else {
			u.Orphan = true
			d.Orphans = append(d.Orphans, u)
		}
	}
	for bid, st := range apps {
		if _, ok := all[bid]; ok || bid == s.BundleID || !st.Installed || st.ModelsRoot == "" || !SameHub(st.ModelsRoot, s.Hub) {
			continue
		}
		d.Unclaimed = append(d.Unclaimed, userOf(bid, apps))
		d.Unsure = true
	}
	sortUsers(d.Users)
	sortUsers(d.Orphans)
	sortUsers(d.Unclaimed)
	for _, f := range famruntime.LoadedBy(s.Family, s.BundleID, ref.Repo, ref.Revision) {
		d.LoadedBy = append(d.LoadedBy, userOf(f.BundleID, apps))
	}
	d.FreeBytes, d.LastSnapshot = freeable(s.Hub, ref)
	return d, nil
}

func sortUsers(u []User) {
	slices.SortFunc(u, func(a, b User) int { return strings.Compare(a.BundleID, b.BundleID) })
}

// Scope of a deletion.
type Scope int

const (
	// ThisApp removes the model from this app; the files go only when no
	// other app needs them (the default button).
	ThisApp Scope = iota
	// AllApps deletes the files whoever uses them, and records a tombstone
	// so that the other apps drop their claims ("Delete for all apps…",
	// after a second confirmation).
	AllApps
)

// Result is what Delete did.
type Result struct {
	Decision Decision // as decided holding the lock
	Deleted  bool     // the files went
	Freed    int64    // bytes removed
	// Tombstoned: other apps' claims were overridden (AllApps).
	Tombstoned bool
	// OwnModels are this app's other models that still use the repo
	// (DeleteModel): with ThisApp the claim and files stay for them; with
	// AllApps they lost it too.
	OwnModels []string
}

// Delete removes ref from this app, following spec §7.6 step 1:
//
//  1. this app's claim goes first (whatever happens next);
//  2. the repo lock is taken exclusively without waiting: when another app
//     downloads or claims, the claim is restored and a *BusyError returned
//     (TextBusy: "try again later");
//  3. the claims and runtime files are read again;
//  4. ThisApp: when another app uses the model, the files stay;
//  5. otherwise the snapshot goes (DeleteSnapshot);
//  6. AllApps with other users also writes a tombstone.
//
// Auxiliary repos (a tokenizer) are separate refs: delete them the same
// way.
func (s *Store) Delete(ctx context.Context, ref Ref, scope Scope) (Result, error) {
	if !ValidRepo(ref.Repo) || ref.Revision == "" {
		return Result{}, fmt.Errorf("%s: %w", ref, errBadRepo)
	}
	mine, _ := s.Claims()
	had, wasClaimed := mine.Find(ref)
	if wasClaimed {
		if err := s.Unclaim(ref); err != nil {
			return Result{}, err
		}
	}
	lock, err := TryLockRepo(s.Hub, ref.Repo, family.Exclusive, s.who(OpDelete))
	if err != nil {
		if wasClaimed {
			_ = s.update(func(c *Claims) bool {
				c.Claims = append(c.Claims, had)
				return true
			})
		}
		return Result{}, err
	}
	defer lock.Release()
	d, err := s.Decide(ref)
	if err != nil {
		return Result{}, err
	}
	res := Result{Decision: d}
	if scope == ThisApp && d.InUse() {
		return res, nil
	}
	freed, err := DeleteSnapshot(s.Hub, ref)
	res.Freed = freed
	if err != nil {
		return res, err
	}
	res.Deleted = true
	if scope == AllApps && (len(d.Users) > 0 || len(d.LoadedBy) > 0 || d.Unsure) {
		if err := AddTombstone(ctx, s.Family, s.Hub, ref, s.BundleID, s.now()); err != nil {
			return res, err
		}
		res.Tombstoned = true
	}
	_ = s.Family.Post(family.TopicModels)
	return res, nil
}

// DeleteModel removes one of this app's models (a catalogue id) from ref:
// the model's main repo, or one of its auxiliary repos. When other models
// of this app still use ref, ThisApp only releases model from the claim
// (Release) and keeps everything, reporting them in Result.OwnModels;
// AllApps deletes anyway, as Delete does. Otherwise it is Delete.
// Call it for the model's repo and for each of its auxiliary repos.
func (s *Store) DeleteModel(ctx context.Context, ref Ref, model string, scope Scope) (Result, error) {
	if !ValidRepo(ref.Repo) || ref.Revision == "" {
		return Result{}, fmt.Errorf("%s: %w", ref, errBadRepo)
	}
	mine, _ := s.Claims()
	if c, ok := mine.Find(ref); ok && model != "" {
		others := slices.DeleteFunc(c.Users(), func(u string) bool { return u == model })
		if len(others) > 0 {
			if scope == ThisApp {
				if _, err := s.Release(ref, model); err != nil {
					return Result{}, err
				}
				d, err := s.Decide(ref)
				return Result{Decision: d, OwnModels: others}, err
			}
			res, err := s.Delete(ctx, ref, scope)
			res.OwnModels = others
			return res, err
		}
	}
	return s.Delete(ctx, ref, scope)
}

// snapshotBlobs maps the blobs a snapshot links (absolute paths) to their
// sizes.
func snapshotBlobs(snap string) map[string]int64 {
	out := map[string]int64{}
	repoDir := filepath.Dir(filepath.Dir(snap))
	_ = filepath.WalkDir(snap, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		target, err := ResolveBlob(repoDir, p)
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

// DeleteSnapshot removes ref's files (spec §7.6 step 1.5): the snapshot
// (and an unfinished one), the refs naming it, the blobs no remaining
// snapshot links (with their partial downloads, except blobs whose lock
// someone holds); the repo directory and its mlx-audio copy when no
// snapshot is left. The caller holds the repo lock exclusively and has
// decided (Decide) that the files may go. It returns the bytes removed.
func DeleteSnapshot(hub string, ref Ref) (int64, error) {
	if !ValidRepo(ref.Repo) || ref.Revision == "" || strings.ContainsAny(ref.Revision, `/\`) || ref.Revision == "." || ref.Revision == ".." {
		return 0, fmt.Errorf("%s: %w", ref, errBadRepo)
	}
	repoDir := RepoDir(hub, ref.Repo)
	if _, err := os.Stat(repoDir); os.IsNotExist(err) {
		return 0, nil
	}
	snaps := filepath.Join(repoDir, "snapshots")
	if err := os.RemoveAll(filepath.Join(snaps, ref.Revision)); err != nil {
		return 0, err
	}
	_ = os.RemoveAll(filepath.Join(snaps, ref.Revision+".incomplete"))
	_ = filepath.WalkDir(filepath.Join(repoDir, "refs"), func(p string, d fs.DirEntry, err error) error {
		if err == nil && !d.IsDir() {
			if data, err := os.ReadFile(p); err == nil && strings.TrimSpace(string(data)) == ref.Revision {
				os.Remove(p)
			}
		}
		return nil
	})
	used, others := usedBlobNames(repoDir, ref.Revision)
	blobs := filepath.Join(repoDir, "blobs")
	entries, _ := os.ReadDir(blobs)
	var freed int64
	busy := false
	for _, e := range entries {
		name := e.Name()
		base, _, _ := strings.Cut(name, ".incomplete")
		if used[base] {
			continue
		}
		// A blob someone is downloading (another app, a Python tool)
		// stays; its lock tells.
		if validBlob(base) {
			l, err := TryLockBlob(hub, ref.Repo, base, Owner{Op: OpDelete})
			if err != nil {
				busy = true
				continue
			}
			freed += removeSize(filepath.Join(blobs, name))
			l.Release()
		} else {
			freed += removeSize(filepath.Join(blobs, name))
		}
	}
	if others == 0 && !busy {
		if err := os.RemoveAll(repoDir); err != nil {
			return freed, err
		}
		if err := os.RemoveAll(EngineCopy(hub, ref.Repo)); err != nil {
			return freed, err
		}
	}
	return freed, nil
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

// Orphan is an uninstalled app's claims (spec §7.6 step 4).
type Orphan struct {
	User
	Claims []Claim
	// Bytes is what cleaning up would free: the claimed snapshots no
	// installed app claims.
	Bytes int64
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

func usedByInstalled(all map[string]Claims, apps map[string]AppState, skip string, ref Ref, hub string) bool {
	for bid, cf := range all {
		if bid == skip || !installed(apps, bid) || (cf.Root != "" && !SameHub(cf.Root, hub)) {
			continue
		}
		if _, ok := cf.Find(ref); ok {
			return true
		}
	}
	return false
}

// CleanOrphans deletes the models only the uninstalled app bundleID
// claimed (each as Delete would, skipping busy repos), then removes that
// app's claims, manifest and runtime files. It returns what happened to
// each claim.
func (s *Store) CleanOrphans(ctx context.Context, bundleID string) ([]Result, error) {
	apps := s.apps()
	if bundleID == s.BundleID || installed(apps, bundleID) {
		return nil, fmt.Errorf("%s is installed: its claims are not orphans", bundleID)
	}
	cf, err := ReadClaims(s.Family, bundleID)
	if err != nil {
		return nil, err
	}
	var out []Result
	var errs []error
	for _, c := range cf.Claims {
		ref := c.Ref()
		if !ValidRepo(ref.Repo) {
			continue
		}
		lock, err := TryLockRepo(s.Hub, ref.Repo, family.Exclusive, s.who(OpDelete))
		if err != nil {
			errs = append(errs, err)
			continue
		}
		d, err := s.Decide(ref)
		if err != nil {
			lock.Release()
			errs = append(errs, err)
			continue
		}
		res := Result{Decision: d}
		if !d.InUse() && !d.Mine {
			res.Freed, err = DeleteSnapshot(s.Hub, ref)
			res.Deleted = err == nil
			if err != nil {
				errs = append(errs, err)
			}
		}
		lock.Release()
		out = append(out, res)
	}
	if len(errs) > 0 {
		return out, errors.Join(errs...)
	}
	for _, p := range []string{s.Family.ClaimsFile(bundleID), s.Family.AppFile(bundleID), s.Family.RuntimeFile(bundleID)} {
		if err := os.Remove(p); err != nil && !os.IsNotExist(err) {
			return out, err
		}
	}
	_ = s.Family.Post(family.TopicModels)
	_ = s.Family.Post(family.TopicApps)
	return out, nil
}

// Usage splits the disk space of this app's claimed models (blobs counted
// once): Mine only this app uses, Shared other apps use too (spec §7.6
// step 6).
type Usage struct {
	Mine, Shared int64
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

// Found is a wanted model found on disk, for the first-run reuse card
// (spec §7.10.1).
type Found struct {
	Want   Claim
	Status Status
	// By are the other apps claiming it (the card says "Installed by
	// <first>"); empty when no app claims it (an older app or a Python
	// tool fetched it).
	By []User
}

// Survey checks which of the wanted models (this app's recommended pack)
// are already installed, and by whom. Claiming the installed ones with
// OriginReused is "Use them".
func (s *Store) Survey(want ...Claim) []Found {
	apps := s.apps()
	all, _ := ReadAllClaims(s.Family)
	var out []Found
	for _, w := range want {
		st := Check(s.Hub, w)
		if st.State != Installed {
			continue
		}
		f := Found{Want: w, Status: st}
		for bid, cf := range all {
			if bid == s.BundleID || (cf.Root != "" && !SameHub(cf.Root, s.Hub)) || !installed(apps, bid) {
				continue
			}
			if c, ok := cf.Find(w.Ref()); ok {
				u := userOf(bid, apps)
				u.Claim = c
				f.By = append(f.By, u)
			}
		}
		sortUsers(f.By)
		out = append(out, f)
	}
	return out
}

// EngineCopyMinAge: empty mlx-audio directories younger than this may be
// another app's engine at work (spec §7.8).
const EngineCopyMinAge = 10 * time.Minute

// CleanEngineCopies removes the empty directories under $HUB/mlx-audio
// that are older than minAge (EngineCopyMinAge): mlx-audio-swift leaves
// them after failed loads, and its next load would then delete the hub
// snapshot. It returns their names.
func CleanEngineCopies(hub string, minAge time.Duration) ([]string, error) {
	dir := EngineCopiesDir(hub)
	entries, err := os.ReadDir(dir)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var removed []string
	for _, e := range entries {
		p := filepath.Join(dir, e.Name())
		if !e.IsDir() || nonEmpty(p) {
			continue
		}
		fi, err := e.Info()
		if err != nil || time.Since(fi.ModTime()) < minAge {
			continue
		}
		if os.Remove(p) == nil {
			removed = append(removed, e.Name())
		}
	}
	return removed, nil
}
