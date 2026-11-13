// Import: bringing the family's models over from Hugging Face caches into
// the model library (spec §7.1.1, v1.0.9).

package models

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"syscall"
	"time"

	"github.com/hehehai/actnowkit/family"
)

// OpImport names the lock owner of an import.
const OpImport = "import"

// OriginImported marks a claim made by Import.
const OriginImported = "imported"

// KindImported is the schema kind of models/imported/<bundle id>.json.
const KindImported = "models.imported"

// ImportedFile is $FAMILY/models/imported/<bundle id>.json: what this app
// already brought over into which $HUB, so that a model the user deletes
// afterwards is not imported again at the next launch. Only that app
// writes it.
func ImportedFile(d family.Dir, bundleID string) string {
	return filepath.Join(d.ModelsDir(), "imported", bundleID+".json")
}

// Imported is the content of ImportedFile.
type Imported struct {
	Schema   string        `json:"schema"`
	BundleID string        `json:"bundleId"`
	Updated  time.Time     `json:"updated"`
	Hubs     []ImportedHub `json:"hubs"`
	Extra    family.Extra  `json:"-"`
}

func (i *Imported) UnmarshalJSON(b []byte) error {
	type plain Imported
	return family.UnmarshalKnown(b, (*plain)(i), &i.Extra)
}

func (i Imported) MarshalJSON() ([]byte, error) {
	type plain Imported
	return family.MarshalKnown(plain(i), i.Extra)
}

// ImportedHub is what was handled for one destination $HUB: refs
// ("repo@commit") imported, already there, or found in no source.
type ImportedHub struct {
	Root string   `json:"root"`
	Refs []string `json:"refs"`
}

// ImportResult is what Import did with one model.
type ImportResult struct {
	Ref  Ref
	From string // the source hub; "" when nothing was brought over
	// Action: "imported", "present" (already in $HUB), "absent" (in no
	// source), "busy" (a lock is held: tried again next time), "no-space"
	// (a copy across volumes would not fit), "error".
	Action string
	Linked int64 // bytes hard-linked (no new disk space)
	Copied int64 // bytes copied (another volume)
	Err    error
}

// Actions of ImportResult.
const (
	ImportImported = "imported"
	ImportPresent  = "present"
	ImportAbsent   = "absent"
	ImportBusy     = "busy"
	ImportNoSpace  = "no-space"
	ImportError    = "error"
)

// ImportReserve is the free space a copy across volumes leaves.
const ImportReserve int64 = 2e9

// Import brings this app's models into s.Hub from the Hugging Face caches
// in from (usually LegacyHubs(s.Hub)), so that moving the library
// (spec v1.0.9) downloads nothing again:
//
//   - the models are want (the app's catalogue entries, main and auxiliary
//     repos as separate claims, with Files/Aux/Sizes as for Check) and the
//     claims of this app's claims file when it names one of from as root
//     (the app's previous library);
//   - each is looked up in s.Hub first (another app imported or downloaded
//     it), then in each source in turn, and taken from the first where it
//     is installed (Check);
//   - files are hard-linked (same volume: no space used, and the source
//     stays untouched for the tools and older app versions that use it);
//     across volumes they are copied when the copy leaves ImportReserve
//     free, else the model is skipped ("no-space");
//   - the snapshot is built as snapshots/<commit>.incomplete and renamed,
//     holding the destination repo's lock exclusively and the source
//     repo's shared (a busy lock skips the model: "busy"); the refs of the
//     source that name the commit are copied when the destination lacks
//     them; blob sizes are recorded, and the repo is marked for Time
//     Machine as backup (models.backup) says;
//   - the app's previous claims are claimed again here (OriginImported);
//     want is not claimed: the app claims what it uses as before (its
//     migration when set up, its first-run reuse card on a fresh install);
//   - nothing in the sources is changed or deleted, and models that no
//     family app wants (the user's own) are not looked at.
//
// Handled refs (imported, present, absent) are recorded in ImportedFile
// and skipped next time, so that a model deleted after the import does
// not come back; busy, no-space and failed ones are tried again at the
// next call. Call it at launch before showing model states, and after the
// model folder changed. It is safe to run from several apps at once.
func (s *Store) Import(ctx context.Context, from []string, backup bool, want ...Claim) ([]ImportResult, error) {
	var sources []string
	for _, f := range from {
		if f != "" && !SameHub(f, s.Hub) && isDir(f) && !slices.ContainsFunc(sources, func(o string) bool { return SameHub(o, f) }) {
			sources = append(sources, filepath.Clean(f))
		}
	}
	if err := EnsureHub(s.Family, s.Hub); err != nil {
		return nil, err
	}
	rec, _ := s.readImported()
	done := map[string]bool{}
	for _, h := range rec.Hubs {
		if SameHub(h.Root, s.Hub) {
			for _, r := range h.Refs {
				done[r] = true
			}
		}
	}
	// The app's previous claims (its old library) join want, and are
	// claimed again.
	mine := map[Ref]bool{}
	if own, err := s.Claims(); err == nil && own.Root != "" && slices.ContainsFunc(sources, func(f string) bool { return SameHub(f, own.Root) }) {
		for _, c := range own.Claims {
			mine[c.Ref()] = true
			if !slices.ContainsFunc(want, func(w Claim) bool { return w.Ref() == c.Ref() }) {
				want = append(want, c)
			}
		}
	}
	var out []ImportResult
	var handled []string
	for _, w := range want {
		if err := ctx.Err(); err != nil {
			break
		}
		if !ValidRepo(w.Repo) || !IsCommit(w.Revision) || done[w.Ref().String()] {
			continue
		}
		done[w.Ref().String()] = true
		r := s.importOne(ctx, sources, w, backup, mine[w.Ref()])
		out = append(out, r)
		switch r.Action {
		case ImportImported, ImportPresent, ImportAbsent:
			handled = append(handled, w.Ref().String())
		}
	}
	if len(handled) > 0 || len(rec.Hubs) == 0 {
		if err := s.recordImported(rec, handled); err != nil {
			return out, err
		}
	}
	var errs []error
	for _, r := range out {
		if r.Err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", r.Ref, r.Err))
		}
	}
	return out, errors.Join(errs...)
}

func (s *Store) readImported() (Imported, error) {
	var rec Imported
	err := family.Load(ImportedFile(s.Family, s.BundleID), KindImported, 1, &rec)
	if family.IsNotExist(err) {
		err = nil
	}
	return rec, err
}

func (s *Store) recordImported(rec Imported, refs []string) error {
	i := slices.IndexFunc(rec.Hubs, func(h ImportedHub) bool { return SameHub(h.Root, s.Hub) })
	if i < 0 {
		rec.Hubs = append(rec.Hubs, ImportedHub{Root: s.Hub, Refs: []string{}})
		i = len(rec.Hubs) - 1
	}
	h := &rec.Hubs[i]
	h.Refs = append(h.Refs, refs...)
	slices.Sort(h.Refs)
	h.Refs = slices.Compact(h.Refs)
	rec.Schema = family.SchemaOf(KindImported, 1)
	rec.BundleID, rec.Updated = s.BundleID, s.now()
	path := ImportedFile(s.Family, s.BundleID)
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	return family.WriteJSON(path, rec)
}

func (s *Store) importOne(ctx context.Context, sources []string, w Claim, backup, claim bool) ImportResult {
	ref := w.Ref()
	res := ImportResult{Ref: ref}
	c := w
	c.Origin = OriginImported
	if Check(s.Hub, w).State == Installed {
		res.Action = ImportPresent
		if claim {
			if err := s.Claim(ctx, c); err != nil && !errors.Is(err, ErrNotInstalled) {
				res.Action, res.Err = ImportError, err
			}
		}
		return res
	}
	src := ""
	for _, f := range sources {
		if Check(f, w).State == Installed {
			src = f
			break
		}
	}
	if src == "" {
		res.Action = ImportAbsent
		return res
	}
	res.From = src
	dst, err := TryLockRepo(s.Hub, ref.Repo, family.Exclusive, s.who(OpImport))
	if err != nil {
		res.Action, res.Err = ImportBusy, nil
		return res
	}
	defer dst.Release()
	sl, err := TryLockRepo(src, ref.Repo, family.Shared, s.who(OpImport))
	if err != nil && errors.Is(err, family.ErrLocked) {
		res.Action = ImportBusy
		return res
	}
	if sl != nil {
		defer sl.Release()
	}
	// Under the locks: someone may have finished it meanwhile.
	if Check(s.Hub, w).State == Installed {
		res.Action = ImportPresent
	} else {
		if Check(src, w).State != Installed {
			res.Action = ImportAbsent
			return res
		}
		linked, copied, err := linkSnapshotFrom(src, s.Hub, ref)
		res.Linked, res.Copied = linked, copied
		switch {
		case errors.Is(err, errNoSpace):
			res.Action = ImportNoSpace
			return res
		case err != nil:
			res.Action, res.Err = ImportError, err
			return res
		}
		res.Action = ImportImported
	}
	if claim {
		if err := s.ClaimLocked(ctx, c); err != nil && !errors.Is(err, ErrNotInstalled) {
			res.Action, res.Err = ImportError, err
			return res
		}
	}
	_ = MarkRepoBackup(s.Hub, ref.Repo, backup)
	return res
}

var errNoSpace = errors.New("not enough free space to copy across volumes")

// srcFile is a file of the source snapshot.
type srcFile struct {
	rel  string // path in the snapshot, slash-separated
	blob string // blob name in the destination
	real string // the file holding the bytes
	size int64
}

// linkSnapshotFrom builds dst's snapshot of ref from src's: blobs hard-
// linked (or copied across volumes), links ../../blobs/<blob>, refs naming
// the commit. Both repo locks are held.
func linkSnapshotFrom(src, dst string, ref Ref) (linked, copied int64, err error) {
	snap := SnapshotDir(src, ref)
	var files []srcFile
	err = filepath.WalkDir(snap, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		rel, _ := filepath.Rel(snap, p)
		f := srcFile{rel: filepath.ToSlash(rel)}
		real, err := filepath.EvalSymlinks(p)
		if err != nil {
			return err
		}
		fi, err := os.Stat(real)
		if err != nil {
			return err
		}
		if !fi.Mode().IsRegular() {
			return fmt.Errorf("%s: not a regular file", rel)
		}
		f.real, f.size = real, fi.Size()
		if t, err := os.Readlink(p); err == nil && validBlob(filepath.Base(t)) {
			f.blob = filepath.Base(t)
		} else {
			// A plain file in the snapshot (copied by a tool): name its
			// blob by content, as an LFS blob is.
			h, err := hashFile(real)
			if err != nil {
				return err
			}
			f.blob = h
		}
		files = append(files, f)
		return nil
	})
	if err != nil {
		return 0, 0, err
	}
	repoDir := RepoDir(dst, ref.Repo)
	blobs := filepath.Join(repoDir, "blobs")
	if err := os.MkdirAll(blobs, 0o755); err != nil {
		return 0, 0, err
	}
	var todo []srcFile
	for _, f := range files {
		if fi, err := os.Stat(filepath.Join(blobs, f.blob)); err == nil && fi.Mode().IsRegular() && fi.Size() == f.size {
			continue
		}
		todo = append(todo, f)
	}
	// Hard links first; whatever fails with EXDEV is copied, if it fits.
	var cross []srcFile
	for _, f := range todo {
		err := linkBlob(f.real, filepath.Join(blobs, f.blob))
		switch {
		case err == nil:
			linked += f.size
		case errors.Is(err, syscall.EXDEV) || errors.Is(err, syscall.EPERM) || errors.Is(err, syscall.EMLINK):
			cross = append(cross, f)
		default:
			return linked, copied, err
		}
	}
	if len(cross) > 0 {
		var need int64
		for _, f := range cross {
			need += f.size
		}
		if free := freeSpace(blobs); free >= 0 && free-need < ImportReserve {
			return linked, copied, errNoSpace
		}
		for _, f := range cross {
			if err := copyBlob(f.real, filepath.Join(blobs, f.blob), f.size); err != nil {
				return linked, copied, err
			}
			copied += f.size
		}
	}
	// The snapshot, built beside its place and renamed in.
	snaps := filepath.Join(repoDir, "snapshots")
	final := filepath.Join(snaps, ref.Revision)
	if isDir(final) {
		// Partial (missing links): add what is missing in place.
		if err := linkFiles(final, files, false); err != nil {
			return linked, copied, err
		}
	} else {
		tmp := final + ".incomplete"
		_ = os.RemoveAll(tmp)
		if err := linkFiles(tmp, files, true); err != nil {
			_ = os.RemoveAll(tmp)
			return linked, copied, err
		}
		if err := os.Rename(tmp, final); err != nil {
			return linked, copied, err
		}
	}
	// Refs of the source naming the commit, when the destination has none
	// of that name or its names a snapshot that is not there.
	srcRefs := filepath.Join(RepoDir(src, ref.Repo), "refs")
	_ = filepath.WalkDir(srcRefs, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		data, err := os.ReadFile(p)
		if err != nil || strings.TrimSpace(string(data)) != ref.Revision {
			return nil
		}
		rel, _ := filepath.Rel(srcRefs, p)
		to := filepath.Join(repoDir, "refs", rel)
		if cur, err := os.ReadFile(to); err == nil {
			if c := strings.TrimSpace(string(cur)); c == ref.Revision || isDir(filepath.Join(snaps, c)) {
				return nil
			}
		}
		_ = os.MkdirAll(filepath.Dir(to), 0o755)
		_ = os.WriteFile(to+".tmp", []byte(ref.Revision), 0o644)
		_ = os.Rename(to+".tmp", to)
		return nil
	})
	sizes := map[string]int64{}
	for _, f := range files {
		sizes[f.blob] = f.size
	}
	return linked, copied, RecordBlobSizes(dst, ref.Repo, sizes)
}

// linkFiles makes <dir>/<rel> → ../../blobs/<blob> (relative, like
// huggingface_hub); fresh: dir is new.
func linkFiles(dir string, files []srcFile, fresh bool) error {
	for _, f := range files {
		link := filepath.Join(dir, filepath.FromSlash(f.rel))
		if err := os.MkdirAll(filepath.Dir(link), 0o755); err != nil {
			return err
		}
		target := strings.Repeat("../", strings.Count(f.rel, "/")+2) + "blobs/" + f.blob
		if !fresh {
			if cur, err := os.Readlink(link); err == nil && cur == target {
				continue
			}
			_ = os.Remove(link)
		}
		if err := os.Symlink(target, link); err != nil {
			return err
		}
	}
	return nil
}

// linkBlob hard-links src to dst (replacing a wrong-sized dst).
func linkBlob(src, dst string) error {
	tmp := dst + ".import-tmp"
	_ = os.Remove(tmp)
	if err := os.Link(src, tmp); err != nil {
		var le *os.LinkError
		if errors.As(err, &le) {
			return le.Err
		}
		return err
	}
	return os.Rename(tmp, dst)
}

// copyBlob copies src to dst through a temporary file.
func copyBlob(src, dst string, size int64) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	tmp := dst + ".import-tmp"
	out, err := os.OpenFile(tmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	n, err := io.Copy(out, in)
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	if err == nil && n != size {
		err = fmt.Errorf("%s: copied %d of %d bytes", src, n, size)
	}
	if err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, dst)
}

func hashFile(p string) (string, error) {
	f, err := os.Open(p)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
