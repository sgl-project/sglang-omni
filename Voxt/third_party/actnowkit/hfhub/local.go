// The cache on disk: status, verification, deletion.

package hfhub

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/hehehai/actnowkit/family"
	"github.com/hehehai/actnowkit/models"
)

// commitOf is the snapshot e uses on disk: its pinned commit, else what
// its ref (refs/main) points at; "" when unknown.
func commitOf(repoDir string, e Entry) string {
	rev := revisionOf(e)
	if isCommit(rev) {
		return rev
	}
	if !safeRel(rev) {
		return ""
	}
	data, err := os.ReadFile(filepath.Join(repoDir, "refs", filepath.FromSlash(rev)))
	if err != nil {
		return ""
	}
	if commit := strings.TrimSpace(string(data)); isCommit(commit) {
		return commit
	}
	return ""
}

// Status reports what of e is on disk, from the files alone (no network).
func (c *Client) Status(ctx context.Context, e Entry) (Local, error) {
	return Status(c.opts.CacheDir, e)
}

// Status is Client.Status for a cache directory. It applies the family's
// "installed" rules (models.Check: the snapshot exists, every link
// reaches its blob, the listed files are there, a main repo has a weights
// file) to e and each auxiliary repo, resolving branches through refs/.
// An entry whose auxiliary repos are not all there is Partial, and its
// Missing lists them as "<repo>:".
func Status(cacheDir string, e Entry) (Local, error) {
	l := statusOne(cacheDir, e, false)
	for _, a := range e.Aux {
		al := statusOne(cacheDir, auxEntry(e, a), true)
		l.Bytes += al.Bytes
		for _, d := range al.Damaged {
			l.Damaged = append(l.Damaged, a.Repo+":"+d)
		}
		if al.State != Downloaded {
			l.Missing = append(l.Missing, a.Repo+":")
			if l.State == Downloaded || al.State == Partial {
				l.State = Partial
			}
		}
	}
	return l, nil
}

// statusOne is the status of one repo.
func statusOne(cacheDir string, e Entry, aux bool) Local {
	var l Local
	if !models.ValidRepo(e.Repo) {
		l.State = Missing
		return l
	}
	repoDir := RepoDir(cacheDir, e.Repo)
	commit := commitOf(repoDir, e)
	if commit == "" {
		// Unknown commit (a branch never resolved here): a repo directory
		// with something in it is a download under way.
		st := models.Check(cacheDir, models.Claim{Repo: e.Repo, Revision: strings.Repeat("0", 40), Aux: aux})
		l.State, l.Bytes = Missing, st.Incomplete
		if st.State != models.Missing {
			l.State = Partial
		}
		return l
	}
	st := models.Check(cacheDir, claimOf(e, commit, aux))
	l.Snapshot, l.Missing, l.Damaged, l.Bytes = st.Snapshot, st.Missing, st.Damaged, st.Bytes
	switch st.State {
	case models.Installed:
		l.State = Downloaded
	case models.Partial:
		l.State = Partial
		l.Bytes += st.Incomplete
	default:
		l.State = Missing
	}
	return l
}

// claimOf is the claim that describes e at commit.
func claimOf(e Entry, commit string, aux bool) models.Claim {
	c := models.Claim{Repo: e.Repo, Revision: commit, Model: e.ID, Family: e.Family, Role: e.Role, Aux: aux}
	for _, f := range e.Files {
		c.Files = append(c.Files, f.Path)
		if f.Size > 0 && !strings.ContainsAny(f.Path, "*?[") {
			if c.Sizes == nil {
				c.Sizes = map[string]int64{}
			}
			c.Sizes[f.Path] = f.Size
		}
	}
	return c
}

func isDir(p string) bool {
	st, err := os.Stat(p)
	return err == nil && st.IsDir()
}

// Verify hashes every file of e's snapshot against the HF API's listing
// (or e.Files when they all carry hashes) and returns those missing or
// bad; bad blobs are removed (holding their blob locks; a blob being
// downloaded is left alone), so that any app's next Download fetches them
// again. Each repo is verified holding its repo lock shared. Files of
// auxiliary repos are named "<repo>:<path>".
func (c *Client) Verify(ctx context.Context, e Entry) ([]string, error) {
	defer c.lockRepo(e.Repo)()
	removed := false
	defer func() {
		if removed {
			c.post()
		}
	}()
	bad, err := c.verifyOne(ctx, e, &removed)
	if err != nil {
		return nil, err
	}
	for _, a := range e.Aux {
		ae := auxEntry(e, a)
		if commitOf(RepoDir(c.opts.CacheDir, a.Repo), ae) == "" {
			bad = append(bad, a.Repo+":")
			continue
		}
		unlock := func() {}
		if a.Repo != e.Repo {
			unlock = c.lockRepo(a.Repo)
		}
		ab, err := c.verifyOne(ctx, ae, &removed)
		unlock()
		if err != nil {
			return nil, err
		}
		for _, p := range ab {
			bad = append(bad, a.Repo+":"+p)
		}
	}
	return bad, nil
}

func (c *Client) verifyOne(ctx context.Context, e Entry, removed *bool) ([]string, error) {
	if !models.ValidRepo(e.Repo) {
		return nil, fmt.Errorf("%q is not a Hugging Face repo id", e.Repo)
	}
	hub := c.opts.CacheDir
	repoDir := RepoDir(hub, e.Repo)
	commit := commitOf(repoDir, e)
	if commit == "" {
		return nil, fmt.Errorf("%s is not downloaded", e.Repo)
	}
	var files []remoteFile
	if hashed := len(e.Files) > 0 && !slices.ContainsFunc(e.Files, func(f File) bool { return f.SHA256 == "" }); hashed {
		for _, f := range e.Files {
			files = append(files, remoteFile{File: f, blob: strings.ToLower(f.SHA256), lfs: true})
		}
	} else {
		pinned := e
		pinned.Revision = commit
		p, err := c.plan(ctx, pinned)
		if err != nil {
			return nil, err
		}
		files = p.files
	}
	lock, err := models.LockRepo(ctx, hub, e.Repo, family.Shared, c.who(models.OpVerify))
	if err != nil {
		return nil, err
	}
	defer lock.Release()
	var bad []string
	blobsDir := filepath.Join(repoDir, "blobs")
	snap := filepath.Join(repoDir, "snapshots", commit)
	for _, f := range files {
		link := filepath.Join(snap, filepath.FromSlash(f.Path))
		blob, err := models.ResolveBlob(repoDir, link)
		if err != nil {
			bad = append(bad, f.Path)
			continue
		}
		ok, err := checkBlob(ctx, blob, f)
		if err != nil {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			bad = append(bad, f.Path)
			continue
		}
		if ok {
			continue
		}
		bad = append(bad, f.Path)
		// Only a blob of this repo's blobs/, under its lock.
		name := filepath.Base(blob)
		if real, err := filepath.EvalSymlinks(blobsDir); err != nil || filepath.Dir(blob) != real {
			continue
		}
		bl, err := models.TryLockBlob(hub, e.Repo, name, c.who(models.OpVerify))
		if err != nil {
			continue // being downloaded (or repaired) right now
		}
		if os.Remove(blob) == nil {
			*removed = true
		}
		bl.Release()
	}
	return bad, nil
}

// Delete removes e from this app (spec §7.6). With Options.Store it is
// models.Store.DeleteModel(…, e.ID, …) for e's repo and then each
// auxiliary repo, each judged on its own: e's name leaves this app's
// claim, the claim goes when no other entry of this app uses the repo,
// and the files go only when no other app claims or loads them (scope
// ThisApp), or whoever does (AllApps, with a tombstone). An auxiliary
// repo another entry of this app still uses stays (Result.OwnModels).
// The results are in that order.
//
// Without a Store, the snapshot is deleted (models.DeleteSnapshot) holding
// the repo lock exclusively, and auxiliary repos stay (other entries may
// share them). Either way a repo lock held elsewhere (a download) makes
// it fail at once with a *models.BusyError naming its holder.
func (c *Client) Delete(ctx context.Context, e Entry, scope models.Scope) ([]models.Result, error) {
	hub := c.opts.CacheDir
	refOf := func(e Entry) (models.Ref, bool) {
		commit := commitOf(RepoDir(hub, e.Repo), e)
		return models.Ref{Repo: e.Repo, Revision: commit}, commit != "" && models.ValidRepo(e.Repo)
	}
	if c.opts.Store == nil {
		ref, ok := refOf(e)
		if !ok {
			return nil, nil
		}
		lock, err := models.TryLockRepo(hub, ref.Repo, family.Exclusive, c.who(models.OpDelete))
		if err != nil {
			return nil, err
		}
		defer lock.Release()
		freed, err := models.DeleteSnapshot(hub, ref)
		return []models.Result{{Decision: models.Decision{Ref: ref}, Deleted: err == nil, Freed: freed}}, err
	}
	var out []models.Result
	var errs []error
	entries := []Entry{e}
	for _, a := range e.Aux {
		entries = append(entries, auxEntry(e, a))
	}
	for i, x := range entries {
		ref, ok := refOf(x)
		if !ok {
			continue
		}
		res, err := c.opts.Store.DeleteModel(ctx, ref, e.ID, scope)
		if err != nil {
			if i == 0 {
				return nil, err // the model stays: so do its auxiliary repos
			}
			errs = append(errs, err)
			continue
		}
		out = append(out, res)
	}
	return out, errors.Join(errs...)
}

// EngineCopiesDir is where mlx-audio-swift keeps its flat copies of the
// repos it resolves by id: $HUB/mlx-audio (spec §7.8).
func EngineCopiesDir(cacheDir string) string { return models.EngineCopiesDir(cacheDir) }

// EngineCopy is mlx-audio-swift's copy of one repo: $HUB/mlx-audio/<owner>_<name>.
func EngineCopy(cacheDir, repo string) string { return models.EngineCopy(cacheDir, repo) }

// CleanEngineCopies removes the empty directories mlx-audio-swift leaves
// in EngineCopiesDir after a failed load, older than
// models.EngineCopyMinAge (younger ones may be another app's engine at
// work). Left there, the next load of that repo finds an "incomplete"
// model and deletes the repo's snapshot from the hub as well. Call it
// before starting an engine. It returns what it removed.
func CleanEngineCopies(cacheDir string) ([]string, error) {
	return models.CleanEngineCopies(cacheDir, models.EngineCopyMinAge)
}
