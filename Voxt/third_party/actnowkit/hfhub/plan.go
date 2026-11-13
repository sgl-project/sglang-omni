// Download plans: the files of a model and what of them the cache already has.

package hfhub

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/hehehai/actnowkit/models"
)

// remoteFile is a file to download and the blob it becomes.
type remoteFile struct {
	File
	blob string // blob name: the LFS sha256, else the git blob id
	lfs  bool
}

type plan struct {
	Plan
	entry  Entry // the repo's own entry (auxEntry for auxiliary repos)
	aux    bool
	repo   string
	rev    string // what was asked: "main", a branch or a commit
	commit string
	ep     string
	files  []remoteFile
}

// Plan lists the files of e at its revision and checks the disk. Files
// are e's own; the byte counts include its auxiliary repos.
func (c *Client) Plan(ctx context.Context, e Entry) (Plan, error) {
	_, p, err := c.planAll(ctx, e)
	return p, err
}

// blobRE is what a blob name may be: a git blob id or a sha256. Anything
// else from the API is refused (it becomes a path).
var blobRE = regexp.MustCompile(`^(?:[0-9a-f]{40}|[0-9a-f]{64})$`)

func (c *Client) plan(ctx context.Context, e Entry) (*plan, error) {
	if !models.ValidRepo(e.Repo) {
		return nil, fmt.Errorf("%q is not a Hugging Face repo id: %w", e.Repo, ErrNotFound)
	}
	rev := revisionOf(e)
	info, ep, err := c.info(ctx, e.Repo, rev)
	if err != nil {
		return nil, err
	}
	if !isCommit(info.SHA) {
		return nil, fmt.Errorf("%s: %s sent a bad commit %q", e.Repo, ep, info.SHA)
	}
	p := &plan{entry: e, repo: e.Repo, rev: rev, commit: info.SHA, ep: ep}
	p.Revision, p.Endpoint, p.Gated = info.SHA, ep, info.gated()
	restrict := len(e.Files) > 0
	blobs := map[string]bool{} // files with the same content share one blob, counted once
	want := map[string]File{}
	for _, f := range e.Files {
		want[f.Path] = f
	}
	for _, s := range info.Siblings {
		f := remoteFile{File: File{Path: s.Path, Size: s.Size}, blob: strings.ToLower(s.BlobID)}
		if s.LFS != nil {
			f.lfs, f.SHA256, f.blob = true, strings.ToLower(s.LFS.SHA256), strings.ToLower(s.LFS.SHA256)
			f.Size = s.LFS.Size
		}
		if restrict {
			w, ok := want[s.Path]
			if !ok {
				continue
			}
			if w.SHA256 != "" && !strings.EqualFold(w.SHA256, f.SHA256) {
				return nil, fmt.Errorf("%s/%s: the catalogue pins sha256 %s, %s lists %s: %w", e.Repo, s.Path, w.SHA256, ep, f.SHA256, ErrChecksum)
			}
			delete(want, s.Path)
		}
		if !blobRE.MatchString(f.blob) || !safeRel(f.Path) || f.Size < 0 {
			return nil, fmt.Errorf("%s: bad file entry %q", e.Repo, f.Path)
		}
		p.files = append(p.files, f)
		p.Files = append(p.Files, f.File)
		if !blobs[f.blob] {
			blobs[f.blob] = true
			p.TotalBytes += f.Size
			p.Bytes += f.Size - c.have(e.Repo, f)
		}
	}
	for missing := range want {
		return nil, fmt.Errorf("%s@%s has no %s: %w", e.Repo, rev, missing, ErrNotFound)
	}
	return p, nil
}

// auxEntry is an auxiliary repo as an entry of its own.
func auxEntry(e Entry, a Aux) Entry {
	return Entry{ID: e.ID, Repo: a.Repo, Revision: a.Revision, Files: a.Files, Family: e.Family, Role: e.Role}
}

// planAll plans e's auxiliary repos and e (last), adds them up and checks
// the disk for all of them.
func (c *Client) planAll(ctx context.Context, e Entry) ([]*plan, Plan, error) {
	var plans []*plan
	var sum Plan
	for _, a := range e.Aux {
		p, err := c.plan(ctx, auxEntry(e, a))
		if err != nil {
			return nil, Plan{}, err
		}
		p.aux = true
		plans = append(plans, p)
		sum.TotalBytes += p.TotalBytes
		sum.Bytes += p.Bytes
	}
	p, err := c.plan(ctx, e)
	if err != nil {
		return nil, Plan{}, err
	}
	plans = append(plans, p)
	sum.Revision, sum.Files, sum.Endpoint, sum.Gated = p.Revision, p.Files, p.Endpoint, p.Gated
	sum.TotalBytes += p.TotalBytes
	sum.Bytes += p.Bytes
	sum.FreeBytes = freeBytes(c.opts.CacheDir)
	if c.opts.ReserveBytes > 0 && sum.FreeBytes >= 0 && sum.Bytes+c.opts.ReserveBytes > sum.FreeBytes {
		return plans, sum, fmt.Errorf("%s needs %s more and keeps %s free; %s free: %w",
			e.Repo, HumanSize(sum.Bytes), HumanSize(c.opts.ReserveBytes), HumanSize(sum.FreeBytes), ErrDiskSpace)
	}
	return plans, sum, nil
}

// safeRel reports whether a repo path stays inside the snapshot.
func safeRel(p string) bool {
	return p != "" && !strings.HasPrefix(p, "/") && !strings.Contains(p, `\`) && path.Clean(p) == p && !strings.HasPrefix(p, "../") && p != ".."
}

// blobPath is blobs/<blob> of repo.
func (c *Client) blobPath(repo, blob string) string {
	return filepath.Join(RepoDir(c.opts.CacheDir, repo), "blobs", blob)
}

// complete reports whether blob is in place with its size.
func complete(blob string, size int64) bool {
	st, err := os.Stat(blob)
	return err == nil && st.Mode().IsRegular() && st.Size() == size
}

// have is how many bytes of f are on disk already (complete, or written
// into its .incomplete file).
func (c *Client) have(repo string, f remoteFile) int64 {
	blob := c.blobPath(repo, f.blob)
	if complete(blob, f.Size) {
		return f.Size
	}
	return readPartial(blob, f.Size).done
}

// chunks is how many ranges of chunk a file of size downloads in: 1
// under two chunks.
func chunks(size, chunk int64) int {
	if size < 2*chunk {
		return 1
	}
	return int((size + chunk - 1) / chunk)
}

// partsFile is blobs/<blob>.incomplete.parts (see the package doc).
type partsFile struct {
	Size  int64  `json:"size"`
	Chunk int64  `json:"chunk"`
	Done  []bool `json:"done"`
}

// valid reports whether p describes a chunked download of size bytes.
func (p *partsFile) valid(size int64) bool {
	return p.Size == size && p.Chunk > 0 && size >= 2*p.Chunk && int64(len(p.Done)) == (size+p.Chunk-1)/p.Chunk
}

// doneBytes is how many bytes the done ranges hold.
func (p *partsFile) doneBytes() int64 {
	var done int64
	for i, d := range p.Done {
		if d {
			done += min(p.Chunk, p.Size-int64(i)*p.Chunk)
		}
	}
	return done
}

// partial is what of a blob's .incomplete file can be resumed.
type partial struct {
	done  int64
	parts *partsFile // nil: written in order, done bytes long
	// stale: the files do not match the blob (a parts file that does not
	// fit, an .incomplete longer than the blob); start again.
	stale bool
}

// readPartial reads what of blob's .incomplete is done: from its parts
// file when there is one, else the length of the file (written in order,
// by us or by huggingface_hub, which uses the same name).
func readPartial(blob string, size int64) partial {
	if data, err := os.ReadFile(blob + ".incomplete.parts"); err == nil {
		var pf partsFile
		if json.Unmarshal(data, &pf) == nil && pf.valid(size) {
			return partial{done: pf.doneBytes(), parts: &pf}
		}
		return partial{stale: true}
	}
	if st, err := os.Stat(blob + ".incomplete"); err == nil {
		if st.Size() > size {
			return partial{stale: true}
		}
		return partial{done: st.Size()}
	}
	return partial{}
}
