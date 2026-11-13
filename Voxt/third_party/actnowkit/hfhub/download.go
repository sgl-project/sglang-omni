// Downloading a plan: the family locks, tasks in parallel, ranges, resume
// and retries.

package hfhub

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/hehehai/actnowkit/family"
	"github.com/hehehai/actnowkit/models"
)

// task is one request: a whole file, or one range of a big one.
type task struct {
	file  *fileState
	index int // chunk index
	from  int64
	to    int64 // exclusive
}

// fileState is a blob being downloaded, its lock held.
type fileState struct {
	remoteFile
	path   string // blobs/<blob>.incomplete
	f      *os.File
	lock   *models.HeldLock
	mu     sync.Mutex
	parts  *partsFile // nil: sequential
	left   int        // tasks not done
	closed bool
}

// close closes the file and releases the blob lock. Safe twice.
func (fs *fileState) close() {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	if fs.closed {
		return
	}
	fs.closed = true
	fs.f.Close()
	fs.lock.Release()
}

// finishAttempts bounds how often a download goes back to fetching when
// blobs vanished between its shared and exclusive repo locks (a deletion
// in between).
const finishAttempts = 3

// Download fetches what Plan lists into the cache: the auxiliary repos
// first, then e's (see the package doc for the locks and claims). An
// error leaves what was fetched for resuming. With Options.Store the
// repos are claimed: the main one while its snapshot is finished, the
// auxiliary ones after it.
func (c *Client) Download(ctx context.Context, e Entry, progress func(Progress)) error {
	defer c.lockRepo(e.Repo)()
	plans, sum, err := c.planAll(ctx, e)
	if err != nil {
		return err
	}
	var base int64
	var auxClaims []models.Claim
	for _, p := range plans {
		unlock := func() {}
		if p.repo != e.Repo {
			unlock = c.lockRepo(p.repo)
		}
		claim, err := c.fetchRepo(ctx, p, func(pr Progress) {
			if progress != nil {
				pr.Repo, pr.Bytes, pr.TotalBytes = e.Repo, base+pr.Bytes, sum.TotalBytes
				progress(pr)
			}
		})
		unlock()
		if err != nil {
			return err
		}
		if p.aux && claim != nil {
			auxClaims = append(auxClaims, *claim)
		}
		base += p.TotalBytes
	}
	if c.opts.Store != nil {
		if len(auxClaims) > 0 {
			// Each takes its repo's shared lock and checks it is still
			// installed.
			if err := c.opts.Store.Claim(ctx, auxClaims...); err != nil {
				return err
			}
		}
		c.post()
	}
	return nil
}

// fetchRepo downloads one repo's plan holding its repo lock: shared while
// fetching blobs, exclusive while finishing the snapshot. It returns the
// repo's claim (nil without a Store); the main repo's is already made.
func (c *Client) fetchRepo(ctx context.Context, p *plan, progress func(Progress)) (*models.Claim, error) {
	hub := c.opts.CacheDir
	fetched := false
	var lastErr error
	for range finishAttempts {
		lock, err := models.LockRepo(ctx, hub, p.repo, family.Shared, c.who(models.OpDownload))
		if err != nil {
			return nil, err
		}
		f, err := c.fetchBlobs(ctx, p, progress)
		fetched = fetched || f
		if err == nil {
			// Others may act between giving up the shared lock and getting
			// the exclusive one: check the blobs again.
			err = lock.Upgrade(ctx, models.OpFinish)
		}
		if err != nil {
			lock.Release()
			return nil, err
		}
		if missing := c.missingBlobs(p); len(missing) > 0 {
			lock.Release()
			lastErr = fmt.Errorf("%s: %s was removed while the download finished", p.repo, missing[0])
			continue
		}
		claim, err := c.finishRepo(ctx, p, fetched)
		lock.Release()
		return claim, err
	}
	return nil, lastErr
}

// missingBlobs are the paths of p whose blobs are not in place.
func (c *Client) missingBlobs(p *plan) []string {
	var missing []string
	for _, f := range p.files {
		if !complete(c.blobPath(p.repo, f.blob), f.Size) {
			missing = append(missing, f.Path)
		}
	}
	return missing
}

// finishRepo links the snapshot and writes the ref (the caller holds the
// repo lock exclusively), then claims the main repo.
func (c *Client) finishRepo(ctx context.Context, p *plan, fetched bool) (*models.Claim, error) {
	repoDir := RepoDir(c.opts.CacheDir, p.repo)
	if err := linkSnapshot(repoDir, p.commit, p.files); err != nil {
		return nil, err
	}
	sizes := map[string]int64{}
	for _, f := range p.files {
		sizes[f.blob] = f.Size
	}
	if err := models.RecordBlobSizes(c.opts.CacheDir, p.repo, sizes); err != nil {
		return nil, err
	}
	if c.opts.Store != nil {
		_ = models.MarkRepoBackup(c.opts.CacheDir, p.repo, c.backup())
	}
	ref := p.rev
	if isCommit(ref) {
		// A pinned commit is found by its path; tools that resolve the
		// repo by id (an engine for auxiliary repos and for families it
		// cannot load from a directory, the Python tools) read refs/main.
		// Point it here unless it names another snapshot that is there.
		ref = "main"
		if old, err := os.ReadFile(filepath.Join(repoDir, "refs", "main")); err == nil {
			if commit := strings.TrimSpace(string(old)); commit != p.commit && isDir(filepath.Join(repoDir, "snapshots", commit)) {
				ref = ""
			}
		}
	}
	if ref != "" {
		if err := writeAtomic(filepath.Join(repoDir, "refs", filepath.FromSlash(ref)), []byte(p.commit)); err != nil {
			return nil, err
		}
	}
	if c.opts.Store == nil {
		return nil, nil
	}
	claim := models.Claim{
		Repo: p.repo, Revision: p.commit, Model: p.entry.ID, Family: p.entry.Family, Role: p.entry.Role,
		Aux: p.aux, Origin: models.OriginDownloaded,
	}
	if !fetched {
		claim.Origin = models.OriginReused // everything was there already
	}
	for _, f := range p.entry.Files {
		claim.Files = append(claim.Files, f.Path)
	}
	if p.aux {
		return &claim, nil
	}
	if err := c.opts.Store.ClaimLocked(ctx, claim); err != nil {
		if errors.Is(err, models.ErrNotInstalled) {
			return nil, fmt.Errorf("%s: downloaded, but it has no weights file (list the files the entry uses): %w", p.repo, err)
		}
		return nil, err
	}
	return &claim, nil
}

// fetchBlobs downloads the blobs of a plan that are not in place, each
// holding its blob lock. It reports whether it wrote any.
func (c *Client) fetchBlobs(ctx context.Context, p *plan, progress func(Progress)) (bool, error) {
	hub := c.opts.CacheDir
	if err := os.MkdirAll(filepath.Join(RepoDir(hub, p.repo), "blobs"), 0o755); err != nil {
		return false, err
	}

	var done atomic.Int64
	var current, waiting atomic.Value
	current.Store("")
	waiting.Store("")
	report := throttle(c.progressEvery, func(final bool) {
		if progress != nil {
			progress(Progress{Repo: p.repo, File: current.Load().(string), Bytes: done.Load(), TotalBytes: p.TotalBytes,
				Mirror: p.ep, WaitingFor: waiting.Load().(string)})
		}
	})

	// What is missing, each blob once: two files with the same content
	// (Marvis's prompts/*.txt) are one blob, or the two writers would race
	// on its .incomplete.
	var todo []remoteFile
	counted := map[string]int64{} // bytes of each blob counted in done; producer only
	seen := map[string]bool{}
	for _, f := range p.files {
		if seen[f.blob] {
			continue
		}
		seen[f.blob] = true
		if complete(c.blobPath(p.repo, f.blob), f.Size) {
			done.Add(f.Size)
			continue
		}
		h := c.have(p.repo, f)
		counted[f.blob] = h
		done.Add(h)
		todo = append(todo, f)
	}
	report(true)
	if len(todo) == 0 {
		return false, nil
	}
	recount := func(blob string, now int64) {
		done.Add(now - counted[blob])
		counted[blob] = now
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var firstErr error
	var errOnce sync.Once
	fail := func(err error) {
		errOnce.Do(func() { firstErr = err; cancel() })
	}
	var openMu sync.Mutex
	var open []*fileState
	defer func() {
		// The workers have stopped: release what is left (failures,
		// cancellation); partial files stay for resuming.
		for _, fs := range open {
			fs.close()
		}
	}()
	var fetched atomic.Bool
	queue := make(chan task)

	// start prepares a blob whose lock is held and queues its tasks.
	start := func(f remoteFile, lock *models.HeldLock) error {
		final := c.blobPath(p.repo, f.blob)
		if complete(final, f.Size) { // another app finished it meanwhile
			lock.Release()
			recount(f.blob, f.Size)
			report(false)
			return nil
		}
		fs, tasks, have, err := c.prepare(f, final, lock)
		if err != nil {
			lock.Release()
			return err
		}
		openMu.Lock()
		open = append(open, fs)
		openMu.Unlock()
		fetched.Store(true)
		recount(f.blob, have)
		if len(tasks) == 0 {
			return c.finishFile(ctx, fs)
		}
		fs.left = len(tasks)
		for _, t := range tasks {
			select {
			case queue <- t:
			case <-ctx.Done():
				return ctx.Err()
			}
		}
		return nil
	}

	// The producer takes blob locks in order, as workers become free;
	// blobs another app is downloading come last, waiting for it.
	go func() {
		defer close(queue)
		var busy []remoteFile
		for _, f := range todo {
			if ctx.Err() != nil {
				return
			}
			lock, err := models.TryLockBlob(hub, p.repo, f.blob, c.who(models.OpDownload))
			var be *models.BusyError
			if errors.As(err, &be) {
				busy = append(busy, f)
				continue
			}
			if err == nil {
				err = start(f, lock)
			}
			if err != nil {
				fail(fmt.Errorf("%s/%s: %w", p.repo, f.Path, err))
				return
			}
		}
		for _, f := range busy {
			final := c.blobPath(p.repo, f.blob)
			lock, err := c.waitBlob(ctx, p.repo, f.blob, func(owner string) {
				// Its progress, from the other app's files.
				if complete(final, f.Size) {
					recount(f.blob, f.Size)
				} else {
					recount(f.blob, readPartial(final, f.Size).done)
				}
				waiting.Store(owner)
				report(false)
			})
			if waiting.Load().(string) != "" {
				waiting.Store("")
				report(false)
			}
			if err == nil {
				err = start(f, lock)
			}
			if err != nil {
				fail(fmt.Errorf("%s/%s: %w", p.repo, f.Path, err))
				return
			}
		}
	}()

	var wg sync.WaitGroup
	for range c.opts.Parallel {
		wg.Go(func() {
			for t := range queue {
				current.Store(t.file.Path)
				err := c.fetch(ctx, p, t, func(n int64) { done.Add(n); report(false) })
				if err != nil {
					fail(fmt.Errorf("%s/%s: %w", p.repo, t.file.Path, err))
					continue
				}
				if err := c.taskDone(ctx, t); err != nil {
					fail(err)
				}
			}
		})
	}
	wg.Wait() // the queue is closed: the producer is done too
	if firstErr != nil {
		return fetched.Load(), firstErr
	}
	if err := ctx.Err(); err != nil {
		return fetched.Load(), err
	}
	report(true)
	return fetched.Load(), nil
}

// waitBlob waits for a blob lock another process holds, calling waiting
// with its owner's bundle id at every try.
func (c *Client) waitBlob(ctx context.Context, repo, blob string, waiting func(owner string)) (*models.HeldLock, error) {
	looked := false
	for {
		lock, err := models.TryLockBlob(c.opts.CacheDir, repo, blob, c.who(models.OpDownload))
		var be *models.BusyError
		if !errors.As(err, &be) {
			return lock, err
		}
		owner := UnknownOwner
		if be.Owner == nil {
			// The holder writes its owner file just after taking the
			// lock, and removes it as it lets go: look again, then try
			// the lock once more, before calling it unknown.
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(25 * time.Millisecond):
			}
			if o, ok := models.ReadOwner(models.BlobLockFile(c.opts.CacheDir, repo, blob)); ok {
				be.Owner = &o
			} else if !looked {
				looked = true
				continue
			}
		}
		looked = false
		if be.Owner != nil && be.Owner.BundleID != "" {
			owner = be.Owner.BundleID
		}
		waiting(owner)
		t := time.NewTimer(c.waitPoll)
		select {
		case <-ctx.Done():
			t.Stop()
			return nil, ctx.Err()
		case <-t.C:
		}
	}
}

// prepare opens a blob's .incomplete file for writing (its lock is held)
// and lists the tasks still to do, resuming what is there; have is the
// bytes already written.
func (c *Client) prepare(f remoteFile, final string, lock *models.HeldLock) (fs *fileState, tasks []task, have int64, err error) {
	inc := final + ".incomplete"
	pt := readPartial(final, f.Size)
	if pt.stale {
		os.Remove(inc)
		os.Remove(inc + ".parts")
		pt = partial{}
	}
	fs = &fileState{remoteFile: f, path: inc, lock: lock}
	if pt.parts == nil && chunks(f.Size, c.opts.ChunkSize) > 1 {
		// Start a parts file: chunks inside what a sequential writer left
		// are done. It is written before the .incomplete file exists.
		chunk := c.opts.ChunkSize
		parts := &partsFile{Size: f.Size, Chunk: chunk, Done: make([]bool, chunks(f.Size, chunk))}
		for i := range parts.Done {
			parts.Done[i] = min(int64(i+1)*chunk, f.Size) <= pt.done
		}
		if err := writeParts(inc+".parts", parts); err != nil {
			return nil, nil, 0, err
		}
		pt = partial{done: parts.doneBytes(), parts: parts}
	}
	if pt.parts != nil {
		fs.parts = pt.parts
		for i, d := range pt.parts.Done {
			if !d {
				from := int64(i) * pt.parts.Chunk
				tasks = append(tasks, task{file: fs, index: i, from: from, to: min(from+pt.parts.Chunk, f.Size)})
			}
		}
	} else if pt.done < f.Size || f.Size == 0 {
		tasks = append(tasks, task{file: fs, from: pt.done, to: f.Size})
	}
	fh, err := os.OpenFile(inc, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil, nil, 0, err
	}
	fs.f = fh
	return fs, tasks, pt.done, nil
}

// throttle calls fn at most once every d, and always when final.
func throttle(d time.Duration, fn func(final bool)) func(final bool) {
	var mu sync.Mutex
	var last time.Time
	return func(final bool) {
		mu.Lock()
		defer mu.Unlock()
		if !final && time.Since(last) < d {
			return
		}
		last = time.Now()
		fn(final)
	}
}

// fetch downloads one task, retrying transient failures from where it got.
func (c *Client) fetch(ctx context.Context, p *plan, t task, add func(int64)) error {
	u := p.ep + "/" + p.repo + "/resolve/" + p.commit + "/" + escapePath(t.file.Path)
	token := c.token()
	from := t.from
	var err error
	for attempt := range 4 {
		if attempt > 0 {
			select {
			case <-time.After(time.Duration(attempt*attempt) * 500 * time.Millisecond):
			case <-ctx.Done():
				return ctx.Err()
			}
		}
		if from >= t.to && t.to > 0 {
			return nil
		}
		var n int64
		n, err = c.fetchRange(ctx, p.ep, u, token, t, from, add)
		from += n
		if err == nil || ctx.Err() != nil || !retryable(err) {
			break
		}
	}
	if ctx.Err() != nil {
		return ctx.Err()
	}
	return err
}

type permanent struct{ error }

func (p permanent) Unwrap() error { return p.error }

func retryable(err error) bool {
	var pe permanent
	return !errors.As(err, &pe)
}

// fetchRange writes bytes [from, t.to) of the file at their offsets and
// returns how many it wrote.
func (c *Client) fetchRange(ctx context.Context, ep, u, token string, t task, from int64, add func(int64)) (int64, error) {
	req, err := c.newRequest(ctx, ep, u, token)
	if err != nil {
		return 0, permanent{err}
	}
	ranged := from > 0 || t.to < t.file.Size
	if ranged {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", from, t.to-1))
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	switch {
	case resp.StatusCode == http.StatusPartialContent:
	case resp.StatusCode == http.StatusOK:
		if ranged {
			if t.file.parts != nil {
				return 0, permanent{fmt.Errorf("%s does not serve ranges", ep)}
			}
			// The server ignored the range: start the file again.
			add(-from)
			from = 0
			if err := t.file.f.Truncate(0); err != nil {
				return 0, permanent{err}
			}
		}
	case resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden:
		return 0, permanent{fmt.Errorf("%w (HTTP %d)", ErrGated, resp.StatusCode)}
	case resp.StatusCode == http.StatusNotFound:
		return 0, permanent{ErrNotFound}
	case resp.StatusCode == http.StatusRequestedRangeNotSatisfiable:
		return 0, permanent{fmt.Errorf("HTTP 416 for bytes %d-%d", from, t.to-1)}
	default:
		return 0, fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	buf := make([]byte, 256<<10)
	off := from
	for off < t.to {
		n, rerr := resp.Body.Read(buf[:min(int64(len(buf)), t.to-off)])
		if n > 0 {
			if _, err := t.file.f.WriteAt(buf[:n], off); err != nil {
				return off - from, permanent{err}
			}
			off += int64(n)
			add(int64(n))
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			return off - from, rerr
		}
	}
	if off < t.to {
		return off - from, io.ErrUnexpectedEOF
	}
	return off - from, nil
}

// taskDone records a finished task and finishes its file after the last.
func (c *Client) taskDone(ctx context.Context, t task) error {
	fs := t.file
	fs.mu.Lock()
	fs.left--
	last := fs.left == 0
	var err error
	if fs.parts != nil {
		fs.parts.Done[t.index] = true
		err = writeParts(fs.path+".parts", fs.parts)
	}
	fs.mu.Unlock()
	if err != nil {
		return err
	}
	if last {
		return c.finishFile(ctx, fs)
	}
	return nil
}

// finishFile checks a complete .incomplete blob, moves it in place and
// releases its lock.
func (c *Client) finishFile(ctx context.Context, fs *fileState) error {
	defer fs.close()
	if err := fs.f.Sync(); err != nil {
		return err
	}
	ok, err := checkBlob(ctx, fs.path, fs.remoteFile)
	if err != nil {
		return err
	}
	if !ok {
		if err := fs.f.Close(); err != nil {
			return err
		}
		os.Remove(fs.path)
		os.Remove(fs.path + ".parts")
		return fmt.Errorf("%s: %w", fs.Path, ErrChecksum)
	}
	if err := fs.f.Close(); err != nil {
		return err
	}
	if err := os.Rename(fs.path, strings.TrimSuffix(fs.path, ".incomplete")); err != nil {
		return err
	}
	for attempt := 0; ; attempt++ {
		err := os.Remove(fs.path + ".parts")
		if err == nil || os.IsNotExist(err) {
			break
		}
		if runtime.GOOS != "windows" || attempt == 99 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	return nil
}
