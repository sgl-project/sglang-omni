package mirror

// File: one file from the fastest source, checked by its SHA-256, the
// next source taking over when one fails, stalls, crawls or serves
// something else.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"io"
	"math"
	"net/http"
	"os"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// FileOptions are File's.
type FileOptions struct {
	// Dir is where the temporary file is made ("": os.TempDir()); on
	// the same volume as where it goes, it can be renamed into place.
	Dir string
	// SHA256 is the file's digest (hex): what is downloaded must match.
	SHA256 string
	// Size is the file's size when known, for progress before a server
	// says it (0: the server's).
	Size int64
	// Limit is the most bytes accepted (0: 4 GiB).
	Limit int64
	// Progress, when set, gets bytes done and the total (0 when unknown).
	// Done can go back when a source is left for another.
	Progress func(done, total int64)
}

// ErrChecksum: what a source served does not match the digest.
var ErrChecksum = errors.New("checksum mismatch")

// ChecksumError is a download that did not match its digest.
type ChecksumError struct{ URL, Got string }

func (e *ChecksumError) Error() string {
	return fmt.Sprintf("%s: checksum mismatch (got %s)", e.URL, e.Got)
}

func (e *ChecksumError) Is(target error) bool { return target == ErrChecksum }

// Why a source is left.
var (
	ErrStalled = errors.New("the download stalled")
	ErrTooSlow = errors.New("the download is too slow")
	errLost    = errors.New("another source was faster")
)

// File downloads p (a path under the sources, "/owner/repo/releases/
// download/tag/asset") into a temporary file in o.Dir and checks its
// SHA-256; it returns the file (the caller renames or removes it) and
// the source it came from.
//
// Every source starts at once; the first to deliver Lead bytes (or the
// whole file) wins and the others stop. When the winner fails, stalls
// (Stall), falls below MinRate over RateWindow, or serves a file that
// does not match, the sources left race again from the start; a source
// left for stalling or crawling races once more, on a new connection
// (the last attempt is never left for being slow). A failure
// of every source returns their errors joined (errors.Is(err,
// ErrChecksum) when one served a wrong file).
func (c *Client) File(ctx context.Context, p string, o FileOptions) (string, string, error) {
	want := strings.ToLower(strings.TrimSpace(o.SHA256))
	if len(want) != 64 {
		return "", "", fmt.Errorf("mirror: %s: no SHA-256 to check against", p)
	}
	if o.Limit <= 0 {
		o.Limit = 4 << 30
	}
	pr := &progress{fn: o.Progress, size: o.Size}
	left := c.sources()
	retried := map[string]bool{}
	var errs []error
	for len(left) > 0 {
		pr.reset()
		r, lost := c.race(ctx, p, left, o, pr)
		errs = append(errs, lost...)
		if r == nil {
			break
		}
		left = slices.DeleteFunc(left, func(b string) bool { return b == r.base })
		err := c.finish(r, len(left) > 0 || !retried[r.base], o, pr)
		if err == nil {
			if got := hex.EncodeToString(r.h.Sum(nil)); got != want {
				err = &ChecksumError{URL: r.url, Got: got}
			}
		}
		if err == nil {
			name := r.f.Name()
			if err = r.f.Close(); err == nil {
				r.f = nil
				r.close(false)
				c.logf("mirror: %s from %s", p, r.base)
				return name, r.base, nil
			}
		}
		r.close(true)
		if ctx.Err() != nil {
			return "", "", ctx.Err()
		}
		c.logf("mirror: %s: %v", r.url, err)
		errs = append(errs, fmt.Errorf("%s: %w", r.base, err))
		if (errors.Is(err, ErrStalled) || errors.Is(err, ErrTooSlow)) && !retried[r.base] {
			// Once more on a new connection: a CDN's addresses differ
			// (some Cloudflare ones crawl from China, others do not).
			retried[r.base] = true
			left = append(left, r.base)
			c.client().CloseIdleConnections()
		}
	}
	if ctx.Err() != nil {
		return "", "", ctx.Err()
	}
	return "", "", errors.Join(errs...)
}

// racer is one source's download.
type racer struct {
	base, url string
	ctx       context.Context
	cancel    context.CancelCauseFunc
	resp      *http.Response
	f         *os.File
	h         hash.Hash
	size      int64 // the server's (0: unknown)
	done      atomic.Int64
	eof       bool
}

func (r *racer) open(hc *http.Client, dir string) error {
	req, err := http.NewRequestWithContext(r.ctx, http.MethodGet, r.url, nil)
	if err != nil {
		return err
	}
	resp, err := hc.Do(req)
	if err != nil {
		return err
	}
	r.resp = resp
	if resp.StatusCode != http.StatusOK {
		return &StatusError{URL: r.url, Status: resp.Status, Code: resp.StatusCode}
	}
	r.size = max(resp.ContentLength, 0)
	f, err := os.CreateTemp(dir, ".download-*")
	if err != nil {
		return err
	}
	r.f, r.h = f, sha256.New()
	return nil
}

// read reads until target bytes are done or the end, at most limit.
func (r *racer) read(target, limit int64, report func(*racer)) error {
	buf := make([]byte, 256<<10)
	for !r.eof && r.done.Load() < target {
		n, err := r.resp.Body.Read(buf)
		if n > 0 {
			d := r.done.Add(int64(n))
			if d > limit {
				return fmt.Errorf("%s is larger than %d bytes", r.url, limit)
			}
			if _, werr := r.f.Write(buf[:n]); werr != nil {
				return werr
			}
			r.h.Write(buf[:n])
			report(r)
		}
		if err == io.EOF {
			r.eof = true
			return nil
		}
		if err != nil {
			if cause := context.Cause(r.ctx); cause != nil && !errors.Is(cause, context.Canceled) {
				return cause
			}
			return err
		}
	}
	return nil
}

// close ends the download, removing the file when remove.
func (r *racer) close(remove bool) {
	if r.resp != nil {
		r.resp.Body.Close()
	}
	if r.f != nil {
		r.f.Close()
		if remove {
			os.Remove(r.f.Name())
		}
	}
	r.cancel(nil)
}

// race starts p at every base and returns the first to read Lead bytes
// (or all of it), the others stopped, with the errors of those that
// failed before; nil when all failed.
func (c *Client) race(ctx context.Context, p string, bases []string, o FileOptions, pr *progress) (*racer, []error) {
	lead := or(c.Lead, DefaultLead)
	type result struct {
		r   *racer
		err error
	}
	ch := make(chan result, len(bases))
	racers := make([]*racer, len(bases))
	for i, b := range bases {
		rctx, cancel := context.WithCancelCause(ctx)
		r := &racer{base: b, url: b + p, ctx: rctx, cancel: cancel}
		racers[i] = r
		go func() {
			err := r.open(c.client(), o.Dir)
			if err == nil {
				err = r.read(lead, o.Limit, pr.racing)
			}
			ch <- result{r, err}
		}()
	}
	var errs []error
	for i := range bases {
		a := <-ch
		if a.err == nil {
			pr.solo(a.r)
			for _, r := range racers {
				if r != a.r {
					r.cancel(errLost)
				}
			}
			go func(n int) { // the others end soon: clean up after them
				for range n {
					(<-ch).r.close(true)
				}
			}(len(bases) - i - 1)
			return a.r, errs
		}
		a.r.close(true)
		if ctx.Err() == nil {
			c.logf("mirror: %s: %v", a.r.url, a.err)
			errs = append(errs, fmt.Errorf("%s: %w", a.r.base, a.err))
		}
	}
	return nil, errs
}

// finish reads the rest of the winner's file; when others are left, a
// watch leaves it if it stalls or crawls.
func (c *Client) finish(r *racer, watch bool, o FileOptions, pr *progress) error {
	if watch {
		stop := c.watch(r)
		defer stop()
	}
	return r.read(math.MaxInt64, o.Limit, pr.racing)
}

// watch cancels r with ErrStalled or ErrTooSlow; stop ends the watch.
func (c *Client) watch(r *racer) (stop func()) {
	stall := or(c.Stall, DefaultStall)
	window := or(c.RateWindow, DefaultRateWindow)
	rate := or(c.MinRate, DefaultMinRate)
	tick := max(min(stall, window)/10, 5*time.Millisecond)
	type sample struct {
		at   time.Time
		done int64
	}
	quit := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		t := time.NewTicker(tick)
		defer t.Stop()
		start := time.Now()
		samples := []sample{{start, r.done.Load()}}
		last, lastAt := samples[0].done, start
		for {
			select {
			case <-quit:
				return
			case <-r.ctx.Done():
				return
			case now := <-t.C:
				d := r.done.Load()
				if d != last {
					last, lastAt = d, now
				} else if now.Sub(lastAt) >= stall {
					r.cancel(ErrStalled)
					return
				}
				samples = append(samples, sample{now, d})
				// The oldest sample at least a window old: the rate since.
				k := -1
				for i, s := range samples {
					if now.Sub(s.at) >= window {
						k = i
					}
				}
				if k >= 0 {
					s := samples[k]
					if float64(d-s.done)/now.Sub(s.at).Seconds() < float64(rate) {
						r.cancel(ErrTooSlow)
						return
					}
					samples = samples[k:]
				}
			}
		}
	}()
	return func() {
		close(quit)
		wg.Wait()
	}
}

// progress reports one number for the racers: the most done while they
// race, the winner's after.
type progress struct {
	fn   func(done, total int64)
	size int64

	mu     sync.Mutex
	winner *racer
	best   int64
}

func (p *progress) racing(r *racer) {
	if p.fn == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	d := r.done.Load()
	if p.winner != nil {
		if r != p.winner {
			return
		}
	} else if d <= p.best {
		return
	}
	p.best = d
	total := p.size
	if total <= 0 {
		total = r.size
	}
	p.fn(d, total)
}

// reset starts a new race.
func (p *progress) reset() {
	p.mu.Lock()
	p.winner, p.best = nil, 0
	p.mu.Unlock()
}

// solo: from now on only r reports.
func (p *progress) solo(r *racer) {
	p.mu.Lock()
	p.winner = r
	p.mu.Unlock()
}
